"""
Process-level memory hygiene for the serving container.

This module exists because the serving process is memory-bound, not CPU-bound.
Measured steady-state resident set on a small instance:

    python + fastapi/uvicorn/pydantic   ~46 MB
    numpy + faiss + onnxruntime         ~36 MB
    chunk store + FAISS + BM25 index    ~24 MB
    embedder ONNX session (fp32)       ~108 MB
    cross-encoder ONNX session (fp32)   ~95 MB
    ------------------------------------------
    baseline                           ~310 MB

On a 512 MB instance that leaves very little headroom, so three things must be
controlled or the process gets OOM-killed and restarted:

1. **Nothing optional may be imported.** ``pandas`` alone costs ~44 MB and
   ``scikit-learn`` ~46 MB; neither is needed to answer a question. Optional
   dependencies are therefore imported lazily, inside the function that needs
   them (see :mod:`backend.evaluation.evaluator`).

2. **Concurrency must be bounded.** Starlette runs ``def`` endpoints in a
   thread pool whose default size is 40. Because ONNX Runtime allocates
   activation buffers per ``Run()`` call, 40 concurrent ``/ask`` requests
   allocate 40 sets of transformer activations at once - hundreds of MB of
   transient memory from a single traffic spike. :data:`MODEL_LOCK` serializes
   model inference and :func:`limit_thread_pool` caps the pool, which converts a
   memory spike into a queue. On a fractional-CPU instance this costs no real
   throughput: the ONNX sessions are already pinned to one thread each.

3. **Freed memory must be returned to the OS.** glibc keeps freed blocks in
   per-thread arenas rather than releasing them, so RSS ratchets upward request
   after request and looks exactly like a leak. :func:`release_memory` runs a
   collection and then ``malloc_trim(0)`` to hand the freed pages back, and
   :func:`configure_allocator` caps the number of arenas.
"""
from __future__ import annotations

import ctypes
import ctypes.util
import gc
import os
import platform
import sys
import threading

# glibc mallopt parameter numbers (from <malloc.h>).
_M_TRIM_THRESHOLD = -1
_M_MMAP_THRESHOLD = -3
_M_ARENA_MAX = -8

_IS_LINUX = sys.platform.startswith("linux")

# Serializes ONNX Runtime inference across request threads. ONNX sessions are
# thread-safe, but each concurrent Run() allocates its own activation buffers,
# so unbounded parallel inference is the single easiest way to blow the memory
# limit. Holding this while running the model bounds peak transient memory to
# one inference regardless of how many requests arrive at once.
MODEL_LOCK = threading.Lock()

# Serializes the *construction* of ONNX sessions, which is separate from running
# them. Model loading is memoized with functools.lru_cache, but lru_cache only
# locks its own bookkeeping - it does not hold a lock while calling the wrapped
# function, so two threads that miss the cache at the same time BOTH build a
# session. Each one is ~100 MB, so a pair of simultaneous first requests after a
# deploy would spike ~200 MB above baseline before one copy is discarded, which
# is enough to trip the memory limit on a small instance. Loading and inference
# never nest, so this is a separate lock rather than reusing MODEL_LOCK.
MODEL_LOAD_LOCK = threading.Lock()

# Default cap on the Starlette/anyio thread pool that runs sync endpoints.
# Each in-flight request holds its own candidate pool, tokenized batches and
# response tree, so this multiplies peak memory directly. Two keeps a single slow
# LLM call from blocking the queue while bounding that multiplier; override with
# MAX_REQUEST_THREADS on a larger instance.
DEFAULT_MAX_REQUEST_THREADS = 2

_libc = None
_libc_loaded = False


def _get_libc():
    """Return a ctypes handle to libc, or None when unavailable (e.g. Windows)."""
    global _libc, _libc_loaded
    if _libc_loaded:
        return _libc
    _libc_loaded = True
    if not _IS_LINUX:
        return None
    try:
        name = ctypes.util.find_library("c") or "libc.so.6"
        _libc = ctypes.CDLL(name, use_errno=True)
    except Exception:  # pragma: no cover - non-glibc platforms
        _libc = None
    return _libc


def configure_allocator() -> None:
    """Cap glibc arenas and make the allocator return pages to the OS sooner.

    ``MALLOC_ARENA_MAX`` is normally set as an environment variable, but glibc
    only reads it at startup, so it is useless once Python is running. The same
    knob is reachable at runtime through ``mallopt``, which takes effect for
    arenas created afterwards - i.e. before the request thread pool spins up.
    Without this, every worker thread can create its own 64 MB arena and RSS
    grows with thread count instead of with real usage.
    """
    libc = _get_libc()
    if libc is None:
        return
    try:
        arena_max = int(os.getenv("MALLOC_ARENA_MAX", "2"))
        libc.mallopt(_M_ARENA_MAX, arena_max)
        # Return freed top-of-heap memory to the OS at a 64 KB threshold instead
        # of glibc's adaptive default, which can grow to many MB per arena.
        libc.mallopt(_M_TRIM_THRESHOLD, 64 * 1024)
        # Serve large blocks (transformer activations) with mmap so freeing them
        # unmaps immediately rather than parking them in the heap.
        libc.mallopt(_M_MMAP_THRESHOLD, 256 * 1024)
    except Exception:  # pragma: no cover
        pass


def configure_thread_limits() -> None:
    """Pin every native math library to a single thread.

    faiss, ONNX Runtime, and NumPy's BLAS each default to one thread per CPU and
    allocate per-thread scratch space. On a fractional-CPU instance the extra
    threads buy no speed and cost real memory, so they are pinned to one. The
    environment variables must be set before the libraries are imported, which
    is why this runs at process start.
    """
    for var in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ.setdefault(var, "1")
    # The Rust tokenizers library forks a thread pool per tokenizer otherwise.
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def configure_faiss_threads() -> None:
    """Pin faiss to one OpenMP thread (safe no-op if faiss is not imported)."""
    faiss = sys.modules.get("faiss")
    if faiss is None:
        return
    try:
        faiss.omp_set_num_threads(1)
    except Exception:  # pragma: no cover
        pass


def release_memory(collect: bool = True) -> None:
    """Return freed heap pages to the OS.

    CPython's allocator frees objects back to its own pools and glibc keeps those
    pools mapped, so resident memory only ever grows across requests even when
    nothing is retained. Calling ``malloc_trim`` after a request releases the
    unused pages, which is what keeps RSS flat over a long-running deploy.
    """
    if collect:
        gc.collect()
    libc = _get_libc()
    if libc is None:
        return
    try:
        libc.malloc_trim(0)
    except Exception:  # pragma: no cover
        pass


def rss_mb() -> float | None:
    """Current resident set size in MB, or None if it cannot be determined.

    Reads ``/proc/self/statm`` directly so no extra dependency (psutil) is
    needed just to report memory.
    """
    if _IS_LINUX:
        try:
            with open("/proc/self/statm", "r") as fh:
                pages = int(fh.read().split()[1])
            return round(pages * os.sysconf("SC_PAGE_SIZE") / (1024 * 1024), 1)
        except Exception:
            return None
    # Best effort elsewhere (local development; production is Linux).
    try:
        import resource

        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        divisor = 1024 * 1024 if platform.system() == "Darwin" else 1024
        return round(usage / divisor, 1)
    except Exception:
        pass
    try:  # Windows has neither /proc nor resource
        import psutil

        return round(psutil.Process().memory_info().rss / (1024 * 1024), 1)
    except Exception:
        return None


def limit_thread_pool(max_threads: int | None = None) -> int | None:
    """Cap the anyio thread pool Starlette uses for sync endpoints.

    Must be called from inside the running event loop (i.e. during lifespan
    startup). Returns the cap applied, or None if it could not be applied.
    """
    if max_threads is None:
        max_threads = int(
            os.getenv("MAX_REQUEST_THREADS", str(DEFAULT_MAX_REQUEST_THREADS))
        )
    try:
        import anyio.to_thread

        limiter = anyio.to_thread.current_default_thread_limiter()
        limiter.total_tokens = max_threads
        return max_threads
    except Exception:  # pragma: no cover
        return None


def configure_low_memory_runtime() -> None:
    """Apply every process-level memory setting. Safe to call more than once."""
    configure_thread_limits()
    configure_allocator()
    configure_faiss_threads()
