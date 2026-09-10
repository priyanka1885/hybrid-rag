"""
FastAPI application for Hybrid RAG for Financial Reports.

Endpoints:
    GET  /health      -> service + LLM + index status (+ resident memory)
    POST /ask         -> grounded answer + citations + verification + retrieval details
    GET  /evaluation  -> real dataset stats + retrieval comparison metrics

Startup loads configuration and indexes once (FAISS + BM25 + chunk store) and
constructs the model wrappers. It does NOT rebuild the dataset. If indexes are
missing, the app still starts and reports the problem via /health so the user
gets a helpful message instead of a crash.

MEMORY
------
This process is memory-bound: two fp32 ONNX sessions plus the indexes put the
baseline around 310 MB, so a 512 MB instance has little headroom and any
avoidable allocation risks an out-of-memory restart. Three rules follow from
that and are enforced here:

* **Import nothing optional.** ``backend.evaluation.evaluator`` is imported
  inside the ``/evaluation`` handler, not at module scope, because its
  dependency chain used to drag ``pandas`` (~44 MB) into every deploy even
  though answering a question never needs it.
* **Compute each expensive thing once.** ``/health`` and ``/evaluation`` are
  polled repeatedly and their answers barely change, so both are cached. They
  previously re-fetched the remote model catalogue and re-read the whole corpus
  and QA CSV on every single request.
* **Give memory back.** :func:`backend.runtime.release_memory` runs after a
  request finishes so freed pages return to the OS instead of accumulating as
  ever-growing RSS, and concurrency is capped so parallel inference cannot
  multiply peak usage.
"""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

# Allow `uvicorn backend.main:app` and `uvicorn main:app` (run from backend/).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Imported before anything that pulls in numpy/faiss/onnxruntime so the native
# thread limits are in place when those libraries initialize.
from backend.runtime import (  # noqa: E402
    configure_low_memory_runtime,
    limit_thread_pool,
    release_memory,
    rss_mb,
)

configure_low_memory_runtime()

from contextlib import asynccontextmanager  # noqa: E402

from fastapi import FastAPI, HTTPException, Request  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402

from backend.api.schemas import (  # noqa: E402
    AskRequest,
    AskResponse,
    EvaluationResponse,
    HealthResponse,
)
from backend.config import settings  # noqa: E402
from backend.generation.llm import OllamaClient  # noqa: E402
from backend.rag_engine import RagEngine  # noqa: E402
from backend.retrieval.metadata import build_explicit_filter  # noqa: E402

# Engine is loaded lazily so the server can start even if indexes are missing.
_engine: RagEngine | None = None
_load_error: str | None = None

# One shared LLM client. It is stateless apart from the pooled HTTP session, and
# building a new one per request also rebuilt that pool.
_llm_client = OllamaClient()

# /evaluation is a static report: the chunk corpus and the QA CSV only change
# when ingestion re-runs, and the metrics come from a file written offline.
# Caching the assembled response keeps a page refresh from re-reading both.
_evaluation_cache: EvaluationResponse | None = None
_evaluation_lock = threading.Lock()

# Trim the heap at most this often, so a burst of small requests does not pay
# for a collection on every one.
_TRIM_MIN_INTERVAL_SECONDS = 5.0
_last_trim = 0.0
_trim_lock = threading.Lock()


def _maybe_release_memory() -> None:
    """Return freed pages to the OS, rate-limited."""
    global _last_trim
    now = time.monotonic()
    with _trim_lock:
        if now - _last_trim < _TRIM_MIN_INTERVAL_SECONDS:
            return
        _last_trim = now
    release_memory()


def _warmup_llm() -> None:
    """Best-effort startup check that the hosted LLM (OpenRouter) is reachable."""
    try:
        status = _llm_client.health()
        if status.get("reachable") and status.get("model_available"):
            print(
                f"[startup] LLM '{_llm_client.model}' is reachable via OpenRouter.",
                file=sys.stderr,
            )
            _llm_client.warmup()
        else:
            print("[startup] LLM not configured/reachable yet (set OPENROUTER_API_KEY).",
                  file=sys.stderr)
    except Exception as exc:  # never let this break startup
        print(f"[startup] LLM check skipped: {exc}", file=sys.stderr)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _engine, _load_error

    # Cap the thread pool Starlette uses for sync endpoints. The default is 40,
    # and because each in-flight /ask runs transformer inference that allocates
    # its own activation buffers, 40 simultaneous requests would allocate 40 sets
    # at once. Capping turns a memory spike into a short queue. Must happen
    # inside the running loop, hence here rather than at import time.
    capped = limit_thread_pool()

    try:
        _engine = RagEngine().load()
        _load_error = None
    except Exception as exc:  # indexes missing or corrupt
        _engine = None
        _load_error = str(exc)
        print(f"[startup] Could not load indexes: {exc}", file=sys.stderr)

    # Preload the LLM into memory in a background thread so startup isn't blocked.
    threading.Thread(target=_warmup_llm, daemon=True).start()

    # Index loading and ONNX graph optimization leave a lot of transient memory
    # behind; hand it back before serving so the steady-state baseline is honest.
    release_memory()
    print(
        f"[startup] ready (request threads={capped}, rss={rss_mb()} MB)",
        file=sys.stderr,
    )
    yield


app = FastAPI(
    title="Hybrid RAG for Financial Reports",
    description="Hybrid retrieval (Dense + BM25) + reranking + grounded Llama 3.1 generation with citation verification.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://hybrid-rag-frontend-fjdc.onrender.com",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def reclaim_memory(request: Request, call_next):
    """Release the heap after each request.

    A request builds a large, short-lived object graph (candidate chunks,
    tokenized batches, activation arrays, the serialized response). CPython
    returns those objects to its own pools and glibc keeps the pools mapped, so
    resident memory ratchets up request after request and eventually trips the
    instance limit even though nothing is actually retained. Trimming here is
    what keeps RSS flat across a long-running deploy.
    """
    response = await call_next(request)
    _maybe_release_memory()
    return response


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        llm=_llm_client.health(),
        indexes_loaded=_engine is not None,
        memory_mb=rss_mb(),
    )


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    question = (req.question or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question must not be empty.")
    if _engine is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Search indexes are not available. Run `python scripts/ingest.py` "
                f"to build them. ({_load_error or 'unknown error'})"
            ),
        )
    explicit_filter = build_explicit_filter(company=req.company, year=req.year)
    try:
        result = _engine.answer(question, metadata_filter=explicit_filter)
    except Exception as exc:  # never leak a raw stack trace
        raise HTTPException(status_code=500, detail=f"Failed to answer the question: {exc}")
    return AskResponse(**result)


def _build_evaluation() -> EvaluationResponse:
    """Assemble the /evaluation payload. Called at most once per process."""
    # Imported here, not at module scope: this module's dependency chain reaches
    # pandas, which costs ~44 MB resident and is never needed to answer a
    # question. See the MEMORY section in this module's docstring.
    from backend.evaluation.evaluator import dataset_stats, load_results

    try:
        # Reuse the engine's already-loaded chunk store instead of reading and
        # parsing the whole corpus a second time.
        stats = dataset_stats(store=_engine.store if _engine is not None else None)
    except Exception as exc:
        return EvaluationResponse(
            available=False,
            num_documents=0,
            num_chunks=0,
            num_eval_questions=0,
            message=f"Dataset stats unavailable: {exc}. Run ingestion first.",
        )

    results = load_results()
    if not results:
        return EvaluationResponse(
            available=False,
            num_documents=stats["num_documents"],
            num_chunks=stats["num_chunks"],
            num_eval_questions=stats["num_eval_questions"],
            message="No evaluation results yet. Run `python scripts/evaluate.py` to generate them.",
        )

    return EvaluationResponse(
        available=True,
        num_documents=stats["num_documents"],
        num_chunks=stats["num_chunks"],
        num_eval_questions=stats["num_eval_questions"],
        k=results.get("k"),
        methods=results.get("methods", {}),
        metric_names=results.get("metric_names", []),
    )


@app.get("/evaluation", response_model=EvaluationResponse)
def evaluation() -> EvaluationResponse:
    global _evaluation_cache
    cached = _evaluation_cache
    if cached is not None:
        return cached
    with _evaluation_lock:
        if _evaluation_cache is None:
            built = _build_evaluation()
            # Only cache a successful report, so a transient failure (e.g.
            # ingestion still running) is retried on the next request.
            if built.available:
                _evaluation_cache = built
            release_memory()
            return built
        return _evaluation_cache


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.main:app", host=settings.API_HOST, port=settings.API_PORT, reload=False)
