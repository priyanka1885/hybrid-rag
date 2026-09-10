"""Hybrid RAG for Financial Reports - backend package.

Native math libraries (ONNX Runtime, faiss, NumPy's BLAS) read their thread
count from the environment at import time and allocate per-thread scratch
space. Pinning them to a single thread here - before any submodule pulls those
libraries in - keeps resident memory proportional to real work instead of to
CPU count, which matters on a memory-limited instance. ``setdefault`` is used
so an explicit ``OMP_NUM_THREADS=8`` (e.g. for a local ingestion run) still
wins. See :mod:`backend.runtime`.
"""
from backend.runtime import configure_thread_limits

configure_thread_limits()
