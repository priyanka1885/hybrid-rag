"""
FastAPI application for Hybrid RAG for Financial Reports.

Endpoints:
    GET  /health      -> service + LLM + index status
    POST /ask         -> grounded answer + citations + verification + retrieval details
    GET  /evaluation  -> real dataset stats + retrieval comparison metrics

Startup loads configuration and indexes once (FAISS + BM25 + chunk store) and
constructs the model wrappers. It does NOT rebuild the dataset. If indexes are
missing, the app still starts and reports the problem via /health so the user
gets a helpful message instead of a crash.
"""
from __future__ import annotations

import sys
import threading
from pathlib import Path

# Allow `uvicorn backend.main:app` and `uvicorn main:app` (run from backend/).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from contextlib import asynccontextmanager  # noqa: E402

from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402

from backend.api.schemas import (  # noqa: E402
    AskRequest,
    AskResponse,
    EvaluationResponse,
    HealthResponse,
)
from backend.config import settings  # noqa: E402
from backend.evaluation.evaluator import dataset_stats, load_results  # noqa: E402
from backend.generation.llm import OllamaClient  # noqa: E402
from backend.rag_engine import RagEngine  # noqa: E402
from backend.retrieval.metadata import build_explicit_filter  # noqa: E402

# Engine is loaded lazily so the server can start even if indexes are missing.
_engine: RagEngine | None = None
_load_error: str | None = None


def _warmup_llm() -> None:
    """Best-effort background warmup so the first question isn't a cold start."""
    try:
        client = OllamaClient()
        status = client.health()
        if status.get("reachable") and status.get("model_available"):
            print(f"[startup] Warming up LLM '{client.model}' in background…", file=sys.stderr)
            if client.warmup():
                print("[startup] LLM warm and ready.", file=sys.stderr)
    except Exception as exc:  # never let warmup break startup
        print(f"[startup] LLM warmup skipped: {exc}", file=sys.stderr)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _engine, _load_error
    try:
        _engine = RagEngine().load()
        _load_error = None
    except Exception as exc:  # indexes missing or corrupt
        _engine = None
        _load_error = str(exc)
        print(f"[startup] Could not load indexes: {exc}", file=sys.stderr)

    # Preload the LLM into memory in a background thread so startup isn't blocked.
    threading.Thread(target=_warmup_llm, daemon=True).start()
    yield


app = FastAPI(
    title="Hybrid RAG for Financial Reports",
    description="Hybrid retrieval (Dense + BM25) + reranking + grounded Llama 3.1 generation with citation verification.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    llm_status = OllamaClient().health()
    return HealthResponse(
        status="ok",
        llm=llm_status,
        indexes_loaded=_engine is not None,
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


@app.get("/evaluation", response_model=EvaluationResponse)
def evaluation() -> EvaluationResponse:
    try:
        stats = dataset_stats()
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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.main:app", host=settings.API_HOST, port=settings.API_PORT, reload=False)
