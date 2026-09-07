"""Tests for the FastAPI endpoints and answerability behavior."""
from conftest import requires_indexes
from fastapi.testclient import TestClient

from backend.main import app


def test_health():
    with TestClient(app) as client:
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert "llm" in body
        assert "indexes_loaded" in body


@requires_indexes
def test_ask_empty_question_rejected():
    with TestClient(app) as client:
        resp = client.post("/ask", json={"question": "   "})
        assert resp.status_code == 400


@requires_indexes
def test_ask_out_of_scope_question():
    with TestClient(app) as client:
        resp = client.post("/ask", json={"question": "What is the capital of France?"})
        assert resp.status_code == 200
        body = resp.json()
        # Unrelated question must be refused, not answered.
        assert body["status"] in ("out_of_scope", "insufficient_evidence")
        assert "outside the scope" in body["answer"].lower() or "sufficient evidence" in body["answer"].lower()
        assert body["citations"] == []
        # Retrieval details are still exposed for inspection.
        assert "dense_results" in body["retrieval_details"]
        assert "bm25_results" in body["retrieval_details"]


@requires_indexes
def test_ask_returns_retrieval_details_structure():
    with TestClient(app) as client:
        resp = client.post("/ask", json={"question": "Clicks revenue 2022"})
        assert resp.status_code == 200
        rd = resp.json()["retrieval_details"]
        for key in ("dense_results", "bm25_results", "hybrid_results", "reranked_results"):
            assert key in rd


@requires_indexes
def test_evaluation_endpoint():
    with TestClient(app) as client:
        resp = client.get("/evaluation")
        assert resp.status_code == 200
        body = resp.json()
        # Real dataset stats (never hardcoded).
        assert body["num_documents"] > 0
        assert body["num_chunks"] > 0
