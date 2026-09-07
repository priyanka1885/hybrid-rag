"""Tests for dense, BM25, hybrid retrieval and reranking."""
from conftest import requires_indexes

QUESTION = "What was Clicks Group revenue in 2022?"


@requires_indexes
def test_dense_retrieval_returns_scored_results(dense):
    results = dense.search(QUESTION, top_k=5)
    assert len(results) > 0
    r = results[0]
    for key in ("chunk_id", "document_name", "page_number", "dense_score", "text"):
        assert key in r
    # Dense scores are cosine sims in [-1, 1].
    assert -1.01 <= r["dense_score"] <= 1.01


@requires_indexes
def test_bm25_retrieval_returns_scored_results(bm25):
    results = bm25.search(QUESTION, top_k=5)
    assert len(results) > 0
    r = results[0]
    assert "bm25_score" in r and r["bm25_score"] >= 0
    assert r["page_number"] >= 1


@requires_indexes
def test_hybrid_fusion_combines_scores(hybrid):
    out = hybrid.search(QUESTION, top_k=5)
    assert out["dense_results"] and out["bm25_results"] and out["hybrid_results"]
    top = out["hybrid_results"][0]
    # Hybrid keeps all three scores inspectable.
    assert "hybrid_score" in top
    assert "dense_norm" in top and "bm25_norm" in top
    # Hybrid results sorted descending by hybrid_score.
    scores = [r["hybrid_score"] for r in out["hybrid_results"]]
    assert scores == sorted(scores, reverse=True)


@requires_indexes
def test_hybrid_alpha_extremes(hybrid):
    # alpha=1 -> pure dense ordering signal; alpha=0 -> pure bm25 signal.
    dense_only = hybrid.search(QUESTION, alpha=1.0, top_k=5)["hybrid_results"]
    bm25_only = hybrid.search(QUESTION, alpha=0.0, top_k=5)["hybrid_results"]
    assert dense_only and bm25_only


@requires_indexes
def test_reranker_orders_and_limits(hybrid, reranker):
    candidates = hybrid.search(QUESTION, top_k=10)["hybrid_results"]
    top_k = 4
    reranked = reranker.rerank(QUESTION, candidates, top_k=top_k)
    assert 0 < len(reranked) <= top_k
    scores = [r["rerank_score"] for r in reranked]
    assert scores == sorted(scores, reverse=True)
    assert reranked[0]["final_rank"] == 1
