"""Tests for the retrieval/fusion/tokenizer/evidence-assembly fixes.

These target the diagnosed root cause of the "insufficient evidence" fallback:
strong retrieval hits being suppressed by fusion or cut by the cross-encoder
before reaching the LLM, plus numeric lexical retrieval. Pure-logic tests here;
model-dependent retrieval quality is exercised in test_retrieval.py.
"""
from backend.rag_engine import RagEngine
from backend.retrieval.bm25 import tokenize


# --- Section 6: financial-aware BM25 tokenization --------------------------
def test_space_grouped_thousands_tokenize_equal():
    # "64 392" and "64,392" and "64392" must produce the SAME lexical token so
    # an exact-number query hits a table row regardless of source formatting.
    assert tokenize("64 392") == tokenize("64,392") == tokenize("64392") == ["64392"]


def test_comma_decimal_tokenizes_like_dot_decimal():
    assert tokenize("63,4") == tokenize("63.4") == ["63.4"]


def test_large_space_grouped_number():
    assert tokenize("1 234 567") == ["1234567"]


def test_percentage_matches_on_numeric_part():
    # The numeric part is what makes the query and doc match lexically.
    assert "12.5" in tokenize("12.5%")


def test_year_is_preserved_as_token():
    assert "2023" in tokenize("in 2023 the company")


def test_words_and_numbers_coexist():
    toks = tokenize("Revenue was R 1,234 million")
    assert "revenue" in toks and "1234" in toks and "million" in toks


# --- Sections 2/3: rank-aware evidence assembly (safety net) ---------------
def _row(cid, **kw):
    base = {
        "chunk_id": cid,
        "document_name": "Doc",
        "page_number": 1,
        "content_type": "text",
        "text": f"text-{cid}",
    }
    base.update(kw)
    return base


def test_safety_net_includes_top_hybrid_chunk_cut_by_reranker():
    # Cross-encoder ranks a table row LAST, but both retrievers rank it #1.
    # The rank-aware safety net must still put it into the LLM evidence.
    gt = _row("gt", content_type="table")
    reranked_full = [_row(f"r{i}", rerank_score=1.0 - i) for i in range(12)] + [
        dict(gt, rerank_score=-9.0)
    ]
    fused = {
        "hybrid_results": [dict(gt, hybrid_score=1.0)] + [_row(f"r{i}") for i in range(5)],
        "dense_results": [dict(gt)],
        "bm25_results": [dict(gt)],
    }
    evidence = RagEngine._assemble_evidence(reranked_full, fused)
    ids = {e["chunk_id"] for e in evidence}
    assert "gt" in ids  # recovered by the safety net despite being reranked last


def test_safety_net_includes_strong_single_modality_dense_hit():
    # A chunk dense ranks #2 but BM25 misses entirely (RRF discounts it) must
    # still reach the LLM via the per-modality safety net.
    gt = _row("gt")
    reranked_full = [_row(f"r{i}", rerank_score=1.0 - i) for i in range(12)] + [
        dict(gt, rerank_score=-5.0)
    ]
    fused = {
        "hybrid_results": [_row(f"r{i}") for i in range(5)],  # gt not in hybrid top
        "dense_results": [_row("r0"), dict(gt)],  # gt is dense #2
        "bm25_results": [_row("r1")],
    }
    evidence = RagEngine._assemble_evidence(reranked_full, fused)
    assert "gt" in {e["chunk_id"] for e in evidence}


def test_assembled_evidence_has_no_duplicates():
    gt = _row("gt")
    reranked_full = [dict(gt, rerank_score=5.0)] + [
        _row(f"r{i}", rerank_score=1.0 - i) for i in range(12)
    ]
    fused = {
        "hybrid_results": [dict(gt)],
        "dense_results": [dict(gt)],
        "bm25_results": [dict(gt)],
    }
    evidence = RagEngine._assemble_evidence(reranked_full, fused)
    ids = [e["chunk_id"] for e in evidence]
    assert len(ids) == len(set(ids))  # gt appears exactly once


def test_evidence_rank_is_sequential():
    reranked_full = [_row(f"r{i}", rerank_score=1.0 - i) for i in range(3)]
    fused = {"hybrid_results": [], "dense_results": [], "bm25_results": []}
    evidence = RagEngine._assemble_evidence(reranked_full, fused)
    assert [e["evidence_rank"] for e in evidence] == [1, 2, 3]
