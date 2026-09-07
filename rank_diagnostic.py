"""
rank_diagnostic.py — Per-stage ranking diagnostic for the THREE key queries.

Traces exactly what RagEngine does to every candidate chunk for:

    Q1  "What were Sasol's direct Scope 1 GHG emissions in 2023?"      (works)
    Q2  "How much direct carbon output did Sasol produce in FY2023?"   (paraphrase)
    Q3  "Compare Scope 1 emissions between Sasol and Impala Platinum"  (comparison)

For each query it prints the full recall trace the task requires:

  1.  Original query
  2.  Expanded / normalized RETRIEVAL query (concept synonyms + FY->year)
  3.  BM25 top candidates
  4.  FAISS top candidates
  5.  RRF rank
  6.  Whether the target Scope-1 value chunk (58 644) appears
  7.  Its BM25 rank
  8.  Its FAISS rank
  9.  Its RRF rank
  10. Reranker rank
  11. Final evidence rank
  12. Reason if it is dropped

It mirrors the REAL RagEngine wiring exactly: same metadata-filter resolution,
the same controlled query expansion for recall, the same full-pool rerank on the
ORIGINAL question, the same table boost, and the same evidence assembly - so the
numbers match production. READ-ONLY: no generation, no writes.

Run:  python rank_diagnostic.py
"""
from __future__ import annotations

import sys

# UTF-8 stdout so the glyphs survive redirection to a file on Windows.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):  # pragma: no cover
    pass

from backend.config import settings
from backend.rag_engine import (
    RagEngine,
    apply_table_boost,
    build_retrieval_query,
    is_data_seeking_query,
)

QUERIES = [
    "What were Sasol's direct Scope 1 GHG emissions in 2023?",
    "How much direct carbon output did Sasol produce in FY2023?",
    "Compare Scope 1 emissions between Sasol and Impala Platinum",
]
TOP_N = 10

BAR = "=" * 120
SUB = "-" * 120


def _contains_answer(text: str) -> bool:
    """Detect the answer-bearing chunk by CONTENT (the 58 644 figure)."""
    norm = text.replace(" ", "").replace(",", "").lower()
    return "58644" in norm


def _source(chunk: dict) -> str:
    ct = chunk.get("content_type", "text")
    return "TABLE" if ct in ("table", "figure") else ct.upper()


def _r(x, nd=4, width=9):
    if x is None:
        return f"{'-':>{width}}"
    return f"{x:>{width}.{nd}f}" if isinstance(x, (int, float)) else f"{str(x):>{width}}"


def _rank(x, width=4):
    return f"{x:>{width}}" if x else f"{'-':>{width}}"


def run_query(engine: RagEngine, question: str) -> None:
    print("\n" + BAR)
    print(f"QUERY: {question!r}")
    print(BAR)

    # --- mirror RagEngine.answer() retrieval path exactly -------------------
    active_filter = engine._resolve_filter(question, None)
    expanded = build_retrieval_query(question)

    # (1) original / (2) expanded retrieval query
    print(f"[1] Original query        : {expanded.original!r}")
    print(f"[2] Expanded RETRIEVAL q  : {expanded.retrieval_text!r}")
    print(f"    data_seeking          : {is_data_seeking_query(question)}")
    print(f"    concepts fired        : {expanded.concepts or '(none)'}")
    print(f"    concept terms added   : {expanded.added_terms or '(none)'}")
    print(f"    FY->year normalized   : {expanded.normalized_years or '(none)'}")
    print(f"    metadata filter       : "
          f"{active_filter.describe() if active_filter else {'active': False}}")

    fused = engine.hybrid.search(expanded.retrieval_text, metadata_filter=active_filter)
    dense_results = fused["dense_results"]
    bm25_results = fused["bm25_results"]
    candidate_pool = fused.get("candidate_pool") or fused["hybrid_results"]

    # Full-pool cross-encoder rerank on the ORIGINAL question (as production),
    # then the same table-aware boost RagEngine.answer() applies.
    reranked_full = engine.reranker.rerank(question, candidate_pool, top_k=len(candidate_pool))
    reranked_full = apply_table_boost(question, reranked_full)
    evidence = engine._assemble_evidence(reranked_full, fused)

    boost_on = settings.ENABLE_TABLE_RERANK_BOOST and is_data_seeking_query(question)
    print(f"\n    table boost           : enabled={settings.ENABLE_TABLE_RERANK_BOOST} "
          f"data_seeking={is_data_seeking_query(question)} boost=+{settings.TABLE_RERANK_BOOST} "
          f"-> {'APPLIED' if boost_on else 'not applied'}")
    print(f"    stage sizes           : dense={len(dense_results)} bm25={len(bm25_results)} "
          f"pool={len(candidate_pool)} reranked={len(reranked_full)} "
          f"rerank_top_k={settings.RERANK_TOP_K} evidence={len(evidence)}")

    # --- rank lookups per stage --------------------------------------------
    bm25_rank = {r["chunk_id"]: i for i, r in enumerate(bm25_results, start=1)}
    dense_rank = {r["chunk_id"]: i for i, r in enumerate(dense_results, start=1)}
    rrf_rank = {c["chunk_id"]: i for i, c in enumerate(candidate_pool, start=1)}
    rerank_rank = {c["chunk_id"]: c["final_rank"] for c in reranked_full}
    rerank_score = {c["chunk_id"]: c["rerank_score"] for c in reranked_full}
    bm25_score = {r["chunk_id"]: r["bm25_score"] for r in bm25_results}
    dense_score = {r["chunk_id"]: r["dense_score"] for r in dense_results}
    evidence_pos = {e["chunk_id"]: e.get("evidence_rank") for e in evidence}
    evidence_via = {
        e["chunk_id"]: ("SAFETY-NET" if e.get("safety_net") else "rerank-topK") for e in evidence
    }

    # --- (3) BM25 top / (4) FAISS top / (5) RRF top ------------------------
    print("\n" + SUB)
    print(f"[3] BM25 top {TOP_N}")
    print(SUB)
    for r in bm25_results[:TOP_N]:
        print(f"  #{r['rank']:>2}  {r['chunk_id'][:16]:<16} {_source(r):<6} "
              f"bm25={_r(r['bm25_score']).strip():<9} p{r.get('page_number')} "
              f"{r.get('document_name','')[:34]:<34} {'<== 58 644' if _contains_answer(r['text']) else ''}")

    print("\n" + SUB)
    print(f"[4] FAISS top {TOP_N}")
    print(SUB)
    for r in dense_results[:TOP_N]:
        print(f"  #{r['rank']:>2}  {r['chunk_id'][:16]:<16} {_source(r):<6} "
              f"dense={_r(r['dense_score']).strip():<8} p{r.get('page_number')} "
              f"{r.get('document_name','')[:34]:<34} {'<== 58 644' if _contains_answer(r['text']) else ''}")

    print("\n" + SUB)
    print(f"[5] RRF fused top {TOP_N}")
    print(SUB)
    for i, c in enumerate(candidate_pool[:TOP_N], start=1):
        print(f"  #{i:>2}  {c['chunk_id'][:16]:<16} {_source(c):<6} "
              f"rrf={_r(c.get('rrf_score'), 5).strip():<9} "
              f"bm25#={_rank(bm25_rank.get(c['chunk_id'])).strip():<4} "
              f"faiss#={_rank(dense_rank.get(c['chunk_id'])).strip():<4} p{c.get('page_number')} "
              f"{'<== 58 644' if _contains_answer(c['text']) else ''}")

    # --- (6..12) target chunk survival trace -------------------------------
    print("\n" + SUB)
    print("TARGET Scope-1 value chunk (58 644) — full rank trace")
    print(SUB)
    answer_chunks = [c for c in candidate_pool if _contains_answer(c["text"])]
    # Also look wider than the pool to explain a recall miss precisely.
    in_bm25 = [r for r in bm25_results if _contains_answer(r["text"])]
    in_dense = [r for r in dense_results if _contains_answer(r["text"])]

    if not answer_chunks:
        print("  [6] target in fused candidate pool : NO")
        print(f"      target in BM25 top-{len(bm25_results)}  : "
              f"{'yes (rank ' + str(bm25_rank[in_bm25[0]['chunk_id']]) + ')' if in_bm25 else 'NO'}")
        print(f"      target in FAISS top-{len(dense_results)} : "
              f"{'yes (rank ' + str(dense_rank[in_dense[0]['chunk_id']]) + ')' if in_dense else 'NO'}")
        print("  [12] DROP REASON: target never entered the dense+BM25 candidate union "
              "(recall miss) -> not reranked, not in evidence.")
        return

    for c in answer_chunks:
        cid = c["chunk_id"]
        fr = rerank_rank.get(cid)
        in_topk = bool(fr and fr <= settings.RERANK_TOP_K)
        in_evidence = cid in evidence_pos
        print(f"  chunk_id                 : {cid}  ({_source(c)}, p{c.get('page_number')})")
        print(f"  [6]  in candidate pool   : YES")
        print(f"  [7]  BM25 rank / score   : {bm25_rank.get(cid, '-')} / {_r(bm25_score.get(cid)).strip()}")
        print(f"  [8]  FAISS rank / score  : {dense_rank.get(cid, '-')} / {_r(dense_score.get(cid)).strip()}")
        print(f"  [9]  RRF rank / score    : {rrf_rank.get(cid, '-')} / {_r(c.get('rrf_score'), 5).strip()}")
        print(f"  [10] Rerank rank / score : {fr} / {_r(rerank_score.get(cid), 4).strip()} "
              f"(in rerank top-{settings.RERANK_TOP_K}: {in_topk})")
        if in_evidence:
            print(f"  [11] Final evidence rank : #{evidence_pos[cid]} (via {evidence_via[cid]})")
            print(f"  [12] Drop reason         : NOT DROPPED — reaches the LLM.")
        else:
            print(f"  [11] Final evidence rank : NOT IN EVIDENCE")
            print(f"  [12] Drop reason         : demoted below rerank top-{settings.RERANK_TOP_K} "
                  "and not rescued by the safety net.")


def main() -> None:
    print(BAR)
    print("RANKING DIAGNOSTIC — three queries, per-stage scores (mirrors production)")
    print(BAR)
    engine = RagEngine().load()
    print(f"Corpus: {len(engine.store)} chunks | embedding={settings.EMBEDDING_MODEL}")
    print(f"Reranker: {settings.RERANKER_MODEL}")
    print(f"Config : RRF_K={settings.RRF_K} ALPHA={settings.HYBRID_ALPHA} "
          f"TOP_K_DENSE={settings.TOP_K_DENSE} TOP_K_BM25={settings.TOP_K_BM25} "
          f"RERANK_TOP_K={settings.RERANK_TOP_K}")

    for q in QUERIES:
        run_query(engine, q)

    print("\n" + BAR)
    print("DIAGNOSTIC COMPLETE")
    print(BAR)


if __name__ == "__main__":
    main()
