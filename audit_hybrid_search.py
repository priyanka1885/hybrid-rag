"""
audit_hybrid_search.py — End-to-end diagnostic of the Hybrid Search pipeline.

Verifies that the three retrieval stages work individually AND fuse together
correctly:

    BM25 (lexical)  ─┐
                     ├─> Reciprocal Rank Fusion ─> Cross-Encoder Reranker ─> top-5
    FAISS (dense)   ─┘

It is a READ-ONLY audit: it loads the persisted indexes (ChunkStore + FAISS +
BM25) and the cross-encoder, then runs a battery of queries and prints the RAW
scores at every stage so you can see the components working together.

Sections
  A. Component Verification
       A1. BM25 only   — exact number/terminology query
       A2. FAISS only  — semantic paraphrase query
       A3. Hybrid + Reranker — combined pass, final top-5 reranked
  B. Edge Cases (robustness)
       B1. Exact phrase / number
       B2. Semantic variation (paraphrase)
       B3. Cross-company comparison + metadata filtering check

Run:  python audit_hybrid_search.py
"""
from __future__ import annotations

import sys
import time

# Force UTF-8 stdout so the box-drawing / arrow glyphs in the report survive
# being piped or redirected to a file on Windows (the default cp1252 codec
# cannot encode '└─' / '<==' and would crash the run under redirection).
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):  # pragma: no cover - non-reconfigurable stream
    pass

from backend.config import settings
from backend.rag_engine import RagEngine, apply_table_boost, build_retrieval_query
from backend.retrieval.bm25 import BM25Retriever, tokenize
from backend.retrieval.dense import DenseRetriever
from backend.retrieval.hybrid import HybridRetriever
from backend.retrieval.metadata import build_explicit_filter, detect_filter
from backend.retrieval.store import ChunkStore
from backend.reranking.reranker import Reranker

# --- formatting helpers -----------------------------------------------------
BAR = "=" * 84
SUB = "-" * 84


def h1(title: str) -> None:
    print("\n" + BAR)
    print(title)
    print(BAR)


def h2(title: str) -> None:
    print("\n" + SUB)
    print(title)
    print(SUB)


def _short(cid: str, n: int = 14) -> str:
    return (cid[:n]) if cid else "?"


def _preview(text: str, n: int = 90) -> str:
    t = " ".join(text.split())
    return (t[:n] + "…") if len(t) > n else t


def _fmt(x, nd=4):
    return f"{x:.{nd}f}" if isinstance(x, (int, float)) else str(x)


def _doc_tag(r: dict) -> str:
    """Compact 'company p<page>' tag so cross-company leakage is obvious."""
    return f"{r.get('document_name', '?')} p{r.get('page_number')}"


# The Sasol Scope 1 answer figure (58 644). We detect the answer-bearing chunk
# by CONTENT rather than a hardcoded id, so the check stays valid if chunk ids
# change: normalize away spaces/commas and look for the literal figure.
def _contains_answer(text: str) -> bool:
    norm = text.replace(" ", "").replace(",", "").lower()
    return "58644" in norm


# --- pretty printers for each stage ----------------------------------------
def print_bm25(results: list[dict], k: int) -> None:
    print(f"BM25 raw scores (top {k}) — lexical term-frequency, NO embeddings:")
    print(f"  {'rk':>2}  {'bm25_score':>11}  {'chunk_id':<15} {'type':<8} loc / preview")
    for r in results[:k]:
        print(
            f"  {r['rank']:>2}  {_fmt(r['bm25_score']):>11}  {_short(r['chunk_id']):<15} "
            f"{r.get('content_type', 'text'):<8} {_doc_tag(r)}"
        )
        print(f"        └─ {_preview(r['text'])}")


def print_dense(results: list[dict], k: int) -> None:
    print(f"FAISS raw scores (top {k}) — cosine similarity in [-1, 1]:")
    print(f"  {'rk':>2}  {'dense_score':>11}  {'chunk_id':<15} {'type':<8} loc / preview")
    for r in results[:k]:
        print(
            f"  {r['rank']:>2}  {_fmt(r['dense_score']):>11}  {_short(r['chunk_id']):<15} "
            f"{r.get('content_type', 'text'):<8} {_doc_tag(r)}"
        )
        print(f"        └─ {_preview(r['text'])}")


def print_hybrid(results: list[dict], k: int) -> None:
    print(f"HYBRID fused (top {k}) — Reciprocal Rank Fusion (alpha-weighted):")
    print(
        f"  {'rk':>2}  {'rrf':>8} {'hybrid':>7}  {'dRnk':>4} {'bRnk':>4}  "
        f"{'dense':>7} {'bm25':>8}  {'chunk_id':<15} loc"
    )
    for r in results[:k]:
        dr = r.get("dense_rank")
        br = r.get("bm25_rank")
        print(
            f"  {r['rank']:>2}  {_fmt(r.get('rrf_score'), 5):>8} {_fmt(r.get('hybrid_score'), 4):>7}  "
            f"{str(dr) if dr else '-':>4} {str(br) if br else '-':>4}  "
            f"{_fmt(r.get('dense_score'), 3):>7} {_fmt(r.get('bm25_score'), 3):>8}  "
            f"{_short(r['chunk_id']):<15} {_doc_tag(r)}"
        )


def print_reranked(results: list[dict], k: int) -> None:
    print(f"CROSS-ENCODER RERANKED (final top {k}) — joint (question, chunk) relevance:")
    print(
        f"  {'fr':>2}  {'rerank_score':>12}  {'rrf_was':>8} {'dRnk':>4} {'bRnk':>4}  "
        f"{'chunk_id':<15} loc / preview"
    )
    for r in results[:k]:
        dr = r.get("dense_rank")
        br = r.get("bm25_rank")
        print(
            f"  {r.get('final_rank'):>2}  {_fmt(r.get('rerank_score')):>12}  "
            f"{_fmt(r.get('rrf_score'), 5):>8} {str(dr) if dr else '-':>4} {str(br) if br else '-':>4}  "
            f"{_short(r['chunk_id']):<15} {_doc_tag(r)}"
        )
        print(f"        └─ {_preview(r['text'], 110)}")


def print_evidence(evidence: list[dict], k_rerank: int) -> None:
    """Print the exact evidence set _assemble_evidence() hands to the LLM.

    'origin' shows HOW each chunk got in: 'rerank-topK' (the cross-encoder's own
    top RERANK_TOP_K) or 'SAFETY-NET' (force-included by hybrid/per-modality
    rank even though the cross-encoder demoted it). 'ans?' flags the chunk that
    actually carries the 58 644 figure.
    """
    print(f"FINAL EVIDENCE -> LLM  ({len(evidence)} chunks = cross-encoder top-{k_rerank} "
          f"+ rank-aware safety net):")
    print(
        f"  {'#':>2}  {'origin':<11} {'rerank':>8} {'dRnk':>4} {'bRnk':>4} {'rrf':>8}  "
        f"{'ans?':<4} {'chunk_id':<15} loc"
    )
    for e in evidence:
        origin = "SAFETY-NET" if e.get("safety_net") else "rerank-topK"
        dr = e.get("dense_rank")
        br = e.get("bm25_rank")
        ans = "YES" if _contains_answer(e["text"]) else "-"
        marker = "  <== 58 644" if _contains_answer(e["text"]) else ""
        print(
            f"  {e.get('evidence_rank'):>2}  {origin:<11} {_fmt(e.get('rerank_score')):>8} "
            f"{str(dr) if dr else '-':>4} {str(br) if br else '-':>4} {_fmt(e.get('rrf_score'), 5):>8}  "
            f"{ans:<4} {_short(e['chunk_id']):<15} {_doc_tag(e)}{marker}"
        )


def confirm_safety_net(evidence: list[dict], rerank_topk: list[dict]) -> None:
    """State plainly whether the 58 644 table chunk reached the LLM, and how."""
    hits = [e for e in evidence if _contains_answer(e["text"])]
    in_topk = any(_contains_answer(e["text"]) for e in rerank_topk)
    print()
    if not hits:
        print("  RESULT: the '58 644' Scope 1 table chunk is NOT in the final LLM context.")
        return
    e = hits[0]
    via = "SAFETY NET" if e.get("safety_net") else "cross-encoder top-K"
    print(f"  RESULT: the '58 644' Scope 1 table chunk ({_short(e['chunk_id'])}) IS in the "
          f"final LLM context (evidence #{e.get('evidence_rank')}, via {via}).")
    if e.get("safety_net") and not in_topk:
        print("          The cross-encoder had demoted it BELOW the top-K; the rank-aware "
              "safety net rescued it — exactly the guarantee we wanted to verify.")
    elif not e.get("safety_net"):
        print("          It ranked into the cross-encoder top-K on its own (no safety net needed).")


# --- pipeline holder --------------------------------------------------------
class Pipeline:
    def __init__(self):
        h2("Loading indexes and models (one-time)")
        t0 = time.time()
        self.store = ChunkStore.load()
        print(f"  ChunkStore   : {len(self.store)} chunks")
        self.dense = DenseRetriever(self.store).load()
        print(f"  FAISS index  : {self.dense.index.ntotal} vectors, dim={self.dense.index.d}, "
              f"model={self.dense.meta.get('embedding_model')}")
        self.bm25 = BM25Retriever(self.store).load()
        print(f"  BM25 index   : corpus_size={self.bm25.bm25.corpus_size}")
        self.hybrid = HybridRetriever(self.dense, self.bm25)
        self.reranker = Reranker()
        print(f"  Reranker     : {self.reranker.model_name} (cross-encoder, lazy-loaded)")
        print(f"  Config       : TOP_K_DENSE={settings.TOP_K_DENSE} TOP_K_BM25={settings.TOP_K_BM25} "
              f"RRF_K={settings.RRF_K} ALPHA={settings.HYBRID_ALPHA} RERANK_TOP_K={settings.RERANK_TOP_K}")
        print(f"  Loaded in {time.time() - t0:.1f}s")


# === SECTION A: COMPONENT VERIFICATION =====================================
def a1_bm25_only(p: Pipeline, k: int = 8) -> None:
    h1("A1. BM25 ONLY  —  exact number + terminology query")
    q = "Sasol Scope 1 58 644 emissions 2023"
    print(f"Query      : {q!r}")
    print(f"Tokenized  : {tokenize(q)}   (note: '58 644' -> canonicalized numeric token)")
    print()
    results = p.bm25.search(q, top_k=k)
    print_bm25(results, k)


def a2_faiss_only(p: Pipeline, k: int = 8) -> None:
    h1("A2. FAISS ONLY  —  semantic paraphrase query")
    q = "What were Sasol's direct Scope 1 GHG emissions in 2023?"
    print(f"Query      : {q!r}")
    print("(no exact '58 644' token — relies purely on embedding similarity)\n")
    results = p.dense.search(q, top_k=k)
    print_dense(results, k)


def a3_hybrid_rerank(p: Pipeline) -> None:
    h1("A3. HYBRID + RERANKER  —  combined pass, final top-5")
    q = "What were Sasol's direct Scope 1 GHG emissions in 2023?"
    print(f"Query      : {q!r}")
    filt = detect_filter(q)
    print(f"Auto filter: {filt.describe()}")
    expanded = build_retrieval_query(q)
    if expanded.changed:
        print(f"Expanded retrieval query: {expanded.retrieval_text!r}")
    print()

    fused = p.hybrid.search(expanded.retrieval_text, metadata_filter=filt if not filt.is_empty() else None)

    h2("Stage 1 — BM25 leg (raw)")
    print_bm25(fused["bm25_results"], 6)
    h2("Stage 2 — FAISS leg (raw)")
    print_dense(fused["dense_results"], 6)
    h2("Stage 3 — RRF fusion")
    print_hybrid(fused["hybrid_results"], 8)

    h2("Stage 4 — Cross-encoder rerank (full pool, top-5 shown)")
    pool = fused.get("candidate_pool") or fused["hybrid_results"]
    print(f"Reranking candidate pool of {len(pool)} fused chunks…\n")
    # Rerank the FULL pool exactly as RagEngine does (incl. table-aware boost),
    # so _assemble_evidence has the same input it sees in production.
    reranked_full = p.reranker.rerank(q, pool, top_k=len(pool))
    reranked_full = apply_table_boost(q, reranked_full)
    print_reranked(reranked_full[:5], 5)

    print("\nFINAL top-5 reranked chunk IDs:")
    for r in reranked_full[:5]:
        print(f"  [{r['final_rank']}] {r['chunk_id']}  rerank={_fmt(r['rerank_score'])}  ({_doc_tag(r)})")

    h2("Stage 5 — _assemble_evidence(): cross-encoder top-K + rank-aware safety net")
    print(f"Config: RERANK_TOP_K={settings.RERANK_TOP_K}  "
          f"HYBRID_SAFETY_TOP_K={settings.HYBRID_SAFETY_TOP_K}  "
          f"MODALITY_SAFETY_TOP_K={settings.MODALITY_SAFETY_TOP_K}")
    print("(this is the EXACT evidence list RagEngine.answer() hands to the LLM)\n")
    evidence = RagEngine._assemble_evidence(reranked_full, fused)
    print_evidence(evidence, settings.RERANK_TOP_K)
    confirm_safety_net(evidence, reranked_full[: settings.RERANK_TOP_K])


# === SECTION B: EDGE CASES =================================================
def _run_full(p: Pipeline, q: str, filt=None, show_legs: int = 4,
              check_answer: bool = True) -> list[dict]:
    """Run the whole pipeline for one query and print every stage's raw scores,
    ending with the assembled LLM evidence (cross-encoder top-K + safety net).

    Mirrors production: dense+BM25 see the controlled EXPANDED recall query,
    while the cross-encoder reranker keeps the ORIGINAL question."""
    active = filt if (filt and not filt.is_empty()) else None
    expanded = build_retrieval_query(q)
    if expanded.changed:
        print(f"Expanded retrieval query: {expanded.retrieval_text!r}")
        print(f"  (concepts={expanded.concepts} added={expanded.added_terms} "
              f"years={expanded.normalized_years})\n")
    fused = p.hybrid.search(expanded.retrieval_text, metadata_filter=active)
    print_bm25(fused["bm25_results"], show_legs)
    print()
    print_dense(fused["dense_results"], show_legs)
    print()
    print_hybrid(fused["hybrid_results"], 5)
    print()
    pool = fused.get("candidate_pool") or fused["hybrid_results"]
    reranked_full = p.reranker.rerank(q, pool, top_k=len(pool))
    reranked_full = apply_table_boost(q, reranked_full)
    print_reranked(reranked_full[:5], 5)
    print()
    evidence = RagEngine._assemble_evidence(reranked_full, fused)
    print_evidence(evidence, settings.RERANK_TOP_K)
    if check_answer:
        confirm_safety_net(evidence, reranked_full[: settings.RERANK_TOP_K])
    return reranked_full[:5]


def b1_exact_number(p: Pipeline) -> None:
    h1("B1. EDGE — exact phrase / number  'Scope 1 58 644'")
    q = "Scope 1 58 644"
    print(f"Query: {q!r}   (tokenized: {tokenize(q)})\n")
    _run_full(p, q, detect_filter(q))


def b2_semantic_variation(p: Pipeline) -> None:
    h1("B2. EDGE — semantic variation (paraphrase, FYxx shorthand)")
    q = "How much direct carbon output did Sasol produce in FY2023?"
    print(f"Query: {q!r}")
    filt = detect_filter(q)
    print(f"Auto filter: {filt.describe()}   (FY2023 -> year 2023, 'Sasol' -> company)\n")
    _run_full(p, q, filt)


def b3_cross_company(p: Pipeline) -> None:
    h1("B3. EDGE — cross-company comparison + metadata filtering check")
    q = "Compare Scope 1 emissions between Sasol and Impala Platinum"

    h2("B3a. NO filter (auto-detect) — should surface BOTH companies")
    filt = detect_filter(q)
    print(f"Query      : {q!r}")
    print(f"Auto filter: {filt.describe()}\n")
    reranked = _run_full(p, q, filt, check_answer=False)
    companies = {r.get("document_name") for r in reranked}
    print(f"\n  -> distinct companies in final top-5: {sorted(companies)}")

    h2("B3b. EXPLICIT filter company=Sasol — should EXCLUDE all other companies")
    sasol = build_explicit_filter(company="Sasol")
    print(f"Explicit filter: {sasol.describe()}\n")
    allowed = p.store.allowed_indices(sasol)
    print(f"Eligible chunks after filter: {len(allowed) if allowed is not None else 'ALL'}\n")
    fused = p.hybrid.search(q, metadata_filter=sasol)
    print_hybrid(fused["hybrid_results"], 6)
    leaked = [
        r for r in fused["hybrid_results"]
        if "sasol" not in (r.get("document_name", "").lower())
    ]
    print(f"\n  -> cross-company leakage past the Sasol filter: {len(leaked)} chunks "
          f"({'NONE — filter clean' if not leaked else 'LEAK DETECTED'})")


# ---------------------------------------------------------------------------
def main() -> None:
    h1("HYBRID SEARCH PIPELINE AUDIT  (FAISS + BM25 + RRF + Cross-Encoder)")
    p = Pipeline()

    # A. Component verification
    a1_bm25_only(p)
    a2_faiss_only(p)
    a3_hybrid_rerank(p)

    # B. Edge cases
    b1_exact_number(p)
    b2_semantic_variation(p)
    b3_cross_company(p)

    h1("AUDIT COMPLETE")
    print("All three stages produced raw scores above. Review that:")
    print("  • BM25 ranks exact-number/terminology hits highly (A1, B1)")
    print("  • FAISS ranks paraphrases highly without the literal tokens (A2, B2)")
    print("  • RRF fuses both and the cross-encoder promotes the true answer (A3)")
    print("  • The metadata filter cleanly scopes cross-company queries (B3)")


if __name__ == "__main__":
    main()
