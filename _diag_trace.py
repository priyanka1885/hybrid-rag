"""Temporary diagnostic: trace retrieval + rerank + verification for two questions.
Does NOT modify any pipeline files. Deleted after diagnosis."""
from __future__ import annotations

from backend.retrieval.store import ChunkStore
from backend.retrieval.dense import DenseRetriever
from backend.retrieval.bm25 import BM25Retriever
from backend.retrieval.hybrid import HybridRetriever
from backend.reranking.reranker import Reranker
from backend.citations.verifier import verify_claim, _financial_figures

store = ChunkStore.load()
dense = DenseRetriever(store).load()
bm25 = BM25Retriever(store).load()
hybrid = HybridRetriever(dense, bm25)
reranker = Reranker()

QUESTIONS = [
    "What are Sasol's total greenhouse gas emissions?",
    "What are Pick n Pay's total number of employees?",
]

# Chunks we know contain the true answer, for locating their rank.
ANSWER_CHUNKS = {
    "What are Sasol's total greenhouse gas emissions?": ["a371fddafa17e4fd", "38b23531b7271660"],
    "What are Pick n Pay's total number of employees?": ["bf34aa74525699d0", "112bf2f50af4ba79"],
}


def short(t, n=110):
    t = " ".join(t.split())
    return t[:n] + ("..." if len(t) > n else "")


for q in QUESTIONS:
    print("\n\n############################################################")
    print("QUESTION:", q)
    print("############################################################")
    fused = hybrid.search(q)
    dense_r = fused["dense_results"]
    bm25_r = fused["bm25_results"]
    hyb = fused["hybrid_results"]
    pool = fused["candidate_pool"]

    # rank position of the true answer chunk in each stage
    ans_ids = ANSWER_CHUNKS[q]

    def rankpos(results, key):
        for i, r in enumerate(results, 1):
            if r["chunk_id"] in ans_ids:
                return i, r.get(key)
        return None, None

    print("\n-- DENSE top 8 --")
    for r in dense_r[:8]:
        mark = " <== ANSWER" if r["chunk_id"] in ans_ids else ""
        print(f"  {r['rank']:>2} dense={r['dense_score']:.4f} p{r['page_number']} {r['chunk_id']} {r['content_type']}{mark} :: {short(r['text'],80)}")
    dp, ds = rankpos(dense_r, "dense_score")
    print(f"   answer chunk dense rank = {dp} score={ds}")

    print("\n-- BM25 top 8 --")
    for r in bm25_r[:8]:
        mark = " <== ANSWER" if r["chunk_id"] in ans_ids else ""
        print(f"  {r['rank']:>2} bm25={r['bm25_score']:.3f} p{r['page_number']} {r['chunk_id']} {r['content_type']}{mark} :: {short(r['text'],80)}")
    bp, bs = rankpos(bm25_r, "bm25_score")
    print(f"   answer chunk bm25 rank = {bp} score={bs}")

    print("\n-- HYBRID top 10 --")
    for r in hyb[:10]:
        mark = " <== ANSWER" if r["chunk_id"] in ans_ids else ""
        print(f"  {r['rank']:>2} hyb={r['hybrid_score']:.4f} (d={r['dense_norm']:.3f} b={r['bm25_norm']:.3f}) p{r['page_number']} {r['chunk_id']} {r['content_type']}{mark} :: {short(r['text'],70)}")
    hp, hs = rankpos(hyb, "hybrid_score")
    print(f"   answer chunk hybrid rank = {hp} score={hs}")

    # rank of answer chunk within full candidate pool handed to reranker
    pp, _ = rankpos(pool, "hybrid_score")
    print(f"   answer chunk position in candidate_pool (size {len(pool)}) = {pp}")

    reranked = reranker.rerank(q, pool)
    print(f"\n-- RERANKED top {len(reranked)} (this is what the LLM sees) --")
    for r in reranked:
        mark = " <== ANSWER" if r["chunk_id"] in ans_ids else ""
        print(f"  {r['final_rank']:>2} rerank={r['rerank_score']:.4f} p{r['page_number']} {r['chunk_id']} {r['content_type']}{mark} :: {short(r['text'],75)}")
    rp, rs = rankpos(reranked, "rerank_score")
    print(f"   answer chunk rerank position = {rp} score={rs}  (RERANK_TOP_K cutoff)")

    # Is the answer chunk even in the reranked top-k the LLM receives?
    in_context = any(r["chunk_id"] in ans_ids for r in reranked)
    print(f"   ANSWER CHUNK IN LLM CONTEXT? {in_context}")
