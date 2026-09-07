"""
Hybrid retrieval: transparent RANK-based fusion of dense and BM25 results.

    Dense results  ─┐
                    ├─> Reciprocal Rank Fusion (RRF) -> sort
    BM25 results   ─┘

Dense and lexical retrieval have complementary strengths, so we combine them.
We fuse by RANK (Reciprocal Rank Fusion) rather than by min-max-normalized
score. Why RRF for financial-report retrieval:

* It is scale-free. Min-max fusion assigns 0 to the missing modality, so a
  chunk found by only ONE retriever (e.g. a terse number-dense table row that
  BM25 ranks #1 but the dense bi-encoder misses) is halved and unfairly
  suppressed. RRF only ever ADDS a positive contribution for each retriever a
  chunk appears in, so strong single-modality hits are not zero-penalized.
* It is robust to the wildly different score distributions of a cosine
  bi-encoder vs BM25 term-frequency scores, which min-max normalization handles
  poorly when one distribution is skewed.
* It is transparent and standard. Every original score and rank is preserved on
  the candidate for inspection.

All scores are kept on each candidate (dense_score, bm25_score, dense_norm,
bm25_norm, dense_rank, bm25_rank, rrf_score, hybrid_score) so the pipeline stays
fully inspectable. ``hybrid_score`` is the min-max-normalized RRF value in
[0, 1] so downstream signal thresholds remain meaningful; ordering is by RRF.
"""
from __future__ import annotations

from backend.config import settings
from backend.retrieval.bm25 import BM25Retriever
from backend.retrieval.dense import DenseRetriever


def _min_max_normalize(values: dict[str, float]) -> dict[str, float]:
    if not values:
        return {}
    lo = min(values.values())
    hi = max(values.values())
    span = hi - lo
    if span <= 1e-12:
        # All equal -> neutral 1.0 if there is a positive signal, else 0.
        return {k: (1.0 if hi > 0 else 0.0) for k in values}
    return {k: (v - lo) / span for k, v in values.items()}


class HybridRetriever:
    def __init__(self, dense: DenseRetriever, bm25: BM25Retriever):
        self.dense = dense
        self.bm25 = bm25

    def search(
        self,
        question: str,
        top_k: int | None = None,
        alpha: float | None = None,
        top_k_dense: int | None = None,
        top_k_bm25: int | None = None,
        metadata_filter=None,
    ) -> dict:
        """Run both retrievers and fuse their scores.

        Returns a dict with dense_results, bm25_results, and hybrid_results so
        callers can inspect every stage. When ``metadata_filter`` is supplied,
        both retrievers are scoped to the matching documents BEFORE search, so
        the fused pool contains no cross-company chunks.
        """
        top_k = top_k or settings.TOP_K_HYBRID
        alpha = settings.HYBRID_ALPHA if alpha is None else alpha

        # Resolve the eligible chunk indices once and share them with both
        # retrievers (None means "no restriction").
        allowed_ids = self.dense.store.allowed_indices(metadata_filter)

        dense_results = self.dense.search(
            question, top_k=top_k_dense or settings.TOP_K_DENSE, allowed_ids=allowed_ids
        )
        bm25_results = self.bm25.search(
            question, top_k=top_k_bm25 or settings.TOP_K_BM25, allowed_ids=allowed_ids
        )

        # Per-retriever rank (1-based) and min-max-normalized scores. The
        # normalized scores are kept purely for transparency/inspection; fusion
        # ordering uses ranks (RRF), not these scores.
        dense_rank = {r["chunk_id"]: i for i, r in enumerate(dense_results, start=1)}
        bm25_rank = {r["chunk_id"]: i for i, r in enumerate(bm25_results, start=1)}
        dense_norm = _min_max_normalize({r["chunk_id"]: r["dense_score"] for r in dense_results})
        bm25_norm = _min_max_normalize({r["chunk_id"]: r["bm25_score"] for r in bm25_results})

        # Collect the union of candidate chunks, deduplicated by chunk_id.
        candidates: dict[str, dict] = {}
        for r in dense_results:
            candidates[r["chunk_id"]] = dict(r)
        for r in bm25_results:
            if r["chunk_id"] in candidates:
                candidates[r["chunk_id"]].update(
                    {k: v for k, v in r.items() if k not in ("rank",)}
                )
            else:
                candidates[r["chunk_id"]] = dict(r)

        # Reciprocal Rank Fusion. A chunk contributes 1/(K+rank) for each
        # retriever it appears in, weighted by alpha (dense) / 1-alpha (bm25).
        # A chunk present in only one retriever keeps its full contribution for
        # that modality instead of being penalized to zero.
        k = settings.RRF_K
        fused: list[dict] = []
        for cid, cand in candidates.items():
            dr = dense_rank.get(cid)
            br = bm25_rank.get(cid)
            rrf = 0.0
            if dr is not None:
                rrf += alpha * (1.0 / (k + dr))
            if br is not None:
                rrf += (1.0 - alpha) * (1.0 / (k + br))
            cand = dict(cand)
            cand["dense_score"] = cand.get("dense_score")
            cand["bm25_score"] = cand.get("bm25_score")
            cand["dense_rank"] = dr
            cand["bm25_rank"] = br
            cand["dense_norm"] = round(dense_norm.get(cid, 0.0), 6)
            cand["bm25_norm"] = round(bm25_norm.get(cid, 0.0), 6)
            cand["rrf_score"] = rrf
            cand.pop("rank", None)
            fused.append(cand)

        # Normalize RRF into [0, 1] as the reported hybrid_score so downstream
        # signal thresholds (e.g. MIN_HYBRID_SCORE) stay meaningful. Ordering is
        # by raw RRF (identical ordering, just a rescale).
        hybrid_norm = _min_max_normalize({c["chunk_id"]: c["rrf_score"] for c in fused})
        for c in fused:
            c["hybrid_score"] = round(hybrid_norm.get(c["chunk_id"], 0.0), 6)

        fused.sort(key=lambda c: c["rrf_score"], reverse=True)
        hybrid_results = fused[:top_k]
        for rank, r in enumerate(hybrid_results, start=1):
            r["rank"] = rank

        # Candidate pool for reranking: the FULL fused union (capped only to
        # avoid a pathologically large cross-encoder batch). Because the pool is
        # the whole union, no chunk is dropped before reranking regardless of
        # its fusion score - fusion ordering only decides the rank-aware safety
        # net applied AFTER reranking (see RagEngine).
        candidate_pool = fused[: settings.RETRIEVAL_POOL_SIZE]

        return {
            "dense_results": dense_results,
            "bm25_results": bm25_results,
            "hybrid_results": hybrid_results,
            "candidate_pool": candidate_pool,
            "alpha": alpha,
        }
