"""
Cross-encoder reranking.

    (Question, Candidate Chunk) -> Cross Encoder -> Relevance Score -> Sort

Unlike the bi-encoder used for dense retrieval (which embeds question and
passage separately), a cross-encoder jointly encodes the question and each
candidate passage, producing a much more precise relevance score. It is
expensive, so we only apply it to the small candidate set returned by hybrid
retrieval - never the whole corpus.
"""
from __future__ import annotations

from functools import lru_cache

from backend.config import settings


@lru_cache(maxsize=2)
def _load_cross_encoder(model_name: str):
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name)


class Reranker:
    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or settings.RERANKER_MODEL
        self._model = None

    @property
    def model(self):
        if self._model is None:
            self._model = _load_cross_encoder(self.model_name)
        return self._model

    def rerank(self, question: str, candidates: list[dict], top_k: int | None = None) -> list[dict]:
        """Score each candidate against the question and return the top_k.

        Adds a ``rerank_score`` and a ``final_rank`` to each returned item.
        """
        top_k = top_k or settings.RERANK_TOP_K
        if not candidates:
            return []
        pairs = [[question, c["text"]] for c in candidates]
        scores = self.model.predict(pairs)
        scored: list[dict] = []
        for cand, score in zip(candidates, scores):
            item = dict(cand)
            item["rerank_score"] = float(score)
            scored.append(item)
        scored.sort(key=lambda c: c["rerank_score"], reverse=True)
        top = scored[:top_k]
        for rank, item in enumerate(top, start=1):
            item["final_rank"] = rank
        return top
