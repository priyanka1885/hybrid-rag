"""
BM25 lexical retrieval.

    User Question -> Tokenization -> BM25 -> Lexical Search -> Results

IMPORTANT: BM25 does NOT use embeddings. It scores documents purely on term
frequency / inverse document frequency statistics over tokenized text. This
makes it strong for exact financial terminology, company names, numbers, and
rare phrases that dense embeddings can smooth over.
"""
from __future__ import annotations

import pickle
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

from backend.config import BM25_PATH, settings
from backend.retrieval.store import ChunkStore
from backend.text_normalize import canonicalize_numbers


_TOKEN_RE = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")


def tokenize(text: str) -> list[str]:
    """Financial-aware lowercase lexical tokenizer (no embeddings involved).

    Before tokenizing, numbers are canonicalized so the many ways a financial
    figure is written collapse to ONE lexical token, applied identically at
    index and query time:

    * space-grouped thousands  "64 392"    -> "64392"
    * comma-grouped thousands  "64,392"    -> "64392"
    * comma decimals           "63,4"      -> "63.4"
    * plain / decimal          "64392" / "63.4" unchanged

    Without this, a query for "64,392" tokenizes to ["64392"] while a table
    chunk containing "64 392" tokenizes to ["64", "392"], so the exact-number
    query never lexically hits the row that answers it. Canonicalizing both
    sides fixes numeric retrieval while keeping BM25 purely lexical. Decimal
    numbers stay intact (e.g. "95.7"); percentages match on their numeric part.
    """
    return _TOKEN_RE.findall(canonicalize_numbers(text).lower())


class BM25Retriever:
    def __init__(self, store: ChunkStore):
        self.store = store
        self.bm25: BM25Okapi | None = None

    def build(self) -> "BM25Retriever":
        corpus = [tokenize(t) for t in self.store.texts]
        self.bm25 = BM25Okapi(corpus)
        return self

    def save(self, path: Path | None = None) -> None:
        path = path or BM25_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.bm25, f)

    def load(self, path: Path | None = None) -> "BM25Retriever":
        path = path or BM25_PATH
        if not path.exists():
            raise FileNotFoundError(
                f"BM25 index not found at {path}. Run `python scripts/ingest.py` first."
            )
        with open(path, "rb") as f:
            self.bm25 = pickle.load(f)
        if self.bm25.corpus_size != len(self.store):
            raise ValueError(
                "BM25 index size does not match chunk store. Re-run ingestion "
                f"(index={self.bm25.corpus_size}, chunks={len(self.store)})."
            )
        return self

    def search(
        self,
        question: str,
        top_k: int | None = None,
        allowed_ids: set[int] | None = None,
    ) -> list[dict]:
        top_k = top_k or settings.TOP_K_BM25
        if self.bm25 is None:
            raise RuntimeError("BM25 index not loaded. Call build() or load() first.")
        tokens = tokenize(question)
        scores = self.bm25.get_scores(tokens)
        # Rank only the eligible chunks. When metadata pre-filtering is active,
        # ``allowed_ids`` restricts scoring to the matching documents so lexical
        # hits from other companies never enter the results.
        candidate_idxs = range(len(scores)) if allowed_ids is None else allowed_ids
        ranked = sorted(candidate_idxs, key=lambda i: scores[i], reverse=True)[:top_k]
        results: list[dict] = []
        for rank, idx in enumerate(ranked, start=1):
            chunk = self.store.get(idx)
            results.append(
                {
                    "rank": rank,
                    "chunk_id": chunk["chunk_id"],
                    "document_name": chunk["document_name"],
                    "document_file": chunk["document_file"],
                    "page_number": chunk["page_number"],
                    "text": chunk["text"],
                    "content_type": chunk.get("content_type", "text"),
                    "visual_ref": chunk.get("visual_ref"),
                    "bm25_score": float(scores[idx]),
                }
            )
        return results
