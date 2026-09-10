"""
Dense (semantic) retrieval.

    User Question -> Embedding Model -> FAISS -> Top-K Dense Results

The question is embedded with the SAME model used at ingestion time, then a
FAISS inner-product search returns the most semantically similar chunks.
This captures paraphrase / conceptual matches that lexical search misses.
"""
from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np

from backend.config import FAISS_INDEX_PATH, FAISS_META_PATH, settings
from backend.embeddings.embedder import Embedder
from backend.retrieval.store import ChunkStore
from backend.runtime import configure_faiss_threads

# faiss defaults to one OpenMP thread per CPU, each with its own scratch space.
# The corpus here is a few thousand vectors, so extra threads buy nothing and
# only add resident memory. Applied here rather than at process start because the
# setting only exists once faiss itself has been imported.
configure_faiss_threads()


class DenseRetriever:
    def __init__(self, store: ChunkStore, embedder: Embedder | None = None):
        self.store = store
        self.embedder = embedder or Embedder()
        self.index = None
        self.meta: dict = {}

    def load(self, index_path: Path | None = None, meta_path: Path | None = None) -> "DenseRetriever":
        index_path = index_path or FAISS_INDEX_PATH
        meta_path = meta_path or FAISS_META_PATH
        if not index_path.exists():
            raise FileNotFoundError(
                f"FAISS index not found at {index_path}. Run `python scripts/ingest.py` first."
            )
        self.index = faiss.read_index(str(index_path))
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                self.meta = json.load(f)
        # Sanity check: index size must match the chunk store.
        if self.index.ntotal != len(self.store):
            raise ValueError(
                "FAISS index size does not match chunk store. Re-run ingestion "
                f"(index={self.index.ntotal}, chunks={len(self.store)})."
            )
        # Sanity check: the index MUST have been built with the same embedding
        # model the query will use, otherwise question and chunk vectors live in
        # different spaces and retrieval silently returns garbage. An
        # embedding-model mismatch is a common, hard-to-spot cause of bad
        # retrieval, so we fail loudly instead.
        indexed_model = self.meta.get("embedding_model")
        if indexed_model and indexed_model != self.embedder.model_name:
            raise ValueError(
                "FAISS index was built with a different embedding model than the "
                f"one configured for querying (index='{indexed_model}', "
                f"query='{self.embedder.model_name}'). Re-run ingestion or set "
                "EMBEDDING_MODEL to match the index."
            )
        # Sanity check: query embedding dimension must match the index.
        indexed_dim = self.meta.get("dimension")
        if indexed_dim and self.index.d != indexed_dim:
            raise ValueError(
                f"FAISS index dimension ({self.index.d}) does not match metadata "
                f"dimension ({indexed_dim})."
            )
        return self

    def search(
        self,
        question: str,
        top_k: int | None = None,
        allowed_ids: set[int] | None = None,
    ) -> list[dict]:
        """Return the top_k semantically closest chunks.

        When ``allowed_ids`` is provided (metadata pre-filtering), only chunks
        in that set are eligible. The flat index is searched exhaustively and
        the results are filtered to the allowed set before taking top_k, so the
        restriction is exact - a matching chunk is never lost because it fell
        outside an unfiltered top_k window.
        """
        top_k = top_k or settings.TOP_K_DENSE
        if self.index is None:
            raise RuntimeError("Dense index not loaded. Call load() first.")
        q = self.embedder.encode_one(question).reshape(1, -1)
        search_k = self.index.ntotal if allowed_ids is not None else min(top_k, self.index.ntotal)
        scores, idxs = self.index.search(q, search_k)
        results: list[dict] = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx < 0:
                continue
            if allowed_ids is not None and int(idx) not in allowed_ids:
                continue
            chunk = self.store.get(int(idx))
            results.append(
                {
                    "rank": len(results) + 1,
                    "chunk_id": chunk["chunk_id"],
                    "document_name": chunk["document_name"],
                    "document_file": chunk["document_file"],
                    "page_number": chunk["page_number"],
                    "text": chunk["text"],
                    "content_type": chunk.get("content_type", "text"),
                    "visual_ref": chunk.get("visual_ref"),
                    # Cosine similarity in [-1, 1] (embeddings are normalized).
                    "dense_score": float(score),
                }
            )
            if len(results) >= top_k:
                break
        return results


def build_faiss_index(store: ChunkStore, embedder: Embedder, show_progress: bool = True):
    """Embed all chunks and build a normalized inner-product FAISS index."""
    # Ingestion is a one-off offline job with the whole machine to itself, so a
    # larger batch than the serving default is worth the extra transient memory.
    vectors = embedder.encode(store.texts, batch_size=32, show_progress=show_progress)
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    meta = {
        "embedding_model": embedder.model_name,
        "dimension": dim,
        "num_vectors": int(index.ntotal),
    }
    return index, meta


def save_faiss_index(index, meta: dict, index_path: Path | None = None, meta_path: Path | None = None) -> None:
    index_path = index_path or FAISS_INDEX_PATH
    meta_path = meta_path or FAISS_META_PATH
    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
