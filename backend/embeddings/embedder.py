"""
Local embedding model wrapper.

Uses sentence-transformers with a configurable open-source model (default
all-MiniLM-L6-v2). No paid API is required. Embeddings are L2-normalized so
that inner-product search in FAISS is equivalent to cosine similarity.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from backend.config import settings


@lru_cache(maxsize=2)
def _load_model(model_name: str):
    # Imported lazily so that importing this module is cheap and does not
    # require torch to be present until embeddings are actually needed.
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


class Embedder:
    """Thin wrapper around a sentence-transformers model."""

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or settings.EMBEDDING_MODEL
        self._model = None

    @property
    def model(self):
        if self._model is None:
            self._model = _load_model(self.model_name)
        return self._model

    @property
    def dimension(self) -> int:
        return int(self.model.get_sentence_embedding_dimension())

    def encode(self, texts: list[str], batch_size: int = 32, show_progress: bool = False) -> np.ndarray:
        """Return an (n, dim) float32 array of L2-normalized embeddings."""
        if not texts:
            return np.zeros((0, self.dimension), dtype="float32")
        vecs = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return np.asarray(vecs, dtype="float32")

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]
