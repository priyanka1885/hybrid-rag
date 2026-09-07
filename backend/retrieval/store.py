"""
Chunk store: load and hold the persisted chunk records.

The ingestion pipeline writes chunks to data/processed/chunks.jsonl. Both the
dense and BM25 retrievers reference chunks by their position in this list, so
loading it once and sharing it keeps everything consistent.
"""
from __future__ import annotations

import json
from pathlib import Path

from backend.config import CHUNKS_PATH


class ChunkStore:
    """In-memory list of chunk dicts, indexed by row position."""

    def __init__(self, chunks: list[dict]):
        self.chunks = chunks
        self._by_id = {c["chunk_id"]: c for c in chunks}
        # Structured company/year metadata for each chunk, aligned by row index.
        # Derived from the document name/file so no re-ingestion is required.
        from backend.retrieval.metadata import build_document_meta

        self.metadata = [
            build_document_meta(c["document_file"], c.get("document_name"))
            for c in chunks
        ]

    def allowed_indices(self, metadata_filter=None) -> set[int] | None:
        """Row indices whose document satisfies ``metadata_filter``.

        Returns ``None`` when no filter is active (meaning "all chunks"), so
        callers can cheaply distinguish "no restriction" from "an empty result".
        """
        if metadata_filter is None or metadata_filter.is_empty():
            return None
        return {
            i for i, meta in enumerate(self.metadata) if metadata_filter.matches(meta)
        }

    def __len__(self) -> int:
        return len(self.chunks)

    def get(self, idx: int) -> dict:
        return self.chunks[idx]

    def get_by_id(self, chunk_id: str) -> dict | None:
        return self._by_id.get(chunk_id)

    @property
    def texts(self) -> list[str]:
        return [c["text"] for c in self.chunks]

    @classmethod
    def load(cls, path: Path | None = None) -> "ChunkStore":
        path = path or CHUNKS_PATH
        if not path.exists():
            raise FileNotFoundError(
                f"Chunk file not found at {path}. Run `python scripts/ingest.py` first."
            )
        chunks: list[dict] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    chunks.append(json.loads(line))
        if not chunks:
            raise ValueError(f"Chunk file {path} is empty. Re-run ingestion.")
        return cls(chunks)

    def save(self, path: Path | None = None) -> None:
        path = path or CHUNKS_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for c in self.chunks:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
