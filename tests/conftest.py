import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.config import CHUNKS_PATH, FAISS_INDEX_PATH, BM25_PATH  # noqa: E402

INDEXES_READY = CHUNKS_PATH.exists() and FAISS_INDEX_PATH.exists() and BM25_PATH.exists()

requires_indexes = pytest.mark.skipif(
    not INDEXES_READY,
    reason="Indexes not built. Run `python scripts/ingest.py` before running these tests.",
)


@pytest.fixture(scope="session")
def store():
    from backend.retrieval.store import ChunkStore

    return ChunkStore.load()


@pytest.fixture(scope="session")
def dense(store):
    from backend.retrieval.dense import DenseRetriever

    return DenseRetriever(store).load()


@pytest.fixture(scope="session")
def bm25(store):
    from backend.retrieval.bm25 import BM25Retriever

    return BM25Retriever(store).load()


@pytest.fixture(scope="session")
def hybrid(dense, bm25):
    from backend.retrieval.hybrid import HybridRetriever

    return HybridRetriever(dense, bm25)


@pytest.fixture(scope="session")
def reranker():
    from backend.reranking.reranker import Reranker

    return Reranker()
