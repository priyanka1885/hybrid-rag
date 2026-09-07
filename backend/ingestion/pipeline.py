"""
End-to-end ingestion pipeline.

    Financial Report PDFs
        -> PDF text extraction (page numbers preserved)
        -> cleaning
        -> chunking + metadata
        -> embeddings -> FAISS index
        -> BM25 index
        -> persist all artifacts

Running this is idempotent: if artifacts already exist and ``force`` is False,
the pipeline reuses them instead of rebuilding.
"""
from __future__ import annotations

from backend.config import (
    BM25_PATH,
    CHUNKS_PATH,
    FAISS_INDEX_PATH,
    PDF_DIR,
    ensure_dirs,
    settings,
)
from backend.embeddings.embedder import Embedder
from backend.ingestion.chunker import chunk_pages
from backend.ingestion.pdf_loader import extract_pages, find_pdfs
from backend.retrieval.bm25 import BM25Retriever
from backend.retrieval.dense import build_faiss_index, save_faiss_index
from backend.retrieval.store import ChunkStore


def artifacts_exist() -> bool:
    return CHUNKS_PATH.exists() and FAISS_INDEX_PATH.exists() and BM25_PATH.exists()


def run_ingestion(force: bool = False, verbose: bool = True) -> dict:
    """Build (or reuse) all indexes. Returns a summary dict."""
    ensure_dirs()

    def log(msg: str):
        if verbose:
            print(msg, flush=True)

    if artifacts_exist() and not force:
        log("Indexes already exist. Reusing them (use --force to rebuild).")
        store = ChunkStore.load()
        return {
            "reused": True,
            "num_documents": len({c["document_file"] for c in store.chunks}),
            "num_chunks": len(store),
        }

    pdfs = find_pdfs(PDF_DIR)
    if not pdfs:
        raise FileNotFoundError(
            f"No PDF files found under {PDF_DIR}. Set DOCUMENTS_DIR in .env to the "
            "folder that contains your financial-report PDFs."
        )
    log(f"Found {len(pdfs)} PDF(s) under {PDF_DIR}")

    # 1. Extract + clean pages.
    all_pages = []
    for pdf in pdfs:
        pages = list(extract_pages(pdf))
        log(f"  {pdf.name}: extracted text from {len(pages)} page(s)")
        all_pages.extend(pages)
    if not all_pages:
        raise ValueError("No extractable text found in any PDF. Are these scanned images?")

    # 2. Chunk with page-level metadata.
    chunks = chunk_pages(all_pages, settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)
    log(f"Created {len(chunks)} chunks (size={settings.CHUNK_SIZE}, overlap={settings.CHUNK_OVERLAP})")

    store = ChunkStore([c.to_dict() for c in chunks])
    store.save()
    log(f"Persisted chunks -> {CHUNKS_PATH}")

    # 3. Embeddings + FAISS.
    log(f"Embedding chunks with {settings.EMBEDDING_MODEL} ...")
    embedder = Embedder()
    index, meta = build_faiss_index(store, embedder, show_progress=verbose)
    save_faiss_index(index, meta)
    log(f"Persisted FAISS index -> {FAISS_INDEX_PATH} (dim={meta['dimension']})")

    # 4. BM25.
    log("Building BM25 index ...")
    bm25 = BM25Retriever(store).build()
    bm25.save()
    log(f"Persisted BM25 index -> {BM25_PATH}")

    return {
        "reused": False,
        "num_documents": len(pdfs),
        "num_chunks": len(store),
        "embedding_model": settings.EMBEDDING_MODEL,
        "dimension": meta["dimension"],
    }
