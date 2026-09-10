"""
Central configuration for the Hybrid RAG for Financial Reports system.

All tunable parameters are read from environment variables (see .env.example)
so nothing important is hardcoded across the codebase. Import `settings` from
this module everywhere else.
"""
from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover - dotenv is optional at runtime
    pass


def _get(name: str, default: str) -> str:
    val = os.getenv(name)
    return val if val is not None and val != "" else default


def _get_int(name: str, default: int) -> int:
    try:
        return int(_get(name, str(default)))
    except ValueError:
        return default


def _get_float(name: str, default: float) -> float:
    try:
        return float(_get(name, str(default)))
    except ValueError:
        return default


def _get_bool(name: str, default: bool) -> bool:
    val = _get(name, str(default)).strip().lower()
    return val in ("1", "true", "yes", "on")


# --- Paths -----------------------------------------------------------------
# Project root is the parent of the backend/ directory.
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"

# Where the source financial-report PDFs live. This is the RAG knowledge base.
# PDFs are discovered RECURSIVELY under this directory, so nested subfolders
# are fine. Configurable via DOCUMENTS_DIR in .env. Individual PDF file names
# are never hardcoded anywhere in the pipeline.
_default_docs_dir = ROOT_DIR / "Structured data-20250319T105519Z-001"
DOCUMENTS_DIR = Path(_get("DOCUMENTS_DIR", str(_default_docs_dir)))
# Backwards-compatible alias used internally by the ingestion pipeline.
PDF_DIR = DOCUMENTS_DIR

# Where the QA evaluation CSV lives. This is used ONLY for evaluation /
# benchmarking, never as part of the retrieval knowledge base.
_default_qa = ROOT_DIR / "Data_ret.csv"
QA_CSV_PATH = Path(_get("QA_CSV_PATH", str(_default_qa)))

PROCESSED_DIR = DATA_DIR / "processed"
# Rendered page images used as the citation "visual reference" for evidence
# derived from tables / scanned pages / figures. One PNG per page that yields
# visual content; text-only chunks continue to rely on document + page number.
PAGE_IMAGE_DIR = PROCESSED_DIR / "page_images"
INDEXES_DIR = DATA_DIR / "indexes"
FAISS_DIR = INDEXES_DIR / "faiss"
BM25_DIR = INDEXES_DIR / "bm25"

CHUNKS_PATH = PROCESSED_DIR / "chunks.jsonl"
FAISS_INDEX_PATH = FAISS_DIR / "index.faiss"
FAISS_META_PATH = FAISS_DIR / "meta.json"
BM25_PATH = BM25_DIR / "bm25.pkl"
EVAL_RESULTS_PATH = PROCESSED_DIR / "evaluation.json"


class Settings:
    """Runtime configuration, sourced from environment variables."""

    # --- LLM (OpenRouter, OpenAI-compatible chat completions) ---
    # The LLM is served by OpenRouter's hosted API using the free Llama 3.1 8B
    # Instruct model. Model name and base URL stay configurable via .env so the
    # model can be swapped without code changes; nothing is hardcoded elsewhere.
    LLM_MODEL: str = _get("LLM_MODEL", "inclusionai/ling-3.0-flash-fin:free")
    LLM_BASE_URL: str = _get("LLM_BASE_URL", "https://openrouter.ai/api")
    # API key for OpenRouter. Read ONLY from the environment - never hardcoded
    # and never committed. Empty string means "not configured", which health/
    # generation surface as a friendly message instead of crashing.
    OPENROUTER_API_KEY: str = _get("OPENROUTER_API_KEY", "")
    # Request timeout (seconds) for a single chat-completion call. Hosted models
    # respond quickly, but free-tier requests can queue, so keep it generous.
    LLM_TIMEOUT: int = _get_int("LLM_TIMEOUT", 120)
    LLM_TEMPERATURE: float = _get_float("LLM_TEMPERATURE", 0.0)

    # --- Embeddings ---
    EMBEDDING_MODEL: str = _get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    # Same batch * seq^2 memory budget as the reranker (see RERANK_TOKEN_BUDGET).
    # At serving time only one short question is embedded per request so this
    # barely binds; it matters during ingestion, where thousands of chunks are
    # embedded and a fixed batch of long chunks would allocate a large attention
    # block. Embeddings are unaffected by batch composition (mean pooling is
    # attention-mask weighted), so this is purely a memory/speed dial.
    EMBED_TOKEN_BUDGET: int = _get_int("EMBED_TOKEN_BUDGET", 262144)

    # --- Reranker ---
    RERANKER_MODEL: str = _get("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    # Memory budget for cross-encoder inference, which is the largest transient
    # allocation the service makes. A transformer's peak allocation is dominated
    # by the attention matrix, sized batch * heads * seq_len^2 * 4 bytes - note
    # the SQUARE on sequence length. Budgeting batch * seq^2 (rather than batch
    # alone) keeps that block bounded no matter how long the retrieved chunks
    # are: short table rows still batch widely, and the occasional long passage
    # runs in a small batch. 262144 * 12 heads * 4 bytes is roughly 12 MB per
    # attention tensor. Raise it to trade memory for a little latency; lower it
    # if the instance is tight. Scores are unaffected either way - padding is
    # masked out - so answerability thresholds stay calibrated.
    RERANK_TOKEN_BUDGET: int = _get_int("RERANK_TOKEN_BUDGET", 262144)
    # Hard cap on pairs per batch; the token budget usually binds first.
    RERANK_BATCH_SIZE: int = _get_int("RERANK_BATCH_SIZE", 16)
    # Table-aware rerank boost. Cross-encoders (ms-marco) reward fluent narrative
    # prose over terse numeric table rows, so an exact-value row (e.g. "Direct
    # scope 1 58 644") can be demoted below generic header/narrative chunks. For
    # data-seeking questions (values, totals, years, emissions...) we add a small
    # constant logit boost to TABLE/FIGURE chunks AFTER reranking and re-sort, so
    # exact-data rows are promoted above loosely relevant prose. The rank-aware
    # safety net already guarantees PRESENCE in the evidence; this fixes ORDER.
    # Disable or tune via .env.
    ENABLE_TABLE_RERANK_BOOST: bool = _get_bool("ENABLE_TABLE_RERANK_BOOST", True)
    TABLE_RERANK_BOOST: float = _get_float("TABLE_RERANK_BOOST", 2.0)

    # --- Chunking (word-based) ---
    CHUNK_SIZE: int = _get_int("CHUNK_SIZE", 200)
    CHUNK_OVERLAP: int = _get_int("CHUNK_OVERLAP", 40)
    # Table-aware chunking: number-dense regions (financial tables) are also
    # split into finer windows so a single metric label + its value (e.g.
    # "Total greenhouse gas ... 64392") is not diluted by hundreds of other
    # numbers in a large window, which makes such rows retrievable and
    # rerankable. A chunk whose fraction of numeric tokens exceeds
    # TABLE_NUMERIC_DENSITY is treated as tabular. These finer chunks are added
    # ALONGSIDE the standard chunks; narrative text is unaffected.
    TABLE_CHUNK_SIZE: int = _get_int("TABLE_CHUNK_SIZE", 90)
    TABLE_CHUNK_OVERLAP: int = _get_int("TABLE_CHUNK_OVERLAP", 30)
    TABLE_NUMERIC_DENSITY: float = _get_float("TABLE_NUMERIC_DENSITY", 0.35)
    # Header-aware atomic table segmentation (chunker.py). A page's word stream
    # is split into narrative vs. number-dense TABLE regions. Table regions are
    # emitted whole (never cut mid-row) and carry the preceding narrative
    # "lead-in" (the metric title + column headers) so a value like "58 644"
    # always travels with its label AND its year header.
    #   TABLE_DETECT_WINDOW  - local word window used to measure numeric density
    #                          when classifying each position as table/narrative.
    #   TABLE_BRIDGE_GAP     - a short non-numeric gap (label words, subtotals)
    #                          shorter than this is absorbed into the table so a
    #                          single table is not shattered into fragments.
    #   TABLE_LEADIN_WORDS   - max narrative words walked back from a table start
    #                          to capture its header/label lead-in.
    #   TABLE_MAX_WORDS      - max words in an atomic table chunk; larger tables
    #                          are split on value-group boundaries with the
    #                          lead-in re-prepended to every sub-block (keeps the
    #                          chunk within the embedder's token window).
    TABLE_DETECT_WINDOW: int = _get_int("TABLE_DETECT_WINDOW", 12)
    TABLE_BRIDGE_GAP: int = _get_int("TABLE_BRIDGE_GAP", 6)
    TABLE_LEADIN_WORDS: int = _get_int("TABLE_LEADIN_WORDS", 25)
    TABLE_MAX_WORDS: int = _get_int("TABLE_MAX_WORDS", 180)

    # --- Retrieval ---
    # "Retrieve wide, rerank narrow": the hybrid stage gathers a broad candidate
    # pool so a precise-but-terse table row (which dense/BM25 rank lower than
    # fluent narrative prose) still reaches the cross-encoder, which then
    # promotes the truly relevant evidence. Widening the pool improves recall
    # without lowering any answerability/quality threshold.
    TOP_K_DENSE: int = _get_int("TOP_K_DENSE", 60)
    TOP_K_BM25: int = _get_int("TOP_K_BM25", 60)
    TOP_K_HYBRID: int = _get_int("TOP_K_HYBRID", 60)
    RERANK_TOP_K: int = _get_int("RERANK_TOP_K", 6)
    HYBRID_ALPHA: float = _get_float("HYBRID_ALPHA", 0.5)
    # Fusion is rank-based (Reciprocal Rank Fusion). RRF is scale-free, so it
    # avoids the failure mode of min-max fusion where a chunk found by only one
    # retriever gets 0 for the missing modality and is unfairly suppressed. K
    # dampens the influence of very deep ranks (standard RRF constant).
    RRF_K: int = _get_int("RRF_K", 60)
    # Upper bound on the candidate union handed to the cross-encoder. The pool
    # is the FULL dense+BM25 union; this only caps pathologically large unions.
    RETRIEVAL_POOL_SIZE: int = _get_int("RETRIEVAL_POOL_SIZE", 40)
    # Rank-aware safety net: after cross-encoder reranking, always include the
    # top-N chunks by hybrid (RRF) rank in the evidence sent to the LLM, even if
    # the cross-encoder demoted them. This prevents the observed failure where a
    # chunk both retrievers rank at the very top is dropped by a noisy
    # cross-encoder before it can be used. Set to 0 to disable.
    HYBRID_SAFETY_TOP_K: int = _get_int("HYBRID_SAFETY_TOP_K", 5)
    # Additionally guarantee the top-N hits from EACH individual retriever
    # (dense and BM25) reach the LLM. RRF necessarily discounts a hit found by
    # only one retriever; this protects a strong single-modality hit (e.g. a
    # terse table row BM25 loves but the bi-encoder misses, or vice versa).
    MODALITY_SAFETY_TOP_K: int = _get_int("MODALITY_SAFETY_TOP_K", 3)

    # --- PDF text extraction ---
    # The default pdfplumber extract_text() orders glyphs by absolute position,
    # which interleaves the columns of two-page "spread" report layouts and
    # scrambles table rows (a metric label gets separated from its value, e.g.
    # "Scope 1" ends up far from "58 644"). When True (default) the loader reads
    # words in the PDF's native text-flow order instead, which follows the
    # content stream and preserves reading order across columns, keeping each
    # row's label and numbers together. Set False to restore the raw
    # position-sorted extractor.
    LAYOUT_AWARE_EXTRACTION: bool = _get_bool("LAYOUT_AWARE_EXTRACTION", True)

    # --- Query expansion (recall) ---
    # Controlled query expansion for data-seeking questions: the dense + BM25
    # RETRIEVAL query is widened with a small, curated set of concept synonyms
    # (scope 1 / direct / GHG / greenhouse gas ...) and fiscal-year
    # normalization (FY2023 -> 2023) so a terse exact-value table row a
    # paraphrase shares no surface with still enters the candidate union and
    # reaches the cross-encoder. The reranker and LLM always receive the
    # ORIGINAL question, so precision/grounding are unchanged. Expansion is
    # gated to data-seeking queries and to a curated concept map, so retrieval
    # is not flooded with loosely related chunks. Disable via .env to restore
    # the original original-query-only retrieval behaviour.
    ENABLE_QUERY_EXPANSION: bool = _get_bool("ENABLE_QUERY_EXPANSION", True)

    # --- Metadata pre-filtering ---
    # The corpus mixes several unrelated companies' reports. When True (default)
    # retrieval is scoped to the company/year a question is about (auto-detected
    # from the question text, or supplied explicitly by the API) BEFORE dense +
    # BM25 search, which removes cross-company context contamination. Detection
    # degrades safely: if no company/year is found the filter is empty and
    # retrieval behaves exactly as before. Set to False to disable entirely.
    ENABLE_METADATA_FILTER: bool = _get_bool("ENABLE_METADATA_FILTER", True)

    # --- Debugging ---
    # When True, RagEngine.answer() attaches a compact per-stage "debug_trace"
    # (dense/bm25/hybrid/pool/rerank/evidence/verification) to its result and
    # logs it. Off by default so production responses stay lean.
    RETRIEVAL_DEBUG: bool = _get_bool("RETRIEVAL_DEBUG", False)
    # If the LLM refuses ("insufficient evidence") despite strong reranked
    # evidence, retry generation ONCE with a stronger evidence-focused prompt.
    ENABLE_REFUSAL_RETRY: bool = _get_bool("ENABLE_REFUSAL_RETRY", True)

    # --- Answerability / hallucination thresholds ---
    # Minimum cross-encoder relevance score for the top evidence to be
    # considered "in scope". Below this we refuse to answer.
    MIN_RERANK_SCORE: float = _get_float("MIN_RERANK_SCORE", -4.0)
    # Minimum hybrid score signal (normalized 0..1) for the best candidate.
    MIN_HYBRID_SCORE: float = _get_float("MIN_HYBRID_SCORE", 0.15)

    # --- Citation verification ---
    # Token-overlap ratio above which a claim is considered SUPPORTED.
    VERIFY_SUPPORTED_THRESHOLD: float = _get_float("VERIFY_SUPPORTED_THRESHOLD", 0.45)
    VERIFY_PARTIAL_THRESHOLD: float = _get_float("VERIFY_PARTIAL_THRESHOLD", 0.2)

    # --- API ---
    API_HOST: str = _get("API_HOST", "0.0.0.0")
    API_PORT: int = _get_int("API_PORT", 8000)

    # --- Multimodal / visual ingestion (OPT-IN, reversible) -----------------
    # Master switch. When False (the default) the ingestion pipeline behaves
    # EXACTLY as the original text-only pipeline: no tables, no OCR, no page
    # images, and existing chunk ids are byte-identical. Turn on via
    # ENABLE_VISUAL_INGEST=true in .env to add table / scanned-page / figure
    # support. Every sub-feature below is additionally gated by this flag.
    ENABLE_VISUAL_INGEST: bool = _get_bool("ENABLE_VISUAL_INGEST", False)
    # Extract vector (real) PDF tables via pdfplumber and index them as
    # structured Markdown (row/column layout preserved), not flattened text.
    ENABLE_TABLE_EXTRACTION: bool = _get_bool("ENABLE_TABLE_EXTRACTION", True)
    # OCR scanned pages, image-based tables, and text inside figures/charts.
    # Requires the Tesseract engine to be installed; if it is missing, OCR is
    # skipped gracefully (page images / tables still work) and a note is logged.
    ENABLE_OCR: bool = _get_bool("ENABLE_OCR", True)
    OCR_LANGUAGE: str = _get("OCR_LANGUAGE", "eng")
    # Optional absolute path to the tesseract executable (Windows installs are
    # often not on PATH). Leave empty to rely on PATH.
    TESSERACT_CMD: str = _get("TESSERACT_CMD", "")
    # DPI used to rasterize a page for OCR / for the citation page image.
    PAGE_RENDER_DPI: int = _get_int("PAGE_RENDER_DPI", 150)
    # A page whose extractable text layer has fewer than this many characters
    # is treated as a scanned/image page and OCR'd as a whole. Above it, the
    # text layer is trusted and the page is NOT OCR'd, so OCR never duplicates
    # content already available as real PDF text.
    OCR_MIN_CHARS: int = _get_int("OCR_MIN_CHARS", 20)
    # Figure/chart selection: an embedded image is only OCR'd if it covers at
    # least this fraction of the page area (skips small decorative logos/icons)
    # ...
    FIGURE_MIN_AREA_RATIO: float = _get_float("FIGURE_MIN_AREA_RATIO", 0.05)
    # ... and is only kept as evidence if OCR yields at least this many
    # meaningful alphanumeric characters (skips images with no real text).
    FIGURE_MIN_OCR_CHARS: int = _get_int("FIGURE_MIN_OCR_CHARS", 12)


settings = Settings()


# Friendly document names for the source PDFs. Keys are the exact PDF file
# names; values are human-readable report titles used in citations.
DOCUMENT_NAME_MAP: dict[str, str] = {
    "2021ESG.pdf": "Tongaat Hulett ESG Report 2021",
    "2022-Absa-Group-limited-Environmental-Social-and-Governance-Data-sheet.pdf": "Absa Group ESG Data Sheet 2022",
    "Clicks-Sustainability-Report-2022.pdf": "Clicks Sustainability Report 2022",
    "DISTELL ESG Appendix 2022.pdf": "Distell ESG Appendix 2022",
    "ESG-spreads.pdf": "Impala Platinum ESG Report 2023",
    "picknpay-esg-report-spreads-2023.pdf": "Pick n Pay ESG Report 2023",
    "SASOL Sustainability Report 2023 20-09_0.pdf": "Sasol Sustainability Report 2023",
}


def friendly_document_name(file_name: str) -> str:
    """Return a human-readable report title for a given PDF file name."""
    return DOCUMENT_NAME_MAP.get(file_name, Path(file_name).stem)


def ensure_dirs() -> None:
    """Create the data/processed and index directories if missing."""
    dirs = [PROCESSED_DIR, FAISS_DIR, BM25_DIR]
    if settings.ENABLE_VISUAL_INGEST:
        dirs.append(PAGE_IMAGE_DIR)
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
