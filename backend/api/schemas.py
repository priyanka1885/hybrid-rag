"""Pydantic schemas for the API request/response models."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="A question about the financial reports.")
    # Optional explicit metadata scope. When omitted, the company/year are
    # auto-detected from the question. Supplying them enforces a strict filter
    # so retrieval only considers the matching report(s).
    company: Optional[str] = Field(
        None, description="Restrict retrieval to this company (e.g. 'Sasol', 'Pick n Pay')."
    )
    year: Optional[int] = Field(
        None, description="Restrict retrieval to reports for this year (e.g. 2023)."
    )


class Citation(BaseModel):
    citation_id: int
    document_name: str
    page_number: int
    chunk_id: str
    supporting_text: str
    # "text" (default) | "table" | "ocr_page" | "figure". Present so the UI can
    # label how the evidence was obtained.
    content_type: str = "text"
    # Path to a rendered image of the source page for visual evidence, so a
    # citation can link back to the actual table/figure/scanned page.
    visual_ref: Optional[str] = None


class PerCitationVerification(BaseModel):
    citation_id: int
    document_name: str
    page_number: int
    status: str
    coverage: float
    numbers_matched: bool
    # Compact, human-readable reason for the status (e.g. a year-value
    # relationship mismatch). Optional so existing frontends are unaffected.
    reason: Optional[str] = None


class Verification(BaseModel):
    overall_status: str
    summary: str
    per_citation: list[PerCitationVerification] = []


class RetrievalRow(BaseModel):
    rank: Optional[int] = None
    final_rank: Optional[int] = None
    chunk_id: str
    document_name: str
    page_number: int
    content_type: str = "text"
    visual_ref: Optional[str] = None
    dense_score: Optional[float] = None
    bm25_score: Optional[float] = None
    dense_norm: Optional[float] = None
    bm25_norm: Optional[float] = None
    hybrid_score: Optional[float] = None
    rerank_score: Optional[float] = None
    text_preview: str


class RetrievalDetails(BaseModel):
    dense_results: list[RetrievalRow] = []
    bm25_results: list[RetrievalRow] = []
    hybrid_results: list[RetrievalRow] = []
    reranked_results: list[RetrievalRow] = []
    alpha: Optional[float] = None
    # The metadata scope actually applied to this query (company/year), so the
    # UI can show that cross-company contamination was prevented.
    applied_filter: Optional[dict] = None


class AskResponse(BaseModel):
    answer: str
    citations: list[Citation] = []
    verification: Verification
    retrieval_details: RetrievalDetails
    llm_available: bool = True
    status: str = "ok"


class HealthResponse(BaseModel):
    status: str
    llm: dict
    indexes_loaded: bool


class MethodMetrics(BaseModel):
    label: str
    recall_at_k: float
    precision_at_k: float
    mrr: float
    hit_rate: float


class EvaluationResponse(BaseModel):
    available: bool
    num_documents: int
    num_chunks: int
    num_eval_questions: int
    k: Optional[int] = None
    methods: dict[str, MethodMetrics] = {}
    metric_names: list[str] = []
    message: Optional[str] = None
