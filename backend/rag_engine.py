"""
RAG engine: the orchestration layer that wires the whole pipeline together.

    Question
      -> Dense + BM25
      -> Hybrid fusion
      -> Cross-encoder rerank
      -> Answerability check
      -> Llama 3.1 grounded generation
      -> Citation mapping
      -> Citation verification
      -> Answer + citations + verification + retrieval_details

Indexes and models are loaded once and reused (the API holds a single engine).
"""
from __future__ import annotations

import logging
import re

from backend.citations.mapper import all_evidence_as_citations, build_citations
from backend.citations.verifier import filter_supported_claims, verify_answer
from backend.config import settings
from backend.generation.llm import (
    LLMUnavailableError,
    OllamaClient,
    _looks_tabular,
    looks_like_metadata_echo,
)
from backend.text_normalize import strip_retrieval_metadata

logger = logging.getLogger("rag.engine")
from backend.reranking.reranker import Reranker
from backend.retrieval.bm25 import BM25Retriever
from backend.retrieval.dense import DenseRetriever
from backend.retrieval.hybrid import HybridRetriever
from backend.retrieval.metadata import MetadataFilter, detect_filter
from backend.retrieval.query_expansion import ExpandedQuery, expand_query
from backend.retrieval.store import ChunkStore

INSUFFICIENT_MSG = (
    "I couldn't find sufficient evidence in the provided financial reports to "
    "answer this reliably."
)
OUT_OF_SCOPE_MSG = (
    "This question is outside the scope of the available financial reports."
)


# Data-seeking question detector: numeric / metric intent (a value, total,
# count, percentage, an explicit year, or a metric noun). These are exactly the
# questions a cross-encoder tends to mis-rank by preferring fluent prose over
# the terse table row that actually holds the number.
_DATA_QUERY_RE = re.compile(
    r"\b(how much|how many|what (?:were|was|is|are)|total|number of|amount|"
    r"emissions?|scope\s*[123]?|tonnes?|tco2e?|co2|kilotons?|percent|percentage|"
    r"rate|ratio|figure|value|revenue|volume|consumption|intensity|spend)\b",
    re.IGNORECASE,
)
_DIGIT_RE = re.compile(r"\d")


def is_data_seeking_query(question: str) -> bool:
    """True when a question seeks a numeric/data metric (value, total, year...).

    Triggers the table-aware rerank boost. Kept deliberately broad: a stray
    boost on a table for a borderline question is harmless (presence is already
    guaranteed by the safety net), whereas missing a numeric question is not.
    """
    q = question or ""
    return bool(_DATA_QUERY_RE.search(q) or _DIGIT_RE.search(q))


def build_retrieval_query(question: str) -> ExpandedQuery:
    """Build the query used for dense + BM25 RECALL (not for rerank/LLM).

    For data-seeking questions we widen recall with a small, curated set of
    concept synonyms and fiscal-year normalization (see
    :mod:`backend.retrieval.query_expansion`) so a terse exact-value table row -
    which a paraphrase shares almost no surface with - still enters the
    candidate union and reaches the cross-encoder. For non-data questions (or
    when nothing fires) the returned ``retrieval_text`` equals the original
    question, so behaviour is unchanged. The cross-encoder reranker and the LLM
    always receive the ORIGINAL question, so precision/grounding are untouched.
    """
    if not settings.ENABLE_QUERY_EXPANSION or not is_data_seeking_query(question):
        return ExpandedQuery(original=question, retrieval_text=question)
    return expand_query(question)


def apply_table_boost(question: str, reranked: list[dict]) -> list[dict]:
    """Promote TABLE/FIGURE chunks for data-seeking queries (post-rerank).

    Adds a constant logit boost (``settings.TABLE_RERANK_BOOST``) to
    table/figure chunks so an exact-value row the cross-encoder demoted is
    re-sorted above generic header/narrative text, then re-assigns
    ``final_rank``. No-op when the boost is disabled or the question is not
    data-seeking. Mutates and re-sorts ``reranked`` in place and returns it;
    the pre-boost score is preserved on ``rerank_score_raw`` for transparency.
    """
    if not settings.ENABLE_TABLE_RERANK_BOOST or not is_data_seeking_query(question):
        return reranked
    boost = settings.TABLE_RERANK_BOOST
    for c in reranked:
        # Promote structured tables/figures AND raw number-dense text rows (a
        # metric label followed by its per-year values). The latter carry the
        # exact metric+values a numeric question asks for but are plain "text"
        # chunks the cross-encoder tends to demote below fluent narrative, so
        # for data-seeking queries we surface them alongside real tables.
        is_tabular = c.get("content_type") in ("table", "figure") or (
            c.get("content_type", "text") == "text" and _looks_tabular(c.get("text", ""))
        )
        if is_tabular:
            c["rerank_score_raw"] = c.get("rerank_score")
            c["rerank_score"] = float(c.get("rerank_score", 0.0)) + boost
            c["table_boost"] = boost
    reranked.sort(key=lambda c: c["rerank_score"], reverse=True)
    for rank, c in enumerate(reranked, start=1):
        c["final_rank"] = rank
    return reranked


def _slim(results: list[dict]) -> list[dict]:
    """Trim retrieval rows for API transport (keep scores + locators)."""
    out = []
    for r in results:
        out.append(
            {
                "rank": r.get("rank"),
                "final_rank": r.get("final_rank"),
                "chunk_id": r["chunk_id"],
                "document_name": r["document_name"],
                "page_number": r["page_number"],
                "content_type": r.get("content_type", "text"),
                "visual_ref": r.get("visual_ref"),
                "dense_score": r.get("dense_score"),
                "bm25_score": r.get("bm25_score"),
                "dense_rank": r.get("dense_rank"),
                "bm25_rank": r.get("bm25_rank"),
                "dense_norm": r.get("dense_norm"),
                "bm25_norm": r.get("bm25_norm"),
                "rrf_score": r.get("rrf_score"),
                "hybrid_score": r.get("hybrid_score"),
                "rerank_score": r.get("rerank_score"),
                "rerank_score_raw": r.get("rerank_score_raw"),
                "table_boost": r.get("table_boost"),
                "text_preview": (r["text"][:240] + "…") if len(r["text"]) > 240 else r["text"],
            }
        )
    return out


class RagEngine:
    def __init__(self):
        self.store: ChunkStore | None = None
        self.dense: DenseRetriever | None = None
        self.bm25: BM25Retriever | None = None
        self.hybrid: HybridRetriever | None = None
        self.reranker: Reranker | None = None
        self.llm: OllamaClient | None = None
        self._ready = False

    # --- lifecycle -------------------------------------------------------
    def load(self) -> "RagEngine":
        """Load indexes and construct model wrappers (lazy model download)."""
        self.store = ChunkStore.load()
        self.dense = DenseRetriever(self.store).load()
        self.bm25 = BM25Retriever(self.store).load()
        self.hybrid = HybridRetriever(self.dense, self.bm25)
        self.reranker = Reranker()
        self.llm = OllamaClient()
        self._ready = True
        return self

    @property
    def ready(self) -> bool:
        return self._ready

    def stats(self) -> dict:
        num_docs = len({c["document_file"] for c in self.store.chunks}) if self.store else 0
        return {
            "num_documents": num_docs,
            "num_chunks": len(self.store) if self.store else 0,
            "embedding_model": settings.EMBEDDING_MODEL,
            "reranker_model": settings.RERANKER_MODEL,
            "llm_model": settings.LLM_MODEL,
        }

    # --- metadata filtering ---------------------------------------------
    def _effective_filter(self, filt: "MetadataFilter | None") -> "MetadataFilter | None":
        """Resolve a filter to one that actually matches chunks.

        Guards against over-filtering: if the requested company+year combination
        matches no chunk (e.g. a company we have for a different year, or a
        mis-detected year), the year constraint is relaxed before giving up so
        we still scope to the right company rather than searching everything.
        Returns ``None`` when no useful restriction remains.
        """
        if filt is None or filt.is_empty():
            return None
        if self.store.allowed_indices(filt):
            return filt
        # Relax the year but keep the company/document scope if that still hits.
        if filt.years and (filt.companies or filt.document_files):
            relaxed = MetadataFilter(
                companies=set(filt.companies),
                document_files=set(filt.document_files),
            )
            if self.store.allowed_indices(relaxed):
                logger.info("Metadata filter relaxed (dropped year %s) to avoid empty pool.",
                            sorted(filt.years))
                return relaxed
        logger.info("Metadata filter matched no chunks; searching full corpus.")
        return None

    def _resolve_filter(self, question: str, explicit: "MetadataFilter | None") -> "MetadataFilter | None":
        """Pick the filter to apply: explicit request wins, else auto-detect."""
        if not settings.ENABLE_METADATA_FILTER:
            return None
        candidate = explicit if (explicit and not explicit.is_empty()) else detect_filter(question)
        return self._effective_filter(candidate)

    # --- core ask --------------------------------------------------------
    def answer(self, question: str, metadata_filter: "MetadataFilter | None" = None) -> dict:
        if not self._ready:
            raise RuntimeError("RagEngine not loaded. Call load() first.")

        question = (question or "").strip()
        if not question:
            raise ValueError("Question must not be empty.")

        # 0. Resolve the metadata scope (explicit filter or auto-detected from
        # the question), with relaxation so we never over-filter to nothing.
        active_filter = self._resolve_filter(question, metadata_filter)

        # 0b. Controlled query expansion for RECALL: for data-seeking questions,
        # widen the dense + BM25 query with curated concept synonyms and
        # fiscal-year normalization so a terse exact-value table row still enters
        # the candidate union. The reranker and LLM below keep the ORIGINAL
        # question, so precision and grounding are unchanged.
        expanded = build_retrieval_query(question)
        retrieval_query = expanded.retrieval_text

        # 1-3. Dense + BM25 + hybrid fusion (scoped to the active filter). Both
        # retrievers see the expanded recall query; everything downstream uses
        # the original question.
        fused = self.hybrid.search(retrieval_query, metadata_filter=active_filter)
        hybrid_results = fused["hybrid_results"]
        # Rerank over the full fused union so a strong single-retriever hit
        # (e.g. a number-dense table row BM25 ranks highly but dense misses)
        # still reaches the cross-encoder instead of being dropped by fusion.
        candidate_pool = fused.get("candidate_pool") or hybrid_results

        retrieval_details = {
            "dense_results": _slim(fused["dense_results"]),
            "bm25_results": _slim(fused["bm25_results"]),
            "hybrid_results": _slim(hybrid_results),
            "reranked_results": [],
            "alpha": fused["alpha"],
            "applied_filter": active_filter.describe() if active_filter else {"active": False},
            "query_expansion": expanded.describe(),
        }

        # No candidates at all -> out of scope.
        if not candidate_pool:
            return self._refusal(OUT_OF_SCOPE_MSG, retrieval_details, reason="no_candidates")

        # 4. Cross-encoder rerank over the FULL pool (one predict call), then
        # assemble the evidence with a rank-aware safety net so strong
        # retrieval-consensus chunks the cross-encoder demoted are still shown
        # to the LLM.
        reranked_full = self.reranker.rerank(question, candidate_pool, top_k=len(candidate_pool))
        # Table-aware boost: for data-seeking questions, promote TABLE/FIGURE
        # chunks so an exact-value row the cross-encoder demoted below generic
        # narrative/header text is re-sorted back to the top before selection.
        reranked_full = apply_table_boost(question, reranked_full)
        reranked = reranked_full[: settings.RERANK_TOP_K]
        evidence = self._assemble_evidence(reranked_full, fused)
        retrieval_details["reranked_results"] = _slim(reranked)
        retrieval_details["evidence"] = _slim(evidence)

        # 5. Answerability / hallucination control (based on the reranker's best
        # score, unchanged - thresholds are NOT loosened).
        top_rerank = reranked_full[0]["rerank_score"] if reranked_full else float("-inf")
        top_hybrid = hybrid_results[0]["hybrid_score"] if hybrid_results else 0.0

        if top_rerank < settings.MIN_RERANK_SCORE or top_hybrid < settings.MIN_HYBRID_SCORE:
            # Weak evidence: distinguish "totally unrelated" from "financial but
            # unsupported" using the reranker signal.
            if top_rerank < settings.MIN_RERANK_SCORE - 3.0:
                msg = OUT_OF_SCOPE_MSG
                reason = "out_of_scope"
            else:
                msg = INSUFFICIENT_MSG
                reason = "insufficient_evidence"
            self._debug(question, retrieval_details, evidence, answer=msg, status=reason)
            return self._refusal(msg, retrieval_details, reason=reason, evidence=reranked)

        # 6. Grounded generation over the assembled evidence.
        try:
            llm_resp = self.llm.generate(question, evidence)
            # 6b. Controlled single retry: if the model refused despite passing
            # the answerability gate (strong evidence is present), OR if it
            # echoed internal retrieval metadata instead of answering, retry
            # ONCE with a stronger evidence-focused prompt. No unbounded loop.
            if (
                self._is_refusal(llm_resp.text) or looks_like_metadata_echo(llm_resp.text)
            ) and settings.ENABLE_REFUSAL_RETRY:
                logger.info("LLM refused or echoed metadata with strong evidence present; "
                            "retrying once (focused).")
                retry = self.llm.generate(question, evidence, focus=True)
                if not (self._is_refusal(retry.text) or looks_like_metadata_echo(retry.text)):
                    llm_resp = retry
        except LLMUnavailableError as exc:
            # No answer was generated, so there is nothing to cite or verify.
            # Return a neutral verification state (not a red "Not Supported")
            # and no citations. Retrieval details remain available for inspection.
            return {
                "answer": str(exc),
                "citations": [],
                "verification": {
                    "overall_status": "NOT_APPLICABLE",
                    "summary": "No answer was generated because the local LLM is unavailable.",
                    "per_citation": [],
                },
                "retrieval_details": retrieval_details,
                "llm_available": False,
                "status": "llm_unavailable",
            }

        # Scrub any internal retrieval metadata the model may have copied
        # (chunk ids, "Page: n", content-type headers) so bookkeeping is never
        # treated as answer content or scored as figures. [n] markers are kept.
        answer_text = strip_retrieval_metadata(llm_resp.text).strip()

        # If the model itself declined (even after the focused retry), or the
        # response was pure metadata that scrubbed to nothing, surface the
        # insufficient-evidence message without fake citations.
        if not answer_text or self._is_refusal(answer_text):
            self._debug(question, retrieval_details, evidence, answer=INSUFFICIENT_MSG,
                        status="insufficient_evidence")
            return {
                "answer": INSUFFICIENT_MSG,
                "citations": [],
                "verification": {
                    "overall_status": "NOT_APPLICABLE",
                    "summary": "The model reported insufficient evidence.",
                    "per_citation": [],
                },
                "retrieval_details": retrieval_details,
                "llm_available": True,
                "status": "insufficient_evidence",
            }

        # 7. Citation mapping (candidate citations for the raw answer). Numbering
        # is anchored to the exact `evidence` list handed to the LLM.
        candidate_citations = build_citations(answer_text, evidence)

        # 8. Claim-level verification gate.
        # Drop any claim whose cited evidence does not support it, so an
        # unsupported claim can never appear as a factual statement. This makes
        # citation verification actually control the final answer.
        gated = filter_supported_claims(answer_text, candidate_citations, evidence)
        answer_text = gated["answer"]

        # If no claim survived verification, we have no supported evidence to
        # stand behind: say the evidence is insufficient instead of guessing.
        if not answer_text:
            self._debug(question, retrieval_details, evidence, answer=INSUFFICIENT_MSG,
                        status="insufficient_evidence", dropped=gated.get("dropped"))
            return {
                "answer": INSUFFICIENT_MSG,
                "citations": [],
                "verification": {
                    "overall_status": "NOT_APPLICABLE",
                    "summary": (
                        "The generated claims were not supported by the cited "
                        "evidence, so no reliable answer could be produced."
                    ),
                    "per_citation": [],
                },
                "retrieval_details": retrieval_details,
                "llm_available": True,
                "status": "insufficient_evidence",
            }

        # 9. Rebuild citations from the filtered answer so only evidence backing
        # the surviving claims is surfaced.
        citations = build_citations(answer_text, evidence)
        if not citations:
            # Answer survived but omitted markers: attach the top evidence so the
            # user still sees the source (never invented).
            citations = all_evidence_as_citations(evidence[:1])

        # 10. Final citation verification on the vetted answer.
        verification = verify_answer(answer_text, citations, evidence)

        self._debug(question, retrieval_details, evidence, answer=answer_text,
                    status="ok", verification=verification)
        return {
            "answer": answer_text,
            "citations": citations,
            "verification": verification,
            "retrieval_details": retrieval_details,
            "llm_available": True,
            "status": "ok",
        }

    # --- evidence assembly / diagnostics --------------------------------
    @staticmethod
    def _is_refusal(text: str) -> bool:
        return INSUFFICIENT_MSG.lower()[:40] in (text or "").lower()

    @staticmethod
    def _assemble_evidence(reranked_full: list[dict], fused: dict) -> list[dict]:
        """Cross-encoder top-K plus a rank-aware safety net.

        The evidence sent to the LLM is the cross-encoder's top RERANK_TOP_K,
        followed by a safety net that guarantees the strongest RETRIEVAL hits
        are present even when the (noisy) cross-encoder demoted them:

        * top HYBRID_SAFETY_TOP_K by fused/RRF rank (retrieval consensus), and
        * top MODALITY_SAFETY_TOP_K from EACH individual retriever (dense and
          BM25) so a strong single-modality hit - e.g. a chunk dense ranks #5
          but BM25 misses, which RRF necessarily discounts - is not lost.

        This directly fixes the observed failure where a chunk both retrievers
        (or one retriever strongly) rank at the top is dropped by the
        cross-encoder before it can be used, WITHOUT loosening any threshold.
        Citation numbers follow this final order.
        """
        rerank_by_id = {c["chunk_id"]: c for c in reranked_full}
        top = [dict(c) for c in reranked_full[: settings.RERANK_TOP_K]]
        seen = {c["chunk_id"] for c in top}

        def _add(rows: list[dict], n: int) -> None:
            for r in rows[:n]:
                cid = r["chunk_id"]
                if cid not in seen:
                    item = dict(rerank_by_id.get(cid, r))
                    item["safety_net"] = True
                    top.append(item)
                    seen.add(cid)

        _add(fused.get("hybrid_results", []), settings.HYBRID_SAFETY_TOP_K)
        _add(fused.get("dense_results", []), settings.MODALITY_SAFETY_TOP_K)
        _add(fused.get("bm25_results", []), settings.MODALITY_SAFETY_TOP_K)

        for i, e in enumerate(top, start=1):
            e["evidence_rank"] = i
        return top

    def _debug(self, question, retrieval_details, evidence, answer, status,
               verification=None, dropped=None) -> None:
        """Emit a compact per-stage trace when RETRIEVAL_DEBUG is on."""
        if not settings.RETRIEVAL_DEBUG:
            return
        try:
            lines = [f"\n=== RETRIEVAL DEBUG === status={status}", f"Q: {question}"]
            for label, key in (("DENSE", "dense_results"), ("BM25", "bm25_results"),
                               ("HYBRID", "hybrid_results")):
                rows = retrieval_details.get(key, [])[:8]
                lines.append(f"[{label}] top {len(rows)}:")
                for r in rows:
                    lines.append(
                        f"  #{r.get('rank')} {r['chunk_id'][:10]} p{r['page_number']} "
                        f"{r.get('content_type')} d={r.get('dense_score')} b={r.get('bm25_score')} "
                        f"h={r.get('hybrid_score')}"
                    )
            lines.append(f"[CANDIDATE POOL] size={len(retrieval_details.get('hybrid_results', []))}+ (full union reranked)")
            lines.append("[EVIDENCE -> LLM]:")
            for e in _slim(evidence):
                lines.append(
                    f"  [{e.get('final_rank') or e.get('rank')}] {e['chunk_id'][:10]} "
                    f"p{e['page_number']} {e.get('content_type')} rerank={e.get('rerank_score')}"
                )
            lines.append(f"[ANSWER] {answer[:300]}")
            if verification:
                lines.append(f"[VERIFICATION] {verification.get('overall_status')}")
                for pc in verification.get("per_citation", []):
                    lines.append(
                        f"  [{pc['citation_id']}] {pc['status']} cov={pc.get('coverage')} "
                        f"nums={pc.get('numbers_matched')} reason={pc.get('reason')}"
                    )
            if dropped:
                lines.append(f"[DROPPED CLAIMS] {len(dropped)}")
                for d in dropped:
                    lines.append(f"  - {d.get('reason')}: {str(d.get('claim'))[:120]}")
            logger.info("\n".join(lines))
        except Exception:  # never let debug logging break a response
            logger.exception("debug trace failed")

    def _refusal(self, message: str, retrieval_details: dict, reason: str, evidence: list[dict] | None = None) -> dict:
        # A refusal contains no factual claim, so verification is not applicable
        # (rather than a red "Not Supported").
        return {
            "answer": message,
            "citations": [],
            "verification": {
                "overall_status": "NOT_APPLICABLE",
                "summary": message,
                "per_citation": [],
            },
            "retrieval_details": retrieval_details,
            "llm_available": True,
            "status": reason,
        }


# Module-level singleton used by the API.
_engine: RagEngine | None = None


def get_engine() -> RagEngine:
    global _engine
    if _engine is None:
        _engine = RagEngine().load()
    return _engine
