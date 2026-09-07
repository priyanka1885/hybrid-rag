"""
Citation mapping.

Parses the [n] markers the LLM emits and maps each one back to the actual
retrieved evidence chunk it refers to. Citations always point at real chunks
with real document names and page numbers - nothing is invented.
"""
from __future__ import annotations

import re

_CITE_RE = re.compile(r"\[(\d+)\]")


def extract_citation_ids(answer: str) -> list[int]:
    """Return the ordered, unique citation numbers referenced in the answer."""
    seen: list[int] = []
    for m in _CITE_RE.finditer(answer):
        n = int(m.group(1))
        if n not in seen:
            seen.append(n)
    return seen


def build_citations(answer: str, evidence: list[dict]) -> list[dict]:
    """Build citation records for every [n] marker that maps to real evidence.

    ``evidence`` is the reranked, numbered evidence list (1-based indexing in
    the answer). Markers that don't map to any evidence item are ignored so we
    never fabricate a citation.
    """
    citations: list[dict] = []
    used_ids = extract_citation_ids(answer)
    for cid in used_ids:
        if 1 <= cid <= len(evidence):
            ev = evidence[cid - 1]
            citations.append(
                {
                    "citation_id": cid,
                    "document_name": ev["document_name"],
                    "page_number": ev["page_number"],
                    "chunk_id": ev["chunk_id"],
                    "supporting_text": ev["text"],
                    "content_type": ev.get("content_type", "text"),
                    "visual_ref": ev.get("visual_ref"),
                }
            )
    return citations


def all_evidence_as_citations(evidence: list[dict]) -> list[dict]:
    """Fallback: expose all evidence as candidate citations (1-based)."""
    return [
        {
            "citation_id": i,
            "document_name": ev["document_name"],
            "page_number": ev["page_number"],
            "chunk_id": ev["chunk_id"],
            "supporting_text": ev["text"],
            "content_type": ev.get("content_type", "text"),
            "visual_ref": ev.get("visual_ref"),
        }
        for i, ev in enumerate(evidence, start=1)
    ]
