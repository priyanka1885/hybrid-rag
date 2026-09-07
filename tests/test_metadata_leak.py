"""Regression tests for retrieval-metadata leaking into answers/verification.

Bug: the LLM sometimes echoed the numbered-evidence header
("Document: ... | Page: 41 | Chunk ID: 620fb5572c74334"). The verifier then read
the page number and the digit runs inside the chunk-id hash as "claimed
figures", none of which are in the evidence, so a numerically CORRECT answer was
rejected as insufficient_evidence.

The fix: keep internal metadata out of the prompt, scrub it out of answers,
ignore id/metadata digits during number extraction, and regenerate metadata-only
responses instead of verifying them. Nothing about the answer is hardcoded - the
expected values come straight from the evidence text.
"""
from backend.citations.mapper import build_citations
from backend.citations.verifier import (
    _financial_figures,
    filter_supported_claims,
    verify_answer,
    verify_claim,
)
from backend.generation.llm import build_context_block, looks_like_metadata_echo
from backend.text_normalize import strip_retrieval_metadata

# Real Sasol page-58 evidence, flattened exactly as extracted from the PDF (no
# Markdown/pipe structure, so relationship reconstruction is unavailable).
GHG_EVIDENCE_TEXT = (
    "Total greenhouse gas (CO2 equivalent) (kilotons) 64 392 63 891 66 273 64 829"
)
EVIDENCE = [
    {
        "chunk_id": "620fb5572c74334",
        "document_name": "Sasol Sustainability Report 2023",
        "page_number": 58,
        "content_type": "text",
        "text": GHG_EVIDENCE_TEXT,
    }
]

QUESTION = "What were Sasol total greenhouse gas emissions in 2023 and 2022?"


def test_prompt_context_excludes_internal_metadata():
    """The LLM prompt must not contain chunk ids or page numbers to echo."""
    block = build_context_block(EVIDENCE)
    assert "Chunk ID" not in block
    assert "620fb5572c74334" not in block
    assert "Page:" not in block
    # ...but the actual evidence content is still present.
    assert "64 392" in block and "63 891" in block


def test_metadata_echo_is_detected():
    echo = (
        "The relevant table is Document: Sasol Sustainability Report 2023 | "
        "Page: 41 | Chunk ID: 620fb5572c74334"
    )
    assert looks_like_metadata_echo(echo) is True


def test_genuine_answer_is_not_flagged_as_metadata_echo():
    good = (
        "Sasol's total greenhouse gas emissions were 64,392 kilotons in 2023 "
        "and 63,891 kilotons in 2022 [1]."
    )
    assert looks_like_metadata_echo(good) is False


def test_number_extraction_ignores_alphanumeric_id_digits():
    # A chunk-id hash must contribute NO figures (620 / 5572 / 74334 ignored).
    assert _financial_figures("620fb5572c74334") == set()
    assert _financial_figures("chunk id 620fb5572c74334 abc123def") == set()
    # A real grouped figure is still extracted.
    assert "64392" in _financial_figures("the value 64 392 in the row")


def test_metadata_polluted_answer_still_supported():
    """The exact failure: a correct answer with leaked metadata must verify."""
    answer = (
        "Sasol's total greenhouse gas emissions were 64,392 kilotons in 2023 "
        "and 63,891 kilotons in 2022. Document: Sasol Sustainability Report 2023 "
        "| Page: 41 | Chunk ID: 620fb5572c74334 [1]"
    )
    res = verify_claim(answer, GHG_EVIDENCE_TEXT)
    assert res["status"] == "SUPPORTED"
    assert res["numbers_matched"] is True


def test_regression_sasol_ghg_2023_2022_end_to_end_verifier():
    """Named regression for QUESTION.

    Expected values come from the evidence: 2023 = 64,392 kilotons,
    2022 = 63,891 kilotons. We drive the post-generation pipeline exactly as
    RagEngine does - scrub metadata, map citations, gate claims, verify - even
    though the raw model output carried leaked metadata, and confirm the answer
    survives with the correct figures instead of collapsing to
    insufficient_evidence.
    """
    # The expected figures are grounded in the evidence (not hardcoded here).
    assert {"64392", "63891"}.issubset(_financial_figures(GHG_EVIDENCE_TEXT))

    raw_llm_output = (
        "Sasol's total greenhouse gas emissions were 64,392 kilotons in 2023 "
        "and 63,891 kilotons in 2022. Document: Sasol Sustainability Report 2023 "
        "| Page: 41 | Chunk ID: 620fb5572c74334 [1]"
    )

    # 1. scrub (as RagEngine does before citation mapping / verification)
    answer_text = strip_retrieval_metadata(raw_llm_output).strip()
    assert "Chunk ID" not in answer_text and "620fb5572c74334" not in answer_text

    # 2. citation mapping + 3. claim-support gate
    citations = build_citations(answer_text, EVIDENCE)
    gated = filter_supported_claims(answer_text, citations, EVIDENCE)
    assert gated["answer"], "correct answer must not be dropped (no false insufficient_evidence)"

    # 4. final verification is SUPPORTED and both figures survive
    final_citations = build_citations(gated["answer"], EVIDENCE)
    verdict = verify_answer(gated["answer"], final_citations, EVIDENCE)
    assert verdict["overall_status"] == "SUPPORTED"
    assert "64,392" in gated["answer"] and "63,891" in gated["answer"]
