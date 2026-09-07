"""Tests for citation mapping and verification."""
from backend.citations.mapper import build_citations, extract_citation_ids
from backend.citations.verifier import verify_answer, verify_claim

EVIDENCE = [
    {
        "chunk_id": "c1",
        "document_name": "Clicks Sustainability Report 2022",
        "page_number": 4,
        "text": "The group reported revenue of 38.9 billion rand for the 2022 financial year.",
    },
    {
        "chunk_id": "c2",
        "document_name": "Clicks Sustainability Report 2022",
        "page_number": 5,
        "text": "Total number of stores increased to 870 across the country.",
    },
]


def test_extract_citation_ids_unique_ordered():
    assert extract_citation_ids("A [1] B [2] C [1]") == [1, 2]
    assert extract_citation_ids("no markers here") == []


def test_build_citations_maps_to_real_evidence():
    answer = "The group reported revenue of 38.9 billion rand [1]."
    cites = build_citations(answer, EVIDENCE)
    assert len(cites) == 1
    c = cites[0]
    assert c["citation_id"] == 1
    assert c["document_name"] == "Clicks Sustainability Report 2022"
    assert c["page_number"] == 4
    assert c["chunk_id"] == "c1"


def test_build_citations_ignores_invalid_markers():
    # [9] does not map to any evidence -> must NOT be fabricated.
    answer = "Some claim [9]."
    cites = build_citations(answer, EVIDENCE)
    assert cites == []


def test_verify_claim_supported():
    res = verify_claim(
        "The group reported revenue of 38.9 billion rand for 2022 [1]",
        EVIDENCE[0]["text"],
    )
    assert res["status"] == "SUPPORTED"
    assert res["numbers_matched"] is True


def test_verify_claim_number_mismatch_not_supported():
    # A hallucinated figure (99.9) not present in evidence.
    res = verify_claim(
        "The group reported revenue of 99.9 billion rand [1]",
        EVIDENCE[0]["text"],
    )
    assert res["numbers_matched"] is False
    assert res["status"] != "SUPPORTED"


def test_verify_answer_overall():
    answer = "The group reported revenue of 38.9 billion rand [1]."
    cites = build_citations(answer, EVIDENCE)
    v = verify_answer(answer, cites)
    assert v["overall_status"] in ("SUPPORTED", "PARTIALLY_SUPPORTED")
    assert len(v["per_citation"]) == 1


def test_verify_answer_no_citations():
    v = verify_answer("Some answer with no citations", [])
    assert v["overall_status"] == "NOT_SUPPORTED"
