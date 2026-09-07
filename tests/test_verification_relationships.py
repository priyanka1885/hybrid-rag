"""Tests for year->value relationship-aware citation verification.

These cover the core failure the fix targets: an answer can pair the right
metric/year with the WRONG value ("8,347.9 in 2022") while both tokens appear in
the evidence. Verification must reject the relationship, not merely confirm the
tokens are present. It must also verify citations independently of claim
grounding and aggregate multiple claims conservatively.
"""
from backend.citations.table_relations import (
    check_relationship,
    claim_year_value_pairs,
    evidence_year_values,
    parse_tables,
)
from backend.citations.verifier import (
    filter_supported_claims,
    verify_answer,
    verify_claim,
)

# The exact reconstructed-table evidence from the task description.
TABLE_EVIDENCE = (
    "| Metric | 2021 | 2022 | 2023 |\n"
    "| --- | --- | --- | --- |\n"
    "| Employees | 7959.0 | 7836.3 | 8347.9 |"
)


# --- table parser ----------------------------------------------------------
def test_parse_wide_table():
    rels = parse_tables(TABLE_EVIDENCE)
    assert rels == [{"Employees": {"2021": "7959.0", "2022": "7836.3", "2023": "8347.9"}}]


def test_parse_long_table():
    ev = (
        "| Year | Employees |\n"
        "| --- | --- |\n"
        "| 2021 | 7959 |\n"
        "| 2022 | 7836 |\n"
        "| 2023 | 8347 |"
    )
    rels = parse_tables(ev)
    assert rels == [{"Employees": {"2021": "7959", "2022": "7836", "2023": "8347"}}]


def test_evidence_year_values():
    yv = evidence_year_values(TABLE_EVIDENCE)
    assert yv["2021"] == {"7959.0"}
    assert yv["2022"] == {"7836.3"}
    assert yv["2023"] == {"8347.9"}


def test_claim_pairs_bind_nearest_year():
    pairs = claim_year_value_pairs("Employees were 7,836.3 in 2022 and 8,347.9 in 2023")
    assert ("7836.3", "2022") in pairs
    assert ("8347.9", "2023") in pairs


def test_check_relationship_wrong_year_value():
    res = check_relationship("Employees were 8,347.9 in 2022", TABLE_EVIDENCE)
    assert res["checked"] is True
    assert res["consistent"] is False


def test_check_relationship_correct_year_value():
    res = check_relationship("Employees were 7,836.3 in 2022", TABLE_EVIDENCE)
    assert res["checked"] is True
    assert res["consistent"] is True


# --- exact failure-case scenarios (tests 1-4) ------------------------------
def test_case1_correct_2022_supported():
    res = verify_claim("Employees were 7,836.3 in 2022 [1].", TABLE_EVIDENCE)
    assert res["status"] == "SUPPORTED"


def test_case2_correct_2023_supported():
    res = verify_claim("Employees were 8,347.9 in 2023 [1].", TABLE_EVIDENCE)
    assert res["status"] == "SUPPORTED"


def test_case3_wrong_year_value_not_supported():
    # 8,347.9 is the 2023 value, not 2022 - both tokens exist, relationship wrong.
    res = verify_claim("Employees were 8,347.9 in 2022 [1].", TABLE_EVIDENCE)
    assert res["status"] == "NOT_SUPPORTED"


def test_case4_absent_value_not_supported():
    res = verify_claim("Employees were 9,999.9 in 2022 [1].", TABLE_EVIDENCE)
    assert res["status"] == "NOT_SUPPORTED"


def test_integer_vs_decimal_tolerance():
    # 7959 (claim) == 7959.0 (evidence): relationship still holds.
    res = verify_claim("Employees were 7959 in 2021 [1].", TABLE_EVIDENCE)
    assert res["status"] == "SUPPORTED"


# --- test 5: citation vs claim grounding are independent -------------------
def test_case5_wrong_citation_not_silently_approved():
    # [2] is unrelated; the correct table is in [1]. Citation [2] must be
    # NOT_SUPPORTED even though [1] contains the answer.
    evidence = [
        {  # [1] - the real table
            "chunk_id": "c1",
            "document_name": "Report",
            "page_number": 10,
            "content_type": "table",
            "text": TABLE_EVIDENCE,
        },
        {  # [2] - unrelated evidence
            "chunk_id": "c2",
            "document_name": "Report",
            "page_number": 22,
            "content_type": "text",
            "text": "The board comprises twelve independent non-executive directors.",
        },
    ]
    answer = "Employees were 7,836.3 in 2022 [2]."
    citations = [
        {
            "citation_id": 2,
            "document_name": "Report",
            "page_number": 22,
            "chunk_id": "c2",
            "supporting_text": evidence[1]["text"],
            "content_type": "text",
            "visual_ref": None,
        }
    ]
    # Verified honestly against its OWN cited text, [2] does NOT support the
    # claim - it is never silently approved.
    v = verify_answer(answer, citations, evidence)
    statuses = {p["citation_id"]: p["status"] for p in v["per_citation"]}
    assert statuses[2] == "NOT_SUPPORTED"
    assert v["overall_status"] == "NOT_SUPPORTED"

    # Recovery: the claim IS supported by another retrieved chunk ([1], the
    # table), so the filter retains it and RE-POINTS the marker to [1] rather
    # than keeping the wrong [2] or dropping a correct answer.
    gated = filter_supported_claims(answer, citations, evidence)
    assert gated["answer"] != ""
    assert "[1]" in gated["answer"]
    assert "[2]" not in gated["answer"]
    assert 1 in gated["kept_ids"]


# --- test 6: a supported claim must not hide an unsupported one ------------
def test_case6_mixed_claims_overall_not_supported():
    evidence = [
        {
            "chunk_id": "c1",
            "document_name": "Report",
            "page_number": 10,
            "content_type": "table",
            "text": TABLE_EVIDENCE,
        },
        {
            "chunk_id": "c2",
            "document_name": "Report",
            "page_number": 40,
            "content_type": "text",
            "text": "Water withdrawal totalled 512 megalitres during the period.",
        },
    ]
    citations = [
        {
            "citation_id": 1,
            "document_name": "Report",
            "page_number": 10,
            "chunk_id": "c1",
            "supporting_text": evidence[0]["text"],
            "content_type": "table",
            "visual_ref": None,
        },
        {
            "citation_id": 2,
            "document_name": "Report",
            "page_number": 40,
            "chunk_id": "c2",
            "supporting_text": evidence[1]["text"],
            "content_type": "text",
            "visual_ref": None,
        },
    ]
    # Claim 1 supported by [1]; claim 2 ("Revenue increased in 2023") is not in [2].
    answer = "Employees were 7,836.3 in 2022 [1]. Revenue increased in 2023 [2]."
    v = verify_answer(answer, citations, evidence)
    assert v["overall_status"] == "NOT_SUPPORTED"


def test_multi_year_single_sentence_all_correct_supported():
    res = verify_claim(
        "Employees were 7,836.3 in 2022 and 8,347.9 in 2023 [1].", TABLE_EVIDENCE
    )
    assert res["status"] == "SUPPORTED"


def test_multi_year_single_sentence_one_wrong_not_supported():
    # Second pair is wrong (8,347.9 is 2023, not 2022) -> whole claim rejected.
    res = verify_claim(
        "Employees were 7,959.0 in 2021 and 8,347.9 in 2022 [1].", TABLE_EVIDENCE
    )
    assert res["status"] == "NOT_SUPPORTED"


def test_marker_after_period_is_not_orphaned():
    # The LLM commonly cites AFTER the period: "... in 2022. [1]". The claim
    # must NOT be dropped as uncited - this was a real cause of correct answers
    # collapsing to the fallback.
    citations = [
        {
            "citation_id": 1,
            "document_name": "Report",
            "page_number": 10,
            "chunk_id": "c1",
            "supporting_text": TABLE_EVIDENCE,
            "content_type": "table",
            "visual_ref": None,
        }
    ]
    answer = "Employees were 7,836.3 in 2022. [1]"
    gated = filter_supported_claims(answer, citations, [
        {"chunk_id": "c1", "document_name": "Report", "page_number": 10,
         "content_type": "table", "text": TABLE_EVIDENCE},
    ])
    assert gated["answer"] != ""
    assert "[1]" in gated["answer"]
    assert 1 in gated["kept_ids"]
