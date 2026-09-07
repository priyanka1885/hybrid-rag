"""Tests for numeric + semantic citation grounding (verify_claim).

These cover the bug where a numerically correct but verbosely phrased financial
claim was down-graded to PARTIALLY_SUPPORTED purely because of low lexical
token overlap with a terse table row. The fix adds a numeric + semantic
grounding path: when the structured relationship check is UNAVAILABLE, a claim
whose figures are all present AND each tied to the claim's own metric label is
SUPPORTED - without weakening contradiction or wrong-metric detection.
"""
from backend.citations.verifier import verify_claim

# Flattened (non-reconstructable) evidence: the Sasol total-GHG row as extracted
# from the PDF. There is no Markdown/pipe structure, so the relationship checker
# returns {"checked": False} and the numeric+semantic path must decide.
FLAT_GHG = (
    "Total greenhouse gas (CO2 equivalent) (kilotons) 64 392 63 891 66 273 64 829"
)


def test_case1_verbose_correct_claim_supported():
    claim = (
        "According to the report, Sasol's total greenhouse gas emissions, "
        "measured as carbon dioxide equivalent, amounted to approximately "
        "64,392 kilotons during the 2023 financial year, compared with roughly "
        "63,891 kilotons recorded in the previous year 2022. [1]"
    )
    res = verify_claim(claim, FLAT_GHG)
    assert res["status"] == "SUPPORTED"
    assert res["numbers_matched"] is True
    # The whole point: lexical coverage is low, yet the claim is SUPPORTED.
    assert res["coverage"] < 0.45


def test_case2_wrong_number_not_supported():
    claim = "Sasol's total greenhouse gas emissions were 70,000 kilotons. [1]"
    res = verify_claim(claim, FLAT_GHG)
    assert res["status"] == "NOT_SUPPORTED"
    assert res["numbers_matched"] is False


# Structured evidence that clearly establishes 2023 -> 64,392 and 2022 -> 63,891.
TABLE_GHG = (
    "| Metric | 2023 | 2022 |\n"
    "| --- | --- | --- |\n"
    "| Total greenhouse gas (CO2 equivalent) (kilotons) | 64 392 | 63 891 |"
)


def test_case3_wrong_relationship_not_supported():
    # Claim swaps the years: says 2023 -> 63,891 and 2022 -> 64,392. Both numbers
    # exist, but the year->value relationship is wrong. Numeric grounding must
    # NOT override the detected contradiction.
    claim = (
        "Total greenhouse gas emissions were 63,891 kilotons in 2023 and "
        "64,392 kilotons in 2022. [1]"
    )
    res = verify_claim(claim, TABLE_GHG)
    assert res["status"] == "NOT_SUPPORTED"


# Evidence where two metrics carry two different figures.
FLAT_MIXED = (
    "Total greenhouse gas (CO2 equivalent) 64 392. Scope 1 emissions 58 644."
)


def test_case4_right_number_wrong_metric_not_supported():
    # 64,392 exists in the evidence, but it belongs to Total GHG, not Scope 1.
    claim = "Scope 1 emissions were 64,392. [1]"
    res = verify_claim(claim, FLAT_MIXED)
    assert res["status"] != "SUPPORTED"
    assert res["status"] in ("NOT_SUPPORTED", "PARTIALLY_SUPPORTED")


def test_case5_terse_correct_claim_supported():
    claim = (
        "Sasol's total greenhouse gas emissions were 64,392 kilotons in 2023 "
        "and 63,891 kilotons in 2022. [1]"
    )
    res = verify_claim(claim, FLAT_GHG)
    assert res["status"] == "SUPPORTED"
    assert res["numbers_matched"] is True
