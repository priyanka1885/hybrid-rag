"""Tests for numeric normalization and format-aware citation verification.

Covers the financial-table OCR case where thousands separators are rendered as
spaces (e.g. "64 392") and the LLM may restate the figure with a comma
("64,392") or plainly ("64392"). All three must be treated as the same value,
while genuinely different figures must still be rejected, and incidental digits
(Scope 1/2, years, percentages, citation markers) must never be merged.
"""
from backend.citations.verifier import verify_claim
from backend.text_normalize import (
    canonicalize_numbers,
    join_comma_thousands,
    join_space_thousands,
)

# A representative OCR'd financial-table row (label + four yearly columns).
EVIDENCE = (
    "Total greenhouse gas (CO equivalent) (kilotons) 64392 63891 66273 64829 Reasonable"
)


# --- normalization: merges true thousands ---------------------------------
def test_join_space_thousands_merges_grouped_numbers():
    assert join_space_thousands("64 392") == "64392"
    assert join_space_thousands("1 234 567") == "1234567"
    assert join_space_thousands("58 644 tons") == "58644 tons"


def test_join_comma_thousands_merges_grouped_numbers():
    assert join_comma_thousands("64,392") == "64392"
    assert join_comma_thousands("1,234,567") == "1234567"


# --- normalization: safety (must NOT merge) -------------------------------
def test_normalization_preserves_scope_labels_years_percentages_markers():
    assert join_space_thousands("Scope 1 and Scope 2") == "Scope 1 and Scope 2"
    assert join_space_thousands("in 2022 and 2023") == "in 2022 and 2023"
    assert join_space_thousands("a 30% reduction by 2030") == "a 30% reduction by 2030"
    assert join_space_thousands("markers [1] and [2]") == "markers [1] and [2]"


def test_normalization_does_not_merge_independent_three_digit_columns():
    # A row of independent 3-digit table cells (each exactly 3 digits) must be
    # left untouched: no cell has the 1-2 digit lead required to start a match.
    assert join_space_thousands("643 635 371 417") == "643 635 371 417"
    assert join_space_thousands("484 977 504 822") == "484 977 504 822"


def test_canonicalize_folds_formats_and_keeps_decimals():
    def nums(t):
        import re
        return set(re.findall(r"\d+(?:\.\d+)?", canonicalize_numbers(t)))

    assert nums("64 392") == {"64392"}
    assert nums("64,392") == {"64392"}
    assert nums("64392") == {"64392"}
    # European decimal comma stays a decimal, not a thousands separator.
    assert nums("63,4") == {"63.4"}
    # A different value stays different.
    assert nums("99,900") == {"99900"}


# --- format-aware citation verification -----------------------------------
def test_verify_space_separated_value_is_supported():
    res = verify_claim(
        "Sasol's total greenhouse gas emissions were 64 392 kilotons CO2e [1]",
        EVIDENCE,
    )
    assert res["numbers_matched"] is True
    assert res["status"] != "NOT_SUPPORTED"


def test_verify_comma_separated_value_is_supported():
    res = verify_claim(
        "Sasol's total greenhouse gas emissions were 64,392 kilotons CO2e [1]",
        EVIDENCE,
    )
    assert res["numbers_matched"] is True
    assert res["status"] != "NOT_SUPPORTED"


def test_verify_plain_value_is_supported():
    res = verify_claim(
        "Sasol's total greenhouse gas emissions were 64392 kilotons CO2e [1]",
        EVIDENCE,
    )
    assert res["numbers_matched"] is True
    assert res["status"] != "NOT_SUPPORTED"


def test_verify_rejects_genuinely_different_value():
    # Hallucination detection must remain intact: 99,900 != 64,392.
    res = verify_claim(
        "Sasol's total greenhouse gas emissions were 99,900 kilotons CO2e [1]",
        EVIDENCE,
    )
    assert res["numbers_matched"] is False
    assert res["status"] == "NOT_SUPPORTED"


def test_verify_scope_labels_do_not_break_numeric_grounding():
    # Bare "Scope 1"/"Scope 2" and "CO2" digits must not force a mismatch.
    res = verify_claim(
        "Scope 1 and Scope 2 greenhouse gas emissions were reported [1]",
        EVIDENCE,
    )
    assert res["numbers_matched"] is True
