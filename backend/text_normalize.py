"""
Shared numeric-text normalization for financial reports.

OCR of financial tables renders thousands separators as spaces, e.g. the total
greenhouse-gas figure ``64 392`` or ``1 234 567``. Left as-is, a plain word/number
tokenizer splits these into ``64`` + ``392``, so an exact figure is never a single
token for BM25 and a correctly grounded claim can fail numeric verification.

This module normalizes *thousands-grouped* numbers back into a single value while
being deliberately conservative so it never merges genuinely separate numbers:

    64 392            -> 64392
    1 234 567         -> 1234567
    64,392            -> 64392        (US comma grouping, verifier side)
    1,234,567         -> 1234567

Safety (things that MUST NOT be merged), guaranteed by the "leading group of only
1-2 digits" rule below:

    Scope 1 / Scope 2        single digits, never matched
    2022 2023                4-digit years, no 3-digit grouping
    30%                      no grouping
    [1] citation markers     single digit
    643 635 371 417          a table row of separate 3-digit values. Every value
                             is exactly 3 digits, so no valid match can *start*
                             (a match needs a 1-2 digit lead not preceded by a
                             digit), leaving the whole run untouched.

The "leading 1-2 digit" rule is what distinguishes a formatted number like
``64 392`` (lead ``64``) from a column of independent 3-digit cells.
"""
from __future__ import annotations

import re

# A thousands-grouped number written with SPACE separators.
#
# The leading group is 1-3 digits so a genuine large value such as ``224 593 325``
# (hundreds of millions) collapses too, not just up to six-figure values. To stay
# safe against a run of INDEPENDENT 3-digit table cells (e.g. ``643 635 371 417``,
# which is four separate cells, not one number) two guards are used:
#   - the match may cover at most three groups ``{1,2}`` trailing 3-digit groups,
#     i.e. up to nine digits; and
#   - it must be neither preceded by a "<digit><space>" nor followed by a
#     "<optional-space><digit>", so it can never start or end in the middle of a
#     longer run of 3-digit cells.
# Together these let an isolated ``224 593 325`` merge while a dense row of
# 3-digit cells is left completely untouched (no partial merges either).
_SPACE_THOUSANDS_RE = re.compile(
    r"(?<![\d.,])(?<![\d.,][ \u00a0])(\d{1,3}(?:[ \u00a0]\d{3}){1,2})(?![ \u00a0]?\d)"
)

# The same, but with COMMA separators (US style). Used only on the claim side of
# verification, where an LLM may emit ``64,392`` or ``224,593,325``. A 1-3 digit
# leading group means large figures collapse correctly (``224,593,325`` ->
# ``224593325``) instead of the two commas leaking through to the decimal rule
# and corrupting the value into ``224.593.325``. Commas are unambiguous thousands
# separators here (table cells are space-separated, not comma-separated), so no
# run-length guard is needed.
_COMMA_THOUSANDS_RE = re.compile(r"(?<![\d.])(\d{1,3}(?:,\d{3})+)(?!\d)")

# A comma used as a decimal separator (South-African / European style: ``63,4``).
_COMMA_DECIMAL_RE = re.compile(r"(?<=\d),(?=\d)")


def join_space_thousands(text: str) -> str:
    """Collapse space-separated thousands groups (``64 392`` -> ``64392``)."""
    return _SPACE_THOUSANDS_RE.sub(
        lambda m: m.group(1).replace(" ", "").replace("\u00a0", ""), text
    )


def join_comma_thousands(text: str) -> str:
    """Collapse comma-separated thousands groups (``64,392`` -> ``64392``)."""
    return _COMMA_THOUSANDS_RE.sub(lambda m: m.group(1).replace(",", ""), text)


def canonicalize_numbers(text: str) -> str:
    """Canonicalize every numeric form to a comparable value string.

    Handles both space- and comma-grouped thousands and comma decimals so that
    ``64 392``, ``64,392`` and ``64392`` all reduce to ``64392`` while a genuine
    European decimal (``63,4``) becomes ``63.4`` and different values stay
    different (``99,900`` -> ``99900`` never equals ``64392``).
    """
    t = join_space_thousands(text)
    t = join_comma_thousands(t)
    t = _COMMA_DECIMAL_RE.sub(".", t)
    return t


# --- Retrieval-metadata scrubbing -----------------------------------------
# An alphanumeric identifier / hash such as a chunk id ("620fb5572c74334"):
# length >= 8 and containing BOTH a letter and a digit. Currency-prefixed
# figures ("R1234") are short (< 8) and pure figures ("58644") have no letter,
# so neither is matched. The digit runs INSIDE such an id ("620", "5572",
# "74334") must never be treated as financial values.
_ID_TOKEN_RE = re.compile(
    r"\b(?=[A-Za-z0-9]*[A-Za-z])(?=[A-Za-z0-9]*\d)[A-Za-z0-9]{8,}\b"
)

# Internal retrieval metadata rendered as "Key: value" (the numbered-evidence
# header the LLM must never copy into an answer). Each segment stops at a pipe
# or newline so only the metadata value is removed, not following prose.
_METADATA_KV_RE = re.compile(
    r"(?i)\b(?:document|source|page|chunk\s*id|content\s*type|rank|score)\b\s*:\s*[^|\n\[.]*"
)


def strip_id_tokens(text: str) -> str:
    """Remove alphanumeric identifiers/hashes (e.g. chunk ids) from ``text``.

    This prevents the digit runs embedded in an id from being mis-read as
    financial figures during numeric grounding.
    """
    return _ID_TOKEN_RE.sub(" ", text)


def strip_retrieval_metadata(text: str) -> str:
    """Strip internal retrieval metadata from a piece of model-facing text.

    Removes "Document/Source/Page/Chunk ID/Content type/Rank/Score: value"
    segments and any leftover id/hash tokens, so retrieval bookkeeping can never
    be treated as answer content. Bracketed ``[n]`` citation markers are left
    intact (handled elsewhere). Orphaned pipe separators and doubled whitespace
    left behind by the removal are tidied up.
    """
    t = _METADATA_KV_RE.sub(" ", text)
    t = strip_id_tokens(t)
    t = re.sub(r"\s*\|\s*", " ", t)  # drop orphan "|" separators from the header
    t = re.sub(r"[ \t]{2,}", " ", t)
    return t
