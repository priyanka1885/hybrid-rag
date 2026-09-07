"""
Structured-table relationship parsing for citation verification.

The visual ingestion pipeline already reconstructs financial tables/charts as
compact Markdown, e.g.::

    | Metric | 2021 | 2022 | 2023 |
    | --- | --- | --- | --- |
    | Employees | 7959.0 | 7836.3 | 8347.9 |

The citation verifier must confirm not only that a claimed value *exists* in the
evidence, but that the claimed **year -> value** (and, where possible,
**metric -> value**) relationship is the one the table actually asserts. Merely
checking that "2022" and "8347.9" both appear somewhere lets a wrong answer
("8,347.9 in 2022") slip through, because 8347.9 is really the 2023 value.

This module parses such tables into ``{metric: {year: value}}`` mappings and
compares a claim's year->value pairs against them. It is deliberately
conservative: if the structure cannot be reconstructed reliably it returns
"not checked" so the caller falls back to the existing lexical/numeric grounding
instead of inventing a relationship. No company, metric, year, or value is ever
hardcoded.
"""
from __future__ import annotations

import re

from backend.text_normalize import canonicalize_numbers

# A 4-digit reporting year (1900-2099). Used both as a table column header and
# as a row key in "long" tables.
_YEAR_RE = re.compile(r"(?:19|20)\d{2}")
_YEAR_FULLMATCH = re.compile(r"(?:19|20)\d{2}\Z")
_NUM_RE = re.compile(r"\d+(?:\.\d+)?")

# Label RANGE such as "40-49" / "40 - 49" / "20 – 29" (age-band labels): their
# endpoints are category labels, not financial figures, and must not be treated
# as claim values. Common hyphen/dash variants are covered.
_RANGE_RE = re.compile(r"\b\d{1,3}\s*[-\u2010-\u2015\u2212]\s*\d{1,3}\b")


def _is_year(token: str) -> bool:
    return bool(_YEAR_FULLMATCH.match(token.strip()))


def _num(value: str) -> str | None:
    """Return the first numeric token of ``value`` in canonical form.

    Canonicalization folds space/comma-grouped thousands and comma decimals so
    ``7,836.3``/``7 836,3``/``7836.3`` all reduce to the same comparable value.
    Returns ``None`` when the string carries no numeric content.
    """
    m = _NUM_RE.search(canonicalize_numbers(value))
    return m.group() if m else None


def values_equal(a: str, b: str) -> bool:
    """True if two value strings denote the same number.

    Tolerant of formatting and of integer/decimal differences, e.g.
    ``7959`` == ``7959.0`` and ``7,836.3`` == ``7836.3``.
    """
    na, nb = _num(a), _num(b)
    if na is None or nb is None:
        return a.strip() == b.strip()
    if na == nb:
        return True
    try:
        return abs(float(na) - float(nb)) < 1e-9
    except ValueError:
        return False


# --- Markdown / pipe table parsing -----------------------------------------
def _split_pipe_row(line: str) -> list[str]:
    line = line.strip()
    if line.startswith("|"):
        line = line[1:]
    if line.endswith("|"):
        line = line[:-1]
    return [c.strip() for c in line.split("|")]


def _is_separator_row(cells: list[str]) -> bool:
    """A Markdown header separator row like ``| --- | --- |``."""
    saw_dash = False
    for c in cells:
        c = c.strip()
        if c == "":
            continue
        if re.fullmatch(r":?-{2,}:?", c):
            saw_dash = True
        else:
            return False
    return saw_dash


def _has_number(cell: str) -> bool:
    return _num(cell) is not None


def _parse_pipe_block(block: list[str]) -> dict[str, dict[str, str]] | None:
    """Parse one contiguous block of pipe-delimited lines into ``{metric:{year:value}}``.

    Handles two orientations conservatively:

    * **wide** - years are column headers, metrics are rows::

          | Metric | 2021 | 2022 | 2023 |
          | Employees | 7959.0 | 7836.3 | 8347.9 |

    * **long** - years are the first column, metrics are the other columns::

          | Year | Employees |
          | 2021 | 7959 |
          | 2022 | 7836 |
    """
    rows = [_split_pipe_row(l) for l in block if l.strip()]
    rows = [r for r in rows if not _is_separator_row(r)]
    if len(rows) < 2:
        return None

    header = rows[0]
    data_rows = rows[1:]

    # --- wide orientation: >= 2 year columns in the header ---
    year_cols = {idx: c.strip() for idx, c in enumerate(header) if _is_year(c)}
    if len(year_cols) >= 2:
        rel: dict[str, dict[str, str]] = {}
        for drow in data_rows:
            if not drow:
                continue
            metric = (drow[0].strip() or "Series")
            ymap: dict[str, str] = {}
            for idx, yr in year_cols.items():
                if idx < len(drow):
                    val = drow[idx].strip()
                    if val and _has_number(val):
                        ymap[yr] = val
            if ymap:
                rel[metric] = ymap
        if rel:
            return rel

    # --- long orientation: first column of data rows are years ---
    first_col_years = [r[0].strip() for r in data_rows if r and _is_year(r[0])]
    if len(first_col_years) >= 2 and len(header) >= 2:
        metric_names = header[1:]
        rel = {}
        for drow in data_rows:
            if not drow or not _is_year(drow[0]):
                continue
            yr = drow[0].strip()
            for ci, mname in enumerate(metric_names, start=1):
                if ci < len(drow):
                    val = drow[ci].strip()
                    if val and _has_number(val):
                        key = mname.strip() or f"col{ci}"
                        rel.setdefault(key, {})[yr] = val
        if rel:
            return rel

    return None


def parse_tables(text: str) -> list[dict[str, dict[str, str]]]:
    """Extract every reconstructable ``{metric: {year: value}}`` table in ``text``.

    Only contiguous pipe-delimited blocks are parsed. Anything that cannot be
    reliably reconstructed yields nothing, so the caller falls back to lexical
    grounding rather than guessing relationships.
    """
    if not text or "|" not in text:
        return []
    tables: list[dict[str, dict[str, str]]] = []
    block: list[str] = []
    for line in text.splitlines() + [""]:
        if "|" in line:
            block.append(line)
        else:
            if block:
                rel = _parse_pipe_block(block)
                if rel:
                    tables.append(rel)
                block = []
    return tables


def evidence_year_values(evidence_text: str) -> dict[str, set[str]]:
    """Return ``{year: {canonical values...}}`` across all parsed tables."""
    yv: dict[str, set[str]] = {}
    for rel in parse_tables(evidence_text):
        for ymap in rel.values():
            for yr, val in ymap.items():
                num = _num(val)
                if num is not None:
                    yv.setdefault(yr, set()).add(num)
    return yv


def claim_year_value_pairs(claim_text: str) -> list[tuple[str, str | None]]:
    """Extract ``(value, year)`` pairs asserted by a claim.

    Each financial value is bound to the nearest year mentioned in the claim
    (by character position). Bare years and age-range endpoints are not treated
    as values. When the claim mentions no year, values are returned with year
    ``None`` (uncheckable against a table).
    """
    t = canonicalize_numbers(claim_text)
    # Blank out range endpoints while preserving character positions.
    t = _RANGE_RE.sub(lambda m: " " * len(m.group()), t)

    years = [(m.start(), m.group()) for m in _YEAR_RE.finditer(t)]

    pairs: list[tuple[str, str | None]] = []
    for m in _NUM_RE.finditer(t):
        s = m.group()
        if "." in s:
            is_value = True
        elif len(s) > 1 and not _is_year(s):
            is_value = True
        else:
            is_value = False
        if not is_value:
            continue
        pairs.append((s, _bind_year(m.start(), years)))
    return pairs


def _bind_year(pos: int, years: list[tuple[int, str]]) -> str | None:
    """Bind a value at ``pos`` to a year.

    Financial phrasing is overwhelmingly "VALUE in YEAR" (the year follows the
    value), e.g. "7,836.3 in 2022 and 8,347.9 in 2023". So the nearest year
    that appears AFTER the value is preferred; only if none follows is the
    nearest preceding year used. This avoids binding "8,347.9" to the earlier
    "2022" merely because it is a few characters closer.
    """
    if not years:
        return None
    after = [(p, y) for p, y in years if p >= pos]
    if after:
        return min(after, key=lambda py: py[0] - pos)[1]
    return min(years, key=lambda py: abs(py[0] - pos))[1]


def check_relationship(claim_text: str, evidence_text: str) -> dict:
    """Check the claim's year->value relationships against the evidence tables.

    Returns a dict::

        {"checked": bool, "consistent": bool, "reason": str}

    * ``checked`` is True only when the evidence yields a structured table AND
      the claim asserts a year that appears in that table (so a real comparison
      was possible).
    * ``consistent`` is False when a claimed year maps to a value the table does
      not associate with that year (the core wrong-relationship case).
    """
    yv = evidence_year_values(evidence_text)
    if not yv:
        return {"checked": False, "consistent": True, "reason": ""}

    checked = False
    for value, year in claim_year_value_pairs(claim_text):
        if year is None or year not in yv:
            continue
        checked = True
        expected = {v for v in yv[year] if v}
        if not any(values_equal(value, ev) for ev in expected):
            return {
                "checked": True,
                "consistent": False,
                "reason": (
                    f"claim maps {year} -> {value}, but the cited table maps "
                    f"{year} -> {sorted(expected)}"
                ),
            }

    if not checked:
        return {"checked": False, "consistent": True, "reason": ""}
    return {
        "checked": True,
        "consistent": True,
        "reason": "year-value relationship matches the cited table",
    }
