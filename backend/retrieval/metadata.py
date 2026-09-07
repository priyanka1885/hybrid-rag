"""
Metadata parsing and pre-filtering for retrieval.

    Question / explicit filter
        -> resolve company + year
        -> restrict the candidate chunks BEFORE dense/BM25 search

Why: the corpus mixes seven unrelated companies' ESG/sustainability reports.
Without a filter, a question about one company ("Sasol Scope 1 emissions FY23")
pulls chunks from every other entity (Distell, Absa, Clicks, ...), diluting the
top-K context window with cross-company noise. This module derives clean
company/year metadata for every document from its friendly title (no
re-ingestion needed) and detects the company/year a question is about so
retrieval can be scoped to the right report(s).

Everything here is pure string logic - no models, no I/O - so it is cheap and
deterministic, and it degrades safely: if nothing is detected, the filter is
empty and retrieval behaves exactly as before.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from backend.config import DOCUMENT_NAME_MAP

# A 4-digit year, 1900-2099.
_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
# Fiscal-year shorthand: FY23, FY 23, FY'23, FY2023.
_FY_RE = re.compile(r"\bfy\s?[''`]?(\d{2}|\d{4})\b", re.IGNORECASE)

# Tokens that describe the report *type*, not the company. Stripped when
# deriving a company name from a friendly document title.
_REPORT_STOPWORDS = {
    "esg", "sustainability", "report", "reports", "data", "sheet", "appendix",
    "spreads", "group", "limited", "ltd", "environmental", "social", "and",
    "governance", "climate", "change", "annual", "integrated", "the",
}

# Extra aliases -> canonical company name (lowercase). These make question
# detection robust to the short names people actually type (tickers, nicknames,
# partial names). Canonical values MUST match the names derived by
# _derive_company() from DOCUMENT_NAME_MAP.
_COMPANY_ALIASES: dict[str, str] = {
    "implats": "impala platinum",
    "impala": "impala platinum",
    "impala platinum": "impala platinum",
    "pnp": "pick n pay",
    "picknpay": "pick n pay",
    "pick n pay": "pick n pay",
    "pick 'n pay": "pick n pay",
    "tongaat": "tongaat hulett",
    "hulett": "tongaat hulett",
    "tongaat hulett": "tongaat hulett",
    "absa": "absa",
    "absa group": "absa",
    "sasol": "sasol",
    "clicks": "clicks",
    "distell": "distell",
}


@dataclass(frozen=True)
class DocumentMeta:
    """Structured metadata for one source document."""

    document_file: str
    document_name: str
    company: str  # canonical, lowercase (e.g. "sasol", "pick n pay")
    company_display: str  # human-readable (e.g. "Sasol")
    year: int | None


def _extract_year(text: str) -> int | None:
    m = _YEAR_RE.search(text)
    return int(m.group(0)) if m else None


def _derive_company(document_name: str) -> str:
    """Strip the year and report-type words to leave the company name."""
    name = _YEAR_RE.sub(" ", document_name)
    tokens = [t for t in re.split(r"[\s\-_]+", name) if t]
    kept = [t for t in tokens if t.lower() not in _REPORT_STOPWORDS]
    company = " ".join(kept).strip()
    return company or document_name.strip()


def build_document_meta(document_file: str, document_name: str | None = None) -> DocumentMeta:
    """Derive :class:`DocumentMeta` for a document from its file/friendly name."""
    friendly = document_name or DOCUMENT_NAME_MAP.get(document_file) or Path(document_file).stem
    display = _derive_company(friendly)
    return DocumentMeta(
        document_file=document_file,
        document_name=friendly,
        company=display.lower(),
        company_display=display,
        year=_extract_year(friendly),
    )


# --- Registry built once from the known document map ------------------------
DOCUMENT_REGISTRY: dict[str, DocumentMeta] = {
    file_name: build_document_meta(file_name, friendly)
    for file_name, friendly in DOCUMENT_NAME_MAP.items()
}

# Canonical company -> display name, and the set of years present in the corpus.
KNOWN_COMPANIES: dict[str, str] = {m.company: m.company_display for m in DOCUMENT_REGISTRY.values()}
KNOWN_YEARS: set[int] = {m.year for m in DOCUMENT_REGISTRY.values() if m.year is not None}

# Full alias table used for detection: every canonical company is its own alias,
# plus the curated shortcuts above.
_ALIAS_TABLE: dict[str, str] = {c: c for c in KNOWN_COMPANIES}
_ALIAS_TABLE.update(_COMPANY_ALIASES)
# Longest aliases first so multi-word names ("impala platinum") win over a
# substring ("impala") when both are present.
_ALIASES_SORTED = sorted(_ALIAS_TABLE.items(), key=lambda kv: len(kv[0]), reverse=True)


def resolve_company(name: str | None) -> str | None:
    """Map a free-text company name to its canonical form, if recognized."""
    if not name:
        return None
    key = re.sub(r"\s+", " ", name.strip().lower())
    if key in _ALIAS_TABLE:
        return _ALIAS_TABLE[key]
    # Fall back to a contained-alias match (e.g. "the sasol company").
    for alias, canonical in _ALIASES_SORTED:
        if re.search(rf"\b{re.escape(alias)}\b", key):
            return canonical
    return key or None


def _normalize_fy(raw: str) -> int | None:
    """Turn an FY capture ('23' or '2023') into a full year."""
    if len(raw) == 4:
        return int(raw)
    if len(raw) == 2:
        return 2000 + int(raw)
    return None


@dataclass
class MetadataFilter:
    """A restriction on which documents' chunks are eligible for retrieval."""

    companies: set[str] = field(default_factory=set)
    years: set[int] = field(default_factory=set)
    document_files: set[str] = field(default_factory=set)

    def is_empty(self) -> bool:
        return not (self.companies or self.years or self.document_files)

    def matches(self, meta: DocumentMeta) -> bool:
        if self.document_files and meta.document_file not in self.document_files:
            return False
        if self.companies and meta.company not in self.companies:
            return False
        if self.years and meta.year not in self.years:
            return False
        return True

    def describe(self) -> dict:
        """A JSON-friendly summary for the API/retrieval trace."""
        return {
            "companies": sorted(KNOWN_COMPANIES.get(c, c) for c in self.companies),
            "years": sorted(self.years),
            "document_files": sorted(self.document_files),
            "active": not self.is_empty(),
        }


def detect_companies(question: str) -> set[str]:
    """Find canonical company names mentioned in a question."""
    q = re.sub(r"\s+", " ", (question or "").lower())
    found: set[str] = set()
    for alias, canonical in _ALIASES_SORTED:
        if re.search(rf"\b{re.escape(alias)}\b", q):
            found.add(canonical)
    return found


def detect_years(question: str, restrict_to_known: bool = True) -> set[int]:
    """Find years (including FYxx shorthand) mentioned in a question.

    Detected years are intersected with the years that actually exist in the
    corpus (``restrict_to_known``) so a stray year the corpus does not contain
    never filters the candidate set down to nothing.
    """
    q = question or ""
    years: set[int] = {int(y) for y in _YEAR_RE.findall(q)}
    for raw in _FY_RE.findall(q):
        y = _normalize_fy(raw)
        if y is not None:
            years.add(y)
    if restrict_to_known:
        years &= KNOWN_YEARS
    return years


def detect_filter(question: str) -> MetadataFilter:
    """Build a :class:`MetadataFilter` from the companies/years in a question."""
    return MetadataFilter(
        companies=detect_companies(question),
        years=detect_years(question),
    )


def build_explicit_filter(
    company: str | None = None,
    year: int | None = None,
    document_file: str | None = None,
) -> MetadataFilter:
    """Build a filter from explicit API parameters (any subset)."""
    companies: set[str] = set()
    if company:
        resolved = resolve_company(company)
        if resolved:
            companies.add(resolved)
    years: set[int] = {int(year)} if year else set()
    files: set[str] = {document_file} if document_file else set()
    return MetadataFilter(companies=companies, years=years, document_files=files)
