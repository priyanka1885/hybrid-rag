"""
Controlled query expansion / normalization for data-seeking retrieval.

    Original question
        -> recognize financial/sustainability CONCEPTS (scope 1, GHG, ...)
        -> append a SMALL, CURATED set of canonical retrieval terms
        -> normalize fiscal-year shorthand (FY2023) to the bare year (2023)
        => expanded RETRIEVAL query (used ONLY for dense + BM25 recall)

Why this exists
---------------
The exact-value rows that answer numerical questions are terse (e.g.
``GHG emissions CO 2 e (kilotons) 2023 2022 Direct scope 1 58 644 ...``). A
paraphrased question such as *"How much direct carbon output did Sasol produce
in FY2023?"* shares almost no lexical surface with that row: the row says
"CO 2 e", not "carbon"; "GHG emissions", not "carbon output"; and the query's
"FY2023" tokenizes to ``fy2023`` which never matches the row's ``2023`` token.
So BM25 ranks the value row far down and it falls out of the dense+BM25
candidate union entirely - a *recall* failure that no post-rerank boost can fix,
because a chunk that never enters the pool is never reranked.

Design principles (deliberately conservative)
---------------------------------------------
* CONTROLLED, not open-ended. Expansion terms come from a curated concept map of
  well-known emissions / sustainability / financial synonyms - never free
  semantic drift - so retrieval is not flooded with loosely related chunks.
* A concept's terms are only added when one of its trigger phrases is actually
  present in the question. A revenue question adds no emissions terms, and an
  emissions question adds no revenue terms.
* Terms already present in the question are not re-added, and the total number
  of added terms is capped, so the expansion stays small.
* The ORIGINAL question is always preserved verbatim and kept dominant; concept
  terms are appended after it.
* Expansion is applied ONLY to the dense + BM25 retrieval query to widen recall.
  The cross-encoder reranker and the LLM always receive the ORIGINAL question,
  so answer precision and grounding are unchanged.
* Numeric normalization is untouched: expansion only appends words / bare years
  before the existing ``canonicalize_numbers`` tokenizer runs, so exact figures
  (``58 644``, ``64,392`` ...) still collapse to their canonical tokens.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# Fiscal-year shorthand (FY23, FY 23, FY'23, FY2023) and plain 4-digit years.
_FY_RE = re.compile(r"\bfy\s?['`\u2019]?(\d{2}|\d{4})\b", re.IGNORECASE)
_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")

# Maximum number of concept terms appended to a single query. Keeps the
# expansion small so recall is widened without diluting the signal.
_MAX_ADDED_TERMS = 12

# Synonym-equivalence groups. If the question already contains ANY member of a
# group, the other members are treated as already present and NOT appended. This
# prevents adding a redundant lexical variant (e.g. "greenhouse gas" when the
# query already says "GHG") whose only effect would be to pull near-duplicate
# distractor chunks into the pool without improving recall for a query that is
# already anchored on the concept. Each group is a set of lowercase terms.
_EQUIVALENCE_GROUPS: tuple[frozenset[str], ...] = (
    frozenset({"ghg", "greenhouse gas", "greenhouse gases", "greenhouse"}),
    frozenset({"co2e", "co2", "carbon dioxide"}),
)


@dataclass(frozen=True)
class _Concept:
    """A curated concept: trigger phrases -> canonical retrieval terms."""

    name: str
    triggers: tuple[re.Pattern, ...]
    terms: tuple[str, ...]


def _c(name: str, triggers: list[str], terms: list[str]) -> _Concept:
    return _Concept(
        name=name,
        triggers=tuple(re.compile(t, re.IGNORECASE) for t in triggers),
        terms=tuple(terms),
    )


# --- The curated concept map ------------------------------------------------
# Each concept fires ONLY when a trigger phrase is present. Terms are the
# canonical vocabulary the underlying financial/sustainability tables actually
# use, so appending them makes the terse value row lexically (and semantically)
# reachable without introducing off-topic vocabulary.
_CONCEPTS: tuple[_Concept, ...] = (
    # Scope 1 = direct emissions. Paraphrases: "direct carbon output",
    # "direct emissions", "direct CO2".
    _c(
        "scope1",
        [
            r"scope\s*1\b",
            r"\bdirect\s+(?:carbon|co2|co\s*2|ghg|greenhouse|emission)",
            r"\bdirect\s+emission",
            r"\bcarbon\s+output\b",
        ],
        ["scope 1", "direct", "emissions", "greenhouse gas", "ghg", "co2e"],
    ),
    # Scope 2 = indirect emissions from purchased energy/electricity.
    _c(
        "scope2",
        [
            r"scope\s*2\b",
            r"\bindirect\s+(?:emission|energy|carbon)",
            r"\bpurchased\s+electricity\b",
        ],
        ["scope 2", "indirect", "emissions", "greenhouse gas", "ghg"],
    ),
    # Scope 3 = value-chain emissions.
    _c(
        "scope3",
        [r"scope\s*3\b", r"\bvalue\s+chain\s+emission"],
        ["scope 3", "emissions", "greenhouse gas", "ghg"],
    ),
    # Generic greenhouse-gas / carbon emissions (covers "carbon emissions",
    # "GHG emissions", "greenhouse gas", "carbon footprint", "CO2 / CO2e").
    _c(
        "ghg",
        [
            r"\bghg\b",
            r"greenhouse",
            r"\bcarbon\b",
            r"\bco2e?\b",
            r"\bco\s*2\b",
            r"\bemission",
            r"carbon\s+footprint",
            r"carbon\s+dioxide",
        ],
        ["emissions", "greenhouse gas", "ghg", "co2e", "carbon"],
    ),
    # Energy consumption metrics.
    _c(
        "energy",
        [r"\benergy\b", r"electricity", r"\bpower\b", r"\bfuel\b", r"consumption"],
        ["energy", "consumption", "electricity"],
    ),
    # Water metrics.
    _c(
        "water",
        [r"\bwater\b", r"\beffluent\b", r"\bwithdrawal\b"],
        ["water", "consumption"],
    ),
    # Waste metrics.
    _c(
        "waste",
        [r"\bwaste\b", r"\brecycl", r"\blandfill\b", r"hazardous"],
        ["waste", "recycled"],
    ),
    # Revenue / financial performance metrics.
    _c(
        "revenue",
        [r"\brevenue\b", r"\bturnover\b", r"\bsales\b", r"\bprofit\b", r"\bearnings\b", r"\bincome\b"],
        ["revenue", "turnover"],
    ),
)


@dataclass
class ExpandedQuery:
    """Result of query expansion, carrying full diagnostics."""

    original: str
    retrieval_text: str
    concepts: list[str] = field(default_factory=list)
    added_terms: list[str] = field(default_factory=list)
    normalized_years: list[str] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return self.retrieval_text.strip() != self.original.strip()

    def describe(self) -> dict:
        return {
            "original": self.original,
            "retrieval_text": self.retrieval_text,
            "concepts": self.concepts,
            "added_terms": self.added_terms,
            "normalized_years": self.normalized_years,
            "changed": self.changed,
        }


def _normalize_years(question: str) -> list[str]:
    """Return bare 4-digit year strings implied by the question.

    ``FY2023`` / ``FY23`` -> ``2023``. Plain years (``2023``) are also returned
    so the bare token is guaranteed present for BM25 (the FY-shorthand token
    ``fy2023`` otherwise never lexically matches a table's ``2023`` column
    header). Only the fiscal-year forms produce a NEW token worth appending;
    plain years are already in the question, so they are reported for
    transparency but skipped by the caller if already present.
    """
    years: list[str] = []
    for raw in _FY_RE.findall(question):
        if len(raw) == 2:
            years.append(str(2000 + int(raw)))
        elif len(raw) == 4:
            years.append(raw)
    for y in _YEAR_RE.findall(question):
        years.append(y)
    # Deduplicate, preserve order.
    seen: set[str] = set()
    out: list[str] = []
    for y in years:
        if y not in seen:
            seen.add(y)
            out.append(y)
    return out


def _phrase_in(term: str, haystack_lower: str) -> bool:
    """True when every word of ``term`` already appears in the question."""
    return all(
        re.search(rf"\b{re.escape(w)}\b", haystack_lower) for w in term.lower().split()
    )


def _already_present(term: str, haystack_lower: str) -> bool:
    """True when ``term`` (or an EQUIVALENT synonym) already appears in the query.

    A term is considered present if either its own words are all in the query,
    or the query already contains any member of a synonym-equivalence group the
    term belongs to (so "greenhouse gas" is skipped when the query says "GHG").
    """
    if _phrase_in(term, haystack_lower):
        return True
    key = term.lower()
    for group in _EQUIVALENCE_GROUPS:
        if key in group and any(_phrase_in(member, haystack_lower) for member in group):
            return True
    return False


def expand_query(question: str) -> ExpandedQuery:
    """Build a controlled, expanded RETRIEVAL query from ``question``.

    Returns an :class:`ExpandedQuery`. When no concept fires and no fiscal-year
    normalization is needed, ``retrieval_text`` equals the original question, so
    callers can pass it straight through with no behavioural change.
    """
    q = (question or "").strip()
    q_lower = q.lower()

    fired_concepts: list[str] = []
    added: list[str] = []
    seen_terms: set[str] = set()

    for concept in _CONCEPTS:
        if any(t.search(q_lower) for t in concept.triggers):
            fired_concepts.append(concept.name)
            for term in concept.terms:
                key = term.lower()
                if key in seen_terms:
                    continue
                if _already_present(term, q_lower):
                    continue
                seen_terms.add(key)
                added.append(term)

    # Fiscal-year normalization: append the bare year only when it is a NEW
    # token (i.e. the question used FY-shorthand, not the plain year already).
    all_years = _normalize_years(q)
    year_tokens_added: list[str] = []
    for y in all_years:
        if not re.search(rf"\b{y}\b", q):
            year_tokens_added.append(y)

    # Cap the number of appended concept terms (years are cheap, kept separate).
    if len(added) > _MAX_ADDED_TERMS:
        added = added[:_MAX_ADDED_TERMS]

    appended = added + year_tokens_added
    retrieval_text = f"{q} {' '.join(appended)}".strip() if appended else q

    return ExpandedQuery(
        original=q,
        retrieval_text=retrieval_text,
        concepts=fired_concepts,
        added_terms=added,
        normalized_years=year_tokens_added,
    )
