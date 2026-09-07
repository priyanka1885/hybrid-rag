"""
Citation verification layer.

    Generated Answer -> claims + citations -> compare with cited evidence -> status

Purpose: a citation should not merely exist; the cited evidence should actually
support the claim. This is a practical, lightweight lexical-overlap check (not
a heavyweight NLI research system):

For each citation we measure how much of the answer's informative content is
covered by the cited supporting text, giving special weight to numeric tokens
(financial figures). We return one of:

    SUPPORTED | PARTIALLY_SUPPORTED | NOT_SUPPORTED

both per-citation and as an overall verdict.
"""
from __future__ import annotations

import re

from backend.citations.table_relations import check_relationship
from backend.config import settings
from backend.text_normalize import (
    canonicalize_numbers,
    join_comma_thousands,
    strip_id_tokens,
    strip_retrieval_metadata,
)

_WORD_RE = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")
_NUM_RE = re.compile(r"\d+(?:\.\d+)?")
# A space-grouped thousands sequence (e.g. "58 644", "1 234 567"). Used ONLY for
# verification matching: canonicalize_numbers() intentionally leaves such groups
# unmerged when they sit inside a run of values (e.g. a table row
# "58 644 57 284"), to protect display/index tokens. For MATCHING a claim's
# "58,644" against that row we additionally recognize each grouped sequence as
# its joined value. The pattern segments "58 644 57 284" into "58644" and
# "57284" (each value is <=3-digit lead + 3-digit groups); a genuine run of
# independent 3-digit cells ("643 635 371 417") joins into one long number and
# so cannot spuriously match a short claimed figure.
_SPACE_GROUP_RE = re.compile(r"\d{1,3}(?:[ \u00a0]\d{3})+")

# A label RANGE such as "40-49", "40 - 49" or "20 – 29" (age-band table labels).
# These endpoints are category labels, not financial figures, so they are
# removed before figure extraction. Uses common hyphen/dash variants.
_RANGE_RE = re.compile(r"\b\d{1,3}\s*[-\u2010-\u2015\u2212]\s*\d{1,3}\b")


def _strip_ranges(text: str) -> str:
    """Remove label ranges like ``40-49`` so their endpoints aren't treated as
    financial figures during numeric grounding."""
    return _RANGE_RE.sub(" ", text)

# Common English stopwords + citation-marker noise, so overlap focuses on
# informative terms.
_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with", "was",
    "were", "is", "are", "be", "been", "as", "at", "by", "it", "its", "that",
    "this", "from", "during", "period", "company", "group", "report", "reported",
    "which", "their", "there", "have", "has", "had", "we", "our", "they", "these",
    "than", "then", "also", "into", "over", "per", "about", "not", "no",
}


def _tokens(text: str) -> list[str]:
    return [t for t in _WORD_RE.findall(text.lower()) if t not in _STOPWORDS]


def _numbers(text: str) -> set[str]:
    # Financial reports render the same value in several ways:
    #   - European / South-African decimals with a comma: "63,4", "130,07"
    #   - Space-grouped thousands from OCR of tables: "64 392", "1 234 567"
    #   - US comma-grouped thousands: "64,392", "1,234,567"
    # canonicalize_numbers() folds all of these to one comparable form (e.g.
    # "64 392", "64,392" and "64392" all become "64392"; "63,4" becomes "63.4").
    # Without this, "64 392" parses as two integers (64 and 392) and a correctly
    # grounded figure fails numeric matching, wrongly rejecting a supported claim
    # as NOT_SUPPORTED. Genuinely different values (99900 vs 64392) still differ.
    # Drop alphanumeric identifiers/hashes first (e.g. a leaked chunk id
    # "620fb5572c74334") so the digit runs inside them are never mistaken for
    # financial values. Pure figures ("58 644") and short currency-prefixed
    # values ("R1234") are untouched.
    text = strip_id_tokens(text)
    normalized = canonicalize_numbers(text)
    nums = set(_NUM_RE.findall(normalized))
    # Also register each space-grouped thousands sequence as its joined value,
    # so "58 644" in a multi-value table row matches a claimed "58,644"/"58644"
    # even though canonicalize_numbers() left the row's groups unmerged.
    for m in _SPACE_GROUP_RE.finditer(text):
        nums.add(m.group(0).replace(" ", "").replace("\u00a0", ""))
    return nums


def _financial_figures(text: str) -> set[str]:
    """Extract only meaningful financial figures for numeric grounding.

    Numeric grounding must compare *figures*, not incidental single digits that
    come from chemical notation ("CO2" -> 2) or ordinal labels ("Scope 1",
    "Scope 2" -> 1, 2). Those bare single digits rarely appear in the terse
    supporting evidence, so requiring them to match wrongly rejects correctly
    grounded claims (e.g. an answer that cites "63.4 million tons CO2, Scope 1
    and 2") and forces a false "insufficient evidence" refusal.

    We therefore keep only decimals (e.g. 63.4) and multi-digit integers
    (e.g. 870, 2022) - the forms actual financial figures take - and drop bare
    single digits. This does not weaken hallucination detection: a mismatched
    real figure (99.9 vs 38.9) is a decimal and is still fully checked.

    We additionally drop bare 4-digit year values (1900-2099). In table-based
    evidence the reporting year is often only a column header and ends up in a
    different chunk than the data row, so the year token is absent from the
    cited supporting text. Requiring it to match wrongly rejects correctly
    grounded claims that mention the year for context (e.g. "64,392 kilotons
    CO2e in 2023"). Genuine financial figures - decimals and non-year
    multi-digit integers - are still fully checked, so a mismatched real value
    (99,999 vs 64,392) is still rejected.

    Finally, we drop the endpoints of a label RANGE such as "40-49 years" or
    "40 - 49" (the age-band labels in demographic tables). These are category
    labels, not financial figures, and the band endpoints live in a different
    chunk (the row header) than the actual data value. Requiring "40" and "49"
    to match wrongly rejected correctly grounded answers whose real value was a
    small single-digit count. The genuine answer figure (a decimal or non-range
    multi-digit number) is still fully checked.
    """
    figures: set[str] = set()
    for n in _numbers(_strip_ranges(text)):
        if "." in n:
            figures.add(n)
        elif len(n) > 1:
            if len(n) == 4 and 1900 <= int(n) <= 2099:
                # Bare 4-digit year used for context; not a financial figure.
                continue
            figures.add(n)
    return figures


def _strip_citation_markers(text: str) -> str:
    return re.sub(r"\[\d+\]", " ", text)


# Verbose "connective" words that carry no metric/entity meaning. A factually
# correct financial claim is often phrased verbosely ("according to the report,
# ... amounted to approximately ... during the ... financial year, compared with
# ... recorded in the previous year"). Those words never appear in a terse table
# row, so they must NOT be required for semantic grounding. They are removed
# (in addition to the stopwords above) when isolating the claim's meaningful
# metric/entity terms. This list is generic - no company or metric is hardcoded.
_FILLER = {
    "according", "amounted", "approximately", "roughly", "recorded", "compared",
    "previous", "year", "years", "overall", "indicating", "measured", "reflects",
    "reflecting", "respectively", "versus", "vs", "around", "approx", "circa",
    "financial", "figure", "figures", "value", "values", "totaling", "totalling",
    "stood", "stands", "came", "reaching", "reached", "roughly", "some",
    "meaningfully", "slight", "slightly", "increase", "decrease",
}


def _metric_terms(text: str) -> set[str]:
    """Isolate a claim's meaningful metric/entity terms.

    Starts from the informative tokens (stopwords already removed) and further
    drops verbose connective FILLER words, pure numbers, and single characters,
    leaving the terms that actually name the metric/entity (e.g. ``greenhouse``,
    ``gas``, ``emissions``, ``revenue``, ``kilotons``). Used for semantic
    grounding, so a verbose-but-correct claim is judged on its metric words, not
    on filler that a terse table row never contains.
    """
    terms: set[str] = set()
    for t in _tokens(text):
        if t in _FILLER or len(t) == 1:
            continue
        if _NUM_RE.fullmatch(t):  # a pure number is not a metric term
            continue
        terms.add(t)
    return terms


def _stream_tokens(text: str) -> list[str]:
    """Ordered lowercase token stream with grouped thousands joined per value.

    Uses :data:`_SPACE_GROUP_RE` (which segments a table row such as
    ``64 392 63 891`` into the individual values ``64392`` and ``63891``, exactly
    as numeric grounding does) plus comma-thousands joining, so a claimed
    canonical figure like ``64392`` appears as a single token and can be located
    positionally within the evidence.
    """
    joined = _SPACE_GROUP_RE.sub(
        lambda m: m.group(0).replace(" ", "").replace("\u00a0", ""), text
    )
    joined = join_comma_thousands(joined)
    return _WORD_RE.findall(joined.lower())


def _label_terms_for_number(supporting_text: str, figure: str, window: int = 8) -> tuple[bool, set[str]]:
    """Return ``(found, label_terms)`` for a claimed ``figure`` in the evidence.

    The *label* of a value in a flattened table row is the metric name that
    immediately precedes its run of numbers, e.g. in
    ``Total greenhouse gas (CO2 equivalent) (kilotons) 64 392 63 891`` the label
    of ``64392`` and ``63891`` is ``{total, greenhouse, gas, co2, equivalent,
    kilotons}``. For each occurrence of ``figure`` we skip the contiguous run of
    numeric cells it belongs to, then collect up to ``window`` preceding
    meaningful words, stopping at the next number (the previous value's
    boundary) so a neighbouring metric's label is never absorbed. ``found`` is
    False when the figure does not occur as a discrete token.
    """
    toks = _stream_tokens(supporting_text)
    found = False
    labels: set[str] = set()
    for i, tok in enumerate(toks):
        if tok != figure:
            continue
        found = True
        j = i - 1
        # Skip the numeric run (sibling cells) immediately left of the figure.
        while j >= 0 and _NUM_RE.fullmatch(toks[j]):
            j -= 1
        # Collect the preceding label words; stop at the previous value's number.
        count = 0
        while j >= 0 and count < window:
            tj = toks[j]
            if _NUM_RE.fullmatch(tj):
                break
            if tj not in _STOPWORDS and tj not in _FILLER and len(tj) > 1:
                labels.add(tj)
                count += 1
            j -= 1
    return found, labels


def _numeric_semantic_grounding(
    claim_clean: str, supporting_text: str, claim_numbers: set[str]
) -> dict:
    """Assess whether each claimed figure is tied to the claim's own metric.

    For every meaningful figure in the claim we locate it in the evidence and
    read the metric label that precedes it (:func:`_label_terms_for_number`). A
    figure is:

    * **grounded**   - its evidence label shares a metric term with the claim;
    * **mismatched** - it is present but under a *different* metric label
      (e.g. the claim asks for Scope 1 but the value sits under Total GHG);
    * **unknown**    - it cannot be located as a discrete token.

    Returns ``matched_all`` (every figure grounded to the claimed metric, with
    the claim actually naming a metric) and ``any_mismatch`` (at least one
    figure tied to a *different* metric). Nothing is hardcoded; grounding is by
    positional metric-label overlap, so it generalizes across reports.
    """
    metric_terms = _metric_terms(claim_clean)
    if not claim_numbers or not metric_terms:
        return {"matched_all": False, "any_mismatch": False}

    matched_all = True
    any_mismatch = False
    for fig in claim_numbers:
        found, labels = _label_terms_for_number(supporting_text, fig)
        if not found or not labels:
            matched_all = False  # cannot confirm association
            continue
        if labels & metric_terms:
            continue  # figure is grounded to the claimed metric
        matched_all = False
        any_mismatch = True  # figure belongs to a different metric label
    return {"matched_all": matched_all, "any_mismatch": any_mismatch}


def _has_strong_numeric_semantic_grounding(
    claim_clean: str, supporting_text: str, claim_numbers: set[str], numbers_matched: bool
) -> bool:
    """True when a data claim's figures all match AND are each tied to the
    claimed metric - strong enough to SUPPORT without high lexical coverage."""
    if not claim_numbers or not numbers_matched:
        return False
    return _numeric_semantic_grounding(claim_clean, supporting_text, claim_numbers)["matched_all"]


def verify_claim(claim_text: str, supporting_text: str) -> dict:
    """Score how well ``supporting_text`` supports ``claim_text``.

    Returns a dict with status, coverage ratio, numeric-match info, and a short
    ``reason``. Verification proceeds in priority order:

    1. **Relationship contradiction (structured tables).** When the cited
       evidence reconstructs into a table AND the claim asserts a year the table
       covers, a wrong ``year -> value`` mapping (e.g. "8,347.9 in 2022" when the
       table says 2022 -> 7,836.3) is NOT_SUPPORTED even though both tokens
       appear. Contradictions always win.

    2. **Relationship consistent.** If that same structured check confirms the
       claim's year-value pairs, the claim is SUPPORTED.

    3. **Numeric + semantic grounding (flattened evidence).** When the table
       could NOT be reconstructed (``rel["checked"] is False`` - unavailable,
       not suspicious), a data claim is SUPPORTED if every meaningful claimed
       figure appears in the evidence AND each is tied to the claim's own metric
       label (so a verbose-but-correct claim is not penalised for filler words,
       while a right-number/wrong-metric claim is not rescued).

    4. **Lexical + numeric fallback (original behavior).** Otherwise, informative
       token overlap plus the requirement that any claimed figure appears in the
       evidence. A figure that is present but tied to a *different* metric caps
       the verdict at PARTIALLY_SUPPORTED rather than SUPPORTED.
    """
    # Strip citation markers AND any internal retrieval metadata the model may
    # have echoed ("Page: 41 | Chunk ID: 620fb5572c...") so metadata numbers are
    # never scored as claimed financial figures.
    claim_clean = strip_retrieval_metadata(_strip_citation_markers(claim_text))
    claim_tokens = set(_tokens(claim_clean))
    support_tokens = set(_tokens(supporting_text))

    if not claim_tokens:
        return {
            "status": "NOT_SUPPORTED",
            "coverage": 0.0,
            "numbers_matched": True,
            "reason": "empty claim",
        }

    overlap = claim_tokens & support_tokens
    coverage = len(overlap) / len(claim_tokens)

    # Numeric grounding: any financial figures in the claim must appear in the
    # supporting evidence. A mismatched number is a strong hallucination signal.
    # We compare only real figures (decimals / multi-digit numbers), not bare
    # single digits from chemical notation ("CO2") or scope labels ("Scope 1"),
    # which would otherwise cause valid grounded claims to be rejected.
    # Citation markers ([1]) are stripped first so they don't count as figures.
    claim_numbers = _financial_figures(claim_clean)
    support_numbers = _financial_figures(supporting_text)
    if claim_numbers:
        numbers_matched = claim_numbers.issubset(support_numbers)
    else:
        numbers_matched = True

    # Stage 1: structured year -> value relationship check. This takes priority
    # because it verifies the actual claimed relationship, which token/number
    # presence alone cannot. A detected contradiction always wins and can never
    # be overridden by the numeric/semantic or lexical paths below.
    rel = check_relationship(claim_clean, supporting_text)
    if rel["checked"] and not rel["consistent"]:
        return {
            "status": "NOT_SUPPORTED",
            "coverage": round(coverage, 3),
            "numbers_matched": False,
            "reason": rel["reason"],
        }
    # Stage 2: the same structured check confirmed the claim's year-value pairs.
    if rel["checked"] and rel["consistent"]:
        return {
            "status": "SUPPORTED",
            "coverage": round(coverage, 3),
            "numbers_matched": True,
            "reason": rel["reason"],
        }

    # Relationship UNAVAILABLE (rel["checked"] is False): the evidence could not
    # be reconstructed into a table. This is not a contradiction, so a strong
    # numeric + semantic match may still support the claim.
    grounding = _numeric_semantic_grounding(claim_clean, supporting_text, claim_numbers)

    # Stage 3: strong numeric + semantic grounding. Every meaningful claimed
    # figure is present in the evidence AND each is tied to the claim's own
    # metric label. This supports a verbose-but-correct financial claim without
    # requiring high lexical coverage, while still refusing a right-number /
    # wrong-metric claim (handled as a mismatch below).
    if claim_numbers and numbers_matched and grounding["matched_all"]:
        return {
            "status": "SUPPORTED",
            "coverage": round(coverage, 3),
            "numbers_matched": True,
            "reason": "numeric and semantic grounding",
        }

    # Stage 4: lexical + numeric fallback (original behavior), with a
    # wrong-metric guard.
    if coverage >= settings.VERIFY_SUPPORTED_THRESHOLD and numbers_matched:
        if grounding["any_mismatch"]:
            # The claimed figure exists but under a DIFFERENT metric label, so
            # incidental lexical overlap must not promote it to SUPPORTED.
            status = "PARTIALLY_SUPPORTED"
            reason = "figure present but tied to a different metric in the evidence"
        else:
            status = "SUPPORTED"
            reason = "lexical and numeric grounding"
    elif coverage >= settings.VERIFY_PARTIAL_THRESHOLD:
        if numbers_matched:
            status = "PARTIALLY_SUPPORTED"
            reason = "partial lexical overlap"
        else:
            status = "NOT_SUPPORTED"
            reason = "claimed figure not found in cited evidence"
    else:
        status = "NOT_SUPPORTED"
        reason = "insufficient overlap with cited evidence"

    return {
        "status": status,
        "coverage": round(coverage, 3),
        "numbers_matched": numbers_matched,
        "reason": reason,
    }


_STATUS_RANK = {"SUPPORTED": 2, "PARTIALLY_SUPPORTED": 1, "NOT_SUPPORTED": 0}
_RANK_STATUS = {v: k for k, v in _STATUS_RANK.items()}


def _better(a: str, b: str) -> str:
    """Return the stronger of two statuses."""
    return a if _STATUS_RANK[a] >= _STATUS_RANK[b] else b


def _combined_support_text(ids: list[int], cmap: dict[int, dict]) -> str:
    """Concatenate the supporting text of several co-cited chunks.

    Used for UNION grounding of a multi-citation claim (see
    :func:`_verify_claim_union`).
    """
    return "\n\n".join(cmap[i]["supporting_text"] for i in ids if i in cmap)


def _verify_claim_union(claim_text: str, ids: list[int], cmap: dict[int, dict]) -> dict:
    """Verify a claim against the UNION of the chunks it actually cites.

    A single sentence may legitimately draw figures from several *co-cited*
    chunks - e.g. a cross-company comparison whose Sasol figure lives in one
    cited chunk and whose Impala figure lives in another. Verifying such a
    sentence against each chunk *individually* wrongly rejects it, because no
    single chunk contains every figure. Grounding against the union treats a
    figure as supported when it appears in ANY co-cited chunk, while still
    rejecting a figure that appears in NONE of them - so strictness is
    preserved and a hallucinated value cannot pass. This is a plain reuse of
    :func:`verify_claim` over the concatenated cited text; no thresholds change.
    """
    return verify_claim(claim_text, _combined_support_text(ids, cmap))


def _contributing_ids(claim_text: str, ids: list[int], cmap: dict[int, dict]) -> list[int]:
    """Of the co-cited ``ids``, which chunks actually contribute to the claim.

    A chunk contributes when it supplies at least one of the claim's financial
    figures (the common comparison case), or - for a figure-less claim - when it
    individually shows lexical support. Non-contributing co-cited markers are
    therefore stripped rather than silently kept, preserving the existing
    "a wrong citation is never kept as valid" behaviour. Falls back to all cited
    ids only if attribution is inconclusive.
    """
    claim_figs = _financial_figures(_strip_citation_markers(claim_text))
    contributors: list[int] = []
    for i in ids:
        c = cmap.get(i)
        if not c:
            continue
        if claim_figs:
            if _financial_figures(c["supporting_text"]) & claim_figs:
                contributors.append(i)
        elif verify_claim(claim_text, c["supporting_text"])["status"] != "NOT_SUPPORTED":
            contributors.append(i)
    return contributors or [i for i in ids if i in cmap]


def _claim_grounding(
    claim_text: str, evidence: list[dict] | None
) -> tuple[bool, int | None]:
    """Report whether *some* retrieved chunk supports the claim (grounding).

    This is tracked SEPARATELY from citation support: it never upgrades the
    status of a wrongly-cited chunk. It only records that the claim is grounded
    somewhere in the retrieved evidence, and which chunk (1-based) supports it,
    so a wrong ``[n]`` marker can be reported without being silently rewritten.
    """
    if not evidence:
        return False, None
    for idx, ev in enumerate(evidence, start=1):
        res = verify_claim(claim_text, ev.get("text", ""))
        if res["status"] != "NOT_SUPPORTED":
            return True, idx
    return False, None


def _aggregate(statuses: list[str]) -> str:
    """Conservative aggregation over per-claim statuses.

    * any NOT_SUPPORTED         -> NOT_SUPPORTED
    * all SUPPORTED             -> SUPPORTED
    * otherwise (mix, no fail)  -> PARTIALLY_SUPPORTED

    A supported claim can never hide an unsupported one.
    """
    if not statuses:
        return "NOT_SUPPORTED"
    if any(s == "NOT_SUPPORTED" for s in statuses):
        return "NOT_SUPPORTED"
    if all(s == "SUPPORTED" for s in statuses):
        return "SUPPORTED"
    return "PARTIALLY_SUPPORTED"


def verify_answer(
    answer: str, citations: list[dict], evidence: list[dict] | None = None
) -> dict:
    """Verify an answer claim-by-claim and compute a conservative verdict.

    Each sentence is a claim. A claim's citations are verified against ONLY the
    evidence they actually point at (no silent fallback to other chunks), so a
    wrong ``[n]`` marker is reported as NOT_SUPPORTED for that citation rather
    than being rescued by a different chunk that happens to contain the value.

    The overall verdict is the conservative aggregation across every claim, so a
    single unsupported factual claim forces the whole answer to NOT_SUPPORTED.
    """
    if not citations:
        return {
            "overall_status": "NOT_SUPPORTED",
            "summary": "No citations were produced for this answer.",
            "per_citation": [],
        }

    cmap = {c["citation_id"]: c for c in citations}
    # Best per-citation verification result across the claims that cite it.
    per_cit: dict[int, dict] = {}
    claim_statuses: list[str] = []

    for sentence in _split_sentences(answer):
        ids = _sentence_citation_ids(sentence)
        if not ids:
            continue
        claim_best = "NOT_SUPPORTED"
        for i in ids:
            c = cmap.get(i)
            if not c:
                continue
            res = verify_claim(sentence, c["supporting_text"])
            prev = per_cit.get(i)
            if prev is None or _STATUS_RANK[res["status"]] > _STATUS_RANK[prev["status"]]:
                per_cit[i] = res
            claim_best = _better(claim_best, res["status"])

        # Union fallback for multi-citation claims: if no single cited chunk
        # supports the sentence but it cites several chunks, verify against their
        # union so a legitimate comparison (one figure per cited chunk) is
        # recognized. Reflect the joint support on the contributing citations so
        # the per-citation view is consistent with the overall verdict.
        valid_ids = [i for i in ids if i in cmap]
        if claim_best == "NOT_SUPPORTED" and len(valid_ids) > 1:
            union_res = _verify_claim_union(sentence, valid_ids, cmap)
            if union_res["status"] != "NOT_SUPPORTED":
                claim_best = union_res["status"]
                for i in _contributing_ids(sentence, valid_ids, cmap):
                    prev = per_cit.get(i)
                    if prev is None or _STATUS_RANK[union_res["status"]] > _STATUS_RANK[prev["status"]]:
                        per_cit[i] = union_res
        claim_statuses.append(claim_best)

    # Citations not referenced by any parsed sentence: verify against the whole
    # answer (still cited-only) so every citation gets a status.
    for c in citations:
        if c["citation_id"] not in per_cit:
            per_cit[c["citation_id"]] = verify_claim(answer, c["supporting_text"])

    # If no citation-bearing sentence was found, fall back to per-citation
    # statuses for aggregation.
    if not claim_statuses:
        claim_statuses = [per_cit[c["citation_id"]]["status"] for c in citations]

    overall = _aggregate(claim_statuses)
    summary = {
        "SUPPORTED": "Every factual claim is supported by its cited evidence.",
        "PARTIALLY_SUPPORTED": "Some claims are only partially supported by their cited evidence.",
        "NOT_SUPPORTED": "At least one factual claim is not supported by its cited evidence.",
    }[overall]

    per_citation = []
    for c in citations:
        res = per_cit[c["citation_id"]]
        per_citation.append(
            {
                "citation_id": c["citation_id"],
                "document_name": c["document_name"],
                "page_number": c["page_number"],
                "status": res["status"],
                "coverage": res["coverage"],
                "numbers_matched": res["numbers_matched"],
                "reason": res.get("reason", ""),
            }
        )

    return {
        "overall_status": overall,
        "summary": summary,
        "per_citation": per_citation,
    }


# Sentence splitter for claim-level enforcement. Splits on sentence-ending
# punctuation followed by whitespace. Decimal figures like "63.4" are safe
# because the punctuation must be followed by whitespace to trigger a split.
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def has_citation_markers(text: str) -> bool:
    """True if the text contains at least one [n] citation marker."""
    return bool(re.search(r"\[\d+\]", text))


_MARKERS_ONLY_RE = re.compile(r"^(?:\[\d+\]\s*)+$")


def _split_sentences(text: str) -> list[str]:
    """Split into sentence-level claims, keeping citation markers with claims.

    The LLM very often places the citation AFTER the sentence-ending period
    ("Revenue was R10bn in 2022. [1]"). A naive split on ".\\s+" then produces a
    separate "[1]" fragment, orphaning the marker so the real claim looks
    uncited and gets wrongly dropped - a common cause of a correct, grounded
    answer collapsing to the insufficient-evidence fallback. We reattach any
    marker-only fragment to the preceding claim so the citation stays with the
    statement it supports.
    """
    parts = [s.strip() for s in _SENT_SPLIT_RE.split(text.strip()) if s.strip()]
    merged: list[str] = []
    for p in parts:
        if merged and _MARKERS_ONLY_RE.match(p):
            merged[-1] = f"{merged[-1]} {p}".strip()
        else:
            merged.append(p)
    return merged


def _sentence_citation_ids(sentence: str) -> list[int]:
    seen: list[int] = []
    for m in re.finditer(r"\[(\d+)\]", sentence):
        n = int(m.group(1))
        if n not in seen:
            seen.append(n)
    return seen


def _keep_only_markers(sentence: str, keep_ids: list[int]) -> str:
    """Remove ``[n]`` markers whose id is not in ``keep_ids``.

    Preserves citation markers only for citations that actually support the
    kept claim, and tidies the leftover whitespace.
    """
    keep = set(keep_ids)

    def _repl(m: re.Match) -> str:
        return m.group(0) if int(m.group(1)) in keep else ""

    cleaned = re.sub(r"\[(\d+)\]", _repl, sentence)
    cleaned = re.sub(r"\s+([.!?,;:])", r"\1", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return cleaned.strip()


def _remap_marker(sentence: str, new_id: int) -> str:
    """Strip all citation markers from ``sentence`` and cite ``new_id`` instead.

    Used for citation RECOVERY: when the marker the LLM attached is wrong but a
    different retrieved chunk genuinely supports the claim, we re-point the
    marker to the correct evidence rather than keeping the wrong one or silently
    approving it.
    """
    stripped = re.sub(r"\s*\[\d+\]", "", sentence)
    stripped = re.sub(r"\s+([.!?,;:])", r"\1", stripped).strip()
    m = re.search(r"([.!?]+)$", stripped)
    if m:
        return f"{stripped[:m.start()].rstrip()} [{new_id}]{m.group(1)}"
    return f"{stripped} [{new_id}]"


def filter_supported_claims(
    answer: str, citations: list[dict], evidence: list[dict] | None = None
) -> dict:
    """Keep only claims that a cited OR retrieved evidence source supports.

    Each sentence in ``answer`` is treated as an independent claim. Citation
    support and claim grounding are tracked SEPARATELY:

    * A sentence is kept if at least one of the citations it *actually cites*
      supports it (SUPPORTED or PARTIALLY_SUPPORTED). Non-supporting markers are
      stripped so a wrong citation is never silently kept as valid.
    * RECOVERY (layer 4): if none of the cited markers support the claim but a
      DIFFERENT chunk in the retrieved ``evidence`` set genuinely supports it,
      the claim is retained and its marker is RE-POINTED to that chunk. This is
      not silent approval of the wrong citation - the marker is corrected to the
      evidence that actually supports the claim, and only evidence that is part
      of the retrieved set is ever used.
    * If nothing (cited or retrieved) supports the claim, it is dropped.

    The year->value relationship check inside :func:`verify_claim` means a
    sentence asserting several year-value pairs is only kept when every asserted
    pair matches the evidence; a single wrong pair drops the whole sentence, and
    a wrong year-value claim cannot be "recovered" because it matches no chunk.

    Returns a dict with:
        answer:   the filtered answer text (may be empty if nothing survived)
        kept_ids: citation ids still referenced by the kept claims
        dropped:  list of {claim, reason, ...} for the removed claims
    """
    cmap = {c["citation_id"]: c for c in citations}
    kept_sentences: list[str] = []
    kept_ids: list[int] = []
    dropped: list[dict] = []

    for sentence in _split_sentences(answer):
        ids = _sentence_citation_ids(sentence)
        if not ids:
            # An uncited sentence cannot be grounded in evidence.
            dropped.append({"claim": sentence, "reason": "uncited"})
            continue

        # Verify against each CITED chunk independently (no cross-chunk rescue
        # for the marker the model chose).
        supporting_ids: list[int] = []
        claim_best = "NOT_SUPPORTED"
        for i in ids:
            c = cmap.get(i)
            if not c:
                continue
            res = verify_claim(sentence, c["supporting_text"])
            if res["status"] != "NOT_SUPPORTED":
                supporting_ids.append(i)
            claim_best = _better(claim_best, res["status"])

        if supporting_ids:
            kept_sentences.append(_keep_only_markers(sentence, supporting_ids))
            for i in supporting_ids:
                if i not in kept_ids:
                    kept_ids.append(i)
            continue

        # Union grounding for multi-citation claims: a single sentence may
        # legitimately draw one figure from each of several co-cited chunks
        # (e.g. a cross-company comparison). When no chunk supports it alone but
        # the union of its cited chunks does, keep the claim and retain only the
        # co-cited markers that actually contribute. Strictness is preserved: a
        # figure present in none of the cited chunks still fails, and a wrong
        # marker that contributes nothing is still stripped.
        valid_ids = [i for i in ids if i in cmap]
        if len(valid_ids) > 1:
            union_res = _verify_claim_union(sentence, valid_ids, cmap)
            if union_res["status"] != "NOT_SUPPORTED":
                contributors = _contributing_ids(sentence, valid_ids, cmap)
                kept_sentences.append(_keep_only_markers(sentence, contributors))
                for i in contributors:
                    if i not in kept_ids:
                        kept_ids.append(i)
                continue

        # Recovery: is the claim supported by any OTHER retrieved evidence chunk?
        grounded, grounded_by = _claim_grounding(sentence, evidence)
        if grounded and grounded_by:
            kept_sentences.append(_remap_marker(sentence, grounded_by))
            if grounded_by not in kept_ids:
                kept_ids.append(grounded_by)
            dropped.append(
                {
                    "claim": sentence,
                    "reason": "recited",  # marker corrected to supporting evidence
                    "original_ids": ids,
                    "grounded_by": grounded_by,
                }
            )
            continue

        dropped.append({"claim": sentence, "reason": "not_supported", "original_ids": ids})

    return {
        "answer": " ".join(kept_sentences).strip(),
        "kept_ids": kept_ids,
        "dropped": dropped,
    }
