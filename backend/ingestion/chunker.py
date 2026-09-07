"""
Chunking for financial-report pages.

Strategy: HEADER-AWARE ATOMIC SEGMENTATION. Each page's word stream is first
segmented into NARRATIVE and TABLE regions based on local numeric density, then
each region is chunked differently:

  * Narrative regions -> sliding word window with overlap (as before), so prose
    retrieval is unchanged.
  * Table regions (number-dense runs) -> kept ATOMIC. A table region is never
    cut mid-row; it is emitted as a single chunk together with the narrative
    "lead-in" that immediately precedes it (the metric title / column headers
    such as "GHG emissions CO2e (kilotons) 2023 2022"), so a value like
    "58 644" always travels with its label AND its column header. If a table
    region is larger than the embedder can encode, it is split into row
    sub-blocks and the detected header lead-in is RE-PREPENDED to every
    sub-block, so headers are never detached from data.

Word-based sizing is reproducible and avoids extremely small or large chunks.
Chunk size / overlap and the table-segmentation knobs are configurable via
config.settings. Structured visual records (content_type table/figure/ocr_page)
coming from the ingestion pipeline are still indexed whole.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, asdict

from backend.config import friendly_document_name, settings
from backend.ingestion.pdf_loader import PageText

_ALPHA_WORD_RE = re.compile(r"[A-Za-z]{2,}")
# A token that IS a number (not merely one that contains a digit): 58, 58.6,
# 57,284, 4.12, 30%, 2023, -12. This deliberately excludes alphanumeric
# identifiers like "w0", "CO2" or "Scope1", which contain a digit but are not
# tabular values, so narrative text is never misread as a table.
_NUMERIC_TOKEN_RE = re.compile(r"^[+\-\u2013(]?\d[\d.,%/)]*$")


def _is_numeric_token(word: str) -> bool:
    """True only when the whole token is a numeric value (see regex above)."""
    return bool(_NUMERIC_TOKEN_RE.match(word))


def _numeric_density(words: list[str]) -> float:
    """Fraction of whitespace tokens that contain a digit.

    Financial tables are number-dense (typically > 0.35) while narrative prose
    is not (typically < 0.1), so this cleanly identifies tabular regions
    without any document- or metric-specific keywords.
    """
    if not words:
        return 0.0
    numeric = sum(1 for w in words if _is_numeric_token(w))
    return numeric / len(words)


@dataclass
class Chunk:
    chunk_id: str
    document_file: str
    document_name: str
    page_number: int
    text: str
    content_type: str = "text"
    visual_ref: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _make_chunk_id(document_file: str, page_number: int, text: str) -> str:
    """Stable id derived from source + content, so re-ingesting is idempotent."""
    h = hashlib.sha1(f"{document_file}|{page_number}|{text}".encode("utf-8")).hexdigest()
    return h[:16]


def _mk(page: PageText, doc_name: str, text: str, content_type: str) -> Chunk:
    """Construct a Chunk carrying the page's citation metadata."""
    return Chunk(
        chunk_id=_make_chunk_id(page.document_file, page.page_number, text),
        document_file=page.document_file,
        document_name=doc_name,
        page_number=page.page_number,
        text=text,
        content_type=content_type,
        visual_ref=page.visual_ref,
    )


# --- region segmentation ----------------------------------------------------
def _table_mask(words: list[str], win: int, density: float) -> list[bool]:
    """Mark each word position that sits inside a number-dense neighbourhood.

    A word is "in a table" if the local window of ``win`` words centred on it
    has a numeric density >= ``density``. Using a neighbourhood (not the single
    token) tolerates the label words interleaved among the numbers - e.g.
    "Direct scope 1 58 644 57 284" - while still firing only inside genuinely
    tabular runs, not in prose that merely mentions a year.
    """
    n = len(words)
    if n == 0:
        return []
    is_num = [_is_numeric_token(w) for w in words]
    # Prefix sums for O(1) window density.
    pref = [0] * (n + 1)
    for i, v in enumerate(is_num):
        pref[i + 1] = pref[i] + (1 if v else 0)
    half = max(1, win // 2)
    mask = [False] * n
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        cnt = pref[hi] - pref[lo]
        if cnt / (hi - lo) >= density:
            mask[i] = True
    return mask


def _regions(words: list[str], win: int, density: float, gap: int) -> list[tuple[str, int, int]]:
    """Return contiguous ``(kind, start, end)`` spans, kind in table/narrative.

    Short non-table gaps between two table stretches (a mid-table subtotal
    label, a sparse row) shorter than ``gap`` are absorbed into the table so a
    single table is not shattered into fragments.
    """
    n = len(words)
    if n == 0:
        return []
    mask = _table_mask(words, win, density)

    # Bridge short narrative gaps that are sandwiched between table stretches.
    i = 0
    while i < n:
        if not mask[i]:
            j = i
            while j < n and not mask[j]:
                j += 1
            if 0 < i and j < n and (j - i) <= gap:
                for k in range(i, j):
                    mask[k] = True
            i = j
        else:
            i += 1

    # Collapse the mask into contiguous spans.
    regions: list[tuple[str, int, int]] = []
    i = 0
    while i < n:
        kind = "table" if mask[i] else "narrative"
        j = i
        while j < n and (("table" if mask[j] else "narrative") == kind):
            j += 1
        regions.append((kind, i, j))
        i = j
    return regions


def _lead_in(words: list[str], start: int, max_lead: int) -> int:
    """Index where a table's header/label lead-in should begin.

    Walks back from ``start`` over up to ``max_lead`` narrative words to capture
    the metric label and column headers that introduce the table (e.g.
    "GHG emissions CO2e (kilotons) 2023 2022"). Stops at a sentence boundary so
    a whole paragraph of prose is not swallowed into the table chunk.
    """
    lo = max(0, start - max_lead)
    begin = start
    for k in range(start - 1, lo - 1, -1):
        w = words[k]
        if w.endswith((".", ":", ";")) and _ALPHA_WORD_RE.search(w):
            break  # previous sentence boundary
        begin = k
    return begin


# --- windowing --------------------------------------------------------------
def _window_span(
    page: PageText,
    doc_name: str,
    words: list[str],
    lo: int,
    hi: int,
    chunk_size: int,
    chunk_overlap: int,
    content_type: str,
) -> list[Chunk]:
    """Slide a fixed-size word window over ``words[lo:hi]`` producing Chunks."""
    if hi <= lo:
        return []
    if chunk_overlap >= chunk_size:
        # Guard against a misconfiguration that would loop forever.
        chunk_overlap = max(0, chunk_size // 4)
    step = chunk_size - chunk_overlap
    out: list[Chunk] = []
    for start in range(lo, hi, step):
        piece = words[start : min(start + chunk_size, hi)]
        if not piece:
            break
        out.append(_mk(page, doc_name, " ".join(piece), content_type))
        if start + chunk_size >= hi:
            break
    return out


def _atomic_table(
    page: PageText,
    doc_name: str,
    words: list[str],
    header_lo: int,
    tbl_lo: int,
    tbl_hi: int,
    max_words: int,
    content_type: str,
) -> list[Chunk]:
    """Emit a table region as atomic chunk(s) that always carry the header.

    ``header_lo..tbl_lo`` is the narrative lead-in (metric label + column
    headers); ``tbl_lo..tbl_hi`` is the number-dense table body. If lead-in +
    body fits in ``max_words`` it becomes ONE chunk. Otherwise the body is split
    into sub-blocks and the lead-in is re-prepended to each, so no data row is
    ever separated from its header.
    """
    header = words[header_lo:tbl_lo]
    body = words[tbl_lo:tbl_hi]
    if not body:
        return []

    if len(header) + len(body) <= max_words:
        text = " ".join(words[header_lo:tbl_hi]).strip()
        return [_mk(page, doc_name, text, content_type)] if text else []

    # Too large for one chunk: split the body, repeating the header on each part
    # so the column/row headers are never detached from the data.
    out: list[Chunk] = []
    budget = max(1, max_words - len(header))
    for s in range(0, len(body), budget):
        piece = body[s : s + budget]
        if not piece:
            break
        text = " ".join(header + piece).strip()
        if text:
            out.append(_mk(page, doc_name, text, content_type))
    return out


def _whole_chunk(page: PageText, doc_name: str) -> list[Chunk]:
    """Keep a structured visual record (table/figure/ocr_page) as one chunk.

    Tables are serialized Markdown; figures are OCR'd label text. Splitting
    these into word windows would destroy their structure, so they are indexed
    whole with their original content_type and visual_ref.
    """
    text = page.text.strip()
    if not text:
        return []
    return [_mk(page, doc_name, text, page.content_type)]


def chunk_page(page: PageText, chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    """Segment a page into narrative + atomic-table regions and chunk each.

    Narrative regions are windowed exactly as before; number-dense table regions
    are kept atomic and prepended with their header/label lead-in so a metric
    label, its column headers and its values stay together in one chunk (or, for
    oversized tables, in row sub-blocks that each repeat the header).
    """
    doc_name = friendly_document_name(page.document_file)

    # Structured visual records are already atomic; keep them intact.
    if page.content_type in ("table", "figure", "ocr_page"):
        return _whole_chunk(page, doc_name)

    words = page.text.split()
    if not words:
        return []

    density = settings.TABLE_NUMERIC_DENSITY
    win = settings.TABLE_DETECT_WINDOW
    gap = settings.TABLE_BRIDGE_GAP
    max_lead = settings.TABLE_LEADIN_WORDS
    max_tbl = settings.TABLE_MAX_WORDS

    regions = _regions(words, win, density, gap)

    # No tabular region at all -> behave exactly like the old narrative path.
    if not any(kind == "table" for kind, _, _ in regions):
        return _window_span(
            page, doc_name, words, 0, len(words), chunk_size, chunk_overlap, "text"
        )

    chunks: list[Chunk] = []
    for kind, lo, hi in regions:
        if kind == "narrative":
            chunks.extend(
                _window_span(
                    page, doc_name, words, lo, hi, chunk_size, chunk_overlap, "text"
                )
            )
        else:
            # Walk back over the preceding narrative words to capture the metric
            # title / column headers that introduce this table. The lead-in is
            # intentionally re-included here (it also appears in the adjacent
            # narrative chunk) so the table chunk is fully self-contained.
            header_lo = _lead_in(words, lo, max_lead)
            chunks.extend(
                _atomic_table(
                    page, doc_name, words, header_lo, lo, hi, max_tbl, "table"
                )
            )
    return chunks


def chunk_pages(pages: list[PageText], chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    """Chunk a list of pages, deduplicating by chunk_id."""
    seen: set[str] = set()
    out: list[Chunk] = []
    for page in pages:
        for chunk in chunk_page(page, chunk_size, chunk_overlap):
            if chunk.chunk_id in seen:
                continue
            seen.add(chunk.chunk_id)
            out.append(chunk)
    return out
