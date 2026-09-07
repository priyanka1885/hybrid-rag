"""
audit_chunking.py — Comprehensive diagnostic of the chunking strategy.

Checks whether the current chunker + chunk store is breaking retrieved context
for multimodal (text / table / image) financial-report documents. It is a
READ-ONLY audit: it loads the persisted chunk store (ChunkStore.load()) and
reports on five critical dimensions:

  1. Table Integrity & Boundary Splitting
  2. Chunk Size & Overlap Distribution
  3. Image & Visual Data Handling
  4. Orphaned Numeric Chunks
  5. Context Window Completeness (Sasol Scope 1 / 58 644)

Run:  python audit_chunking.py
"""
from __future__ import annotations

import re
import statistics
from collections import Counter

from backend.config import settings
from backend.retrieval.store import ChunkStore

# --- small formatting helpers ----------------------------------------------
BAR = "=" * 78
SUB = "-" * 78


def h1(title: str) -> None:
    print("\n" + BAR)
    print(title)
    print(BAR)


def h2(title: str) -> None:
    print("\n" + SUB)
    print(title)
    print(SUB)


# --- text analysis primitives ----------------------------------------------
_DIGIT_RE = re.compile(r"\d")
_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
_ALPHA_WORD_RE = re.compile(r"[A-Za-z]{3,}")  # a "real" word (>=3 letters)
# A number token: 58, 58.6, 58,644, 58 644 handled via normalization below.
_NUMBER_TOKEN_RE = re.compile(r"^[+-]?\d[\d.,]*%?$")
_HEADER_HINT_RE = re.compile(
    r"\b(units?|20\d{2}|19\d{2}|total|scope|kilotons?|tonnes?|co\s*2|%|ratio|intensity)\b",
    re.IGNORECASE,
)
# Metric-label hints: an alphabetic phrase that names a measured quantity.
_METRIC_HINT_RE = re.compile(
    r"\b(scope|emission|energy|water|waste|carbon|ghg|intensity|revenue|"
    r"employees?|electricity|consumption|total|direct|indirect|fuel|"
    r"greenhouse|diesel|petrol|renewable|spend|cost|rate|number)\b",
    re.IGNORECASE,
)


def word_tokens(text: str) -> list[str]:
    return text.split()


def numeric_density(text: str) -> float:
    words = word_tokens(text)
    if not words:
        return 0.0
    numeric = sum(1 for w in words if _DIGIT_RE.search(w))
    return numeric / len(words)


def alpha_word_count(text: str) -> int:
    return len(_ALPHA_WORD_RE.findall(text))


def number_token_count(text: str) -> int:
    return sum(1 for w in word_tokens(text) if _DIGIT_RE.search(w))


def is_markdown_table(text: str) -> bool:
    return "|" in text and "---" in text


def looks_tabular(chunk: dict) -> bool:
    """A chunk is 'tabular' if declared a table/figure, is markdown, or is
    number-dense the same way the chunker itself detects tables."""
    if chunk.get("content_type") in ("table", "figure"):
        return True
    t = chunk["text"]
    if is_markdown_table(t):
        return True
    return numeric_density(t) >= settings.TABLE_NUMERIC_DENSITY and number_token_count(t) >= 4


def md_columns(text: str) -> int:
    """Approximate column count of a markdown table (max cells across rows)."""
    best = 0
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("|"):
            cells = [c for c in line.strip("|").split("|")]
            best = max(best, len(cells))
    return best


def md_has_header_labels(text: str) -> bool:
    """True if the markdown header row carries at least one alphabetic label
    (i.e. a real column header like 'Units' or a metric name), not just years
    or numbers."""
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("|") and "---" not in line:
            cells = [c.strip() for c in line.strip("|").split("|")]
            # header is the first pipe row
            return any(_ALPHA_WORD_RE.search(c) for c in cells)
    return False


def md_row_label_ratio(text: str) -> float:
    """Fraction of data rows whose first cell contains an alphabetic label."""
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("|") and "---" not in line:
            rows.append([c.strip() for c in line.strip("|").split("|")])
    data_rows = rows[1:] if len(rows) > 1 else rows
    if not data_rows:
        return 0.0
    labelled = sum(1 for r in data_rows if r and _ALPHA_WORD_RE.search(r[0]))
    return labelled / len(data_rows)


# ---------------------------------------------------------------------------
def load_store() -> ChunkStore:
    store = ChunkStore.load()
    print(f"Loaded ChunkStore: {len(store)} chunks from persisted store.")
    return store


# === CHECK 1 ================================================================
def check_table_integrity(store: ChunkStore) -> dict:
    h1("CHECK 1  Table Integrity & Boundary Splitting")
    tabular = [c for c in store.chunks if looks_tabular(c)]
    md_tables = [c for c in tabular if is_markdown_table(c["text"])]

    detached_headers = []   # data rows but no alpha header / labels
    detached_data = []      # markdown table w/ header but rows lack labels
    single_col = []         # markdown table collapsed to 1 data column
    orphan_numeric_tables = []  # tabular chunk that is essentially numbers only

    for c in md_tables:
        t = c["text"]
        cols = md_columns(t)
        has_labels = md_has_header_labels(t)
        row_label_ratio = md_row_label_ratio(t)
        if cols <= 2:
            single_col.append(c)
        if not has_labels and row_label_ratio < 0.25:
            detached_headers.append(c)
        elif has_labels and row_label_ratio < 0.25:
            detached_data.append(c)
        if alpha_word_count(t) <= 2 and number_token_count(t) >= 3:
            orphan_numeric_tables.append(c)

    # Boundary splitting in TEXT chunks: a header row (years / Units) that ends
    # a chunk with the data rows presumably starting in the next chunk, or a
    # chunk that opens with number rows but has no preceding header.
    boundary_split = []
    for c in store.chunks:
        if c.get("content_type") != "text":
            continue
        t = c["text"]
        words = word_tokens(t)
        if len(words) < 10:
            continue
        # tail is a lone header (years / Units) with no numeric data following
        tail = " ".join(words[-12:])
        head = " ".join(words[:12])
        tail_is_header = bool(_YEAR_RE.search(tail)) and numeric_density(tail) < 0.35
        head_is_data = number_token_count(head) >= 4 and alpha_word_count(head) <= 2
        if tail_is_header or head_is_data:
            boundary_split.append(c)

    print(f"Tabular chunks (declared/markdown/number-dense): {len(tabular)}")
    print(f"  of which markdown tables:                      {len(md_tables)}")
    print(f"Markdown tables collapsed to <=2 columns:        {len(single_col)}")
    print(f"Tables with DETACHED headers (no labels+no rows):{len(detached_headers)}")
    print(f"Tables with header but DETACHED data rows:       {len(detached_data)}")
    print(f"Numeric-only 'tables' (labels stripped):         {len(orphan_numeric_tables)}")
    print(f"TEXT chunks with boundary-split header/data:     {len(boundary_split)}")

    def show(label, items, n=3):
        if not items:
            return
        print(f"\n  Examples — {label}:")
        for c in items[:n]:
            snippet = c["text"].replace("\n", " \\n ")[:180]
            print(f"    [{c['chunk_id']} p{c['page_number']} {c['content_type']}] {snippet}")

    show("single-column markdown tables", single_col)
    show("detached headers", detached_headers)
    show("numeric-only tables", orphan_numeric_tables)
    show("text boundary splits", boundary_split)

    return {
        "tabular": len(tabular),
        "md_tables": len(md_tables),
        "single_col": len(single_col),
        "detached_headers": len(detached_headers),
        "detached_data": len(detached_data),
        "orphan_numeric_tables": len(orphan_numeric_tables),
        "boundary_split_text": len(boundary_split),
    }


# === CHECK 2 ================================================================
def check_size_distribution(store: ChunkStore) -> dict:
    h1("CHECK 2  Chunk Size & Overlap Distribution")

    def stats(values):
        if not values:
            return {}
        values = sorted(values)
        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "mean": round(statistics.mean(values), 1),
            "median": int(statistics.median(values)),
            "p10": values[int(0.10 * (len(values) - 1))],
            "p90": values[int(0.90 * (len(values) - 1))],
        }

    by_type: dict[str, list[int]] = {}
    char_by_type: dict[str, list[int]] = {}
    for c in store.chunks:
        ct = c.get("content_type", "text")
        by_type.setdefault(ct, []).append(len(word_tokens(c["text"])))
        char_by_type.setdefault(ct, []).append(len(c["text"]))

    all_words = [w for v in by_type.values() for w in v]
    all_chars = [w for v in char_by_type.values() for w in v]

    print(f"Configured  CHUNK_SIZE={settings.CHUNK_SIZE} words, "
          f"CHUNK_OVERLAP={settings.CHUNK_OVERLAP} words "
          f"(step={settings.CHUNK_SIZE - settings.CHUNK_OVERLAP})")
    print(f"Configured  TABLE_CHUNK_SIZE={settings.TABLE_CHUNK_SIZE} words, "
          f"TABLE_CHUNK_OVERLAP={settings.TABLE_CHUNK_OVERLAP} words")

    print("\nWord-token length (approx tokens):")
    print(f"  ALL: {stats(all_words)}")
    for ct, vals in sorted(by_type.items()):
        print(f"  {ct:10s}: {stats(vals)}")

    print("\nCharacter length:")
    print(f"  ALL: {stats(all_chars)}")
    for ct, vals in sorted(char_by_type.items()):
        print(f"  {ct:10s}: {stats(vals)}")

    tiny = [c for c in store.chunks if len(word_tokens(c["text"])) < 15]
    tiny_tables = [c for c in tiny if looks_tabular(c)]
    print(f"\nVery small chunks (<15 words):        {len(tiny)}")
    print(f"  of which tabular (context-starved):  {len(tiny_tables)}")

    return {"tiny": len(tiny), "tiny_tables": len(tiny_tables),
            "word_stats": stats(all_words)}


# === CHECK 3 ================================================================
def check_visual_handling(store: ChunkStore) -> dict:
    h1("CHECK 3  Image & Visual Data Handling")
    types = Counter(c.get("content_type", "text") for c in store.chunks)
    print("content_type distribution:")
    for ct, n in types.most_common():
        print(f"  {ct:10s}: {n}")

    has_image_type = types.get("image", 0)
    has_caption_type = types.get("caption", 0)
    figures = types.get("figure", 0)
    ocr_pages = types.get("ocr_page", 0)
    tables = types.get("table", 0)

    with_vref = [c for c in store.chunks if c.get("visual_ref")]
    vref_by_type = Counter(c.get("content_type") for c in with_vref)

    # Caption-like text living inside plain text chunks (never promoted to a
    # dedicated caption/image chunk).
    caption_markers = re.compile(r"\b(figure|fig\.|chart|exhibit|table)\s*\d+", re.IGNORECASE)
    inline_captions = [c for c in store.chunks
                       if c.get("content_type") == "text" and caption_markers.search(c["text"])]

    print(f"\nDedicated 'image' chunks:   {has_image_type}")
    print(f"Dedicated 'caption' chunks: {has_caption_type}")
    print(f"'figure' chunks (OCR text): {figures}")
    print(f"'ocr_page' chunks:          {ocr_pages}")
    print(f"'table' chunks:             {tables}")
    print(f"Chunks with a visual_ref (page image): {len(with_vref)}  {dict(vref_by_type)}")
    print(f"Caption-like text buried in plain 'text' chunks: {len(inline_captions)}")

    print(f"\nVisual ingest enabled at config: ENABLE_VISUAL_INGEST="
          f"{settings.ENABLE_VISUAL_INGEST}, ENABLE_OCR={settings.ENABLE_OCR}")

    if has_image_type == 0 and has_caption_type == 0:
        print("\n  NOTE: There is NO 'image' or 'caption' content_type in the store.")
        print("        Visual content is represented only as OCR'd 'figure'/'table' text.")
        print("        Chart geometry / captions are not preserved as dedicated blocks.")

    return {"image": has_image_type, "caption": has_caption_type,
            "figure": figures, "ocr_page": ocr_pages, "table": tables,
            "inline_captions": len(inline_captions), "with_vref": len(with_vref)}


# === CHECK 4 ================================================================
def check_orphaned_numeric(store: ChunkStore) -> dict:
    h1("CHECK 4  Orphaned Numeric Chunks")
    orphans = []
    for c in store.chunks:
        t = c["text"]
        nums = number_token_count(t)
        if nums < 3:
            continue
        # No metric label anywhere in the chunk, and few real words overall.
        has_metric = bool(_METRIC_HINT_RE.search(t))
        alpha = alpha_word_count(t)
        if not has_metric and alpha <= max(2, nums // 4):
            orphans.append(c)

    by_type = Counter(c.get("content_type") for c in orphans)
    print(f"Chunks with >=3 numbers but NO metric label nearby: {len(orphans)}")
    print(f"  by content_type: {dict(by_type)}")
    print("\n  Examples:")
    for c in orphans[:6]:
        snippet = c["text"].replace("\n", " \\n ")[:160]
        print(f"    [{c['chunk_id']} p{c['page_number']} {c['content_type']} "
              f"nums={number_token_count(c['text'])} alpha={alpha_word_count(c['text'])}] {snippet}")

    return {"orphans": len(orphans), "by_type": dict(by_type)}


# === CHECK 5 ================================================================
def check_context_completeness(store: ChunkStore) -> dict:
    h1("CHECK 5  Context Window Completeness (Sasol Scope 1 / 58 644)")

    def norm(s: str) -> str:
        return s.replace(" ", "").lower()

    target_num = "58644"
    hits = []
    for c in store.chunks:
        t = c["text"]
        if target_num in norm(t) or re.search(r"\bscope\s*1\b", t, re.IGNORECASE):
            # keep Sasol-focused but include any doc that matches the metric
            hits.append(c)

    # Prioritise Sasol emissions chunks that actually carry 58 644.
    hits.sort(key=lambda c: (target_num not in norm(c["text"]),
                             "sasol" not in c.get("document_file", "").lower()))

    print(f"Chunks matching '58 644' or 'Scope 1': {len(hits)}")
    complete = 0
    for c in hits[:10]:
        t = c["text"]
        has_value = target_num in norm(t)
        has_label = bool(re.search(r"\b(direct\s+)?scope\s*1\b", t, re.IGNORECASE))
        years = sorted(set(_YEAR_RE.findall(t)))  # returns prefixes; recompute below
        year_hits = sorted(set(re.findall(r"\b(?:19|20)\d{2}\b", t)))
        has_header = len(year_hits) >= 1
        both = has_value and has_label and has_header
        complete += 1 if both else 0

        print("\n" + SUB)
        print(f"chunk_id={c['chunk_id']}  page={c['page_number']}  "
              f"type={c['content_type']}  doc={c['document_file']}")
        print(f"  contains value 58 644 : {has_value}")
        print(f"  contains label Scope 1: {has_label}")
        print(f"  contains year headers : {has_header}  {year_hits}")
        print(f"  COMPLETE (value+label+header in one block): {both}")
        print("  FULL TEXT:")
        print("  " + c["text"].replace("\n", "\n  "))

    print("\n" + SUB)
    print(f"Chunks that are self-contained (value+label+header together): "
          f"{complete}/{min(len(hits),10)} shown")
    return {"hits": len(hits), "complete_shown": complete}


# ---------------------------------------------------------------------------
def verdict(c1, c2, c3, c4, c5) -> None:
    h1("VERDICT")
    defects = []
    if c1["single_col"] or c1["orphan_numeric_tables"] or c1["detached_headers"]:
        defects.append(
            f"Table splitting: {c1['single_col']} single-column tables, "
            f"{c1['orphan_numeric_tables']} numeric-only tables, "
            f"{c1['detached_headers']} tables with detached headers — "
            "row labels and column headers are separated from their data.")
    if c1["boundary_split_text"]:
        defects.append(
            f"{c1['boundary_split_text']} text chunks split a table across a "
            "window boundary (header or data row orphaned).")
    if c2["tiny_tables"]:
        defects.append(
            f"{c2['tiny_tables']} tabular chunks are <15 words — too little "
            "surrounding context for a metric row to be self-explaining.")
    if c3["image"] == 0 and c3["caption"] == 0:
        defects.append(
            "No dedicated 'image'/'caption' content_type: visual/caption data "
            "is only OCR text, chart structure and captions are not atomic blocks.")
    if c4["orphans"]:
        defects.append(
            f"{c4['orphans']} orphaned numeric chunks (numbers with no metric "
            "label in the same chunk).")

    if defects:
        print("CHUNKING DEFECTS DETECTED:\n")
        for i, d in enumerate(defects, 1):
            print(f"  {i}. {d}")
        print("\n=> Tables/multimodal blocks are NOT consistently atomic. "
              "A table-aware, atomic-block chunker is recommended (see proposal).")
    else:
        print("No significant chunking defects detected.")


def main() -> None:
    h1("CHUNKING AUDIT")
    store = load_store()
    c1 = check_table_integrity(store)
    c2 = check_size_distribution(store)
    c3 = check_visual_handling(store)
    c4 = check_orphaned_numeric(store)
    c5 = check_context_completeness(store)
    verdict(c1, c2, c3, c4, c5)


if __name__ == "__main__":
    main()
