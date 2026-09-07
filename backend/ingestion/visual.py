"""
Multimodal (visual) extraction helpers for the ingestion pipeline.

This module is the ONLY place that knows about PyMuPDF / OCR. It converts the
non-text content of a PDF page into plain-text records that the EXISTING
retrieval pipeline (BM25 + all-MiniLM-L6-v2 + hybrid + reranker + verifier) can
consume unchanged. Each record also carries a ``visual_ref`` - the path to a
rendered image of the source page - so a citation can point back to the actual
visual.

Design constraints honoured here:
  * Selective OCR - a whole page is only OCR'd when it has essentially no text
    layer (a scanned page); embedded images are only OCR'd when they are large
    enough to be a real figure/table and actually contain readable text. Small
    decorative logos/icons are skipped.
  * No duplication - OCR never runs on a page that already has a real text
    layer, and a figure's OCR text is dropped if it is already contained in the
    page's text layer.
  * Conservative graphs/charts - we index only the text OCR can actually read
    (titles, axis/legend/data labels) plus a marker. We never infer numeric
    values from plotted lines/bars.

Everything is best-effort: if PyMuPDF or the Tesseract engine is unavailable,
the helpers degrade gracefully (return nothing) instead of raising, so the
text-only pipeline is never broken.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

from backend.config import PAGE_IMAGE_DIR, settings

_ALNUM_RE = re.compile(r"[A-Za-z0-9]")
_SAFE_RE = re.compile(r"[^A-Za-z0-9._-]+")
_ALPHA_WORD_RE = re.compile(r"[A-Za-z]{2,}")
_DIGIT_RE = re.compile(r"\d")


@dataclass
class VisualRecord:
    """A text representation of some visual page content."""

    content_type: str  # "table" | "ocr_page" | "figure"
    text: str
    visual_ref: Optional[str] = None


# --- capability probing (cached) -------------------------------------------
@lru_cache(maxsize=1)
def _pymupdf():
    """Return the pymupdf module, or None if it is not installed."""
    try:
        import pymupdf  # type: ignore

        return pymupdf
    except Exception:
        try:
            import fitz as pymupdf  # type: ignore  # older name

            return pymupdf
        except Exception:
            return None


@lru_cache(maxsize=1)
def _pytesseract():
    """Return (pytesseract, Image) if OCR is usable, else (None, None)."""
    try:
        import pytesseract  # type: ignore
        from PIL import Image  # type: ignore
    except Exception:
        return None, None
    if settings.TESSERACT_CMD:
        pytesseract.pytesseract.tesseract_cmd = settings.TESSERACT_CMD
    try:
        # Fails if the tesseract engine binary is not reachable.
        pytesseract.get_tesseract_version()
    except Exception:
        return None, None
    return pytesseract, Image


def pymupdf_available() -> bool:
    return _pymupdf() is not None


def ocr_available() -> bool:
    """True only if OCR is enabled AND the engine is actually usable."""
    if not settings.ENABLE_OCR:
        return False
    pt, _ = _pytesseract()
    return pt is not None


def capabilities() -> dict:
    """Human-readable snapshot of what the visual layer can do right now."""
    return {
        "visual_ingest": settings.ENABLE_VISUAL_INGEST,
        "pymupdf": pymupdf_available(),
        "table_extraction": settings.ENABLE_TABLE_EXTRACTION,
        "ocr_enabled": settings.ENABLE_OCR,
        "ocr_usable": ocr_available(),
    }


# --- low-level helpers ------------------------------------------------------
def _meaningful_len(text: str) -> int:
    return len(_ALNUM_RE.findall(text or ""))


def open_document(pdf_path: Path):
    """Open a PDF with PyMuPDF, or return None if unavailable."""
    mod = _pymupdf()
    if mod is None:
        return None
    try:
        return mod.open(str(pdf_path))
    except Exception:
        return None


def _page_image_path(pdf_path: Path, page_number: int) -> Path:
    stem = _SAFE_RE.sub("_", pdf_path.stem)
    return PAGE_IMAGE_DIR / f"{stem}_p{page_number}.png"


def render_page_png(vdoc, pdf_path: Path, page_index: int) -> Optional[str]:
    """Render a page to a PNG (once) and return its path as the visual ref."""
    if vdoc is None:
        return None
    out = _page_image_path(pdf_path, page_index + 1)
    try:
        if out.exists():
            return str(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        page = vdoc[page_index]
        mod = _pymupdf()
        zoom = settings.PAGE_RENDER_DPI / 72.0
        matrix = mod.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=matrix)
        pix.save(str(out))
        return str(out)
    except Exception:
        return None


def _ocr_pixmap_text(pix) -> str:
    pt, Image = _pytesseract()
    if pt is None:
        return ""
    try:
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        return pt.image_to_string(img, lang=settings.OCR_LANGUAGE) or ""
    except Exception:
        return ""


def ocr_page_text(vdoc, page_index: int) -> str:
    """OCR a full page (used for scanned pages with no text layer)."""
    if vdoc is None or not ocr_available():
        return ""
    try:
        page = vdoc[page_index]
        mod = _pymupdf()
        zoom = settings.PAGE_RENDER_DPI / 72.0
        pix = page.get_pixmap(matrix=mod.Matrix(zoom, zoom))
        return _ocr_pixmap_text(pix)
    except Exception:
        return ""


# --- table extraction (vector tables via pdfplumber) ------------------------
def _table_to_markdown(table: list[list]) -> str:
    """Serialize a pdfplumber table (list of rows) to a Markdown pipe table.

    Row/column structure is preserved so a metric label and its value stay on
    the same row instead of being flattened into a bag of words.
    """
    rows = []
    for raw_row in table:
        cells = [(c if c is not None else "").replace("\n", " ").strip() for c in raw_row]
        if any(cells):
            rows.append(cells)
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    header = rows[0]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * width) + " |"]
    for r in rows[1:]:
        lines.append("| " + " | ".join(r) + " |")
    return "\n".join(lines)


def _cell_is_numeric(cell: str) -> bool:
    """True if a cell is a bare number (58, 58 644, 4.12, 31%), not a label.

    A cell that carries any real word (>=2 letters) is treated as a label, so a
    footnote like "Refer to page 63" is not mistaken for a numeric value.
    """
    cell = (cell or "").strip()
    if not cell or _ALPHA_WORD_RE.search(cell):
        return False
    return bool(_DIGIT_RE.search(cell))


def _table_is_valid(rows: list[list]) -> bool:
    """Reject tables whose row labels / column headers were lost by extraction.

    pdfplumber's line-based cell detection on the two-page "spread" layouts
    these ESG reports use frequently collapses a multi-column table into one or
    two columns of bare numbers, dropping the metric-label column and/or the
    year header row (e.g. producing ``| 2023 | ... | 58 644 |`` with no
    "Direct scope 1" label). Such a table is worse than useless for retrieval -
    a value with no label - and the layout-aware TEXT stream already retains the
    full row ("Direct scope 1 58 644 57 284") intact. So we discard the table
    representation when, per the audit rule, more than 50% of the header cells
    OR more than 50% of the row-label cells are empty AND the body is
    number-dominated, or when the table collapsed to <=2 columns of numbers.
    """
    cleaned: list[list[str]] = []
    for raw_row in rows:
        cells = [(c if c is not None else "").replace("\n", " ").strip() for c in raw_row]
        if any(cells):
            cleaned.append(cells)
    if not cleaned:
        return False

    width = max(len(r) for r in cleaned)
    cleaned = [r + [""] * (width - len(r)) for r in cleaned]
    header = cleaned[0]
    data = cleaned[1:] if len(cleaned) > 1 else cleaned

    # Numeric dominance of the body (non-empty cells only).
    body_cells = [c for r in data for c in r if c.strip()]
    if not body_cells:
        return False
    numeric_ratio = sum(1 for c in body_cells if _cell_is_numeric(c)) / len(body_cells)

    # A table dominated by prose (e.g. an assurance / governance list) is kept
    # regardless of width - the label-loss failure mode is specific to numeric
    # grids.
    if numeric_ratio < 0.5:
        return True

    # Collapsed to a single value column: labels are certainly gone.
    if width <= 2:
        return False

    # Fraction of header cells with no label text, and fraction of data rows
    # whose first (label) column is empty.
    header_empty_ratio = sum(1 for c in header if not c.strip()) / (len(header) or 1)
    first_col = [r[0] for r in data]
    label_empty_ratio = sum(1 for c in first_col if not c.strip()) / (len(first_col) or 1)

    if header_empty_ratio > 0.5 or label_empty_ratio > 0.5:
        return False
    return True


def extract_tables_markdown(plumber_page) -> list[str]:
    """Return Markdown for each meaningful vector table on a pdfplumber page.

    Tables whose labels/headers were destroyed by extraction (collapsed numeric
    columns) are discarded via ``_table_is_valid`` so we fall back to the
    layout-aware text stream, which keeps each row's label and values together.
    """
    if not settings.ENABLE_TABLE_EXTRACTION:
        return []
    try:
        tables = plumber_page.extract_tables() or []
    except Exception:
        return []
    out: list[str] = []
    for tbl in tables:
        if not _table_is_valid(tbl):
            continue  # label-less / collapsed numeric grid -> rely on text stream
        md = _table_to_markdown(tbl)
        # Require at least a header + one data row and some real content.
        if md and md.count("\n") >= 2 and _meaningful_len(md) >= 4:
            out.append(md)
    return out


# --- coordinate-based chart / borderless-"table" reconstruction -------------
# Some financial-report pages present their numbers as vector INFOGRAPHICS
# (small bar charts with data labels positioned graphically) rather than ruled
# tables. pdfplumber's line-based ``extract_tables`` finds nothing on such a
# page, so it falls through to ``extract_text`` which linearizes the graphic
# labels into a jumbled reading order and destroys the column-to-value (e.g.
# year -> value) mapping. This helper rebuilds that mapping directly from word
# geometry: it locates year-header axes, then binds each year column to the
# numeric data label sitting above it, so a metric label and its per-year
# values survive as one coherent row. It is only used as a FALLBACK when the
# line-based extractor returns nothing (see ``extract_visual_records``), so it
# never competes with a genuine ruled table.
_YEAR_RE = re.compile(r"^(19|20)\d{2}$")
_VALUE_RE = re.compile(r"^\d[\d.,]*$")  # numeric data label (year cells excluded separately)
_ROW_TOL = 4.0            # px: words within this vertical distance share a visual row
_COL_TOL = 18.0           # px: a value binds to a year column if left edges align within this
_COL_GAP_MAX = 120.0      # px: adjacent year columns wider apart than this start a new chart
_THOUSANDS_GAP_MAX = 12.0  # px: gap under which "8 347.9" is rejoined into one value token
_MAX_VALUE_BAND = 160.0   # px: a bar's data label sits within this distance above its axis
_MAX_LABEL_BAND = 42.0    # px: a chart title sits within this distance above its data labels
_LABEL_XPAD = 40.0        # px: horizontal slack when matching a title to a chart's columns


def _page_cells(plumber_page) -> list[dict]:
    """Flatten a page into positioned word cells (best-effort)."""
    words = plumber_page.extract_words(use_text_flow=False) or []
    cells: list[dict] = []
    for w in words:
        try:
            cells.append(
                {
                    "text": w["text"],
                    "x0": float(w["x0"]),
                    "x1": float(w["x1"]),
                    "top": float(w["top"]),
                }
            )
        except (KeyError, TypeError, ValueError):
            continue
    return cells


def _merge_row_numbers(cells: list[dict]) -> list[dict]:
    """Rejoin space-separated thousands (e.g. "8" + "347.9" -> "8347.9").

    South-African reports use a space as the thousands separator, which
    pdfplumber emits as two adjacent word tokens. Merging them keeps a value
    like ``8 347.9`` as a single alignable token.
    """
    out: list[dict] = []
    i = 0
    while i < len(cells):
        c = cells[i]
        if i + 1 < len(cells) and re.fullmatch(r"\d{1,3}", c["text"]):
            nxt = cells[i + 1]
            if (
                re.fullmatch(r"\d{3}([.,]\d+)?", nxt["text"])
                and (nxt["x0"] - c["x1"]) < _THOUSANDS_GAP_MAX
            ):
                out.append(
                    {
                        "text": c["text"] + nxt["text"],
                        "x0": c["x0"],
                        "x1": nxt["x1"],
                        "top": c["top"],
                    }
                )
                i += 2
                continue
        out.append(c)
        i += 1
    return out


def _group_rows(cells: list[dict]) -> list[dict]:
    """Cluster cells into visual rows by vertical position, left-to-right."""
    rows: list[dict] = []
    for c in sorted(cells, key=lambda z: (z["top"], z["x0"])):
        placed = False
        for r in rows:
            if abs(r["top"] - c["top"]) <= _ROW_TOL:
                r["cells"].append(c)
                placed = True
                break
        if not placed:
            rows.append({"top": c["top"], "cells": [c]})
    for r in rows:
        r["cells"].sort(key=lambda z: z["x0"])
        r["cells"] = _merge_row_numbers(r["cells"])
    rows.sort(key=lambda z: z["top"])
    return rows


def _year_axis_groups(rows: list[dict]) -> list[dict]:
    """Find year-header axes, splitting side-by-side charts on the same row.

    A single visual row can carry the x-axis of several charts placed next to
    each other (e.g. "2021 2022 2023   2021 2022 2023"). Each ascending run of
    years is treated as a separate chart axis.
    """
    groups: list[dict] = []
    for r in rows:
        years = [(int(c["text"]), c["x0"]) for c in r["cells"] if _YEAR_RE.match(c["text"])]
        if len(years) < 2:
            continue
        run = [years[0]]
        runs = []
        for y in years[1:]:
            if y[0] > run[-1][0] and (y[1] - run[-1][1]) < _COL_GAP_MAX:
                run.append(y)
            else:
                runs.append(run)
                run = [y]
        runs.append(run)
        for run in runs:
            if len(run) >= 2:
                xs = [x for _, x in run]
                groups.append(
                    {"top": r["top"], "cols": run, "xa": min(xs), "xb": max(xs)}
                )
    return groups


def _series_values(group: dict, rows: list[dict], groups: list[dict]) -> list[tuple]:
    """Bind each year column to the numeric data label sitting above its axis.

    Bar-chart data labels sit above the x-axis, so values are searched in the
    vertical band directly above the header, bounded below by the nearest
    higher chart sharing the same horizontal region (prevents a lower chart
    from stealing an upper chart's numbers).
    """
    lower = group["top"] - _MAX_VALUE_BAND
    for h in groups:
        if h is group or h["top"] >= group["top"]:
            continue
        x_overlap = not (h["xb"] < group["xa"] - _COL_TOL or h["xa"] > group["xb"] + _COL_TOL)
        if x_overlap:
            lower = max(lower, h["top"])
    values: list[tuple] = []
    for year, cx in group["cols"]:
        best = None
        for r in rows:
            if not (lower < r["top"] < group["top"]):
                continue
            for c in r["cells"]:
                if _YEAR_RE.match(c["text"]) or not _VALUE_RE.match(c["text"]):
                    continue
                if abs(c["x0"] - cx) <= _COL_TOL and (best is None or c["top"] > best["top"]):
                    best = c
        values.append((year, best["text"] if best else None, best["top"] if best else None))
    return values


def _series_label(values: list[tuple], group: dict, rows: list[dict]) -> str:
    """Find the chart title: the nearest text row just above the data labels."""
    tops = [t for _, _, t in values if t is not None]
    if not tops:
        return ""
    min_top = min(tops)
    best = None
    for r in rows:
        if not (min_top - _MAX_LABEL_BAND <= r["top"] <= min_top + _ROW_TOL):
            continue
        texts = [
            c
            for c in r["cells"]
            if re.search(r"[A-Za-z]", c["text"])
            and not _YEAR_RE.match(c["text"])
            and not (c["x1"] < group["xa"] - _LABEL_XPAD or c["x0"] > group["xb"] + _LABEL_XPAD)
        ]
        if texts and (best is None or abs(r["top"] - min_top) < abs(best[0] - min_top)):
            best = (r["top"], texts)
    if not best:
        return ""
    return " ".join(c["text"] for c in sorted(best[1], key=lambda z: z["x0"]))


def _series_to_markdown(label: str, values: list[tuple]) -> str:
    years = [str(y) for y, _, _ in values]
    vals = [v if v is not None else "" for _, v, _ in values]
    header = "| " + " | ".join(["Metric"] + years) + " |"
    sep = "| " + " | ".join(["---"] * (len(years) + 1)) + " |"
    row = "| " + " | ".join([label or "Series"] + vals) + " |"
    return "\n".join([header, sep, row])


def extract_chart_series_markdown(plumber_page) -> list[str]:
    """Reconstruct year->value chart series from word geometry as Markdown.

    Returns one compact Markdown table per detected chart. Conservative: a
    series is only emitted when at least two year columns each bind to an
    aligned numeric value, so narrative pages produce nothing.
    """
    if not settings.ENABLE_TABLE_EXTRACTION:
        return []
    try:
        rows = _group_rows(_page_cells(plumber_page))
    except Exception:
        return []
    groups = _year_axis_groups(rows)
    out: list[str] = []
    for group in groups:
        values = _series_values(group, rows, groups)
        if sum(1 for _, v, _ in values if v is not None) < 2:
            continue
        md = _series_to_markdown(_series_label(values, group, rows), values)
        if _meaningful_len(md) >= 4:
            out.append(md)
    return out


# --- figure / chart extraction ---------------------------------------------
def _is_subset_of_page(text: str, page_text: str) -> bool:
    """True if most of ``text`` tokens already appear in the page text layer.

    Used to drop a figure whose OCR text merely repeats the surrounding
    narrative (avoids duplicate competing chunks).
    """
    toks = set(re.findall(r"[a-z0-9]+", text.lower()))
    if not toks:
        return True
    page_toks = set(re.findall(r"[a-z0-9]+", (page_text or "").lower()))
    if not page_toks:
        return False
    overlap = len(toks & page_toks) / len(toks)
    return overlap >= 0.8


def extract_figure_texts(vdoc, page_index: int, page_text: str) -> list[str]:
    """OCR sufficiently large embedded images and return figure text records.

    Conservative: only images covering >= FIGURE_MIN_AREA_RATIO of the page are
    considered (skips logos/icons); an image is kept only if OCR yields
    >= FIGURE_MIN_OCR_CHARS meaningful characters and that text is not already
    present in the page's text layer. We index only the OCR'd labels/text plus a
    marker; we do NOT infer plotted numeric values.
    """
    if vdoc is None or not ocr_available():
        return []
    mod = _pymupdf()
    try:
        page = vdoc[page_index]
        page_area = abs(page.rect.width * page.rect.height) or 1.0
        images = page.get_images(full=True)
    except Exception:
        return []

    seen: set[str] = set()
    records: list[str] = []
    for img in images:
        xref = img[0]
        if xref in seen:
            continue
        seen.add(xref)
        try:
            rects = page.get_image_rects(xref)
        except Exception:
            rects = []
        area_ratio = 0.0
        for r in rects:
            area_ratio = max(area_ratio, abs(r.width * r.height) / page_area)
        if area_ratio < settings.FIGURE_MIN_AREA_RATIO:
            continue  # decorative / small image
        try:
            pix = mod.Pixmap(vdoc, xref)
            if pix.n - pix.alpha >= 4:  # CMYK/other -> convert to RGB
                pix = mod.Pixmap(mod.csRGB, pix)
            ocr_text = _ocr_pixmap_text(pix)
        except Exception:
            ocr_text = ""
        ocr_text = " ".join(ocr_text.split())
        if _meaningful_len(ocr_text) < settings.FIGURE_MIN_OCR_CHARS:
            continue  # no real text in the image
        if _is_subset_of_page(ocr_text, page_text):
            continue  # already covered by the text layer
        records.append(f"[FIGURE p.{page_index + 1}] {ocr_text}")
    return records


# --- orchestration ----------------------------------------------------------
def extract_visual_records(
    vdoc, pdf_path: Path, page_index: int, plumber_page, page_text: str
) -> list[VisualRecord]:
    """Produce all visual records for a single page and attach a page image ref.

    ``page_text`` is the cleaned text-layer text already extracted for the page
    (empty string if the page has no text layer).
    """
    if not settings.ENABLE_VISUAL_INGEST:
        return []

    records: list[VisualRecord] = []

    # 1. Vector tables (structured, no OCR needed).
    table_mds = extract_tables_markdown(plumber_page)
    for md in table_mds:
        records.append(VisualRecord("table", md))
    # 1b. Fallback for chart/infographic pages that have NO ruled table: rebuild
    #     the year->value mapping from word geometry so it is not lost to the
    #     flattened text layer. Only runs when the line-based extractor found
    #     nothing, so it never competes with a genuine vector table.
    if not table_mds:
        for md in extract_chart_series_markdown(plumber_page):
            records.append(VisualRecord("table", md))

    scanned = _meaningful_len(page_text) < settings.OCR_MIN_CHARS
    if scanned:
        # 2. Scanned / image page -> OCR the whole page. This only runs when the
        #    text layer is essentially empty, so it never duplicates real text.
        ocr_text = ocr_page_text(vdoc, page_index)
        if _meaningful_len(ocr_text) >= settings.OCR_MIN_CHARS:
            from backend.ingestion.pdf_loader import clean_text  # local import

            records.append(VisualRecord("ocr_page", clean_text(ocr_text)))
    else:
        # 3. Figures / charts on a normal text page.
        for fig in extract_figure_texts(vdoc, page_index, page_text):
            records.append(VisualRecord("figure", fig))

    # Attach one rendered page image as the citation visual reference for every
    # visual record produced on this page.
    if records:
        vref = render_page_png(vdoc, pdf_path, page_index)
        if vref:
            for r in records:
                r.visual_ref = vref
    return records
