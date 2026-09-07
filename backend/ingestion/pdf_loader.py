"""
PDF text extraction that preserves page numbers.

Uses pdfplumber for reliable text extraction. Each page is returned with its
1-based page number, which is mandatory for building accurate citations later.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import pdfplumber

from backend.config import settings
from backend.ingestion import visual
from backend.text_normalize import join_space_thousands


@dataclass
class PageText:
    """Text extracted from a single PDF page.

    ``content_type`` distinguishes ordinary text-layer text ("text") from
    visual-derived records ("table", "ocr_page", "figure"). ``visual_ref`` is an
    optional path to a rendered image of the source page, used so a citation can
    point back to the actual visual.
    """

    document_file: str  # PDF file name, e.g. "Clicks-Sustainability-Report-2022.pdf"
    page_number: int  # 1-based page number
    text: str
    content_type: str = "text"
    visual_ref: str | None = None


_WS_RE = re.compile(r"[ \t]+")
_MULTI_NL_RE = re.compile(r"\n{3,}")


def clean_text(raw: str) -> str:
    """Normalize whitespace while preserving meaningful line breaks.

    Financial reports contain tables and figures; we keep single newlines so
    row structure survives, but collapse runs of spaces and excessive blank
    lines that add noise to retrieval.
    """
    if not raw:
        return ""
    # Normalize weird unicode spaces and bullets.
    text = raw.replace("\u00a0", " ").replace("\u200b", "")
    # Collapse horizontal whitespace.
    lines = [_WS_RE.sub(" ", line).strip() for line in text.split("\n")]
    text = "\n".join(line for line in lines if line != "")
    text = _MULTI_NL_RE.sub("\n\n", text)
    # Rejoin space-separated thousands from OCR of financial tables so a figure
    # like "64 392" survives as a single token/value for retrieval and citation
    # verification. Conservative: never merges Scope 1/2, years, or 3-digit
    # table columns (see backend.text_normalize).
    text = join_space_thousands(text)
    return text.strip()


def _extract_page_text(page) -> str:
    """Extract a page's text layer in correct reading order.

    ``page.extract_text()`` sorts glyphs by absolute (x, y) position. On the
    two-column / two-page "spread" layouts these ESG reports use, that
    interleaves the columns and shreds table rows - a metric label ends up
    separated from its value (e.g. "Scope 1" far from "58 644"), and adjacent
    columns' characters get intermixed. We instead read words in the PDF's
    native text-flow order (``use_text_flow=True``), which follows the content
    stream and preserves human reading order across columns, so each table row's
    label and its numbers stay together. Falls back to the default extractor if
    word extraction yields nothing (e.g. an image-only page).
    """
    try:
        words = page.extract_words(
            x_tolerance=2,
            y_tolerance=2,
            use_text_flow=True,
            keep_blank_chars=False,
        )
    except Exception:
        words = []
    if words:
        return " ".join(w["text"] for w in words)
    try:
        return page.extract_text() or ""
    except Exception:
        return ""


def extract_pages(pdf_path: Path) -> Iterator[PageText]:
    """Yield cleaned text for every page of a PDF, with 1-based page numbers.

    Ordinary text-layer text is always yielded first (content_type="text"),
    exactly as the original text-only pipeline did. When ENABLE_VISUAL_INGEST is
    on, additional records for vector tables, scanned pages (OCR), and figures
    are also yielded (see backend.ingestion.visual). Pages that contain no
    extractable text AND no recoverable visual content are skipped.
    """
    file_name = pdf_path.name
    visual_on = settings.ENABLE_VISUAL_INGEST
    vdoc = visual.open_document(pdf_path) if visual_on else None
    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            for idx, page in enumerate(pdf.pages):
                page_number = idx + 1
                if settings.LAYOUT_AWARE_EXTRACTION:
                    raw = _extract_page_text(page)
                else:
                    try:
                        raw = page.extract_text() or ""
                    except Exception:
                        raw = ""
                cleaned = clean_text(raw)
                if cleaned:
                    yield PageText(
                        document_file=file_name,
                        page_number=page_number,
                        text=cleaned,
                    )
                if visual_on:
                    for rec in visual.extract_visual_records(
                        vdoc, pdf_path, idx, page, cleaned
                    ):
                        yield PageText(
                            document_file=file_name,
                            page_number=page_number,
                            text=rec.text,
                            content_type=rec.content_type,
                            visual_ref=rec.visual_ref,
                        )
    finally:
        if vdoc is not None:
            try:
                vdoc.close()
            except Exception:
                pass


def find_pdfs(pdf_dir: Path) -> list[Path]:
    """Return a sorted list of all PDF files under a directory (recursive).

    Discovery is recursive so nested folders (e.g. an extracted archive with a
    subdirectory of reports) are handled automatically. No file names are
    hardcoded; every ``*.pdf`` found becomes part of the knowledge base.
    """
    if not pdf_dir.exists():
        return []
    return sorted(pdf_dir.rglob("*.pdf"))
