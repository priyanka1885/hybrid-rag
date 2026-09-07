"""Tests for PDF ingestion and chunking."""
from pathlib import Path

from backend.config import PDF_DIR, settings
from backend.ingestion.chunker import chunk_page, chunk_pages
from backend.ingestion.pdf_loader import PageText, clean_text, extract_pages, find_pdfs


def test_find_pdfs_recursive():
    pdfs = find_pdfs(PDF_DIR)
    assert len(pdfs) > 0, "Expected to find financial-report PDFs in DOCUMENTS_DIR"
    assert all(p.suffix.lower() == ".pdf" for p in pdfs)


def test_clean_text_collapses_whitespace():
    raw = "Hello    world\n\n\n\nFoo\u00a0bar   "
    cleaned = clean_text(raw)
    assert "    " not in cleaned
    assert "\n\n\n" not in cleaned
    assert "\u00a0" not in cleaned


def test_extract_pages_have_page_numbers():
    pdfs = find_pdfs(PDF_DIR)
    pages = list(extract_pages(pdfs[0]))
    assert len(pages) > 0
    # Page numbers must be 1-based and present (mandatory for citations).
    assert all(isinstance(p.page_number, int) and p.page_number >= 1 for p in pages)
    assert all(p.text.strip() for p in pages)
    assert all(p.document_file == pdfs[0].name for p in pages)


def test_chunk_page_metadata_and_overlap():
    words = " ".join(f"w{i}" for i in range(500))
    page = PageText(document_file="Report.pdf", page_number=42, text=words)
    chunks = chunk_page(page, chunk_size=100, chunk_overlap=20)
    assert len(chunks) > 1
    for c in chunks:
        assert c.page_number == 42
        assert c.document_file == "Report.pdf"
        assert c.chunk_id
        assert c.text
    # Overlap: consecutive chunks should share some words.
    first_words = set(chunks[0].text.split())
    second_words = set(chunks[1].text.split())
    assert first_words & second_words


def test_chunk_ids_are_unique():
    pages = [
        PageText(document_file="A.pdf", page_number=1, text=" ".join(f"a{i}" for i in range(300))),
        PageText(document_file="A.pdf", page_number=2, text=" ".join(f"b{i}" for i in range(300))),
    ]
    chunks = chunk_pages(pages, settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))


def test_table_region_is_atomic_with_header_leadin():
    # A number-dense (tabular) region must be emitted as ATOMIC table chunk(s)
    # that carry the preceding metric-label / column-header lead-in, so a value
    # like "58644" never appears without its label or year header.
    from backend.config import settings

    lead = "GHG emissions CO2e kilotons 2023 2022 Direct scope 1"
    body = " ".join(["58644", "57284", "5748", "6607", "36664", "37557"] * 20)
    page = PageText(document_file="R.pdf", page_number=1, text=f"{lead} {body}")

    chunks = chunk_page(page, settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)
    table_chunks = [c for c in chunks if c.content_type == "table"]

    assert table_chunks, "number-dense region should yield atomic table chunk(s)"
    # The value and its label/header must live together in the same chunk.
    assert any(
        "58644" in c.text and ("Direct" in c.text or "2023" in c.text)
        for c in table_chunks
    ), "table value must travel with its label and column header"


def test_narrative_page_has_no_table_chunks():
    # A digit-free prose page must be windowed as ordinary text only - the
    # atomic-table path must never fire on narrative content.
    from backend.config import settings

    prose_vocab = ["alpha", "beta", "gamma", "delta", "omega", "sigma"]
    narrative = " ".join(prose_vocab[i % len(prose_vocab)] for i in range(600))
    page = PageText(document_file="R.pdf", page_number=2, text=narrative)

    chunks = chunk_page(page, settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)
    assert chunks and all(c.content_type == "text" for c in chunks)


def test_clean_text_joins_space_separated_thousands():
    # OCR thousands separators become spaces; cleaning must rejoin them so the
    # figure survives as a single token, while leaving Scope labels untouched.
    cleaned = clean_text("Total greenhouse gas 64 392 kilotons Scope 1 and Scope 2")
    assert "64392" in cleaned
    assert "Scope 1 and Scope 2" in cleaned
