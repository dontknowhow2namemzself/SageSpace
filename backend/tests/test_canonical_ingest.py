"""Tests for the top-level canonical ingest entry point."""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

import core.database as database
from core import canonical
from core.canonical import db as canonical_db
from core.canonical import ingest as canonical_ingest


@pytest.fixture
def temp_db(monkeypatch):
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    monkeypatch.setattr(database, "DB_PATH", Path(tmp.name))
    database.init_db()
    yield Path(tmp.name)
    Path(tmp.name).unlink(missing_ok=True)


def _seed_book(book_id: str, file_path: str) -> None:
    from core.database import get_conn

    conn = get_conn()
    conn.execute(
        "INSERT INTO books (id, title, upload_date, raptor_status, file_path, ingest_version) "
        "VALUES (?, ?, ?, ?, ?, 1)",
        (book_id, "T", "2026-01-01T00:00:00Z", "ready", file_path),
    )
    conn.commit()
    conn.close()


def test_unsupported_format_raises(temp_db):
    _seed_book("bk_bad", "/tmp/some.txt")
    with pytest.raises(ValueError, match="Unsupported source format"):
        canonical_ingest.ingest_canonical("bk_bad", "/tmp/some.txt")


def test_ingest_real_epub_end_to_end(temp_db):
    """Run the full ingest entry point against a real EPUB from uploads/.
    Verifies: rows land in sections/blocks/ingestion_reports and the book's
    ingest_version is bumped to 2.
    """
    epub_files = sorted(
        (Path(__file__).parent.parent / "uploads").glob("*.epub")
    )
    if not epub_files:
        pytest.skip("no real EPUB in uploads/")

    book_id = "bk_real_epub_e2e"
    _seed_book(book_id, str(epub_files[0]))
    report = canonical_ingest.ingest_canonical(book_id, str(epub_files[0]))

    # Report sanity
    assert report["locator_type"] == "epub"
    assert report["final_blocks"] > 0
    assert report["sections_total"] > 0

    # DB persisted everything atomically
    blocks = canonical_db.get_blocks(book_id)
    sections = canonical_db.get_sections(book_id)
    assert len(blocks) == report["final_blocks"]
    assert len(sections) == report["sections_total"]

    # Book row was upgraded
    book_row = database.get_book(book_id)
    assert book_row["ingest_version"] == 2

    # Ingestion report stored and matches
    stored = canonical_db.get_ingestion_report(book_id)
    assert stored is not None
    assert stored["report"]["final_blocks"] == report["final_blocks"]


def test_ingest_real_pdf_end_to_end(temp_db):
    pdf_files = sorted(
        (Path(__file__).parent.parent / "uploads").glob("*.pdf")
    )
    if not pdf_files:
        pytest.skip("no real PDF in uploads/")

    book_id = "bk_real_pdf_e2e"
    _seed_book(book_id, str(pdf_files[0]))
    report = canonical_ingest.ingest_canonical(book_id, str(pdf_files[0]))

    assert report["locator_type"] == "pdf"
    assert report["pages_total"] > 0
    assert report["final_blocks"] > 0

    blocks = canonical_db.get_blocks(book_id)
    # Spot-check: every block has a 1-based page in its locator and a non-empty section_id
    for b in blocks[:50]:
        assert b["locator"]["page"] >= 1
        assert b["section_id"] is not None
        assert b["book_offset_end"] >= b["book_offset_start"]


def test_re_ingest_keeps_block_ids_stable(temp_db):
    """The critical contract: re-ingesting must produce identical block_ids.
    This is what guarantees existing citations stay valid across re-ingests.
    """
    epub_files = sorted(
        (Path(__file__).parent.parent / "uploads").glob("*.epub")
    )
    if not epub_files:
        pytest.skip("no real EPUB in uploads/")

    book_id = "bk_idempotent"
    _seed_book(book_id, str(epub_files[0]))

    canonical_ingest.ingest_canonical(book_id, str(epub_files[0]))
    ids_run1 = [b["block_id"] for b in canonical_db.get_blocks(book_id)]
    secs_run1 = [s["section_id"] for s in canonical_db.get_sections(book_id)]

    canonical_ingest.ingest_canonical(book_id, str(epub_files[0]))
    ids_run2 = [b["block_id"] for b in canonical_db.get_blocks(book_id)]
    secs_run2 = [s["section_id"] for s in canonical_db.get_sections(book_id)]

    assert ids_run1 == ids_run2
    assert secs_run1 == secs_run2

    # No orphans accumulated.
    assert len(ids_run2) == len(ids_run1)
