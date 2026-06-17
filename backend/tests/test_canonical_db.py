"""Tests for the canonical text DB layer (sections / blocks / reports).

These tests run against an isolated tempfile DB so they do not touch the
project's real sagespace.db.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

import core.database as database
from core.canonical import db as canonical_db
from core.canonical.ids import make_block_id, make_section_id
from core.canonical.models import Block, Section


@pytest.fixture
def temp_db(monkeypatch):
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    monkeypatch.setattr(database, "DB_PATH", Path(tmp.name))
    database.init_db()
    yield Path(tmp.name)
    Path(tmp.name).unlink(missing_ok=True)


def _make_book(book_id: str = "bk_test") -> str:
    # Manually insert a book row (bypass create_book to keep id stable for assertions).
    from core.database import get_conn

    conn = get_conn()
    conn.execute(
        "INSERT INTO books (id, title, upload_date, raptor_status, ingest_version) "
        "VALUES (?, ?, ?, ?, 1)",
        (book_id, "Test Book", "2026-01-01T00:00:00Z", "ready"),
    )
    conn.commit()
    conn.close()
    return book_id


def _fixture_sections_and_blocks(book_id: str) -> tuple[list[Section], list[Block]]:
    sec0 = Section(
        section_id=make_section_id(book_id, 0),
        book_id=book_id,
        order_idx=0,
        label="Chapter 1",
        level=1,
        source="outline",
    )
    sec1 = Section(
        section_id=make_section_id(book_id, 1),
        book_id=book_id,
        order_idx=1,
        label="Chapter 2",
        level=1,
        source="outline",
    )
    blocks = [
        Block(
            block_id=make_block_id(book_id, 0),
            book_id=book_id,
            order_idx=0,
            kind="heading",
            text="Chapter 1",
            book_offset_start=0,
            book_offset_end=9,
            locator_type="pdf",
            locator={"page": 1, "bbox": [0, 0, 100, 20]},
            section_id=sec0.section_id,
        ),
        Block(
            block_id=make_block_id(book_id, 1),
            book_id=book_id,
            order_idx=1,
            kind="paragraph",
            text="Hello world.",
            book_offset_start=10,
            book_offset_end=22,
            locator_type="pdf",
            locator={"page": 1, "bbox": [0, 30, 100, 50]},
            section_id=sec0.section_id,
            norm_flags={"dehyphenated": True},
        ),
        Block(
            block_id=make_block_id(book_id, 2),
            book_id=book_id,
            order_idx=2,
            kind="paragraph",
            text="Second chapter starts.",
            book_offset_start=23,
            book_offset_end=45,
            locator_type="pdf",
            locator={"page": 2, "bbox": [0, 0, 100, 20]},
            section_id=sec1.section_id,
        ),
    ]
    return [sec0, sec1], blocks


def test_block_id_is_stable_across_calls():
    a1 = make_block_id("bk_a", 42)
    a2 = make_block_id("bk_a", 42)
    b = make_block_id("bk_b", 42)
    a_other_idx = make_block_id("bk_a", 43)
    assert a1 == a2, "same (book_id, order_idx) must produce identical id"
    assert a1 != b, "different book_id must produce different id"
    assert a1 != a_other_idx, "different order_idx must produce different id"
    assert a1.startswith("blk_")


def test_replace_canonical_book_persists_and_reads_back(temp_db):
    book_id = _make_book()
    sections, blocks = _fixture_sections_and_blocks(book_id)
    canonical_db.replace_canonical_book(
        book_id, sections, blocks, report={"dropped_headers": 0}
    )

    # ingest_version was bumped to 2
    from core.database import get_book

    assert get_book(book_id)["ingest_version"] == 2

    # Reads return rows in order
    out_secs = canonical_db.get_sections(book_id)
    assert [s["order_idx"] for s in out_secs] == [0, 1]
    assert out_secs[0]["source"] == "outline"

    out_blocks = canonical_db.get_blocks(book_id)
    assert [b["order_idx"] for b in out_blocks] == [0, 1, 2]
    # Round-trip: locator JSON → dict, norm_flags JSON → dict
    assert out_blocks[1]["norm_flags"] == {"dehyphenated": True}
    assert out_blocks[0]["locator"] == {"page": 1, "bbox": [0, 0, 100, 20]}


def test_get_blocks_filters_by_section_and_offset(temp_db):
    book_id = _make_book()
    sections, blocks = _fixture_sections_and_blocks(book_id)
    canonical_db.replace_canonical_book(book_id, sections, blocks, report={})

    sec0_id = sections[0].section_id
    in_sec0 = canonical_db.get_blocks(book_id, section_id=sec0_id)
    assert [b["order_idx"] for b in in_sec0] == [0, 1]

    # Offset window that catches only the third block
    win = canonical_db.get_blocks(book_id, from_offset=23, to_offset=100)
    assert [b["order_idx"] for b in win] == [2]


def test_get_blocks_pagination_cursor(temp_db):
    book_id = _make_book()
    sections, blocks = _fixture_sections_and_blocks(book_id)
    canonical_db.replace_canonical_book(book_id, sections, blocks, report={})

    page1 = canonical_db.get_blocks(book_id, limit=2)
    assert [b["order_idx"] for b in page1] == [0, 1]
    last_idx = page1[-1]["order_idx"]
    page2 = canonical_db.get_blocks(book_id, limit=2, after_order_idx=last_idx)
    assert [b["order_idx"] for b in page2] == [2]


def test_replace_is_idempotent_and_block_ids_stay_stable(temp_db):
    """Re-ingesting the same book must produce the same block_ids and not
    leave orphan rows. This is the contract that lets citations survive
    re-ingest (see docs/ARCHITECTURE.md §canonical-refactor).
    """
    book_id = _make_book()
    sections, blocks = _fixture_sections_and_blocks(book_id)
    canonical_db.replace_canonical_book(book_id, sections, blocks, report={"run": 1})

    first_ids = [b["block_id"] for b in canonical_db.get_blocks(book_id)]

    # Re-run with the same logical input
    canonical_db.replace_canonical_book(book_id, sections, blocks, report={"run": 2})
    second_ids = [b["block_id"] for b in canonical_db.get_blocks(book_id)]
    assert first_ids == second_ids

    # No accumulation of orphan blocks / sections
    assert len(canonical_db.get_blocks(book_id)) == 3
    assert len(canonical_db.get_sections(book_id)) == 2

    # Report from the latest run survives, older one is replaced
    rep = canonical_db.get_ingestion_report(book_id)
    assert rep is not None
    assert rep["report"] == {"run": 2}


def test_replace_rolls_back_on_error(temp_db, monkeypatch):
    """If the write fails midway, the book's canonical state must be
    untouched (no partial section/block rows, ingest_version unchanged).
    """
    book_id = _make_book()
    sections, blocks = _fixture_sections_and_blocks(book_id)
    # Seed a valid state first
    canonical_db.replace_canonical_book(book_id, sections, blocks, report={"v": 1})

    bad_block = Block(
        block_id="dup",
        book_id=book_id,
        order_idx=0,
        kind="paragraph",
        text="x",
        book_offset_start=0,
        book_offset_end=1,
        locator_type="pdf",
        locator={},
    )
    # Two blocks with the same block_id will violate the PK and abort the txn
    with pytest.raises(Exception):
        canonical_db.replace_canonical_book(
            book_id, sections, [bad_block, bad_block], report={"v": 2}
        )

    # Original 3 blocks should still be there
    assert len(canonical_db.get_blocks(book_id)) == 3
    rep = canonical_db.get_ingestion_report(book_id)
    assert rep is not None and rep["report"] == {"v": 1}


# ── PR4: kind + printed_number persistence ──────────────────────────────────


def test_replace_persists_section_kind_and_printed_number(temp_db):
    """The normalizer fills (kind, printed_number) on every Section.
    The DB layer must round-trip both fields without loss; downstream
    get_chapter_summary depends on the values being queryable after a
    re-ingest."""
    book_id = _make_book("bk_kind")
    sections = [
        Section(
            section_id=make_section_id(book_id, 0),
            book_id=book_id,
            order_idx=0,
            label="Cover",
            level=1,
            source="outline",
            kind="cover",
            printed_number=None,
        ),
        Section(
            section_id=make_section_id(book_id, 1),
            book_id=book_id,
            order_idx=1,
            label="CHAPTER I",
            level=1,
            source="outline",
            kind="chapter",
            printed_number=1,
        ),
        Section(
            section_id=make_section_id(book_id, 2),
            book_id=book_id,
            order_idx=2,
            label="CHAPTER V",
            level=1,
            source="outline",
            kind="chapter",
            printed_number=5,
        ),
    ]
    canonical_db.replace_canonical_book(book_id, sections, [], report={})

    out = canonical_db.get_sections(book_id)
    assert [s["kind"] for s in out] == ["cover", "chapter", "chapter"]
    assert [s["printed_number"] for s in out] == [None, 1, 5]


def test_kind_defaults_to_other_when_normalizer_does_not_set_it(temp_db):
    """The Section dataclass defaults kind='other' and printed_number=None.
    Pre-PR4 normalizers that have not been updated must still produce
    insertable rows (DB schema has NOT NULL default 'other' on kind)."""
    book_id = _make_book("bk_default")
    # Construct WITHOUT explicit kind/printed_number; rely on dataclass defaults.
    sections = [
        Section(
            section_id=make_section_id(book_id, 0),
            book_id=book_id,
            order_idx=0,
            label="Some Heading",
            level=1,
            source="inferred",
        ),
    ]
    canonical_db.replace_canonical_book(book_id, sections, [], report={})
    out = canonical_db.get_sections(book_id)
    assert out[0]["kind"] == "other"
    assert out[0]["printed_number"] is None


def test_backfill_populates_kind_and_printed_number_on_existing_rows(tmp_path, monkeypatch):
    """Books ingested before PR4 should not require manual re-ingest.
    init_db's idempotent migration ADDs the columns AND backfills
    existing rows using classify_section_kind / parse_printed_number
    on the label."""
    from core import database as db_module
    import sqlite3

    # Build a "legacy" DB without kind / printed_number columns.
    db_path = tmp_path / "legacy.db"
    monkeypatch.setattr(db_module, "DB_PATH", db_path)
    # Build only the columns that existed pre-PR4.
    legacy_conn = sqlite3.connect(str(db_path))
    legacy_conn.row_factory = sqlite3.Row
    legacy_conn.executescript("""
        CREATE TABLE books (
            id TEXT PRIMARY KEY, title TEXT NOT NULL, author TEXT,
            total_chunks INTEGER, total_chapters INTEGER,
            upload_date TEXT, raptor_status TEXT DEFAULT 'pending',
            file_path TEXT, ingest_version INTEGER NOT NULL DEFAULT 1
        );
        CREATE TABLE sections (
            section_id TEXT PRIMARY KEY,
            book_id TEXT NOT NULL,
            parent_section_id TEXT,
            order_idx INTEGER NOT NULL,
            level INTEGER NOT NULL DEFAULT 1,
            label TEXT NOT NULL,
            source TEXT NOT NULL DEFAULT 'inferred'
        );
    """)
    legacy_conn.execute(
        "INSERT INTO books (id, title, upload_date, ingest_version) VALUES (?,?,?,?)",
        ("bk_old", "Alice", "2026-01-01T00:00:00Z", 2),
    )
    legacy_conn.executemany(
        "INSERT INTO sections (section_id, book_id, order_idx, label) VALUES (?,?,?,?)",
        [
            ("sec_0", "bk_old", 0, "Cover"),
            ("sec_1", "bk_old", 1, "CHAPTER I. Down the Rabbit-Hole"),
            ("sec_2", "bk_old", 2, "CHAPTER II. The Pool of Tears"),
            ("sec_3", "bk_old", 3, "CHAPTER V. Advice from a Caterpillar"),
            ("sec_4", "bk_old", 4, "Appendix A"),
        ],
    )
    legacy_conn.commit()
    legacy_conn.close()

    # Now run init_db on the legacy DB -- should ALTER + backfill.
    db_module.init_db()

    out = canonical_db.get_sections("bk_old")
    by_label = {s["label"]: s for s in out}
    assert by_label["Cover"]["kind"] == "front_matter"
    assert by_label["Cover"]["printed_number"] is None
    assert by_label["CHAPTER I. Down the Rabbit-Hole"]["kind"] == "chapter"
    assert by_label["CHAPTER I. Down the Rabbit-Hole"]["printed_number"] == 1
    assert by_label["CHAPTER V. Advice from a Caterpillar"]["kind"] == "chapter"
    assert by_label["CHAPTER V. Advice from a Caterpillar"]["printed_number"] == 5
    assert by_label["Appendix A"]["kind"] == "back_matter"
