"""Tests for the FTS5 keyword index + search (PR2 step 1).

Covers the `keyword_search` engine end to end at the SQLite layer:
  * the FTS5 index finds exact entities and orders by relevance,
  * per-book isolation (book_id filter),
  * populate is a clean rebuild (re-ingest replaces stale rows),
  * the init_db backfill fills pre-PR2 books, and delete_book clears it,
  * the match-query builder is injection-safe against FTS5 operators.
"""
import sqlite3

import pytest

import core.database as db
from core import keyword_search as ks


@pytest.fixture
def isolated_db(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "test.db")
    db.init_db()


def _seed_fts(book_id, blocks):
    """blocks: list[(block_id, text)] -> populate blocks_fts directly."""
    conn = db.get_conn()
    db.populate_blocks_fts(conn, book_id, [(bid, book_id, txt) for bid, txt in blocks])
    conn.commit()
    conn.close()


def _insert_block(book_id, block_id, text, order_idx=0):
    conn = db.get_conn()
    conn.execute("INSERT OR IGNORE INTO books (id, title) VALUES (?, ?)", (book_id, "T"))
    conn.execute(
        """INSERT INTO blocks
           (block_id, book_id, section_id, order_idx, kind, text,
            book_offset_start, book_offset_end, locator_type, locator_json)
           VALUES (?, ?, NULL, ?, 'paragraph', ?, 0, 0, 'pdf', '{}')""",
        (block_id, book_id, order_idx, text),
    )
    conn.commit()
    conn.close()


# ── search ──────────────────────────────────────────────────────────────────


def test_finds_exact_entity(isolated_db):
    _seed_fts("bk", [
        ("b1", "the Cheshire Cat grinned at Alice from a tree branch"),
        ("b2", "the Queen of Hearts shouted off with their heads"),
    ])
    hits = ks.search_blocks_fts("bk", "Cheshire Cat")
    assert [h["block_id"] for h in hits] == ["b1"]
    assert "Cat" in hits[0]["text"]
    assert "[" in hits[0]["snippet"]  # snippet highlights the match


def test_and_semantics_requires_all_terms(isolated_db):
    _seed_fts("bk", [
        ("b1", "Alice met the Cheshire Cat"),
        ("b2", "Alice met the white rabbit"),
    ])
    # both terms must be present -> only b1
    assert [h["block_id"] for h in ks.search_blocks_fts("bk", "Cheshire Cat")] == ["b1"]
    # single term still works
    assert {h["block_id"] for h in ks.search_blocks_fts("bk", "Alice")} == {"b1", "b2"}


def test_book_isolation(isolated_db):
    _seed_fts("bk1", [("a", "the Cheshire Cat")])
    _seed_fts("bk2", [("b", "the Cheshire Cat")])
    assert [h["block_id"] for h in ks.search_blocks_fts("bk1", "Cheshire")] == ["a"]
    assert [h["block_id"] for h in ks.search_blocks_fts("bk2", "Cheshire")] == ["b"]


def test_no_match_returns_empty(isolated_db):
    _seed_fts("bk", [("b1", "Alice in Wonderland")])
    assert ks.search_blocks_fts("bk", "Gandalf") == []


def test_limit_caps_results(isolated_db):
    _seed_fts("bk", [(f"b{i}", "Alice falls down the hole") for i in range(20)])
    assert len(ks.search_blocks_fts("bk", "Alice", limit=5)) == 5


# ── populate / rebuild ──────────────────────────────────────────────────────


def test_populate_is_idempotent_rebuild(isolated_db):
    _seed_fts("bk", [("b1", "old text alpha")])
    _seed_fts("bk", [("b1", "new text beta")])  # re-populate same book
    assert ks.search_blocks_fts("bk", "alpha") == []
    assert [h["block_id"] for h in ks.search_blocks_fts("bk", "beta")] == ["b1"]


# ── backfill + delete cascade ───────────────────────────────────────────────


def test_init_db_backfills_existing_blocks(isolated_db):
    # Insert a block WITHOUT touching FTS (simulates a pre-PR2 book).
    _insert_block("bk", "b1", "the Mad Hatter hosted a tea party")
    conn = db.get_conn()
    conn.execute("DELETE FROM blocks_fts WHERE book_id = 'bk'")  # ensure not indexed
    conn.commit()
    conn.close()
    assert ks.search_blocks_fts("bk", "Hatter") == []
    db.init_db()  # runs _backfill_blocks_fts
    assert [h["block_id"] for h in ks.search_blocks_fts("bk", "Hatter")] == ["b1"]


def test_delete_book_clears_fts(isolated_db):
    _insert_block("bk", "b1", "the Cheshire Cat")
    _seed_fts("bk", [("b1", "the Cheshire Cat")])
    assert ks.search_blocks_fts("bk", "Cheshire")
    db.delete_book("bk")
    assert ks.search_blocks_fts("bk", "Cheshire") == []


# ── injection safety ────────────────────────────────────────────────────────


@pytest.mark.parametrize("bad", [
    "", "   ", '"', '*', 'Cat)', '(Cat', 'Cat AND', 'cat: "grin"',
    'NEAR(a b)', 'a OR b', '^Cat', 'Cat-Hatter', '"""',
])
def test_malformed_queries_never_raise(isolated_db, bad):
    _seed_fts("bk", [("b1", "the Cat (grinning) said: hi to the Hatter")])
    out = ks.search_blocks_fts("bk", bad)
    assert isinstance(out, list)  # graceful, never raises


def test_operator_like_input_still_matches_literal_words(isolated_db):
    _seed_fts("bk", [("b1", "the Cat and the Hatter")])
    # "Cat AND Hatter" must be treated as literal words, not an FTS5 boolean
    # expression that errors -- and it should find the block.
    assert [h["block_id"] for h in ks.search_blocks_fts("bk", "Cat AND Hatter")] == ["b1"]


def test_build_match_query_quotes_and_ands_tokens():
    assert ks._build_match_query("Cheshire Cat") == '"Cheshire" AND "Cat"'
    assert ks._build_match_query("  ") is None
    assert ks._build_match_query('"); DROP') == '"DROP"'  # operators stripped/quoted
