"""Tests for the /api/ingest endpoint's input sanitization.

The actual EPUB/PDF indexing pipeline is covered by test_canonical_*
(real fixtures, end-to-end). These tests focus on what /ingest does to
the user-supplied title and author *before* it hands off to the
background indexer: cap length, strip control chars, sanitize the
filename-stem fallback so it can't bypass the cleaning either.
"""
from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient

import core.database as db_module
from core import database as db


@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    monkeypatch.setattr(db_module, "DB_PATH", tmp_path / "test.db")
    db_module.init_db()


@pytest.fixture
def client(tmp_path, monkeypatch):
    """A TestClient that:
      * routes uploads into a tmp dir (no real on-disk pollution);
      * mocks the heavy background indexer (we only care what /ingest
        wrote BEFORE the BackgroundTask runs)."""
    from api import ingest as ingest_module
    monkeypatch.setattr(ingest_module, "UPLOAD_DIR", tmp_path / "uploads")
    (tmp_path / "uploads").mkdir()
    monkeypatch.setattr(ingest_module, "_build_index", lambda *a, **kw: None)

    from main import app
    return TestClient(app)


def _upload(client: TestClient, **fields):
    """Post a tiny dummy EPUB so the endpoint accepts it; the indexer
    is mocked out so contents don't matter."""
    files = {"file": ("book.epub", io.BytesIO(b"dummy"), "application/epub+zip")}
    return client.post("/api/ingest", files=files, data=fields)


def _book_title(book_id: str) -> str:
    row = db.get_book(book_id)
    return (row or {}).get("title", "")


def _book_author(book_id: str) -> str | None:
    row = db.get_book(book_id)
    return (row or {}).get("author")


# ── length cap ────────────────────────────────────────────────────────────


def test_title_longer_than_cap_is_truncated(client):
    long_title = "T" * 1000
    res = _upload(client, title=long_title, author="A")
    assert res.status_code == 200
    title = _book_title(res.json()["book_id"])
    assert len(title) == 200
    assert title == "T" * 200


def test_author_longer_than_cap_is_truncated(client):
    res = _upload(client, title="Book", author="A" * 1000)
    author = _book_author(res.json()["book_id"])
    assert author is not None and len(author) == 200


# ── whitespace + control chars ────────────────────────────────────────────


def test_title_strips_surrounding_whitespace(client):
    res = _upload(client, title="  Alice in Wonderland\n", author="Lewis Carroll")
    assert _book_title(res.json()["book_id"]) == "Alice in Wonderland"


def test_control_chars_are_dropped(client):
    """Newlines / NULs in the title are a log-injection + XSS vector
    (Next.js escapes JSX-rendered text, but the defense-in-depth rule
    is sanitize at the boundary)."""
    res = _upload(client, title="Alice\x00<script>alert(1)</script>\r\nLF",
                  author="A\x07thor")
    title = _book_title(res.json()["book_id"])
    assert "\x00" not in title and "\r" not in title and "\n" not in title
    # Bracketed text survives (Next.js escapes it at render); only the
    # control bytes are stripped here.
    assert title == "Alice<script>alert(1)</script>LF"
    assert _book_author(res.json()["book_id"]) == "Athor"


# ── filename-stem fallback ────────────────────────────────────────────────


def test_filename_stem_fallback_is_also_sanitized(client):
    """The fallback path (empty title -> filename stem) must run the
    same cleaning — the upload filename is user-controlled too."""
    files = {"file": ("  Alice\x00<x>\n  .epub",
                      io.BytesIO(b"dummy"),
                      "application/epub+zip")}
    res = client.post("/api/ingest", files=files, data={"title": "", "author": ""})
    assert res.status_code == 200
    title = _book_title(res.json()["book_id"])
    assert "\x00" not in title and "\n" not in title
    assert title.strip() == title and title  # non-empty after strip


def test_all_whitespace_title_falls_through_to_filename_then_untitled(client):
    """A title that is purely whitespace/control chars (== empty after
    sanitization) must NOT be saved as-is. We fall through to the
    filename stem, and if THAT is also empty, to a literal 'Untitled'."""
    files = {"file": ("\x00\x00.epub", io.BytesIO(b"dummy"),
                      "application/epub+zip")}
    res = client.post("/api/ingest", files=files,
                      data={"title": "   \n\t\x00   ", "author": ""})
    assert res.status_code == 200
    assert _book_title(res.json()["book_id"]) == "Untitled"


# ── empty author is preserved as NULL (not "") ────────────────────────────


def test_empty_author_persists_as_none(client):
    res = _upload(client, title="Book", author="")
    assert _book_author(res.json()["book_id"]) is None


def test_whitespace_only_author_persists_as_none(client):
    res = _upload(client, title="Book", author="   \n\t   ")
    assert _book_author(res.json()["book_id"]) is None
