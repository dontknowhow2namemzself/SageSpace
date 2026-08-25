"""Tests for the plain-function tools (compute_reading_progress, run_export).

PR5 stripped @tool / ContextVar plumbing from core/tools.py. These
helpers are now ordinary Python functions taking explicit
(book_id, session_id, ...) args; the pipeline's chat orchestrator
calls them directly for the reading_progress / export_notes intents.

Tests cover:
  * compute_reading_progress with valid + missing inputs
  * run_export markdown + pdf paths and the empty-history skip case
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import core.database as db_module
import core.tools as tools_module
from core.tools import compute_reading_progress, run_export


@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    monkeypatch.setattr(db_module, "DB_PATH", tmp_path / "test.db")
    monkeypatch.setattr(tools_module, "DATA_DIR", tmp_path)
    db_module.init_db()


@pytest.fixture
def book_and_session():
    book_id = db_module.create_book("Test Book", "Author", "/tmp/t.pdf")
    db_module.update_book_status(book_id, "ready", total_chunks=100, total_chapters=5)
    session_id = db_module.create_session(book_id)
    # Progress speaks CITED (per-fact attribution), not retrieved: the
    # retrieved ledger below must not count toward the percentage.
    db_module.record_retrieved_chunks(
        session_id, book_id, [f"chunk_{i:04d}" for i in range(80)]
    )
    db_module.record_cited_chunks(
        session_id, book_id, [f"chunk_{i:04d}" for i in range(30)]
    )
    return book_id, session_id


# ── compute_reading_progress ───────────────────────────────────────────────


def test_compute_reading_progress_returns_percentage(book_and_session):
    book_id, session_id = book_and_session
    data = compute_reading_progress(book_id, session_id)
    assert data["digested_pct"] == "30.0%"  # 30 cited, NOT 80 retrieved
    assert data["total_chunks"] == 100
    assert data["cited_chunk_count"] == 30
    assert data["available"] is True


def test_compute_reading_progress_returns_safe_defaults_when_missing():
    """Bad inputs MUST NOT raise -- the synthesizer downgrades to a
    polite 'no progress data' reply when available=False."""
    data = compute_reading_progress("nonexistent_book", "nonexistent_session")
    assert data["available"] is False
    assert data["digested_pct"] == "0.0%"
    assert data["total_chunks"] == 0


# ── run_export ────────────────────────────────────────────────────────────


def test_run_export_markdown(book_and_session, tmp_path, monkeypatch):
    book_id, session_id = book_and_session
    history = [
        {"role": "user", "content": "What is this book about?"},
        {"role": "assistant", "content": "It is about Zen and motorcycle maintenance."},
    ]
    db_module.save_conversation(session_id, json.dumps(history))

    result = run_export(book_id, session_id, format="markdown")
    assert result["available"] is True
    assert result["format"] == "markdown"
    assert result["path"].endswith(".md")


def test_run_export_skips_when_history_empty(book_and_session):
    """Trying to export an empty session must not write a file or
    error out -- the synthesizer renders a polite 'nothing to export'
    line based on available=False."""
    book_id, session_id = book_and_session
    # No save_conversation -> conversation_json is the default empty []
    result = run_export(book_id, session_id, format="markdown")
    assert result["available"] is False
    assert result["reason"] == "empty_history"
    assert result["path"] == ""


def test_run_export_skips_when_session_missing():
    result = run_export("any_book", "no_such_session", format="markdown")
    assert result["available"] is False
    assert result["reason"] == "session_not_found"
