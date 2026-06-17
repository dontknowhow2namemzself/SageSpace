"""Tests for the fast-lane memory_notes DB layer (design §A).

Covers add_memory_note (dedup + empty-skip + provenance), list ordering, the
cross-session recent_user_questions reader (the "chat-messages table" is really
sessions.conversation_json), and library_titles.
"""
import json
import sqlite3

import pytest

import core.database as db_module
from core.database import (
    init_db,
    add_memory_note,
    list_memory_notes,
    get_memory_note,
    update_memory_note,
    delete_memory_note,
    recent_user_questions,
    library_titles,
    create_book,
    create_session,
    save_conversation,
)


@pytest.fixture(autouse=True)
def use_test_db(tmp_path, monkeypatch):
    monkeypatch.setattr(db_module, "DB_PATH", tmp_path / "test.db")
    init_db()


def _exec(sql, params=()):
    conn = sqlite3.connect(str(db_module.DB_PATH))
    conn.execute(sql, params)
    conn.commit()
    conn.close()


def _set_start_time(session_id: str, iso: str):
    _exec("UPDATE sessions SET start_time=? WHERE id=?", (iso, session_id))


# ── add_memory_note ──────────────────────────────────────────────────────────


def test_add_memory_note_returns_id_and_persists():
    note_id = add_memory_note("用户希望被称为小王", type="fact")
    assert note_id
    notes = list_memory_notes()
    assert len(notes) == 1
    assert notes[0]["text"] == "用户希望被称为小王"
    assert notes[0]["type"] == "fact"


def test_add_memory_note_dedups_identical_text():
    first = add_memory_note("用户在研读斯多葛哲学", type="interest")
    second = add_memory_note("用户在研读斯多葛哲学", type="interest")
    assert first is not None
    assert second is None  # dedup -> skipped, not a second row
    assert len(list_memory_notes()) == 1


def test_add_memory_note_skips_empty_and_whitespace():
    assert add_memory_note("", type="fact") is None
    assert add_memory_note("   ", type="fact") is None
    assert list_memory_notes() == []


def test_add_memory_note_strips_whitespace():
    add_memory_note("  小王  ", type="fact")
    assert list_memory_notes()[0]["text"] == "小王"


def test_add_memory_note_records_provenance():
    book_id = create_book("Meditations", "Aurelius", "/tmp/m.pdf")
    add_memory_note(
        "用户在研读斯多葛", type="interest",
        source_book_id=book_id, source_locator="sess-1",
    )
    note = list_memory_notes()[0]
    assert note["source_book_id"] == book_id
    assert note["source_locator"] == "sess-1"


# ── list ordering ────────────────────────────────────────────────────────────


def test_list_memory_notes_newest_first():
    add_memory_note("first", type="fact")
    add_memory_note("second", type="fact")
    # The two inserts can share a microsecond; pin created_at to assert DESC.
    _exec("UPDATE memory_notes SET created_at='2026-01-01T00:00:00' WHERE text='first'")
    _exec("UPDATE memory_notes SET created_at='2026-02-01T00:00:00' WHERE text='second'")
    assert [n["text"] for n in list_memory_notes()] == ["second", "first"]


# ── recent_user_questions (cross-session conversation_json reader) ────────────


def test_recent_user_questions_empty_when_no_sessions():
    assert recent_user_questions() == []


def test_recent_user_questions_extracts_user_turns_newest_first():
    book_id = create_book("B", None, "/tmp/b.pdf")
    sid = create_session(book_id)
    save_conversation(sid, json.dumps([
        {"role": "user", "content": "Q1"},
        {"role": "assistant", "content": "A1"},
        {"role": "user", "content": "Q2"},
    ]))
    # User turns only, reversed within the session (tail is newest).
    assert recent_user_questions() == ["Q2", "Q1"]


def test_recent_user_questions_orders_newest_session_first():
    book_id = create_book("B", None, "/tmp/b.pdf")
    old = create_session(book_id)
    _set_start_time(old, "2026-01-01T00:00:00")
    new = create_session(book_id)
    _set_start_time(new, "2026-03-01T00:00:00")
    save_conversation(old, json.dumps([{"role": "user", "content": "old-q"}]))
    save_conversation(new, json.dumps([{"role": "user", "content": "new-q"}]))
    assert recent_user_questions() == ["new-q", "old-q"]


def test_recent_user_questions_respects_limit():
    book_id = create_book("B", None, "/tmp/b.pdf")
    sid = create_session(book_id)
    save_conversation(sid, json.dumps(
        [{"role": "user", "content": f"q{i}"} for i in range(20)]
    ))
    qs = recent_user_questions(limit=5)
    assert len(qs) == 5
    assert qs[0] == "q19"  # newest first


def test_recent_user_questions_skips_blank_and_bad_json():
    book_id = create_book("B", None, "/tmp/b.pdf")
    good = create_session(book_id)
    _set_start_time(good, "2026-02-01T00:00:00")
    bad = create_session(book_id)
    _set_start_time(bad, "2026-01-01T00:00:00")
    save_conversation(good, json.dumps([
        {"role": "user", "content": "  "},   # blank -> skipped
        {"role": "user", "content": "real"},
    ]))
    _exec("UPDATE sessions SET conversation_json='not valid json' WHERE id=?", (bad,))
    assert recent_user_questions() == ["real"]


# ── library_titles ───────────────────────────────────────────────────────────


def test_library_titles():
    create_book("Alice", "Carroll", "/tmp/a.pdf")
    create_book("Meditations", "Aurelius", "/tmp/m.pdf")
    assert set(library_titles()) == {"Alice", "Meditations"}


# ── finalize capture path (design §A) ────────────────────────────────────────


def test_capture_memory_note_writes_from_state():
    from core.graph.nodes import _capture_memory_note
    from core.pipeline.types import IntentDecision

    book_id = create_book("B", None, "/tmp/b.pdf")
    state = {
        "intent": IntentDecision(
            kind="smalltalk", memory_note="用户叫小王", memory_note_type="fact"
        ),
        "book_id": book_id,
        "session_id": "sess-x",
    }
    _capture_memory_note(state)
    notes = list_memory_notes()
    assert len(notes) == 1
    assert notes[0]["text"] == "用户叫小王"
    assert notes[0]["type"] == "fact"
    assert notes[0]["source_book_id"] == book_id
    assert notes[0]["source_locator"] == "sess-x"


def test_capture_memory_note_noop_when_no_note():
    from core.graph.nodes import _capture_memory_note
    from core.pipeline.types import IntentDecision

    state = {
        "intent": IntentDecision(kind="search", search_query="x"),
        "book_id": "b", "session_id": "s",
    }
    _capture_memory_note(state)  # memory_note defaults to None -> no write
    assert list_memory_notes() == []


# ── get / update / delete ("What I remember" panel) ──────────────────────────


def test_get_memory_note():
    nid = add_memory_note("hello", type="fact")
    assert get_memory_note(nid)["text"] == "hello"
    assert get_memory_note("nope") is None


def test_update_memory_note_text():
    nid = add_memory_note("old text", type="fact")
    assert update_memory_note(nid, "new text") is True
    assert get_memory_note(nid)["text"] == "new text"


def test_update_memory_note_can_change_type():
    nid = add_memory_note("x", type="fact")
    update_memory_note(nid, "x", type="interest")
    assert get_memory_note(nid)["type"] == "interest"


def test_update_memory_note_ignores_invalid_type():
    nid = add_memory_note("x", type="fact")
    update_memory_note(nid, "x2", type="bogus")  # invalid type -> kept as-is
    note = get_memory_note(nid)
    assert note["text"] == "x2"
    assert note["type"] == "fact"


def test_update_memory_note_empty_text_rejected():
    nid = add_memory_note("keep", type="fact")
    assert update_memory_note(nid, "   ") is False
    assert get_memory_note(nid)["text"] == "keep"  # unchanged


def test_update_memory_note_missing_returns_false():
    assert update_memory_note("ghost", "x") is False


def test_delete_memory_note():
    nid = add_memory_note("bye", type="fact")
    assert delete_memory_note(nid) is True
    assert get_memory_note(nid) is None
    assert delete_memory_note(nid) is False  # already gone -> API 404
