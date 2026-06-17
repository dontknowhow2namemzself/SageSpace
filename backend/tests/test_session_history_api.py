"""Tests for the session-history sidebar API:

  GET    /api/chat/sessions/{book_id}   - list summaries, newest first
  DELETE /api/chat/session/{session_id} - delete one conversation

Delete semantics under test: the session row and its retrieval_events go,
but retrieved_chunks rows are KEPT — digested progress is a property of
the book, deleting a chat must not un-read it.
"""
import json

import pytest
from fastapi.testclient import TestClient

import core.database as db_module
from core import database as db


@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    monkeypatch.setattr(db_module, "DB_PATH", tmp_path / "test.db")
    db_module.init_db()


@pytest.fixture
def client():
    from main import app
    return TestClient(app)


def _make_book() -> str:
    book_id = db.create_book("History Book", "Author", "/tmp/h.pdf")
    db.update_book_status(book_id, "ready", total_chunks=10, total_chapters=2)
    return book_id


def _set_start_time(session_id: str, iso: str) -> None:
    conn = db.get_conn()
    conn.execute("UPDATE sessions SET start_time=? WHERE id=?", (iso, session_id))
    conn.commit()
    conn.close()


# ── GET /api/chat/sessions/{book_id} ───────────────────────────────────────


def test_list_sessions_unknown_book_404(client):
    assert client.get("/api/chat/sessions/nope").status_code == 404


def _seed_conversation(session_id: str, first_question: str) -> None:
    db.save_conversation(session_id, json.dumps([
        {"role": "user", "content": first_question},
        {"role": "assistant", "content": "It is..."},
    ]))


def test_list_sessions_newest_first_with_previews(client):
    book_id = _make_book()
    s_old = db.create_session(book_id)
    s_new = db.create_session(book_id)
    _set_start_time(s_old, "2026-06-01T10:00:00+00:00")
    _set_start_time(s_new, "2026-06-09T10:00:00+00:00")
    _seed_conversation(s_old, "What is saponification?")
    _seed_conversation(s_new, "Who wrote this?")

    res = client.get(f"/api/chat/sessions/{book_id}")
    assert res.status_code == 200
    sessions = res.json()["sessions"]
    assert [s["id"] for s in sessions] == [s_new, s_old]

    newest, oldest = sessions
    assert newest["preview"] == "Who wrote this?"
    assert oldest["message_count"] == 2
    assert oldest["preview"] == "What is saponification?"


def test_list_sessions_excludes_never_used_sessions(client):
    """Page loads / eval runs create sessions freely; ones that never got
    a message are not history and must not reach the sidebar."""
    book_id = _make_book()
    s_used = db.create_session(book_id)
    db.create_session(book_id)  # never used
    _seed_conversation(s_used, "Hello?")

    sessions = client.get(f"/api/chat/sessions/{book_id}").json()["sessions"]
    assert [s["id"] for s in sessions] == [s_used]


def test_list_sessions_scoped_to_book(client):
    book_a, book_b = _make_book(), _make_book()
    sid_a = db.create_session(book_a)
    sid_b = db.create_session(book_b)
    _seed_conversation(sid_a, "About book A")
    _seed_conversation(sid_b, "About book B")

    sessions = client.get(f"/api/chat/sessions/{book_a}").json()["sessions"]
    assert [s["id"] for s in sessions] == [sid_a]


# ── DELETE /api/chat/session/{session_id} ──────────────────────────────────


def test_delete_session_unknown_404(client):
    assert client.delete("/api/chat/session/nope").status_code == 404


def test_delete_session_removes_row_and_events_keeps_progress(client):
    book_id = _make_book()
    session_id = db.create_session(book_id)
    keep_id = db.create_session(book_id)

    db.record_retrieved_chunks(session_id, book_id, ["chk_1", "chk_2"])
    event_id = db.create_retrieval_event(
        session_id=session_id, book_id=book_id,
        query_text="q", multi_query_variants_json="[]", hyde_hypothesis="",
        raw_hits_count=2, new_raw_hits_count=2, summary_hits_count=0,
    )
    db.add_event_chunks(event_id, [
        {"chunk_id": "chk_1", "raptor_level": 0, "chapter": 1, "page": 1,
         "rank": 1, "origin": "multi_query", "is_new_lighting": 1,
         "preview_text": "p"},
    ])

    res = client.delete(f"/api/chat/session/{session_id}")
    assert res.status_code == 200
    assert res.json() == {"deleted": session_id}

    # Session row + its retrieval events are gone; the sibling survives.
    assert db.get_session(session_id) is None
    assert db.get_session(keep_id) is not None
    assert db.get_retrieval_events(session_id) == []
    assert db.get_event_chunks(event_id) == []

    # Digested progress is preserved: the book still counts the lit chunks.
    assert set(db.get_all_retrieved_chunk_ids_for_book(book_id)) == {"chk_1", "chk_2"}


def test_delete_session_gcs_langgraph_checkpoints(client, monkeypatch):
    book_id = _make_book()
    session_id = db.create_session(book_id)

    cleared: list[str] = []
    import core.graph.build as build_module
    monkeypatch.setattr(
        build_module, "gc_checkpoints_for_session",
        lambda sid: cleared.append(sid) or True,
    )

    assert client.delete(f"/api/chat/session/{session_id}").status_code == 200
    assert cleared == [session_id]
