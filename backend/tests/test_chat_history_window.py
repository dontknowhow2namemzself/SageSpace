"""Sliding-window regression: long sessions must not blow up the
LangGraph state / per-turn checkpoint.

api/chat.py POST /chat loads sessions.conversation_json (full archive,
unbounded by design — used for sidebar restore) and feeds it into a
new turn's init_state. Only the LAST `_HISTORY_WINDOW_MESSAGES` enter
the working set; the full archive stays in the DB."""
from __future__ import annotations

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
def book_and_session():
    bid = db.create_book("Book", "Author", "/tmp/x.epub")
    db.update_book_status(bid, "ready", total_chunks=10, total_chapters=1)
    sid = db.create_session(bid)
    return bid, sid


def _seed_history(session_id: str, n_turns: int):
    """Seed n_turns user+assistant pairs into conversation_json."""
    history = []
    for i in range(n_turns):
        history.append({"role": "user", "content": f"Q{i}"})
        history.append({"role": "assistant", "content": f"A{i}"})
    db.save_conversation(session_id, json.dumps(history))


@pytest.fixture
def client_capturing_init_state(monkeypatch):
    """Replace the chat graph with a stub whose .stream captures
    init_state so we can assert what the LangGraph state actually saw."""
    captured: dict = {}

    class _FakeGraph:
        def stream(self, init, config, stream_mode):
            captured["init"] = init
            return iter(())  # no frames; the SSE response will be empty

        def get_state(self, config):  # not paused
            class _S:
                next = ()
                tasks = ()
            return _S()

    import api.chat as chat_module
    monkeypatch.setattr(chat_module, "get_chat_graph", lambda: _FakeGraph())
    monkeypatch.setattr(chat_module, "persist_usage_from_callback",
                        lambda *a, **kw: None)

    from main import app
    return TestClient(app), captured


# ── what enters the LangGraph state ────────────────────────────────────────


def test_init_state_history_capped_to_window(
    client_capturing_init_state, book_and_session
):
    """A 50-turn conversation feeds only the last
    _HISTORY_WINDOW_MESSAGES into the graph state."""
    client, captured = client_capturing_init_state
    book_id, session_id = book_and_session
    _seed_history(session_id, n_turns=50)

    resp = client.post(
        "/api/chat",
        json={"book_id": book_id, "session_id": session_id, "message": "next"},
    )
    list(resp.iter_lines())  # drain

    from api.chat import _HISTORY_WINDOW_MESSAGES
    init_history = captured["init"]["history"]
    assert len(init_history) == _HISTORY_WINDOW_MESSAGES
    # It is the MOST RECENT messages, not the first ones.
    assert init_history[-1] == {"role": "assistant", "content": "A49"}
    assert init_history[0]["content"].startswith(("Q", "A"))


def test_init_state_history_short_session_unchanged(
    client_capturing_init_state, book_and_session
):
    """A 3-turn conversation passes through untouched: the cap is
    last-N, not exactly-N."""
    client, captured = client_capturing_init_state
    book_id, session_id = book_and_session
    _seed_history(session_id, n_turns=3)

    client.post(
        "/api/chat",
        json={"book_id": book_id, "session_id": session_id, "message": "next"},
    ).iter_lines()

    assert len(captured["init"]["history"]) == 6


# ── what stays in the DB ───────────────────────────────────────────────────


def test_full_conversation_archive_survives_in_db(
    client_capturing_init_state, book_and_session
):
    """The cap is applied at the LangGraph state boundary only — the DB
    archive (sessions.conversation_json) is what the sidebar restores
    from and must stay complete."""
    client, _ = client_capturing_init_state
    book_id, session_id = book_and_session
    _seed_history(session_id, n_turns=50)

    client.post(
        "/api/chat",
        json={"book_id": book_id, "session_id": session_id, "message": "next"},
    ).iter_lines()

    stored = json.loads(db.get_session(session_id)["conversation_json"])
    assert len(stored) == 100  # 50 turns × 2 messages — full archive
    assert stored[0]["content"] == "Q0"
    assert stored[-1]["content"] == "A49"
