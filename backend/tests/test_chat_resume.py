"""Tests for the chat interrupt/resume HTTP seam (PR3 step 2).

Two layers:
  * unit tests for the chat.py interrupt helpers (TTL, payload extraction),
  * a TestClient integration of the full flow — POST /chat halts with an
    `ask_user` frame, POST /chat/resume continues the turn — plus the
    expiry safe-degrade and the no-pending stale-tolerance.

The node internals are mocked at core.graph.nodes so the real compiled
graph + SqliteSaver run offline.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document

import core.database as db_module
from core.pipeline.types import AnswerDraft, IntentDecision, RetrievalResult


# ── unit: interrupt helpers ──────────────────────────────────────────────────


class _Intr:
    def __init__(self, value):
        self.value = value


class _Task:
    def __init__(self, interrupts):
        self.interrupts = interrupts


class _Snap:
    def __init__(self, next_=(), tasks=(), created_at=None):
        self.next = next_
        self.tasks = tasks
        self.created_at = created_at


def test_interrupt_value_extracts_payload():
    from api.chat import _interrupt_value
    assert _interrupt_value(_Snap(tasks=[_Task(())])) is None
    val = {"kind": "clarify", "prompt": "who?"}
    assert _interrupt_value(_Snap(tasks=[_Task((_Intr(val),))])) == val


def test_pending_ask_user_wraps_payload():
    from api.chat import _pending_ask_user
    paused = _Snap(next_=("clarify",), tasks=[_Task((_Intr({"kind": "clarify", "prompt": "q"}),))])
    done = _Snap(next_=(), tasks=[])

    class _G:
        def __init__(self, snap):
            self._snap = snap

        def get_state(self, cfg):
            return self._snap

    assert _pending_ask_user(_G(paused), {}) == {"type": "ask_user", "kind": "clarify", "prompt": "q"}
    assert _pending_ask_user(_G(done), {}) is None


def test_interrupt_expired_uses_absolute_30min():
    from api.chat import _interrupt_expired
    now = datetime.now(timezone.utc)
    assert _interrupt_expired(_Snap(created_at=(now - timedelta(minutes=10)).isoformat())) is False
    assert _interrupt_expired(_Snap(created_at=(now - timedelta(minutes=40)).isoformat())) is True
    # lenient on missing / unparseable timestamps -> not expired
    assert _interrupt_expired(_Snap(created_at=None)) is False
    assert _interrupt_expired(_Snap(created_at="garbage")) is False


# ── integration: /chat -> ask_user -> /chat/resume ──────────────────────────


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(db_module, "DB_PATH", tmp_path / "test.db")
    import core.graph.build as build_mod
    monkeypatch.setattr(build_mod, "_graph", None)  # fresh graph w/ temp checkpoint DB
    from main import app
    db_module.init_db()
    return TestClient(app)


def _ready_book_session():
    bid = db_module.create_book("Alice", "Carroll", "/tmp/a.pdf")
    db_module.update_book_status(bid, "ready", total_chunks=10, total_chapters=2)
    return bid, db_module.create_session(bid)


def _mock_nodes(monkeypatch, *, ambiguous, captured):
    import core.graph.nodes as G
    if ambiguous:
        intent = IntentDecision(
            kind="search", search_query="fate", ambiguous=True,
            clarify_question="你指的是谁?", clarify_options=["疯帽子", "白兔"],
        )
    else:
        intent = IntentDecision(kind="search", search_query="cheshire cat")
    monkeypatch.setattr(G, "classify_intent", lambda m, h: intent)
    monkeypatch.setattr(G, "get_vectorstore", lambda book_id: object())
    monkeypatch.setattr(G, "decompose_question",
                        lambda q: {"is_compound": False, "subquestions": [q]})

    def fake_gather(**kw):
        captured["message"] = kw.get("message")
        # Real Document — evidence is checkpointed, so it must serialize.
        return [Document(page_content="x", metadata={"chunk_id": "chk1"})]

    def fake_assemble(*, book_id, session_id, query_text, docs, vectorstore):
        eid = db_module.create_retrieval_event(
            session_id=session_id, book_id=book_id, query_text=query_text,
            multi_query_variants_json="[]", hyde_hypothesis="",
            raw_hits_count=1, new_raw_hits_count=1, summary_hits_count=0,
        )
        db_module.add_event_chunks(eid, [
            {"chunk_id": "chk1", "raptor_level": 0, "chapter": 1, "page": 1,
             "rank": 1, "origin": "keyword", "is_new_lighting": 1,
             "preview_text": "rabbit"},
        ])
        return RetrievalResult(
            docs=[{"text": "rabbit", "chunk_id": "chk1", "raptor_level": 0}],
            sources=[], event_id=eid,
            sse_payload=json.dumps({"type": "retrieval_update", "sources": [],
                                    "chapter_clusters": []}),
        )

    monkeypatch.setattr(G, "gather_evidence", fake_gather)
    monkeypatch.setattr(G, "assemble_retrieval_result", fake_assemble)
    monkeypatch.setattr(G, "synthesize_answer",
                        lambda **kw: AnswerDraft(text="<fact>The White Rabbit.</fact>",
                                                 is_error_response=False))


def _frames(resp_text):
    return [json.loads(ln[6:]) for ln in resp_text.splitlines() if ln.startswith("data: ")]


def test_chat_halts_with_ask_user_on_ambiguous(client, monkeypatch):
    bid, sid = _ready_book_session()
    _mock_nodes(monkeypatch, ambiguous=True, captured={})
    resp = client.post("/api/chat", json={"book_id": bid, "session_id": sid,
                                          "message": "他后来怎么样了?"})
    frames = _frames(resp.text)
    assert frames[-1]["type"] == "ask_user"           # terminal frame of leg 1
    assert frames[-1]["prompt"] == "你指的是谁?"
    assert frames[-1]["options"] == ["疯帽子", "白兔"]
    assert all(f["type"] not in ("token", "done", "stream_end") for f in frames)


def test_resume_completes_turn_with_answer_folded_in(client, monkeypatch):
    bid, sid = _ready_book_session()
    captured: dict = {}
    _mock_nodes(monkeypatch, ambiguous=True, captured=captured)
    client.post("/api/chat", json={"book_id": bid, "session_id": sid,
                                   "message": "他后来怎么样了?"})            # leg 1: interrupt
    resp = client.post("/api/chat/resume", json={"session_id": sid, "answer": "白兔"})
    types = [f["type"] for f in _frames(resp.text)]
    assert "token" in types and types[-1] == "stream_end"
    assert "白兔" in captured["message"]                                   # folded in


def test_resume_no_pending_returns_notice(client, monkeypatch):
    bid, sid = _ready_book_session()
    _mock_nodes(monkeypatch, ambiguous=False, captured={})
    resp = client.post("/api/chat/resume", json={"session_id": sid, "answer": "x"})
    frames = _frames(resp.text)
    assert frames[0]["type"] == "notice" and frames[0]["kind"] == "no_pending"
    assert frames[-1]["type"] == "stream_end"


def test_resume_expired_degrades_broad_with_notice(client, monkeypatch):
    bid, sid = _ready_book_session()
    captured: dict = {}
    _mock_nodes(monkeypatch, ambiguous=True, captured=captured)
    client.post("/api/chat", json={"book_id": bid, "session_id": sid,
                                   "message": "他后来怎么样了?"})            # leg 1: interrupt
    import api.chat as chat_mod
    monkeypatch.setattr(chat_mod, "_interrupt_expired", lambda snap: True)  # force expiry
    resp = client.post("/api/chat/resume", json={"session_id": sid, "answer": "白兔"})
    frames = _frames(resp.text)
    assert frames[0]["type"] == "notice" and frames[0]["kind"] == "clarify_expired"
    assert [f["type"] for f in frames][-1] == "stream_end"
    # safe-degrade: proceeded BROAD on the original question, answer NOT folded in
    assert "白兔" not in captured["message"]
    assert "他后来怎么样了?" in captured["message"]


def test_delete_book_gcs_thread_checkpoints(client, monkeypatch):
    import sqlite3
    import core.graph.build as build_mod

    bid, sid = _ready_book_session()
    _mock_nodes(monkeypatch, ambiguous=True, captured={})
    client.post("/api/chat", json={"book_id": bid, "session_id": sid,
                                   "message": "他后来怎么样了?"})  # writes a checkpoint

    def _count():
        cp = sqlite3.connect(str(build_mod.checkpoint_db_path()))
        try:
            return cp.execute("SELECT COUNT(*) FROM checkpoints WHERE thread_id=?",
                              (sid,)).fetchone()[0]
        finally:
            cp.close()

    assert _count() > 0                      # the paused turn left checkpoints
    client.delete(f"/api/books/{bid}")       # delete cascade -> GC by ownership
    assert _count() == 0                      # thread's checkpoints dropped
