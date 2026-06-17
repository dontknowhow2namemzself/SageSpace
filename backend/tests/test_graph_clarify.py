"""Tests for the clarify HITL gate + interrupt/resume seam (PR3 step 1).

Drives the real compiled graph (with an isolated SqliteSaver) and asserts
the LangGraph-level mechanism the design's §7 seam rests on:

  * an ambiguous intent halts the turn AT the clarify node (interrupt),
    surfacing the ask_user payload via get_state(),
  * resuming with Command(resume=answer) continues into retrieve, with the
    user's answer folded into the question the agent sees,
  * a clear intent passes straight through (no interrupt).
"""
from __future__ import annotations

import json
import sqlite3

import pytest
from langchain_core.documents import Document
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

import core.database as db_module
from core.graph import build_chat_graph
from core.graph import nodes as G
from core.pipeline.types import AnswerDraft, IntentDecision, RetrievalResult


@pytest.fixture
def isolated_db(tmp_path, monkeypatch):
    monkeypatch.setattr(db_module, "DB_PATH", tmp_path / "test.db")
    db_module.init_db()
    return tmp_path


def _book_and_session():
    book_id = db_module.create_book("Alice in Wonderland", "Carroll", "/tmp/a.pdf")
    db_module.update_book_status(book_id, "ready", total_chunks=10, total_chapters=2)
    return book_id, db_module.create_session(book_id)


def _compiled(tmp_path):
    conn = sqlite3.connect(str(tmp_path / "cp.db"), check_same_thread=False)
    return build_chat_graph(checkpointer=SqliteSaver(conn)), conn


def _doc(cid):
    # Real Document: the evidence channel is checkpointed, so it must be
    # serializable (a bare fake object is not).
    return Document(page_content="x", metadata={"chunk_id": cid})


def _ambiguous_intent():
    return IntentDecision(
        kind="search", search_query="character fate",
        ambiguous=True, clarify_question="你指的是谁?",
        clarify_options=["疯帽子", "白兔"], clarify_multi=False,
    )


def _wire_downstream(monkeypatch, captured):
    """Mock retrieve/synthesize so a resumed turn completes offline, and
    capture the message the agent receives (to prove the clarification was
    folded in)."""
    monkeypatch.setattr(G, "get_vectorstore", lambda book_id: object())
    # plan node runs an LLM for search intents; keep it offline + simple.
    monkeypatch.setattr(G, "decompose_question",
                        lambda q: {"is_compound": False, "subquestions": [q]})

    def fake_gather(**kw):
        captured["message"] = kw.get("message")
        return [_doc("chk_cat")]

    def fake_assemble(*, book_id, session_id, query_text, docs, vectorstore):
        eid = db_module.create_retrieval_event(
            session_id=session_id, book_id=book_id, query_text=query_text,
            multi_query_variants_json="[]", hyde_hypothesis="",
            raw_hits_count=1, new_raw_hits_count=1, summary_hits_count=0,
        )
        db_module.add_event_chunks(eid, [
            {"chunk_id": "chk_cat", "raptor_level": 0, "chapter": 1, "page": 1,
             "rank": 1, "origin": "keyword", "is_new_lighting": 1,
             "preview_text": "the white rabbit"},
        ])
        return RetrievalResult(
            docs=[{"text": "the white rabbit", "chunk_id": "chk_cat", "raptor_level": 0}],
            sources=[], event_id=eid,
            sse_payload=json.dumps({"type": "retrieval_update", "sources": [],
                                    "chapter_clusters": []}),
        )

    def fake_synth(**kw):
        captured["synth_question"] = kw.get("question")
        return AnswerDraft(text="<fact>The White Rabbit.</fact>", is_error_response=False)

    monkeypatch.setattr(G, "gather_evidence", fake_gather)
    monkeypatch.setattr(G, "assemble_retrieval_result", fake_assemble)
    monkeypatch.setattr(G, "synthesize_answer", fake_synth)


def _cfg(session_id):
    return {"configurable": {"thread_id": session_id}}


# ── interrupt ────────────────────────────────────────────────────────────────


def test_ambiguous_intent_interrupts_at_clarify(isolated_db, monkeypatch):
    book_id, session_id = _book_and_session()
    monkeypatch.setattr(G, "classify_intent", lambda m, h: _ambiguous_intent())
    _wire_downstream(monkeypatch, {})

    graph, _ = _compiled(isolated_db)
    init = {"book_id": book_id, "session_id": session_id, "message": "他后来怎么样了?",
            "book_title": "Alice", "history": []}
    frames = list(graph.stream(init, config=_cfg(session_id), stream_mode="custom"))

    # The turn halted before retrieve: no tool/token frames yet.
    assert all(f.get("type") not in ("token", "retrieval_update") for f in frames)

    snap = graph.get_state(_cfg(session_id))
    assert snap.next == ("clarify",)                     # paused AT the gate
    payload = snap.tasks[0].interrupts[0].value          # the ask_user content
    assert payload["kind"] == "clarify"
    assert payload["prompt"] == "你指的是谁?"
    assert payload["options"] == ["疯帽子", "白兔"]
    assert payload["multi"] is False


def test_resume_folds_answer_into_retrieval(isolated_db, monkeypatch):
    book_id, session_id = _book_and_session()
    monkeypatch.setattr(G, "classify_intent", lambda m, h: _ambiguous_intent())
    captured: dict = {}
    _wire_downstream(monkeypatch, captured)

    graph, _ = _compiled(isolated_db)
    init = {"book_id": book_id, "session_id": session_id, "message": "他后来怎么样了?",
            "book_title": "Alice", "history": []}
    list(graph.stream(init, config=_cfg(session_id), stream_mode="custom"))  # leg 1 -> interrupt

    # leg 2: user answered "白兔"
    frames = list(graph.stream(Command(resume="白兔"), config=_cfg(session_id),
                               stream_mode="custom"))

    assert [f.get("type") for f in frames][-1] == "stream_end"      # turn completed
    assert graph.get_state(_cfg(session_id)).next == ()            # fully done
    # the clarification reached BOTH retrieve and synthesize (the synthesize
    # gap was the "选了疯帽子没起作用" bug — answer hedged the ambiguity).
    assert "白兔" in captured["message"]
    assert "他后来怎么样了?" in captured["message"]                 # original kept too
    assert "白兔" in captured["synth_question"]                     # <- the fix
    assert "他后来怎么样了?" in captured["synth_question"]


def test_clear_intent_passes_through(isolated_db, monkeypatch):
    book_id, session_id = _book_and_session()
    monkeypatch.setattr(
        G, "classify_intent",
        lambda m, h: IntentDecision(kind="search", search_query="cheshire cat"),
    )
    captured: dict = {}
    _wire_downstream(monkeypatch, captured)

    graph, _ = _compiled(isolated_db)
    init = {"book_id": book_id, "session_id": session_id, "message": "Who is the Cheshire Cat?",
            "book_title": "Alice", "history": []}
    frames = list(graph.stream(init, config=_cfg(session_id), stream_mode="custom"))

    assert graph.get_state(_cfg(session_id)).next == ()           # never paused
    assert [f.get("type") for f in frames][-1] == "stream_end"
    assert captured["message"] == "Who is the Cheshire Cat?"      # no clarification appended
