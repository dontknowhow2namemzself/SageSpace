"""Tests for the LangGraph chat-turn port (core/graph/).

PR1 is a "same behavior, new substrate" port: the four nodes wrap the
existing pipeline functions and must emit the SAME structured SSE
payloads, in the SAME order, that the pre-graph `event_stream()` did.
These tests drive the compiled graph directly (with an isolated
SqliteSaver) and assert:

  * the per-intent frame sequence (tool frames + retrieval_update +
    token + attribution + done + stream_end) matches the old contract,
  * conversation history is persisted exactly once,
  * the graph itself does NOT account token usage (that moved to the
    request level in api/chat.py -- finalize passes usage_callback=None),
  * the SqliteSaver actually writes a checkpoint for the thread.

The node internals (intent / retrieval / synthesis / finalize) are
mocked at the seam so these stay fast + offline; their real behavior is
covered by the test_pipeline_* / test_finalize suites.
"""
from __future__ import annotations

import json
import sqlite3

import pytest
from langchain_core.documents import Document
from langgraph.checkpoint.sqlite import SqliteSaver

import core.database as db_module
from core.graph import build_chat_graph
from core.graph import nodes as G
from core.pipeline.types import AnswerDraft, IntentDecision, RetrievalResult


# ── Fixtures / helpers ──────────────────────────────────────────────────────


@pytest.fixture
def isolated_db(tmp_path, monkeypatch):
    monkeypatch.setattr(db_module, "DB_PATH", tmp_path / "test.db")
    db_module.init_db()
    return tmp_path


@pytest.fixture(autouse=True)
def _mock_attribution_mapper(monkeypatch):
    """The per-fact attribution mapper is a real LLM call inside
    synthesize_node — graph tests must never touch the network.
    Defaults every fact to unattributed; tests that assert specific
    routing override this with their own monkeypatch."""
    import core.pipeline.finalize as finalize_module
    monkeypatch.setattr(
        finalize_module, "_map_facts_to_chunks",
        lambda fact_texts, docs: [[] for _ in fact_texts],
    )


def _book_and_session():
    book_id = db_module.create_book("Alice in Wonderland", "Carroll", "/tmp/a.pdf")
    db_module.update_book_status(book_id, "ready", total_chunks=10, total_chapters=2)
    session_id = db_module.create_session(book_id)
    return book_id, session_id


def _compiled(tmp_path):
    """Compile the real graph with an isolated on-disk SqliteSaver so we
    can assert a checkpoint row was written."""
    conn = sqlite3.connect(str(tmp_path / "cp.db"), check_same_thread=False)
    return build_chat_graph(checkpointer=SqliteSaver(conn)), conn


def _run(graph, *, book_id, session_id, message, history=None):
    payloads: list[dict] = []
    # Mirrors api/chat.py's init_state, INCLUDING the per-turn resets:
    # the thread checkpoint keeps the previous turn's state, and a path
    # that skips retrieve (smalltalk) must not inherit a stale
    # RetrievalResult (finalize would attach onto the wrong event).
    init = {
        "book_id": book_id,
        "session_id": session_id,
        "message": message,
        "book_title": "Alice in Wonderland",
        "history": history if history is not None else [],
        "retrieval": None,
        "fact_attribution": None,
    }
    for p in graph.stream(
        init, config={"configurable": {"thread_id": session_id}}, stream_mode="custom"
    ):
        payloads.append(p)
    return payloads


def _stub_synth(text: str):
    return lambda **kw: AnswerDraft(text=text, is_error_response=False)


# ── search / book_overview path ─────────────────────────────────────────────


def test_graph_search_emits_full_frame_sequence(isolated_db, monkeypatch):
    book_id, session_id = _book_and_session()

    monkeypatch.setattr(
        G, "classify_intent",
        lambda msg, hist: IntentDecision(kind="search", search_query="cheshire cat"),
    )
    monkeypatch.setattr(G, "get_vectorstore", lambda book_id: object())
    monkeypatch.setattr(G, "decompose_question",
                        lambda q: {"is_compound": False, "subquestions": [q]})

    retrieval = RetrievalResult(
        docs=[{"text": "the Cheshire Cat grinned at Alice", "chunk_id": "chk_cat",
               "raptor_level": 0}],
        sources=[],
        event_id=None,
        sse_payload=json.dumps({"type": "retrieval_update", "sources": [],
                                "chapter_clusters": []}),
    )

    # PR2: the search path runs the bounded ReAct agent (gather_evidence)
    # then assembles a RetrievalResult. Mock the agent (its own tool frames
    # are covered in test_retrieve_agent) and the assembly, but mimic the
    # retrieval_event side effect so finalize can link an answer_attribution.
    def fake_assemble(**kw):
        eid = db_module.create_retrieval_event(
            session_id=session_id, book_id=book_id, query_text="cheshire cat",
            multi_query_variants_json="[]", hyde_hypothesis="",
            raw_hits_count=1, new_raw_hits_count=1, summary_hits_count=0,
        )
        db_module.add_event_chunks(eid, [
            {"chunk_id": "chk_cat", "raptor_level": 0, "chapter": 6, "page": 1,
             "rank": 1, "origin": "keyword", "is_new_lighting": 1,
             "preview_text": "the Cheshire Cat"},
        ])
        return retrieval

    monkeypatch.setattr(
        G, "gather_evidence",
        lambda **kw: [Document(page_content="x", metadata={"chunk_id": "chk_cat"})],
    )
    monkeypatch.setattr(G, "assemble_retrieval_result", fake_assemble)
    monkeypatch.setattr(
        G, "synthesize_answer",
        _stub_synth("<fact>The Cheshire Cat grinned at Alice.</fact>"),
    )
    # The per-fact attribution mapper is a real LLM call — mock it like
    # the synthesizer above.
    import core.pipeline.finalize as finalize_module
    monkeypatch.setattr(
        finalize_module, "_map_facts_to_chunks",
        lambda fact_texts, docs: [["chk_cat"] for _ in fact_texts],
    )

    graph, conn = _compiled(isolated_db)
    payloads = _run(graph, book_id=book_id, session_id=session_id,
                    message="who is the cat")

    # With the agent mocked, the search path emits the assembled
    # retrieval_update, then the answer + finalize frames. (The agent's own
    # per-tool tool_start/tool_end frames are asserted in test_retrieve_agent.)
    assert [p["type"] for p in payloads] == [
        "retrieval_update", "token", "answer_attribution", "done", "stream_end",
    ]

    # token carries the answer with per-<fact> data-* injection
    token = next(p for p in payloads if p["type"] == "token")
    assert 'data-fact-id="f1"' in token["content"]
    assert 'data-chunk-ids="chk_cat"' in token["content"]

    # conversation persisted once: user + assistant appended
    saved = json.loads(db_module.get_session(session_id)["conversation_json"])
    assert saved == [
        {"role": "user", "content": "who is the cat"},
        {"role": "assistant", "content": token["content"]},
    ]

    # graph does NOT account usage (that's request-level in chat.py)
    s = db_module.get_session(session_id)
    assert s["total_tokens_in"] == 0 and s["total_tokens_out"] == 0

    # SqliteSaver wrote a checkpoint for this thread
    n = conn.execute(
        "SELECT COUNT(*) FROM checkpoints WHERE thread_id = ?", (session_id,)
    ).fetchone()[0]
    assert n > 0


def test_graph_fanout_merges_subquestion_evidence(isolated_db, monkeypatch):
    """A compound question fans out one ReAct branch per sub-question; the
    join node assembles ONE RetrievalResult over the merged, deduped union."""
    book_id, session_id = _book_and_session()
    monkeypatch.setattr(
        G, "classify_intent",
        lambda m, h: IntentDecision(kind="search", search_query="x"),
    )
    monkeypatch.setattr(G, "get_vectorstore", lambda book_id: object())
    monkeypatch.setattr(
        G, "decompose_question",
        lambda q: {"is_compound": True, "subquestions": ["sub one", "sub two"]},
    )

    seen_subqs = []

    def fake_gather(**kw):
        sq = kw["message"]
        seen_subqs.append(sq)
        cid = "chk_a" if "one" in sq else "chk_b"
        return [Document(page_content=sq, metadata={"chunk_id": cid, "raptor_level": 0})]

    captured = {}

    def fake_assemble(*, book_id, session_id, query_text, docs, vectorstore):
        captured["chunk_ids"] = sorted(d.metadata["chunk_id"] for d in docs)
        eid = db_module.create_retrieval_event(
            session_id=session_id, book_id=book_id, query_text=query_text,
            multi_query_variants_json="[]", hyde_hypothesis="",
            raw_hits_count=len(docs), new_raw_hits_count=len(docs), summary_hits_count=0,
        )
        db_module.add_event_chunks(eid, [
            {"chunk_id": d.metadata["chunk_id"], "raptor_level": 0, "chapter": 1,
             "page": 1, "rank": 1, "origin": "semantic", "is_new_lighting": 1,
             "preview_text": "x"} for d in docs
        ])
        return RetrievalResult(
            docs=[{"text": "x", "chunk_id": d.metadata["chunk_id"], "raptor_level": 0}
                  for d in docs],
            sources=[], event_id=eid,
            sse_payload=json.dumps({"type": "retrieval_update", "sources": [],
                                    "chapter_clusters": []}),
        )

    monkeypatch.setattr(G, "gather_evidence", fake_gather)
    monkeypatch.setattr(G, "assemble_retrieval_result", fake_assemble)
    synth_calls = []
    monkeypatch.setattr(
        G, "synthesize_answer",
        lambda **kw: synth_calls.append(1) or AnswerDraft(
            text="<fact>Merged answer.</fact>", is_error_response=False),
    )

    graph, _ = _compiled(isolated_db)
    payloads = _run(graph, book_id=book_id, session_id=session_id,
                    message="What is X, and what is Y?")

    assert set(seen_subqs) == {"sub one", "sub two"}      # both branches ran
    assert captured["chunk_ids"] == ["chk_a", "chk_b"]    # join merged the union
    assert len(synth_calls) == 1                          # synthesized ONCE
    assert [p["type"] for p in payloads][-1] == "stream_end"


def test_balance_by_subquestion_round_robins():
    """One branch must not monopolize the context cap: interleave the
    branches so each sub-question is represented near the top."""
    from core.graph.nodes import _balance_by_subquestion

    def d(cid, sq):
        return Document(page_content=cid, metadata={"chunk_id": cid, "sub_question": sq})

    # branch A returned 3 docs, branch B only 1 -> round-robin keeps B near top
    out = _balance_by_subquestion([d("a1", "A"), d("a2", "A"), d("a3", "A"), d("b1", "B")])
    ids = [x.metadata["chunk_id"] for x in out]
    assert ids[:2] == ["a1", "b1"]                # A, B interleaved first
    assert set(ids) == {"a1", "a2", "a3", "b1"}   # nothing dropped


def test_balance_by_subquestion_single_is_noop():
    from core.graph.nodes import _balance_by_subquestion

    docs = [Document(page_content="x", metadata={"chunk_id": f"c{i}", "sub_question": "S"})
            for i in range(3)]
    assert [x.metadata["chunk_id"] for x in _balance_by_subquestion(docs)] == ["c0", "c1", "c2"]


def test_balance_diversifies_sections_within_a_single_branch():
    """A single branch that fetched several sections must spread them across
    the cap -- the "final three chapters" failure mode, where get_chapter x3
    in ONE branch let the first chapter's chunks (listed section-by-section)
    fill the whole cap and drop the other two."""
    from core.graph.nodes import _balance_by_subquestion

    def d(cid, sec):
        return Document(page_content=cid,
                        metadata={"chunk_id": cid, "sub_question": "Q", "section_id": sec})

    # one branch, 3 sections (X/Y/Z) x 2 docs each, listed section-by-section
    docs = [d("x1", "X"), d("x2", "X"),
            d("y1", "Y"), d("y2", "Y"),
            d("z1", "Z"), d("z2", "Z")]
    ids = [x.metadata["chunk_id"] for x in _balance_by_subquestion(docs)]
    assert ids[:3] == ["x1", "y1", "z1"]                      # one per section up front
    assert set(ids) == {"x1", "x2", "y1", "y2", "z1", "z2"}   # nothing dropped


def test_balance_nests_sections_inside_subquestions():
    """Both levels at once: outer keeps each fan-out branch represented,
    inner spreads each branch across its sections."""
    from core.graph.nodes import _balance_by_subquestion

    def d(cid, sq, sec):
        return Document(page_content=cid,
                        metadata={"chunk_id": cid, "sub_question": sq, "section_id": sec})

    docs = [d("a_x", "A", "X"), d("a_y", "A", "Y"),   # branch A spans 2 sections
            d("b_x", "B", "X")]                        # branch B, 1 section
    ids = [x.metadata["chunk_id"] for x in _balance_by_subquestion(docs)]
    assert ids[0] == "a_x" and ids[1] == "b_x"         # outer: A, B interleaved first
    assert set(ids) == {"a_x", "a_y", "b_x"}


# ── chapter_summary path ────────────────────────────────────────────────────


def test_graph_chapter_summary_uses_chapter_tool(isolated_db, monkeypatch):
    book_id, session_id = _book_and_session()
    monkeypatch.setattr(
        G, "classify_intent",
        lambda msg, hist: IntentDecision(kind="chapter_summary", chapter_number=6),
    )
    monkeypatch.setattr(G, "get_vectorstore", lambda book_id: object())
    monkeypatch.setattr(
        G, "chapter_summary_retrieval",
        lambda **kw: RetrievalResult(
            docs=[{"text": "Pig and Pepper", "chunk_id": "c6", "raptor_level": 1}],
            sources=[], event_id=None,
            sse_payload=json.dumps({"type": "retrieval_update", "sources": []}),
        ),
    )
    monkeypatch.setattr(G, "synthesize_answer", _stub_synth("<fact>Chapter six.</fact>"))

    graph, _ = _compiled(isolated_db)
    payloads = _run(graph, book_id=book_id, session_id=session_id,
                    message="tell me about chapter 6")

    assert [p["type"] for p in payloads] == [
        "tool_start", "tool_end", "retrieval_update", "token", "done", "stream_end",
    ]
    assert payloads[0] == {"type": "tool_start", "tool": "chapter_summary"}
    assert payloads[1] == {"type": "tool_end", "tool": "chapter_summary"}


# ── non-retrieval paths (no tool frames) ────────────────────────────────────


def test_graph_smalltalk_skips_retrieval(isolated_db, monkeypatch):
    book_id, session_id = _book_and_session()
    monkeypatch.setattr(G, "classify_intent",
                        lambda msg, hist: IntentDecision(kind="smalltalk"))
    monkeypatch.setattr(G, "get_vectorstore", lambda book_id: object())
    monkeypatch.setattr(G, "synthesize_answer",
                        _stub_synth("<commentary>I only know this book.</commentary>"))

    graph, _ = _compiled(isolated_db)
    payloads = _run(graph, book_id=book_id, session_id=session_id, message="hi there")

    # No tool_start / tool_end / retrieval_update for smalltalk.
    assert [p["type"] for p in payloads] == ["token", "done", "stream_end"]


def test_graph_smalltalk_turn_does_not_clobber_previous_attribution(
    isolated_db, monkeypatch
):
    """Stale-checkpoint regression: turn 1 (search) attaches per-fact
    attribution to its retrieval event; turn 2 (smalltalk on the SAME
    thread) skips retrieve, so without the per-turn state reset it would
    inherit turn 1's RetrievalResult from the checkpoint and overwrite
    that event's facts with an empty list — going dark on the Reading
    Map."""
    book_id, session_id = _book_and_session()

    monkeypatch.setattr(
        G, "classify_intent",
        lambda msg, hist: IntentDecision(kind="search", search_query="cat"),
    )
    monkeypatch.setattr(G, "get_vectorstore", lambda book_id: object())
    monkeypatch.setattr(G, "decompose_question",
                        lambda q: {"is_compound": False, "subquestions": [q]})

    def fake_assemble(**kw):
        eid = db_module.create_retrieval_event(
            session_id=session_id, book_id=book_id, query_text="cat",
            multi_query_variants_json="[]", hyde_hypothesis="",
            raw_hits_count=1, new_raw_hits_count=1, summary_hits_count=0,
        )
        db_module.add_event_chunks(eid, [
            {"chunk_id": "chk_cat", "raptor_level": 0, "chapter": 6, "page": 1,
             "rank": 1, "origin": "keyword", "is_new_lighting": 1,
             "preview_text": "the Cheshire Cat"},
        ])
        return RetrievalResult(
            docs=[{"text": "the Cheshire Cat grinned", "chunk_id": "chk_cat",
                   "raptor_level": 0}],
            sources=[], event_id=eid, sse_payload=None,
        )

    monkeypatch.setattr(
        G, "gather_evidence",
        lambda **kw: [Document(page_content="x", metadata={"chunk_id": "chk_cat"})],
    )
    monkeypatch.setattr(G, "assemble_retrieval_result", fake_assemble)
    monkeypatch.setattr(
        G, "synthesize_answer", _stub_synth("<fact>The Cat grinned.</fact>")
    )
    import core.pipeline.finalize as finalize_module
    monkeypatch.setattr(
        finalize_module, "_map_facts_to_chunks",
        lambda fact_texts, docs: [["chk_cat"] for _ in fact_texts],
    )

    graph, _ = _compiled(isolated_db)
    _run(graph, book_id=book_id, session_id=session_id, message="who is the cat")

    event = db_module.get_retrieval_events(session_id)[0]
    stored = json.loads(event["answer_attribution_json"])
    assert stored["facts"][0]["chunk_ids"] == ["chk_cat"]

    # Turn 2: smalltalk on the same thread.
    monkeypatch.setattr(G, "classify_intent",
                        lambda msg, hist: IntentDecision(kind="smalltalk"))
    monkeypatch.setattr(G, "synthesize_answer",
                        _stub_synth("<commentary>my pleasure</commentary>"))
    _run(graph, book_id=book_id, session_id=session_id, message="thanks!")

    # Turn 1's event still carries its facts — not clobbered to [].
    event = db_module.get_retrieval_events(session_id)[0]
    stored = json.loads(event["answer_attribution_json"])
    assert stored["facts"][0]["chunk_ids"] == ["chk_cat"]


def test_graph_reading_progress_weaves_progress_data(isolated_db, monkeypatch):
    book_id, session_id = _book_and_session()
    monkeypatch.setattr(G, "classify_intent",
                        lambda msg, hist: IntentDecision(kind="reading_progress"))
    monkeypatch.setattr(G, "get_vectorstore", lambda book_id: object())
    seen = {}

    def fake_compute(b, s):
        seen["called"] = (b, s)
        return {"digested_pct": "20%"}

    monkeypatch.setattr(G, "compute_reading_progress", fake_compute)

    def synth_capturing(**kw):
        seen["progress_data"] = kw.get("progress_data")
        return AnswerDraft(text="<commentary>20% read.</commentary>")

    monkeypatch.setattr(G, "synthesize_answer", synth_capturing)

    graph, _ = _compiled(isolated_db)
    payloads = _run(graph, book_id=book_id, session_id=session_id,
                    message="how much have I read")

    assert [p["type"] for p in payloads] == ["token", "done", "stream_end"]
    # progress_data flowed through the graph state into synthesize
    assert seen["called"] == (book_id, session_id)
    assert seen["progress_data"] == {"digested_pct": "20%"}
