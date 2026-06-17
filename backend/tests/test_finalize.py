"""Tests for the shared per-turn finalizer (core/pipeline/finalize.py).

The two paths in api/chat.py (structured fast path + ReAct slow path)
both delegate to finalize_turn after their visible token frame, so this
module is the contract point for "everything that happens after the
answer text is on the wire". We cover:

  - history is appended + saved exactly once
  - token usage flows through when a callback handler is provided
  - answer_attribution SSE only fires when there are linked retrieval
    events for this session
  - done + stream_end always fire, in order, at the end

The online faithfulness probe + its env gating were retired in v1.0
(see core/pipeline/finalize.py docstring). The offline replacement
lives in the sage-eval project.
"""
from __future__ import annotations

import json

import pytest

import core.database as db_module
from core.pipeline import finalize as F


# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    monkeypatch.setattr(db_module, "DB_PATH", tmp_path / "test.db")
    db_module.init_db()


@pytest.fixture
def session_no_events():
    """A book + empty session with NO retrieval_events recorded."""
    book_id = db_module.create_book("Book", "Author", "/tmp/b.pdf")
    db_module.update_book_status(book_id, "ready", total_chunks=10, total_chapters=1)
    session_id = db_module.create_session(book_id)
    return book_id, session_id


@pytest.fixture
def session_with_event():
    """A book + session that already has one retrieval_event with 2 chunks,
    simulating the state after search_book_content ran during the turn."""
    book_id = db_module.create_book("Book", "Author", "/tmp/b.pdf")
    db_module.update_book_status(book_id, "ready", total_chunks=10, total_chapters=1)
    session_id = db_module.create_session(book_id)
    event_id = db_module.create_retrieval_event(
        session_id=session_id, book_id=book_id, query_text="q",
        multi_query_variants_json="[]", hyde_hypothesis="",
        raw_hits_count=2, new_raw_hits_count=2, summary_hits_count=0,
    )
    db_module.add_event_chunks(event_id, [
        {"chunk_id": "chk_a", "raptor_level": 0, "chapter": 1, "page": 1,
         "rank": 1, "origin": "hyde", "is_new_lighting": 1,
         "preview_text": "preview a"},
        {"chunk_id": "chk_b", "raptor_level": 0, "chapter": 1, "page": 1,
         "rank": 2, "origin": "multi_query", "is_new_lighting": 1,
         "preview_text": "preview b"},
    ])
    return book_id, session_id, event_id


# ── Helpers ─────────────────────────────────────────────────────────────────


async def _collect(agen) -> list[dict]:
    """Drain an async generator of SSE frames into a list of parsed dicts.
    Each frame is `data: {...}\\n\\n`; we strip the prefix and parse JSON.
    """
    out: list[dict] = []
    async for frame in agen:
        assert frame.startswith("data: ") and frame.endswith("\n\n"), frame
        out.append(json.loads(frame[len("data: "):].strip()))
    return out


# ── Conversation persistence ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_finalize_appends_and_saves_conversation(session_no_events):
    _, session_id = session_no_events
    history: list[dict] = []
    frames = await _collect(F.finalize_turn(
        session_id=session_id,
        user_message="hello",
        assistant_response="hi there",
        history=history,
    ))
    # history mutated in place: user + assistant turns appended
    assert history == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi there"},
    ]
    # conversation_json persisted
    saved = db_module.get_session(session_id)["conversation_json"]
    assert json.loads(saved) == history
    # SSE: just done + stream_end (no events -> no attribution; faithfulness off)
    assert [f["type"] for f in frames] == ["done", "stream_end"]


# ── Token usage ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_finalize_persists_token_usage_when_callback_given(session_no_events):
    _, session_id = session_no_events

    class _FakeCallback:
        usage_metadata = {
            "openai/gpt-4o-mini": {"input_tokens": 100, "output_tokens": 50},
        }

    await _collect(F.finalize_turn(
        session_id=session_id,
        user_message="q",
        assistant_response="a",
        history=[],
        usage_callback=_FakeCallback(),
    ))
    s = db_module.get_session(session_id)
    assert s["total_tokens_in"] == 100
    assert s["total_tokens_out"] == 50
    # cost is whatever model_pricing.json maps gpt-4o-mini to; just sanity
    # that the row got written (positive or zero, not None)
    assert s["total_cost_usd"] is not None


@pytest.mark.asyncio
async def test_finalize_skips_usage_when_no_callback(session_no_events):
    _, session_id = session_no_events
    await _collect(F.finalize_turn(
        session_id=session_id, user_message="q", assistant_response="a",
        history=[], usage_callback=None,
    ))
    s = db_module.get_session(session_id)
    assert s["total_tokens_in"] == 0
    assert s["total_tokens_out"] == 0


# ── Attribution ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_finalize_emits_attribution_when_events_exist(session_with_event):
    _, session_id, event_id = session_with_event
    frames = await _collect(F.finalize_turn(
        session_id=session_id, user_message="q", assistant_response="some answer",
        history=[],
    ))
    types = [f["type"] for f in frames]
    assert types == ["answer_attribution", "done", "stream_end"]
    attr = frames[0]
    assert attr["retrieval_event_ids"] == [event_id]
    assert set(attr["chunk_ids"]) == {"chk_a", "chk_b"}
    # the attribution payload was persisted onto the event row as well
    event = db_module.get_retrieval_events(session_id)[0]
    assert event["answer_attribution_json"], "attribution should be attached"


@pytest.mark.asyncio
async def test_finalize_merges_per_fact_attribution_into_payload(session_with_event):
    """The graph path computes per-fact attribution at synthesize time
    and passes it through finalize. The facts must land BOTH in the SSE
    answer_attribution frame (eval clients read them) and in the
    attached event row (the Reading Map derives its lit set from them)."""
    _, session_id, event_id = session_with_event
    fact_attribution = {
        "retrieval_event_ids": [event_id],
        "chunk_ids": ["chk_a", "chk_b"],
        "raptor_ids": [],
        "facts": [
            {"fact_id": "f1", "text": "cited claim",
             "chunk_ids": ["chk_b"], "retrieval_event_ids": [event_id]},
        ],
    }
    frames = await _collect(F.finalize_turn(
        session_id=session_id, user_message="q",
        assistant_response="<fact>cited claim</fact>", history=[],
        fact_attribution=fact_attribution,
    ))
    attr = frames[0]
    assert attr["type"] == "answer_attribution"
    assert attr["facts"][0]["chunk_ids"] == ["chk_b"]
    event = db_module.get_retrieval_events(session_id)[0]
    stored = json.loads(event["answer_attribution_json"])
    assert stored["facts"][0]["chunk_ids"] == ["chk_b"]


@pytest.mark.asyncio
async def test_finalize_attaches_only_to_current_turn_events(session_with_event):
    """Cross-turn clobber regression: the linked-event list spans the
    session's last 3 events, but the attach (a row OVERWRITE) must hit
    only THIS turn's event — otherwise turn N rewrites turn N-1's
    per-fact facts and the Reading Map's lit chunks go dark."""
    book_id, session_id, e1 = session_with_event

    # Turn 1: facts cite chk_a, attached to e1.
    await _collect(F.finalize_turn(
        session_id=session_id, user_message="q1",
        assistant_response="<fact>a</fact>", history=[],
        fact_attribution={"facts": [
            {"fact_id": "f1", "text": "a", "chunk_ids": ["chk_a"],
             "retrieval_event_ids": [e1]},
        ]},
        current_event_ids=[e1],
    ))

    # Turn 2: a NEW event e2; facts cite chk_b, attached to e2 only.
    e2 = db_module.create_retrieval_event(
        session_id=session_id, book_id=book_id, query_text="q2",
        multi_query_variants_json="[]", hyde_hypothesis="",
        raw_hits_count=1, new_raw_hits_count=1, summary_hits_count=0,
    )
    db_module.add_event_chunks(e2, [
        {"chunk_id": "chk_b", "raptor_level": 0, "chapter": 1, "page": 2,
         "rank": 1, "origin": "hyde", "is_new_lighting": 1,
         "preview_text": "preview b"},
    ])
    await _collect(F.finalize_turn(
        session_id=session_id, user_message="q2",
        assistant_response="<fact>b</fact>", history=[],
        fact_attribution={"facts": [
            {"fact_id": "f1", "text": "b", "chunk_ids": ["chk_b"],
             "retrieval_event_ids": [e2]},
        ]},
        current_event_ids=[e2],
    ))

    stored = {
        e["id"]: json.loads(e["answer_attribution_json"] or "null")
        for e in db_module.get_retrieval_events(session_id)
    }
    # e1 still carries turn 1's facts — NOT overwritten by turn 2.
    assert stored[e1]["facts"][0]["chunk_ids"] == ["chk_a"]
    assert stored[e2]["facts"][0]["chunk_ids"] == ["chk_b"]


@pytest.mark.asyncio
async def test_finalize_no_fact_turn_writes_empty_facts_key(session_with_event):
    """A modern turn with no <fact> tags must still write facts=[] on its
    event: api/debug.py treats a MISSING facts key as a legacy row and
    falls back to lighting the full turn-level chunk union."""
    _, session_id, event_id = session_with_event
    frames = await _collect(F.finalize_turn(
        session_id=session_id, user_message="q",
        assistant_response="<commentary>no facts here</commentary>",
        history=[], fact_attribution=None,
        current_event_ids=[event_id],
    ))
    attr = frames[0]
    assert attr["type"] == "answer_attribution"
    assert attr["facts"] == []
    stored = json.loads(
        db_module.get_retrieval_events(session_id)[0]["answer_attribution_json"]
    )
    assert stored["facts"] == []


@pytest.mark.asyncio
async def test_finalize_turn_without_events_attaches_nothing(session_with_event):
    """A turn that produced NO retrieval event (smalltalk following a
    search turn) must not re-attach attribution to earlier events —
    that would replace their per-fact facts with a facts-less payload."""
    _, session_id, e1 = session_with_event

    # Turn 1 attaches real facts to e1.
    await _collect(F.finalize_turn(
        session_id=session_id, user_message="q1",
        assistant_response="<fact>a</fact>", history=[],
        fact_attribution={"facts": [
            {"fact_id": "f1", "text": "a", "chunk_ids": ["chk_a"],
             "retrieval_event_ids": [e1]},
        ]},
        current_event_ids=[e1],
    ))
    # Smalltalk turn: no new event, current_event_ids=[].
    frames = await _collect(F.finalize_turn(
        session_id=session_id, user_message="thanks!",
        assistant_response="<commentary>my pleasure</commentary>",
        history=[], fact_attribution=None,
        current_event_ids=[],
    ))
    # Frame still emitted (eval clients read it) ...
    assert frames[0]["type"] == "answer_attribution"
    # ... but e1's stored facts are untouched.
    stored = json.loads(
        db_module.get_retrieval_events(session_id)[0]["answer_attribution_json"]
    )
    assert stored["facts"][0]["chunk_ids"] == ["chk_a"]


@pytest.mark.asyncio
async def test_finalize_records_cited_chunks_raw_only(session_with_event):
    """Finalize writes the reader-facing cited ledger from the turn's
    facts: raw chunk ids only — a RAPTOR summary citation shows in the
    popup but must not count as reading progress."""
    book_id, session_id, event_id = session_with_event
    await _collect(F.finalize_turn(
        session_id=session_id, user_message="q",
        assistant_response="<fact>a</fact><fact>b</fact>", history=[],
        fact_attribution={"facts": [
            {"fact_id": "f1", "text": "a", "chunk_ids": ["chk_a", "chk_b"],
             "retrieval_event_ids": [event_id]},
            {"fact_id": "f2", "text": "b",
             "chunk_ids": ["raptor_l1_sec_x", "chk_b"],
             "retrieval_event_ids": [event_id]},
        ]},
        current_event_ids=[event_id],
    ))
    assert set(db_module.get_cited_chunk_ids(session_id)) == {"chk_a", "chk_b"}
    assert set(db_module.get_all_cited_chunk_ids_for_book(book_id)) == {"chk_a", "chk_b"}


@pytest.mark.asyncio
async def test_finalize_no_facts_records_no_cited_chunks(session_with_event):
    _, session_id, event_id = session_with_event
    await _collect(F.finalize_turn(
        session_id=session_id, user_message="q",
        assistant_response="<commentary>nothing factual</commentary>",
        history=[], fact_attribution=None,
        current_event_ids=[event_id],
    ))
    assert db_module.get_cited_chunk_ids(session_id) == []


@pytest.mark.asyncio
async def test_finalize_skips_attribution_when_no_events(session_no_events):
    _, session_id = session_no_events
    frames = await _collect(F.finalize_turn(
        session_id=session_id, user_message="q", assistant_response="a",
        history=[],
    ))
    assert [f["type"] for f in frames] == ["done", "stream_end"]


# ── Frame ordering invariants ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_finalize_done_always_precedes_stream_end(session_with_event):
    _, session_id, _ = session_with_event
    frames = await _collect(F.finalize_turn(
        session_id=session_id, user_message="q", assistant_response="ok",
        history=[],
    ))
    types = [f["type"] for f in frames]
    assert types.index("done") < types.index("stream_end")
    assert types[-1] == "stream_end"


@pytest.mark.asyncio
async def test_finalize_never_emits_quality_update(session_with_event):
    """v1.0: the online faithfulness probe is gone. quality_update SSE
    frames should never appear, regardless of inputs."""
    _, session_id, _ = session_with_event
    frames = await _collect(F.finalize_turn(
        session_id=session_id, user_message="q",
        assistant_response="<fact>grounded</fact>", history=[],
    ))
    assert "quality_update" not in [f["type"] for f in frames]
