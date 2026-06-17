"""Chat-turn graph nodes.

Each node is a thin wrapper over an existing `core/pipeline/*` function
-- the PR1 port is "same behavior, new substrate", so the actual
intent/retrieval/synthesis/finalization logic is reused verbatim and
its unit tests stay green. The nodes' only new job is to:

  1. read/write the typed `GraphState` channels, and
  2. emit the turn's SSE payloads via LangGraph's `get_stream_writer()`,
     in the exact order + timing the pre-graph `event_stream()` did.

Payload contract: nodes write **plain dicts** to the stream writer.
`api/chat.py` owns the `data: {...}\n\n` SSE framing on the way out, so
the graph stays transport-agnostic (a non-SSE consumer -- a test, a
LangSmith eval -- can read the same structured payloads).

Why `get_stream_writer()` rather than returning frames in state: it
streams *incrementally mid-node*, so `tool_start` is on the wire before
the retrieval call runs (preserving the frontend's tool spinner), and
it is the same mechanism PR3 will use to surface `ask_user` frames from
the (future) clarify node.
"""
from __future__ import annotations

import json
import logging
import os

from langgraph.config import get_stream_writer
from langgraph.types import Send, interrupt

from core.pipeline.finalize import (
    build_answer_attribution,
    build_finalize_payloads,
    inject_fact_attribution,
)
from core.pipeline.intent import classify_intent
from core.pipeline.plan import decompose_question, estimate_cost
from core.pipeline.retrieve import assemble_retrieval_result, chapter_summary_retrieval
from core.pipeline.synthesize import synthesize_answer
from core.pipeline.types import IntentDecision
from core.raptor import get_vectorstore
from core.tools import compute_reading_progress, run_export

from core import database as db
from core.graph.retrieve_agent import gather_evidence
from core.graph.state import GraphState


logger = logging.getLogger(__name__)


# Preserve the pre-graph chat_debug.log breadcrumb trail (backend/chat_debug.log).
# Best-effort dev aid only -- never raises, never part of the SSE contract.
_CHAT_DEBUG_LOG = os.path.join(os.path.dirname(__file__), "..", "..", "chat_debug.log")


def _debug_log(stage: str, payload) -> None:
    try:
        with open(_CHAT_DEBUG_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps({"stage": stage, "payload": payload}, ensure_ascii=False) + "\n")
    except Exception:
        pass


def _resolved_question(state: GraphState) -> str:
    """The turn's question with any clarify answer folded in (PR3).

    Used by BOTH retrieve (so the agent searches the right subject) and
    synthesize (so the answer commits to it instead of re-hedging the
    ambiguity). `state["message"]` stays the raw user input — finalize
    persists that, not this augmented form. No clarification -> raw message.
    """
    message = state["message"]
    clar = state.get("clarification")
    if clar and clar.get("answer"):
        return (
            f'{message}\n\n[Clarification — the user was asked '
            f'"{clar.get("question", "")}" and answered "{clar["answer"]}". '
            f'Treat that answer as the intended subject of the question.]'
        )
    return message


# ── Node 1: classify_intent ─────────────────────────────────────────────────


def classify_intent_node(state: GraphState) -> dict:
    """LLM intent router (design §5, KEPT). Emits no SSE frame -- matches
    the pre-graph behavior where intent classification was silent."""
    message = state["message"]
    try:
        intent = classify_intent(message, state.get("history", []))
    except Exception as exc:  # classify_intent already self-guards; belt & braces
        _debug_log("intent_classifier_error", str(exc))
        intent = IntentDecision(kind="search", search_query=message)
    _debug_log("intent", {
        "kind": intent.kind,
        "chapter_number": intent.chapter_number,
        "search_query": intent.search_query,
        "export_format": intent.export_format,
        "memory_note": intent.memory_note,
    })
    return {"intent": intent}


# ── Node 2: clarify (HITL①, PR3) ────────────────────────────────────────────


def clarify_node(state: GraphState) -> dict:
    """Human-in-the-loop gate before retrieval (design §6/§7).

    方案甲: the ambiguity judgement already happened inside
    `classify_intent` (one LLM call), so this node makes NO LLM call and
    has NO side effects before the interrupt — which is mandatory, because
    on resume LangGraph re-runs the node from its top (§7.3). It simply:

      * passes straight through when the question is clear, or
      * `interrupt()`s with the clarifying question + options when it is
        ambiguous, halting the turn. The graph checkpoints; the stream
        ends; `api/chat.py` surfaces the payload as an `ask_user` frame.
        A later `POST /api/chat/resume` re-enters here with
        `Command(resume=<answer>)`, so `interrupt()` returns the answer and
        the turn continues into retrieve.

    On expiry / no safe resume the design's safe default is "proceed
    broad" — that degrade lives in `api/chat.py` (it just doesn't resume
    here), so this node stays a pure gate.
    """
    intent: IntentDecision = state["intent"]
    if not intent.ambiguous or not intent.clarify_question:
        return {}  # clear question -> no interrupt, fall through to retrieve

    answer = interrupt(
        {
            "kind": "clarify",
            "prompt": intent.clarify_question,
            "options": intent.clarify_options,
            "multi": intent.clarify_multi,
        }
    )
    # Reached only after resume. `answer` is whatever the user sent back.
    return {"clarification": {"question": intent.clarify_question, "answer": answer}}


# ── Node 2.5: plan (decompose compound questions, PR4) ──────────────────────


def plan_node(state: GraphState) -> dict:
    """Decompose a compound question into sub-questions for the retrieve
    fan-out (design §3/§5). Only the agent retrieval intents (search /
    book_overview) are decomposed; every other kind passes straight through.

    Emits no SSE frame (planning is silent). Stores `plan` +
    `cost_estimate`. NOTE: PR4 step 1 wires this node in but retrieve still
    runs on the whole question; step 2 adds the Send fan-out that consumes
    `plan["subquestions"]`."""
    intent: IntentDecision = state["intent"]
    if intent.kind not in ("search", "book_overview"):
        return {}  # nothing to decompose for chapter/progress/export/smalltalk

    plan = decompose_question(_resolved_question(state))
    _debug_log("plan", plan)
    # Only a genuinely compound question gets the coarse "researching N
    # sub-questions" panel; simple questions stream their tool frames as
    # usual (the agent's per-tool spinner). join emits the matching
    # fanout_end. (PR4 step 2b.)
    if plan["is_compound"]:
        get_stream_writer()({"type": "fanout_start", "subquestions": plan["subquestions"]})
    return {"plan": plan, "cost_estimate": estimate_cost(len(plan["subquestions"]))}


# ── Node 3: retrieve (dispatch on intent.kind) ──────────────────────────────


def retrieve_node(state: GraphState) -> dict:
    """Gather this turn's grounding context. Dispatches on intent.kind to
    the same handlers the pre-graph chat loop used; retrieval-bearing
    intents emit tool_start -> (work) -> tool_end -> retrieval_update,
    exactly as before. The non-retrieval intents (progress / export /
    smalltalk) populate their own channel and emit nothing here -- their
    output is woven in at synthesize."""
    writer = get_stream_writer()
    intent: IntentDecision = state["intent"]
    book_id = state["book_id"]
    session_id = state["session_id"]

    # Built per-turn from book_id (a thin Chroma client constructor, same
    # cost as the pre-graph one-per-request build) so no non-serializable
    # vectorstore handle has to live in checkpointed state.
    vectorstore = get_vectorstore(book_id)

    if intent.kind == "chapter_summary" and intent.chapter_number is not None:
        writer({"type": "tool_start", "tool": "chapter_summary"})
        retrieval = chapter_summary_retrieval(
            book_id=book_id,
            printed_number=intent.chapter_number,
            session_id=session_id,
            vectorstore=vectorstore,
        )
        writer({"type": "tool_end", "tool": "chapter_summary"})
        if retrieval.sse_payload:
            writer(json.loads(retrieval.sse_payload))
        return {"retrieval": retrieval, "tool_name": "chapter_summary"}

    # NOTE: search / book_overview no longer land here -- they fan out via
    # decompose -> Send -> retrieve_subq -> join (PR4). retrieve_node now
    # only handles the deterministic / non-agent intents.

    if intent.kind == "reading_progress":
        return {"progress_data": compute_reading_progress(book_id, session_id)}

    if intent.kind == "export_notes":
        fmt = intent.export_format or "markdown"
        return {
            "export_info": run_export(
                book_id=book_id, session_id=session_id, format=fmt
            )
        }

    if intent.kind == "smalltalk":
        return {"is_smalltalk": True}

    # Defensive: an unknown kind falls through with no grounding context;
    # synthesize will produce a clean no-context refusal.
    return {}


# ── Node 3b: retrieve fan-out — one ReAct branch per sub-question (PR4) ──────


def route_after_decompose(state: GraphState):
    """Conditional edge after `decompose`. The agent retrieval intents fan
    out one Send -> retrieve_subq per sub-question (width 1 for a simple
    question -- uniform); every other intent goes to the single retrieve
    node. Send carries only what a branch needs (no full turn state)."""
    intent: IntentDecision = state["intent"]
    if intent.kind not in ("search", "book_overview"):
        return "retrieve"
    plan = state.get("plan") or {}
    subqs = plan.get("subquestions") or [_resolved_question(state)]
    book_title = state.get("book_title", "this book")
    return [
        Send("retrieve_subq", {
            "sub_question": sq,
            "book_id": state["book_id"],
            "book_title": book_title,
        })
        for sq in subqs
    ]


def retrieve_subq_node(state: GraphState) -> dict:
    """One fan-out branch: run the bounded ReAct agent on a single
    sub-question and contribute its evidence to the shared (reducer)
    channel. Receives only the Send payload (sub_question + book_id +
    book_title), not the full turn state; assembly happens once in `join`."""
    book_id = state["book_id"]
    sub_question = state["sub_question"]
    docs = gather_evidence(
        message=sub_question,
        book_id=book_id,
        book_title=state.get("book_title", "this book"),
        vectorstore=get_vectorstore(book_id),
    )
    # Tag each doc with its source sub-question so `join` can balance the
    # final context across branches (otherwise one branch's chunks can fill
    # the whole MAX_CONTEXT_DOCS cap and starve the others — a "compare A and
    # B" answer would then cover A well but barely mention B).
    for d in docs:
        if getattr(d, "metadata", None) is not None:
            d.metadata["sub_question"] = sub_question
    return {"evidence": docs}


def _round_robin(queues: list) -> list:
    """Deal one item from each non-empty queue per pass, preserving each
    queue's internal order, until all are drained. Interleaves the groups so
    none monopolizes a downstream cap."""
    queues = [list(q) for q in queues if q]
    out: list = []
    while any(queues):
        for q in queues:
            if q:
                out.append(q.pop(0))
    return out


def _group_by_meta(docs: list, key: str) -> "OrderedDict":
    from collections import OrderedDict

    groups: "OrderedDict[str, list]" = OrderedDict()
    for d in docs:
        val = (getattr(d, "metadata", None) or {}).get(key, "")
        groups.setdefault(val, []).append(d)
    return groups


def _balance_by_subquestion(docs: list) -> list:
    """Two-level balance so the downstream MAX_SYNTH_DOCS cap preserves
    coverage instead of letting the front of the list monopolize it:

      outer -- round-robin by `sub_question` so each fan-out branch survives
               the cap (a "compare A and B" answer must cover both, not just
               whichever branch's evidence happened to sort first);
      inner -- within each branch, round-robin by `section_id` so a single
               branch's multi-section evidence is spread too (a "final three
               chapters" question fetches 3 sections in ONE branch -- without
               this the first section's chunks fill the whole cap and the
               other two are dropped before the synthesizer ever sees them).

    Each (branch, section) queue keeps its internal rank order, so the most
    relevant doc of each group still leads. A single sub-question with a
    single section is a no-op (original order preserved)."""
    by_sq = _group_by_meta(docs, "sub_question")
    # inner: section-diversify each branch's contribution
    branch_queues = []
    for branch_docs in by_sq.values():
        by_sec = _group_by_meta(branch_docs, "section_id")
        branch_queues.append(
            _round_robin(list(by_sec.values())) if len(by_sec) > 1 else list(branch_docs)
        )
    # outer: round-robin across branches (keeps each fan-out branch represented)
    if len(branch_queues) <= 1:
        return branch_queues[0] if branch_queues else list(docs)
    return _round_robin(branch_queues)


def join_node(state: GraphState) -> dict:
    """Gather the fan-out branches' merged evidence and assemble the turn's
    SINGLE RetrievalResult (design §5 `join`: synthesize once over the
    union). Evidence is already deduped across branches by the `evidence`
    reducer; we balance it across sub-questions before the context cap so
    every branch survives. Emits the retrieval_update frame."""
    writer = get_stream_writer()
    book_id = state["book_id"]
    intent: IntentDecision = state["intent"]
    # Close the coarse fan-out panel (only opened for a compound question).
    if (state.get("plan") or {}).get("is_compound"):
        writer({"type": "fanout_end"})
    retrieval = assemble_retrieval_result(
        book_id=book_id,
        session_id=state["session_id"],
        query_text=intent.search_query or state["message"],
        docs=_balance_by_subquestion(state.get("evidence") or []),
        vectorstore=get_vectorstore(book_id),
    )
    if retrieval.sse_payload:
        writer(json.loads(retrieval.sse_payload))
    return {"retrieval": retrieval, "tool_name": "react_retrieve"}


# ── Node 3: synthesize ──────────────────────────────────────────────────────


def synthesize_node(state: GraphState) -> dict:
    """Produce the sage answer and put it on the wire (design §5, KEPT).

    Mirrors the pre-graph sequence exactly: synthesize -> build session
    attribution -> inject per-<fact> data-* anchors -> emit ONE `token`
    frame carrying the full enriched answer. The attribution rebuild in
    finalize is independent, so doing the inline injection here changes
    nothing downstream."""
    writer = get_stream_writer()
    retrieval = state.get("retrieval")

    draft = synthesize_answer(
        # Resolved question folds in any clarify answer (PR3) so the answer
        # commits to the clarified subject instead of re-hedging the ambiguity.
        question=_resolved_question(state),
        book_title=state.get("book_title", "this book"),
        history=state.get("history", []),
        retrieval=retrieval,
        progress_data=state.get("progress_data"),
        export_info=state.get("export_info"),
        is_smalltalk=state.get("is_smalltalk", False),
    )
    _debug_log("synth_response_head", draft.text[:300])

    # Inline <fact> data-* enrichment so the citation card has linkable
    # spans the moment the token frame lands (finalize rebuilds the
    # attribution separately for the retrieval_events rows). Note this
    # includes ONE synchronous mapper-LLM call (gpt-4o-mini) routing each
    # fact to its supporting chunk -- the token frame ships ~1s later in
    # exchange for attribution that is honest about its grounding.
    attribution = build_answer_attribution(state["session_id"])
    enriched_text, fact_payload = inject_fact_attribution(
        draft.text,
        attribution,
        retrieval_docs=(retrieval.docs if retrieval is not None else None),
    )
    writer({"type": "token", "content": enriched_text})

    return {
        "answer_text": enriched_text,
        "is_error_response": draft.is_error_response,
        "fact_attribution": fact_payload,
    }


# ── Node 4: finalize ────────────────────────────────────────────────────────


def _capture_memory_note(state: GraphState) -> None:
    """Fast lane (design §A): silently persist an explicitly-stated user fact.

    The note was produced by classify_intent (no extra LLM call); we write it
    HERE, at finalize, so only a COMMITTED turn captures memory -- a turn
    abandoned at a clarify interrupt (finalize never runs) correctly writes
    nothing. Best-effort: a memory write must never break or delay the user's
    answer, so any failure is swallowed to the debug log."""
    intent = state.get("intent")
    note = getattr(intent, "memory_note", None) if intent else None
    if not note:
        return
    try:
        db.add_memory_note(
            text=note,
            type=getattr(intent, "memory_note_type", None) or "interest",
            source_book_id=state.get("book_id"),
            source_locator=state.get("session_id"),
        )
    except Exception as exc:  # never let memory capture break the turn
        _debug_log("memory_note_error", str(exc))


def finalize_node(state: GraphState) -> dict:
    """Persist the turn and emit the closing frames (design §5, KEPT).

    Token usage is intentionally NOT persisted here (usage_callback=None):
    the graph path accounts usage once at the request level in
    api/chat.py, since a turn can span multiple LLM calls now and
    multiple HTTP requests once interrupt/resume lands (design §7.6)."""
    writer = get_stream_writer()
    # This turn's retrieval event (if any): attribution is persisted onto
    # THIS event only, so earlier turns' per-fact facts (the Reading
    # Map's lit set) are never clobbered by the session-scoped rebuild.
    retrieval = state.get("retrieval")
    current_event_ids = (
        [retrieval.event_id]
        if retrieval is not None and retrieval.event_id else []
    )
    for payload in build_finalize_payloads(
        session_id=state["session_id"],
        user_message=state["message"],
        assistant_response=state.get("answer_text", ""),
        history=state.get("history", []),
        usage_callback=None,
        fact_attribution=state.get("fact_attribution"),
        current_event_ids=current_event_ids,
    ):
        writer(payload)
    # Fast-lane memory capture: silent, best-effort, after the turn is closed.
    _capture_memory_note(state)
    # Bound this thread's checkpoint footprint. Local import avoids a
    # circular dependency (build.py imports node functions from here).
    # Best-effort: prune_thread_checkpoints already swallows its own
    # errors, but we belt-and-suspenders in case something else trips.
    try:
        from core.graph.build import prune_thread_checkpoints
        prune_thread_checkpoints(state["session_id"])
    except Exception as exc:
        _debug_log("checkpoint_prune_error", str(exc))
    # history was appended in place; echo it back so the channel + the
    # final checkpoint reflect the persisted conversation explicitly.
    return {"history": state.get("history", [])}
