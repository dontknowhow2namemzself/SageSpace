"""Assemble + compile the chat-turn graph.

This is the LangGraph substrate that replaces the hand-rolled node
sequence in the old `api/chat.py`. The skeleton (design §3):

    classify_intent ─► clarify ─► decompose ─┬─► retrieve_subq ×N ─► join ─┐
                                             └─► retrieve ───────────────► synthesize ─► finalize

PR1 ported the four-node chain; PR3 added `clarify` (HITL interrupt/resume);
PR4 added `decompose` + the Send fan-out: search/book_overview run one
bounded-ReAct `retrieve_subq` per sub-question (width 1 for simple Qs),
merged in `join`; every other intent takes the single `retrieve`. Both
paths converge at `synthesize` (one coherent answer over the union).
`retrieve` is the bounded ReAct agent (PR2). Still to come: the `Send`
fan-out that runs `retrieve` per sub-question + a `join` (PR4 step 2).

Checkpointer
------------
Compiled with the **synchronous** `SqliteSaver` (design §7.3). Two
deliberate choices:

  * Sync, not async: `langgraph-checkpoint-sqlite`'s async saver is
    broken against the installed `aiosqlite` (it calls a removed
    `Connection.is_alive()`); the sync saver uses stdlib `sqlite3` and
    is never affected. `graph.stream(stream_mode="custom")` still
    streams frames incrementally, so nothing is lost.
  * Its own DB file (`sagespace_checkpoints.db`, sibling of the app
    DB), connection opened `check_same_thread=False` because LangGraph
    runs nodes on worker threads. Keeping it off the app's
    connection-per-call file avoids SQLite write-lock contention.
    `thread_id = session_id`, so GC-by-ownership (design §8 Q1) rides
    the existing session/book delete cascade.

Even though PR1 never interrupts (nothing resumes yet), wiring the
checkpointer now is what makes the assignment's "state flows through
nodes & edges" durable, and it is the foundation PR3's interrupt/resume
seam builds on.
"""
from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph


logger = logging.getLogger(__name__)

from core import database as db
from core.graph.nodes import (
    clarify_node,
    classify_intent_node,
    finalize_node,
    join_node,
    plan_node,
    retrieve_node,
    retrieve_subq_node,
    route_after_decompose,
    synthesize_node,
)
from core.graph.state import GraphState


def build_chat_graph(checkpointer=None):
    """Build + compile the four-node chat graph.

    `checkpointer` is injectable so tests can pass an isolated saver (or
    None for a checkpoint-free run). Production uses `get_chat_graph()`,
    which supplies the shared SqliteSaver singleton.
    """
    graph = StateGraph(GraphState)
    graph.add_node("classify_intent", classify_intent_node)
    graph.add_node("clarify", clarify_node)
    # Node name 'decompose' (not 'plan') -- a node name must never equal a
    # state-channel key, and the result lands in state['plan'] (same rule
    # that makes the nodes classify_intent/retrieve, not intent/retrieval).
    graph.add_node("decompose", plan_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("retrieve_subq", retrieve_subq_node)
    graph.add_node("join", join_node)
    graph.add_node("synthesize", synthesize_node)
    graph.add_node("finalize", finalize_node)

    # PR3: clarify (HITL①) sits between intent and retrieve -- passes through
    # for clear questions, interrupt()s for ambiguous ones (design §7).
    # PR4: plan decomposes compound questions; the Send fan-out it feeds is
    # wired in step 2 (for now retrieve still runs on the whole question).
    graph.add_edge(START, "classify_intent")
    graph.add_edge("classify_intent", "clarify")
    graph.add_edge("clarify", "decompose")
    # PR4: decompose fans out one Send -> retrieve_subq per sub-question for
    # search/book_overview (width 1 for simple questions), joined into a
    # single RetrievalResult; every other intent takes the single retrieve
    # node. Both paths converge at synthesize.
    graph.add_conditional_edges(
        "decompose", route_after_decompose, ["retrieve_subq", "retrieve"]
    )
    graph.add_edge("retrieve_subq", "join")
    graph.add_edge("join", "synthesize")
    graph.add_edge("retrieve", "synthesize")
    graph.add_edge("synthesize", "finalize")
    graph.add_edge("finalize", END)

    return graph.compile(checkpointer=checkpointer)


def checkpoint_db_path() -> Path:
    """Checkpoint DB path: sibling of the app DB. Derived from
    `db.DB_PATH` at call time so test monkeypatching is respected."""
    return Path(db.DB_PATH).with_name("sagespace_checkpoints.db")


def gc_checkpoints_for_book(book_id: str) -> int:
    """Drop LangGraph checkpoints for every session (thread) of a book —
    GC by ownership (design §8 Q1). thread_id = session_id, so deleting a
    book should also drop its turns' checkpoints rather than orphan them in
    the checkpoint DB. Best-effort: never raises into the delete path;
    returns how many threads were cleared.

    Must be called BEFORE the session rows are deleted (it enumerates them)."""
    if not checkpoint_db_path().exists():
        return 0  # no chat ever happened -> no checkpoint DB to touch
    try:
        saver = get_chat_graph().checkpointer
    except Exception:
        return 0
    cleared = 0
    for session_id in db.get_session_ids_for_book(book_id):
        try:
            saver.delete_thread(session_id)
            cleared += 1
        except Exception:
            pass
    return cleared


def gc_checkpoints_for_session(session_id: str) -> bool:
    """Drop LangGraph checkpoints for ONE session (thread) — the per-session
    sibling of gc_checkpoints_for_book, used when the user deletes a single
    conversation from the history sidebar. Best-effort: never raises."""
    if not checkpoint_db_path().exists():
        return False
    try:
        get_chat_graph().checkpointer.delete_thread(session_id)
        return True
    except Exception:
        return False


# Per-thread retention cap. Each turn writes ~7-10 checkpoints, so 30
# keeps roughly the last 3-4 turns' worth — enough that resume from a
# pending clarify interrupt (always the LATEST checkpoint and its
# parent chain) is unaffected, while bounding per-thread row count on
# long sessions.
_CHECKPOINT_RETENTION = 30


class _NullLock:
    """No-op context manager for checkpointers that expose no .lock."""
    def __enter__(self): return self
    def __exit__(self, *_): return False


_NULL_LOCK = _NullLock()


def prune_thread_checkpoints(
    session_id: str, keep_last: int = _CHECKPOINT_RETENTION
) -> int:
    """Cap one thread's checkpoint footprint to the last `keep_last` rows.

    Called from finalize_node after every completed turn so that long
    sessions cannot grow the checkpoint DB indefinitely. Deletes the
    matching `writes` rows first, then the older `checkpoints` rows
    themselves — `writes` references `checkpoints` by
    (thread_id, checkpoint_id) and we never want a dangling write.

    Why resume stays safe: an interrupt resume always re-enters from
    the LATEST pending checkpoint and walks its parent chain. The
    latest chain is by definition inside the retention window; the
    rows dropped are completed turns' debris.

    Returns the number of checkpoint rows deleted. Best-effort: any
    failure is logged and swallowed — the user's turn already
    finalized and must not be affected by a maintenance error.
    """
    if keep_last <= 0:
        return 0
    if not checkpoint_db_path().exists():
        return 0
    try:
        saver = get_chat_graph().checkpointer
        conn = saver.conn
        keep_clause = (
            "checkpoint_id IN ("
            " SELECT checkpoint_id FROM checkpoints "
            " WHERE thread_id = ? "
            " ORDER BY checkpoint_id DESC LIMIT ?"
            ")"
        )
        # SqliteSaver serializes its own writes via .lock; share it so
        # this prune cannot interleave with an in-flight checkpoint
        # put from a parallel turn.
        with getattr(saver, "lock", _NULL_LOCK):
            conn.execute(
                f"DELETE FROM writes WHERE thread_id = ? AND NOT {keep_clause}",
                (session_id, session_id, keep_last),
            )
            cur = conn.execute(
                f"DELETE FROM checkpoints WHERE thread_id = ? AND NOT {keep_clause}",
                (session_id, session_id, keep_last),
            )
            conn.commit()
            return cur.rowcount or 0
    except Exception:
        logger.exception(
            "prune_thread_checkpoints failed for session %s", session_id
        )
        return 0


_graph = None


def get_chat_graph():
    """Process-wide compiled graph singleton, backed by the shared
    SqliteSaver. Built lazily on first chat turn; the underlying sqlite3
    connection lives for the process lifetime (closed on exit)."""
    global _graph
    if _graph is None:
        conn = sqlite3.connect(str(checkpoint_db_path()), check_same_thread=False)
        _graph = build_chat_graph(checkpointer=SqliteSaver(conn))
    return _graph
