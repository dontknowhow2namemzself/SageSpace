"""Chat SSE endpoint + interrupt/resume seam.

Thin driver over the LangGraph chat-turn workflow (core/graph/). The turn

    classify_intent ─► clarify ─► retrieve ─► synthesize ─► finalize

runs inside a compiled `StateGraph` checkpointed by a SqliteSaver
(`thread_id = session_id`). This module's job:

  1. validate the request + load session/history,
  2. drive `graph.stream(stream_mode="custom")`, framing each structured
     payload the nodes emit as one `data: {...}\n\n` SSE frame,
  3. account this turn's token usage at the request level, and
  4. (PR3) bridge the `clarify` human-in-the-loop interrupt to the
     one-directional SSE stream via checkpoint → close → resume (§7).

The interrupt/resume seam (design §7.2, pattern B): when the `clarify`
node `interrupt()`s, the graph checkpoints and `graph.stream` ends. We
detect the pause via `get_state()`, emit a terminal `ask_user` frame, and
close the stream — no long-held connection. The frontend collects the
answer and calls `POST /chat/resume`, which re-enters the SAME thread with
`Command(resume=answer)` and streams the rest. A paused turn is two HTTP
requests stitched by `thread_id`; the checkpoint is the glue.

Expiry (§8 Q2): a clarify interrupt is answerable for an absolute 30 min
(checkpoint timestamp). A late resume safe-degrades — it proceeds with the
conservative default (broad search, no disambiguation) and tells the user
why via a `notice` frame, rather than failing.
"""
import json
import logging
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.callbacks import get_usage_metadata_callback
from langgraph.types import Command

from models.schemas import ChatRequest, ResumeRequest

from core import database as db
from core.graph import get_chat_graph
from core.pipeline.finalize import persist_usage_from_callback


router = APIRouter()
logger = logging.getLogger(__name__)

# A clarify interrupt is answerable for this long (absolute, from the
# interrupt checkpoint's timestamp). Past it, resume safe-degrades to the
# conservative default. Borrowed from web-session norms (design §8 Q1);
# tune here if "resume tomorrow" ever becomes a goal.
_CLARIFY_TTL = timedelta(minutes=30)

# Sliding window applied when loading the persisted conversation into a
# new turn's LangGraph state (= what gets checkpointed). 20 messages
# ≈ 10 user/assistant turns — leaves comfortable headroom for the
# synthesizer's last-4 / intent classifier's last-6 inner slices while
# keeping per-turn checkpoint size bounded as sessions grow. The full
# history stays intact in sessions.conversation_json for sidebar restore.
_HISTORY_WINDOW_MESSAGES = 20

_EXPIRED_MSG = (
    "This clarification waited more than 30 minutes, so I ran a broad search "
    "on your original question. Ask me again if you'd like something more precise."
)
_NO_PENDING_MSG = "This question can no longer be resumed — please ask again."


# ── Routes ────────────────────────────────────────────────────────────────


@router.post("/chat/session")
def create_session(book_id: str):
    book = db.get_book(book_id)
    if not book:
        raise HTTPException(status_code=404, detail="Book not found")
    session_id = db.create_session(book_id)
    return {"session_id": session_id}


@router.get("/chat/session/{session_id}")
def get_session(session_id: str):
    session = db.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return {
        "session_id": session["id"],
        "book_id": session["book_id"],
        "conversation": json.loads(session.get("conversation_json", "[]")),
    }


@router.get("/chat/sessions/{book_id}")
def list_sessions(book_id: str):
    """Session history for the sidebar: newest first, with a first-question
    preview per session."""
    book = db.get_book(book_id)
    if not book:
        raise HTTPException(status_code=404, detail="Book not found")
    return {"sessions": db.list_sessions_for_book(book_id)}


@router.delete("/chat/session/{session_id}")
def delete_session(session_id: str):
    """Delete one conversation: the session row, its retrieval events, and
    the thread's LangGraph checkpoints. The book's digested progress
    (retrieved_chunks) is preserved — deleting a chat does not un-read
    the book."""
    session = db.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    from core.graph.build import gc_checkpoints_for_session
    gc_checkpoints_for_session(session_id)
    db.delete_session(session_id)
    return {"deleted": session_id}


@router.post("/chat")
async def chat(req: ChatRequest):
    # Tag every log line in this request (and every downstream LangGraph
    # node — contextvars propagate through await) with the IDs that
    # uniquely identify this turn. Set as early as possible so a 404
    # below is still attributable.
    from core.observability import book_id_var, session_id_var
    session_id_var.set(req.session_id)
    book_id_var.set(req.book_id)
    logger.info("chat_turn_received", extra={"msg_len": len(req.message or "")})

    book = db.get_book(req.book_id)
    if not book:
        raise HTTPException(status_code=404, detail="Book not found")
    if not book["raptor_status"].startswith("ready"):
        raise HTTPException(status_code=400, detail="Book index not ready yet")
    session = db.get_session(req.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    history = json.loads(session.get("conversation_json", "[]"))
    init_state = {
        "book_id": req.book_id,
        "session_id": req.session_id,
        "message": req.message,
        "book_title": book.get("title") or "this book",
        # Sliding window into the LangGraph state (and therefore the
        # per-turn checkpoint that serializes it): the full conversation
        # lives in conversation_json for sidebar restore, but only the
        # last _HISTORY_WINDOW_MESSAGES enter this turn's working set.
        # Bounds checkpoint DB growth on long sessions and keeps the
        # synth/intent prompts in their tested ranges. Their inner
        # [-4:]/[-6:] slices stay as defense-in-depth.
        "history": history[-_HISTORY_WINDOW_MESSAGES:],
        # Per-turn keys are EXPLICITLY reset: the thread checkpoint keeps
        # the previous turn's state, and a path that skips retrieve (e.g.
        # smalltalk) would otherwise inherit a stale RetrievalResult —
        # finalize would then attach this turn's (empty) facts onto the
        # PREVIOUS turn's event, clobbering its per-fact attribution.
        # (/chat/resume deliberately does NOT reset: a resumed leg must
        # keep its own turn's retrieval.)
        "retrieval": None,
        "fact_attribution": None,
    }
    graph = get_chat_graph()
    config = _turn_config(req.session_id, req.book_id)

    async def event_stream():
        # One usage callback per HTTP leg: every LLM call across nodes rolls
        # up here and is persisted (accumulated) once after the leg drains.
        # A turn that interrupts spans two legs; each persists its own leg's
        # usage, so the session total is the sum (design §7.6).
        with get_usage_metadata_callback() as usage_callback:
            for frame in _stream_leg(graph, init_state, config):
                yield frame
            persist_usage_from_callback(req.session_id, usage_callback)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.post("/chat/resume")
async def chat_resume(req: ResumeRequest):
    """Resume a turn paused at a clarify interrupt with the user's answer."""
    from core.observability import book_id_var, session_id_var
    session_id_var.set(req.session_id)

    session = db.get_session(req.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    book_id_var.set(session["book_id"])

    graph = get_chat_graph()
    config = _turn_config(req.session_id, session["book_id"])
    snap = graph.get_state(config)

    # Idempotent + stale-tolerant: nothing pending (already answered, never
    # asked, or checkpoint GC'd) -> a clear, non-scary notice, not a crash.
    if _interrupt_value(snap) is None:
        return StreamingResponse(
            _notice_only_stream({"type": "notice", "kind": "no_pending",
                                 "message": _NO_PENDING_MSG}),
            media_type="text/event-stream",
        )

    expired = _interrupt_expired(snap)

    async def event_stream():
        with get_usage_metadata_callback() as usage_callback:
            if expired:
                # Safe-degrade: an empty resume value makes the clarify gate
                # fall through without folding a clarification in, so retrieve
                # runs broad on the original question. Tell the user why.
                yield _sse({"type": "notice", "kind": "clarify_expired",
                            "message": _EXPIRED_MSG})
                resume_value = ""
            else:
                resume_value = req.answer
            for frame in _stream_leg(graph, Command(resume=resume_value), config):
                yield frame
            persist_usage_from_callback(req.session_id, usage_callback)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ── Streaming + interrupt helpers ───────────────────────────────────────────


def _stream_leg(graph, graph_input, config):
    """Drive one leg of a turn: yield each node payload as an SSE frame, and
    if the leg ends paused at an interrupt, yield the terminal `ask_user`
    frame. (Also forward-compatible with a second interrupt on the resume
    leg, e.g. PR4's cost-confirm.)"""
    for payload in graph.stream(graph_input, config=config, stream_mode="custom"):
        yield _sse(payload)
    ask = _pending_ask_user(graph, config)
    if ask is not None:
        yield _sse(ask)


def _pending_ask_user(graph, config) -> dict | None:
    """If the graph is paused at an interrupt, the ask_user frame to emit."""
    snap = graph.get_state(config)
    if not snap.next:
        return None
    value = _interrupt_value(snap)
    if not isinstance(value, dict):
        return None
    return {"type": "ask_user", **value}


def _interrupt_value(snap):
    """The payload of the pending interrupt (if any) on a state snapshot."""
    for task in getattr(snap, "tasks", ()) or ():
        interrupts = getattr(task, "interrupts", ()) or ()
        if interrupts:
            return interrupts[0].value
    return None


def _interrupt_expired(snap) -> bool:
    """True if the interrupt checkpoint is older than the clarify TTL.
    Lenient on any parse failure (treat as not-expired -> let resume proceed)."""
    created = getattr(snap, "created_at", None)
    if not created:
        return False
    try:
        dt = created if isinstance(created, datetime) else datetime.fromisoformat(created)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return datetime.now(timezone.utc) - dt > _CLARIFY_TTL
    except Exception:
        return False


def _turn_config(session_id: str, book_id: str) -> dict:
    """LangGraph run config. thread_id = session_id stitches a turn's
    checkpoints (and an interrupt/resume across two requests); run_name /
    tags / metadata make the turn legible in LangSmith."""
    return {
        "configurable": {"thread_id": session_id},
        "run_name": "sage_chat_turn",
        "tags": ["sage_chat"],
        "metadata": {"book_id": book_id, "session_id": session_id},
    }


async def _notice_only_stream(notice: dict):
    """A minimal terminal stream: one notice + stream_end (used when there is
    nothing to resume, so the frontend's reader loop still closes cleanly)."""
    yield _sse(notice)
    yield _sse({"type": "stream_end"})


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
