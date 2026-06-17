"""Graph state — the typed payload that flows through the chat-turn graph.

This is the LangGraph port of the implicit "state" that the pre-graph
`api/chat.py` event loop carried in local variables (intent, retrieval,
answer text). Making it an explicit, typed channel object is what the
training assignment means by *state flows through nodes & edges*; the
SqliteSaver checkpointer (see `build.py`) then persists this object per
super-step, which is the durable version of that same requirement.

PR1 keeps the field set deliberately minimal — exactly what the four
ported nodes (classify_intent / retrieve / synthesize / finalize) read
and write. Later PRs extend it (plan / sub-questions / clarification /
user_memory per design §4); the channels added here are forward
compatible with that table.

Every field is optional (`total=False`): a turn seeds only the entry
inputs, and each node contributes its own channel(s). No reducers are
needed in PR1 because every channel is written by exactly one node, so
the default last-write-wins behavior is correct.

NOTE (LangGraph gotcha): a node name must never equal a state-channel
key, or `StateGraph.add_node` raises "already being used as a state
key". That is why the nodes are named `classify_intent` / `retrieve`
(not `intent` / `retrieval`).
"""
from __future__ import annotations

from typing import Annotated, TypedDict

from core.pipeline.types import IntentDecision, RetrievalResult


def merge_evidence(current: list, new: list) -> list:
    """Reducer for the fan-out `evidence` channel (PR4): concatenate the
    parallel sub-question branches' Documents, deduped by chunk_id (first
    wins). Each branch already did its own internal dedup / origin-merge in
    the ReAct agent; this only stitches the branches together. Robust to a
    None/absent accumulator (the first branch seeds it)."""
    merged = list(current or [])
    seen = {
        d.metadata.get("chunk_id")
        for d in merged
        if getattr(d, "metadata", None)
    }
    for d in (new or []):
        meta = getattr(d, "metadata", None) or {}
        cid = meta.get("chunk_id")
        if cid and cid in seen:
            continue
        if cid:
            seen.add(cid)
        merged.append(d)
    return merged


class GraphState(TypedDict, total=False):
    # ── Entry inputs (seeded by api/chat.py before the first super-step) ──
    book_id: str
    session_id: str
    message: str
    book_title: str
    history: list[dict]

    # ── classify_intent node ──
    intent: IntentDecision

    # ── clarify node (HITL①, PR3) ──
    # The user's answer to the clarifying question, folded back in after
    # interrupt()/resume. retrieve reads it to disambiguate the query.
    # {"question": str, "answer": str} | absent when no clarification happened.
    clarification: dict | None

    # ── plan node (PR4) ──
    # Decomposition of a compound question for the retrieve fan-out.
    # {"is_compound": bool, "subquestions": [str, ...]} (always >=1 sub-q).
    plan: dict | None
    # Rough fan-out cost (informational; feeds Token-Usage + a future
    # cost-confirm gate). {"n_subq", "est_calls", "est_latency_s"}.
    cost_estimate: dict | None

    # ── retrieve fan-out (PR4) ──
    # Per-branch input carried by each Send (the sub-question that branch
    # retrieves). Lives in the schema so Send payloads validate.
    sub_question: str
    # Evidence Documents gathered across ALL fan-out branches, merged +
    # deduped by the reducer; the join node assembles the turn's single
    # RetrievalResult from it.
    evidence: Annotated[list, merge_evidence]

    # ── retrieve node (exactly one of these is populated per turn, chosen
    #    by intent.kind — mirrors the pre-graph dispatch in chat.py) ──
    retrieval: RetrievalResult | None
    progress_data: dict | None
    export_info: dict | None
    is_smalltalk: bool
    tool_name: str | None          # which tool ran, for LangSmith/debug clarity

    # ── synthesize node ──
    answer_text: str               # visible answer AFTER <fact> data-* injection
    is_error_response: bool
    # Normalized per-fact attribution payload from inject_fact_attribution
    # (facts[i].chunk_ids = each fact's matched chunks). finalize merges it
    # into the attached/emitted answer_attribution; the Reading Map derives
    # its "lit" set from these cited chunks (not from raw retrieval hits).
    fact_attribution: dict | None
