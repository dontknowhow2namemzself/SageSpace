"""Bounded ReAct retrieval agent (PR2).

The `retrieve` node's brain for free-form questions: a small, EXPLICIT
LangGraph `agent <-> tools` loop that decides, per query, which
retrieval tool(s) to call (semantic / keyword / get_chapter /
expand_neighbors), observes the results, and stops when it has enough.

This is deliberately a *custom, bounded* agent rather than a free-roaming
one (design §1, the PR5 lesson):
  * the skeleton is two explicit nodes with an explicit looping edge,
  * the loop has a HARD iteration cap (MAX_ITERATIONS) -- on hitting it
    we proceed with whatever evidence was gathered, never spin,
  * the agent only *gathers* evidence; it never writes the answer (that
    stays in the separate synthesize node).

Its only job is to return a deduped list of evidence Documents (tagged
with metadata['origin']); the retrieve node turns that into the turn's
single RetrievalResult via pipeline.retrieve.assemble_retrieval_result.

LangSmith sees the whole trajectory: each agent LLM call (with its
tool-call decisions) and each tools step -- the assignment's ReAct-trace
deliverable.
"""
from __future__ import annotations

import logging
from typing import Annotated, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
import os

from core.retrieval_tools import (
    tool_expand_neighbors,
    tool_get_chapter,
    tool_keyword_search,
    tool_semantic_search,
)


logger = logging.getLogger(__name__)

# Hard stop on agent<->tools cycles (controllability, the PR5 lesson). On
# hitting it the loop ends and we synthesize from whatever was gathered.
MAX_ITERATIONS = 5
# Passages shown back to the agent per observation (keeps its context lean).
_OBSERVE_LIMIT = 6


# ── Tool argument schemas (what the LLM controls; book_id/vectorstore are
#    injected by the tools node, never by the model) ──────────────────────────


class semantic_search(BaseModel):
    """Vector search for conceptual / thematic questions -- descriptions,
    motives, themes, "what/why/how" — where meaning matters more than exact
    wording. Weak on bare proper nouns; use keyword_search for those."""
    query: str = Field(..., description="A focused search query describing the "
                       "entity, event, or theme to find (not the user's literal "
                       "sentence).")


class keyword_search(BaseModel):
    """Exact-term lookup over the book, like Ctrl+F. Best for proper nouns and
    coined terms — character names, place names, specific phrases — which
    vector search embeds poorly. Pass the literal name(s)/term(s)."""
    terms: str = Field(..., description="The exact name(s) or term(s) to find, "
                       "e.g. 'Cheshire Cat' or 'Mock Turtle'.")


class get_chapter(BaseModel):
    """Pull a whole chapter's summary + passages by its printed number. Use
    when the user references a chapter by number (e.g. 'what happens in
    chapter 6')."""
    printed_number: int = Field(..., description="The chapter's printed number, "
                                "e.g. 6.")


class expand_neighbors(BaseModel):
    """Widen context around a passage you already retrieved. Pass a chunk_id
    seen in a previous observation to pull the passages immediately around it
    in reading order."""
    chunk_id: str = Field(..., description="A chunk_id from a previous "
                          "observation to expand context around.")


_TOOL_SCHEMAS = [semantic_search, keyword_search, get_chapter, expand_neighbors]


_SYSTEM_PROMPT = """\
You are the retrieval brain for a Q&A system about the book "{book_title}".
Your ONLY job is to GATHER the passages that will let another component
answer the user's question. You do NOT write the answer yourself.

Tools:
- semantic_search(query): meaning-based vector search. Use for conceptual /
  thematic questions (descriptions, motives, themes, what/why/how).
- keyword_search(terms): exact-term "Ctrl+F". Use when the question hinges on
  a specific PROPER NOUN or coined term (a character, place, or exact phrase),
  because vector search is weak on those. Using BOTH semantic_search and
  keyword_search is encouraged when a question names an entity AND asks about
  it conceptually.
- get_chapter(printed_number): when the user references a chapter by number.
- expand_neighbors(chunk_id): widen context around a strong hit (pass a
  chunk_id from a prior observation) when you need the surrounding text.

How to work:
1. Pick the best tool for the question and call it. You MUST make at least one
   tool call before stopping.
2. Read each observation. If you now have enough relevant passages, STOP by
   replying with a brief note that you're done (and NO tool call).
3. If a named entity wasn't found by semantic search, try keyword_search. If a
   strong hit needs more surrounding context, use expand_neighbors.
4. Be efficient — a few well-chosen calls, not an exhaustive sweep.

Do not answer the user's question; only gather evidence, then stop."""


# ── Graph state ──────────────────────────────────────────────────────────────


class ReactState(TypedDict, total=False):
    messages: Annotated[list, add_messages]
    evidence: list          # accumulated Documents (deduped, origins merged)
    book_id: str
    vectorstore: object     # Chroma handle; ephemeral subgraph -> never serialized
    iterations: int


# ── Nodes ────────────────────────────────────────────────────────────────────


def _agent_llm() -> ChatOpenAI:
    """Cheap retrieval-tier model (design §8 Q3), with tools bound."""
    return ChatOpenAI(
        model="openai/gpt-4o-mini",
        openai_api_base=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        temperature=0,
    ).bind_tools(_TOOL_SCHEMAS)


def agent_node(state: ReactState) -> dict:
    """One reasoning step: the LLM looks at the question + observations so
    far and either requests tool(s) or stops."""
    ai = _agent_llm().invoke(state["messages"])
    return {"messages": [ai]}


def tools_node(state: ReactState) -> dict:
    """Execute every tool the agent just requested, accumulate the evidence,
    and feed compact observations back. Emits tool_start/tool_end to the
    parent SSE stream so the UI + LangSmith show the trajectory."""
    writer = get_stream_writer()
    last = state["messages"][-1]
    book_id = state["book_id"]
    vectorstore = state.get("vectorstore")
    # The sub-question this ReAct branch is working on (the HumanMessage
    # gather_evidence seeded). Threaded into get_chapter so the chapter's
    # limited level-0 slots go to the most relevant chunks.
    subquestion = next(
        (m.content for m in state["messages"] if isinstance(m, HumanMessage)), ""
    )

    tool_messages: list = []
    new_docs: list[Document] = []
    for call in getattr(last, "tool_calls", None) or []:
        name = call["name"]
        args = call.get("args") or {}
        writer({"type": "tool_start", "tool": name})
        try:
            docs = _dispatch_tool(name, args, book_id, vectorstore, subquestion)
        except Exception as exc:  # a tool error must not kill the turn
            logger.warning("retrieve tool %s failed: %s", name, str(exc)[:160])
            docs = []
        writer({"type": "tool_end", "tool": name})
        new_docs.extend(docs)
        tool_messages.append(
            ToolMessage(content=_format_observation(name, docs), tool_call_id=call["id"])
        )

    merged = _merge_evidence(state.get("evidence", []), new_docs)
    # Count tool ROUNDS (the hard cap): at most MAX_ITERATIONS of them.
    return {
        "messages": tool_messages,
        "evidence": merged,
        "iterations": state.get("iterations", 0) + 1,
    }


def should_continue(state: ReactState) -> str:
    """Loop to tools only if the agent asked for some AND we're under the
    hard cap on tool rounds; otherwise end and synthesize from what we
    have (never spin -- the PR5 lesson)."""
    last = state["messages"][-1]
    wants_tools = bool(getattr(last, "tool_calls", None))
    if wants_tools and state.get("iterations", 0) < MAX_ITERATIONS:
        return "tools"
    return END


# ── Dispatch + observation formatting ────────────────────────────────────────


def _dispatch_tool(
    name: str, args: dict, book_id: str, vectorstore, subquestion: str = ""
) -> list[Document]:
    if name == "semantic_search":
        return tool_semantic_search(args.get("query", ""), vectorstore)
    if name == "keyword_search":
        return tool_keyword_search(args.get("terms", ""), book_id, vectorstore)
    if name == "get_chapter":
        try:
            n = int(args.get("printed_number"))
        except (TypeError, ValueError):
            return []
        return tool_get_chapter(n, book_id, vectorstore, query=subquestion)
    if name == "expand_neighbors":
        return tool_expand_neighbors(args.get("chunk_id", ""), book_id, vectorstore)
    logger.warning("retrieve agent requested unknown tool: %s", name)
    return []


def _format_observation(name: str, docs: list[Document]) -> str:
    """Compact, chunk_id-bearing observation so the agent can decide what to
    do next (and target expand_neighbors) without bloating its context."""
    if not docs:
        return f"{name}: no passages found."
    lines = [f"{name}: {len(docs)} passage(s)."]
    for i, d in enumerate(docs[:_OBSERVE_LIMIT], 1):
        label = d.metadata.get("section_label") or f"Chapter {d.metadata.get('chapter', '?')}"
        cid = d.metadata.get("chunk_id", "?")
        snippet = " ".join((d.page_content or "")[:160].split())
        lines.append(f"{i}. [{label}] (chunk_id={cid}) {snippet}…")
    return "\n".join(lines)


# ── Evidence merge (dedup by chunk_id, merge semantic∩keyword -> "both") ──────


def _merge_evidence(current: list, new: list) -> list:
    merged = list(current or [])
    pos_by_cid = {
        d.metadata.get("chunk_id"): i
        for i, d in enumerate(merged)
        if d.metadata.get("chunk_id")
    }
    for d in (new or []):
        cid = d.metadata.get("chunk_id")
        if cid and cid in pos_by_cid:
            ex = merged[pos_by_cid[cid]]
            ex.metadata["origin"] = _merge_origin(
                ex.metadata.get("origin"), d.metadata.get("origin")
            )
        else:
            if cid:
                pos_by_cid[cid] = len(merged)
            merged.append(d)
    return merged


def _merge_origin(a: str | None, b: str | None) -> str:
    """A chunk found by both keyword and a non-keyword tool is the design's
    'both'; otherwise keep the first seen."""
    if a == b or not b:
        return a or "semantic"
    if not a:
        return b
    if "keyword" in (a, b):
        return "both"
    return a


# ── Build + public entrypoint ────────────────────────────────────────────────


_agent_graph = None


def get_retrieve_agent():
    """Compiled agent<->tools subgraph singleton (no checkpointer: its state
    is transient working memory for one turn)."""
    global _agent_graph
    if _agent_graph is None:
        g = StateGraph(ReactState)
        g.add_node("agent", agent_node)
        g.add_node("tools", tools_node)
        g.add_edge(START, "agent")
        g.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
        g.add_edge("tools", "agent")
        # checkpointer=False: this subgraph is ephemeral working memory for one
        # turn. Without it the subgraph would INHERIT the main graph's
        # SqliteSaver (via ambient config) and try to checkpoint its state --
        # which holds the non-serializable Chroma handle -- and would also
        # pollute the chat thread's checkpoint namespace. False opts out cleanly.
        _agent_graph = g.compile(checkpointer=False)
    return _agent_graph


def gather_evidence(
    message: str, book_id: str, book_title: str, vectorstore
) -> list[Document]:
    """Run the bounded ReAct loop for `message` and return the gathered
    evidence Documents. Always returns at least one semantic pass' worth of
    docs (a safety net) so a search turn never ends up with nothing to
    ground on if the agent stops too early."""
    agent = get_retrieve_agent()
    init: ReactState = {
        "messages": [
            SystemMessage(_SYSTEM_PROMPT.format(book_title=book_title or "this book")),
            HumanMessage(message),
        ],
        "evidence": [],
        "book_id": book_id,
        "vectorstore": vectorstore,
        "iterations": 0,
    }
    # recursion_limit allows MAX_ITERATIONS full agent+tools cycles plus the
    # final agent stop, as a backstop beneath our own iteration cap.
    out = agent.invoke(init, {"recursion_limit": 2 * MAX_ITERATIONS + 2})
    evidence = out.get("evidence") or []
    if not evidence:
        evidence = tool_semantic_search(message, vectorstore)
    return evidence
