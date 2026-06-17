"""Pipeline contracts.

Every chat turn flows through a fixed sequence of nodes
(intent -> retrieve -> synthesize -> finalize). The dataclasses here
are the typed payloads that move between them. Each node consumes one
input shape and produces one output shape; there is no ContextVar,
implicit global state, or LangChain Tool magic in this layer.

A turn does not require every node:
  * smalltalk and reading_progress / export_notes routes skip retrieve.
  * synthesize is always invoked so every turn produces a sage-persona
    answer with optional <fact>/<commentary> tags.
  * finalize is always invoked so every turn emits the standard SSE
    closing frames (attribution / done / stream_end).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


IntentKind = Literal[
    "chapter_summary",   # "Tell me about chapter N" -> needs printed_number
    "search",            # Free-form question that needs retrieval
    "book_overview",     # "What is this book about" -> wide-net retrieval
    "reading_progress",  # "How much have I read"
    "export_notes",      # "Save these notes"
    "smalltalk",         # Off-topic; refuse politely, never claim book facts
]

ExportFormat = Literal["pdf", "markdown"]


@dataclass
class IntentDecision:
    """Output of pipeline.intent.classify_intent.

    `kind` is mandatory; the other fields are populated only when their
    intent applies. Downstream dispatch reads only the fields relevant
    to the chosen kind.
    """
    kind: IntentKind
    # search / book_overview: the retrieval query the LLM rephrased the
    # user message into (or the raw user message when no rephrase needed).
    search_query: str | None = None
    # chapter_summary: the author-printed chapter number (1 for "Chapter
    # 1" / "第一章" / "Chapter One" / "I"). Resolves via PR4's
    # sections.printed_number column.
    chapter_number: int | None = None
    # export_notes: which file format the user asked for.
    export_format: ExportFormat | None = None

    # clarify (HITL①, PR3 — design §6): the intent LLM (方案甲) also judges
    # whether a free-form question is too ambiguous to retrieve well (e.g. an
    # unresolved pronoun / entity — "他后来怎么样了?"). When `ambiguous`, the
    # clarify node interrupt()s with `clarify_question` + `clarify_options`
    # and the turn pauses until the user answers (design §7). Only ever set
    # for search / book_overview; the other kinds have nothing to disambiguate.
    ambiguous: bool = False
    clarify_question: str | None = None
    clarify_options: list[str] = field(default_factory=list)
    clarify_multi: bool = False

    # Memory fast lane (memory-system-design.md §A): the SAME intent LLM call
    # also judges whether the user EXPLICITLY stated a durable fact/interest
    # about themselves this turn ("叫我小王", "我在啃斯多葛") -- zero extra calls.
    # `memory_note` is that one short note (in the user's language) or None for
    # the ~majority of turns; `memory_note_type` is fact|interest. finalize
    # writes it silently to memory_notes; nothing here is user-visible.
    memory_note: str | None = None
    memory_note_type: str | None = None


@dataclass
class RetrievalSource:
    """One citation in the SSE retrieval_update payload + future
    answer_attribution. Mirrors the existing source_refs shape so the
    frontend's citation card renders unchanged.
    """
    label: str
    chunk_id: str
    text: str                       # snippet for the citation card
    chapter: int = 0
    page: int = 0
    # Canonical anchor fields (present when resolve_citation succeeded).
    citation_id: str | None = None
    section_id: str | None = None
    section_label: str | None = None
    primary_block_id: str | None = None
    block_ids: list[str] = field(default_factory=list)
    retrieved_layer: Literal["raw", "raptor"] | None = None


@dataclass
class RetrievalResult:
    """Output of pipeline.retrieve.run_retrieval / chapter_summary_text.

    `docs` carries the actual chunk text + metadata the synthesizer
    feeds the LLM. `sources` is the UI-facing slice (top-N with
    citation anchors). `sse_payload` is the pre-formatted SSE frame
    the chat router emits as retrieval_update after the retrieval
    side-effects have already been written.
    """
    docs: list[dict]                # {text, chunk_id, section_label, ...}
    sources: list[RetrievalSource]
    event_id: str | None = None     # retrieval_events row id, if persisted
    sse_payload: str | None = None  # ready-to-yield SSE frame


@dataclass
class AnswerDraft:
    """Output of pipeline.synthesize.synthesize_answer.

    `text` is the user-visible answer with <fact>/<commentary> tags but
    WITHOUT data-* attribute injection -- injection happens in the graph
    synthesize node (core/graph/nodes.py:synthesize_node), which
    enriches the text before emitting the token frame.
    `is_error_response` is retained for API compatibility (the online
    faithfulness probe that consumed it was retired in v1.0).
    """
    text: str
    is_error_response: bool = False
