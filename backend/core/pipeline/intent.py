"""Intent classifier (pipeline node 1).

One LLM call per turn that maps the user message into a structured
IntentDecision. Replaces the regex-and-dispatch scaffolding that lived
in api/chat.py through PR3 (the _CHAPTER_NUM_RE / _READING_PROGRESS_RE
/ _EXPORT_NOTES_RE / _BOOK_OVERVIEW_RE / _CHAPTER_SUMMARY_INTENT_RE
blob), and removes the brittle pinball between "讲什么" / "讲了什么"
phrasings that kept producing user-reported misroutes.

Why an LLM:
  * Regex was already collapsing under its own weight by PR3's A1
    hotfix -- every new phrasing the user reported needed a new
    branch in the alternation.
  * gpt-4o-mini with structured output returns a clean JSON object
    in ~200ms and ~70 tokens. Cost per turn ~$0.00005.
  * The model handles the chapter-number normalization (Arabic /
    Roman / Chinese / English-word) that the parser had to do
    manually for the regex path.

Determinism: temperature=0, JSON schema enforced via
ChatOpenAI.with_structured_output. The judge prompt explicitly lists
the 6 intent kinds and gives one positive example per kind. If the
model returns something unexpected we fail loud (the IntentDecision
dataclass field types reject it on construction).
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import Literal as _Literal

from core.pipeline.types import IntentDecision


logger = logging.getLogger(__name__)


# Pydantic schema mirrors IntentDecision so we can use with_structured_output
# and still hand back our internal dataclass at the API boundary.
class _IntentSchema(BaseModel):
    """The user's intent in this chat turn.

    Choose exactly one `kind`. Fill the field that matches:
      * chapter_summary -> chapter_number (the printed number the
        user asked for; convert "Chapter Five" / "第五章" / "CHAPTER V"
        all to 5)
      * search          -> search_query (a 4-10 word retrieval query
        capturing the user's actual question, NOT the user's exact
        wording verbatim if it is conversational)
      * book_overview   -> leave others null
      * reading_progress, export_notes -> leave search/chapter null;
        for export_notes set export_format if the user specified one
      * smalltalk       -> the user asked something off-topic about
        this book (weather, your name, etc.). Leave all other fields null.
    """
    kind: _Literal[
        "chapter_summary",
        "search",
        "book_overview",
        "reading_progress",
        "export_notes",
        "smalltalk",
    ] = Field(..., description="The intent category for this turn.")
    chapter_number: int | None = Field(
        default=None,
        description=(
            "The author-printed chapter number for kind='chapter_summary'. "
            "Convert Arabic / Roman / Chinese / English-word forms to int. "
            "Null for any other kind."
        ),
    )
    search_query: str | None = Field(
        default=None,
        description=(
            "A 4-10 word retrieval query for kind='search' or "
            "kind='book_overview'. Phrase as keywords describing the "
            "entity / event / theme the user wants -- not as a question. "
            "Null for any other kind."
        ),
    )
    export_format: _Literal["pdf", "markdown"] | None = Field(
        default=None,
        description=(
            "File format for kind='export_notes'. Default to 'markdown' "
            "unless the user explicitly said 'pdf'. Null for any other kind."
        ),
    )
    # Ambiguity judgement (HITL① clarify, 方案甲: folded into this one call
    # so the clarify node stays a side-effect-free interrupt gate).
    ambiguous: bool = Field(
        default=False,
        description=(
            "True ONLY for kind=search/book_overview when the question cannot "
            "be retrieved reliably without asking the user one thing -- e.g. an "
            "unresolved pronoun or referent ('他/她/it/they' with no antecedent), "
            "or a vague target ('that part', 'the other one'). FIRST resolve any "
            "pronoun from the Recent conversation: if a prior turn already "
            "established who 'he/she/it' is, it is NOT ambiguous -> false. Do "
            "NOT set true just because a question is broad; only when a SHORT "
            "clarification would materially change what to search for. Always "
            "false for chapter_summary / reading_progress / export_notes / smalltalk."
        ),
    )
    clarify_question: str | None = Field(
        default=None,
        description=(
            "When ambiguous=true: one short question to ask the user. It MUST "
            "be written in the SAME language as the user's CURRENT message -- "
            "English question -> ask in English ('Who do you mean by \"she\"?'); "
            "Chinese question -> ask in Chinese ('你指的是谁?'). Null otherwise."
        ),
    )
    clarify_options: list[str] = Field(
        default_factory=list,
        description=(
            "When ambiguous=true: 2-4 concrete answer options to pick from "
            "(likely referents from the conversation/book context), in the SAME "
            "language as the user's current message -- e.g. for an English "
            "question use English names ('Mad Hatter', 'White Rabbit'), not "
            "Chinese. Empty when not ambiguous or no good options exist (the "
            "user can still type a free-text answer)."
        ),
    )
    clarify_multi: bool = Field(
        default=False,
        description="When ambiguous=true: whether the user may pick MORE than "
                    "one option. Usually false.",
    )
    # Memory fast lane (independent of `kind` -- judged on every turn).
    memory_note: str | None = Field(
        default=None,
        description=(
            "Did the user EXPLICITLY state a DURABLE fact or preference about "
            "THEMSELVES this turn -- their name, what they're working on / "
            "preparing, a subject they're into / studying ('叫我小王', '我在准备"
            "领导力演讲', 'I'm really into Stoicism')? If yes -> one short note "
            "capturing it, ALWAYS written in ENGLISH (translate it even if the "
            "user wrote in another language). Otherwise null. This is "
            "NOT about the book's content and NOT about transient/this-question "
            "state -- only lasting facts the user would expect remembered. The "
            "VAST MAJORITY of turns are null; only fill it on a clear signal."
        ),
    )
    memory_note_type: _Literal["fact", "interest"] | None = Field(
        default=None,
        description=(
            "When memory_note is set: 'interest' for a topic/subject the user "
            "likes or is studying; 'fact' for anything else about them (name, "
            "role, what they're working on). Null when memory_note is null."
        ),
    )


_SYSTEM_PROMPT = """\
You classify a user's chat-turn intent in a book-grounded conversation.
The user is talking to a sage who has read one specific book. Return
ONE intent kind from the schema. Be strict; do not invent kinds.

Examples (one per kind):

  user: "讲讲第5章" / "Tell me about Chapter V" / "What's in chapter 5"
  -> kind=chapter_summary, chapter_number=5

  user: "Who is the Cheshire Cat?"
  -> kind=search, search_query="Cheshire Cat description appearance traits"

  user: "What is this book about?" / "介绍下这本书"
  -> kind=book_overview, search_query="book overview main themes central storyline"

  user: "How much have I read?" / "进度怎么样"
  -> kind=reading_progress

  user: "Save these notes as PDF" / "导出笔记 pdf"
  -> kind=export_notes, export_format="pdf"

  user: "What's the weather today?" / "你叫什么名字"
  -> kind=smalltalk

When in doubt between chapter_summary and search: if the user named
a specific chapter number AND asked about its content, prefer
chapter_summary. If they asked about a character / event / theme that
happens to be in a chapter, prefer search.

Ambiguity (clarify): for kind=search/book_overview only. A pronoun or
referent ("他/她/it/they/that one") is NOT automatically ambiguous.
FIRST try to resolve it from the Recent conversation shown in the user
message. Set ambiguous=true ONLY when the referent cannot be resolved
from the recent conversation NOR from the message itself. If it IS
resolvable from the conversation, keep ambiguous=false and just write
search_query for the already-resolved subject.

clarify_question and clarify_options MUST be written in the SAME language
as the user's CURRENT message (English question -> English; Chinese ->
Chinese).

  (no prior conversation) user: "他后来怎么样了?"
  -> kind=search, ambiguous=true, clarify_question="你指的是谁?",
     clarify_options=["疯帽子","白兔","柴郡猫"]

  (no prior conversation) user: "Who is she?"
  -> kind=search, ambiguous=true,
     clarify_question="Who do you mean by 'she'?",
     clarify_options=["the Queen of Hearts","the Duchess","Alice"]

  Recent conversation:
    user: 疯帽子是谁?
    sage: <facts about the Hatter>
  user: "他后来怎么样了?"     (他 clearly = the Hatter, from the prior turn)
  -> kind=search, search_query="Hatter fate outcome ending", ambiguous=false

  user: "Who is the Cheshire Cat?"   (explicit -- never clarify)
  -> kind=search, ambiguous=false

Memory note (fast lane): on EVERY turn, also decide if the user explicitly
stated a durable fact/preference about themselves. This is orthogonal to
`kind` -- a smalltalk turn can carry a note, and a search turn usually does
NOT. Most turns -> memory_note=null. Only a clear, lasting self-statement
fills it. ALWAYS write the note in ENGLISH, even when the user wrote in another
language (translate it; transliterate names). It is shown to the user verbatim.

  user: "叫我小王"               -> memory_note="User wants to be called Xiao Wang", memory_note_type="fact"
  user: "我最近在啃斯多葛哲学"     -> memory_note="User is studying Stoic philosophy", memory_note_type="interest"
  user: "I'm learning AI Harness"  -> memory_note="User is learning about AI Harness", memory_note_type="interest"
  user: "Who is the Cheshire Cat?"  -> memory_note=null   (a book question, not a self-fact)
  user: "What's the weather?"       -> memory_note=null   (transient, not durable)
"""


def _build_classifier() -> ChatOpenAI:
    return ChatOpenAI(
        model="openai/gpt-4o-mini",
        openai_api_base=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        temperature=0,
    )


def classify_intent(
    message: str, history: list[dict] | None = None
) -> IntentDecision:
    """Map a user message to a structured IntentDecision.

    `history` is a recent slice of [{role, content}, ...] used only for
    context-dependent classification (e.g. "and what about chapter 6?"
    after a chapter_summary turn). On any LLM failure (network blip,
    schema reject, etc.) we default to kind="search" with the raw
    message as the query -- this keeps the chat working even if the
    classifier is unreachable.
    """
    try:
        llm = _build_classifier()
        structured = llm.with_structured_output(_IntentSchema)
        history_lines = _format_history_for_prompt(history or [])
        prompt = _build_user_prompt(message, history_lines)
        result: Any = structured.invoke(
            [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]
        )
        return _schema_to_decision(result, fallback_query=message)
    except Exception as exc:
        logger.warning(
            "classify_intent failed: %s. Defaulting to kind=search.", str(exc)[:160]
        )
        return IntentDecision(kind="search", search_query=message)


# ── Helpers ────────────────────────────────────────────────────────────────


def _format_history_for_prompt(history: list[dict]) -> str:
    if not history:
        return ""
    # Last 6 messages is plenty for "and what about ...?" follow-up
    # resolution without bloating the classifier prompt. Defense-in-depth:
    # api/chat.py already caps history at _HISTORY_WINDOW_MESSAGES before
    # it reaches the graph state.
    lines = []
    for msg in history[-6:]:
        role = "user" if msg.get("role") == "user" else "sage"
        content = (msg.get("content") or "")[:200]
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _build_user_prompt(message: str, history_lines: str) -> str:
    if history_lines:
        return f"Recent conversation:\n{history_lines}\n\nUser message: {message}"
    return f"User message: {message}"


def _schema_to_decision(
    schema: _IntentSchema | dict, fallback_query: str
) -> IntentDecision:
    """Coerce the structured-output payload (pydantic model OR a plain
    dict, depending on langchain version) into our internal dataclass.
    """
    if isinstance(schema, dict):
        data = schema
    else:
        data = schema.model_dump() if hasattr(schema, "model_dump") else dict(schema)

    kind = data.get("kind", "search")
    chapter_number = data.get("chapter_number")
    search_query = data.get("search_query")
    export_format = data.get("export_format")

    # For search / book_overview, a missing search_query degrades to the
    # raw user message rather than failing the whole turn.
    if kind in ("search", "book_overview") and not search_query:
        search_query = fallback_query

    # For export_notes, default markdown if the model omitted format.
    if kind == "export_notes" and not export_format:
        export_format = "markdown"

    # Clarify (HITL①): only meaningful for free-form retrieval, and only when
    # the model actually produced a question. Clamp everywhere else so a
    # stray ambiguous=true can never interrupt a progress/export/smalltalk
    # turn (which have nothing to disambiguate).
    clarify_question = data.get("clarify_question")
    ambiguous = (
        bool(data.get("ambiguous"))
        and kind in ("search", "book_overview")
        and bool(clarify_question)
    )
    clarify_options = data.get("clarify_options") or []
    if not isinstance(clarify_options, list):
        clarify_options = []

    # Memory fast lane: normalize the note. Keep a type only when there is a
    # note; default a present-but-untyped note to 'interest' (the cheaper-to-be-
    # wrong bucket -- it just widens recommendation signal).
    memory_note = data.get("memory_note")
    memory_note = memory_note.strip() if isinstance(memory_note, str) else None
    memory_note = memory_note or None
    memory_note_type = data.get("memory_note_type") if memory_note else None
    if memory_note and memory_note_type not in ("fact", "interest"):
        memory_note_type = "interest"

    return IntentDecision(
        kind=kind,
        chapter_number=chapter_number if isinstance(chapter_number, int) else None,
        search_query=search_query,
        export_format=export_format,
        ambiguous=ambiguous,
        clarify_question=clarify_question if ambiguous else None,
        clarify_options=[str(o) for o in clarify_options][:4] if ambiguous else [],
        clarify_multi=bool(data.get("clarify_multi")) if ambiguous else False,
        memory_note=memory_note,
        memory_note_type=memory_note_type,
    )
