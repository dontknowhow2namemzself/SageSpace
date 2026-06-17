"""Tests for the LLM intent classifier.

We do NOT call the real LLM here -- the structured-output behavior of
ChatOpenAI is exercised in integration. These tests cover:

  * IntentDecision coercion: structured output (dict or pydantic) ->
    our internal dataclass with correct field population
  * default-fill behavior: search_query falls back to raw message,
    export_format defaults to markdown
  * LLM failure recovery: classifier returns search-intent rather than
    raising (so a flaky network does not break chat)
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from core.pipeline import intent as intent_mod
from core.pipeline.intent import classify_intent
from core.pipeline.types import IntentDecision


def _patch_structured_output(monkeypatch, payload: dict | Exception):
    """Patch ChatOpenAI.with_structured_output(...).invoke(...) to return
    `payload` (or raise it if Exception). Returns the patched LLM mock
    so tests can also assert call counts."""
    mock_structured = MagicMock()
    if isinstance(payload, Exception):
        mock_structured.invoke.side_effect = payload
    else:
        mock_structured.invoke.return_value = payload

    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value = mock_structured

    monkeypatch.setattr(intent_mod, "_build_classifier", lambda: mock_llm)
    return mock_llm


def test_chapter_summary_intent_round_trips_chapter_number(monkeypatch):
    _patch_structured_output(monkeypatch, {
        "kind": "chapter_summary",
        "chapter_number": 5,
        "search_query": None,
        "export_format": None,
    })
    out = classify_intent("讲讲第5章", history=[])
    assert out.kind == "chapter_summary"
    assert out.chapter_number == 5
    assert out.search_query is None


def test_search_intent_defaults_to_raw_message_when_query_missing(monkeypatch):
    """Model said kind=search but omitted search_query -- the
    classifier must fall back to the raw user message rather than
    handing the retriever an empty string."""
    _patch_structured_output(monkeypatch, {
        "kind": "search",
        "chapter_number": None,
        "search_query": None,
        "export_format": None,
    })
    out = classify_intent("Who is the Cheshire Cat?", history=[])
    assert out.kind == "search"
    assert out.search_query == "Who is the Cheshire Cat?"


def test_export_notes_defaults_to_markdown_when_format_missing(monkeypatch):
    _patch_structured_output(monkeypatch, {
        "kind": "export_notes",
        "chapter_number": None,
        "search_query": None,
        "export_format": None,
    })
    out = classify_intent("save these notes", history=[])
    assert out.kind == "export_notes"
    assert out.export_format == "markdown"


def test_export_notes_respects_explicit_pdf(monkeypatch):
    _patch_structured_output(monkeypatch, {
        "kind": "export_notes",
        "chapter_number": None,
        "search_query": None,
        "export_format": "pdf",
    })
    out = classify_intent("export as pdf", history=[])
    assert out.export_format == "pdf"


def test_reading_progress_intent_has_no_extra_fields(monkeypatch):
    _patch_structured_output(monkeypatch, {
        "kind": "reading_progress",
        "chapter_number": None,
        "search_query": None,
        "export_format": None,
    })
    out = classify_intent("how much have I read", history=[])
    assert out.kind == "reading_progress"
    assert out.chapter_number is None
    assert out.search_query is None


def test_book_overview_falls_back_to_user_message_for_query(monkeypatch):
    """book_overview also wants a search_query; same defaulting rule
    as kind=search."""
    _patch_structured_output(monkeypatch, {
        "kind": "book_overview",
        "chapter_number": None,
        "search_query": None,
        "export_format": None,
    })
    out = classify_intent("What's this book about?", history=[])
    assert out.kind == "book_overview"
    assert out.search_query == "What's this book about?"


def test_smalltalk_intent_leaves_other_fields_null(monkeypatch):
    _patch_structured_output(monkeypatch, {
        "kind": "smalltalk",
        "chapter_number": None,
        "search_query": None,
        "export_format": None,
    })
    out = classify_intent("how is the weather today?", history=[])
    assert out.kind == "smalltalk"
    assert out.search_query is None
    assert out.chapter_number is None


def test_llm_failure_falls_back_to_search_intent(monkeypatch):
    """The classifier MUST NOT raise on transient LLM errors -- the
    chat turn keeps working by defaulting to a search over the user
    message verbatim."""
    _patch_structured_output(monkeypatch, RuntimeError("simulated network outage"))
    out = classify_intent("Tell me about chapter 5", history=[])
    assert out.kind == "search"
    assert out.search_query == "Tell me about chapter 5"


def test_pydantic_model_return_also_coerces_correctly(monkeypatch):
    """langchain versions differ on whether with_structured_output
    returns a pydantic instance or a dict. Both must coerce cleanly."""
    from core.pipeline.intent import _IntentSchema
    model_instance = _IntentSchema(
        kind="chapter_summary",
        chapter_number=12,
        search_query=None,
        export_format=None,
    )
    _patch_structured_output(monkeypatch, model_instance)
    out = classify_intent("第十二章讲什么", history=[])
    assert out.kind == "chapter_summary"
    assert out.chapter_number == 12


def test_history_included_in_prompt(monkeypatch):
    """The classifier must pass recent history into the prompt so it
    can resolve 'and what about chapter 6?' follow-ups. We capture
    the system+user messages sent to the LLM and verify the prior
    turn appears in there."""
    mock_structured = MagicMock()
    mock_structured.invoke.return_value = {
        "kind": "chapter_summary", "chapter_number": 6,
        "search_query": None, "export_format": None,
    }
    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value = mock_structured
    monkeypatch.setattr(intent_mod, "_build_classifier", lambda: mock_llm)

    history = [
        {"role": "user", "content": "讲讲第5章"},
        {"role": "assistant", "content": "<fact>Chapter V is about the Caterpillar.</fact>"},
    ]
    classify_intent("and chapter 6?", history=history)

    call_args = mock_structured.invoke.call_args
    sent_messages = call_args[0][0]  # first positional arg is the message list
    user_message_content = sent_messages[-1]["content"]
    assert "讲讲第5章" in user_message_content, user_message_content
    assert "and chapter 6?" in user_message_content


# ── clarify ambiguity (PR3, 方案甲) ─────────────────────────────────────────


def test_ambiguous_search_passes_through(monkeypatch):
    _patch_structured_output(monkeypatch, {
        "kind": "search", "search_query": "character fate",
        "ambiguous": True, "clarify_question": "你指的是谁?",
        "clarify_options": ["疯帽子", "白兔"], "clarify_multi": False,
    })
    out = classify_intent("他后来怎么样了?", history=[])
    assert out.ambiguous is True
    assert out.clarify_question == "你指的是谁?"
    assert out.clarify_options == ["疯帽子", "白兔"]


def test_ambiguous_clamped_off_for_non_retrieval_kinds(monkeypatch):
    # Model wrongly flags smalltalk as ambiguous -> must be clamped to False
    # (there is nothing to retrieve, so no clarify interrupt should fire).
    _patch_structured_output(monkeypatch, {
        "kind": "smalltalk", "ambiguous": True,
        "clarify_question": "which one?", "clarify_options": ["a", "b"],
    })
    out = classify_intent("你叫什么?", history=[])
    assert out.ambiguous is False
    assert out.clarify_question is None
    assert out.clarify_options == []


def test_ambiguous_requires_a_question(monkeypatch):
    # ambiguous=true but no question -> not actionable -> clamp to False.
    _patch_structured_output(monkeypatch, {
        "kind": "search", "search_query": "x",
        "ambiguous": True, "clarify_question": None, "clarify_options": [],
    })
    out = classify_intent("something", history=[])
    assert out.ambiguous is False


def test_ambiguous_options_capped_and_stringified(monkeypatch):
    _patch_structured_output(monkeypatch, {
        "kind": "search", "search_query": "x", "ambiguous": True,
        "clarify_question": "who?", "clarify_options": ["a", "b", "c", "d", "e", 6],
    })
    out = classify_intent("who?", history=[])
    assert out.clarify_options == ["a", "b", "c", "d"]  # capped at 4, all str


def test_clear_intent_defaults_ambiguous_false(monkeypatch):
    _patch_structured_output(monkeypatch, {
        "kind": "search", "search_query": "cheshire cat",
    })
    out = classify_intent("Who is the Cheshire Cat?", history=[])
    assert out.ambiguous is False


# ── memory fast lane (design §A) ─────────────────────────────────────────────


def test_memory_note_round_trips_with_type(monkeypatch):
    _patch_structured_output(monkeypatch, {
        "kind": "smalltalk",
        "memory_note": "用户希望被称为小王",
        "memory_note_type": "fact",
    })
    out = classify_intent("叫我小王", history=[])
    assert out.memory_note == "用户希望被称为小王"
    assert out.memory_note_type == "fact"


def test_memory_note_rides_on_a_search_turn(monkeypatch):
    """The note is orthogonal to kind -- a search turn can still carry one."""
    _patch_structured_output(monkeypatch, {
        "kind": "search", "search_query": "stoicism control",
        "memory_note": "用户在研读斯多葛哲学", "memory_note_type": "interest",
    })
    out = classify_intent("斯多葛怎么看待掌控?", history=[])
    assert out.kind == "search"
    assert out.memory_note == "用户在研读斯多葛哲学"
    assert out.memory_note_type == "interest"


def test_no_memory_note_is_none_none(monkeypatch):
    _patch_structured_output(monkeypatch, {
        "kind": "search", "search_query": "cheshire cat",
    })
    out = classify_intent("Who is the Cheshire Cat?", history=[])
    assert out.memory_note is None
    assert out.memory_note_type is None


def test_memory_note_blank_is_dropped(monkeypatch):
    """Whitespace-only note -> normalized to None (no type either)."""
    _patch_structured_output(monkeypatch, {
        "kind": "smalltalk", "memory_note": "   ", "memory_note_type": "fact",
    })
    out = classify_intent("hi", history=[])
    assert out.memory_note is None
    assert out.memory_note_type is None


def test_memory_note_without_type_defaults_to_interest(monkeypatch):
    """A present note with a missing/invalid type defaults to 'interest'."""
    _patch_structured_output(monkeypatch, {
        "kind": "smalltalk", "memory_note": "用户喜欢科幻", "memory_note_type": None,
    })
    out = classify_intent("我爱看科幻", history=[])
    assert out.memory_note == "用户喜欢科幻"
    assert out.memory_note_type == "interest"
