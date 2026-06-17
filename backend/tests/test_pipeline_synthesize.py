"""Tests for the Synthesize pipeline node.

The 4 input shapes correspond to the 6 intent kinds:

  retrieval  -> kind in {search, chapter_summary, book_overview}
  progress   -> kind = reading_progress
  export     -> kind = export_notes (NO LLM call -- deterministic)
  smalltalk  -> kind = smalltalk

We mock the synthesizer LLM to assert each branch picks the right
prompt and produces the expected envelope shape (<fact> for grounded,
<commentary>-only otherwise).
"""
from __future__ import annotations

from unittest.mock import MagicMock

from core.pipeline import synthesize as synth_mod
from core.pipeline.synthesize import synthesize_answer
from core.pipeline.types import RetrievalResult


def _mock_llm(monkeypatch, content: str | Exception = "<fact>mocked answer</fact>"):
    mock = MagicMock()
    if isinstance(content, Exception):
        mock.invoke.side_effect = content
    else:
        mock.invoke.return_value = MagicMock(content=content)
    monkeypatch.setattr(synth_mod, "_build_synth_llm", lambda: mock)
    return mock


# ── retrieval-grounded path ───────────────────────────────────────────────


def test_synthesize_retrieval_path_invokes_llm_with_context_block(monkeypatch):
    llm = _mock_llm(
        monkeypatch,
        "<fact>The Cheshire Cat grins.</fact><commentary>It is unsettling.</commentary>",
    )
    retrieval = RetrievalResult(
        docs=[
            {"text": "the Cat grinned", "chunk_id": "chk_a",
             "section_label": "CHAPTER VI", "chapter": 6, "page": 65,
             "raptor_level": 0},
        ],
        sources=[],
    )
    draft = synthesize_answer(
        question="who is the cheshire cat",
        book_title="Alice in Wonderland",
        retrieval=retrieval,
    )
    assert "<fact>" in draft.text
    assert draft.is_error_response is False
    # The CONTEXT block must have made it into the prompt
    sent = llm.invoke.call_args[0][0]
    joined = "\n".join(m["content"] for m in sent)
    assert "the Cat grinned" in joined
    assert "CHAPTER VI" in joined


def test_synthesize_retrieval_empty_docs_falls_through_to_refusal(monkeypatch):
    """retrieval is not None but docs is empty -- still no context to
    ground on. Must NOT invoke the synthesizer LLM with empty CONTEXT
    (which would invite hallucination); instead emit a refusal."""
    llm = _mock_llm(monkeypatch, "<fact>WRONG: should not run</fact>")
    retrieval = RetrievalResult(docs=[], sources=[])
    draft = synthesize_answer(
        question="anything", book_title="Some Book", retrieval=retrieval,
    )
    assert draft.is_error_response is True
    assert "<fact>" not in draft.text
    assert llm.invoke.called is False


def test_synthesize_retrieval_llm_failure_returns_error_draft(monkeypatch):
    _mock_llm(monkeypatch, RuntimeError("simulated outage"))
    retrieval = RetrievalResult(
        docs=[{"text": "x", "chunk_id": "c", "chapter": 1, "page": 1, "raptor_level": 0}],
        sources=[],
    )
    draft = synthesize_answer(
        question="q", book_title="B", retrieval=retrieval,
    )
    assert draft.is_error_response is True


# ── progress path ─────────────────────────────────────────────────────────


def test_synthesize_progress_uses_progress_prompt(monkeypatch):
    llm = _mock_llm(
        monkeypatch,
        "<commentary>You have read 30% of the book.</commentary>",
    )
    progress = {
        "digested_pct": "30.0%",
        "cited_chunk_count": 30,
        "total_chunks": 100,
    }
    draft = synthesize_answer(
        question="how much have I read?",
        book_title="Test",
        progress_data=progress,
    )
    assert "<fact>" not in draft.text  # progress is session data, not book content
    assert "<commentary>" in draft.text


def test_synthesize_progress_falls_back_when_llm_returns_empty(monkeypatch):
    """If the synthesizer LLM returns an empty string, we still produce
    a useful commentary line by interpolating the numbers ourselves."""
    _mock_llm(monkeypatch, "")
    progress = {
        "digested_pct": "42.0%",
        "cited_chunk_count": 42,
        "total_chunks": 100,
    }
    draft = synthesize_answer(
        question="progress?", book_title="T", progress_data=progress,
    )
    assert "42.0%" in draft.text


# ── export path (no LLM call) ─────────────────────────────────────────────


def test_synthesize_export_does_not_call_llm(monkeypatch):
    """Export is a deterministic file write -- the synthesizer must
    NOT spend LLM tokens to confirm a file path."""
    llm = _mock_llm(monkeypatch, "WRONG: should not run")
    export = {"format": "markdown", "path": "/exports/session_x.md", "available": True}
    draft = synthesize_answer(
        question="save my notes", book_title="T", export_info=export,
    )
    assert "/exports/session_x.md" in draft.text
    assert "MARKDOWN" in draft.text.upper()
    assert llm.invoke.called is False
    assert draft.is_error_response is False


# ── smalltalk path ────────────────────────────────────────────────────────


def test_synthesize_smalltalk_uses_smalltalk_prompt_and_no_fact_tags(monkeypatch):
    llm = _mock_llm(
        monkeypatch,
        "<commentary>I cannot see the sky from here.</commentary>",
    )
    draft = synthesize_answer(
        question="what is the weather today?",
        book_title="Alice",
        is_smalltalk=True,
    )
    assert "<commentary>" in draft.text
    assert "<fact>" not in draft.text  # never claim book content for smalltalk
    # System prompt should signal off-topic refusal posture
    sys_msg = llm.invoke.call_args[0][0][0]["content"]
    assert "off-topic" in sys_msg.lower() or "redirect" in sys_msg.lower()


def test_synthesize_smalltalk_llm_empty_returns_default_refusal(monkeypatch):
    _mock_llm(monkeypatch, "")
    draft = synthesize_answer(
        question="how is the weather", book_title="T", is_smalltalk=True,
    )
    assert "<commentary>" in draft.text
    assert "outside the book" in draft.text.lower() or "ask me about" in draft.text.lower()
    assert draft.is_error_response is False


# ── no-input fallthrough ──────────────────────────────────────────────────


def test_synthesize_with_no_inputs_returns_refusal_without_calling_llm(monkeypatch):
    """Caller forgot to set ANY input -- synthesize must NOT invent a
    book-grounded answer. Treat like smalltalk-empty: refusal, no LLM."""
    llm = _mock_llm(monkeypatch, "WRONG: should not run")
    draft = synthesize_answer(question="q", book_title="T")
    assert "<fact>" not in draft.text
    assert draft.is_error_response is True
    assert llm.invoke.called is False


# ── PR7: tag balancer ──────────────────────────────────────────────────────


from core.pipeline.synthesize import _balance_tags, _flatten_nested_tags
from core.pipeline.types import RetrievalResult


# ── nested-tag flattener ───────────────────────────────────────────────────


def test_flatten_nested_fact_keeps_outer_span_and_full_text():
    """The live 2026-06-10 bug: the model nested <fact> inside <fact>.
    Downstream matchers pair the INNERMOST close, beheading the fact
    (the attribution mapper then sees half the sentence) and orphaning
    a visible </fact>. The flattener must keep ONE full span."""
    text = (
        '<fact>At the tea party in <fact>CHAPTER VII</fact>, the Hatter '
        'asks Alice, "Why is a raven like a writing-desk?"</fact>'
    )
    assert _flatten_nested_tags(text) == (
        '<fact>At the tea party in CHAPTER VII, the Hatter '
        'asks Alice, "Why is a raven like a writing-desk?"</fact>'
    )


def test_flatten_nested_commentary_inside_fact_dropped():
    text = "<fact>a <commentary>b</commentary> c</fact>"
    assert _flatten_nested_tags(text) == "<fact>a b c</fact>"


def test_flatten_drops_stray_close_without_open():
    text = "plain text </fact> more text"
    assert _flatten_nested_tags(text) == "plain text  more text"


def test_flatten_drops_mismatched_close_keeps_span_for_balancer():
    """</commentary> while a <fact> is open closes nothing; the fact
    stays open and _balance_tags terminates it at the end."""
    text = "<fact>claim</commentary> tail"
    flattened = _flatten_nested_tags(text)
    assert flattened == "<fact>claim tail"
    assert _balance_tags(flattened) == "<fact>claim tail</fact>"


def test_flatten_drops_leftover_nested_open():
    text = "<fact>a <fact>b"
    assert _flatten_nested_tags(text) == "<fact>a b"


def test_flatten_leaves_wellformed_text_untouched():
    text = '<fact data-fact-id="f1">a</fact><commentary>b</commentary>'
    assert _flatten_nested_tags(text) == text


def test_balance_tags_closes_unclosed_commentary():
    """Reproduces the user-reported export bug: LLM forgot the closing
    </commentary>. The balancer appends it at the end so the frontend
    does not render the trailing text as one giant commentary block."""
    text = "<commentary>The chapter's mockery of order, where the more"
    assert _balance_tags(text) == (
        "<commentary>The chapter's mockery of order, where the more</commentary>"
    )


def test_balance_tags_closes_unclosed_fact():
    text = "<fact>Chapter VI introduces the Cheshire Cat"
    assert _balance_tags(text) == (
        "<fact>Chapter VI introduces the Cheshire Cat</fact>"
    )


def test_balance_tags_leaves_balanced_text_untouched():
    text = "<fact>a</fact><commentary>b</commentary>"
    assert _balance_tags(text) == text


def test_balance_tags_handles_attribute_decorated_opening():
    """After inject_fact_attribution decorates the tag with data-* attrs
    the opener is <fact data-fact-id="f1" ...>. The balancer must still
    recognize it as a <fact> opening."""
    text = '<fact data-fact-id="f1" data-chunk-ids="chk_a">unclosed'
    out = _balance_tags(text)
    assert out.endswith("</fact>")


def test_balance_tags_handles_interleaved_open_close():
    """fact then commentary then close commentary, no close for fact:
    LIFO close order means we close fact last."""
    text = "<fact>a<commentary>b</commentary>"
    out = _balance_tags(text)
    assert out.endswith("</fact>")
    assert "</commentary>" in out


def test_balance_tags_ignores_empty_and_plain_text():
    assert _balance_tags("") == ""
    assert _balance_tags("plain text with no tags") == "plain text with no tags"


def test_synthesize_answer_applies_tag_balancer(monkeypatch):
    """End-to-end: a synthesize call whose LLM forgot a close tag
    returns an AnswerDraft whose text IS already balanced."""
    _mock_llm(monkeypatch, "<fact>a</fact><commentary>b unclosed")
    retrieval = RetrievalResult(
        docs=[{"text": "x", "chunk_id": "c", "chapter": 1, "page": 1, "raptor_level": 0}],
        sources=[],
    )
    draft = synthesize_answer(
        question="q", book_title="T", retrieval=retrieval,
    )
    assert draft.text.endswith("</commentary>")


# ── _strip_doubled_answer (model emits the whole answer twice) ───────────────

_BLOCK = (
    "<fact>The White Rabbit hurries past muttering that he shall be late, and "
    "Alice, burning with curiosity, follows him down the rabbit-hole.</fact>\n"
    "<commentary>A dreamlike beginning that pulls her out of ordinary life.</commentary>"
)


def test_strip_doubled_answer_exact_repeat():
    doubled = _BLOCK + "\n\n" + _BLOCK
    assert synth_mod._strip_doubled_answer(doubled) == _BLOCK


def test_strip_doubled_answer_whitespace_insensitive_repeat():
    doubled = _BLOCK + "\n\n\n   " + _BLOCK
    assert synth_mod._strip_doubled_answer(doubled).strip() == _BLOCK.strip()


def test_strip_doubled_answer_reworded_tail():
    # Real failure (thread 3364b190 turn5): identical <fact>s, the closing
    # <commentary> reworded in the second copy -> still a duplicate.
    facts = (
        "<fact>CONTEXT does not say the rabbit visited any 'white house'.</fact>"
        "<fact>It only says the rabbit ran close by and told Alice to fetch "
        "a pair of gloves and a fan.</fact>"
    )
    first = facts + "<commentary>So the material only shows it appeared and sent Alice off.</commentary>"
    second = facts + "<commentary>So we can only say it appeared and dispatched Alice; nothing about a white house.</commentary>"
    out = synth_mod._strip_doubled_answer(first + "\n\n" + second)
    assert out == first.strip()
    assert out.count("CONTEXT does not say") == 1


def test_strip_doubled_answer_keeps_genuine_single():
    # A long, non-repeating answer must pass through untouched.
    single = (
        "<fact>The Queen of Hearts rules with sudden temper.</fact>"
        "<fact>The Cheshire Cat grins and fades to a smile.</fact>"
        "<commentary>Two very different kinds of madness sit side by side.</commentary>"
    )
    assert synth_mod._strip_doubled_answer(single) == single


def test_strip_doubled_answer_short_untouched():
    assert synth_mod._strip_doubled_answer("<fact>short</fact>") == "<fact>short</fact>"


def test_synthesize_answer_strips_model_doubling(monkeypatch):
    """End to end: the LLM returns A\\n\\nA, synthesize_answer ships one A."""
    _mock_llm(monkeypatch, _BLOCK + "\n\n" + _BLOCK)
    retrieval = RetrievalResult(
        docs=[{"text": "x", "chunk_id": "c", "chapter": 1, "page": 1, "raptor_level": 0}],
        sources=[],
    )
    draft = synthesize_answer(question="q", book_title="T", retrieval=retrieval)
    assert draft.text.count("The White Rabbit hurries past") == 1
