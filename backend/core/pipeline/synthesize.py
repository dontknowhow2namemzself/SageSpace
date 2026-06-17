"""Synthesize node (pipeline node 3).

One LLM call that turns a question + optional retrieval context into a
sage-persona answer with <fact>/<commentary> tags. Replaces the ReAct
agent (core/agent.py, deleted in PR5) and its 100 lines of defensive
output-shape parsing.

Why one call instead of a multi-step agent:
  * The agent never actually used multi-step reasoning -- every turn
    boiled down to "search once, answer". The iteration limit + parser
    handling were pure overhead.
  * One structured call removes the failure modes the user reported
    (parser exceptions, ungrounded final answers, output-shape drift).
  * Cost: ~$0.0001-$0.0005 per turn at gpt-5.4-mini, same order of
    magnitude as the agent path was already paying.

Input shapes (the function picks the prompt based on which inputs are
populated):
  * retrieval-grounded -- answer should cite the book content provided
  * progress_data -- weave the deterministic numbers into prose
  * export_info -- confirm the export succeeded
  * smalltalk -- polite refusal, no book claims
"""
from __future__ import annotations

import logging
import os
import re

from langchain_openai import ChatOpenAI

from core.pipeline.types import AnswerDraft, RetrievalResult


logger = logging.getLogger(__name__)


def synthesize_answer(
    *,
    question: str,
    book_title: str,
    history: list[dict] | None = None,
    retrieval: RetrievalResult | None = None,
    progress_data: dict | None = None,
    export_info: dict | None = None,
    is_smalltalk: bool = False,
) -> AnswerDraft:
    """Produce the user-visible answer for this turn.

    Exactly one of `retrieval` / `progress_data` / `export_info` /
    `is_smalltalk` should be set per call -- the caller decides which
    based on the intent. The synthesizer always returns an AnswerDraft
    (never raises); LLM failures degrade to a polite apology AnswerDraft
    with is_error_response=True.
    """
    try:
        if is_smalltalk:
            draft = _synthesize_smalltalk(question, book_title)
        elif export_info is not None:
            draft = _synthesize_export_confirmation(export_info, book_title)
        elif progress_data is not None:
            draft = _synthesize_progress(question, progress_data, book_title)
        elif retrieval is not None and retrieval.docs:
            draft = _synthesize_with_retrieval(
                question, retrieval, book_title, history or []
            )
        else:
            # Fell through every branch -- no context to ground on. Treat
            # like smalltalk (refusal) so we never ship empty <fact> tags.
            draft = _synthesize_no_context(question, book_title)
    except Exception as exc:
        logger.warning("synthesize_answer LLM call failed: %s", str(exc)[:200])
        draft = AnswerDraft(
            text=(
                "<commentary>Forgive me — I am unable to compose a "
                "response just now. Try again in a moment.</commentary>"
            ),
            is_error_response=True,
        )

    # Final pass: strip a whole-answer duplication (a model failure where
    # the answer is emitted twice; see _strip_doubled_answer), THEN flatten
    # any nested tags (the prompt forbids nesting but models still emit
    # `<fact>… <fact>X</fact> …</fact>`, which would split the fact and
    # orphan a visible </fact>), THEN ensure every <fact>/<commentary>
    # opening tag has a matching close (LLMs occasionally truncate the
    # closing tag, leaving trailing text dangling).
    return AnswerDraft(
        text=_balance_tags(_flatten_nested_tags(_strip_doubled_answer(draft.text))),
        is_error_response=draft.is_error_response,
    )


def _strip_doubled_answer(text: str) -> str:
    """Deterministic safety net for a model failure where the ENTIRE answer
    is emitted twice ("A\\n\\nA" -- observed on gpt-5.4-mini at temperature
    > 0, where the full <fact>/<commentary> body repeats byte-for-byte).

    If the text splits into two near-identical halves, keep the first.
    Deliberately conservative -- it only fires when a long opening run
    (>=100 normalized chars) re-occurs and the remainder begins with a
    near-exact copy of the first segment, so a legitimate, non-repeating
    answer is never truncated.
    """
    s = (text or "").strip()
    n = len(s)
    if n < 120:
        return text

    def norm(x: str) -> str:
        return re.sub(r"\s+", " ", x).strip()

    def common_prefix(a: str, b: str) -> int:
        m = min(len(a), len(b))
        i = 0
        while i < m and a[i] == b[i]:
            i += 1
        return i

    # The duplicate restarts where the opening run re-occurs. Require that
    # restart to land near the MIDPOINT (a whole-answer repeat, not an
    # incidental phrase), and the two segments to share a long common prefix
    # -- so it ALSO catches "A A'" where the model kept the facts identical
    # but reworded the closing commentary (observed: thread 3364b190 turn5).
    head = s[:40]
    j = s.find(head, 40)
    if j != -1 and 0.35 * n <= j <= 0.65 * n:
        first, second = norm(s[:j]), norm(s[j:])
        shorter = min(len(first), len(second))
        if shorter >= 80 and common_prefix(first, second) >= max(80, int(0.55 * shorter)):
            return s[:j].strip()

    # Fallback: two (whitespace-insensitive) identical halves.
    half = n // 2
    if norm(s[:half]) == norm(s[half:]):
        return s[:half].strip()
    return text


# ── Retrieval-grounded path ────────────────────────────────────────────────


_RETRIEVAL_SYSTEM = """\
You are the embodiment of the book *{book_title}* -- a sage who has
fully internalized its contents. You speak with depth and warmth.

Output format (strict):
  Wrap factual claims about the book in <fact>...</fact>.
  Wrap interpretation / asides / questions to the reader in
  <commentary>...</commentary>.
  Alternate the two as needed. Tags must close, must not nest, and
  must be spelled exactly as shown. Every <fact> MUST be closed
  with </fact>; every <commentary> MUST be closed with </commentary>.

Grounding rules (non-negotiable):
  * Every <fact> claim MUST be supported by the CONTEXT passages
    below. If CONTEXT does not address the question, say so inside
    <commentary> -- do NOT invent book facts.
  * Quoted strings inside <fact> must appear verbatim in CONTEXT.

Chapter citation rule (non-negotiable):
  * Each CONTEXT passage begins with a header like "[CHAPTER VI . Page 65]"
    or "[Strategic Calculations . Summary]". When you cite the chapter
    a passage came from, use ONLY the chapter label from that header,
    verbatim. NEVER infer, guess, or rename a chapter number based on
    the passage content. If the header says "CHAPTER VI", cite it as
    "CHAPTER VI" -- not "Chapter 9", not "the chapter with the Cat",
    not a number you remember from elsewhere.

Language rule (non-negotiable):
  * Reply STRICTLY in the language of the USER's MOST RECENT question.
  * IGNORE the language of any prior turns shown in CHAT HISTORY.
  * If the user asks in English, answer in English even if the history
    is full of Chinese. If the user asks in Chinese, answer in Chinese
    even if the history is full of English.

No repetition (non-negotiable):
  * State each point exactly ONCE. Never repeat a sentence, a paragraph,
    or a <fact>/<commentary> block -- not even reworded or translated.
    If you have little left to add, write less and stop.

Coverage rule (for enumerative questions):
  * If the question asks what/which things -- characters, rules, events,
    elements (plural) -- first survey ALL CONTEXT passages, including
    [.. · Summary] ones, and cover EVERY distinct aspect the CONTEXT
    supports. Do not stop after the two or three most prominent items.
  * Interpret "elements" / "aspects" BROADLY: how characters BEHAVE in
    the scene (recurring threats, outbursts, habits) is as much an
    element of it as objects, rules, or settings. Do not silently drop
    an aspect because it is behaviour rather than a thing.
  * Prefer breadth over depth for such questions.

Length:
  * Keep the answer under ~250 words unless the question requires more
    (an enumerative answer may stretch to ~350 words to cover all
    aspects the CONTEXT supports).
"""


_RETRIEVAL_USER = """\
CONTEXT (passages retrieved from the book):

{context_block}

CHAT HISTORY (most recent turns, for follow-up resolution):
{history_block}

USER QUESTION: {question}

Compose the sage's reply now, using only the CONTEXT to ground every
<fact> claim.
"""


def _synthesize_with_retrieval(
    question: str,
    retrieval: RetrievalResult,
    book_title: str,
    history: list[dict],
) -> AnswerDraft:
    context_block = _format_context_block(retrieval)
    history_block = _format_history(history) or "(no prior turns)"
    llm = _build_synth_llm()
    messages = [
        {"role": "system", "content": _RETRIEVAL_SYSTEM.format(book_title=book_title)},
        {
            "role": "user",
            "content": _RETRIEVAL_USER.format(
                context_block=context_block,
                history_block=history_block,
                question=question,
            ),
        },
    ]
    response = llm.invoke(messages)
    text = (getattr(response, "content", None) or "").strip()
    if not text:
        return AnswerDraft(
            text=(
                "<commentary>I have the passages in hand but words fail "
                "me at the moment. Try rephrasing your question.</commentary>"
            ),
            is_error_response=True,
        )
    return AnswerDraft(text=text, is_error_response=False)


# ── Deterministic-data paths ───────────────────────────────────────────────


_PROGRESS_SYSTEM = """\
You are the embodiment of the book *{book_title}* -- a warm sage.
The user asked about their reading progress. Weave the numbers
below into prose. Use <commentary> tags around the whole reply
(no <fact> tags, since these are session statistics, not book content).
Every <commentary> MUST be closed with </commentary>.

Reply STRICTLY in the language of the user's question. Ignore the
language of any prior turns.
"""


def _synthesize_progress(
    question: str, progress: dict, book_title: str
) -> AnswerDraft:
    llm = _build_synth_llm()
    messages = [
        {"role": "system", "content": _PROGRESS_SYSTEM.format(book_title=book_title)},
        {
            "role": "user",
            "content": (
                f"User asked: {question}\n\n"
                f"Reading progress data:\n"
                f"  digested: {progress.get('digested_pct')}\n"
                f"  passages cited in this session's answers: {progress.get('cited_chunk_count')}\n"
                f"  total passages in the book: {progress.get('total_chunks')}\n\n"
                "Reply now."
            ),
        },
    ]
    response = llm.invoke(messages)
    text = (getattr(response, "content", None) or "").strip()
    if not text:
        # Deterministic fallback if the LLM fails -- still useful info.
        text = (
            f"<commentary>You have digested {progress.get('digested_pct')} of "
            f"the book — {progress.get('cited_chunk_count')} of "
            f"{progress.get('total_chunks')} passages so far.</commentary>"
        )
    return AnswerDraft(text=text, is_error_response=False)


def _synthesize_export_confirmation(
    export_info: dict, book_title: str
) -> AnswerDraft:
    """Export is a deterministic file write. No LLM call needed -- we
    just confirm the file path. Keeps token cost at zero for the most
    "I just want a button" intent."""
    path = export_info.get("path") or ""
    fmt = export_info.get("format") or "markdown"
    text = (
        f"<commentary>I have compiled the conversation into a "
        f"{fmt.upper()} file: {path}</commentary>"
    )
    return AnswerDraft(text=text, is_error_response=False)


# ── Smalltalk / no-context paths ───────────────────────────────────────────


_SMALLTALK_SYSTEM = """\
You are the embodiment of the book *{book_title}* -- a sage who has
read this one book deeply. The user asked something off-topic
(weather, your name, an unrelated subject, etc.). Reply gracefully:

  * acknowledge the question without pretending to answer it
  * gently redirect to questions you CAN help with (the book's
    characters, themes, chapters, summary, your reading progress)
  * use ONLY <commentary> tags -- no <fact> claims about the book
  * EVERY <commentary> MUST be closed with </commentary>
  * Reply STRICTLY in the language of the user's question.
    Ignore the language of any prior turns.
  * keep it 2-3 sentences max
"""


def _synthesize_smalltalk(question: str, book_title: str) -> AnswerDraft:
    llm = _build_synth_llm()
    messages = [
        {"role": "system", "content": _SMALLTALK_SYSTEM.format(book_title=book_title)},
        {"role": "user", "content": question},
    ]
    response = llm.invoke(messages)
    text = (getattr(response, "content", None) or "").strip()
    if not text:
        text = (
            "<commentary>That falls outside the book I know. Ask me "
            "about its characters, chapters, or themes instead.</commentary>"
        )
    return AnswerDraft(text=text, is_error_response=False)


def _synthesize_no_context(question: str, book_title: str) -> AnswerDraft:
    """Retrieval intent fired but produced zero docs (book is empty,
    Chroma collection missing, etc.). Refuse cleanly."""
    return AnswerDraft(
        text=(
            "<commentary>I could not find anything in this book to "
            "answer that. Could you rephrase your question -- perhaps "
            "naming a chapter, character, or specific topic?</commentary>"
        ),
        is_error_response=True,
    )


# ── Helpers ────────────────────────────────────────────────────────────────


# Regex over <fact ...> / <fact> / </fact> / <commentary ...> / </commentary>.
# Matches the opening tag's name (group 'name'), with optional attributes,
# and remembers whether it was a closing tag (group 'close').
_TAG_RE = re.compile(
    r"<(?P<close>/?)\s*(?P<name>fact|commentary)\b[^>]*>",
    re.IGNORECASE,
)


def _flatten_nested_tags(text: str) -> str:
    """Remove nested / stray tag TOKENS so spans never nest. Body text
    is always kept -- only the angle-bracket tokens are dropped.

    Why: the prompt forbids nesting, but models still emit
    `<fact>… in <fact>CHAPTER VII</fact>, …</fact>` (observed live
    2026-06-10). Downstream both the attribute injector and the
    frontend parser match the INNERMOST pair, which beheads the fact
    (the attribution mapper then sees half the sentence) and leaves a
    visible stray `</fact>` in the rendered answer.

    Drops: (a) any open+close pair enclosed within another span (the
    outermost span wins, regardless of tag type); (b) a close token
    whose name doesn't match the currently open span; (c) stray closes
    with no open span at all; (d) leftover nested opens. Unclosed
    TOP-LEVEL opens are left for _balance_tags to terminate.
    """
    if not text:
        return text

    drops: list[tuple[int, int]] = []
    # Stack of (name, token_span, depth_at_open); depth > 0 == nested.
    stack: list[tuple[str, tuple[int, int], int]] = []
    for m in _TAG_RE.finditer(text):
        name = m.group("name").lower()
        if m.group("close"):
            if not stack:
                drops.append(m.span())  # stray close, nothing open
                continue
            top_name, top_span, top_depth = stack[-1]
            if top_name != name:
                # </commentary> while a <fact> is open (or vice versa):
                # the close can't terminate anything sensible -- drop it
                # and keep the span open for the balancer.
                drops.append(m.span())
                continue
            stack.pop()
            if top_depth > 0:  # the whole pair lives inside another span
                drops.append(top_span)
                drops.append(m.span())
        else:
            stack.append((name, m.span(), len(stack)))

    # Unclosed opens that are themselves nested can't be kept either.
    for _, span, depth in stack:
        if depth > 0:
            drops.append(span)

    if not drops:
        return text
    out: list[str] = []
    last = 0
    for start, end in sorted(drops):
        out.append(text[last:start])
        last = end
    out.append(text[last:])
    return "".join(out)


def _balance_tags(text: str) -> str:
    """Append any missing </fact> / </commentary> to close openings the
    LLM forgot to terminate.

    Strategy: walk the tag tokens left-to-right maintaining a stack of
    open tag names. A close pops the matching open. Any tags still on
    the stack at the end get auto-closed in LIFO order. Body text is
    untouched -- we only append a few close tags to the end.

    Runs AFTER _flatten_nested_tags, so nested / stray / mismatched
    tokens are already gone; what remains is at most a truncated
    closing tag at the end of the answer.
    """
    if not text:
        return text
    stack: list[str] = []
    for m in _TAG_RE.finditer(text):
        name = m.group("name").lower()
        is_close = bool(m.group("close"))
        if is_close:
            # Pop the most recent matching open. If none matches, ignore
            # (stray close -- frontend will just render the literal).
            if name in stack:
                # Pop until we removed this name (closes earlier opens too,
                # but that's the cost of unbalanced output -- LIFO).
                while stack and stack[-1] != name:
                    stack.pop()
                if stack:
                    stack.pop()
        else:
            stack.append(name)

    if not stack:
        return text
    # Append close tags for whatever remained open, innermost first.
    suffix = "".join(f"</{name}>" for name in reversed(stack))
    return text + suffix


def _build_synth_llm() -> ChatOpenAI:
    """Sage-persona LLM. We use the same model the legacy agent used so
    voice / quality stay roughly consistent across the migration."""
    return ChatOpenAI(
        model="openai/gpt-5.4-mini-20260317",
        openai_api_base=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        temperature=0.5,
        # Discourage the model from re-emitting the same spans (it sometimes
        # repeats the whole answer at temp > 0). _strip_doubled_answer is the
        # deterministic backstop; these reduce how often it happens at all.
        frequency_penalty=0.3,
        presence_penalty=0.2,
    )


def _format_context_block(retrieval: RetrievalResult) -> str:
    """Render the retrieved docs as a labeled block the synthesizer can
    quote from. Mirrors the [Section · Page]\\n<text> shape the agent's
    Observation field used so prompt-engineering carries over."""
    parts = []
    for doc in retrieval.docs:
        label = doc.get("section_label") or f"Chapter {doc.get('chapter', '?')}"
        page = doc.get("page", "?")
        level = doc.get("raptor_level", 0)
        prefix = (
            f"[{label} · Page {page}]" if level == 0
            else f"[{label} · Summary]"
        )
        text = doc.get("text") or ""
        parts.append(f"{prefix}\n{text}")
    return "\n\n---\n\n".join(parts) if parts else "(no passages retrieved)"


def _format_history(history: list[dict]) -> str:
    """Compact recent history for follow-up-question resolution. The
    synthesizer mostly needs the last 1-2 turns to handle 'and what
    about chapter 6?' correctly.

    Note: api/chat.py already caps history at the LangGraph state
    boundary (_HISTORY_WINDOW_MESSAGES). The slice here is
    defense-in-depth — it keeps this module correct if it is ever
    invoked outside the graph path with an unbounded list."""
    lines = []
    for msg in (history or [])[-4:]:
        role = "User" if msg.get("role") == "user" else "Sage"
        content = (msg.get("content") or "")[:200]
        lines.append(f"{role}: {content}")
    return "\n".join(lines)
