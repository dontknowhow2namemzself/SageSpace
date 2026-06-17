"""Shared per-turn finalization for the chat SSE pipeline.

finalize_turn() consolidates the closing steps so both api/chat.py paths
run the same closer. Callers yield their visible `token` frame first,
then delegate to finalize_turn for everything that comes after.

Answer quality measurement
--------------------------

The online faithfulness probe that used to run here (gated by the
EVAL_FAITHFULNESS_ENABLED env var) was removed in v1.0. It was never
methodologically sound — the prompt was "is the answer grounded? yes/no"
which the judge model overwhelmingly answered "yes", reporting ~100%
faithfulness regardless of actual grounding.

The offline replacement lives in sage-eval, a companion benchmark
project:

  - Faithfulness uses RAGAs-style 2-step decomposition (extract claims
    from the answer, then per-claim grounding check against retrieved
    context). Cost ~$0.002/question (eval side, not user side).
  - Five other quality scorers (Recall@8, Precision@8, MRR@8,
    Completeness, Refusal Correctness) measure dimensions the online
    probe couldn't reach.

The `is_error_response` parameter is retained for API backwards
compatibility but no longer affects behavior.
"""
from __future__ import annotations

import json
import logging
import os
import re
from html import escape
from typing import AsyncIterator

from core import database as db
from core.pricing import calculate_cost_for_model


logger = logging.getLogger(__name__)


_FACT_TAG_RE = re.compile(r"<fact>([\s\S]*?)</fact>", flags=re.IGNORECASE)


def _sse(payload: dict) -> str:
    """Format a dict as one SSE `data: ...\\n\\n` frame."""
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


# ── Public API ──────────────────────────────────────────────────────────────


async def finalize_turn(
    *,
    session_id: str,
    user_message: str,
    assistant_response: str,
    history: list[dict],
    is_error_response: bool = False,
    usage_callback: object | None = None,
    fact_attribution: dict | None = None,
    current_event_ids: list[str] | None = None,
) -> AsyncIterator[str]:
    """Persist a completed assistant turn and yield the closing SSE frames.

    Yields, in order:
      1. answer_attribution -- only when there are recent retrieval_events
         for this session whose chunks can be linked back to the answer
      2. done -- UI stops the streaming spinner here
      3. stream_end -- UI stops reading on this

    Always (regardless of streaming):
      * appends user + assistant turns to `history` (mutated in place) and
        calls db.save_conversation
      * persists token usage from `usage_callback` when provided
      * attaches the attribution payload to every linked retrieval_event

    Args:
        session_id: chat session id
        user_message: the raw user input for this turn
        assistant_response: the visible answer text (may already contain
            <fact data-*=...> attributes injected upstream by the agent)
        history: list of prior {role, content} turns; appended in place
        is_error_response: retained for API backwards compatibility but
            no longer used (the online faithfulness probe that consumed
            this signal was retired in v1.0; see module docstring).
        usage_callback: a LangChain get_usage_metadata_callback() handler.
            Pass None for paths that did not invoke an LLM directly.

    Note: this generator does NOT emit the `token` frame for the visible
    answer. Callers must yield that themselves before delegating, so
    upstream paths retain control over things like <fact> injection
    that need to happen on the wire copy.
    """
    del is_error_response  # explicitly unused — see module docstring

    for payload in build_finalize_payloads(
        session_id=session_id,
        user_message=user_message,
        assistant_response=assistant_response,
        history=history,
        usage_callback=usage_callback,
        fact_attribution=fact_attribution,
        current_event_ids=current_event_ids,
    ):
        yield _sse(payload)


def build_finalize_payloads(
    *,
    session_id: str,
    user_message: str,
    assistant_response: str,
    history: list[dict],
    usage_callback: object | None = None,
    fact_attribution: dict | None = None,
    current_event_ids: list[str] | None = None,
) -> list[dict]:
    """Synchronous core of per-turn finalization.

    Does all the closing side-effects (conversation persistence, token
    usage, attribution attach) and RETURNS the closing SSE payloads as
    plain dicts -- it does not format or yield them. This split lets two
    very different callers share one implementation:

      * `finalize_turn` (async generator) wraps each dict with `_sse`
        and yields it -- the contract the pre-graph chat paths + the
        finalize unit tests rely on.
      * the LangGraph `finalize` node writes each dict to the graph
        stream via `get_stream_writer()`; api/chat.py owns the SSE
        framing on the way out.

    `history` is appended to IN PLACE (user + assistant turns) and saved,
    matching the long-standing finalize_turn contract.

    Returns the payload dicts in order: an optional `answer_attribution`
    (only when this session has linkable retrieval events), then `done`,
    then `stream_end`.
    """
    # 1. Conversation persistence
    history.append({"role": "user", "content": user_message})
    history.append({"role": "assistant", "content": assistant_response})
    db.save_conversation(session_id, json.dumps(history, ensure_ascii=False))

    # 2. Token usage (only meaningful for paths that ran an LLM and passed
    #    their callback here; the graph path accounts usage at the request
    #    level instead -- see api/chat.py + design §7.6 -- and passes None).
    if usage_callback is not None:
        _persist_usage_from_callback(session_id, usage_callback)

    payloads: list[dict] = []

    # 3. Answer attribution -- link recent retrieval events to this answer
    attribution = _build_answer_attribution(session_id)
    if attribution:
        # Merge the per-fact attribution computed at synthesize time (the
        # rebuild here only knows turn-level unions). The Reading Map's
        # "lit" set is derived from these per-fact chunk_ids, and the SSE
        # answer_attribution frame carries them to eval clients.
        # ALWAYS write the facts key, even as []: api/debug.py treats a
        # MISSING facts key as "legacy row from before per-fact
        # attribution" and falls back to over-lighting the turn-level
        # union -- a modern no-<fact> turn must not look legacy.
        attribution["facts"] = (fact_attribution or {}).get("facts") or []

        # Persist onto THIS TURN's events only. attach_event_answer_attribution
        # overwrites the row, and the linked-event list spans the session's
        # last 3 events -- attaching to all of them would clobber earlier
        # turns' per-fact facts (the Reading Map's lit set lives in those
        # rows). Callers that know their turn's events pass them in; the
        # legacy default is the newest linked event. A turn that produced
        # no retrieval event (smalltalk after a search turn) attaches
        # nothing and leaves history intact.
        linked = attribution.get("retrieval_event_ids", [])  # oldest-first
        attach_to = (
            linked[-1:] if current_event_ids is None
            else [eid for eid in current_event_ids if eid in linked]
        )
        for event_id in attach_to:
            db.attach_event_answer_attribution(event_id, attribution)

        # Reader-facing progress ledger: raw chunks this answer CITED.
        # Summary-node citations are excluded — the shelf %, Insight
        # panel and Reading Map all count book text only.
        cited_raw_ids = list(dict.fromkeys(
            cid
            for fact in attribution["facts"]
            for cid in (fact.get("chunk_ids") or [])
            if cid and not cid.startswith("raptor_")
        ))
        if cited_raw_ids:
            session = db.get_session(session_id)
            if session and session.get("book_id"):
                db.record_cited_chunks(
                    session_id, session["book_id"], cited_raw_ids
                )

        payloads.append({"type": "answer_attribution", **attribution})

    # 4. Done frame
    payloads.append({"type": "done"})

    # 5. Stream end
    payloads.append({"type": "stream_end"})

    return payloads


def persist_usage_from_callback(session_id: str, callback_handler: object) -> None:
    """Public wrapper over the token-usage persister.

    The graph chat path (api/chat.py) wraps the whole turn in one
    `get_usage_metadata_callback()` and persists usage ONCE at the
    request level after the stream drains (a turn may span >1 LLM call
    across nodes, and -- from PR3 -- >1 HTTP request across an
    interrupt/resume; design §7.6). Exposed as a function so chat.py
    doesn't reach into the private helper.
    """
    _persist_usage_from_callback(session_id, callback_handler)


# ── Helpers shared with chat.py upstream paths ──────────────────────────────


def build_answer_attribution(session_id: str) -> dict | None:
    """Public alias for the attribution builder.

    Exported so api/chat.py can pre-build attribution for its inline
    <fact> data-* injection (which has to happen before the token frame
    is yielded). finalize_turn() rebuilds independently to stay
    self-contained for callers that don't pre-build.
    """
    return _build_answer_attribution(session_id)


def inject_fact_attribution(
    answer: str,
    attribution: dict | None,
    retrieval_docs: list[dict] | None = None,
) -> tuple[str, dict | None]:
    """Enrich <fact> tags with data-* attribution.

    Each <fact> is mapped to the doc(s) that support it — book-text
    chunks or, when only a chapter summary grounds the fact, the RAPTOR
    summary node — by ONE batched LLM call over `retrieval_docs`
    (see _map_facts_to_chunks).
    A fact the mapper cannot ground gets an EMPTY data-chunk-ids -- the
    UI renders no citation icon and the Reading Map does not light --
    rather than the pre-2026-06-10 fallback of stamping the whole shared
    session chunk list on it (which over-lit the map and made every
    citation chip open the same sources). Pass retrieval_docs=None for
    paths without retrieval context (smalltalk, progress, export): no
    mapper call is made and every fact stays unattributed.
    """
    return _inject_fact_attribution(
        answer, attribution, retrieval_docs=retrieval_docs,
    )


# ── Internal helpers ────────────────────────────────────────────────────────


def _get_recent_retrieval_event_ids(session_id: str, limit: int = 3) -> list[str]:
    try:
        return db.get_recent_event_ids_for_session(session_id, limit=limit)
    except Exception:
        return []


def _build_answer_attribution(session_id: str) -> dict | None:
    event_ids = _get_recent_retrieval_event_ids(session_id, limit=3)
    if not event_ids:
        return None

    event_details = []
    raw_ids: list[str] = []
    summary_ids: list[str] = []
    seen_chunk_ids: set[str] = set()
    seen_summary_ids: set[str] = set()

    for event_id in reversed(event_ids):
        chunks = db.get_event_chunks(event_id)
        if not chunks:
            continue
        event_details.append({"event_id": event_id, "chunks": chunks})
        for chunk in chunks:
            chunk_id = chunk.get("chunk_id") or ""
            if not chunk_id:
                continue
            if int(chunk.get("raptor_level", 0) or 0) > 0:
                if chunk_id not in seen_summary_ids:
                    summary_ids.append(chunk_id)
                    seen_summary_ids.add(chunk_id)
            else:
                if chunk_id not in seen_chunk_ids:
                    raw_ids.append(chunk_id)
                    seen_chunk_ids.add(chunk_id)

    if not event_details:
        return None

    return {
        "retrieval_event_ids": [item["event_id"] for item in event_details],
        "chunk_ids": raw_ids,
        "raptor_ids": summary_ids,
        "events": event_details,
    }


def _inject_fact_attribution(
    answer: str,
    attribution: dict | None,
    retrieval_docs: list[dict] | None = None,
) -> tuple[str, dict | None]:
    """Enrich every <fact>...</fact> in `answer` with data-* attrs so
    the frontend can link each fact span back to its supporting chunk.

    Per-fact routing is ONE batched LLM call (_map_facts_to_chunks)
    over ALL docs that fed the synthesizer (book text + labeled chapter
    summaries, book text preferred): the mapper returns,
    for each fact, the id(s) of the chunk(s) that most directly support
    it, or an empty list when nothing does. Empty means empty: the
    data-chunk-ids attribute is written as "" and the fact renders
    without a citation icon. There is deliberately NO fallback to the
    shared session chunk list -- that fallback (removed 2026-06-10)
    over-lit the Reading Map and pointed every unmatched fact's chip at
    the same sources, claiming a precision the system didn't have.

    Returns (enriched_answer, normalized_payload). When the answer has
    no <fact> tags, returns the input unchanged and None for payload.
    """
    if not answer or not attribution:
        return answer, None

    shared_chunk_ids = attribution.get("chunk_ids") or []
    summary_ids = attribution.get("raptor_ids") or []
    event_ids = attribution.get("retrieval_event_ids") or []

    # Candidate pool: every doc that fed the synthesizer — book text AND
    # RAPTOR chapter summaries. Summaries are legitimate citation targets
    # (2026-06-10): overview/comparison facts are often grounded ONLY in
    # a chapter summary, and pointing at it beats going unattributed or
    # hijacking an unrelated raw chunk. The popup renders a summary
    # node's own text labeled as AI-generated; the Reading Map filters
    # raptor ids out of its lit set (the map stays raw-only).
    candidate_docs: list[dict] = [
        d for d in (retrieval_docs or []) if d.get("chunk_id")
    ]

    fact_texts = [
        (m.group(1) or "").strip() for m in _FACT_TAG_RE.finditer(answer)
    ]
    if not fact_texts:
        return answer, None

    per_fact_ids = _map_facts_to_chunks(fact_texts, candidate_docs)
    per_fact_ids = _enforce_quote_grounding(
        fact_texts, per_fact_ids, candidate_docs
    )

    facts: list[dict] = []

    def replacer(match: re.Match) -> str:
        idx = len(facts)
        fact_text = (match.group(1) or "").strip()
        fact_id = f"f{idx + 1}"
        chunk_ids_for_this_fact = per_fact_ids[idx]

        facts.append(
            {
                "fact_id": fact_id,
                "text": fact_text,
                "chunk_ids": chunk_ids_for_this_fact,
                "retrieval_event_ids": event_ids,
            }
        )
        # No per-fact raptor attribution: the only available raptor id
        # list is the TURN-LEVEL union of retrieved summary nodes, and
        # stamping it on every fact misread as "this fact came from
        # these summaries" (removed 2026-06-08). The union stays
        # available at the payload top level.
        attrs = [
            f'data-fact-id="{fact_id}"',
            f'data-chunk-ids="{escape(",".join(chunk_ids_for_this_fact), quote=True)}"',
            f'data-event-ids="{escape(",".join(event_ids), quote=True)}"',
        ]
        return f"<fact {' '.join(attrs)}>{fact_text}</fact>"

    enriched = _FACT_TAG_RE.sub(replacer, answer)
    if not facts:
        return answer, None

    return enriched, {
        "retrieval_event_ids": event_ids,
        # session-level union (UI consumers that want "every chunk this
        # turn touched" still get it). per-fact narrowing is in
        # `facts[i].chunk_ids`.
        "chunk_ids": shared_chunk_ids,
        "raptor_ids": summary_ids,
        "facts": facts,
    }


# ── LLM-based per-fact source mapping ───────────────────────────────────────


_ATTRIBUTION_MODEL = "openai/gpt-4o-mini"
_MAX_CHUNKS_PER_FACT = 2
_MAX_PASSAGE_CHARS = 800

_ATTRIBUTION_PROMPT = """\
You map factual statements from a book-assistant's answer to the source \
passages that support them.

FACTS:
{facts}

PASSAGES:
{passages}

For each fact, pick the passage(s) where a reader could verify the fact.
Rules:
- At most {max_per_fact} passages per fact; one is usually right.
- If NO passage actually supports a fact, use an empty list. Never guess.
- A fact that quotes the book verbatim is supported ONLY by a passage
  containing that quote -- a passage merely about the same scene does
  not count.
- Passages marked (chapter summary) are AI-generated summaries. Prefer
  (book text) passages; cite a chapter summary ONLY when no book-text
  passage supports the fact.
- Refer to passages by their [number] only.

Respond ONLY with JSON, one entry per fact, in order:
{{"mappings": [{{"fact": 1, "passages": [3]}}, {{"fact": 2, "passages": []}}]}}
"""


def _build_attribution_llm():
    """Small, cheap, deterministic mapper. Imported lazily so unit tests
    that monkeypatch the mapper never touch langchain wiring."""
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=_ATTRIBUTION_MODEL,
        openai_api_base=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        temperature=0,
        model_kwargs={"response_format": {"type": "json_object"}},
    )


def _map_facts_to_chunks(
    fact_texts: list[str], docs: list[dict]
) -> list[list[str]]:
    """One batched LLM call mapping each fact to its supporting chunk(s).

    Returns one chunk-id list per fact (possibly empty), aligned with
    `fact_texts`. The model sees numbered facts and numbered passages
    and answers in passage NUMBERS; numbers are validated and converted
    to chunk ids server-side, so a hallucinated id can never escape.

    Degradation: on any failure (call error, unparseable JSON) every
    fact gets [] -- an unattributed fact renders without a citation
    icon, which is honest; guessing is not.
    """
    empty: list[list[str]] = [[] for _ in fact_texts]
    if not fact_texts or not docs:
        return empty

    facts_block = "\n".join(
        f"{i + 1}. {t}" for i, t in enumerate(fact_texts)
    )
    passages_block = "\n\n".join(
        "[{n}] ({kind}) {text}".format(
            n=i + 1,
            kind=(
                "chapter summary"
                if (d.get("raptor_level") or 0) > 0 else "book text"
            ),
            text=(d.get("text") or "")[:_MAX_PASSAGE_CHARS],
        )
        for i, d in enumerate(docs)
    )
    prompt = _ATTRIBUTION_PROMPT.format(
        facts=facts_block,
        passages=passages_block,
        max_per_fact=_MAX_CHUNKS_PER_FACT,
    )

    try:
        response = _build_attribution_llm().invoke(
            [{"role": "user", "content": prompt}]
        )
        raw = (getattr(response, "content", None) or "").strip()
        # Parsing stays INSIDE the try: a valid-JSON-wrong-shape response
        # must degrade like any other mapper failure, never abort the
        # turn before the token frame ships.
        parsed = _parse_attribution_mapping(raw, len(fact_texts), docs)
    except Exception:
        logger.warning("fact-attribution mapper call failed", exc_info=True)
        return empty

    return parsed if parsed is not None else empty


_QUOTED_SPAN_RE = re.compile(r'["“”]([^"“”]{4,})["“”]')

# Punctuation hugging the inside of quote marks ("pack of cards," in
# American style) must not decide a match: strip span edges before
# containment so “pack of cards,” still matches “…pack of cards: the…”.
_QUOTE_EDGE_PUNCT_RE = re.compile(
    r"^[\s\.,;:!?。，；：！？—–\-…]+|[\s\.,;:!?。，；：！？—–\-…]+$"
)


def _normalize_for_quote_match(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


def _extract_quote_spans(fact_text: str) -> list[str]:
    """Quoted spans (4+ chars between straight/curly double quotes),
    normalized and punctuation-trimmed at both edges."""
    spans: list[str] = []
    for m in _QUOTED_SPAN_RE.findall(fact_text or ""):
        q = _QUOTE_EDGE_PUNCT_RE.sub("", _normalize_for_quote_match(m))
        if len(q) >= 4:
            spans.append(q)
    return spans


def _enforce_quote_grounding(
    fact_texts: list[str],
    per_fact_ids: list[list[str]],
    docs: list[dict],
) -> list[list[str]]:
    """Deterministic VERIFY-don't-override guard for verbatim quotes.

    The synthesizer prompt guarantees quoted strings inside <fact> appear
    verbatim in CONTEXT, so a quoted fact's citation must contain the
    quote. Per quoted fact:
      1. mapper picks that CONTAIN a quote are kept -- the mapper's
         contextual judgment wins whenever it is string-verifiable
         (a containment-first search would let the same phrase from an
         unrelated chapter hijack the citation: observed live 2026-06-10,
         "pack of cards" routed to the Chapter XI trial chunk instead of
         the Chapter XII summary);
      2. otherwise search the pool for containment -- book text first,
         chapter summaries second, rank order within each;
      3. nothing contains the quote -> [] (no icon; an adjacent-scene
         citation that cannot verify the quote misleads).
    Facts without quotes pass through untouched.
    """
    norm_text: dict[str, str] = {}
    level_by_id: dict[str, int] = {}
    pool_order: list[str] = []
    for d in docs:
        cid = d.get("chunk_id")
        if cid and cid not in norm_text:
            norm_text[cid] = _normalize_for_quote_match(d.get("text") or "")
            level_by_id[cid] = int(d.get("raptor_level") or 0)
            pool_order.append(cid)

    out: list[list[str]] = []
    for fact_text, ids in zip(fact_texts, per_fact_ids):
        quotes = _extract_quote_spans(fact_text)
        if not quotes:
            out.append(ids)
            continue

        def contains_quote(cid: str) -> bool:
            text = norm_text.get(cid, "")
            return any(q in text for q in quotes)

        verified_picks = [cid for cid in ids if contains_quote(cid)]
        if verified_picks:
            out.append(verified_picks[:_MAX_CHUNKS_PER_FACT])
            continue
        raw_hits = [
            cid for cid in pool_order
            if level_by_id[cid] == 0 and contains_quote(cid)
        ]
        summary_hits = [
            cid for cid in pool_order
            if level_by_id[cid] > 0 and contains_quote(cid)
        ]
        out.append((raw_hits or summary_hits)[:_MAX_CHUNKS_PER_FACT])
    return out


def _parse_attribution_mapping(
    raw: str, fact_count: int, docs: list[dict]
) -> list[list[str]] | None:
    """Validate the mapper's JSON and convert passage numbers to chunk ids.

    Tolerant of partial junk: out-of-range or non-integer passage numbers
    are dropped, duplicate ids deduped, lists capped at
    _MAX_CHUNKS_PER_FACT, facts missing from the response get [].
    Returns None only when the response is not parseable at all (the
    caller then degrades to all-empty).
    """
    try:
        data = json.loads(raw)
    except (TypeError, ValueError):
        logger.warning("fact-attribution mapper returned unparseable JSON")
        return None
    if not isinstance(data, dict):
        return None
    mappings = data.get("mappings")
    if not isinstance(mappings, list):
        return None

    def _is_index(v: object) -> bool:
        # bool is a subclass of int -- JSON true would otherwise pass as
        # 1 (and hash-collide with key 1 in by_fact).
        return isinstance(v, int) and not isinstance(v, bool)

    by_fact: dict[int, list] = {}
    for m in mappings:
        if isinstance(m, dict) and _is_index(m.get("fact")):
            by_fact[m["fact"]] = m.get("passages")

    out: list[list[str]] = []
    for i in range(fact_count):
        ids: list[str] = []
        nums = by_fact.get(i + 1)
        if isinstance(nums, list):
            for n in nums:
                if not _is_index(n) or not (1 <= n <= len(docs)):
                    continue
                cid = docs[n - 1].get("chunk_id")
                if cid and cid not in ids:
                    ids.append(cid)
                if len(ids) >= _MAX_CHUNKS_PER_FACT:
                    break
        out.append(ids)
    return out


def _persist_usage_from_callback(session_id: str, callback_handler) -> None:
    usage_by_model = getattr(callback_handler, "usage_metadata", {}) or {}
    if not usage_by_model:
        return

    total_in = 0
    total_out = 0
    total_cost = 0.0

    for model_name, usage in usage_by_model.items():
        input_tokens = int(usage.get("input_tokens", 0) or 0)
        output_tokens = int(usage.get("output_tokens", 0) or 0)
        total_in += input_tokens
        total_out += output_tokens
        model_cost = calculate_cost_for_model(model_name, input_tokens, output_tokens)
        if model_cost == 0.0 and (input_tokens or output_tokens):
            logger.warning(
                "Usage recorded for model without configured price: %s (input=%s, output=%s)",
                model_name,
                input_tokens,
                output_tokens,
            )
        total_cost += model_cost

    if total_in or total_out or total_cost:
        db.update_session_tokens(session_id, total_in, total_out, total_cost)
