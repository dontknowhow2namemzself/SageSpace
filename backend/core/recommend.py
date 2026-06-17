"""Book recommendation pipeline (memory-system-design.md §B).

`recommend()` gathers the user's interests (explicit memory notes + library
titles + recent questions), asks an LLM for N fresh picks grounded in those
interests (forcing one cross-genre 'stretch'), validates each against the book
catalog (dropping hallucinations), and persists the survivors as 'suggested'
rows. The home "For you" block reads them; Want-to-read / Dismiss / Shuffle
just transition status. No background job / cron -- recompute is lazy (API
layer) or explicit (Shuffle).
"""
from __future__ import annotations

import logging
import os
import re

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from core import books_api
from core import database as db


logger = logging.getLogger(__name__)

DEFAULT_N = 3

# How many recent questions to PULL before filtering (we keep the best 15).
_QUESTION_POOL = 40
_KEEP_QUESTIONS = 15

# Procedural / meta / smalltalk questions that are NOT interest signals. The
# real signal would be each turn's intent kind, but we don't persist it, and
# re-classifying 15+ stored questions per recommend is too costly -- so this is
# a cheap, deterministic heuristic over the raw text (EN + ZH).
_NOISE_RE = re.compile(
    r"how\s+(much|far)\s+have\s+i\s+read"
    r"|\bprogress\b"
    r"|(export|save)\b.{0,20}\b(notes?|pdf|markdown)"
    r"|summar[iy][sz]e\s+(this|the)\s+book"
    r"|what(?:'s|\s+is)\s+(this|the)\s+book\s+about"
    r"|^(hi|hello|hey)\b"
    r"|nice\s+to\s+meet\s+you"
    r"|(who|what)\s+are\s+you"
    r"|进度|读了多少|导出|保存.{0,6}笔记|介绍.{0,4}这本书|这本书.{0,4}(讲|关于)|你好|你是谁|你叫什么",
    re.IGNORECASE,
)


def _is_interest_question(q: str) -> bool:
    """True when a past question plausibly reveals a durable interest worth
    feeding the recommender. Drops fragments, reading-progress / export /
    smalltalk / 'summarize this book' chatter -- the noise that otherwise
    dilutes (and out-numbers) the real signals."""
    q = (q or "").strip()
    if len(q) < 10:                       # fragments: "tell", "tell me"
        return False
    has_cjk = bool(re.search(r"[一-鿿]", q))
    if not has_cjk and len(q.split()) < 3:
        return False
    return not _NOISE_RE.search(q)


class _RecPick(BaseModel):
    """One LLM-proposed book (pre-validation)."""
    title: str = Field(..., description="The book's title as it appears in a catalog.")
    author: str = Field(..., description="The book's author.")
    reason: str = Field(
        ...,
        description=(
            "One sentence IN ENGLISH on why THIS reader would like it -- it MUST "
            "reference the specific interest it is grounded in."
        ),
    )
    which_interest: str = Field(
        ...,
        description="The specific reader interest this pick ties to, IN ENGLISH.",
    )
    is_stretch: bool = Field(
        default=False,
        description="True for the ONE cross-genre / unexpected pick.",
    )


class _RecSchema(BaseModel):
    picks: list[_RecPick] = Field(default_factory=list)


_SYSTEM_PROMPT = """\
You are a well-read librarian recommending books to ONE reader, from what they
have told you and what they have been reading. Recommend exactly {n} real,
findable books (use widely-known editions/titles so each can be looked up in a
public catalog).

Rules:
- Ground EACH pick in a SPECIFIC interest from the reader's list. The `reason`
  must name/echo that interest and say why this book fits -- one sentence.
- Make exactly ONE pick a 'stretch': a cross-genre / unexpected book a curious
  reader with these interests might love but would not have searched for. Set
  is_stretch=true on that one only.
- NEVER recommend a book that appears in the 'already owned or recommended'
  list.
- Vary authors -- do not return {n} books by the same author.
- `reason` and `which_interest` MUST contain ONLY English (the app is
  English-facing). The reader's signals may be in another language; when one is,
  translate it and write ONLY the English form -- do NOT include the original
  words, transliteration, or a parenthetical. `title` and `author` stay in the
  book's original/catalog language.
"""


def _build_recommender() -> ChatOpenAI:
    # A little warmth (vs the temperature=0 used for classify/plan): recommending
    # is a discovery task, so some variety across runs is desirable. Freshness
    # across Shuffle is mainly driven by the growing exclude list, not temperature.
    return ChatOpenAI(
        model="openai/gpt-4o-mini",
        openai_api_base=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        temperature=0.6,
    )


def llm_recommend(
    interests: list[str], exclude: list[str], n: int = DEFAULT_N
) -> list[_RecPick]:
    """Ask the LLM for `n` grounded picks. Returns [] on any failure (a recs
    block that can't be filled degrades to empty, never an error)."""
    if n <= 0:
        return []
    interests_block = "\n".join(f"- {s}" for s in interests) or "(no signals yet)"
    exclude_block = "\n".join(f"- {t}" for t in exclude) or "(none)"
    user = (
        f"Reader's interests / recent reading (most useful first):\n"
        f"{interests_block}\n\n"
        f"Already owned or already recommended -- DO NOT suggest any of these:\n"
        f"{exclude_block}\n\n"
        f"Recommend exactly {n} books."
    )
    try:
        structured = _build_recommender().with_structured_output(_RecSchema)
        result = structured.invoke([
            {"role": "system", "content": _SYSTEM_PROMPT.format(n=n)},
            {"role": "user", "content": user},
        ])
        data = result.model_dump() if hasattr(result, "model_dump") else dict(result)
        picks = [
            _RecPick(**p) if isinstance(p, dict) else p
            for p in (data.get("picks") or [])
        ]
        return picks[:n]
    except Exception as exc:
        logger.warning("llm_recommend failed: %s", str(exc)[:160])
        return []


def recommend(n: int = DEFAULT_N) -> list[dict]:
    """Generate, validate, and persist up to `n` fresh recommendations.

    Returns the inserted rows (status='suggested'). Cold start -- no library,
    no notes, no questions -- returns [] so the frontend shows its onboarding
    nudge instead of an empty recs block.
    """
    notes = [row["text"] for row in db.list_memory_notes()]
    library = db.library_titles()
    # Pull a wider pool of recent questions, then keep only the substantive ones
    # (procedural / smalltalk get filtered) up to the cap -- so 15 real signals,
    # not 15 mostly-noise rows.
    questions = [
        q for q in db.recent_user_questions(_QUESTION_POOL)
        if _is_interest_question(q)
    ][:_KEEP_QUESTIONS]

    if not library and not notes and not questions:
        return []

    interests = _dedupe(notes + library + questions)
    exclude = set(db.recommended_titles()) | set(library)
    picks = llm_recommend(interests, sorted(exclude), n)

    out: list[dict] = []
    seen_lower = {t.lower() for t in exclude}
    for p in picks:
        title = (p.title or "").strip()
        if not title or title.lower() in seen_lower:
            continue  # model ignored the exclude list / dup within batch -> skip
        try:
            meta = books_api.lookup(title, p.author)
        except books_api.BookLookupUnavailable as exc:
            # Catalog unreachable (e.g. Google Books 429) -- can't validate, but
            # that is NOT a hallucination signal. Keep the pick (the prompt asks
            # for real, findable books) without enriched blurb/year rather than
            # blank the block.
            logger.warning("catalog unavailable for %r; keeping unvalidated: %s",
                           title, str(exc)[:120])
            meta = None
        else:
            if meta is None:
                continue  # clean 'no such book' -> hallucination guard -> drop
        final_title = meta.title if meta else title
        if final_title.lower() in seen_lower:
            continue  # catalog's canonical title collides with an excluded one
        rec_id = db.insert_recommendation(
            title=final_title,
            author=(meta.author if meta else None) or p.author,
            blurb=meta.blurb if meta else None,
            reason=p.reason,
            which_interest=p.which_interest,
            status="suggested",
        )
        seen_lower.add(final_title.lower())
        row = db.get_recommendation(rec_id)
        if row:
            out.append(row)
    return out


def _dedupe(items: list[str]) -> list[str]:
    """Order-preserving, case-insensitive de-dup of interest strings."""
    seen: set[str] = set()
    out: list[str] = []
    for s in items:
        s = (s or "").strip()
        if s and s.lower() not in seen:
            seen.add(s.lower())
            out.append(s)
    return out
