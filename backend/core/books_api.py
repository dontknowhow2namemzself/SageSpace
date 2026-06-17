"""Pluggable book-metadata lookup (memory-system-design.md §B).

recommend() proposes books from the user's interests; before any of them is
shown we VALIDATE each against a real catalog -- a pick that doesn't resolve to
a real book is dropped (the hallucination guard). The source sits behind a
small interface so it can be swapped later (like ReadITloder's TTS engines);
the MVP impl is the Google Books volumes API: metadata ONLY
({title, author, year, blurb}), NO covers (the app draws its own card visual).

API key: set GOOGLE_BOOKS_API_KEY in the env to lift the quota to ~1000
requests/day (free, no billing -- enable "Books API" in a Google Cloud project
and create an API key). WITHOUT a key the call still works but uses the shared
anonymous quota, which is tiny and 429s almost immediately on a dev/cloud IP
(that 429 -> BookLookupUnavailable -> recommend() keeps the pick degraded).
"""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from typing import Protocol

import httpx


logger = logging.getLogger(__name__)

_GOOGLE_BOOKS_URL = "https://www.googleapis.com/books/v1/volumes"
_TIMEOUT_S = 6.0


class BookLookupUnavailable(Exception):
    """The catalog could not be REACHED (network / timeout / 429 rate-limit /
    5xx / bad payload) -- distinct from a clean 'no such book' (which is None).

    The distinction matters for the hallucination guard: a failed *validation*
    is not evidence a book is fake. recommend() drops a clean not-found but
    KEEPS (degraded) a pick whose validation was merely unavailable, so a
    rate-limited keyless catalog does not blank the whole recs block."""


@dataclass
class BookMeta:
    """Catalog metadata for one validated book. No cover -- by design."""
    title: str
    author: str | None = None
    year: str | None = None
    blurb: str | None = None


class BookMetadataSource(Protocol):
    """Swap target: anything that can resolve a (title, author) guess to real
    catalog metadata, or None when there is no confident match."""

    def lookup(self, title: str, author: str | None = None) -> BookMeta | None:
        ...


class GoogleBooksSource:
    """Google Books volumes API. Metadata only.

    Sends GOOGLE_BOOKS_API_KEY (read from the env per call) when present, which
    lifts the quota from the shared-anonymous tier (429s fast) to ~1000/day.

    `country` is sent because the volumes endpoint now requires it in some
    regions (otherwise it 403s); US is a safe default for an English catalog.
    """

    def __init__(self, timeout_s: float = _TIMEOUT_S):
        self._timeout_s = timeout_s

    def lookup(self, title: str, author: str | None = None) -> BookMeta | None:
        """Return catalog metadata, or None for a clean 'no such book'.

        Raises BookLookupUnavailable when the catalog could not be reached
        (network / non-200 incl. 429 / bad JSON) -- so the caller can tell
        "couldn't validate" apart from "validated: doesn't exist."
        """
        title = (title or "").strip()
        if not title:
            return None
        q = f'intitle:"{title}"'
        if author and author.strip():
            q += f'+inauthor:"{author.strip()}"'
        # Pull a few candidates (not just the top hit): Google often ranks
        # third-party "Summary of X" / study-guide knockoffs above the real
        # book, so we score + pick rather than blindly take items[0].
        params = {"q": q, "maxResults": 5, "country": "US"}
        api_key = os.getenv("GOOGLE_BOOKS_API_KEY")
        if api_key:
            params["key"] = api_key
        try:
            resp = httpx.get(
                _GOOGLE_BOOKS_URL,
                params=params,
                timeout=self._timeout_s,
            )
        except Exception as exc:  # connect/timeout/DNS
            raise BookLookupUnavailable(str(exc)) from exc
        if resp.status_code != 200:
            # 429 (keyless quota) / 5xx / 4xx: we could not validate -- NOT a
            # signal the book is fake.
            raise BookLookupUnavailable(f"HTTP {resp.status_code}")
        try:
            data = resp.json()
        except Exception as exc:
            raise BookLookupUnavailable(f"bad payload: {exc}") from exc
        return _select_best(data, title, author)  # None == clean 'no match'


# Third-party knockoffs that pollute Google Books results (summaries, study
# guides, etc.). A volume whose title/subtitle/publisher matches gets penalized
# so the real edition wins when both are present.
_JUNK_RE = re.compile(
    r"\b(summary|summaries|study guide|workbook|analysis|key takeaways|"
    r"conversation starters|sparknotes|cliffs?notes|quicklet|trivia|"
    r"sidekick|instaread|blinkist)\b",
    re.IGNORECASE,
)


def _norm(s: str) -> str:
    """Lowercase + punctuation-to-space, for loose title/author comparison."""
    return re.sub(r"[^\w\s]", " ", (s or "").lower()).strip()


def _tokens(s: str) -> set[str]:
    return {w for w in _norm(s).split() if len(w) > 1}


def _score_volume(
    info: dict, want_title_norm: str, want_title_tokens: set[str],
    want_author_tokens: set[str],
) -> float | None:
    """Higher = better match for the LLM's intended (title, author). Returns
    None for an unusable volume (no title)."""
    title = (info.get("title") or "").strip()
    if not title:
        return None
    t_norm = _norm(title)
    score = 0.0
    # Title closeness: exact > substring > token overlap.
    if want_title_norm and t_norm == want_title_norm:
        score += 10
    elif want_title_norm and (want_title_norm in t_norm or t_norm in want_title_norm):
        score += 5
    else:
        score += 2 * len(_tokens(title) & want_title_tokens)
    # Author agreement.
    authors = " ".join(info.get("authors") or [])
    if want_author_tokens and (_tokens(authors) & want_author_tokens):
        score += 3
    # Knockoff penalty (title + subtitle + publisher).
    haystack = f"{title} {info.get('subtitle') or ''} {info.get('publisher') or ''}"
    if _JUNK_RE.search(haystack):
        score -= 8
    # Tie-breaker: when title/author scores tie, prefer the edition that
    # actually carries a description, so we don't store blurb=NULL while a
    # blurb-bearing edition of the same book was available.
    if (info.get("description") or "").strip():
        score += 0.5
    return score


def _select_best(
    data: dict, want_title: str, want_author: str | None
) -> BookMeta | None:
    """Pick the best-scoring volume among the candidates. Returns the best even
    if its score is negative (a real-but-polluted hit beats dropping the book);
    only returns None when there are no usable items."""
    items = (data or {}).get("items") or []
    wt_norm = _norm(want_title)
    wt_tokens = _tokens(want_title)
    wa_tokens = _tokens(want_author or "")

    best_info: dict | None = None
    best_score: float | None = None
    for it in items:
        info = (it or {}).get("volumeInfo") or {}
        s = _score_volume(info, wt_norm, wt_tokens, wa_tokens)
        if s is None:
            continue
        if best_score is None or s > best_score:
            best_score, best_info = s, info
    return _info_to_meta(best_info) if best_info else None


def _info_to_meta(info: dict) -> BookMeta:
    title = (info.get("title") or "").strip()
    authors = info.get("authors") or []
    author = ", ".join(a for a in authors if a) if authors else None
    published = info.get("publishedDate") or ""
    year = published.split("-", 1)[0] if published else None
    blurb = (info.get("description") or "").strip() or None
    return BookMeta(title=title, author=author, year=year, blurb=blurb)


# Default singleton + module-level convenience. Reassign `_default_source` (or
# monkeypatch `lookup`) to swap the backend -- e.g. tests, or a future
# Open Library source.
_default_source: BookMetadataSource = GoogleBooksSource()


def lookup(title: str, author: str | None = None) -> BookMeta | None:
    """Resolve a (title, author) guess to real catalog metadata, or None."""
    return _default_source.lookup(title, author)
