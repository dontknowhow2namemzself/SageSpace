"""Tests for the Google Books metadata lookup (core/books_api.py).

The real HTTP call is never made -- httpx.get is stubbed. Covers: parse of a
real volume; the clean not-found -> None path; the transient-failure ->
BookLookupUnavailable path (network error, 429/non-200) that lets recommend()
distinguish "couldn't validate" from "doesn't exist"; the empty-title
short-circuit; and the intitle/inauthor query assembly.
"""
import pytest

import core.books_api as books_api
from core.books_api import GoogleBooksSource, BookMeta, BookLookupUnavailable


class _FakeResp:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def json(self):
        return self._payload


def _patch_get(monkeypatch, *, payload=None, status_code=200, exc=None, capture=None):
    def fake_get(url, params=None, timeout=None):
        if capture is not None:
            capture["url"] = url
            capture["params"] = params
        if exc:
            raise exc
        return _FakeResp(payload, status_code=status_code)
    monkeypatch.setattr(books_api.httpx, "get", fake_get)


def test_lookup_parses_volume(monkeypatch):
    _patch_get(monkeypatch, payload={"items": [{"volumeInfo": {
        "title": "Alice's Adventures in Wonderland",
        "authors": ["Lewis Carroll"],
        "publishedDate": "1865-11-26",
        "description": "A girl tumbles down a rabbit hole.",
    }}]})
    meta = GoogleBooksSource().lookup("Alice in Wonderland", "Carroll")
    assert isinstance(meta, BookMeta)
    assert meta.title == "Alice's Adventures in Wonderland"
    assert meta.author == "Lewis Carroll"
    assert meta.year == "1865"  # trimmed from the full date
    assert "rabbit hole" in meta.blurb


def test_lookup_clean_not_found_returns_none(monkeypatch):
    """200 with no items == validated 'no such book' -> None (drop as a
    hallucination)."""
    _patch_get(monkeypatch, payload={"items": []})
    assert GoogleBooksSource().lookup("No Such Book", "Nobody") is None


def test_lookup_network_error_raises_unavailable(monkeypatch):
    _patch_get(monkeypatch, exc=RuntimeError("connection reset"))
    with pytest.raises(BookLookupUnavailable):
        GoogleBooksSource().lookup("Anything")


def test_lookup_rate_limited_raises_unavailable(monkeypatch):
    """429 (keyless quota) is 'couldn't validate', NOT 'doesn't exist'."""
    _patch_get(monkeypatch, payload={}, status_code=429)
    with pytest.raises(BookLookupUnavailable):
        GoogleBooksSource().lookup("Popular Book")


def test_lookup_empty_title_short_circuits(monkeypatch):
    called = {"hit": False}

    def boom(*a, **k):
        called["hit"] = True
        raise AssertionError("should not call the network for an empty title")

    monkeypatch.setattr(books_api.httpx, "get", boom)
    assert GoogleBooksSource().lookup("   ") is None
    assert called["hit"] is False


def test_lookup_builds_intitle_inauthor_query(monkeypatch):
    capture = {}
    _patch_get(monkeypatch, payload={"items": []}, capture=capture)
    GoogleBooksSource().lookup("Meditations", "Marcus Aurelius")
    assert 'intitle:"Meditations"' in capture["params"]["q"]
    assert 'inauthor:"Marcus Aurelius"' in capture["params"]["q"]


def test_lookup_includes_api_key_when_env_set(monkeypatch):
    monkeypatch.setenv("GOOGLE_BOOKS_API_KEY", "test-key-123")
    capture = {}
    _patch_get(monkeypatch, payload={"items": []}, capture=capture)
    GoogleBooksSource().lookup("Meditations")
    assert capture["params"].get("key") == "test-key-123"


def test_lookup_omits_api_key_when_env_unset(monkeypatch):
    monkeypatch.delenv("GOOGLE_BOOKS_API_KEY", raising=False)
    capture = {}
    _patch_get(monkeypatch, payload={"items": []}, capture=capture)
    GoogleBooksSource().lookup("Meditations")
    assert "key" not in capture["params"]


def test_lookup_handles_missing_authors_and_description(monkeypatch):
    _patch_get(monkeypatch, payload={"items": [{"volumeInfo": {"title": "Bare Book"}}]})
    meta = GoogleBooksSource().lookup("Bare Book")
    assert meta.title == "Bare Book"
    assert meta.author is None
    assert meta.year is None
    assert meta.blurb is None


def test_module_level_lookup_delegates(monkeypatch):
    _patch_get(monkeypatch, payload={"items": [{"volumeInfo": {"title": "X"}}]})
    assert books_api.lookup("X").title == "X"


# ── candidate selection (avoid summary/study-guide knockoffs) ────────────────


def test_lookup_requests_multiple_candidates(monkeypatch):
    capture = {}
    _patch_get(monkeypatch, payload={"items": []}, capture=capture)
    GoogleBooksSource().lookup("X")
    assert int(capture["params"]["maxResults"]) >= 5  # not just the top hit


def test_lookup_prefers_real_edition_over_summary(monkeypatch):
    # Google ranks the knockoff first; we must still pick Kahneman's real book.
    _patch_get(monkeypatch, payload={"items": [
        {"volumeInfo": {
            "title": "Summary of Thinking, Fast and Slow",
            "authors": ["Readtrepreneur Publishing"],
            "description": "A summary of the book.",
        }},
        {"volumeInfo": {
            "title": "Thinking, Fast and Slow",
            "authors": ["Daniel Kahneman"],
            "publishedDate": "2011",
            "description": "The real book.",
        }},
    ]})
    meta = GoogleBooksSource().lookup("Thinking, Fast and Slow", "Daniel Kahneman")
    assert meta.title == "Thinking, Fast and Slow"
    assert meta.author == "Daniel Kahneman"


def test_lookup_penalizes_summary_by_publisher_and_subtitle(monkeypatch):
    # Exact-title knockoff loses to a looser-title real edition via the penalty.
    _patch_get(monkeypatch, payload={"items": [
        {"volumeInfo": {
            "title": "Atomic Habits",
            "subtitle": "A Study Guide and Workbook",
            "authors": ["SparkNotes"],
        }},
        {"volumeInfo": {
            "title": "Atomic Habits",
            "authors": ["James Clear"],
            "description": "Tiny changes, remarkable results.",
        }},
    ]})
    meta = GoogleBooksSource().lookup("Atomic Habits", "James Clear")
    assert meta.author == "James Clear"


def test_lookup_tiebreak_prefers_edition_with_description(monkeypatch):
    # Two identical-scoring editions (same title + author); the one WITH a
    # description must win even though Google listed it second, so blurb != NULL.
    _patch_get(monkeypatch, payload={"items": [
        {"volumeInfo": {"title": "Atomic Habits", "authors": ["James Clear"]}},  # no desc
        {"volumeInfo": {"title": "Atomic Habits", "authors": ["James Clear"],
                        "description": "Tiny changes, remarkable results."}},
    ]})
    meta = GoogleBooksSource().lookup("Atomic Habits", "James Clear")
    assert meta.blurb == "Tiny changes, remarkable results."


def test_lookup_returns_best_even_when_only_knockoff(monkeypatch):
    # If a summary is the ONLY hit, return it (real book preserved) rather than
    # dropping the pick to None (which would fire the hallucination guard).
    _patch_get(monkeypatch, payload={"items": [
        {"volumeInfo": {"title": "Summary of Foo", "authors": ["X Publishing"]}},
    ]})
    assert GoogleBooksSource().lookup("Foo", "Real Author") is not None
