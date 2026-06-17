"""Tests for the recommend() pipeline (core/recommend.py).

The LLM and the catalog HTTP call are both stubbed; we test the orchestration:
interest gathering, exclude filtering, the hallucination guard (lookup None ->
drop), in-batch dedup, persistence as 'suggested', and the cold-start
short-circuit. Plus llm_recommend's parse/cap/failure behavior.
"""
import json
from unittest.mock import MagicMock

import pytest

import core.database as db_module
import core.recommend as rec_mod
import core.books_api as books_api
from core.recommend import recommend, llm_recommend, _RecPick, _is_interest_question
from core.database import (
    init_db, create_book, add_memory_note, list_recommendations,
    create_session, save_conversation,
)


@pytest.fixture(autouse=True)
def use_test_db(tmp_path, monkeypatch):
    monkeypatch.setattr(db_module, "DB_PATH", tmp_path / "test.db")
    init_db()


def _pick(title, author="A", reason="因为你喜欢X", which="X", stretch=False):
    return _RecPick(
        title=title, author=author, reason=reason,
        which_interest=which, is_stretch=stretch,
    )


def _patch_llm(monkeypatch, picks):
    monkeypatch.setattr(
        rec_mod, "llm_recommend", lambda interests, exclude, n=3: list(picks)
    )


def _patch_lookup_echo(monkeypatch, found=True):
    """lookup -> BookMeta echoing the requested title (found) or None."""
    def fake(title, author=None):
        if not found:
            return None
        return books_api.BookMeta(title=title, author=author, year="2020", blurb="b")
    monkeypatch.setattr(books_api, "lookup", fake)


# ── recommend() orchestration ────────────────────────────────────────────────


def test_recommend_inserts_validated_picks(monkeypatch):
    create_book("Owned", "Auth", "/tmp/o.pdf")  # leaves cold start
    _patch_llm(monkeypatch, [_pick("New One"), _pick("New Two")])
    _patch_lookup_echo(monkeypatch, found=True)
    out = recommend()
    assert {r["title"] for r in out} == {"New One", "New Two"}
    assert {r["status"] for r in out} == {"suggested"}
    assert len(list_recommendations(status="suggested")) == 2


def test_recommend_drops_hallucinations(monkeypatch):
    create_book("Owned", "Auth", "/tmp/o.pdf")
    _patch_llm(monkeypatch, [_pick("Real"), _pick("Fake")])

    def fake_lookup(title, author=None):
        return books_api.BookMeta(title=title) if title == "Real" else None

    monkeypatch.setattr(books_api, "lookup", fake_lookup)
    out = recommend()
    assert {r["title"] for r in out} == {"Real"}  # unfound pick dropped


def test_recommend_keeps_pick_when_catalog_unavailable(monkeypatch):
    """A 429/network failure is 'couldn't validate', not 'fake' -> keep the
    pick (degraded: no blurb), don't blank the block."""
    create_book("Owned", "Auth", "/tmp/o.pdf")
    _patch_llm(monkeypatch, [_pick("Real Book", author="Real Author")])

    def unavailable(title, author=None):
        raise books_api.BookLookupUnavailable("HTTP 429")

    monkeypatch.setattr(books_api, "lookup", unavailable)
    out = recommend()
    assert len(out) == 1
    assert out[0]["title"] == "Real Book"      # title kept from the LLM pick
    assert out[0]["author"] == "Real Author"   # author kept from the LLM pick
    assert out[0]["blurb"] is None             # degraded: no enriched metadata
    assert out[0]["status"] == "suggested"


def test_recommend_excludes_library_and_prior_recs(monkeypatch):
    create_book("Owned Book", "Auth", "/tmp/o.pdf")
    db_module.insert_recommendation("Past Rec", "A", "b", "r", "X", status="seen")
    _patch_llm(monkeypatch, [_pick("Owned Book"), _pick("Past Rec"), _pick("Brand New")])
    _patch_lookup_echo(monkeypatch, found=True)
    out = recommend()
    assert {r["title"] for r in out} == {"Brand New"}  # owned + already-shown skipped


def test_recommend_dedups_within_batch(monkeypatch):
    create_book("Owned", "Auth", "/tmp/o.pdf")
    _patch_llm(monkeypatch, [_pick("Dup"), _pick("Dup")])
    _patch_lookup_echo(monkeypatch, found=True)
    assert len(recommend()) == 1


def test_recommend_cold_start_returns_empty(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("llm_recommend must not run on cold start")

    monkeypatch.setattr(rec_mod, "llm_recommend", boom)
    assert recommend() == []
    assert list_recommendations() == []


def test_recommend_uses_notes_as_signal_without_books(monkeypatch):
    add_memory_note("用户在研读斯多葛", type="interest")  # note alone leaves cold start
    captured = {}

    def fake_llm(interests, exclude, n=3):
        captured["interests"] = interests
        return [_pick("Stoic Book")]

    monkeypatch.setattr(rec_mod, "llm_recommend", fake_llm)
    _patch_lookup_echo(monkeypatch, found=True)
    out = recommend()
    assert {r["title"] for r in out} == {"Stoic Book"}
    assert "用户在研读斯多葛" in captured["interests"]


# ── llm_recommend() parse / guard ────────────────────────────────────────────


def _patch_structured(monkeypatch, payload):
    mock_structured = MagicMock()
    if isinstance(payload, Exception):
        mock_structured.invoke.side_effect = payload
    else:
        mock_structured.invoke.return_value = payload
    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value = mock_structured
    monkeypatch.setattr(rec_mod, "_build_recommender", lambda: mock_llm)


def test_llm_recommend_parses_and_caps_to_n(monkeypatch):
    _patch_structured(monkeypatch, {"picks": [
        {"title": "A", "author": "x", "reason": "r", "which_interest": "i", "is_stretch": False},
        {"title": "B", "author": "y", "reason": "r", "which_interest": "i", "is_stretch": True},
    ]})
    picks = llm_recommend(["interest"], ["owned"], n=1)
    assert len(picks) == 1  # capped at n
    assert picks[0].title == "A"


def test_llm_recommend_failure_returns_empty(monkeypatch):
    _patch_structured(monkeypatch, RuntimeError("LLM down"))
    assert llm_recommend(["i"], [], n=3) == []


# ── recent-question filtering (drop procedural / smalltalk noise) ─────────────


@pytest.mark.parametrize("q,keep", [
    ("tell", False),                                  # fragment
    ("tell me", False),                               # fragment
    ("How much have I read?", False),                 # progress
    ("Summarize this book for me.", False),           # book-meta
    ("What is this book about?", False),              # book-meta
    ("Hi! Nice to meet you. What are you?", False),   # smalltalk
    ("export these notes as pdf", False),             # export
    ("你好，以后叫我小王", False),                       # smalltalk (zh)
    ("介绍下这本书", False),                            # book-meta (zh)
    ("tell me about the Queen of Hearts", True),      # real interest
    ("What does the Cheshire Cat symbolize?", True),  # real interest
    ("我最近在思考社会规则对认知的影响", True),            # real interest (zh)
])
def test_is_interest_question(q, keep):
    assert _is_interest_question(q) is keep


def test_recommend_filters_noisy_questions(monkeypatch):
    book_id = create_book("Owned", "A", "/tmp/o.pdf")
    sid = create_session(book_id)
    save_conversation(sid, json.dumps([
        {"role": "user", "content": "How much have I read?"},                  # noise
        {"role": "user", "content": "tell"},                                    # fragment
        {"role": "user", "content": "What does the Cheshire Cat symbolize?"},   # keep
        {"role": "user", "content": "Hi! Nice to meet you."},                   # noise
    ]))
    captured = {}

    def fake_llm(interests, exclude, n=3):
        captured["interests"] = interests
        return [_pick("Some Book")]

    monkeypatch.setattr(rec_mod, "llm_recommend", fake_llm)
    _patch_lookup_echo(monkeypatch, found=True)
    recommend()

    qs = captured["interests"]
    assert "What does the Cheshire Cat symbolize?" in qs   # substantive kept
    assert "How much have I read?" not in qs               # progress dropped
    assert "tell" not in qs                                # fragment dropped
    assert "Hi! Nice to meet you." not in qs               # smalltalk dropped
