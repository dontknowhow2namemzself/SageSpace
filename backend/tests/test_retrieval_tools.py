"""Tests for the four pure retrieval tools (PR2 step 2).

The tools wrap existing retrieval engines and must (a) tag every doc with
the right `origin`, (b) map keyword/neighbor block hits up to their
containing level-0 chunk, and (c) stay side-effect-free + crash-free.
A FakeVectorstore stands in for Chroma's `.get(where=...)`.
"""
import pytest
from langchain_core.documents import Document

import core.retrieval_tools as rt
from core.canonical import db as canonical_db


class FakeVS:
    """Minimal Chroma stand-in supporting the two .get(where=...) shapes the
    tools issue: by raptor_level (index build) and by chunk_id $in (fetch)."""

    def __init__(self, chunks):
        self.chunks = chunks  # list of dicts, each with 'text' + metadata keys

    def get(self, where=None):
        rows = self.chunks
        where = where or {}
        if "raptor_level" in where:
            rows = [c for c in rows if c.get("raptor_level", 0) == where["raptor_level"]]
        if "chunk_id" in where:
            ids = set(where["chunk_id"]["$in"])
            rows = [c for c in rows if c.get("chunk_id") in ids]
        return {
            "documents": [c.get("text", "") for c in rows],
            "metadatas": [{k: v for k, v in c.items() if k != "text"} for c in rows],
        }


def _chunk(cid, text, block_ids, primary, **extra):
    return {
        "chunk_id": cid, "text": text, "block_ids": block_ids,
        "primary_block_id": primary, "raptor_level": 0, **extra,
    }


# ── semantic_search ─────────────────────────────────────────────────────────


def test_semantic_search_tags_origin(monkeypatch):
    docs = [Document(page_content="x", metadata={"chunk_id": "c1", "retrieval_origin": "hyde"})]
    monkeypatch.setattr(rt, "retrieve_combined", lambda q, vs: docs)
    out = rt.tool_semantic_search("who is alice", FakeVS([]))
    assert [d.metadata["origin"] for d in out] == ["semantic"]


def test_semantic_search_empty_guards():
    assert rt.tool_semantic_search("", FakeVS([])) == []
    assert rt.tool_semantic_search("q", None) == []


# ── keyword_search ──────────────────────────────────────────────────────────


def test_keyword_search_maps_block_hits_to_chunks(monkeypatch):
    vs = FakeVS([
        _chunk("chk_A", "Alice met the Cheshire Cat", "b1,b2", "b1",
               section_label="Ch6", page=65),
        _chunk("chk_B", "the Queen made tarts", "b3", "b3"),
    ])
    # FTS hit lands on block b2, which lives inside chunk chk_A.
    monkeypatch.setattr(
        rt, "search_blocks_fts",
        lambda book_id, terms, limit=8: [{"block_id": "b2", "text": "...", "snippet": "[Cat]"}],
    )
    out = rt.tool_keyword_search("Cheshire Cat", "bk", vs)
    assert [d.metadata["chunk_id"] for d in out] == ["chk_A"]
    assert out[0].metadata["origin"] == "keyword"
    assert "Cheshire" in out[0].page_content


def test_keyword_search_dedups_multiple_blocks_same_chunk(monkeypatch):
    vs = FakeVS([_chunk("chk_A", "Alice and the Cat", "b1,b2,b3", "b1")])
    monkeypatch.setattr(
        rt, "search_blocks_fts",
        lambda *a, **k: [{"block_id": "b1"}, {"block_id": "b2"}, {"block_id": "b3"}],
    )
    out = rt.tool_keyword_search("alice cat", "bk", vs)
    assert [d.metadata["chunk_id"] for d in out] == ["chk_A"]  # one chunk, not three


def test_keyword_search_no_fts_hits(monkeypatch):
    monkeypatch.setattr(rt, "search_blocks_fts", lambda *a, **k: [])
    assert rt.tool_keyword_search("nothing", "bk", FakeVS([])) == []


# ── get_chapter ─────────────────────────────────────────────────────────────


def test_get_chapter_tags_origin(monkeypatch):
    docs = [Document(page_content="ch6 summary", metadata={"chunk_id": "raptor_l1_s6"})]
    monkeypatch.setattr(rt, "fetch_chapter_docs", lambda book_id, n, vs, query=None: docs)
    out = rt.tool_get_chapter(6, "bk", FakeVS([]))
    assert [d.metadata["origin"] for d in out] == ["chapter"]


# ── expand_neighbors ────────────────────────────────────────────────────────


def test_expand_neighbors_returns_adjacent_excluding_seed(monkeypatch):
    vs = FakeVS([
        _chunk("chk_prev", "before", "b4", "b4"),
        _chunk("chk_mid", "seed text", "b5", "b5"),
        _chunk("chk_next", "after", "b6", "b6"),
    ])
    monkeypatch.setattr(canonical_db, "get_block",
                        lambda book_id, bid: {"block_id": "b5", "order_idx": 5})
    monkeypatch.setattr(canonical_db, "get_blocks",
                        lambda book_id, **kw: [{"block_id": "b4"}, {"block_id": "b5"}, {"block_id": "b6"}])
    out = rt.tool_expand_neighbors("chk_mid", "bk", vs)
    cids = {d.metadata["chunk_id"] for d in out}
    assert cids == {"chk_prev", "chk_next"}  # seed chk_mid excluded
    assert all(d.metadata["origin"] == "neighbor" for d in out)


def test_expand_neighbors_anchors_at_edges_not_midpoint(monkeypatch):
    """A seed chunk spanning blocks b10..b18 (9 blocks, wider than the +/-3
    window). Anchoring on the EDGE blocks must still reach the neighbor
    chunks; the old midpoint anchor would have kept the window inside the
    seed (b11..b17) and surfaced nothing."""
    vs = FakeVS([
        _chunk("chk_before", "before", "b9", "b9"),
        _chunk("chk_seed", "long seed",
               "b10,b11,b12,b13,b14,b15,b16,b17,b18", "b14"),
        _chunk("chk_after", "after", "b19", "b19"),
    ])
    order = {f"b{i}": i for i in range(9, 20)}
    monkeypatch.setattr(canonical_db, "get_block",
                        lambda book_id, bid: {"block_id": bid, "order_idx": order[bid]})

    def fake_get_blocks(book_id, *, after_order_idx, limit):
        start = after_order_idx + 1
        return [{"block_id": f"b{i}"} for i in range(start, start + limit)
                if f"b{i}" in order]

    monkeypatch.setattr(canonical_db, "get_blocks", fake_get_blocks)
    out = rt.tool_expand_neighbors("chk_seed", "bk", vs)
    assert {d.metadata["chunk_id"] for d in out} == {"chk_before", "chk_after"}


def test_expand_neighbors_unknown_chunk(monkeypatch):
    assert rt.tool_expand_neighbors("nope", "bk", FakeVS([])) == []
    assert rt.tool_expand_neighbors("", "bk", FakeVS([])) == []


# ── shared helpers ──────────────────────────────────────────────────────────


def test_block_to_chunk_index_first_chunk_wins():
    vs = FakeVS([
        _chunk("chk_A", "a", "b1,b2", "b1"),
        _chunk("chk_B", "b", "b3", "b3"),
        {"chunk_id": "raptor_l1_x", "text": "summary", "block_ids": "b1,b2,b3",
         "primary_block_id": "b1", "raptor_level": 1},  # level-1 ignored
    ])
    idx = rt._block_to_chunk_index(vs)
    assert idx == {"b1": "chk_A", "b2": "chk_A", "b3": "chk_B"}


def test_fetch_chunk_docs_preserves_requested_order():
    vs = FakeVS([_chunk("c1", "one", "b1", "b1"), _chunk("c2", "two", "b2", "b2")])
    out = rt._fetch_chunk_docs(vs, ["c2", "c1"])
    assert [d.metadata["chunk_id"] for d in out] == ["c2", "c1"]
