"""Tests for the canonical chunker.

The chunker is the bridge between the canonical text layer (sections /
blocks) and the retrieval layer (chunks in Chroma). Two correctness
contracts matter most and are both exercised here:

  1. Chunks NEVER cross section boundaries.
  2. Each chunk's block_ids precisely cover the canonical blocks its text
     comes from, so any retrieval hit can be resolved back to canonical
     spans without ambiguity.
"""
from __future__ import annotations

import pytest

from core.canonical.chunker import (
    chunk_blocks,
    decode_block_ids,
    encode_block_ids,
    _build_section_window,
    _resolve_span,
)


def _block(block_id: str, section_id: str, text: str, locator: dict | None = None) -> dict:
    return {
        "block_id": block_id,
        "section_id": section_id,
        "text": text,
        "locator": locator or {},
        "locator_type": "pdf" if locator and "page" in locator else "epub",
    }


# ── encode / decode helpers ────────────────────────────────────────────────


def test_encode_decode_block_ids_roundtrip():
    ids = ["blk_aaa", "blk_bbb", "blk_ccc"]
    assert decode_block_ids(encode_block_ids(ids)) == ids
    assert decode_block_ids("") == []
    assert decode_block_ids(None) == []


# ── _resolve_span ───────────────────────────────────────────────────────────


def test_resolve_span_single_block_chunk():
    win = _build_section_window(
        "s1",
        [_block("b0", "s1", "Hello world"), _block("b1", "s1", "Second one"),
         _block("b2", "s1", "Third here")],
    )
    # "world" sits entirely inside block 0
    out = _resolve_span(win, win.text.index("world"), win.text.index("world") + len("world"))
    assert out["block_ids"] == ["b0"]
    assert out["primary_block_id"] == "b0"
    assert out["first_idx"] == out["last_idx"] == 0


def test_resolve_span_multi_block_chunk_marks_primary_at_midpoint():
    win = _build_section_window(
        "s1",
        [
            _block("b0", "s1", "AAA"),
            _block("b1", "s1", "BBB"),
            _block("b2", "s1", "CCC"),
        ],
    )
    # Whole window
    out = _resolve_span(win, 0, len(win.text))
    assert out["block_ids"] == ["b0", "b1", "b2"]
    # Mid char of "AAA\nBBB\nCCC" lands in BBB
    assert out["primary_block_id"] == "b1"


def test_resolve_span_handles_tiny_chunk_at_boundary():
    win = _build_section_window(
        "s1",
        [_block("b0", "s1", "first"), _block("b1", "s1", "second")],
    )
    # The "\n" separator sits between blocks. A range that lands on the join
    # should still pick a single block, not crash.
    join_pos = len("first")  # index of '\n'
    out = _resolve_span(win, join_pos, join_pos + 1)
    assert out["block_ids"] in (["b0"], ["b1"])


# ── End-to-end chunk_blocks ─────────────────────────────────────────────────


def _stub_section_blocks(
    book_id: str, n_blocks_per_section: int, sections: list[str], words_per_block: int = 20
) -> list[dict]:
    """Build deterministic blocks: each section has N paragraph blocks, each
    block is a unique sentence so we can grep the chunker output.
    """
    out: list[dict] = []
    counter = 0
    for sec in sections:
        for i in range(n_blocks_per_section):
            counter += 1
            words = [f"word{counter:03d}_{w}" for w in range(words_per_block)]
            out.append(_block(f"{book_id}_b{counter:04d}", sec, " ".join(words),
                              locator={"page": 1 + counter // 4}))
    return out


def test_chunks_never_cross_sections():
    """A chunk's block_ids must all belong to the same section."""
    blocks = _stub_section_blocks("bk_t", n_blocks_per_section=6, sections=["s_A", "s_B", "s_C"])
    section_of = {b["block_id"]: b["section_id"] for b in blocks}

    chunks = chunk_blocks("bk_t", blocks, chunk_size=400, chunk_overlap=50)
    assert chunks, "expected non-empty chunk list"
    for ch in chunks:
        bids = decode_block_ids(ch.metadata["block_ids"])
        sections = {section_of[bid] for bid in bids}
        assert len(sections) == 1, (
            f"chunk {ch.metadata['chunk_id']} crosses sections {sections}: "
            f"text={ch.page_content[:60]!r}"
        )
        # primary_block_id must be one of the chunk's block_ids
        assert ch.metadata["primary_block_id"] in bids


def test_chunk_block_ids_cover_actual_text():
    """Every chunk's text must appear verbatim within the concatenation of
    its referenced blocks (allowing for the BLOCK_JOIN separator).
    """
    blocks = _stub_section_blocks("bk_t", n_blocks_per_section=4, sections=["s_A", "s_B"])
    by_id = {b["block_id"]: b for b in blocks}
    chunks = chunk_blocks("bk_t", blocks, chunk_size=350, chunk_overlap=40)
    assert chunks
    for ch in chunks:
        bids = decode_block_ids(ch.metadata["block_ids"])
        joined = "\n".join(by_id[bid]["text"] for bid in bids)
        # Every chunk piece must be present somewhere in the joined text.
        # Strip whitespace to be tolerant of separator splits.
        assert ch.page_content.strip() in joined, (
            f"chunk text not found in its blocks: chunk={ch.page_content[:60]!r}\n"
            f"first 100 of joined: {joined[:100]!r}"
        )


def test_chunk_ids_are_stable_across_runs():
    """Re-chunking the same blocks must produce identical chunk_ids in the
    same order. (Stability is critical for re-ingest without rebuilding the
    entire Chroma collection from scratch later.)
    """
    blocks = _stub_section_blocks("bk_t", n_blocks_per_section=3, sections=["s_A", "s_B"])
    a = chunk_blocks("bk_t", blocks)
    b = chunk_blocks("bk_t", blocks)
    assert [c.metadata["chunk_id"] for c in a] == [c.metadata["chunk_id"] for c in b]


def test_chunk_metadata_has_required_fields_and_legacy_mirrors():
    blocks = _stub_section_blocks("bk_t", n_blocks_per_section=3, sections=["s_A"])
    section_label_by_id = {"s_A": "Chapter One"}
    section_order_by_id = {"s_A": 0}
    chunks = chunk_blocks(
        "bk_t",
        blocks,
        section_label_by_id=section_label_by_id,
        section_order_by_id=section_order_by_id,
    )
    assert chunks
    m = chunks[0].metadata
    # New canonical
    assert m["section_id"] == "s_A"
    assert m["primary_block_id"].startswith("bk_t_b")
    assert m["block_ids"]  # non-empty
    assert m["raptor_level"] == 0
    assert m["book_id"] == "bk_t"
    # Legacy mirrors
    assert m["chapter"] == 1
    assert m["chapter_label"] == "Chapter One"
    # PDF page mirror should be present and >= 1
    assert m["page"] >= 1


def test_chunker_handles_blocks_with_no_section_id():
    """If section_id is None (e.g., a malformed legacy ingest), the chunker
    still runs and emits chunks; section_id metadata is empty string.
    """
    blocks = [
        _block("b0", "", "Paragraph alpha words " * 10),
        _block("b1", "", "Paragraph beta words " * 10),
    ]
    chunks = chunk_blocks("bk_t", blocks)
    assert chunks
    assert all(c.metadata["section_id"] == "" for c in chunks)
    assert all(c.metadata["chapter"] == 0 for c in chunks)


def test_chunker_skips_empty_section():
    """A section made of only whitespace blocks must not crash and produces
    no chunks for that section.
    """
    blocks = [
        _block("b_empty", "s_empty", "   "),
        _block("b_real", "s_real", "Real content goes here. " * 20),
    ]
    chunks = chunk_blocks("bk_t", blocks)
    for ch in chunks:
        assert ch.metadata["section_id"] == "s_real"


# ── End-to-end on a real ingested book ──────────────────────────────────────


def test_chunker_against_real_v2_book():
    """If the dev DB happens to hold a canonicalized book, chunk it and
    verify all global contracts hold. Skips otherwise.
    """
    from pathlib import Path
    import sqlite3

    db_path = Path(__file__).parent.parent / "sagespace.db"
    if not db_path.exists():
        pytest.skip("no real DB available")

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT id, title FROM books WHERE ingest_version = 2 LIMIT 1"
    ).fetchone()
    if not row:
        conn.close()
        pytest.skip("no v2 book in real DB")

    book_id = row["id"]
    from core.canonical import db as canonical_db

    blocks = canonical_db.get_blocks(book_id)
    sections = canonical_db.get_sections(book_id)
    section_label = {s["section_id"]: s["label"] for s in sections}
    section_order = {s["section_id"]: s["order_idx"] for s in sections}
    conn.close()

    chunks = chunk_blocks(
        book_id, blocks,
        section_label_by_id=section_label,
        section_order_by_id=section_order,
    )
    assert chunks, "expected at least one chunk for a non-trivial book"

    # Global contracts on the real book
    block_to_section = {b["block_id"]: b["section_id"] for b in blocks}
    for ch in chunks:
        bids = decode_block_ids(ch.metadata["block_ids"])
        secs = {block_to_section[bid] for bid in bids}
        assert len(secs) == 1, f"chunk crosses sections: {secs}"
        assert ch.metadata["primary_block_id"] in bids
