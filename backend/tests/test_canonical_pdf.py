"""Unit tests for the PDF normalizer helpers.

These tests deliberately avoid loading any real PDF; each helper takes
plain dataclasses so we can construct fixtures inline. A full real-PDF
end-to-end check lives in the manual ingest smoke step.
"""
from __future__ import annotations

import pytest

from core.canonical import normalize_pdf as npdf
from core.canonical.normalize_pdf import (
    _DraftBlock,
    _Line,
    cluster_into_blocks,
    drop_repeating_bands,
    emit_blocks,
    join_pages,
    resolve_sections,
)


def _line(page: int, y0: float, text: str, font_size: float = 10.0) -> _Line:
    return _Line(page=page, text=text, bbox=(50.0, y0, 500.0, y0 + 12.0), font_size=font_size)


# ── drop_repeating_bands ────────────────────────────────────────────────────


def test_drop_repeating_bands_removes_header_repeated_across_pages():
    lines = [
        _line(p, 30.0, "Book Title — Chapter 1") for p in range(5)
    ] + [
        _line(0, 100.0, "First real paragraph"),
        _line(1, 100.0, "Second real paragraph"),
    ]
    kept, audit = drop_repeating_bands(lines, repeat_threshold=3)
    texts = [l.text for l in kept]
    assert "Book Title — Chapter 1" not in texts
    assert "First real paragraph" in texts
    assert audit["dropped_headers_or_footers"] == 5


def test_drop_repeating_bands_keeps_unique_long_text_even_if_y_collides():
    lines = [
        _line(0, 100.0, "Unique sentence A"),
        _line(1, 100.0, "Unique sentence B"),
        _line(2, 100.0, "Unique sentence C"),
    ]
    kept, audit = drop_repeating_bands(lines, repeat_threshold=3)
    assert len(kept) == 3
    assert audit["dropped_headers_or_footers"] == 0


def test_drop_repeating_bands_strips_standalone_page_numbers():
    lines = [
        _line(0, 50.0, "Body text"),
        _line(0, 750.0, "1"),
        _line(1, 50.0, "More body text"),
        _line(1, 750.0, "2"),
    ]
    kept, audit = drop_repeating_bands(lines)
    assert all(not l.text.strip().isdigit() for l in kept)
    assert audit["dropped_page_numbers"] == 2


# ── cluster_into_blocks ─────────────────────────────────────────────────────


def test_cluster_groups_tight_lines_and_splits_on_large_gap():
    lines = [
        _line(0, 100.0, "Line one"),
        _line(0, 113.0, "Line two"),       # close gap → same block
        _line(0, 200.0, "Another block"),  # big gap → new block
    ]
    blocks = cluster_into_blocks(lines, vertical_gap=20.0)
    assert len(blocks) == 2
    assert blocks[0].text == "Line one Line two"
    assert blocks[1].text == "Another block"


def test_cluster_breaks_block_on_page_boundary():
    lines = [
        _line(0, 700.0, "End of page 1"),
        _line(1, 50.0, "Top of page 2"),  # different page → must split
    ]
    blocks = cluster_into_blocks(lines)
    assert len(blocks) == 2
    assert blocks[0].page == 0
    assert blocks[1].page == 1


# ── join_pages ──────────────────────────────────────────────────────────────


def test_join_pages_heals_hyphenated_word_across_break():
    b1 = _DraftBlock(
        page=0,
        lines=[_line(0, 700.0, "univer-")],
        bbox=(50.0, 700.0, 500.0, 712.0),
        font_size=10.0,
    )
    b2 = _DraftBlock(
        page=1,
        lines=[_line(1, 50.0, "sal truth")],
        bbox=(50.0, 50.0, 500.0, 62.0),
        font_size=10.0,
    )
    out, audit = join_pages([b1, b2])
    assert len(out) == 1
    assert "univer" in out[0].text and "sal truth" in out[0].text
    assert audit["dehyphenated_words"] == 1
    assert audit["merged_page_breaks"] == 1
    assert out[0].norm_flags.get("dehyphenated") is True


def test_join_pages_merges_mid_sentence_comma_break():
    b1 = _DraftBlock(
        page=0, lines=[_line(0, 700.0, "He turned around,")],
        bbox=(50.0, 700.0, 500.0, 712.0), font_size=10.0,
    )
    b2 = _DraftBlock(
        page=1, lines=[_line(1, 50.0, "and walked away.")],
        bbox=(50.0, 50.0, 500.0, 62.0), font_size=10.0,
    )
    out, audit = join_pages([b1, b2])
    assert len(out) == 1
    assert audit["merged_page_breaks"] == 1
    assert audit["dehyphenated_words"] == 0


def test_join_pages_does_not_merge_when_sentence_ends_cleanly():
    b1 = _DraftBlock(
        page=0, lines=[_line(0, 700.0, "The end.")],
        bbox=(50.0, 700.0, 500.0, 712.0), font_size=10.0,
    )
    b2 = _DraftBlock(
        page=1, lines=[_line(1, 50.0, "A new chapter begins.")],
        bbox=(50.0, 50.0, 500.0, 62.0), font_size=10.0,
    )
    out, audit = join_pages([b1, b2])
    assert len(out) == 2
    assert audit["merged_page_breaks"] == 0


# ── PR9: cross-chapter bleed regression ────────────────────────────────────


def test_join_pages_does_not_glue_when_next_page_starts_with_chapter_heading():
    """User-reported bug (PR9). Page 67 of a textbook ends with the
    leftover page footer text "Access for free at openstax.org"
    (the footer-band filter missed it). That string ends with the
    lowercase letter "g", so the pre-PR9 rule "prev ends lowercase ->
    merge" glued page 67's tail onto page 68's "CHAPTER 2
    Neurophysiology" heading. PR9 refuses to merge whenever the next
    page's first block looks like a heading.
    """
    prev = _DraftBlock(
        page=67,
        lines=[_line(67, 700.0, "Access for free at openstax.org")],
        bbox=(50.0, 700.0, 500.0, 712.0), font_size=10.0,
    )
    nxt = _DraftBlock(
        page=68,
        lines=[_line(68, 50.0, "CHAPTER 2 Neurophysiology")],
        bbox=(50.0, 50.0, 500.0, 62.0), font_size=18.0,
    )
    out, audit = join_pages([prev, nxt])
    assert len(out) == 2, "must NOT merge across a chapter heading boundary"
    assert audit["merged_page_breaks"] == 0


def test_join_pages_blocks_subsection_heading_too():
    """Sub-section ordinals like "2.1 Foo" also count as headings --
    a page that ended on a lowercase footer should not get fused into
    a sub-section title."""
    prev = _DraftBlock(
        page=10, lines=[_line(10, 700.0, "Access for free at openstax.org")],
        bbox=(0, 0, 0, 0), font_size=10.0,
    )
    nxt = _DraftBlock(
        page=11, lines=[_line(11, 50.0, "2.1 Neural Communication")],
        bbox=(0, 0, 0, 0), font_size=14.0,
    )
    out, _ = join_pages([prev, nxt])
    assert len(out) == 2


def test_join_pages_blocks_chinese_chapter_heading_too():
    """Chinese books with "第二章 ..." headings get the same protection."""
    prev = _DraftBlock(
        page=20, lines=[_line(20, 700.0, "footer text running long")],
        bbox=(0, 0, 0, 0), font_size=10.0,
    )
    nxt = _DraftBlock(
        page=21, lines=[_line(21, 50.0, "第二章 神经生理学")],
        bbox=(0, 0, 0, 0), font_size=18.0,
    )
    out, _ = join_pages([prev, nxt])
    assert len(out) == 2


def test_join_pages_still_merges_legit_mid_clause_lowercase_continuation():
    """Negative regression: a real mid-clause break where BOTH prev
    tail and next head are lowercase MUST still merge. PR9 tightened
    the rule but did not eliminate this happy path."""
    prev = _DraftBlock(
        page=0, lines=[_line(0, 700.0, "the next chapter explores how")],
        bbox=(0, 0, 0, 0), font_size=10.0,
    )
    nxt = _DraftBlock(
        page=1, lines=[_line(1, 50.0, "the brain processes language.")],
        bbox=(0, 0, 0, 0), font_size=10.0,
    )
    out, audit = join_pages([prev, nxt])
    assert len(out) == 1, "real lowercase-to-lowercase continuation should still join"
    assert audit["merged_page_breaks"] == 1


def test_join_pages_does_not_merge_when_prev_lowercase_but_next_uppercase():
    """Even without a heading-pattern next block, an uppercase start
    on the next page suggests a new sentence -- do not blindly glue."""
    prev = _DraftBlock(
        page=0, lines=[_line(0, 700.0, "ending with lowercase word")],
        bbox=(0, 0, 0, 0), font_size=10.0,
    )
    nxt = _DraftBlock(
        page=1, lines=[_line(1, 50.0, "A new sentence begins here.")],
        bbox=(0, 0, 0, 0), font_size=10.0,
    )
    out, _ = join_pages([prev, nxt])
    assert len(out) == 2


def test_join_pages_merged_from_pages_audit_records_both_pages():
    """Pre-PR9, the merged_from_pages audit field only recorded the
    next page's number, so a 3-page-wide merge ended up as [N, N, N]
    instead of [N-1, N, N+1]. PR9 fixes the bookkeeping."""
    p0 = _DraftBlock(
        page=5, lines=[_line(5, 700.0, "ending with lowercase word and")],
        bbox=(0, 0, 0, 0), font_size=10.0,
    )
    p1 = _DraftBlock(
        page=6, lines=[_line(6, 50.0, "continuing lowercase on page 6 and")],
        bbox=(0, 0, 0, 0), font_size=10.0,
    )
    p2 = _DraftBlock(
        page=7, lines=[_line(7, 50.0, "still going on page 7.")],
        bbox=(0, 0, 0, 0), font_size=10.0,
    )
    out, _ = join_pages([p0, p1, p2])
    assert len(out) == 1
    pages = out[0].norm_flags.get("merged_from_pages", [])
    assert 5 in pages and 6 in pages and 7 in pages
    assert pages.count(5) >= 1 and pages.count(7) >= 1


# ── resolve_sections (heuristic / no outline) ───────────────────────────────


class _FakeDoc:
    """Stand-in for fitz.Document. Only exposes get_toc() + len()."""

    def __init__(self, toc=None, pages: int = 1):
        self._toc = toc or []
        self._pages = pages

    def __len__(self):
        return self._pages

    def get_toc(self):
        return self._toc


def test_resolve_sections_uses_outline_when_present():
    doc = _FakeDoc(toc=[[1, "Preface", 1], [1, "Chapter One", 3], [1, "Chapter Two", 8]])
    drafts = [
        _DraftBlock(page=p, lines=[_line(p, 100.0, f"body {p}")],
                    bbox=(0, 0, 0, 0), font_size=10.0)
        for p in range(10)
    ]
    secs, per_block, audit = resolve_sections(doc, drafts, book_id="bk_t")
    assert audit["section_source"] == "outline"
    assert [s.label for s in secs] == ["Preface", "Chapter One", "Chapter Two"]
    assert all(s.source == "outline" for s in secs)
    # Page 0 → Preface; page 3..7 → Chapter One; page 8..9 → Chapter Two
    assert per_block[0] == secs[0].section_id
    assert per_block[3] == secs[1].section_id
    assert per_block[8] == secs[2].section_id


def test_resolve_sections_falls_back_to_heuristic_when_no_outline():
    doc = _FakeDoc(toc=[], pages=3)
    # First block is a big-font heading; rest is body.
    drafts = [
        _DraftBlock(page=0, lines=[_line(0, 100.0, "PART I")], bbox=(0, 0, 0, 0), font_size=18.0),
        _DraftBlock(page=0, lines=[_line(0, 200.0, "body 1")], bbox=(0, 0, 0, 0), font_size=10.0),
        _DraftBlock(page=1, lines=[_line(1, 100.0, "body 2")], bbox=(0, 0, 0, 0), font_size=10.0),
    ]
    secs, per_block, audit = resolve_sections(doc, drafts, book_id="bk_t")
    assert audit["section_source"] == "inferred"
    assert secs[0].source == "inferred"
    assert secs[0].label == "PART I"
    # All blocks assigned to the only inferred section
    assert per_block.count(secs[0].section_id) == 3


def test_resolve_sections_synthetic_body_when_nothing_detected():
    doc = _FakeDoc(toc=[], pages=1)
    drafts = [
        _DraftBlock(page=0, lines=[_line(0, 100.0, "uniform paragraph")],
                    bbox=(0, 0, 0, 0), font_size=10.0),
    ]
    secs, per_block, audit = resolve_sections(doc, drafts, book_id="bk_t")
    assert len(secs) == 1
    assert secs[0].label == "Body"
    assert secs[0].source == "inferred"
    assert per_block == [secs[0].section_id]


# ── emit_blocks ─────────────────────────────────────────────────────────────


def test_emit_blocks_assigns_stable_ids_and_monotonic_offsets():
    drafts = [
        _DraftBlock(page=0, lines=[_line(0, 100.0, "Hello")],
                    bbox=(10, 100, 90, 112), font_size=10.0,
                    norm_flags={"dehyphenated": True}),
        _DraftBlock(page=0, lines=[_line(0, 130.0, "World again")],
                    bbox=(10, 130, 90, 142), font_size=10.0),
    ]
    sec_ids = ["sec_a", "sec_a"]
    blocks = emit_blocks(drafts, sec_ids, book_id="bk_t")
    assert [b.order_idx for b in blocks] == [0, 1]
    assert blocks[0].block_id != blocks[1].block_id
    assert blocks[0].book_offset_end == len("Hello")
    assert blocks[1].book_offset_start == blocks[0].book_offset_end + 1
    assert blocks[1].book_offset_end - blocks[1].book_offset_start == len("World again")
    # Locator is 1-based page + rounded bbox
    assert blocks[0].locator["page"] == 1
    assert blocks[0].locator["bbox"] == [10.0, 100.0, 90.0, 112.0]
    # Section id and norm flags survive
    assert blocks[0].section_id == "sec_a"
    assert blocks[0].norm_flags == {"dehyphenated": True}


def test_emit_blocks_ids_are_deterministic_across_calls():
    drafts = [
        _DraftBlock(page=0, lines=[_line(0, 100.0, "X")], bbox=(0, 0, 0, 0), font_size=10.0)
        for _ in range(5)
    ]
    sec_ids = [None] * 5
    a = emit_blocks(drafts, sec_ids, book_id="bk_t")
    b = emit_blocks(drafts, sec_ids, book_id="bk_t")
    assert [x.block_id for x in a] == [x.block_id for x in b]


# ── End-to-end on a tiny in-memory fixture ──────────────────────────────────


def test_normalize_full_pipeline_smoke(monkeypatch):
    """Build a fake fitz.Document substitute and walk the full normalize()
    function end-to-end.
    """

    class _FakePage:
        def __init__(self, idx, lines):
            self._idx = idx
            self._lines = lines

        def get_text(self, mode):
            assert mode == "dict"
            return {
                "blocks": [
                    {
                        "type": 0,
                        "bbox": (50.0, 100.0, 500.0, 800.0),
                        "lines": [
                            {
                                "bbox": (50.0, y0, 500.0, y0 + 12.0),
                                "spans": [{"text": text, "size": fs}],
                            }
                            for (y0, text, fs) in self._lines
                        ],
                    }
                ]
            }

    class _FakeDocFull:
        def __init__(self, pages):
            self._pages = pages
            self._toc = [[1, "Chapter 1", 1]]

        def __len__(self):
            return len(self._pages)

        def __getitem__(self, i):
            return self._pages[i]

        def get_toc(self):
            return self._toc

    pages = [
        _FakePage(
            0,
            [
                (30.0, "Repeating Header", 8.0),   # appears on every page
                (100.0, "The story starts here,", 10.0),
                (200.0, "1", 8.0),                   # page number
            ],
        ),
        _FakePage(
            1,
            [
                (30.0, "Repeating Header", 8.0),
                (100.0, "and continues smoothly.", 10.0),
                (200.0, "2", 8.0),
            ],
        ),
        _FakePage(
            2,
            [
                (30.0, "Repeating Header", 8.0),
                (100.0, "A standalone closing line.", 10.0),
                (200.0, "3", 8.0),
            ],
        ),
    ]
    doc = _FakeDocFull(pages)
    sections, blocks, report = npdf.normalize(doc, book_id="bk_smoke")

    # Header band + page numbers stripped
    assert report["dropped_headers_or_footers"] >= 3
    assert report["dropped_page_numbers"] == 3

    # Cross-page join: "starts here," + "and continues smoothly." should merge
    merged_texts = [b.text for b in blocks]
    assert any("starts here" in t and "continues smoothly" in t for t in merged_texts)

    # Section source is outline
    assert report["section_source"] == "outline"
    assert all(b.section_id == sections[0].section_id for b in blocks)

    # Offsets monotone
    for prev, nxt in zip(blocks, blocks[1:]):
        assert nxt.book_offset_start > prev.book_offset_end

    # IDs deterministic when re-running
    s2, b2, _ = npdf.normalize(doc, book_id="bk_smoke")
    assert [b.block_id for b in b2] == [b.block_id for b in blocks]
