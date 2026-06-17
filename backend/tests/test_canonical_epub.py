"""Unit tests for the EPUB normalizer.

We construct minimal in-memory EPUB documents via ebooklib's own model so
the tests exercise the real library API (spine, toc, items_of_type) rather
than mocks. No actual .epub file I/O.
"""
from __future__ import annotations

import pytest

from ebooklib import epub

from core.canonical import normalize_epub as nepub
from core.canonical.normalize_epub import (
    _EpubDraft,
    _flatten_toc,
    extract_drafts,
    resolve_sections,
)


# ── Helpers to build a synthetic EpubBook ───────────────────────────────────


def _make_book(items_html: list[tuple[str, str, str]], toc=None):
    """items_html = [(file_name, title, body_html), ...] in desired spine order.
    Returns an ebooklib.epub.EpubBook with spine set accordingly.
    """
    book = epub.EpubBook()
    book.set_identifier("test-id")
    book.set_title("Test")
    book.set_language("en")
    spine: list = []
    for fname, title, html in items_html:
        item = epub.EpubHtml(title=title, file_name=fname, lang="en")
        item.content = (
            f"<html><body>{html}</body></html>"
        )
        book.add_item(item)
        spine.append(item)
    book.spine = spine
    if toc is not None:
        book.toc = toc
    return book


# ── extract_drafts ──────────────────────────────────────────────────────────


def test_extract_drafts_walks_spine_in_order():
    book = _make_book(
        [
            ("a.xhtml", "A", "<h1>Alpha</h1><p>First paragraph.</p>"),
            ("b.xhtml", "B", "<h1>Beta</h1><p>Second paragraph.</p>"),
        ]
    )
    drafts, _epub_types, audit = extract_drafts(book)
    assert audit["spine_items"] == 2
    # spine_idx is monotonic
    assert [d.spine_idx for d in drafts] == sorted(d.spine_idx for d in drafts)
    # In-document order preserved within an item
    a_drafts = [d for d in drafts if d.spine_idx == 0]
    assert a_drafts[0].tag == "h1" and a_drafts[0].text == "Alpha"
    assert a_drafts[1].tag == "p" and a_drafts[1].text == "First paragraph."


def test_extract_drafts_picks_up_each_block_kind():
    html = (
        "<h2>Heading</h2>"
        "<p>Para.</p>"
        "<blockquote>Quoted.</blockquote>"
        "<ul><li>Item one</li><li>Item two</li></ul>"
        "<figure><img src='x'/><figcaption>Pic caption</figcaption></figure>"
    )
    book = _make_book([("only.xhtml", "Only", html)])
    drafts, _epub_types, audit = extract_drafts(book)
    kinds = {d.tag: d.text for d in drafts}
    assert kinds["h2"] == "Heading"
    assert kinds["p"] == "Para."
    assert kinds["blockquote"] == "Quoted."
    assert kinds["figcaption"] == "Pic caption"
    # Both list items captured separately
    li_texts = [d.text for d in drafts if d.tag == "li"]
    assert li_texts == ["Item one", "Item two"]
    assert audit["kinds"]["li"] == 2


def test_extract_drafts_drops_nav_subtrees_and_empty_blocks():
    html = (
        "<nav><ol><li><a href='c1.xhtml'>Chapter 1</a></li></ol></nav>"
        "<p>Real content.</p>"
        "<p></p>"
        "<p>   \n\t  </p>"
    )
    book = _make_book([("p.xhtml", "P", html)])
    drafts, _epub_types, audit = extract_drafts(book)
    assert audit["dropped_nav_subtrees"] == 1
    assert audit["dropped_empty_blocks"] == 2
    assert [d.text for d in drafts] == ["Real content."]


def test_extract_drafts_collapses_internal_whitespace():
    book = _make_book(
        [("p.xhtml", "P", "<p>Lots   of\n\twhitespace\n  here.</p>")]
    )
    drafts, _, _ = extract_drafts(book)
    assert drafts[0].text == "Lots of whitespace here."


# ── _flatten_toc ────────────────────────────────────────────────────────────


def test_flatten_toc_handles_nested_links():
    l1 = epub.Link("a.xhtml", "Ch 1", "id1")
    l11 = epub.Link("a.xhtml#s1", "Ch 1.1", "id11")
    l2 = epub.Link("b.xhtml", "Ch 2", "id2")
    toc = [(l1, [l11]), l2]
    flat = _flatten_toc(toc)
    assert flat == [("Ch 1", "a.xhtml"), ("Ch 1.1", "a.xhtml#s1"), ("Ch 2", "b.xhtml")]


# ── resolve_sections ────────────────────────────────────────────────────────


def test_resolve_sections_uses_toc_when_present():
    book = _make_book(
        [
            ("a.xhtml", "A", "<h1>Alpha heading</h1><p>aaa</p>"),
            ("b.xhtml", "B", "<h1>Beta heading</h1><p>bbb</p>"),
        ],
        toc=[epub.Link("a.xhtml", "Alpha", "id1"), epub.Link("b.xhtml", "Beta", "id2")],
    )
    drafts, _, _ = extract_drafts(book)
    sections, per_draft, audit = resolve_sections(book, drafts, book_id="bk_t")
    assert audit["section_source"] == "toc"
    assert [s.label for s in sections] == ["Alpha", "Beta"]
    assert all(s.source == "toc" for s in sections)
    # Drafts from item A go to section A, drafts from item B go to section B
    a_assigned = {per_draft[i] for i, d in enumerate(drafts) if d.item_href == "a.xhtml"}
    b_assigned = {per_draft[i] for i, d in enumerate(drafts) if d.item_href == "b.xhtml"}
    assert a_assigned == {sections[0].section_id}
    assert b_assigned == {sections[1].section_id}


def test_resolve_sections_gutenberg_style_anchored_toc():
    """Gutenberg EPUBs pack the whole book into a few files; chapters are
    distinguished only by #fragment anchors, often on <a id> elements
    INSIDE the heading. The TOC nests the chapter title (and back-matter
    advert address lines) as children of top-level entries.

    The resolver must: (1) split sections at anchor positions, not file
    starts; (2) use top-level TOC entries only; (3) fold a bare chapter
    marker's child title into its label; (4) not promote advert address
    lines into chapter sections (the Soap Manufacturer bug)."""
    html = (
        '<div id="a-title"><h1>MY BOOK</h1></div>'
        "<p>Title page text.</p>"
        '<h2><a id="a-ch1"></a>CHAPTER I.</h2>'
        '<h3 id="a-ch1t">INTRODUCTION.</h3>'
        "<p>Chapter one body.</p>"
        '<h2 id="a-ch2">CHAPTER II.</h2>'
        "<p>Chapter two body.</p>"
        '<h2 id="a-ads">PUBLISHER &amp; SON,</h2>'
        '<p id="a-addr">8 BROADWAY, LUDGATE HILL, LONDON, E.C.</p>'
        "<p>Advert text.</p>"
    )
    toc = [
        epub.Link("all.xhtml#a-title", "MY BOOK", "t"),
        (
            epub.Link("all.xhtml#a-ch1", "CHAPTER I.", "c1"),
            [epub.Link("all.xhtml#a-ch1t", "INTRODUCTION.", "c1t")],
        ),
        epub.Link("all.xhtml#a-ch2", "CHAPTER II.", "c2"),
        (
            epub.Link("all.xhtml#a-ads", "PUBLISHER & SON,", "ads"),
            [epub.Link("all.xhtml#a-addr",
                       "8 BROADWAY, LUDGATE HILL, LONDON, E.C.", "addr")],
        ),
    ]
    book = _make_book([("all.xhtml", "All", html)], toc=toc)
    drafts, _, _ = extract_drafts(book)
    sections, per_draft, audit = resolve_sections(book, drafts, book_id="bk_g")

    assert audit["section_source"] == "toc"
    # Top-level entries only; the chapter-title child is folded into the
    # parent label, the address child does not become a section.
    assert [s.label for s in sections] == [
        "MY BOOK",
        "CHAPTER I. INTRODUCTION.",
        "CHAPTER II.",
        "PUBLISHER & SON,",
    ]
    assert [(s.kind, s.printed_number) for s in sections] == [
        ("other", None),
        ("chapter", 1),
        ("chapter", 2),
        ("other", None),
    ]

    # Anchor-aware boundaries: each chapter holds its own blocks even
    # though everything lives in one spine file.
    sec_of = {d.text: per_draft[i] for i, d in enumerate(drafts)}
    ch1, ch2, ads = sections[1], sections[2], sections[3]
    assert sec_of["CHAPTER I."] == ch1.section_id
    assert sec_of["INTRODUCTION."] == ch1.section_id
    assert sec_of["Chapter one body."] == ch1.section_id
    assert sec_of["Chapter two body."] == ch2.section_id
    # The advert address line stays in the advert section.
    assert sec_of["8 BROADWAY, LUDGATE HILL, LONDON, E.C."] == ads.section_id


def test_resolve_sections_falls_back_to_h1_when_toc_empty():
    book = _make_book(
        [
            ("a.xhtml", "A", "<h1>Alpha heading</h1><p>aaa</p><h1>Beta heading</h1><p>bbb</p>"),
        ],
        toc=[],
    )
    drafts, _, _ = extract_drafts(book)
    sections, per_draft, audit = resolve_sections(book, drafts, book_id="bk_t")
    assert audit["section_source"] == "inferred"
    assert [s.label for s in sections] == ["Alpha heading", "Beta heading"]
    assert all(s.source == "inferred" for s in sections)
    # The two <p> drafts inherit their preceding <h1>'s section
    p_secs = [per_draft[i] for i, d in enumerate(drafts) if d.tag == "p"]
    assert p_secs == [sections[0].section_id, sections[1].section_id]


def test_resolve_sections_uses_h2_when_no_h1_anywhere():
    book = _make_book(
        [("a.xhtml", "A", "<h2>Only h2</h2><p>body</p>")],
        toc=[],
    )
    drafts, _, _ = extract_drafts(book)
    sections, per_draft, _ = resolve_sections(book, drafts, book_id="bk_t")
    assert [s.label for s in sections] == ["Only h2"]


def test_resolve_sections_synthetic_body_when_no_structure():
    book = _make_book([("a.xhtml", "A", "<p>One</p><p>Two</p>")], toc=[])
    drafts, _, _ = extract_drafts(book)
    sections, per_draft, audit = resolve_sections(book, drafts, book_id="bk_t")
    assert audit["section_source"] == "inferred"
    assert len(sections) == 1
    assert sections[0].label == "Body"
    assert all(s == sections[0].section_id for s in per_draft)


# ── End-to-end ──────────────────────────────────────────────────────────────


def test_normalize_end_to_end_smoke():
    book = _make_book(
        [
            ("a.xhtml", "A", "<h1>Alpha</h1><p>First paragraph.</p><p>Second.</p>"),
            ("b.xhtml", "B", "<h1>Beta</h1><blockquote>Quote.</blockquote>"),
        ],
        toc=[epub.Link("a.xhtml", "Alpha", "id1"), epub.Link("b.xhtml", "Beta", "id2")],
    )
    sections, blocks, report = nepub.normalize(book, book_id="bk_t")

    # Section assignment is final and from toc
    assert report["section_source"] == "toc"
    assert [s.label for s in sections] == ["Alpha", "Beta"]

    # Each block kind is preserved
    by_text = {b.text: b for b in blocks}
    assert by_text["Alpha"].kind == "heading"
    assert by_text["First paragraph."].kind == "paragraph"
    assert by_text["Quote."].kind == "quote"

    # Monotonic offsets, no overlap
    for prev, nxt in zip(blocks, blocks[1:]):
        assert nxt.book_offset_start > prev.book_offset_end
        assert prev.book_offset_start < prev.book_offset_end

    # Locator carries spine_idx + anchor
    assert by_text["Quote."].locator["spine_idx"] == 1
    assert "b.xhtml" in by_text["Quote."].locator["anchor"]

    # IDs deterministic when re-running
    sections2, blocks2, _ = nepub.normalize(book, book_id="bk_t")
    assert [b.block_id for b in blocks2] == [b.block_id for b in blocks]


def test_normalize_against_real_uploaded_epub():
    """Sanity check against one of the real EPUB files in uploads/, if present.
    Skips silently if the corpus changes.
    """
    from pathlib import Path

    candidates = list(Path(__file__).parent.parent.glob("uploads/*.epub"))
    if not candidates:
        pytest.skip("no real EPUB available")

    book = epub.read_epub(str(candidates[0]), options={"ignore_ncx": True})
    sections, blocks, report = nepub.normalize(book, book_id="bk_real_epub")

    assert report["final_blocks"] > 0
    assert report["spine_items"] > 0
    # Sections must exist (toc or inferred); no section_id should be None.
    assert len(sections) > 0
    assert all(b.section_id is not None for b in blocks)
    # Re-run determinism
    _, blocks2, _ = nepub.normalize(book, book_id="bk_real_epub")
    assert [b.block_id for b in blocks2] == [b.block_id for b in blocks]
