"""Normalize an EPUB into canonical Sections + Blocks.

Pipeline (single pass in spine order):
  1. walk_spine         – iterate items in book.spine order (NOT items_of_type)
  2. extract_dom_blocks – per item: p/h1-h6/li/blockquote/figcaption → drafts
                          drop nav/toc/role=doc-pagebreak elements
  3. resolve_sections   – book.toc first (locked); else fall back to <h1>/<h2>
  4. emit_blocks        – stable block_ids, monotonic book_offset, locator
  5. compile_report     – audit counts (dropped_nav, kinds histogram, ...)

Design rules (mirrors normalize_pdf to keep the system uniform):
  * block_id depends only on (book_id, order_idx). Never hash text or href.
  * Section assignment is final here; downstream code MUST NOT re-infer.
  * Inferred sections are explicitly marked source='inferred' for audit.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Literal

from core.canonical.chapter_parse import resolve_kind_and_number
from core.canonical.ids import make_block_id, make_section_id
from core.canonical.models import Block, Section


HEADING_TAGS = ("h1", "h2", "h3", "h4", "h5", "h6")
PARA_LIKE_TAGS = ("p", "blockquote")
LIST_ITEM_TAG = "li"
CAPTION_TAG = "figcaption"

# Tags whose entire subtree must be skipped (TOC navigation, page-break anchors).
SKIP_TAGS = ("nav",)

# EPUB 3 Structural Semantics Vocabulary tokens we recognize on <body> /
# <section> epub:type attributes. The mapping is one-way (EPUB token to
# our internal kind enum). Unknown tokens fall through to the label-text
# heuristic in classify_section_kind. The full SSV list lives at
# https://idpf.github.io/epub-vocabs/structure/ -- we only map the ones
# that round-trip cleanly to our enum.
_EPUB_TYPE_TO_KIND = {
    # front-matter
    "cover": "cover",
    "halftitlepage": "titlepage",
    "titlepage": "titlepage",
    "copyright-page": "front_matter",
    "toc": "toc",
    "landmarks": "front_matter",
    "loi": "front_matter",
    "lot": "front_matter",
    "preamble": "front_matter",
    "dedication": "front_matter",
    "preface": "preface",
    "foreword": "foreword",
    "introduction": "introduction",
    "acknowledgments": "front_matter",
    "frontmatter": "front_matter",
    # body-matter
    "prologue": "prologue",
    "chapter": "chapter",
    "part": "chapter",          # parts wrap chapters; treat as chapter-like
    "subchapter": "chapter",
    "epilogue": "epilogue",
    "bodymatter": "chapter",    # generic body-matter falls into chapter bucket
    # back-matter
    "afterword": "afterword",
    "appendix": "appendix",
    "glossary": "glossary",
    "index": "index",
    "bibliography": "bibliography",
    "colophon": "back_matter",
    "backmatter": "back_matter",
}

# Block-level kinds we want to emit. Order matters: more specific first.
_KIND_BY_TAG = {
    **{h: "heading" for h in HEADING_TAGS},
    "p": "paragraph",
    "blockquote": "quote",
    LIST_ITEM_TAG: "list_item",
    CAPTION_TAG: "caption",
}


# ── Internal draft model ────────────────────────────────────────────────────


@dataclass
class _EpubDraft:
    spine_idx: int          # 0-based position in book.spine
    item_href: str          # for locator + debugging
    tag: str
    text: str
    heading_level: int | None  # 1..6 for headings, None otherwise
    # HTML element ids attached to this block: its own id, ids of its
    # descendants (Gutenberg nests `<a id=...>` anchors inside headings),
    # and ids seen since the previous block (wrapper divs, empty anchors).
    # Used to resolve TOC hrefs with #fragments to exact block positions.
    anchor_ids: tuple[str, ...] = ()
    norm_flags: dict = field(default_factory=dict)


# ── 1+2. Walk spine, extract DOM blocks ─────────────────────────────────────


def extract_drafts(book) -> tuple[list[_EpubDraft], dict[str, str], dict]:
    """Walk the EPUB in spine order; emit one _EpubDraft per block-level
    element with non-trivial text. `book` is an ebooklib.epub.EpubBook.

    Returns (drafts, epub_type_by_href, audit). `epub_type_by_href`
    maps each item href to the EPUB 3 epub:type value found on its
    root <body> or first <section> element, when present -- this is
    the high-signal source for section kind classification (preferred
    over label heuristics).
    """
    import ebooklib
    from bs4 import BeautifulSoup

    audit = {
        "spine_items": 0,
        "dropped_nav_subtrees": 0,
        "dropped_empty_blocks": 0,
        "epub_types_seen": Counter(),
        "kinds": Counter(),
    }
    drafts: list[_EpubDraft] = []
    epub_type_by_href: dict[str, str] = {}

    # ebooklib's book.spine can hold any of:
    #   - (idref, linear) tuples              (canonical OPF representation)
    #   - bare idref strings                  (some readers)
    #   - EpubHtml objects directly           (programmatic construction in tests)
    # Resolve each entry to a concrete item so downstream code only sees items.
    spine = list(book.spine or [])
    audit["spine_items"] = len(spine)

    for spine_idx, entry in enumerate(spine):
        item = None
        if hasattr(entry, "get_type"):
            # Already an EpubItem-like object.
            item = entry
        else:
            idref = entry[0] if isinstance(entry, (list, tuple)) else entry
            item = book.get_item_with_id(idref)
        if item is None or item.get_type() != ebooklib.ITEM_DOCUMENT:
            continue

        soup = BeautifulSoup(item.get_content(), "html.parser")

        # Capture epub:type if the item carries one. We check <body> first,
        # then the first <section>. html.parser preserves "epub:type" as
        # a literal attribute name (no namespace stripping).
        item_epub_type = _read_epub_type(soup)
        if item_epub_type:
            epub_type_by_href[item.get_name()] = item_epub_type
            audit["epub_types_seen"][item_epub_type] += 1

        # Remove navigational subtrees outright (they are not content).
        for tag_name in SKIP_TAGS:
            for n in soup.find_all(tag_name):
                n.decompose()
                audit["dropped_nav_subtrees"] += 1

        # Iterate ALL elements in document order, tracking anchor ids so
        # TOC hrefs with #fragments resolve to exact block positions.
        # (Gutenberg-style EPUBs pack the whole book into a handful of
        # files and rely on anchors for chapter boundaries; matching by
        # file alone collapses every chapter in a file onto one section.)
        pending_ids: list[str] = []
        consumed_ids: set[str] = set()
        for el in soup.find_all(True):
            el_id = el.get("id")
            if el_id and el_id not in consumed_ids:
                pending_ids.append(el_id)
                consumed_ids.add(el_id)
            tag = el.name
            if tag not in _KIND_BY_TAG:
                continue
            text = _clean_text(el.get_text(separator=" "))
            if not text:
                audit["dropped_empty_blocks"] += 1
                continue
            # Descendant ids belong to this block too — anchors often sit
            # INSIDE the heading they mark (<h2><a id="..."/>CHAPTER I.</h2>),
            # and the pre-order walk would otherwise see them only after
            # the block was emitted, shifting the boundary one block late.
            for sub in el.find_all(True):
                sub_id = sub.get("id")
                if sub_id and sub_id not in consumed_ids:
                    pending_ids.append(sub_id)
                    consumed_ids.add(sub_id)
            heading_level = int(tag[1]) if tag in HEADING_TAGS else None
            drafts.append(
                _EpubDraft(
                    spine_idx=spine_idx,
                    item_href=item.get_name(),
                    tag=tag,
                    text=text,
                    heading_level=heading_level,
                    anchor_ids=tuple(pending_ids),
                )
            )
            pending_ids = []
            audit["kinds"][tag] += 1

    # Counter is JSON-friendly as plain dict
    audit["kinds"] = dict(audit["kinds"])
    audit["epub_types_seen"] = dict(audit["epub_types_seen"])
    return drafts, epub_type_by_href, audit


def _read_epub_type(soup) -> str | None:
    """Find the first epub:type attribute on body / section. Returns the
    raw attribute value (may be space-separated tokens) or None.

    BeautifulSoup with html.parser keeps the prefixed attribute name
    intact, but some files use just `type` (no namespace) which would
    collide with input[type] etc. We only honor `type` when found on
    body or section (where input[type] does not apply).
    """
    for tag_name in ("body", "section"):
        el = soup.find(tag_name)
        if el is None:
            continue
        for attr in ("epub:type", "type"):
            value = el.get(attr)
            if value:
                return value.strip()
    return None


def _clean_text(s: str) -> str:
    # Collapse internal whitespace; EPUB DOM often has stray newlines/tabs.
    return " ".join(s.split())


# ── 3. Section resolution ───────────────────────────────────────────────────


def resolve_sections(
    book,
    drafts: list[_EpubDraft],
    book_id: str,
    epub_type_by_href: dict[str, str] | None = None,
) -> tuple[list[Section], list[str | None], dict]:
    """Return (sections, per_draft_section_id, audit).

    Strategy:
      1. book.toc (locked, source='toc'). Top-level entries only (children
         are sub-headings within their parent section). Each entry is
         resolved to a draft position by (file, #fragment-anchor) when the
         href carries a fragment, falling back to the first draft of the
         file. Drafts are then assigned to the latest section whose start
         precedes them.
      2. Heuristic fallback: every heading-level-1 (or top-most heading) draft
         starts a new section; everything else continues current. Marked
         source='inferred'.
      3. Synthetic 'Body' if neither produces a section.

    Each created Section is also tagged with kind + printed_number
    using EPUB epub:type as the high-signal source (via
    `epub_type_by_href` keyed by the section's first draft item_href)
    and falling back to the label-text heuristic.
    """
    epub_type_by_href = epub_type_by_href or {}
    audit = {"section_source": "toc", "sections_inferred": 0, "sections_total": 0}

    toc_entries = _toc_entries_for_sections(getattr(book, "toc", None) or [])
    sections: list[Section] = []

    if toc_entries:
        starts_at_draft: list[int] = []  # index into `drafts` where each section begins
        order = 0
        for title, href in toc_entries:
            href_file, _, fragment = href.partition("#")
            # Anchor-aware: locate the exact block the fragment points at.
            first_match = None
            if fragment:
                first_match = next(
                    (i for i, d in enumerate(drafts)
                     if d.item_href.endswith(href_file) and fragment in d.anchor_ids),
                    None,
                )
            if first_match is None:
                first_match = next(
                    (i for i, d in enumerate(drafts) if d.item_href.endswith(href_file)),
                    None,
                )
            if first_match is None:
                continue
            label = title.strip() or f"Section {order + 1}"
            section_item_href = drafts[first_match].item_href
            override = _kind_override_from_epub_type(
                epub_type_by_href.get(section_item_href)
            )
            kind, printed_number = resolve_kind_and_number(label, override)
            sections.append(
                Section(
                    section_id=make_section_id(book_id, order),
                    book_id=book_id,
                    order_idx=order,
                    label=label,
                    level=1,
                    source="toc",
                    kind=kind,
                    printed_number=printed_number,
                )
            )
            starts_at_draft.append(first_match)
            order += 1

        per_draft = _assign_by_draft_index(len(drafts), sections, starts_at_draft)
        if sections:
            audit["sections_total"] = len(sections)
            return sections, per_draft, audit

    # Heuristic fallback
    audit["section_source"] = "inferred"
    starts_at_draft = []
    order = 0
    for i, d in enumerate(drafts):
        # Treat the first h1 we see (or h2 if no h1 exists) as a section start.
        if d.heading_level == 1 or (d.heading_level == 2 and not _has_any_h1(drafts)):
            label = d.text[:80]
            override = _kind_override_from_epub_type(
                epub_type_by_href.get(d.item_href)
            )
            kind, printed_number = resolve_kind_and_number(label, override)
            sections.append(
                Section(
                    section_id=make_section_id(book_id, order),
                    book_id=book_id,
                    order_idx=order,
                    label=label,
                    level=1,
                    source="inferred",
                    kind=kind,
                    printed_number=printed_number,
                )
            )
            starts_at_draft.append(i)
            audit["sections_inferred"] += 1
            order += 1

    if not sections:
        sections.append(
            Section(
                section_id=make_section_id(book_id, 0),
                book_id=book_id,
                order_idx=0,
                label="Body",
                level=1,
                source="inferred",
                kind="other",
                printed_number=None,
            )
        )
        starts_at_draft.append(0)
        audit["sections_inferred"] += 1

    per_draft = _assign_by_draft_index(len(drafts), sections, starts_at_draft)
    audit["sections_total"] = len(sections)
    return sections, per_draft, audit


def _toc_entries_for_sections(toc) -> list[tuple[str, str]]:
    """Pick the TOC entries that become Sections.

    Top-level entries only, matching the PDF pipeline (top-level outline
    entries). Nested children are sub-headings *within* a chapter:
    flattening them fragments chapters into empty shells and promotes
    junk (publisher-advert address lines in Gutenberg back matter) into
    sections of their own.

    Gutenberg books often list a bare chapter marker ("CHAPTER I.") as
    the parent with the chapter title as its first child; fold that
    child title into the parent label so the section reads
    "CHAPTER I. INTRODUCTION.".

    Falls back to the flattened list when there are fewer than 3
    top-level entries (some books nest everything under a single root).
    """
    top: list[tuple[str, str]] = []
    for entry in toc:
        if isinstance(entry, tuple) and len(entry) == 2 and isinstance(entry[1], list):
            link, children = entry
            href = getattr(link, "href", None)
            title = getattr(link, "title", None)
            if not (href and title):
                continue
            label = title.strip()
            if _is_bare_chapter_marker(label):
                child_titles = _flatten_toc(children)
                if child_titles:
                    label = f"{label} {child_titles[0][0].strip()}"
            top.append((label, href))
        else:
            href = getattr(entry, "href", None)
            title = getattr(entry, "title", None)
            if href and title:
                top.append((title.strip(), href))
    if len(top) >= 3:
        return top
    return _flatten_toc(toc)


_BARE_CHAPTER_MARKER_RE = re.compile(
    r"^\s*(?:CHAPTER|PART|BOOK)\s+(?:\d+|[IVXLCDM]+)\s*[.:]?\s*$",
    re.IGNORECASE,
)


def _is_bare_chapter_marker(label: str) -> bool:
    """True for labels that are ONLY a chapter marker ("CHAPTER VIII.",
    "Part 2") with no title text after the number."""
    return bool(_BARE_CHAPTER_MARKER_RE.match(label))


def _flatten_toc(toc) -> list[tuple[str, str]]:
    """Flatten ebooklib's nested toc tree into [(title, href), ...] in order.

    ebooklib returns toc entries as Link or as nested tuples (Link, [children]).
    """
    out: list[tuple[str, str]] = []
    for entry in toc:
        if isinstance(entry, tuple) and len(entry) == 2 and isinstance(entry[1], list):
            link, children = entry
            href = getattr(link, "href", None)
            title = getattr(link, "title", None)
            if href and title:
                out.append((title, href))
            out.extend(_flatten_toc(children))
        else:
            href = getattr(entry, "href", None)
            title = getattr(entry, "title", None)
            if href and title:
                out.append((title, href))
    return out


def _assign_by_draft_index(
    n_drafts: int, sections: list[Section], starts_at_draft: list[int]
) -> list[str | None]:
    """Given the index in `drafts` where each section begins, label every
    draft with the latest section whose start ≤ its index. Drafts before
    the first start (e.g. Gutenberg licence boilerplate ahead of the
    first TOC anchor) are clamped into the first section — every block
    must carry a section_id (module invariant).
    """
    out: list[str | None] = []
    cur = -1
    for i in range(n_drafts):
        while cur + 1 < len(starts_at_draft) and starts_at_draft[cur + 1] <= i:
            cur += 1
        if cur >= 0:
            out.append(sections[cur].section_id)
        else:
            out.append(sections[0].section_id if sections else None)
    return out


def _has_any_h1(drafts: list[_EpubDraft]) -> bool:
    return any(d.heading_level == 1 for d in drafts)


def _kind_override_from_epub_type(value: str | None) -> str | None:
    """Map a raw epub:type attribute string (possibly space-separated
    tokens) to our internal kind enum. Returns the first recognized
    mapping, or None if nothing matched (caller falls back to label
    heuristic).
    """
    if not value:
        return None
    for token in value.split():
        kind = _EPUB_TYPE_TO_KIND.get(token.strip().lower())
        if kind:
            return kind
    return None


# ── 4. Emit canonical Blocks ────────────────────────────────────────────────


def emit_blocks(
    drafts: list[_EpubDraft],
    per_draft_section_id: list[str | None],
    book_id: str,
) -> list[Block]:
    out: list[Block] = []
    cursor = 0
    for i, (draft, sec_id) in enumerate(zip(drafts, per_draft_section_id)):
        text = draft.text
        kind = _KIND_BY_TAG.get(draft.tag, "paragraph")
        start = cursor
        end = cursor + len(text)
        out.append(
            Block(
                block_id=make_block_id(book_id, i),
                book_id=book_id,
                order_idx=i,
                kind=kind,
                text=text,
                book_offset_start=start,
                book_offset_end=end,
                locator_type="epub",
                locator={
                    "spine_idx": draft.spine_idx,
                    "href": draft.item_href,
                    "tag": draft.tag,
                    # Simplified CFI surrogate; full CFI generation is phase-2.
                    "anchor": f"{draft.item_href}#{draft.tag}[{i}]",
                },
                section_id=sec_id,
                norm_flags=dict(draft.norm_flags),
            )
        )
        cursor = end + 1
    return out


# ── Public entry point ──────────────────────────────────────────────────────


def normalize(book, book_id: str) -> tuple[list[Section], list[Block], dict]:
    """Full pipeline. Returns (sections, blocks, ingestion_report)."""
    drafts, epub_type_by_href, audit_extract = extract_drafts(book)
    sections, per_draft_sec, audit_sec = resolve_sections(
        book, drafts, book_id, epub_type_by_href=epub_type_by_href,
    )
    blocks = emit_blocks(drafts, per_draft_sec, book_id)

    report = {
        "locator_type": "epub",
        "draft_blocks": len(drafts),
        "final_blocks": len(blocks),
        **audit_extract,
        **audit_sec,
        "blocks_per_section": dict(
            Counter(b.section_id or "<none>" for b in blocks)
        ),
    }
    return sections, blocks, report
