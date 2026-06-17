"""Canonical text dataclasses.

These are the in-memory representation produced by normalizers and consumed
by the DB layer and downstream chunker. SQLite is the system of record; this
module deliberately stays plain dataclasses (no ORM) to keep the boundary
narrow and easy to test.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Literal


BlockKind = Literal[
    "paragraph",
    "heading",
    "list_item",
    "quote",
    "footnote",
    "caption",
    "figure",
]

LocatorType = Literal["pdf", "epub"]

SectionSource = Literal["outline", "toc", "inferred"]

# Structural role of a section. Body-matter chapters (kind="chapter")
# are the only sections users mean when they say "Chapter N"; the rest
# are addressable individually (e.g. "the prologue", "the appendix")
# but should NOT consume Chapter-N ordinal slots.
SectionKind = Literal[
    # front-matter
    "cover", "titlepage", "toc", "preface", "foreword", "introduction",
    "front_matter",   # catch-all for unclassified front-matter
    # body-matter
    "prologue",
    "chapter",
    "epilogue",
    # back-matter
    "afterword", "appendix", "glossary", "index", "bibliography",
    "back_matter",    # catch-all
    # unclassified
    "other",
]


@dataclass
class Section:
    section_id: str
    book_id: str
    order_idx: int
    label: str
    level: int = 1
    parent_section_id: str | None = None
    source: SectionSource = "inferred"
    # Structural role. Populated by the normalizer using EPUB epub:type
    # when available (high signal) or classify_section_kind(label) as
    # fallback. Defaults to "other" so unmigrated rows behave sanely.
    kind: SectionKind = "other"
    # The author-printed chapter number, if any. 5 for "CHAPTER V" /
    # "第五章". None for non-chapter sections.
    printed_number: int | None = None


@dataclass
class Block:
    block_id: str
    book_id: str
    order_idx: int
    kind: BlockKind
    text: str
    book_offset_start: int
    book_offset_end: int
    locator_type: LocatorType
    # Free-form per-format locator. PDF: {page, bbox}. EPUB: {spine_idx, cfi, print_page}.
    locator: dict = field(default_factory=dict)
    section_id: str | None = None
    # Audit flags set by the normalizer: dehyphenated, merged_from, dropped_neighbors, etc.
    norm_flags: dict = field(default_factory=dict)

    @property
    def locator_json(self) -> str:
        return json.dumps(self.locator, ensure_ascii=False, sort_keys=True)

    @property
    def norm_flags_json(self) -> str:
        return json.dumps(self.norm_flags, ensure_ascii=False, sort_keys=True)
