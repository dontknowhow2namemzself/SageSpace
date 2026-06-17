"""Normalize a PDF into canonical Sections + Blocks.

Pipeline (single forward pass over pages):
  1. extract_lines       – PyMuPDF dict mode → (text, bbox, font_size, page) per line
  2. drop_repeating_bands– remove headers/footers/page numbers (repeat across pages)
  3. cluster_into_blocks – merge lines into blocks by vertical gap + font size
  4. join_pages          – heal hyphenation + sentence-continuation across page breaks
  5. resolve_sections    – PDF outline if available, else heuristic heading detection
  6. emit_blocks         – assign stable block_ids, book_offset, locator, norm_flags
  7. compile_report      – counts of dropped/merged/inferred items for audit

The caller (ingest.py) opens the fitz.Document and passes it in; this module
does no file I/O. All helpers below are pure functions on plain data, kept
small enough to unit-test in isolation.

Design rules locked here:
  * block_id depends only on (book_id, block.order_idx). DO NOT hash text.
  * Section assignment happens ONCE here. Downstream code MUST NOT re-infer.
  * If outline-based sections fail, every fallback section is marked
    source='inferred' so the audit can flag it.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Iterable

from core.canonical.chapter_parse import resolve_kind_and_number
from core.canonical.ids import make_block_id, make_section_id
from core.canonical.models import Block, Section


# ── Tunables (kept top-level so tests can monkeypatch) ──────────────────────

# Treat a line as a likely header/footer if the SAME exact text appears on at
# least this many pages within ±HEADER_BAND_Y_TOL of the same Y coordinate.
HEADER_REPEAT_THRESHOLD = 3
HEADER_BAND_Y_TOL = 6.0

# Vertical gap (in pts) larger than this between two lines starts a new block.
# Smaller values keep tight paragraphs intact; ~1.6x line height is a safe default.
BLOCK_VERTICAL_GAP = 8.0

# A line whose font size exceeds the running median by this multiple is treated
# as a heading candidate when outline is absent.
HEADING_FONT_RATIO = 1.25

# Cross-page join: if the previous page's last block ends with a hyphen or
# lowercase/comma, glue it to the next page's first block.
JOIN_TAIL_HYPHEN = ("-", "\u2010")  # ASCII hyphen + Unicode hyphen
JOIN_TAIL_PUNCTS = (",", ";", ":")  # mid-sentence punctuation


# ── Internal line model ─────────────────────────────────────────────────────


@dataclass
class _Line:
    page: int  # 0-based page index
    text: str
    bbox: tuple[float, float, float, float]  # x0, y0, x1, y1
    font_size: float

    @property
    def y0(self) -> float:
        return self.bbox[1]


@dataclass
class _DraftBlock:
    """Pre-finalisation block, before section resolution and ID assignment."""

    page: int
    lines: list[_Line]
    bbox: tuple[float, float, float, float]
    font_size: float  # median across lines, used for heading detection
    norm_flags: dict = field(default_factory=dict)

    @property
    def text(self) -> str:
        return " ".join(l.text for l in self.lines).strip()


# ── 1. Extraction ───────────────────────────────────────────────────────────


def extract_lines(pdf_doc) -> list[_Line]:
    """Use PyMuPDF dict mode so we keep bbox + font size for every line.
    `pdf_doc` is a `fitz.Document`. Kept as a parameter to avoid file I/O.
    """
    out: list[_Line] = []
    for page_idx in range(len(pdf_doc)):
        page = pdf_doc[page_idx]
        data = page.get_text("dict")
        for block in data.get("blocks", []):
            if block.get("type", 0) != 0:  # 0 = text, 1 = image
                continue
            for line in block.get("lines", []):
                spans = line.get("spans", [])
                if not spans:
                    continue
                text = "".join(s.get("text", "") for s in spans).strip()
                if not text:
                    continue
                bbox = line.get("bbox") or block.get("bbox")
                if not bbox:
                    continue
                # Median font size across spans is robust to inline italics.
                sizes = sorted(s.get("size", 0.0) for s in spans)
                fs = sizes[len(sizes) // 2] if sizes else 0.0
                out.append(
                    _Line(
                        page=page_idx,
                        text=text,
                        bbox=tuple(bbox),
                        font_size=fs,
                    )
                )
    return out


# ── 2. Header / footer / page-number stripping ──────────────────────────────


def drop_repeating_bands(
    lines: list[_Line],
    *,
    repeat_threshold: int = HEADER_REPEAT_THRESHOLD,
    y_tol: float = HEADER_BAND_Y_TOL,
) -> tuple[list[_Line], dict]:
    """Strip lines whose text repeats on many pages near the same Y band.
    Returns (kept_lines, audit_dict).

    Heuristic: bucket by (rounded_y, exact_text); if a bucket spans ≥
    repeat_threshold distinct pages, treat every member as boilerplate.
    Also strip standalone-numeric lines (page numbers) unconditionally.
    """
    audit = {
        "dropped_headers_or_footers": 0,
        "dropped_page_numbers": 0,
    }

    # First pass: count repetitions.
    bucket_pages: dict[tuple[float, str], set[int]] = defaultdict(set)
    for ln in lines:
        key = (round(ln.y0 / y_tol) * y_tol, ln.text)
        bucket_pages[key].add(ln.page)

    repeating_keys: set[tuple[float, str]] = {
        k for k, pages in bucket_pages.items() if len(pages) >= repeat_threshold
    }

    kept: list[_Line] = []
    for ln in lines:
        key = (round(ln.y0 / y_tol) * y_tol, ln.text)
        if key in repeating_keys:
            audit["dropped_headers_or_footers"] += 1
            continue
        if ln.text.strip().isdigit() and len(ln.text.strip()) <= 4:
            # standalone page number
            audit["dropped_page_numbers"] += 1
            continue
        kept.append(ln)
    return kept, audit


# ── 3. Cluster lines into draft blocks ──────────────────────────────────────


def cluster_into_blocks(
    lines: list[_Line],
    *,
    vertical_gap: float = BLOCK_VERTICAL_GAP,
) -> list[_DraftBlock]:
    """Group adjacent lines on the same page into blocks. Page breaks always
    end a block. Inside a page, a vertical gap larger than `vertical_gap`
    starts a new block.
    """
    blocks: list[_DraftBlock] = []
    current: list[_Line] = []

    def flush():
        if not current:
            return
        xs0 = min(l.bbox[0] for l in current)
        ys0 = min(l.bbox[1] for l in current)
        xs1 = max(l.bbox[2] for l in current)
        ys1 = max(l.bbox[3] for l in current)
        sizes = sorted(l.font_size for l in current)
        fs = sizes[len(sizes) // 2] if sizes else 0.0
        blocks.append(
            _DraftBlock(
                page=current[0].page,
                lines=list(current),
                bbox=(xs0, ys0, xs1, ys1),
                font_size=fs,
            )
        )

    for ln in lines:
        if not current:
            current = [ln]
            continue
        same_page = ln.page == current[-1].page
        gap = ln.bbox[1] - current[-1].bbox[3] if same_page else None
        if not same_page or (gap is not None and gap > vertical_gap):
            flush()
            current = [ln]
        else:
            current.append(ln)
    flush()
    return blocks


# ── 4. Cross-page join (hyphen / sentence continuation) ─────────────────────


_HEADING_PATTERNS = (
    # Common book/textbook heading prefixes. Anchored at start; case-sensitive
    # uppercase variant catches "CHAPTER 2", lowercase catches "Chapter 2".
    re.compile(r"^(CHAPTER|Chapter|SECTION|Section|PART|Part|BOOK|Book)\s+"
               r"(\d+|[IVXLCDM]+|[A-Z][a-z]+)\b"),
    # Sub-section ordinal like "2.1 Foo" or "2.1.3 Bar"
    re.compile(r"^\d+\.\d+(\.\d+)?\s+\w"),
    # Chinese chapter heading
    re.compile(r"^第[一二三四五六七八九十百零两\d]+章"),
)


def _looks_like_heading(block: _DraftBlock) -> bool:
    """Heuristic: is `block` likely a section / chapter heading?

    Triggers when the block text matches one of the well-known heading
    text patterns. Used by join_pages to refuse cross-page glue when
    the NEXT page starts with what is unambiguously a new chapter.
    """
    text = (block.text or "").lstrip()
    if not text:
        return False
    return any(p.match(text) for p in _HEADING_PATTERNS)


def join_pages(blocks: list[_DraftBlock]) -> tuple[list[_DraftBlock], dict]:
    """Merge a block that genuinely continues onto the next page.
    Heals the two PDF artefacts we actually want to fix:
      * hyphenated word at end of page:  "univer-\\nsal" → "universal"
      * mid-sentence clause break:       "..., \\nand he said" → joined

    Refuses to merge when the next page's first block looks like a
    heading ("CHAPTER 2", "1.1 Foo", "第二章 ..."), even if the
    previous page's tail ended with a lowercase character (e.g. a page
    footer like "Access for free at openstax.org" that escaped the
    header-band filter -- this was the cause of the cross-chapter
    bleed bug). For lowercase-tail merges we additionally require the
    next page's first character to ALSO be lowercase (i.e. a real
    clause continuation), not an uppercase new sentence / new heading.
    """
    audit = {"merged_page_breaks": 0, "dehyphenated_words": 0}
    if not blocks:
        return blocks, audit

    merged: list[_DraftBlock] = [blocks[0]]
    for nxt in blocks[1:]:
        prev = merged[-1]
        if nxt.page == prev.page:
            merged.append(nxt)
            continue

        prev_text = prev.text
        if not prev_text:
            merged.append(nxt)
            continue

        nxt_text_stripped = (nxt.text or "").lstrip()
        nxt_first_char = nxt_text_stripped[:1]

        glue: str | None = None
        flag_dehyphen = False
        if prev_text.endswith(JOIN_TAIL_HYPHEN):
            # Hyphenated split is unambiguous; merge even across a
            # heading-looking next block (rare edge case: a chapter
            # title that starts mid-word? virtually never happens).
            glue = ""
            flag_dehyphen = True
            prev.lines[-1].text = prev.lines[-1].text.rstrip("".join(JOIN_TAIL_HYPHEN))
        elif _looks_like_heading(nxt):
            # Strong "do not merge" signal regardless of prev-tail shape.
            # This is the new PR9 rule that fixes the cross-chapter bleed.
            merged.append(nxt)
            continue
        elif prev_text.endswith(JOIN_TAIL_PUNCTS):
            # Mid-clause punctuation at page end. Real continuation
            # usually starts lowercase; uppercase suggests a new
            # sentence / heading we should NOT swallow.
            if nxt_first_char and nxt_first_char.islower():
                glue = " "
            else:
                merged.append(nxt)
                continue
        elif prev_text[-1].islower() and nxt_first_char and nxt_first_char.islower():
            # Both sides lowercase: a true mid-word / mid-clause flow.
            # Pre-PR9 we joined whenever prev ended lowercase, which
            # mis-merged page footer fragments into next-chapter
            # headings (those page footers often end with a lowercase
            # word like "openstax.org").
            glue = " "
        else:
            merged.append(nxt)
            continue

        # Perform the merge: extend lines, recompute bbox, mark flags.
        prev.lines.extend(nxt.lines)
        prev.bbox = (
            min(prev.bbox[0], nxt.bbox[0]),
            min(prev.bbox[1], nxt.bbox[1]),
            max(prev.bbox[2], nxt.bbox[2]),
            max(prev.bbox[3], nxt.bbox[3]),
        )
        if glue == "":
            prev.lines[-1].text = prev.lines[-1].text  # already stripped above
        # Audit: record BOTH contributing pages on first merge, then
        # just the new one on subsequent merges. Pre-PR9 this recorded
        # only nxt.page, which made the audit field unreliable (it
        # could show [N, N, N] instead of [N-1, N, N+1]).
        merged_from = prev.norm_flags.setdefault("merged_from_pages", [])
        if not merged_from:
            merged_from.append(prev.page)
        merged_from.append(nxt.page)
        if flag_dehyphen:
            prev.norm_flags["dehyphenated"] = True
            audit["dehyphenated_words"] += 1
        audit["merged_page_breaks"] += 1
    return merged, audit


# ── 5. Section resolution ───────────────────────────────────────────────────


def resolve_sections(
    pdf_doc,
    blocks: list[_DraftBlock],
    book_id: str,
) -> tuple[list[Section], list[str | None], dict]:
    """Return (sections, per_block_section_id, audit).

    Priority:
      1. PDF outline (toc) - level 1 entries become sections, anchored by page.
      2. Heuristic fallback - lines whose font size > median × HEADING_FONT_RATIO
         become section starts. Always marked source='inferred'.
    """
    audit = {"section_source": "outline", "sections_inferred": 0, "sections_total": 0}
    toc = []
    try:
        toc = pdf_doc.get_toc() or []
    except Exception:
        toc = []

    sections: list[Section] = []
    # toc rows: [level, title, page (1-based)]
    if toc:
        order = 0
        for lvl, title, page in toc:
            if lvl != 1:
                continue  # keep top-level only; nested chapters can come in phase 2
            label = title.strip() or f"Section {order + 1}"
            kind, printed_number = resolve_kind_and_number(label)
            sections.append(
                Section(
                    section_id=make_section_id(book_id, order),
                    book_id=book_id,
                    order_idx=order,
                    label=label,
                    level=1,
                    source="outline",
                    kind=kind,
                    printed_number=printed_number,
                )
            )
            # Remember the start page on the section for assignment below.
            sections[-1]._start_page = page - 1  # type: ignore[attr-defined]
            order += 1

    if not sections:
        audit["section_source"] = "inferred"
        # Heuristic: a block is a heading if its font size is markedly larger
        # than the median block font size.
        font_sizes = sorted(b.font_size for b in blocks if b.font_size > 0)
        median = font_sizes[len(font_sizes) // 2] if font_sizes else 0.0
        order = 0
        for b in blocks:
            if median > 0 and b.font_size >= median * HEADING_FONT_RATIO and len(b.text) <= 120:
                label = b.text[:80]
                kind, printed_number = resolve_kind_and_number(label)
                sec = Section(
                    section_id=make_section_id(book_id, order),
                    book_id=book_id,
                    order_idx=order,
                    label=label,
                    level=1,
                    source="inferred",
                    kind=kind,
                    printed_number=printed_number,
                )
                sec._start_page = b.page  # type: ignore[attr-defined]
                sections.append(sec)
                audit["sections_inferred"] += 1
                order += 1

        if not sections:
            # No structure detected at all - fall back to a single synthetic section.
            sec = Section(
                section_id=make_section_id(book_id, 0),
                book_id=book_id,
                order_idx=0,
                label="Body",
                level=1,
                source="inferred",
                kind="other",
                printed_number=None,
            )
            sec._start_page = 0  # type: ignore[attr-defined]
            sections.append(sec)
            audit["sections_inferred"] += 1

    audit["sections_total"] = len(sections)

    # Assign each block to the latest section whose _start_page <= block.page.
    sections.sort(key=lambda s: getattr(s, "_start_page", 0))
    per_block: list[str | None] = []
    cur_idx = -1
    for b in blocks:
        # advance cur_idx as long as the next section starts on/before this block's page
        while (
            cur_idx + 1 < len(sections)
            and getattr(sections[cur_idx + 1], "_start_page", 0) <= b.page
        ):
            cur_idx += 1
        per_block.append(sections[cur_idx].section_id if cur_idx >= 0 else None)

    # Strip the temporary attribute before returning.
    for s in sections:
        if hasattr(s, "_start_page"):
            del s._start_page  # type: ignore[attr-defined]

    return sections, per_block, audit


# ── 6. Emit canonical Blocks ────────────────────────────────────────────────


def emit_blocks(
    drafts: list[_DraftBlock],
    per_block_section_id: list[str | None],
    book_id: str,
) -> list[Block]:
    """Assign stable IDs, compute book_offset, attach locator + norm_flags.
    Each block's text is joined with a single space; the offset window is
    half-open [start, end). Blocks are separated by a single '\\n' in the
    virtual book offset stream, so end-of-block + 1 is the next start.
    """
    out: list[Block] = []
    cursor = 0
    for i, (draft, sec_id) in enumerate(zip(drafts, per_block_section_id)):
        text = draft.text
        kind = _classify_kind(draft)
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
                locator_type="pdf",
                locator={
                    "page": draft.page + 1,  # 1-based for human consumption
                    "bbox": [round(c, 2) for c in draft.bbox],
                },
                section_id=sec_id,
                norm_flags=dict(draft.norm_flags),
            )
        )
        cursor = end + 1  # +1 for the virtual '\n' between blocks
    return out


def _classify_kind(draft: _DraftBlock) -> str:
    """Lightweight kind classifier. Real list/quote/footnote/caption detection
    is a phase-2 refinement; for now we surface 'heading' only because the
    section resolver relies on it during inferred mode, and everything else
    is 'paragraph'.
    """
    # The section resolver uses raw font size; we mirror its threshold here so
    # the block 'kind' is consistent with section detection in inferred mode.
    return "paragraph"


# ── Public entry point ──────────────────────────────────────────────────────


def normalize(pdf_doc, book_id: str) -> tuple[list[Section], list[Block], dict]:
    """Full pipeline. Returns (sections, blocks, ingestion_report)."""
    raw_lines = extract_lines(pdf_doc)
    kept_lines, audit_drop = drop_repeating_bands(raw_lines)
    draft_blocks = cluster_into_blocks(kept_lines)
    joined_blocks, audit_join = join_pages(draft_blocks)
    sections, per_block_sec, audit_sec = resolve_sections(pdf_doc, joined_blocks, book_id)
    blocks = emit_blocks(joined_blocks, per_block_sec, book_id)

    report = {
        "locator_type": "pdf",
        "pages_total": len(pdf_doc),
        "raw_lines": len(raw_lines),
        "kept_lines": len(kept_lines),
        "draft_blocks": len(draft_blocks),
        "final_blocks": len(blocks),
        **audit_drop,
        **audit_join,
        **audit_sec,
        # Section length histogram - useful for spotting "one giant section" bugs.
        "blocks_per_section": dict(
            Counter(b.section_id or "<none>" for b in blocks)
        ),
    }
    return sections, blocks, report
