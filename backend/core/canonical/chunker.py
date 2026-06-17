"""Derive retrieval chunks from canonical Blocks.

Chunks are NOT a source of truth here; they are a derived projection of the
canonical block stream. Every chunk carries back-references to the blocks
it covers, so any retrieval hit can be resolved to a precise span in the
canonical text layer (see docs/ARCHITECTURE.md §canonical-refactor).

Rules locked here:
  * Splitter sees ONE section at a time. We never let a chunk cross
    sections - that would resurrect the "wrong chapter" pathology of the
    legacy parser.
  * Each chunk records:
      - chunk_id              stable, derived from book_id + global order
      - block_ids             comma-separated list of every covered block
      - primary_block_id      the block containing the chunk's text midpoint;
                              this is what citation jumps will scroll to
      - first/last_char_offset character offsets inside first/last block
      - section_id, chapter (legacy mirror), page (legacy mirror, PDF only)
  * The output is a list of langchain Document objects so it slots into
    the existing raptor.build_raptor_index() without changes.

Chroma metadata values must be primitives (str/int/float/bool). Lists are
encoded as comma-separated strings; see core/canonical/chroma_meta.py.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 100

# Single character used to join blocks inside a section before splitting. The
# normalizer reserves one virtual character between blocks in book_offset, so
# using "\n" here keeps offsets cheap to reason about.
BLOCK_JOIN = "\n"


# ── Helpers to encode/decode list metadata for Chroma (no list support) ─────


def encode_block_ids(block_ids: list[str]) -> str:
    return ",".join(block_ids)


def decode_block_ids(s: str | None) -> list[str]:
    if not s:
        return []
    return [bid for bid in s.split(",") if bid]


# ── Stable chunk_id ─────────────────────────────────────────────────────────


def _chunk_id(book_id: str, order_idx: int) -> str:
    h = hashlib.sha1(f"{book_id}|chunk|{order_idx}".encode("utf-8")).hexdigest()[:8]
    return f"chk_{h}"


# ── Internal: build a [start, end) char range map for blocks in a section ──


@dataclass
class _SectionWindow:
    """One section's blocks projected into a contiguous string + a mapping
    from char offset → block (for reverse-lookup during chunk assembly).
    """

    section_id: str | None
    text: str
    # Aligned with text. block_starts[i] = char offset in `text` where blocks[i] begins.
    blocks: list[dict]
    block_starts: list[int]
    block_ends: list[int]  # exclusive (so block i covers [start, end))


def _build_section_window(section_id: str | None, blocks: list[dict]) -> _SectionWindow:
    parts: list[str] = []
    starts: list[int] = []
    ends: list[int] = []
    cursor = 0
    for i, b in enumerate(blocks):
        if i > 0:
            cursor += len(BLOCK_JOIN)
        starts.append(cursor)
        parts.append(b["text"])
        cursor += len(b["text"])
        ends.append(cursor)
    return _SectionWindow(
        section_id=section_id,
        text=BLOCK_JOIN.join(b["text"] for b in blocks),
        blocks=blocks,
        block_starts=starts,
        block_ends=ends,
    )


# ── Block-span resolution for a given char range inside a section ──────────


def _resolve_span(window: _SectionWindow, start: int, end: int) -> dict:
    """Given a [start, end) char range within `window.text`, return:
        first_idx, last_idx,           indices into window.blocks
        first_char_offset, last_char_offset
        block_ids (in order), primary_block_id
    `primary` = the block containing the midpoint of [start, end).
    Always covers at least one block (handles tiny chunks safely).
    """
    n = len(window.blocks)
    if n == 0:
        raise ValueError("section window has no blocks")

    # bisect manually to keep the dep surface small
    def _find_idx(pos: int) -> int:
        lo, hi = 0, n - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if window.block_starts[mid] <= pos:
                lo = mid
            else:
                hi = mid - 1
        return lo

    first_idx = _find_idx(start)
    # end is exclusive in the char range; locate the block holding end-1
    last_pos = max(start, end - 1)
    last_idx = _find_idx(last_pos)

    first_char_offset = max(0, start - window.block_starts[first_idx])
    last_char_offset = max(
        0, min(window.block_ends[last_idx], end) - window.block_starts[last_idx]
    )

    mid_pos = (start + end) // 2
    primary_idx = _find_idx(mid_pos)

    block_ids = [window.blocks[i]["block_id"] for i in range(first_idx, last_idx + 1)]
    return {
        "first_idx": first_idx,
        "last_idx": last_idx,
        "first_char_offset": first_char_offset,
        "last_char_offset": last_char_offset,
        "block_ids": block_ids,
        "primary_block_id": window.blocks[primary_idx]["block_id"],
    }


# ── Public entry point ──────────────────────────────────────────────────────


def chunk_blocks(
    book_id: str,
    blocks: Iterable[dict],
    *,
    section_label_by_id: dict[str, str] | None = None,
    section_order_by_id: dict[str, int] | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Document]:
    """Produce retrieval chunks (LangChain Documents) from a book's canonical
    blocks. Blocks must be in canonical order_idx order.

    Optional `section_label_by_id` / `section_order_by_id` enrich the legacy
    `chapter` / `chapter_label` mirror fields; pass them in if available.
    """
    section_label_by_id = section_label_by_id or {}
    section_order_by_id = section_order_by_id or {}
    blocks_list = list(blocks)

    # Group consecutive blocks by section_id, preserving order.
    sections: list[tuple[str | None, list[dict]]] = []
    for b in blocks_list:
        sid = b.get("section_id")
        if sections and sections[-1][0] == sid:
            sections[-1][1].append(b)
        else:
            sections.append((sid, [b]))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        # Keep the same separator preferences as the legacy parser so chunk
        # boundaries do not regress against retrieval quality.
        separators=["\n\n", "\n", "。", ".", " "],
    )

    documents: list[Document] = []
    global_order = 0

    for section_id, sec_blocks in sections:
        window = _build_section_window(section_id, sec_blocks)
        if not window.text.strip():
            continue

        # split_text returns plain strings. We need each piece's offset inside
        # window.text. RecursiveCharacterTextSplitter does not surface offsets,
        # so we re-scan: each piece is a contiguous substring (with overlap),
        # so str.find from a running cursor recovers the offset reliably.
        pieces = splitter.split_text(window.text)
        scan_cursor = 0
        for piece in pieces:
            if not piece:
                continue
            # Search forward from cursor - back off by overlap if not found
            # (handles separators consumed by the splitter).
            idx = window.text.find(piece, scan_cursor)
            if idx < 0:
                # Best-effort: search the whole string. Should be rare.
                idx = window.text.find(piece)
                if idx < 0:
                    # Skip pieces we can't anchor; safer than emitting wrong block_ids.
                    continue
            start = idx
            end = idx + len(piece)
            scan_cursor = max(scan_cursor, end - chunk_overlap)

            span = _resolve_span(window, start, end)

            # Locator mirror fields for legacy v1 compatibility (chat / debug UI).
            first_block = window.blocks[span["first_idx"]]
            primary_idx = None
            for i, b in enumerate(window.blocks):
                if b["block_id"] == span["primary_block_id"]:
                    primary_idx = i
                    break
            primary_block = (
                window.blocks[primary_idx] if primary_idx is not None else first_block
            )
            legacy_page = (
                primary_block.get("locator", {}).get("page", 0)
                if primary_block.get("locator_type") == "pdf"
                else 0
            )
            section_order = (
                section_order_by_id.get(section_id, 0) if section_id else 0
            )
            section_label = section_label_by_id.get(section_id, "") if section_id else ""

            metadata = {
                "chunk_id": _chunk_id(book_id, global_order),
                "book_id": book_id,
                "raptor_level": 0,
                # New canonical fields
                "section_id": section_id or "",
                "block_ids": encode_block_ids(span["block_ids"]),
                "primary_block_id": span["primary_block_id"],
                "first_char_offset": span["first_char_offset"],
                "last_char_offset": span["last_char_offset"],
                # Legacy mirrors (kept until phase 3 cleanup)
                "chapter": section_order + 1 if section_id else 0,
                "chapter_label": section_label,
                "page": legacy_page,
            }
            documents.append(Document(page_content=piece, metadata=metadata))
            global_order += 1

    return documents
