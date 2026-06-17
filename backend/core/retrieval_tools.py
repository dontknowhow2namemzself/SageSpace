"""The four retrieval tools the bounded ReAct retrieve agent calls (PR2).

Each is a PURE function returning langchain `Document`s tagged with
`metadata['origin']` -- and with NO database side effects. The retrieve
node persists exactly ONE retrieval_event for the whole turn (after the
agent loop), so its tools must not each persist their own. Purity also
keeps them unit-testable in isolation.

| tool                       | origin     | backed by                         |
|----------------------------|------------|-----------------------------------|
| tool_semantic_search       | "semantic" | retriever.retrieve_combined (MQ+HyDE) |
| tool_keyword_search        | "keyword"  | keyword_search.search_blocks_fts (FTS5) |
| tool_get_chapter           | "chapter"  | pipeline.retrieve.fetch_chapter_docs |
| tool_expand_neighbors      | "neighbor" | canonical block order_idx adjacency |

Keyword hits land on `blocks`, but evidence/attribution is chunk-based,
so keyword + neighbor hits are mapped to their containing level-0 chunk
via a `block_id -> chunk_id` index built from Chroma metadata. The node
dedups by chunk_id and merges origins (a chunk found by both semantic
and keyword retrieval becomes origin "both").
"""
from __future__ import annotations

import logging

from langchain_chroma import Chroma
from langchain_core.documents import Document

from core.canonical import db as canonical_db
from core.canonical.chunker import decode_block_ids
from core.keyword_search import search_blocks_fts
from core.pipeline.retrieve import fetch_chapter_docs
from core.retriever import retrieve_combined


logger = logging.getLogger(__name__)

# Per-tool evidence cap (the node caps the merged union again).
MAX_TOOL_DOCS = 8
# How many canonical BLOCKS beyond the seed chunk's edge blocks
# `expand_neighbors` reaches before mapping back to chunks. 1 = only the
# immediately-adjacent chunk on each side (lo-1 / hi+1 always fall outside the
# seed, so this is non-empty except at the book's very first/last block). Was 3,
# which spilled into the 2nd neighbor and diluted precision — the chronic weak
# spot (Q4 analysis 2026-06-08). Tunable; A/B with sage-eval design-lens precision.
_NEIGHBOR_WINDOW = 1


# ── semantic_search ─────────────────────────────────────────────────────────


def tool_semantic_search(query: str, vectorstore: Chroma) -> list[Document]:
    """Multi-query + HyDE vector retrieval (conceptual / thematic questions)."""
    if vectorstore is None or not query:
        return []
    docs = retrieve_combined(query, vectorstore)
    for d in docs:
        d.metadata["origin"] = "semantic"
    return docs


# ── keyword_search ──────────────────────────────────────────────────────────


def tool_keyword_search(
    terms: str, book_id: str, vectorstore: Chroma
) -> list[Document]:
    """FTS5 exact-term lookup over block text, mapped up to the containing
    level-0 chunks so the hits join the chunk-based evidence pool. Best for
    proper nouns (characters, places, coined terms)."""
    if vectorstore is None or not terms:
        return []
    hits = search_blocks_fts(book_id, terms, limit=MAX_TOOL_DOCS)
    if not hits:
        return []
    index = _block_to_chunk_index(vectorstore)
    chunk_ids = _unique_chunk_ids((h["block_id"] for h in hits), index)
    docs = _fetch_chunk_docs(vectorstore, chunk_ids)
    for d in docs:
        d.metadata["origin"] = "keyword"
    return docs


# ── get_chapter ─────────────────────────────────────────────────────────────


def tool_get_chapter(
    printed_number: int, book_id: str, vectorstore: Chroma,
    query: str = "",
) -> list[Document]:
    """Pull a whole chapter's summary + passages ("what happens in ch N").

    `query` is the agent's sub-question, threaded in by the dispatcher
    (not part of the tool schema the model sees): with it, the chapter's
    level-0 slots go to the chunks most relevant to the sub-question
    instead of an even reading-order spread."""
    docs = fetch_chapter_docs(book_id, printed_number, vectorstore, query=query or None)
    for d in docs:
        d.metadata["origin"] = "chapter"
    return docs


# ── expand_neighbors ────────────────────────────────────────────────────────


def tool_expand_neighbors(
    chunk_id: str, book_id: str, vectorstore: Chroma
) -> list[Document]:
    """Widen context around a strong hit: return the chunks immediately
    adjacent (in reading order) to `chunk_id`, via canonical block
    `order_idx` adjacency. Excludes the seed chunk itself.

    The window is anchored on the seed chunk's EDGE blocks (first + last),
    not its midpoint, and reaches _NEIGHBOR_WINDOW blocks beyond each edge.
    Anchoring on the midpoint would fail for a large seed chunk: a +/-window
    around the centre block could stay entirely inside the seed and surface
    no neighbors at all. Anchoring on the edges always escapes the seed,
    whatever its size.
    """
    if vectorstore is None or not chunk_id:
        return []
    seed = _fetch_chunk_docs(vectorstore, [chunk_id])
    if not seed:
        return []
    lo, hi = _chunk_block_span(book_id, seed[0])
    if lo is None:
        return []

    # Blocks in [lo - window, hi + window], in order. get_blocks only bounds
    # the lower edge (after_order_idx) + count; order_idx is contiguous per
    # book, so the count reaches hi + window. (Near the very start of the
    # book it may reach a block or two further forward — harmless extra
    # context, capped downstream.)
    neighbor_blocks = canonical_db.get_blocks(
        book_id,
        after_order_idx=lo - _NEIGHBOR_WINDOW - 1,
        limit=(hi - lo + 1) + 2 * _NEIGHBOR_WINDOW,
    )
    index = _block_to_chunk_index(vectorstore)
    neighbor_chunk_ids = [
        cid
        for cid in _unique_chunk_ids((b["block_id"] for b in neighbor_blocks), index)
        if cid != chunk_id
    ]
    docs = _fetch_chunk_docs(vectorstore, neighbor_chunk_ids)
    for d in docs:
        d.metadata["origin"] = "neighbor"
    return docs


def _chunk_block_span(book_id: str, doc: Document) -> tuple[int | None, int | None]:
    """The seed chunk's [first, last] block `order_idx` -- the edges the
    neighbor window expands outward from.

    Uses the chunk's `block_ids` (the chunker emits them in reading order,
    so element 0 / -1 are the span endpoints). Falls back to the primary
    (midpoint) block as a zero-width span when no block_ids are present.
    Returns (None, None) when nothing resolves.
    """
    block_ids = decode_block_ids(doc.metadata.get("block_ids"))
    if not block_ids:
        pb = doc.metadata.get("primary_block_id")
        block = canonical_db.get_block(book_id, pb) if pb else None
        if block is None:
            return None, None
        return block["order_idx"], block["order_idx"]

    endpoints = [
        canonical_db.get_block(book_id, block_ids[0]),
        canonical_db.get_block(book_id, block_ids[-1]),
    ]
    idxs = [b["order_idx"] for b in endpoints if b is not None]
    if not idxs:
        return None, None
    return min(idxs), max(idxs)


# ── Shared helpers ──────────────────────────────────────────────────────────


def _block_to_chunk_index(vectorstore: Chroma) -> dict[str, str]:
    """Map every block_id -> its containing level-0 chunk_id, from Chroma
    metadata (`block_ids` is a comma-joined list per chunk). First chunk
    wins on the rare overlap. Empty dict if the store can't be read."""
    if not hasattr(vectorstore, "get"):
        return {}
    try:
        res = vectorstore.get(where={"raptor_level": 0})
    except Exception:
        return {}
    out: dict[str, str] = {}
    for meta in (res.get("metadatas") or []):
        cid = meta.get("chunk_id")
        if not cid:
            continue
        for bid in decode_block_ids(meta.get("block_ids")):
            out.setdefault(bid, cid)
    return out


def _unique_chunk_ids(block_ids, index: dict[str, str]) -> list[str]:
    """Map an ordered iterable of block_ids -> chunk_ids, deduped, order
    preserved (so keyword relevance / reading order survives)."""
    seen: set[str] = set()
    out: list[str] = []
    for bid in block_ids:
        cid = index.get(bid)
        if cid and cid not in seen:
            seen.add(cid)
            out.append(cid)
    return out


def _fetch_chunk_docs(vectorstore: Chroma, chunk_ids: list[str]) -> list[Document]:
    """Batch-fetch level-0 chunk Documents by chunk_id, preserving the
    requested order. Empty list on any store error."""
    if not chunk_ids or not hasattr(vectorstore, "get"):
        return []
    try:
        got = vectorstore.get(where={"chunk_id": {"$in": list(chunk_ids)}})
    except Exception:
        return []
    by_cid: dict[str, Document] = {}
    docs_text = got.get("documents") or []
    metas = got.get("metadatas") or []
    for text, meta in zip(docs_text, metas):
        cid = (meta or {}).get("chunk_id")
        if cid:
            by_cid[cid] = Document(page_content=text or "", metadata=dict(meta))
    return [by_cid[cid] for cid in chunk_ids if cid in by_cid]
