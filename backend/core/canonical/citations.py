"""Citation resolution: chunk hit (raw or RAPTOR summary) → Citation payload.

A Citation payload is the contract between the chat backend and the
frontend's "click → scroll → highlight" UX. It is always rooted in the
canonical block layer, never in a chunk_id.

This module deliberately knows about Chroma (where chunks live with
block_ids metadata) AND SQLite (where blocks / sections / raptor_node_blocks
live). Higher-level code that wants to expose citations should call
resolve_citation() rather than reach into either store directly.

Citation payload shape (also documented in docs/ARCHITECTURE.md):

    {
      "book_id": str,
      "section_id": str | None,
      "section_label": str | None,
      "anchor": {
        "primary_block_id": str,
        "block_ids": [str, ...]
      },
      "source_locator": dict,                # primary block's locator
      "evidence": {
        "snippet": str,                      # first ~280 chars (legacy consumers)
        "text": str,                         # FULL evidence text — the node's own
                                             # Chroma document: chunk text for raw
                                             # hits, the AI-generated summary text
                                             # for raptor hits (the popup labels
                                             # the latter as a chapter summary).
        "retrieved_from": {
          "layer": "raw" | "raptor",
          "node_or_chunk_id": str,
          "raptor_level": int
        }
      }
    }

Returns None when:
  * the chunk_id is unknown in Chroma, or
  * the chunk has no block_ids metadata (i.e. v1 / legacy chunk).
"""
from __future__ import annotations

from typing import Any

from core.canonical import db as canonical_db
from core.canonical.chunker import decode_block_ids


_SNIPPET_MAX = 280


def resolve_citation(book_id: str, chunk_or_node_id: str, vectorstore) -> dict | None:
    """Resolve a chunk/node id (as stored in Chroma) into a Citation payload.

    Args:
        book_id: the book in scope. Required because chunk_ids alone are
                 not globally unique once we host multiple books.
        chunk_or_node_id: a level-0 chunk_id ("chk_xxxx...") or a RAPTOR
                          summary node id ("raptor_l*_c*").
        vectorstore: an open Chroma collection for this book. We only read
                     metadata; embeddings are not touched.
    """
    if not chunk_or_node_id:
        return None

    # Pull metadata from Chroma; we go via .get(where=...) because Chroma
    # doesn't expose "fetch one by id" in a uniform way and IDs aren't
    # always what we stored as chunk_id.
    try:
        hit = vectorstore.get(where={"chunk_id": chunk_or_node_id})
    except Exception:
        return None
    metadatas = (hit or {}).get("metadatas") or []
    documents = (hit or {}).get("documents") or []
    if not metadatas:
        return None
    meta = metadatas[0]
    doc_text = documents[0] if documents else ""

    raptor_level = int(meta.get("raptor_level") or 0)
    layer = "raw" if raptor_level == 0 else "raptor"

    block_ids = decode_block_ids(meta.get("block_ids", ""))
    primary_block_id = meta.get("primary_block_id") or ""

    # RAPTOR summary nodes don't carry block_ids in Chroma metadata; pull
    # them from the raptor_node_blocks reverse index instead. The first
    # block is used as the primary jump target (stable + cheap).
    if not block_ids:
        block_ids = canonical_db.get_node_block_ids(book_id, chunk_or_node_id)
        if block_ids and not primary_block_id:
            primary_block_id = block_ids[0]

    if not block_ids:
        return None

    # Look up primary block's section + locator
    primary_block = canonical_db.get_block(book_id, primary_block_id) if primary_block_id else None
    if primary_block is None:
        # Fall back to the first covered block if primary lookup failed.
        primary_block = canonical_db.get_block(book_id, block_ids[0])
        if primary_block is None:
            return None
        primary_block_id = primary_block["block_id"]

    section_label: str | None = None
    section_id: str | None = primary_block.get("section_id")
    if section_id:
        for s in canonical_db.get_sections(book_id):
            if s["section_id"] == section_id:
                section_label = s["label"]
                break

    # The node's OWN text in both cases. For raptor hits this is the
    # AI-generated summary — since 2026-06-10 summaries are first-class
    # citation targets and the popup labels them explicitly, so showing
    # the summary the fact was actually grounded in is the honest move
    # (the old behavior substituted the primary block's prose, which
    # implied a block-level precision summaries don't have).
    evidence_text = (
        doc_text if layer == "raw" else (doc_text or primary_block.get("text", ""))
    ) or ""
    snippet = evidence_text[:_SNIPPET_MAX]

    return {
        "book_id": book_id,
        "section_id": section_id,
        "section_label": section_label,
        "anchor": {
            "primary_block_id": primary_block_id,
            "block_ids": block_ids,
        },
        "source_locator": primary_block.get("locator", {}),
        "evidence": {
            "snippet": snippet,
            "text": evidence_text,
            "retrieved_from": {
                "layer": layer,
                "node_or_chunk_id": chunk_or_node_id,
                "raptor_level": raptor_level,
            },
        },
    }


def resolve_citations(book_id: str, ids: list[str], vectorstore) -> list[dict]:
    """Batch helper - same as calling resolve_citation per id, but with
    only one Chroma round-trip when possible.
    """
    out: list[dict] = []
    for cid in ids:
        c = resolve_citation(book_id, cid, vectorstore)
        if c is not None:
            out.append(c)
    return out
