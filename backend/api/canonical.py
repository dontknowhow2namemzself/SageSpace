"""Canonical text browse + citation resolution APIs.

These endpoints only serve books with ingest_version >= 2. Asking about a
v1 book returns 409 so the frontend can keep using the legacy chunk-map
UI for it without ambiguity.

Endpoints:
  GET  /books/{book_id}/sections
  GET  /books/{book_id}/blocks
        ?section_id=...
        &from_offset=...&to_offset=...
        &after=...        # pagination cursor (largest order_idx returned)
        &limit=...        # default 200
  GET  /books/{book_id}/blocks/{block_id}
  GET  /books/{book_id}/ingestion-report
  GET  /books/{book_id}/citations/{chunk_or_node_id}
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from core import database as db
from core.canonical import db as canonical_db
from core.canonical.citations import resolve_citation


router = APIRouter()


def _require_v2_book(book_id: str) -> dict:
    book = db.get_book(book_id)
    if book is None:
        raise HTTPException(status_code=404, detail="Book not found")
    if int(book.get("ingest_version") or 1) < 2:
        raise HTTPException(
            status_code=409,
            detail="Book is on legacy ingest_version=1; canonical APIs unavailable. "
                   "Re-ingest the book to enable.",
        )
    return book


@router.get("/books/{book_id}/sections")
def list_sections(book_id: str):
    _require_v2_book(book_id)
    return {"sections": canonical_db.get_sections(book_id)}


@router.get("/books/{book_id}/blocks")
def list_blocks(
    book_id: str,
    section_id: str | None = Query(default=None),
    from_offset: int | None = Query(default=None, ge=0),
    to_offset: int | None = Query(default=None, ge=0),
    after: int | None = Query(default=None, ge=0, description="cursor: largest order_idx seen"),
    limit: int = Query(default=200, ge=1, le=1000),
):
    _require_v2_book(book_id)
    blocks = canonical_db.get_blocks(
        book_id,
        section_id=section_id,
        from_offset=from_offset,
        to_offset=to_offset,
        after_order_idx=after,
        limit=limit,
    )
    next_cursor = blocks[-1]["order_idx"] if len(blocks) == limit else None
    return {"blocks": blocks, "next_cursor": next_cursor}


@router.get("/books/{book_id}/blocks/{block_id}")
def get_block(book_id: str, block_id: str):
    _require_v2_book(book_id)
    block = canonical_db.get_block(book_id, block_id)
    if block is None:
        raise HTTPException(status_code=404, detail="Block not found")
    return block


@router.get("/books/{book_id}/ingestion-report")
def get_ingestion_report(book_id: str):
    _require_v2_book(book_id)
    rep = canonical_db.get_ingestion_report(book_id)
    if rep is None:
        raise HTTPException(status_code=404, detail="No ingestion report stored")
    return rep


@router.get("/books/{book_id}/citations/{chunk_or_node_id}")
def get_citation(book_id: str, chunk_or_node_id: str):
    """Resolve a single chunk_id or RAPTOR node_id into a Citation payload."""
    _require_v2_book(book_id)
    # Open the book's Chroma collection on demand (cheap; no embeddings ride).
    from core.raptor import get_vectorstore
    vs = get_vectorstore(book_id)
    citation = resolve_citation(book_id, chunk_or_node_id, vs)
    if citation is None:
        raise HTTPException(
            status_code=404,
            detail="Citation could not be resolved; chunk_id may be unknown "
                   "or refer to a legacy v1 chunk without block_ids.",
        )
    return citation
