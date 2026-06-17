"""SQLite persistence for the canonical text layer.

Write API:
    replace_canonical_book(book_id, sections, blocks, report)
        Atomic full replace. The only supported write path. Sets the book's
        ingest_version to 2 on success.

Read APIs (all read-only, safe to call from API handlers):
    get_sections(book_id)
    get_blocks(book_id, section_id=None, from_offset=None, to_offset=None,
               limit=None, offset_cursor=None)
    get_block(book_id, block_id)
    get_ingestion_report(book_id)
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Iterable

from core.database import get_conn, populate_blocks_fts
from core.canonical.models import Block, Section


# ── Writes ──────────────────────────────────────────────────────────────────


def replace_canonical_book(
    book_id: str,
    sections: Iterable[Section],
    blocks: Iterable[Block],
    report: dict,
) -> None:
    """Atomically replace the canonical layer for `book_id` and bump the
    book's ingest_version to 2.

    Existing chunks / RAPTOR rows are left untouched - they will be rebuilt by
    a downstream step (phase 2). This function only owns sections / blocks /
    ingestion_reports / books.ingest_version.
    """
    sections_list = list(sections)
    blocks_list = list(blocks)
    now = datetime.now(timezone.utc).isoformat()

    conn = get_conn()
    try:
        conn.execute("BEGIN")

        # Wipe previous canonical rows for this book (idempotent re-ingest).
        conn.execute("DELETE FROM blocks WHERE book_id = ?", (book_id,))
        conn.execute("DELETE FROM sections WHERE book_id = ?", (book_id,))
        conn.execute(
            "DELETE FROM ingestion_reports WHERE book_id = ? AND ingest_version = 2",
            (book_id,),
        )

        # Sections first so blocks' FK is satisfied.
        if sections_list:
            conn.executemany(
                """
                INSERT INTO sections
                    (section_id, book_id, parent_section_id, order_idx,
                     level, label, source, kind, printed_number)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        s.section_id,
                        s.book_id,
                        s.parent_section_id,
                        s.order_idx,
                        s.level,
                        s.label,
                        s.source,
                        s.kind,
                        s.printed_number,
                    )
                    for s in sections_list
                ],
            )

        if blocks_list:
            conn.executemany(
                """
                INSERT INTO blocks
                    (block_id, book_id, section_id, order_idx, kind, text,
                     book_offset_start, book_offset_end,
                     locator_type, locator_json, norm_flags_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        b.block_id,
                        b.book_id,
                        b.section_id,
                        b.order_idx,
                        b.kind,
                        b.text,
                        b.book_offset_start,
                        b.book_offset_end,
                        b.locator_type,
                        b.locator_json,
                        b.norm_flags_json,
                    )
                    for b in blocks_list
                ],
            )

        # Keep the FTS5 keyword index (blocks_fts) in lockstep with the block
        # rows we just wrote. Runs even when blocks_list is empty so a re-ingest
        # that drops all blocks also clears the book's stale keyword index.
        populate_blocks_fts(
            conn,
            book_id,
            [(b.block_id, b.book_id, b.text) for b in blocks_list],
        )

        conn.execute(
            """
            INSERT INTO ingestion_reports (book_id, ingest_version, report_json, created_at)
            VALUES (?, 2, ?, ?)
            """,
            (book_id, json.dumps(report, ensure_ascii=False, sort_keys=True), now),
        )

        conn.execute(
            "UPDATE books SET ingest_version = 2 WHERE id = ?",
            (book_id,),
        )

        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ── Reads ───────────────────────────────────────────────────────────────────


def _row_to_section(row) -> dict:
    # `kind` / `printed_number` may be absent on rows that pre-date the
    # PR4 migration if some external process inserted into sections
    # directly. _backfill_section_kind_and_number in core/database.py
    # populates them on init, but defensively coerce here too.
    keys = row.keys() if hasattr(row, "keys") else []
    return {
        "section_id": row["section_id"],
        "book_id": row["book_id"],
        "parent_section_id": row["parent_section_id"],
        "order_idx": row["order_idx"],
        "level": row["level"],
        "label": row["label"],
        "source": row["source"],
        "kind": row["kind"] if "kind" in keys else "other",
        "printed_number": row["printed_number"] if "printed_number" in keys else None,
    }


def _row_to_block(row) -> dict:
    return {
        "block_id": row["block_id"],
        "book_id": row["book_id"],
        "section_id": row["section_id"],
        "order_idx": row["order_idx"],
        "kind": row["kind"],
        "text": row["text"],
        "book_offset_start": row["book_offset_start"],
        "book_offset_end": row["book_offset_end"],
        "locator_type": row["locator_type"],
        "locator": json.loads(row["locator_json"]),
        "norm_flags": json.loads(row["norm_flags_json"]),
    }


def get_sections(book_id: str) -> list[dict]:
    conn = get_conn()
    try:
        rows = conn.execute(
            "SELECT * FROM sections WHERE book_id = ? ORDER BY order_idx",
            (book_id,),
        ).fetchall()
        return [_row_to_section(r) for r in rows]
    finally:
        conn.close()


def get_blocks(
    book_id: str,
    *,
    section_id: str | None = None,
    from_offset: int | None = None,
    to_offset: int | None = None,
    limit: int | None = None,
    after_order_idx: int | None = None,
) -> list[dict]:
    """Fetch blocks in canonical (order_idx) order.

    Filters are AND-combined. `after_order_idx` is the pagination cursor for
    infinite-scroll: pass the largest order_idx returned by the previous page.
    """
    clauses = ["book_id = ?"]
    params: list = [book_id]

    if section_id is not None:
        clauses.append("section_id = ?")
        params.append(section_id)
    if from_offset is not None:
        clauses.append("book_offset_end >= ?")
        params.append(from_offset)
    if to_offset is not None:
        clauses.append("book_offset_start <= ?")
        params.append(to_offset)
    if after_order_idx is not None:
        clauses.append("order_idx > ?")
        params.append(after_order_idx)

    sql = (
        "SELECT * FROM blocks WHERE "
        + " AND ".join(clauses)
        + " ORDER BY order_idx"
    )
    if limit is not None:
        sql += " LIMIT ?"
        params.append(limit)

    conn = get_conn()
    try:
        rows = conn.execute(sql, params).fetchall()
        return [_row_to_block(r) for r in rows]
    finally:
        conn.close()


def get_block(book_id: str, block_id: str) -> dict | None:
    conn = get_conn()
    try:
        row = conn.execute(
            "SELECT * FROM blocks WHERE book_id = ? AND block_id = ?",
            (book_id, block_id),
        ).fetchone()
        return _row_to_block(row) if row else None
    finally:
        conn.close()


# ── RAPTOR node ↔ block reverse index ───────────────────────────────────────


def replace_raptor_node_blocks(book_id: str, node_block_pairs: list[tuple[str, str]]) -> None:
    """Atomic full replace of raptor_node_blocks for `book_id`.

    `node_block_pairs` is a flat list of (node_id, block_id) tuples. Use a set
    upstream if you want dedup; this function will tolerate duplicates via
    INSERT OR IGNORE.
    """
    conn = get_conn()
    try:
        conn.execute("BEGIN")
        conn.execute("DELETE FROM raptor_node_blocks WHERE book_id = ?", (book_id,))
        if node_block_pairs:
            conn.executemany(
                "INSERT OR IGNORE INTO raptor_node_blocks (book_id, node_id, block_id) "
                "VALUES (?, ?, ?)",
                [(book_id, nid, bid) for nid, bid in node_block_pairs],
            )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def get_node_block_ids(book_id: str, node_id: str) -> list[str]:
    """Return the list of canonical block_ids covered by a RAPTOR summary node.
    Empty list if the node is unknown or has no recorded coverage.
    """
    conn = get_conn()
    try:
        rows = conn.execute(
            "SELECT block_id FROM raptor_node_blocks "
            "WHERE book_id = ? AND node_id = ? ORDER BY block_id",
            (book_id, node_id),
        ).fetchall()
        return [r["block_id"] for r in rows]
    finally:
        conn.close()


def get_ingestion_report(book_id: str) -> dict | None:
    conn = get_conn()
    try:
        row = conn.execute(
            "SELECT report_json, created_at FROM ingestion_reports "
            "WHERE book_id = ? AND ingest_version = 2",
            (book_id,),
        ).fetchone()
        if not row:
            return None
        return {
            "created_at": row["created_at"],
            "report": json.loads(row["report_json"]),
        }
    finally:
        conn.close()
