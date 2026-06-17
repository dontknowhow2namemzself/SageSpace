"""FTS5 keyword search over canonical block text.

The query side of PR2's hybrid retrieval -- the engine behind the
`keyword_search` agent tool. This is the human "Ctrl+F": exact-entity
lookup (characters, places, coined terms) where pure vector search is
weak, because proper nouns embed poorly.

The index itself (`blocks_fts`) is owned by `core/database.py`: schema
in `init_db`, populated by `replace_canonical_book` on ingest and by
`_backfill_blocks_fts` for pre-PR2 books. This module only reads it.

CJK note: the index uses the default `unicode61` tokenizer, which does
not segment CJK runs -- keyword search is English-corpus-only for now
(see README -> Known limitations / deferred work).
"""
from __future__ import annotations

import re
import sqlite3

from core.database import get_conn


# Column index of `text` in blocks_fts (block_id=0, book_id=1, text=2);
# used by snippet() to highlight the matched column.
_TEXT_COL = 2


def _build_match_query(terms: str) -> str | None:
    """Turn free-form user/agent `terms` into a safe FTS5 MATCH string.

    Strategy: extract word tokens, double-quote each as a literal term,
    and AND them (every term must appear in the block -- precise, which
    is the point of keyword search). Quoting is what makes this injection
    safe: any stray FTS5 operator characters (`"`, `*`, `:`, `(`, `^`,
    `-`, `OR`, `NEAR`) become inert because they live inside a quoted
    literal or are dropped by the `\\w+` tokenizer.

    Returns None when no usable token remains (caller -> empty result).
    """
    if not terms:
        return None
    tokens = re.findall(r"\w+", terms, flags=re.UNICODE)
    if not tokens:
        return None
    # Quote each token; a doubled quote escapes a literal quote inside an
    # FTS5 string (defensive -- \\w+ already strips quotes).
    return " AND ".join('"' + t.replace('"', '""') + '"' for t in tokens)


def search_blocks_fts(book_id: str, terms: str, limit: int = 8) -> list[dict]:
    """Return up to `limit` blocks of `book_id` whose text matches `terms`,
    best-rank first.

    Each hit: ``{block_id, text, snippet}`` where `snippet` is a short
    highlighted excerpt (matched terms wrapped in `[...]`). Returns an
    empty list on no match, an unusable query, or any SQLite/FTS error --
    this backs an agent tool and must never raise into the retrieval loop.
    """
    match = _build_match_query(terms)
    if match is None:
        return []
    conn = get_conn()
    try:
        # FTS5 snippet()'s column-index arg must be an integer literal, not a
        # bind parameter -- _TEXT_COL is a trusted module constant, so
        # interpolating it is safe (no user input touches the SQL text).
        rows = conn.execute(
            f"""
            SELECT block_id,
                   text,
                   snippet(blocks_fts, {_TEXT_COL}, '[', ']', ' … ', 12) AS snippet
            FROM blocks_fts
            WHERE blocks_fts MATCH ? AND book_id = ?
            ORDER BY rank
            LIMIT ?
            """,
            (match, book_id, limit),
        ).fetchall()
    except sqlite3.Error:
        return []
    finally:
        conn.close()
    return [
        {"block_id": r["block_id"], "text": r["text"], "snippet": r["snippet"]}
        for r in rows
    ]
