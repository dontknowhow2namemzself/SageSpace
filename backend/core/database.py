import sqlite3
import uuid
import json
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "sagespace.db"


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS books (
            id             TEXT PRIMARY KEY,
            title          TEXT NOT NULL,
            author         TEXT,
            total_chunks   INTEGER,
            total_chapters INTEGER,
            upload_date    TEXT,
            raptor_status  TEXT DEFAULT 'pending',
            file_path      TEXT,
            ingest_version INTEGER NOT NULL DEFAULT 1
        );

        CREATE TABLE IF NOT EXISTS sessions (
            id                TEXT PRIMARY KEY,
            book_id           TEXT REFERENCES books(id),
            start_time        TEXT,
            end_time          TEXT,
            total_tokens_in   INTEGER DEFAULT 0,
            total_tokens_out  INTEGER DEFAULT 0,
            total_cost_usd    REAL DEFAULT 0.0,
            conversation_json TEXT DEFAULT '[]'
        );

        CREATE TABLE IF NOT EXISTS retrieved_chunks (
            session_id   TEXT,
            book_id      TEXT,
            chunk_id     TEXT,
            retrieved_at TEXT,
            PRIMARY KEY (session_id, chunk_id)
        );

        -- Raw (level-0) chunks CITED by an answer's per-fact attribution.
        -- This is the reader-facing progress ledger (2026-06-10): the
        -- shelf %, the Insight panel %, and the Reading Map all speak
        -- "cited" — what the user actually read through answers.
        -- retrieved_chunks above stays as the internal "fed to the
        -- synthesizer" ledger (newly-lit diffing); it is no longer
        -- shown to the user.
        CREATE TABLE IF NOT EXISTS cited_chunks (
            session_id   TEXT,
            book_id      TEXT,
            chunk_id     TEXT,
            cited_at     TEXT,
            PRIMARY KEY (session_id, chunk_id)
        );

        CREATE TABLE IF NOT EXISTS retrieval_events (
            id                         TEXT PRIMARY KEY,
            session_id                 TEXT REFERENCES sessions(id),
            book_id                    TEXT REFERENCES books(id),
            query_text                 TEXT,
            multi_query_variants_json  TEXT,
            hyde_hypothesis            TEXT,
            raw_hits_count             INTEGER DEFAULT 0,
            new_raw_hits_count         INTEGER DEFAULT 0,
            summary_hits_count         INTEGER DEFAULT 0,
            created_at                 TEXT,
            answer_attribution_json    TEXT,
            faithfulness_score         REAL,
            faithfulness_status        TEXT DEFAULT 'pending',
            faithfulness_reasoning     TEXT
        );

        CREATE TABLE IF NOT EXISTS retrieval_event_chunks (
            event_id        TEXT REFERENCES retrieval_events(id),
            chunk_id        TEXT,
            raptor_level    INTEGER,
            chapter         INTEGER,
            page            INTEGER,
            rank            INTEGER,
            origin          TEXT,
            is_new_lighting INTEGER DEFAULT 0,
            preview_text    TEXT,
            PRIMARY KEY (event_id, chunk_id)
        );

        -- Canonical text layer (ingest_version=2). The block is the smallest
        -- addressable semantic unit; chunks / RAPTOR nodes will reference it.
        -- See docs/ARCHITECTURE.md §canonical-refactor.
        --
        -- kind / printed_number are how we resolve "Chapter N" -- without
        -- them we hit sections[N-1] which silently picks up front-matter
        -- slots (cover / ToC) and returns the wrong content.
        --   kind:           cover | titlepage | toc | preface | foreword |
        --                   introduction | front_matter | prologue |
        --                   chapter | epilogue | afterword | appendix |
        --                   glossary | index | bibliography | back_matter |
        --                   other
        --   printed_number: 5 for "CHAPTER V" / "第五章" / "Chapter Five";
        --                   NULL for non-chapter sections.
        CREATE TABLE IF NOT EXISTS sections (
            section_id        TEXT PRIMARY KEY,
            book_id           TEXT NOT NULL REFERENCES books(id),
            parent_section_id TEXT REFERENCES sections(section_id),
            order_idx         INTEGER NOT NULL,
            level             INTEGER NOT NULL DEFAULT 1,
            label             TEXT NOT NULL,
            source            TEXT NOT NULL DEFAULT 'inferred',  -- 'outline' | 'toc' | 'inferred'
            kind              TEXT NOT NULL DEFAULT 'other',
            printed_number    INTEGER
        );
        CREATE INDEX IF NOT EXISTS idx_sections_book_order
            ON sections (book_id, order_idx);

        CREATE TABLE IF NOT EXISTS blocks (
            block_id            TEXT PRIMARY KEY,
            book_id             TEXT NOT NULL REFERENCES books(id),
            section_id          TEXT REFERENCES sections(section_id),
            order_idx           INTEGER NOT NULL,
            kind                TEXT NOT NULL,  -- paragraph | heading | list_item | quote | footnote | caption | figure
            text                TEXT NOT NULL,
            book_offset_start   INTEGER NOT NULL,
            book_offset_end     INTEGER NOT NULL,
            locator_type        TEXT NOT NULL,  -- 'pdf' | 'epub'
            locator_json        TEXT NOT NULL,  -- {page, bbox} or {spine_idx, cfi, print_page}
            norm_flags_json     TEXT NOT NULL DEFAULT '{}'
        );
        CREATE INDEX IF NOT EXISTS idx_blocks_book_order
            ON blocks (book_id, order_idx);
        CREATE INDEX IF NOT EXISTS idx_blocks_section
            ON blocks (section_id);

        -- FTS5 keyword index over canonical block text. Backs the PR2
        -- hybrid-retrieval `keyword_search` tool (the human "Ctrl+F":
        -- exact characters / places / terms that vector search is weak on).
        -- Standalone table (keeps its own copy of the text) for robustness;
        -- book_id stored UNINDEXED so we can filter per book without
        -- full-text-matching against it. Default unicode61 tokenizer does
        -- NOT segment CJK (see README -> Known limitations); the corpus is
        -- English for now. Populated by replace_canonical_book on ingest +
        -- _backfill_blocks_fts for pre-PR2 books.
        CREATE VIRTUAL TABLE IF NOT EXISTS blocks_fts USING fts5(
            block_id UNINDEXED,
            book_id  UNINDEXED,
            text,
            tokenize = 'unicode61'
        );

        -- Audit report per ingest run; one row per (book_id, ingest_version).
        CREATE TABLE IF NOT EXISTS ingestion_reports (
            book_id        TEXT NOT NULL REFERENCES books(id),
            ingest_version INTEGER NOT NULL,
            report_json    TEXT NOT NULL,
            created_at     TEXT NOT NULL,
            PRIMARY KEY (book_id, ingest_version)
        );

        -- Reverse index from RAPTOR summary nodes to the canonical blocks
        -- they ultimately cover. Populated for ingest_version=2 books only.
        -- A node "covers" a block iff some leaf-level (level=0) chunk in its
        -- subtree references that block.
        CREATE TABLE IF NOT EXISTS raptor_node_blocks (
            book_id  TEXT NOT NULL REFERENCES books(id),
            node_id  TEXT NOT NULL,    -- chunk_id of the RAPTOR summary node
            block_id TEXT NOT NULL REFERENCES blocks(block_id),
            PRIMARY KEY (book_id, node_id, block_id)
        );
        CREATE INDEX IF NOT EXISTS idx_raptor_node_blocks_node
            ON raptor_node_blocks (node_id);

        -- Memory MVP: fast-lane user facts (memory-system-design.md "NOW" §A).
        -- One row per EXPLICITLY-stated durable fact/interest the user dropped
        -- in chat ("叫我小王", "我在啃斯多葛"), captured silently by piggybacking
        -- the per-turn classify_intent call (zero extra LLM calls) and written
        -- at finalize. Invisible in the MVP (no panel) -- read as plain text by
        -- recommend(). No embedding yet.
        --   type:           fact | interest
        --   source_book_id: provenance only (the book whose session it was said
        --                   in); NULLABLE. On book delete it is NULLed, NOT
        --                   cascaded -- a user-level fact outlives any one book.
        --   source_locator: the session_id it was captured in (chat origin has
        --                   no page locator).
        CREATE TABLE IF NOT EXISTS memory_notes (
            id             TEXT PRIMARY KEY,
            text           TEXT NOT NULL,
            type           TEXT NOT NULL DEFAULT 'fact',
            source_book_id TEXT REFERENCES books(id),
            source_locator TEXT,
            created_at     TEXT NOT NULL
        );

        -- Memory MVP: book recommendations (memory-system-design.md "NOW" §B).
        -- One row per suggested book. Deliberately has NO book_id -- a rec is a
        -- book the user does NOT own, validated against Google Books; it is
        -- user-level, so it survives book deletes (exclude-by-title still works).
        --   status: suggested(on screen) | seen(shown, no action / 换一批) |
        --           added(+想读) | dismissed(忽略). The full lifecycle is kept
        --           (never deleted) so the eval denominator (GROUP BY status =
        --           add-rate) stays intact.
        --   which_interest: the specific interest this pick is grounded in
        --           (shown as the card's reason). reason/blurb are display text.
        CREATE TABLE IF NOT EXISTS recommendations (
            id             TEXT PRIMARY KEY,
            title          TEXT NOT NULL,
            author         TEXT,
            blurb          TEXT,
            reason         TEXT,
            which_interest TEXT,
            status         TEXT NOT NULL DEFAULT 'suggested',
            created_at     TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_recommendations_status
            ON recommendations (status);
    """)
    # ── Idempotent migrations for older DBs ─────────────────────────────
    existing_cols = {
        row["name"]
        for row in conn.execute("PRAGMA table_info(retrieval_events)").fetchall()
    }
    if "faithfulness_score" not in existing_cols:
        conn.execute("ALTER TABLE retrieval_events ADD COLUMN faithfulness_score REAL")
    if "answer_attribution_json" not in existing_cols:
        conn.execute("ALTER TABLE retrieval_events ADD COLUMN answer_attribution_json TEXT")
    if "faithfulness_status" not in existing_cols:
        conn.execute(
            "ALTER TABLE retrieval_events ADD COLUMN faithfulness_status TEXT DEFAULT 'pending'"
        )
    if "faithfulness_reasoning" not in existing_cols:
        conn.execute("ALTER TABLE retrieval_events ADD COLUMN faithfulness_reasoning TEXT")

    # books.ingest_version: 1 = legacy (page/chapter-based parser),
    # 2 = canonical (block-based, see docs/ARCHITECTURE.md §canonical-refactor).
    # All existing rows are backfilled to 1 so legacy retrieval paths keep working.
    books_cols = {
        row["name"] for row in conn.execute("PRAGMA table_info(books)").fetchall()
    }
    if "ingest_version" not in books_cols:
        conn.execute(
            "ALTER TABLE books ADD COLUMN ingest_version INTEGER NOT NULL DEFAULT 1"
        )

    # sections.kind / sections.printed_number (added in PR4). Older v2 books
    # were ingested before these columns existed; add them now and back-fill
    # the existing rows by running classify_section_kind / parse_printed_number
    # on each label. New ingests overwrite these heuristic values with the
    # higher-signal version from the normalizer (epub:type for EPUB books).
    sections_cols = {
        row["name"] for row in conn.execute("PRAGMA table_info(sections)").fetchall()
    }
    kind_added = False
    if "kind" not in sections_cols:
        conn.execute(
            "ALTER TABLE sections ADD COLUMN kind TEXT NOT NULL DEFAULT 'other'"
        )
        kind_added = True
    if "printed_number" not in sections_cols:
        conn.execute("ALTER TABLE sections ADD COLUMN printed_number INTEGER")
    if kind_added:
        _backfill_section_kind_and_number(conn)

    # Populate the FTS5 keyword index for any v2 book ingested before PR2
    # (idempotent: only books with blocks but no FTS rows are touched).
    _backfill_blocks_fts(conn)

    conn.commit()
    conn.close()


def populate_blocks_fts(conn: sqlite3.Connection, book_id: str, rows) -> None:
    """(Re)build the FTS5 keyword index for one book, in `conn`'s
    transaction. `rows` is an iterable of (block_id, book_id, text) tuples.

    Shared by canonical ingest (replace_canonical_book, to keep the index in
    lockstep with the block rows) and _backfill_blocks_fts. Always wipes the
    book's existing FTS rows first, so re-ingest / re-backfill is idempotent.
    """
    conn.execute("DELETE FROM blocks_fts WHERE book_id = ?", (book_id,))
    conn.executemany(
        "INSERT INTO blocks_fts (block_id, book_id, text) VALUES (?, ?, ?)",
        rows,
    )


def _backfill_blocks_fts(conn: sqlite3.Connection) -> None:
    """One-shot, idempotent population of blocks_fts for books that have
    block rows but no FTS rows yet -- i.e. v2 books ingested before the PR2
    keyword index existed. Books already indexed are skipped, so this is a
    cheap no-op on every subsequent startup."""
    with_blocks = {
        r[0] for r in conn.execute("SELECT DISTINCT book_id FROM blocks").fetchall()
    }
    with_fts = {
        r[0] for r in conn.execute("SELECT DISTINCT book_id FROM blocks_fts").fetchall()
    }
    for book_id in with_blocks - with_fts:
        rows = conn.execute(
            "SELECT block_id, book_id, text FROM blocks WHERE book_id = ?",
            (book_id,),
        ).fetchall()
        populate_blocks_fts(conn, book_id, [(r[0], r[1], r[2]) for r in rows])


def _backfill_section_kind_and_number(conn: sqlite3.Connection) -> None:
    """One-shot pass over every existing section row to populate the new
    kind + printed_number fields using the label heuristic. Called from
    init_db when kind was just ALTER-added so previously-ingested v2
    books get usable values without forcing a full re-ingest.

    New ingests overwrite this with the higher-signal value the
    normalizer produces (epub:type for EPUB, outline-based heuristic
    for PDF).
    """
    from core.canonical.chapter_parse import (
        classify_section_kind,
        parse_printed_number,
    )
    rows = conn.execute(
        "SELECT section_id, label FROM sections WHERE kind = 'other' OR kind IS NULL"
    ).fetchall()
    for row in rows:
        label = row["label"] or ""
        kind = classify_section_kind(label)
        num = parse_printed_number(label) if kind == "chapter" else None
        conn.execute(
            "UPDATE sections SET kind = ?, printed_number = ? WHERE section_id = ?",
            (kind, num, row["section_id"]),
        )


# ── Books ──────────────────────────────────────────────────────────────────

def create_book(title: str, author: str | None, file_path: str) -> str:
    book_id = str(uuid.uuid4())
    conn = get_conn()
    conn.execute(
        "INSERT INTO books (id, title, author, upload_date, raptor_status, file_path) "
        "VALUES (?,?,?,?,?,?)",
        (book_id, title, author, datetime.now(timezone.utc).isoformat(), "pending", file_path),
    )
    conn.commit()
    conn.close()
    return book_id


def get_book(book_id: str) -> dict | None:
    conn = get_conn()
    row = conn.execute("SELECT * FROM books WHERE id=?", (book_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def list_books() -> list[dict]:
    conn = get_conn()
    rows = conn.execute("SELECT * FROM books ORDER BY upload_date DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def update_book_status(
    book_id: str,
    status: str,
    total_chunks: int | None = None,
    total_chapters: int | None = None,
):
    conn = get_conn()
    if total_chunks is not None:
        conn.execute(
            "UPDATE books SET raptor_status=?, total_chunks=?, total_chapters=? WHERE id=?",
            (status, total_chunks, total_chapters, book_id),
        )
    else:
        conn.execute(
            "UPDATE books SET raptor_status=? WHERE id=?", (status, book_id)
        )
    conn.commit()
    conn.close()


def delete_book(book_id: str):
    """Hard-delete a book and every row that references it.

    Deepest dependents first, then the row itself, all in one transaction
    so we cannot leave the books row alive with orphans (or vice-versa)
    on partial failure. Only references the tables that actually exist
    in this codebase - extending later means adding lines here.
    """
    conn = get_conn()
    try:
        conn.execute("BEGIN")
        # ── Chat / retrieval side ──────────────────────────────────────
        conn.execute("""
            DELETE FROM retrieval_event_chunks
            WHERE event_id IN (
                SELECT id FROM retrieval_events WHERE book_id=?
            )
        """, (book_id,))
        conn.execute("DELETE FROM retrieval_events WHERE book_id=?", (book_id,))
        conn.execute("DELETE FROM retrieved_chunks WHERE book_id=?", (book_id,))
        conn.execute("DELETE FROM cited_chunks WHERE book_id=?", (book_id,))
        conn.execute("DELETE FROM sessions WHERE book_id=?", (book_id,))
        # ── Canonical side (v2 books; no-ops for v1) ───────────────────
        conn.execute("DELETE FROM raptor_node_blocks WHERE book_id=?", (book_id,))
        conn.execute("DELETE FROM ingestion_reports WHERE book_id=?", (book_id,))
        conn.execute("DELETE FROM blocks_fts WHERE book_id=?", (book_id,))
        conn.execute("DELETE FROM blocks   WHERE book_id=?", (book_id,))
        conn.execute("DELETE FROM sections WHERE book_id=?", (book_id,))
        # ── Memory side ────────────────────────────────────────────────
        # memory_notes are user-level facts that merely RECORD which book's
        # session they were said in. Preserve the fact, drop the dangling
        # provenance -- deleting a book must not erase "叫我小王". (This is the
        # one book-referencing table that is intentionally NOT cascaded.)
        conn.execute(
            "UPDATE memory_notes SET source_book_id=NULL WHERE source_book_id=?",
            (book_id,),
        )
        # ── The book row itself ────────────────────────────────────────
        conn.execute("DELETE FROM books WHERE id=?", (book_id,))
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ── Sessions ───────────────────────────────────────────────────────────────

def get_session_ids_for_book(book_id: str) -> list[str]:
    """All session ids (= LangGraph thread ids) belonging to a book. Used by
    the book-delete cascade to GC each thread's checkpoints (design §8 Q1)."""
    conn = get_conn()
    try:
        rows = conn.execute(
            "SELECT id FROM sessions WHERE book_id = ?", (book_id,)
        ).fetchall()
        return [r[0] for r in rows]
    finally:
        conn.close()


def create_session(book_id: str) -> str:
    session_id = str(uuid.uuid4())
    conn = get_conn()
    conn.execute(
        "INSERT INTO sessions (id, book_id, start_time) VALUES (?,?,?)",
        (session_id, book_id, datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()
    conn.close()
    return session_id


def get_session(session_id: str) -> dict | None:
    conn = get_conn()
    row = conn.execute("SELECT * FROM sessions WHERE id=?", (session_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def update_session_tokens(
    session_id: str, tokens_in: int, tokens_out: int, cost: float
):
    conn = get_conn()
    conn.execute(
        """UPDATE sessions
           SET total_tokens_in  = total_tokens_in  + ?,
               total_tokens_out = total_tokens_out + ?,
               total_cost_usd   = total_cost_usd   + ?
           WHERE id=?""",
        (tokens_in, tokens_out, cost, session_id),
    )
    conn.commit()
    conn.close()


def save_conversation(session_id: str, history_json: str):
    conn = get_conn()
    conn.execute(
        "UPDATE sessions SET conversation_json=? WHERE id=?",
        (history_json, session_id),
    )
    conn.commit()
    conn.close()


def list_sessions_for_book(book_id: str, limit: int = 100) -> list[dict]:
    """Session summaries for the history sidebar, newest first. Each row
    carries a preview (first user message) and message_count derived from
    conversation_json, so the UI never ships full histories.

    Sessions that never got a message are excluded in SQL: page loads and
    eval runs create sessions freely (hundreds per book), and an empty
    session is not "history" — the active one lives in page state, not
    in this list."""
    conn = get_conn()
    try:
        rows = conn.execute(
            """SELECT id, book_id, start_time, total_tokens_in,
                      total_tokens_out, total_cost_usd, conversation_json
               FROM sessions
               WHERE book_id=?
                 AND conversation_json IS NOT NULL
                 AND conversation_json NOT IN ('', '[]')
               ORDER BY start_time DESC
               LIMIT ?""",
            (book_id, limit),
        ).fetchall()
    finally:
        conn.close()

    out = []
    for r in rows:
        d = dict(r)
        try:
            conversation = json.loads(d.get("conversation_json") or "[]")
        except json.JSONDecodeError:
            conversation = []
        preview = next(
            (m.get("content", "") for m in conversation
             if m.get("role") == "user" and m.get("content")),
            "",
        )
        out.append({
            "id": d["id"],
            "book_id": d["book_id"],
            "start_time": d["start_time"],
            "message_count": len(conversation),
            "preview": preview[:120],
            "total_tokens_in": d.get("total_tokens_in") or 0,
            "total_tokens_out": d.get("total_tokens_out") or 0,
            "total_cost_usd": d.get("total_cost_usd") or 0.0,
        })
    return out


def delete_session(session_id: str):
    """Delete a session row plus its retrieval_events / event_chunks.

    retrieved_chunks AND cited_chunks rows are intentionally KEPT:
    reading progress is a property of the book ("how much of it the
    user read through answers"), and deleting a conversation must not
    un-read the book. LangGraph checkpoints are GC'd separately by the
    caller (core.graph.build.gc_checkpoints_for_session)."""
    conn = get_conn()
    try:
        conn.execute("BEGIN")
        conn.execute(
            """DELETE FROM retrieval_event_chunks
               WHERE event_id IN (
                   SELECT id FROM retrieval_events WHERE session_id=?
               )""",
            (session_id,),
        )
        conn.execute("DELETE FROM retrieval_events WHERE session_id=?", (session_id,))
        conn.execute("DELETE FROM sessions WHERE id=?", (session_id,))
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ── Memory notes (fast lane — memory-system-design.md §A) ────────────────────

def add_memory_note(
    text: str,
    type: str = "fact",
    source_book_id: str | None = None,
    source_locator: str | None = None,
) -> str | None:
    """Silently persist one EXPLICITLY-stated user fact/interest.

    Returns the new row id, or None when nothing was written:
      * empty / whitespace-only text -> skipped (the classifier returns null
        for the ~majority of turns; this is the belt-and-braces guard);
      * identical `text` already stored -> skipped (dedup, so the same fact
        said across sessions does not show up as N copies in the recommend
        prompt). Dedup is on exact text, case/space-sensitive -- cheap and
        good enough for the MVP's low-volume, explicit-only notes.
    """
    text = (text or "").strip()
    if not text:
        return None
    conn = get_conn()
    try:
        dup = conn.execute(
            "SELECT 1 FROM memory_notes WHERE text=? LIMIT 1", (text,)
        ).fetchone()
        if dup:
            return None
        note_id = str(uuid.uuid4())
        conn.execute(
            "INSERT INTO memory_notes "
            "(id, text, type, source_book_id, source_locator, created_at) "
            "VALUES (?,?,?,?,?,?)",
            (note_id, text, type or "fact", source_book_id, source_locator,
             datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()
        return note_id
    finally:
        conn.close()


def list_memory_notes() -> list[dict]:
    """All captured notes, newest first. Read by recommend() as plain text;
    also surfaced in the home "What I remember" panel (view/edit/delete)."""
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM memory_notes ORDER BY created_at DESC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_memory_note(note_id: str) -> dict | None:
    conn = get_conn()
    row = conn.execute(
        "SELECT * FROM memory_notes WHERE id=?", (note_id,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def update_memory_note(note_id: str, text: str, type: str | None = None) -> bool:
    """Edit a note's text (and optionally its type). Returns False when the id
    does not exist or the new text is empty (the panel guards empty saves, this
    is belt-and-braces). The user's honest correction point for what's stored."""
    text = (text or "").strip()
    if not text:
        return False
    conn = get_conn()
    try:
        if type in ("fact", "interest"):
            cur = conn.execute(
                "UPDATE memory_notes SET text=?, type=? WHERE id=?",
                (text, type, note_id),
            )
        else:
            cur = conn.execute(
                "UPDATE memory_notes SET text=? WHERE id=?", (text, note_id)
            )
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()


def delete_memory_note(note_id: str) -> bool:
    """Forget one note. Returns False if the id does not exist (so the API can
    404). The user can delete anything we've remembered, at any time."""
    conn = get_conn()
    try:
        cur = conn.execute("DELETE FROM memory_notes WHERE id=?", (note_id,))
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()


def recent_user_questions(limit: int = 15) -> list[str]:
    """The most-recent user questions across ALL sessions & books, newest first.

    The "chat-messages table" is really a per-session JSON blob
    (sessions.conversation_json = [{role, content}, ...]); there are no
    per-message timestamps. So recency is approximated: walk sessions newest
    first by start_time, and within each session take user turns in reverse
    (the list is appended chronologically, so the tail is newest), until
    `limit` is reached. Library titles remain the floor for recommend(), so a
    thin / empty history is fine (design §B). Returns <= limit non-empty
    user-message strings.
    """
    conn = get_conn()
    try:
        rows = conn.execute(
            "SELECT conversation_json FROM sessions ORDER BY start_time DESC"
        ).fetchall()
    finally:
        conn.close()

    out: list[str] = []
    for row in rows:
        try:
            convo = json.loads(row["conversation_json"] or "[]")
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(convo, list):
            continue
        for msg in reversed(convo):
            if not isinstance(msg, dict) or msg.get("role") != "user":
                continue
            content = (msg.get("content") or "").strip()
            if content:
                out.append(content)
                if len(out) >= limit:
                    return out
    return out


def library_titles() -> list[str]:
    """Every book title currently in the library. The floor for recommend()'s
    interest signal AND part of its exclude set (don't recommend owned books)."""
    conn = get_conn()
    rows = conn.execute(
        "SELECT title FROM books WHERE title IS NOT NULL AND title != ''"
    ).fetchall()
    conn.close()
    return [r["title"] for r in rows]


# ── Progress tracking ──────────────────────────────────────────────────────

def record_retrieved_chunks(session_id: str, book_id: str, chunk_ids: list[str]):
    if not chunk_ids:
        return
    conn = get_conn()
    now = datetime.now(timezone.utc).isoformat()
    conn.executemany(
        "INSERT OR IGNORE INTO retrieved_chunks "
        "(session_id, book_id, chunk_id, retrieved_at) VALUES (?,?,?,?)",
        [(session_id, book_id, cid, now) for cid in chunk_ids],
    )
    conn.commit()
    conn.close()


def get_retrieved_chunk_ids(session_id: str) -> list[str]:
    conn = get_conn()
    rows = conn.execute(
        "SELECT DISTINCT chunk_id FROM retrieved_chunks WHERE session_id=?",
        (session_id,),
    ).fetchall()
    conn.close()
    return [r["chunk_id"] for r in rows]


def get_all_retrieved_chunk_ids_for_book(book_id: str) -> list[str]:
    """Across all sessions. Internal ledger ("fed to the synthesizer");
    user-facing progress reads cited_chunks instead (2026-06-10)."""
    conn = get_conn()
    rows = conn.execute(
        "SELECT DISTINCT chunk_id FROM retrieved_chunks WHERE book_id=?",
        (book_id,),
    ).fetchall()
    conn.close()
    return [r["chunk_id"] for r in rows]


# ── Cited chunks (reader-facing progress ledger) ───────────────────────────


def record_cited_chunks(session_id: str, book_id: str, chunk_ids: list[str]):
    """Raw chunks cited by an answer's per-fact attribution. Written at
    finalize time; the PK dedupes within a session, DISTINCT queries
    dedupe across sessions."""
    if not chunk_ids:
        return
    conn = get_conn()
    now = datetime.now(timezone.utc).isoformat()
    conn.executemany(
        "INSERT OR IGNORE INTO cited_chunks "
        "(session_id, book_id, chunk_id, cited_at) VALUES (?,?,?,?)",
        [(session_id, book_id, cid, now) for cid in chunk_ids],
    )
    conn.commit()
    conn.close()


def get_cited_chunk_ids(session_id: str) -> list[str]:
    conn = get_conn()
    rows = conn.execute(
        "SELECT DISTINCT chunk_id FROM cited_chunks WHERE session_id=?",
        (session_id,),
    ).fetchall()
    conn.close()
    return [r["chunk_id"] for r in rows]


def get_all_cited_chunk_ids_for_book(book_id: str) -> list[str]:
    """Across all sessions, deduped — the bookshelf 'Explored' metric."""
    conn = get_conn()
    rows = conn.execute(
        "SELECT DISTINCT chunk_id FROM cited_chunks WHERE book_id=?",
        (book_id,),
    ).fetchall()
    conn.close()
    return [r["chunk_id"] for r in rows]


# ── Retrieval Events ───────────────────────────────────────────────────────

def create_retrieval_event(
    session_id: str,
    book_id: str,
    query_text: str,
    multi_query_variants_json: str,
    hyde_hypothesis: str,
    raw_hits_count: int,
    new_raw_hits_count: int,
    summary_hits_count: int,
) -> str:
    event_id = str(uuid.uuid4())
    conn = get_conn()
    conn.execute(
        """INSERT INTO retrieval_events
           (id, session_id, book_id, query_text,
            multi_query_variants_json, hyde_hypothesis,
            raw_hits_count, new_raw_hits_count, summary_hits_count, created_at)
           VALUES (?,?,?,?,?,?,?,?,?,?)""",
        (event_id, session_id, book_id, query_text,
         multi_query_variants_json, hyde_hypothesis,
         raw_hits_count, new_raw_hits_count, summary_hits_count,
         datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()
    conn.close()
    return event_id


def add_event_chunks(event_id: str, chunks: list[dict]) -> None:
    if not chunks:
        return
    conn = get_conn()
    conn.executemany(
        """INSERT OR IGNORE INTO retrieval_event_chunks
           (event_id, chunk_id, raptor_level, chapter, page,
            rank, origin, is_new_lighting, preview_text)
           VALUES (:event_id,:chunk_id,:raptor_level,:chapter,:page,
                   :rank,:origin,:is_new_lighting,:preview_text)""",
        [{**c, "event_id": event_id} for c in chunks],
    )
    conn.commit()
    conn.close()


def get_retrieval_events(session_id: str) -> list[dict]:
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM retrieval_events WHERE session_id=? ORDER BY created_at",
        (session_id,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_event_chunks(event_id: str) -> list[dict]:
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM retrieval_event_chunks WHERE event_id=? ORDER BY rank",
        (event_id,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_recent_event_ids_for_session(session_id: str, limit: int = 5) -> list[str]:
    """Most recent retrieval events for a session, newest first.

    Used by the online faithfulness evaluator to attach a score to all
    retrieval events that participated in the just-finished assistant turn.
    """
    conn = get_conn()
    rows = conn.execute(
        "SELECT id FROM retrieval_events WHERE session_id=? "
        "ORDER BY created_at DESC LIMIT ?",
        (session_id, limit),
    ).fetchall()
    conn.close()
    return [r["id"] for r in rows]


def update_event_faithfulness(
    event_id: str,
    status: str,
    score: float | None = None,
    reasoning: str | None = None,
) -> None:
    conn = get_conn()
    conn.execute(
        "UPDATE retrieval_events SET faithfulness_status=?, "
        "faithfulness_score=?, faithfulness_reasoning=? WHERE id=?",
        (status, score, reasoning, event_id),
    )
    conn.commit()
    conn.close()


def attach_event_answer_attribution(event_id: str, attribution: dict) -> None:
    conn = get_conn()
    conn.execute(
        "UPDATE retrieval_events SET answer_attribution_json=? WHERE id=?",
        (json.dumps(attribution, ensure_ascii=False), event_id),
    )
    conn.commit()
    conn.close()


# ── Recommendations (memory-system-design.md §B) ─────────────────────────────

_REC_STATUSES = ("suggested", "seen", "added", "dismissed")


def insert_recommendation(
    title: str,
    author: str | None,
    blurb: str | None,
    reason: str | None,
    which_interest: str | None,
    status: str = "suggested",
) -> str:
    """Persist one recommendation row. Returns the new id."""
    rec_id = str(uuid.uuid4())
    conn = get_conn()
    conn.execute(
        "INSERT INTO recommendations "
        "(id, title, author, blurb, reason, which_interest, status, created_at) "
        "VALUES (?,?,?,?,?,?,?,?)",
        (rec_id, title, author, blurb, reason, which_interest, status,
         datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()
    conn.close()
    return rec_id


def list_recommendations(status: str | None = None) -> list[dict]:
    """Recommendation rows, newest first. Filter by `status` when given."""
    conn = get_conn()
    if status is None:
        rows = conn.execute(
            "SELECT * FROM recommendations ORDER BY created_at DESC"
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM recommendations WHERE status=? ORDER BY created_at DESC",
            (status,),
        ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_recommendation(rec_id: str) -> dict | None:
    conn = get_conn()
    row = conn.execute(
        "SELECT * FROM recommendations WHERE id=?", (rec_id,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def set_recommendation_status(rec_id: str, status: str) -> bool:
    """Transition one rec's status. Returns False if the id does not exist
    (so the API layer can 404). Unknown status strings are rejected."""
    if status not in _REC_STATUSES:
        raise ValueError(f"unknown recommendation status: {status!r}")
    conn = get_conn()
    try:
        cur = conn.execute(
            "UPDATE recommendations SET status=? WHERE id=?", (status, rec_id)
        )
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()


def recommended_titles() -> set[str]:
    """Every title ever recommended, in ANY status -- the exclude set so a book
    is never re-suggested once it has been shown (design §B 'exclude')."""
    conn = get_conn()
    rows = conn.execute("SELECT DISTINCT title FROM recommendations").fetchall()
    conn.close()
    return {r["title"] for r in rows if r["title"]}


def recommendation_stats() -> dict[str, int]:
    """Counts GROUP BY status -- the MVP eval (add-rate = added/total). No
    LLM-judge harness; recs have a free in-product signal (add vs dismiss)."""
    conn = get_conn()
    rows = conn.execute(
        "SELECT status, COUNT(*) AS n FROM recommendations GROUP BY status"
    ).fetchall()
    conn.close()
    return {r["status"]: r["n"] for r in rows}
