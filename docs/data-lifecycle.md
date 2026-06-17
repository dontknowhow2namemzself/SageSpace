# Data lifecycle: what deletion actually deletes

Reference for the two destructive operations in SageSpace. Verified
against the code on 2026-06-10; source of truth is listed per section —
if this doc and the code disagree, the code wins (and this doc should
be updated in the same commit).

## Deleting a book — `DELETE /api/books/{book_id}`

Source of truth: `backend/api/books.py` (endpoint) and
`backend/core/database.py::delete_book` (SQL cascade, single
transaction). LangGraph checkpoint GC runs BEFORE the session rows are
deleted (it enumerates them): `backend/core/graph/build.py::gc_checkpoints_for_book`.

### Removed permanently (local, unrecoverable)

| Data | Where it lived | Notes |
|---|---|---|
| All conversations for the book | `sessions` (incl. `conversation_json`, token/cost totals) | Every session, not just the active one |
| LangGraph checkpoints | `sagespace_checkpoints.db` (`thread_id` = session id) | GC'd per thread before the rows go |
| Reading progress / lit map | `retrieved_chunks` + `cited_chunks` | Digested-% returns to nothing |
| Debug timeline | `retrieval_events` + `retrieval_event_chunks` | |
| Canonical text layer | `sections`, `blocks`, `blocks_fts`, `raptor_node_blocks`, `ingestion_reports` | |
| Vector index | ChromaDB collection `book_<id>` | Level-0 chunks + all RAPTOR summary nodes |
| Files on disk | `uploads/<id>.{epub,pdf}`, cover PNG | Best-effort removal |

### Intentionally kept

| Data | Why |
|---|---|
| `memory_notes` (user memory) | User-level facts survive book deletion; only `source_book_id` is nulled (provenance dropped, fact kept). See the comment in `delete_book`. |
| `recommendations` | User-level by design — carries no `book_id`; exclude-by-title still works after the delete. |
| Exported notes (`backend/exports/*.md` / `.pdf`) | Plain files; nothing tracks or removes them. |
| LangSmith traces | Cloud-side, keyed by `LANGCHAIN_PROJECT`. Local deletion cannot touch them; prune from the LangSmith console if needed. |
| Browser `localStorage` key `sagespace.session.<bookId>` | Stale but harmless — the chat page falls back to creating a session, and the shelf no longer links to the book. |

## Deleting one conversation — `DELETE /api/chat/session/{session_id}`

Source of truth: `backend/api/chat.py::delete_session`,
`backend/core/database.py::delete_session`,
`backend/core/graph/build.py::gc_checkpoints_for_session`.

Removed: the `sessions` row (conversation + token/cost stats), its
`retrieval_events` + `retrieval_event_chunks`, and the thread's
LangGraph checkpoints.

Kept **on purpose**: the session's `retrieved_chunks` (internal
"fed to the synthesizer" ledger) and `cited_chunks` (reader-facing
"cited by answers" ledger — what the shelf %, Insight panel, and
Reading Map count since 2026-06-10) rows. Reading progress is a
property of the *book*, so deleting a chat does not un-read the book.
The book-level percentage on the shelf is unchanged; only the
per-session view dies with the session.

Also untouched: `memory_notes` — a note captured during the session
survives, with `source_locator` still holding the now-deleted session
id (dangling provenance, accepted: the fact matters, its origin pointer
is best-effort). And the checkpoint DB *file*
(`sagespace_checkpoints.db`) itself is never removed — GC deletes
threads inside it, so an empty file persists after the last delete.
