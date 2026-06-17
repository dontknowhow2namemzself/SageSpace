# SageSpace — Memory MVP (Fast Lane + Recommendations)

**Status:** Implemented & verified

The first slice of SageSpace's memory system. Two halves:

- **Fast lane** — silently capture the user's *explicitly stated* durable
  facts/interests from chat, with zero extra LLM calls.
- **Recommendations** — a home **"For you"** block that suggests real books
  grounded in those facts + the user's library + recent questions, plus a
  **"Want to read"** list and a **"What I remember"** panel.

Design philosophy (locked in the design memory): **low-interruption, quietly
delightful** — writes are silent, surfaces are pull-discovered on the home page,
nothing is pushed. The app is **English-facing** (UI + LLM-generated content in
English; the book-chat answer still mirrors the user's question language).

---

## 1. Data model

One SQLite file, `backend/sagespace.db` (plain `sqlite3`, no ORM; see
`core/database.py`). Two new tables:

```sql
memory_notes(
  id, text, type,              -- type = fact | interest
  source_book_id,              -- provenance; NULLed (not cascaded) on book delete
  source_locator,              -- the session_id it was captured in
  created_at
)

recommendations(
  id, title, author, blurb,    -- blurb = Google Books description (may be NULL)
  reason, which_interest,      -- LLM's pitch + the specific interest it ties to
  status,                      -- suggested | seen | added | dismissed
  created_at
)
```

- **No covers** — recommendation cards use the app's own visual.
- **`recommendations` has no `book_id`** → it is *not* part of the book-delete
  cascade; recs are user-level and persist (exclude-by-title still works).
- **`memory_notes` is the one book-referencing table deliberately NOT cascaded**:
  on book delete `source_book_id` is set NULL so a user-level fact ("call me
  Wang") outlives any single book.

> There is **no chat-messages table** — conversation history is a per-session
> JSON blob (`sessions.conversation_json`, a list of `{role, content}`, no
> per-message timestamps). This shaped `recent_user_questions` (see §3).

---

## 2. Part A — Fast lane (capture)

Per turn, the existing `classify_intent` LLM call (one call, gpt-4o-mini) also
emits `memory_note` + `memory_note_type` — **zero extra calls**. The note is
always written in **English** (translated if the user typed another language),
since it is shown verbatim in the "What I remember" panel.

- **Write point:** `finalize_node` (`core/graph/nodes.py`) → `db.add_memory_note`.
  Capturing at *finalize* means only a **committed** turn writes — a turn
  abandoned at a clarify interrupt writes nothing.
- **Guards:** empty/whitespace skipped; **dedup on exact text** (the same fact
  said twice doesn't pile up). Best-effort — a memory write never breaks a turn.
- **Most turns capture nothing** (the classifier returns null unless there's a
  clear, durable self-statement).

Touched: `core/pipeline/intent.py` (schema + prompt), `core/pipeline/types.py`
(`IntentDecision`), `core/graph/nodes.py` (`_capture_memory_note`),
`core/database.py`.

---

## 3. Part B — Recommendations

### Signals & priority (`core/recommend.py` → `recommend()`)

Interests are assembled and de-duped in this order — a **soft** priority (list
order + a "most useful first" line in the prompt; the LLM does the final
weighing, there are no hard weights):

1. **`memory_notes`** — what the user *said* (highest)
2. **library titles** — what they *own* (also the exclude floor)
3. **recent questions** — what they've *shown*

**Recent-question filtering** — `recent_user_questions` returns raw chat turns,
which include procedural/smalltalk noise that out-numbers real signals. So we
pull a wider pool (40) and keep only **substantive** questions (≤15) via
`_is_interest_question`: drop fragments (too short), reading-progress / export /
greeting / "summarize this book" patterns (EN + ZH regex). *(A cleaner future
version would persist each turn's `intent.kind` and filter by that — see §7.)*

**The 15-question window is live**, not a one-time snapshot: it's recomputed on
every `recommend()` run, so it's a sliding window of the user's latest activity.
(Displayed recs are cached as `suggested` rows; they refresh on **Shuffle** or
when the set is consumed — there is no auto-recompute on every chat message.)

### Generation & validation

```
interests + exclude  ─►  llm_recommend (gpt-4o-mini, structured)
                          · 1 forced "stretch" / cross-genre pick
                          · each reason MUST tie to a specific interest (English)
                     ─►  books_api.lookup(title, author)  per pick
                     ─►  insert as status='suggested'
exclude = every recommended title (any status) ∪ library titles
```

**Catalog = Google Books** (`core/books_api.py`, metadata only, pluggable
`BookMetadataSource`). Behavior:

- **Validates the LLM's pick.** A real book → store Google's **canonical**
  title/author/blurb (so a slightly-off LLM title gets corrected).
- **Hallucination guard:** a clean *not-found* (HTTP 200, no match) → `None` →
  the pick is **dropped** (so the batch can be < 3).
- **Graceful degrade:** *couldn't reach* the catalog (429 / network / bad
  payload) → raises `BookLookupUnavailable` → the pick is **kept** with the
  LLM's title/author and no blurb (a failed *validation* is not evidence the
  book is fake). The guard is therefore active only while Google is reachable.
- **Best-of-top-5 selection** (not blindly `items[0]`): score each candidate by
  title closeness + author match − a **knockoff penalty** (Summary / Study Guide
  / Workbook / SparkNotes…), with a **has-description tie-break** so equal
  matches prefer the edition that actually carries a blurb.

`GOOGLE_BOOKS_API_KEY` (in `backend/.env`) lifts the quota to ~1000/day; without
it the keyless endpoint 429s almost immediately and recs degrade (no blurbs).

### Status lifecycle & eval

`suggested` → `seen` (Shuffle) | `added` (Want to read) | `dismissed` (Ignore).
`unsave` returns `added` → `seen`. Rows are **never deleted**, so the eval
denominator holds: **eval = `GROUP BY status`** (add-rate = added/total), exposed
at `GET /api/recommendations/stats` — recs carry a free in-product
quality signal (add vs dismiss), no offline harness needed.

---

## 4. API

```
# Memory notes (api/memory.py)
GET    /api/memory-notes                  list, newest first
PATCH  /api/memory-notes/{id}             edit text (and optionally type)
DELETE /api/memory-notes/{id}             forget a note

# Recommendations (api/recommend.py)
GET    /api/recommendations               current 'suggested' (lazy-compute if none)
GET    /api/recommendations/saved         the Want-to-read list (status='added')
GET    /api/recommendations/stats         GROUP BY status (the eval)
POST   /api/recommendations/refresh       Shuffle: suggested -> seen, recompute
POST   /api/recommendations/{id}/add      Want to read  -> added
POST   /api/recommendations/{id}/dismiss  Ignore        -> dismissed
POST   /api/recommendations/{id}/unsave   remove from list -> seen
```

---

## 5. Frontend (home page, below the shelf, only when the library is non-empty)

- **`components/Recommendations.tsx`** — "For you": 3 cards (interest pill ·
  title · author · grounded reason) with *Want to read* / *Dismiss*, plus
  *Shuffle*. Optimistic UI.
- **`components/WantToReadList.tsx`** — collapsible list of added books;
  self-hides when empty; refreshes when a card is added.
- **`components/MemoryNotes.tsx`** — a quiet collapsible "What I remember" — the
  honest entry point to view / inline-edit / delete captured notes. (Silent
  writes are unchanged; this is read-and-correct only — a light, early take on
  the design's future management panel.)

Card readability: the `.reading-surface` class (`app/globals.css`) gives cards an
opaque walnut backing over the busy hero painting — **tune `--card-opacity`**.
Focus rings: `:focus:not(:focus-visible){outline:none}` drops the mouse-click
outline app-wide while keeping it for keyboard navigation.

---

## 6. Run & verify

- **Backend:** `cd backend && python -m pytest -q` — offline, no API key.
  (`GOOGLE_BOOKS_API_KEY` in `backend/.env` for non-degraded recs.)
- **Frontend:** `cd frontend && npm run dev` → home page; `npx tsc --noEmit` +
  `npx next lint` clean.
- Tests: `test_memory_db`, `test_memory_api`, `test_recommend`,
  `test_recommend_db`, `test_recommend_api`, `test_books_api`, plus extensions to
  `test_pipeline_intent` / `test_database`.

---

## 7. Deferred / future (status-marked so it isn't re-derived)

- 🔮 **`load_memory` (chat personalization)** — inject relevant notes into
  `synthesize` as soft reader-context ("as I recall you mentioned…"), not cited
  as book facts. Currently notes shape **recommendations only**, not chat
  answers. This is the natural next step for "the sage remembers me".
- 🔵 **Cleaner question filtering** — persist each turn's `intent.kind` and keep
  only search/book_overview, instead of the text-heuristic in §3. Also near-dup
  removal of similar questions.
- 🔮 **Hard English guarantee** — translate non-English *signals* before the
  recommend call (a Chinese raw chat question can still echo into
  `which_interest` non-deterministically; rare, dev-testing only).
- 🔵 **Display the stored blurb** on the card (e.g. expand-for-details); it is
  fetched and stored today but not shown.
- 🔵 **Slow lane / reflect**, 🔮 **Connections** — shelved/undecided; the
  latter is where note embeddings would finally appear (memory notes are
  plain text today; the only vector store in the project is ChromaDB for
  **book** retrieval, not memory).
