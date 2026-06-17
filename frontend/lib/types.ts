export interface SourceReference {
  label: string
  chapter?: number
  page?: number
  text: string
  chunk_id?: string
}

/** One card in the home "For you" block (memory-system-design.md §B). A book the
 *  user does NOT own, validated against Google Books, with a reason grounded in
 *  a specific interest. No cover_url — the card uses the app's own visual. */
export interface Recommendation {
  id: string
  title: string
  author: string | null
  blurb: string | null
  reason: string | null
  which_interest: string | null
  status: 'suggested' | 'seen' | 'added' | 'dismissed' | string
  created_at: string
}

/** One captured user fact/interest (memory-system-design.md §A). Normally
 *  invisible "fuel" for recommendations, surfaced in the home "What I remember"
 *  panel so the user can see/correct what's stored. */
export interface MemoryNote {
  id: string
  text: string
  type: string
  source_book_id: string | null
  source_locator: string | null
  created_at: string
}

export interface Book {
  id: string
  title: string
  author: string | null
  file_path?: string | null
  total_chunks: number | null
  total_chapters: number | null
  upload_date: string
  raptor_status: 'pending' | 'building' | 'ready' | string
  digested_pct: number
  /** Backend-relative path (e.g. "/api/books/<id>/cover") when a generated
   *  cover exists. Frontend prefixes NEXT_PUBLIC_API_URL when rendering. */
  cover_url?: string | null
}

export interface ProgressData {
  /** Reader-facing: % of the book's passages CITED by this session's
   *  answers (per-fact attribution) — same semantics as the Reading
   *  Map and the shelf %. */
  digested_pct: number
  cited_chunks: number
  total_chunks: number
  token_stats: {
    tokens_in: number
    tokens_out: number
    cost_usd: number
  }
  chapter_clusters?: ChapterCluster[]
  last_retrieval?: { event_id: string; query_text: string; newly_lit_count: number } | null
}

export interface SessionSummary {
  id: string
  book_id: string
  /** ISO 8601 (UTC) — sessions are listed newest-first by this. */
  start_time: string
  message_count: number
  /** First user question of the conversation, truncated server-side. */
  preview: string
  total_tokens_in: number
  total_tokens_out: number
  total_cost_usd: number
}

export interface Message {
  role: 'user' | 'assistant'
  content: string
  /** True while the assistant turn is in-flight but no tokens have
   *  arrived yet. Used to render the "Sage is turning the page…"
   *  placeholder instead of an empty bubble. Cleared on first non-
   *  empty token. */
  pending?: boolean
  /** A small system banner shown above the answer (PR3): the
   *  clarify-expired notice when a resume arrives after the 30-min
   *  window and the turn safe-degrades to a broad search. */
  notice?: string
}

export interface AnswerFactAttribution {
  fact_id: string
  text: string
  chunk_ids: string[]
  retrieval_event_ids: string[]
}

export interface AnswerAttribution {
  retrieval_event_ids: string[]
  chunk_ids: string[]
  raptor_ids: string[]
  facts: AnswerFactAttribution[]
}

export interface ChatEvent {
  type:
    | 'token'
    | 'tool_start'
    | 'tool_end'
    | 'done'
    | 'error'
    | 'retrieval_update'
    | 'answer_attribution'
    | 'stream_end'
    | 'ask_user'      // PR3: clarify interrupt — pause for a user answer
    | 'notice'        // PR3: system banner (clarify expired / nothing to resume)
    | 'fanout_start'  // PR4: compound question split into parallel sub-question searches
    | 'fanout_end'    // PR4: parallel searches done
  content?: string
  tool?: string
  // ask_user (clarify interrupt) fields
  prompt?: string
  options?: string[]
  multi?: boolean
  // notice fields
  kind?: string
  message?: string
  // fanout_start field
  subquestions?: string[]
  session_lit_chunks?: number
  total_chunks?: number
  newly_lit_count?: number
  newly_lit_chunk_ids?: string[]
  chapter_clusters?: ChapterCluster[]
  sources?: SourceReference[]
  retrieval_event_ids?: string[]
  chunk_ids?: string[]
  raptor_ids?: string[]
  facts?: AnswerFactAttribution[]
}

export interface ChapterCluster {
  chapter: number
  total: number
  lit: number
}

export interface RetrievalUpdateEvent {
  type: 'retrieval_update'
  session_lit_chunks: number
  total_chunks: number
  newly_lit_count: number
  newly_lit_chunk_ids: string[]
  chapter_clusters: ChapterCluster[]
  sources?: SourceReference[]
}

export interface ChunkDetail {
  chunk_id: string
  page: number
  char_length: number
  is_lit: boolean
  first_lit_event_id: string | null
  preview_text: string
}

export interface ChapterChunks {
  chapter: number          // legacy mirror = section.order_idx + 1
  chunks: ChunkDetail[]
  // Canonical section info (PR8). Present for every group on books
  // ingested under v2; the legacy `chapter` field is the fallback when
  // these are not populated.
  section_id?: string
  section_label?: string
  kind?:
    | 'cover' | 'titlepage' | 'toc' | 'preface' | 'foreword'
    | 'introduction' | 'front_matter'
    | 'prologue' | 'chapter' | 'epilogue'
    | 'afterword' | 'appendix' | 'glossary' | 'index' | 'bibliography'
    | 'back_matter' | 'other'
  printed_number?: number | null
  order_idx?: number
}

export interface ChunkMapResponse {
  total_chunks: number
  chapters: ChapterChunks[]
}

export interface ChunkFullResponse {
  chunk_id: string
  chapter: number
  page: number
  raptor_level: number
  char_length: number
  full_text: string
  truncated: boolean
}

// ── Canonical text layer (ingest_version=2) ─────────────────────────────────

export interface CanonicalBlock {
  block_id: string
  book_id: string
  order_idx: number
  kind: string
  text: string
  book_offset_start: number
  book_offset_end: number
  section_id: string | null
  locator_type: 'pdf' | 'epub' | string
  locator: Record<string, unknown>
  norm_flags: Record<string, unknown>
}

export interface CanonicalSection {
  section_id: string
  book_id: string
  parent_section_id: string | null
  order_idx: number
  level: number
  label: string
  source: 'outline' | 'toc' | 'heading' | 'inferred' | string
}

export interface BlocksResponse {
  blocks: CanonicalBlock[]
  next_cursor: number | null
}

export interface CitationPayload {
  book_id: string
  section_id: string | null
  section_label: string | null
  anchor: {
    primary_block_id: string
    block_ids: string[]
  }
  source_locator: Record<string, unknown>
  evidence: {
    snippet: string
    /** Full evidence text: the chunk's own text for raw hits, the
     *  primary block's text for raptor hits. What the popup renders. */
    text: string
    retrieved_from: {
      layer: 'raw' | 'raptor'
      node_or_chunk_id: string
      raptor_level: number
    }
  }
}
