import type {
  ChunkMapResponse,
  ChunkFullResponse,
  Message,
  CitationPayload,
  Recommendation,
  MemoryNote,
  SessionSummary,
} from './types'

// ?? (not ||): production builds set NEXT_PUBLIC_API_URL="" for same-origin
// requests through the nginx proxy — the empty string must survive.
const BASE = process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000'

/** Error carrying the HTTP status so callers can branch on it
 *  (e.g. 409 = legacy-index book) instead of substring-matching
 *  the message. */
export class ApiError extends Error {
  status: number
  constructor(status: number, message: string) {
    super(message)
    this.name = 'ApiError'
    this.status = status
  }
}

async function handleResponse(res: Response) {
  if (!res.ok) {
    const text = await res.text().catch(() => res.statusText)
    throw new ApiError(res.status, `API error ${res.status}: ${text}`)
  }
  return res.json()
}

export async function listBooks() {
  return handleResponse(await fetch(`${BASE}/api/books`))
}

export async function getBook(bookId: string) {
  return handleResponse(await fetch(`${BASE}/api/books/${bookId}`))
}

export function getBookContentUrl(bookId: string) {
  return `${BASE}/api/books/${bookId}/content`
}

export function getBookCoverUrl(bookId: string) {
  return `${BASE}/api/books/${bookId}/cover`
}

export async function deleteBook(bookId: string) {
  return handleResponse(
    await fetch(`${BASE}/api/books/${bookId}`, { method: 'DELETE' })
  )
}

export async function uploadBook(file: File, title?: string, author?: string) {
  const form = new FormData()
  form.append('file', file)
  if (title) form.append('title', title)
  if (author) form.append('author', author)
  return handleResponse(
    await fetch(`${BASE}/api/ingest`, { method: 'POST', body: form })
  )
}

export async function createSession(
  bookId: string
): Promise<{ session_id: string }> {
  return handleResponse(
    await fetch(`${BASE}/api/chat/session?book_id=${bookId}`, { method: 'POST' })
  )
}

export async function getSession(sessionId: string): Promise<{
  session_id: string
  book_id: string
  conversation: Message[]
}> {
  return handleResponse(await fetch(`${BASE}/api/chat/session/${sessionId}`))
}

export async function listSessions(
  bookId: string
): Promise<{ sessions: SessionSummary[] }> {
  return handleResponse(await fetch(`${BASE}/api/chat/sessions/${bookId}`))
}

export async function deleteSession(
  sessionId: string
): Promise<{ deleted: string }> {
  return handleResponse(
    await fetch(`${BASE}/api/chat/session/${sessionId}`, { method: 'DELETE' })
  )
}

export async function getProgress(bookId: string, sessionId: string) {
  return handleResponse(
    await fetch(`${BASE}/api/progress/${bookId}?session_id=${sessionId}`)
  )
}

// ── Recommendations (home "For you" block) ──────────────────────────────────

export async function getRecommendations(): Promise<Recommendation[]> {
  return handleResponse(await fetch(`${BASE}/api/recommendations`))
}

export async function refreshRecommendations(): Promise<Recommendation[]> {
  return handleResponse(
    await fetch(`${BASE}/api/recommendations/refresh`, { method: 'POST' })
  )
}

export async function addRecommendation(id: string) {
  return handleResponse(
    await fetch(`${BASE}/api/recommendations/${id}/add`, { method: 'POST' })
  )
}

export async function dismissRecommendation(id: string) {
  return handleResponse(
    await fetch(`${BASE}/api/recommendations/${id}/dismiss`, { method: 'POST' })
  )
}

export async function getSavedBooks(): Promise<Recommendation[]> {
  return handleResponse(await fetch(`${BASE}/api/recommendations/saved`))
}

export async function unsaveBook(id: string) {
  return handleResponse(
    await fetch(`${BASE}/api/recommendations/${id}/unsave`, { method: 'POST' })
  )
}

// ── Memory notes ("What I remember" panel) ───────────────────────────────────

export async function getMemoryNotes(): Promise<MemoryNote[]> {
  return handleResponse(await fetch(`${BASE}/api/memory-notes`))
}

export async function updateMemoryNote(
  id: string,
  text: string,
  type?: string
): Promise<MemoryNote> {
  return handleResponse(
    await fetch(`${BASE}/api/memory-notes/${id}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text, type }),
    })
  )
}

export async function deleteMemoryNote(id: string) {
  return handleResponse(
    await fetch(`${BASE}/api/memory-notes/${id}`, { method: 'DELETE' })
  )
}

export async function exportNotes(
  sessionId: string,
  format: 'pdf' | 'markdown'
): Promise<Blob> {
  const res = await fetch(`${BASE}/api/export`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ session_id: sessionId, format }),
  })
  if (!res.ok) throw new Error(`Export failed: ${res.status}`)
  return res.blob()
}

export async function getChunkMap(
  bookId: string,
  sessionId: string
): Promise<ChunkMapResponse> {
  return handleResponse(
    await fetch(`${BASE}/api/debug/books/${bookId}/chunk-map?session_id=${sessionId}`)
  )
}

export async function getChunkFull(
  bookId: string,
  chunkId: string
): Promise<ChunkFullResponse> {
  return handleResponse(
    await fetch(`${BASE}/api/debug/books/${bookId}/chunks/${encodeURIComponent(chunkId)}/full`)
  )
}

// ── Canonical (v2) endpoints ────────────────────────────────────────────────
// All four require books.ingest_version >= 2; the backend returns 409 for v1
// books so the caller can fall back to the legacy debug surface cleanly.

export async function getCitation(
  bookId: string,
  chunkOrNodeId: string
): Promise<CitationPayload> {
  return handleResponse(
    await fetch(
      `${BASE}/api/books/${bookId}/citations/${encodeURIComponent(chunkOrNodeId)}`
    )
  )
}

