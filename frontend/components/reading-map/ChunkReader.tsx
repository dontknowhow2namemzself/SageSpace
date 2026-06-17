'use client'
import { useEffect, useMemo, useState } from 'react'
import { ChevronLeft, ChevronRight } from 'lucide-react'
import { ChunkFullResponse, ChunkMapResponse } from '@/lib/types'
import { getChunkFull } from '@/lib/api'
import { groupHeader } from './ChunkMapPanel'

interface Props {
  bookId: string
  chunkId: string | null
  chunkMap: ChunkMapResponse | null
  onNavigate: (chunkId: string) => void
}

interface FlatChunk {
  chunkId: string
  header: string
  page: number
}

/** Cache full-chunk fetches across selections within a single page view. */
const fullChunkCache = new Map<string, ChunkFullResponse>()

/**
 * The right pane of the Reading Map: a plain reader for the selected
 * chunk. Click a square on the map → the chunk's full text opens here,
 * with prev/next (and ←/→ keys) flowing through the book in display
 * order — the map and the reader stay in sync via onNavigate.
 */
export default function ChunkReader({
  bookId, chunkId, chunkMap, onNavigate,
}: Props) {
  const [data, setData] = useState<ChunkFullResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // Flatten the sectioned grid into reading order so prev/next can
  // cross section boundaries. chunkMap groups arrive sorted by
  // canonical order_idx; chunks within a group are in book order.
  const flat = useMemo<FlatChunk[]>(() => {
    if (!chunkMap) return []
    return chunkMap.chapters.flatMap((group) =>
      group.chunks.map((c) => ({
        chunkId: c.chunk_id,
        header: groupHeader(group),
        page: c.page,
      })),
    )
  }, [chunkMap])

  const index = useMemo(
    () => flat.findIndex((c) => c.chunkId === chunkId),
    [flat, chunkId],
  )
  const current = index >= 0 ? flat[index] : null
  const prev = index > 0 ? flat[index - 1] : null
  const next = index >= 0 && index < flat.length - 1 ? flat[index + 1] : null

  useEffect(() => {
    if (!chunkId) {
      setData(null)
      setError(null)
      return
    }
    const cached = fullChunkCache.get(`${bookId}::${chunkId}`)
    if (cached) {
      setData(cached)
      setError(null)
      return
    }
    let cancelled = false
    setLoading(true)
    setError(null)
    setData(null)
    getChunkFull(bookId, chunkId)
      .then((res) => {
        fullChunkCache.set(`${bookId}::${chunkId}`, res)
        if (!cancelled) setData(res)
      })
      .catch((err) => {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : 'Failed to load chunk')
        }
      })
      .finally(() => {
        if (!cancelled) setLoading(false)
      })
    return () => { cancelled = true }
  }, [bookId, chunkId])

  // ←/→ page through the book like turning leaves.
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === 'ArrowLeft' && prev) onNavigate(prev.chunkId)
      if (e.key === 'ArrowRight' && next) onNavigate(next.chunkId)
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [prev, next, onNavigate])

  if (!chunkId) {
    return (
      <div className="flex h-full items-center justify-center p-8 text-center">
        <div className="max-w-xs font-serif italic text-[var(--soft-foreground)]/70">
          Pick a square on the map to read that passage here.
        </div>
      </div>
    )
  }

  return (
    <div className="flex h-full flex-col overflow-hidden">
      {/* Passage header */}
      <div className="flex flex-shrink-0 items-baseline justify-between gap-3 border-b border-amber-100/[0.08] px-6 py-3">
        <div className="min-w-0">
          <div className="truncate font-serif text-sm text-amber-50">
            {current?.header ?? '…'}
          </div>
          <div className="mt-0.5 text-[11px] text-[var(--soft-foreground)]">
            Page {current?.page ?? data?.page ?? '—'}
            {data ? ` · ${data.char_length} chars` : ''}
          </div>
        </div>
        {index >= 0 && (
          <div className="flex-shrink-0 font-mono text-[11px] text-[var(--soft-foreground)]/70">
            {index + 1} / {flat.length}
          </div>
        )}
      </div>

      {/* Passage body — key remounts the scroll container per chunk so
          the reading position resets to the top on navigation. */}
      <div
        key={chunkId}
        className="scrollbar-warm flex-1 overflow-y-auto px-6 py-5"
      >
        {loading && (
          <div className="font-serif italic text-[var(--soft-foreground)] animate-pulse">
            Turning the page…
          </div>
        )}
        {error && <div className="text-sm text-red-400">{error}</div>}
        {data && (
          <div className="mx-auto max-w-prose">
            <div className="whitespace-pre-wrap font-serif text-[15px] leading-loose text-amber-50/90">
              {data.full_text}
            </div>
            {data.truncated && (
              <div className="mt-3 text-[11px] italic text-[var(--soft-foreground)]/70">
                Truncated — original length {data.char_length} chars.
              </div>
            )}
          </div>
        )}
      </div>

      {/* Prev / next navigation */}
      <div className="flex flex-shrink-0 items-center justify-between border-t border-amber-100/[0.08] px-4 py-2.5">
        <button
          type="button"
          disabled={!prev}
          onClick={() => prev && onNavigate(prev.chunkId)}
          className="inline-flex items-center gap-1 rounded-md px-2 py-1 text-xs text-amber-300/80 transition-colors hover:text-amber-100 disabled:cursor-default disabled:opacity-30"
        >
          <ChevronLeft size={14} /> Previous
        </button>
        <span className="text-[10px] uppercase tracking-[0.18em] text-[var(--soft-foreground)]/50">
          ← / → to turn
        </span>
        <button
          type="button"
          disabled={!next}
          onClick={() => next && onNavigate(next.chunkId)}
          className="inline-flex items-center gap-1 rounded-md px-2 py-1 text-xs text-amber-300/80 transition-colors hover:text-amber-100 disabled:cursor-default disabled:opacity-30"
        >
          Next <ChevronRight size={14} />
        </button>
      </div>
    </div>
  )
}
