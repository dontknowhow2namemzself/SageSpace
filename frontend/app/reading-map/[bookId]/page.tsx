'use client'
import { useState, useEffect, Suspense } from 'react'
import { useParams, useSearchParams, useRouter } from 'next/navigation'
import ChunkMapPanel from '@/components/reading-map/ChunkMapPanel'
import ChunkReader from '@/components/reading-map/ChunkReader'
import { getChunkMap, getBook } from '@/lib/api'
import { ChunkMapResponse, Book } from '@/lib/types'

/**
 * Reading Map: the dot-matrix map of the whole book (left) plus a
 * plain reader for the selected chunk (right). Lit squares are the
 * chunks the sage's answers actually cited this session (per-fact
 * data-chunk-ids) — i.e. what the user has read through conversation.
 *
 * The former middle/right debug columns (retrieval timeline, event
 * details, faithfulness, multi-query / HyDE internals) were removed:
 * LangSmith tracing covers all of that for development.
 */
function ReadingMapContent() {
  const { bookId } = useParams<{ bookId: string }>()
  const searchParams = useSearchParams()
  const router = useRouter()
  const sessionId = searchParams.get('session') || ''

  const [book, setBook] = useState<Book | null>(null)
  const [chunkMap, setChunkMap] = useState<ChunkMapResponse | null>(null)
  const [selectedChunkId, setSelectedChunkId] = useState<string | null>(null)

  useEffect(() => {
    getBook(bookId).then(setBook).catch(() => {})
    getChunkMap(bookId, sessionId).then(setChunkMap).catch(() => {})
  }, [bookId, sessionId])

  return (
    <div className="reading-map-background flex h-screen flex-col overflow-hidden">
      {/* Header — page-level frost handles the chrome treatment, so this
          row just needs a subtle bottom separator */}
      <div className="flex flex-shrink-0 items-center gap-4 border-b border-amber-100/10 px-5 py-3">
        <button
          onClick={() => router.back()}
          className="text-sm text-[var(--muted-foreground)] transition-colors hover:text-amber-100"
        >
          ←
        </button>
        <h1 className="font-serif text-base tracking-wide text-amber-50">
          {book?.title ? `${book.title} ` : ''}Reading Map
        </h1>
        <span className="ml-auto font-mono text-xs text-[var(--soft-foreground)]/70">
          session: {sessionId.slice(0, 8) || '—'}
        </span>
      </div>

      {/* Two-column body: map | reader */}
      <div className="flex flex-1 overflow-hidden">
        <div className="flex w-[38%] flex-col overflow-hidden border-r border-amber-100/[0.08]">
          <div className="border-b border-amber-100/[0.08] px-4 pt-3 pb-2 text-[11px] uppercase tracking-[0.22em] text-[var(--soft-foreground)]">
            Full Book Chunk Map
          </div>
          <ChunkMapPanel
            data={chunkMap}
            selectedChunkId={selectedChunkId}
            onSelectChunk={setSelectedChunkId}
          />
        </div>

        <div className="flex flex-1 flex-col overflow-hidden">
          <div className="border-b border-amber-100/[0.08] px-4 pt-3 pb-2 text-[11px] uppercase tracking-[0.22em] text-[var(--soft-foreground)]">
            Reader
          </div>
          <ChunkReader
            bookId={bookId}
            chunkId={selectedChunkId}
            chunkMap={chunkMap}
            onNavigate={setSelectedChunkId}
          />
        </div>
      </div>
    </div>
  )
}

export default function ReadingMapPage() {
  return (
    <Suspense fallback={<div className="reading-map-background h-screen" />}>
      <ReadingMapContent />
    </Suspense>
  )
}
