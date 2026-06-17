'use client'
import { useEffect, useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { X, BookOpen, Loader2 } from 'lucide-react'
import { getCitation, ApiError } from '@/lib/api'
import type { CitationPayload } from '@/lib/types'

/**
 * Minimal "show me the source" modal (redesigned 2026-06-10).
 *
 * Renders exactly what the attribution system can honestly claim: the
 * retrieved chunk's text, in one uniform style, plus a section/page
 * line and the chunk id. The old ±2-block window with a highlighted
 * "primary block" was removed on purpose — attribution resolves to
 * CHUNK granularity, and the amber block highlight implied a per-fact
 * block-level precision that was never computed (primary_block_id is
 * a chunking-time artifact, not per-fact evidence).
 *
 * One request (GET /api/books/{id}/citations/{chunk_id}); no follow-up
 * block fetch. Failure modes degrade to a friendly message:
 *   409 → legacy (v1) book, no canonical layer
 *   404 → unknown / legacy chunk id
 */
interface Props {
  bookId: string
  chunkOrNodeId: string
  onClose: () => void
}

export default function CitationModal({ bookId, chunkOrNodeId, onClose }: Props) {
  const [citation, setCitation] = useState<CitationPayload | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    let cancelled = false
    setLoading(true)
    setError(null)
    setCitation(null)

    ;(async () => {
      try {
        const payload = await getCitation(bookId, chunkOrNodeId)
        if (!cancelled) setCitation(payload)
      } catch (e) {
        if (cancelled) return
        const status = e instanceof ApiError ? e.status : null
        if (status === 409) {
          setError(
            'This book is on the legacy index. Source view is only available for v2 (canonical) books.'
          )
        } else if (status === 404) {
          setError('Could not locate the source for this citation.')
        } else {
          setError('Failed to load source. Please retry.')
        }
      } finally {
        if (!cancelled) setLoading(false)
      }
    })()

    return () => {
      cancelled = true
    }
  }, [bookId, chunkOrNodeId])

  // Close on Escape
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])

  const titleLabel = citation?.section_label || 'Source'
  const isSummary = citation?.evidence.retrieved_from.layer === 'raptor'
  const page = (citation?.source_locator as { page?: number } | undefined)?.page
  // Summary nodes span a whole chapter — a page number would mislead.
  const subtitle = isSummary
    ? 'AI-generated chapter summary — not the book’s own words'
    : typeof page === 'number'
    ? `Page ${page}`
    : null

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm p-4"
        onClick={onClose}
      >
        <motion.div
          initial={{ opacity: 0, scale: 0.96, y: 8 }}
          animate={{ opacity: 1, scale: 1, y: 0 }}
          exit={{ opacity: 0, scale: 0.96 }}
          transition={{ duration: 0.15 }}
          className="relative w-full max-w-2xl max-h-[80vh] overflow-hidden rounded-2xl border border-stone-700 bg-stone-950 shadow-2xl flex flex-col"
          onClick={(e) => e.stopPropagation()}
        >
          {/* Title bar */}
          <div className="flex items-start gap-3 border-b border-stone-800 px-5 py-3 flex-shrink-0">
            <BookOpen size={16} className="mt-1 text-stone-400 flex-shrink-0" />
            <div className="flex-1 min-w-0">
              <p className="font-serif text-stone-200 text-sm truncate">{titleLabel}</p>
              {subtitle && (
                <p className="mt-0.5 text-[11px] uppercase tracking-wide text-stone-500">
                  {subtitle}
                </p>
              )}
            </div>
            <button
              onClick={onClose}
              className="text-stone-500 hover:text-stone-200 transition-colors flex-shrink-0"
              aria-label="Close"
            >
              <X size={18} />
            </button>
          </div>

          {/* Body — the retrieved passage, one uniform style */}
          <div className="flex-1 overflow-y-auto px-5 py-4">
            {loading && (
              <div className="flex items-center gap-2 text-stone-500 text-sm">
                <Loader2 size={14} className="animate-spin" />
                Resolving source…
              </div>
            )}

            {error && !loading && (
              <p className="text-stone-400 text-sm leading-relaxed">{error}</p>
            )}

            {citation && !loading && (
              <p className="whitespace-pre-wrap text-[14px] leading-7 text-stone-300">
                {citation.evidence.text || citation.evidence.snippet || '(no preview available)'}
              </p>
            )}
          </div>

          {/* Footer — the chunk id, nothing else */}
          {citation && !loading && (
            <div className="border-t border-stone-800 px-5 py-2.5 flex-shrink-0">
              <span className="font-mono text-[11px] text-stone-600">{chunkOrNodeId}</span>
            </div>
          )}
        </motion.div>
      </motion.div>
    </AnimatePresence>
  )
}
