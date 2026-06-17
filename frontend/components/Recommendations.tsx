'use client'
import { useState, useEffect, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Sparkles, RefreshCw, BookmarkPlus, X } from 'lucide-react'
import {
  getRecommendations,
  refreshRecommendations,
  addRecommendation,
  dismissRecommendation,
} from '@/lib/api'
import { Recommendation } from '@/lib/types'

/**
 * The home "For you" block (memory-system-design.md §B).
 *
 * Calm + pull-discovered (not pushed): it sits quietly below the shelf, fetched
 * once on mount. Each card ties its reason to a specific interest; "Want to
 * read" / "Dismiss" are optimistic status transitions, "Shuffle" retires the
 * batch and asks for a fresh one. Renders nothing on a cold/empty result so the
 * page stays quiet.
 */
export default function Recommendations({ onAdd }: { onAdd?: () => void }) {
  const [recs, setRecs] = useState<Recommendation[] | null>(null)
  const [loading, setLoading] = useState(true)
  const [refreshing, setRefreshing] = useState(false)
  const [everHadRecs, setEverHadRecs] = useState(false)

  useEffect(() => {
    let alive = true
    getRecommendations()
      .then((data) => {
        if (!alive) return
        setRecs(data)
        if (data.length > 0) setEverHadRecs(true)
      })
      .catch(() => alive && setRecs([])) // backend down -> stay quiet
      .finally(() => alive && setLoading(false))
    return () => {
      alive = false
    }
  }, [])

  const handleRefresh = useCallback(async () => {
    setRefreshing(true)
    try {
      const data = await refreshRecommendations()
      setRecs(data)
      if (data.length > 0) setEverHadRecs(true)
    } catch {
      // leave the current cards in place on failure
    } finally {
      setRefreshing(false)
    }
  }, [])

  // Optimistic: remove the card immediately; tell the parent once the status
  // flip persists so the "Want to read" list can re-pull and show it.
  const handleAdd = useCallback(
    (id: string) => {
      setRecs((cur) => (cur ? cur.filter((r) => r.id !== id) : cur))
      addRecommendation(id)
        .then(() => onAdd?.())
        .catch(() => {})
    },
    [onAdd]
  )

  const handleDismiss = useCallback((id: string) => {
    setRecs((cur) => (cur ? cur.filter((r) => r.id !== id) : cur))
    dismissRecommendation(id).catch(() => {})
  }, [])

  // Cold start / never had anything to show -> render nothing (the empty-shelf
  // state is the onboarding nudge; this block only appears once it has value).
  if (loading) {
    return (
      <div className="mt-12 text-sm text-[var(--soft-foreground)]/70 animate-pulse">
        Finding your next read…
      </div>
    )
  }
  if (!recs || (recs.length === 0 && !everHadRecs)) return null

  return (
    <section className="mt-12">
      {/* Header — calm, with Shuffle on the right */}
      <div className="mb-5 flex items-end justify-between gap-4">
        <div>
          <h2 className="flex items-center gap-2 font-serif text-2xl text-amber-50">
            <Sparkles size={18} strokeWidth={1.6} className="text-amber-200/80" />
            For you
          </h2>
          <p className="mt-1.5 text-sm leading-6 text-[var(--muted-foreground)]">
            Where your reading might wander next.
          </p>
        </div>
        <button
          onClick={handleRefresh}
          disabled={refreshing}
          className="inline-flex shrink-0 items-center gap-1.5 rounded-full border border-amber-200/15 bg-amber-200/[0.04] px-3.5 py-2 text-xs text-[var(--soft-foreground)] transition-colors hover:border-amber-200/25 hover:text-amber-100 disabled:opacity-50"
        >
          <RefreshCw
            size={13}
            className={refreshing ? 'animate-spin' : ''}
            strokeWidth={1.8}
          />
          Shuffle
        </button>
      </div>

      {recs.length === 0 ? (
        <div className="reading-surface rounded-[1.5rem] px-6 py-10 text-center text-sm text-[var(--muted-foreground)]">
          That&apos;s all for now — Shuffle for more.
        </div>
      ) : (
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
          <AnimatePresence mode="popLayout">
            {recs.map((rec) => (
              <RecCard
                key={rec.id}
                rec={rec}
                onAdd={handleAdd}
                onDismiss={handleDismiss}
              />
            ))}
          </AnimatePresence>
        </div>
      )}
    </section>
  )
}

function RecCard({
  rec,
  onAdd,
  onDismiss,
}: {
  rec: Recommendation
  onAdd: (id: string) => void
  onDismiss: (id: string) => void
}) {
  return (
    <motion.div
      layout
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, scale: 0.96 }}
      transition={{ type: 'spring', stiffness: 300, damping: 26 }}
      className="reading-surface flex flex-col rounded-[1.4rem] p-5"
    >
      {/* which_interest — the grounding, shown as a quiet pill */}
      {rec.which_interest && (
        <span className="mb-3 inline-flex w-fit items-center rounded-full border border-amber-200/15 bg-amber-200/[0.08] px-2.5 py-0.5 text-[11px] text-amber-100/90">
          {rec.which_interest}
        </span>
      )}

      <h3 className="font-serif text-lg leading-snug text-amber-50 line-clamp-2">
        {rec.title}
      </h3>
      {rec.author && (
        <p className="mt-1 truncate text-xs text-[var(--muted-foreground)]">
          {rec.author}
        </p>
      )}

      {rec.reason && (
        <p className="mt-3 flex-1 text-sm leading-6 text-[var(--soft-foreground)] line-clamp-4">
          {rec.reason}
        </p>
      )}

      {/* Actions */}
      <div className="mt-5 flex items-center gap-2">
        <button
          onClick={() => onAdd(rec.id)}
          className="inline-flex flex-1 items-center justify-center gap-1.5 rounded-xl border border-amber-200/20 bg-amber-200/[0.10] px-3 py-2 text-xs font-medium text-amber-100 transition-colors hover:bg-amber-200/[0.18]"
        >
          <BookmarkPlus size={14} strokeWidth={1.8} />
          Want to read
        </button>
        <button
          onClick={() => onDismiss(rec.id)}
          aria-label="Dismiss this book"
          className="inline-flex items-center justify-center rounded-xl border border-amber-200/15 bg-amber-200/[0.04] px-3 py-2 text-xs text-[var(--soft-foreground)] transition-colors hover:border-amber-200/25 hover:text-amber-50"
        >
          <X size={14} strokeWidth={1.8} />
          Dismiss
        </button>
      </div>
    </motion.div>
  )
}
