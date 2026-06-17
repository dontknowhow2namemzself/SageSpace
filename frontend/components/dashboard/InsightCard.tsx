'use client'
import { useState } from 'react'
import Link from 'next/link'
import { motion, AnimatePresence } from 'framer-motion'
import { ChevronDown } from 'lucide-react'
import { ProgressData } from '@/lib/types'

/**
 * Single unified card for the chat sidebar — merges the old
 * ProgressPanel + TokenStats into one surface so the sidebar reads
 * as one thought, not two stacked dashboards.
 *
 * Hierarchy:
 *   1. Reading progress (big %, bar, passage count) — the answer to
 *      "how far am I" that a reader actually cares about
 *   2. Total cost — humanized one-line summary
 *   3. Token detail — collapsed by default; for the curious / debugging,
 *      not in the primary read
 *
 * Designed natively for the ~260-320px sidebar width — no transform-
 * scale hack like the previous EmbeddedDashboard.
 */
export default function InsightCard({
  bookId,
  sessionId,
  progress,
  onHide,
}: {
  bookId: string
  sessionId?: string
  progress: ProgressData | null
  /** When provided, renders a "‹ Hide panel" text control in the card's
   *  title row (taking the "This session" slot, mirroring the Reading
   *  Map link's style on the right) that collapses the sidebar. */
  onHide?: () => void
}) {
  const [showTokenDetail, setShowTokenDetail] = useState(false)

  // Styled identically to the Reading Map link on the opposite side of
  // the title row — a quiet text control, not a boxed chip.
  const hideButton = onHide ? (
    <button
      type="button"
      onClick={onHide}
      className="inline-flex items-center gap-1.5 text-[11px] uppercase tracking-[0.22em] text-[var(--soft-foreground)] transition-colors hover:text-amber-200"
    >
      <span className="text-sm leading-none">‹</span>
      Hide panel
    </button>
  ) : null

  // Empty / loading state — no session activity yet.
  if (!progress) {
    return (
      <div className="glass-panel-soft rounded-2xl px-5 py-6">
        {hideButton && <div className="mb-4">{hideButton}</div>}
        <p className="text-center text-sm leading-relaxed text-[var(--soft-foreground)]">
          Your session insights will appear here once you start the conversation.
        </p>
      </div>
    )
  }

  const pct = progress.digested_pct
  const cost = progress.token_stats.cost_usd
  const tokensIn = progress.token_stats.tokens_in
  const tokensOut = progress.token_stats.tokens_out

  // Humanize the cost line. Sub-cent reads as "a fraction of a cent"
  // (the dev who built this is showing it as $0.0003); single-cent reads
  // as "X cents"; anything ≥ $0.01 reads as a plain dollar amount.
  const costCopy =
    cost < 0.005
      ? 'less than a cent so far'
      : cost < 0.05
      ? `~${(cost * 100).toFixed(1)} cents so far`
      : `~$${cost.toFixed(3)} so far`

  const readingMapHref = sessionId
    ? `/reading-map/${bookId}?session=${sessionId}`
    : undefined

  return (
    <div className="glass-panel rounded-2xl p-5">
      {/* Title row — "Hide panel" sits in the old "This session" slot,
          visually paired with the Reading Map link on the right. */}
      <div className="mb-5 flex items-center justify-between gap-3">
        {hideButton ?? (
          <h3 className="font-serif text-base text-amber-50">This session</h3>
        )}
        {readingMapHref && (
          <Link
            href={readingMapHref}
            className="text-[11px] uppercase tracking-[0.22em] text-[var(--soft-foreground)] transition-colors hover:text-amber-200"
          >
            Reading Map →
          </Link>
        )}
      </div>

      {/* Progress — big % + bar + humanized passage count */}
      <div className="mb-6">
        <div className="mb-2 flex items-baseline justify-between gap-3">
          <span className="font-mono text-4xl font-light text-amber-200">
            {pct.toFixed(0)}
            <span className="ml-0.5 text-xl text-amber-200/70">%</span>
          </span>
          <span className="text-[11px] uppercase tracking-[0.22em] text-[var(--soft-foreground)]">
            explored
          </span>
        </div>
        <div className="h-1.5 overflow-hidden rounded-full bg-black/30">
          <div
            className="h-full rounded-full bg-gradient-to-r from-amber-300 via-amber-400 to-amber-500 transition-all duration-700"
            style={{ width: `${Math.min(pct, 100)}%` }}
          />
        </div>
        <p className="mt-3 text-sm leading-relaxed text-[var(--muted-foreground)]">
          {progress.cited_chunks} of {progress.total_chunks} passages cited
          in this conversation’s answers.
        </p>
      </div>

      {/* Secondary lines — session time + cost. Divided from the
          progress block above by a warm hairline (amber-tinted, low
          alpha) so it reads as a soft visual seam rather than a cold
          chrome line. */}
      <div className="space-y-1.5 border-t border-amber-100/10 pt-4 text-sm leading-relaxed text-[var(--muted-foreground)]">
        <p>
          This conversation:{' '}
          <span className="font-mono text-amber-100/80">{costCopy}</span>.
        </p>
      </div>

      {/* Token detail — collapsed by default */}
      <div className="mt-4">
        <button
          type="button"
          onClick={() => setShowTokenDetail((v) => !v)}
          className="flex items-center gap-1.5 text-[11px] uppercase tracking-[0.22em] text-[var(--soft-foreground)] transition-colors hover:text-amber-200"
          aria-expanded={showTokenDetail}
        >
          <ChevronDown
            size={12}
            className={`transition-transform duration-200 ${
              showTokenDetail ? 'rotate-180' : ''
            }`}
          />
          Token detail
        </button>

        <AnimatePresence initial={false}>
          {showTokenDetail && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
              transition={{ duration: 0.18 }}
              className="overflow-hidden"
            >
              <div className="mt-3 space-y-1.5 border-l border-amber-100/[0.08] pl-3 text-xs">
                <div className="flex justify-between font-mono text-[var(--soft-foreground)]">
                  <span>Input</span>
                  <span>{tokensIn.toLocaleString()}</span>
                </div>
                <div className="flex justify-between font-mono text-[var(--soft-foreground)]">
                  <span>Output</span>
                  <span>{tokensOut.toLocaleString()}</span>
                </div>
                <div className="flex justify-between border-t border-amber-100/10 pt-1.5 font-mono text-[var(--muted-foreground)]">
                  <span>Total</span>
                  <span>{(tokensIn + tokensOut).toLocaleString()}</span>
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

    </div>
  )
}
