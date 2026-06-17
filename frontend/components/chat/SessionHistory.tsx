'use client'
import { useCallback, useEffect, useState } from 'react'
import { ChevronDown, History, Trash2 } from 'lucide-react'
import { listSessions, deleteSession } from '@/lib/api'
import { SessionSummary } from '@/lib/types'

/** How many conversations show before "Show more". The full list (up to
 *  the server's 100-row cap) stays in memory; this only trims the DOM. */
const INITIAL_VISIBLE = 8

/** "Jun 8, 14:32" in the viewer's locale — enough to tell sessions apart
 *  without eating the row. */
function formatStart(iso: string): string {
  const d = new Date(iso)
  if (isNaN(d.getTime())) return ''
  return d.toLocaleString(undefined, {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  })
}

export default function SessionHistory({
  bookId,
  currentSessionId,
  refreshToken,
  onOpenSession,
  onDeleted,
}: {
  bookId: string
  currentSessionId: string | null
  /** Bump to refetch the list (new session created, turn finished). */
  refreshToken: number
  /** Open a past conversation; the page swaps session + messages. */
  onOpenSession: (sessionId: string) => void
  /** Called after a session row was deleted server-side; the page
   *  decides what to do when it was the active one. */
  onDeleted: (sessionId: string) => void
}) {
  const [sessions, setSessions] = useState<SessionSummary[]>([])
  const [expanded, setExpanded] = useState(false)

  const load = useCallback(async () => {
    try {
      const { sessions } = await listSessions(bookId)
      setSessions(sessions)
    } catch {
      // Sidebar is best-effort chrome — keep whatever we had.
    }
  }, [bookId])

  useEffect(() => {
    load()
  }, [load, refreshToken])

  const handleDelete = async (
    e: React.MouseEvent,
    session: SessionSummary
  ) => {
    e.stopPropagation()
    const label = session.preview
      ? `"${session.preview.slice(0, 60)}"`
      : 'this conversation'
    if (!confirm(`Delete ${label}? This cannot be undone.`)) return
    try {
      await deleteSession(session.id)
      setSessions((prev) => prev.filter((s) => s.id !== session.id))
      onDeleted(session.id)
    } catch {
      alert('Could not delete the conversation. Please try again.')
    }
  }

  // The server already excludes never-used sessions; a brand-new empty
  // conversation lives in page state, not in this history list.
  const visible = expanded ? sessions : sessions.slice(0, INITIAL_VISIBLE)
  const hiddenCount = sessions.length - INITIAL_VISIBLE

  return (
    <div className="glass-panel-soft rounded-2xl px-4 py-4">
      <div className="mb-3 flex items-center gap-2 px-1 text-xs uppercase tracking-[0.22em] text-[var(--soft-foreground)]">
        <History className="h-3.5 w-3.5 text-amber-300/60" />
        Past Conversations
      </div>

      {visible.length === 0 ? (
        <p className="px-1 pb-1 text-xs leading-relaxed text-[var(--soft-foreground)]/80">
          Conversations with the sage about this book will be kept here.
        </p>
      ) : (
        <ul className="space-y-1.5">
          {visible.map((s) => {
            const isCurrent = s.id === currentSessionId
            return (
              <li key={s.id}>
                <button
                  type="button"
                  onClick={() => !isCurrent && onOpenSession(s.id)}
                  className={`group relative w-full rounded-xl border px-3 py-2 text-left transition-colors ${
                    isCurrent
                      ? 'border-amber-300/30 bg-amber-900/15'
                      : 'border-white/[0.06] bg-black/15 hover:border-amber-300/20 hover:bg-white/[0.03]'
                  }`}
                >
                  <div
                    className={`truncate pr-7 text-sm ${
                      isCurrent ? 'text-amber-100' : 'text-[var(--muted-foreground)]'
                    }`}
                  >
                    {s.preview || 'New conversation'}
                  </div>
                  <div className="mt-0.5 flex items-center gap-2 text-[11px] text-[var(--soft-foreground)]/75">
                    <span>{formatStart(s.start_time)}</span>
                    {s.message_count > 0 && (
                      <span>· {Math.ceil(s.message_count / 2)} Q</span>
                    )}
                    {isCurrent && (
                      <span className="text-amber-200/80">· current</span>
                    )}
                  </div>
                  {/* Hover-reveal delete, mirroring the BookCard pattern. */}
                  <span
                    role="button"
                    aria-label="Delete this conversation"
                    tabIndex={0}
                    onClick={(e) => handleDelete(e, s)}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter' || e.key === ' ') {
                        handleDelete(e as unknown as React.MouseEvent, s)
                      }
                    }}
                    className="absolute right-2 top-1/2 -translate-y-1/2 rounded-full border border-white/10 bg-black/30 p-1 text-[var(--soft-foreground)] opacity-0 backdrop-blur-sm transition-all duration-200 hover:text-red-300 focus:opacity-100 focus:outline-none group-hover:opacity-100"
                  >
                    <Trash2 size={12} />
                  </span>
                </button>
              </li>
            )
          })}
        </ul>
      )}

      {hiddenCount > 0 && (
        <button
          type="button"
          onClick={() => setExpanded((v) => !v)}
          className="mt-2.5 flex w-full items-center justify-center gap-1.5 py-1 text-[11px] uppercase tracking-[0.22em] text-[var(--soft-foreground)] transition-colors hover:text-amber-200"
        >
          <ChevronDown
            size={12}
            className={`transition-transform duration-200 ${expanded ? 'rotate-180' : ''}`}
          />
          {expanded ? 'Show less' : `Show more (${hiddenCount})`}
        </button>
      )}
    </div>
  )
}
