'use client'
import { useState, useEffect, useCallback, forwardRef, useImperativeHandle } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Bookmark, ChevronDown, X } from 'lucide-react'
import { getSavedBooks, unsaveBook } from '@/lib/api'
import { Recommendation } from '@/lib/types'

export interface WantToReadHandle {
  refresh: () => void
}

/**
 * The home "Want to read" list (memory-system-design.md §B).
 *
 * Every book the user marked "Want to read" (status='added'), as a collapsible
 * list. Self-hides when empty. Removing a book is neutral (status -> 'seen':
 * off the list, not a dismissal). Exposes a `refresh` handle so the parent can
 * re-pull after a card is added from the "For you" block.
 */
const WantToReadList = forwardRef<WantToReadHandle>(function WantToReadList(_props, ref) {
  const [saved, setSaved] = useState<Recommendation[] | null>(null)
  const [open, setOpen] = useState(false)

  const load = useCallback(() => {
    getSavedBooks()
      .then(setSaved)
      .catch(() => setSaved([])) // backend down -> empty, stay quiet
  }, [])

  useEffect(() => {
    load()
  }, [load])

  useImperativeHandle(ref, () => ({ refresh: load }), [load])

  const remove = useCallback((id: string) => {
    setSaved((cur) => (cur ? cur.filter((r) => r.id !== id) : cur))
    unsaveBook(id).catch(() => {})
  }, [])

  if (!saved || saved.length === 0) return null // self-hide when empty

  return (
    <div className="mt-10">
      <button
        onClick={() => setOpen((v) => !v)}
        className="inline-flex items-center gap-2 rounded-full border border-amber-200/15 bg-amber-200/[0.06] px-3.5 py-1.5 text-xs text-amber-100/90 transition-colors hover:border-amber-200/30 hover:text-amber-100"
      >
        <Bookmark size={13} strokeWidth={1.7} />
        Want to read
        <span className="rounded-full bg-amber-200/15 px-1.5 text-[10px]">
          {saved.length}
        </span>
        <ChevronDown
          size={13}
          strokeWidth={1.8}
          className={`transition-transform duration-200 ${open ? 'rotate-180' : ''}`}
        />
      </button>

      <AnimatePresence initial={false}>
        {open && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="overflow-hidden"
          >
            <ul className="mt-3 space-y-2">
              <AnimatePresence mode="popLayout">
                {saved.map((book) => (
                  <motion.li
                    layout
                    key={book.id}
                    initial={{ opacity: 0, y: 6 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, scale: 0.97 }}
                    transition={{ type: 'spring', stiffness: 300, damping: 28 }}
                    className="group reading-surface flex items-start gap-3 rounded-xl px-4 py-3"
                  >
                    <Bookmark
                      size={15}
                      strokeWidth={1.6}
                      className="mt-0.5 shrink-0 text-amber-300/70"
                    />
                    <div className="min-w-0 flex-1">
                      <p className="font-serif text-sm text-amber-50">{book.title}</p>
                      {book.author && (
                        <p className="truncate text-xs text-[var(--muted-foreground)]">
                          {book.author}
                        </p>
                      )}
                      {book.reason && (
                        <p className="mt-1 text-xs leading-5 text-[var(--soft-foreground)] line-clamp-2">
                          {book.reason}
                        </p>
                      )}
                    </div>
                    <button
                      onClick={() => remove(book.id)}
                      aria-label={`Remove ${book.title} from Want to read`}
                      className="rounded-md p-1 text-[var(--soft-foreground)] opacity-0 transition-all hover:text-amber-50 focus:opacity-100 group-hover:opacity-100"
                    >
                      <X size={15} strokeWidth={1.8} />
                    </button>
                  </motion.li>
                ))}
              </AnimatePresence>
            </ul>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
})

export default WantToReadList
