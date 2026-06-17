'use client'
import { motion } from 'framer-motion'
import { useRouter } from 'next/navigation'
import { Trash2, BookOpenText } from 'lucide-react'
import { Book } from '@/lib/types'
import { getBookCoverUrl } from '@/lib/api'

export default function BookCard({
  book,
  onDelete,
}: {
  book: Book
  onDelete: (id: string) => void
}) {
  const router = useRouter()
  const isReady = book.raptor_status === 'ready'
  const hasCover = Boolean(book.cover_url)

  return (
    <motion.div
      whileHover={{ y: isReady ? -4 : 0, scale: isReady ? 1.02 : 1 }}
      transition={{ type: 'spring', stiffness: 300, damping: 20 }}
      className={`group glass-panel relative overflow-hidden rounded-[1.4rem] p-4 sm:p-4
        ${isReady ? 'cursor-pointer hover:border-amber-300/20 hover:shadow-[0_24px_50px_rgba(8,6,4,0.3)]' : 'cursor-default opacity-75'}`}
      onClick={() => isReady && router.push(`/chat/${book.id}`)}
    >
      <div className="pointer-events-none absolute inset-x-4 top-0 h-px bg-gradient-to-r from-transparent via-white/40 to-transparent" />

      {/* Cover — generated 2:3 portrait, falls back to icon when absent */}
      <div className="mb-4 aspect-[2/3] w-full overflow-hidden rounded-[1.1rem] border border-white/10 bg-gradient-to-br from-amber-100/10 via-[#8e6442]/18 to-stone-950/70">
        {hasCover ? (
          // eslint-disable-next-line @next/next/no-img-element
          <img
            src={getBookCoverUrl(book.id)}
            alt=""
            className="h-full w-full object-cover"
            loading="lazy"
          />
        ) : (
          <div className="flex h-full w-full items-center justify-center">
            <div className="flex h-16 w-16 items-center justify-center rounded-2xl border border-white/10 bg-black/15 text-amber-50 shadow-inner shadow-white/5">
              <BookOpenText size={28} strokeWidth={1.6} />
            </div>
          </div>
        )}
      </div>

      {/* Title + author */}
      <h3 className="font-serif text-base font-semibold leading-snug text-amber-50 line-clamp-2">
        {book.title}
      </h3>
      {book.author && (
        <p className="mt-1 truncate text-xs text-[var(--muted-foreground)]">{book.author}</p>
      )}

      {/* Progress bar */}
      <div className="mt-4">
        <div className="mb-1.5 flex justify-between text-xs text-[var(--soft-foreground)]">
          <span>Explored</span>
          <span>{book.digested_pct.toFixed(1)}%</span>
        </div>
        <div className="h-1.5 overflow-hidden rounded-full bg-white/10">
          <div
            className="h-full rounded-full bg-gradient-to-r from-amber-300 via-amber-400 to-amber-500 transition-all duration-700"
            style={{ width: `${Math.min(book.digested_pct, 100)}%` }}
          />
        </div>
      </div>

      {/* Status badge */}
      {!isReady && (
        <div className="mt-3 rounded-full border border-white/8 bg-black/10 px-3 py-1 text-center text-xs text-[var(--muted-foreground)] animate-pulse">
          {book.raptor_status === 'building'
            ? 'Indexing...'
            : book.raptor_status === 'pending'
            ? 'Queued...'
            : book.raptor_status}
        </div>
      )}

      {/* Finished badge */}
      {isReady && book.digested_pct >= 100 && (
        <div className="absolute left-3 top-3 rounded-full border border-amber-200/20 bg-amber-200/14 px-2.5 py-1 text-xs text-amber-100 backdrop-blur-sm">
          Completed
        </div>
      )}

      {/* Delete button — fades in on card hover */}
      <button
        aria-label={`Remove ${book.title} from your shelf`}
        className="absolute right-3 top-3 rounded-full border border-white/8 bg-black/30 p-1.5 text-[var(--soft-foreground)] opacity-0 backdrop-blur-sm transition-all duration-200 hover:text-red-300 focus:opacity-100 focus:outline-none focus:ring-1 focus:ring-amber-200/30 group-hover:opacity-100"
        onClick={(e) => {
          e.stopPropagation()
          if (confirm(`Delete "${book.title}" from your shelf?`)) onDelete(book.id)
        }}
      >
        <Trash2 size={14} />
      </button>
    </motion.div>
  )
}
