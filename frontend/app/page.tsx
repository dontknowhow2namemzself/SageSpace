'use client'
import { useState, useEffect, useCallback, useRef } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { LibraryBig } from 'lucide-react'
import BookCard from '@/components/BookCard'
import UploadZone from '@/components/UploadZone'
import Recommendations from '@/components/Recommendations'
import WantToReadList, { WantToReadHandle } from '@/components/WantToReadList'
import MemoryNotes from '@/components/MemoryNotes'
import { listBooks, deleteBook } from '@/lib/api'
import { Book } from '@/lib/types'

export default function BookShelf() {
  const [books, setBooks] = useState<Book[]>([])
  const [showUpload, setShowUpload] = useState(false)
  const [loading, setLoading] = useState(true)
  const wantToReadRef = useRef<WantToReadHandle>(null)

  const refresh = useCallback(async () => {
    try {
      setBooks(await listBooks())
    } catch {
      // backend may not be running yet
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    refresh()
    // Poll every 5s to detect when "building" books become "ready"
    const interval = setInterval(refresh, 5000)
    return () => clearInterval(interval)
  }, [refresh])

  const handleDelete = async (id: string) => {
    await deleteBook(id).catch(() => {})
    refresh()
  }

  return (
    <div className="hero-background flex min-h-screen flex-col px-4 py-6 sm:px-6 lg:px-8 lg:py-10">
      <div className="hero-content-shell mx-auto w-full max-w-6xl flex-1 rounded-[2rem] px-5 py-6 sm:px-8 sm:py-8 lg:px-10 lg:py-10">
        {/* Hero — left-aligned, breathing room */}
        <div className="mb-12 max-w-3xl lg:mb-16">
          <h1 className="font-serif text-5xl tracking-[0.08em] text-[var(--foreground)] sm:text-6xl">
            SageSpace
          </h1>
          <p className="mt-5 max-w-xl text-base leading-7 text-[var(--muted-foreground)] sm:text-lg sm:leading-8">
            Your Sage absorbs every page. The conversation begins where you choose.
          </p>
          <button
            onClick={() => setShowUpload((v) => !v)}
            className="hero-button mt-8 inline-flex items-center justify-center gap-2 rounded-xl px-6 py-3 text-sm font-medium text-amber-50 transition-transform duration-200 hover:-translate-y-0.5"
          >
            {showUpload ? (
              'Close'
            ) : (
              <>
                <span className="text-lg leading-none">+</span>
                Bring a book in
              </>
            )}
          </button>
        </div>

        {/* Upload zone */}
        <AnimatePresence>
          {showUpload && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
              className="mb-8 overflow-hidden"
            >
              <UploadZone
                onUploaded={() => {
                  setShowUpload(false)
                  setTimeout(refresh, 500)
                }}
              />
            </motion.div>
          )}
        </AnimatePresence>

        {/* Book grid */}
        {loading ? (
          <div className="glass-panel-soft rounded-[1.75rem] py-20 text-center text-[var(--soft-foreground)] animate-pulse">
            Opening your space…
          </div>
        ) : books.length === 0 ? (
          <div className="glass-panel rounded-[1.75rem] px-6 py-20 text-center">
            <div className="mx-auto mb-5 flex h-14 w-14 items-center justify-center rounded-2xl border border-white/10 bg-white/5 text-amber-200/70">
              <LibraryBig size={28} strokeWidth={1.4} />
            </div>
            <p className="font-serif text-2xl text-amber-50">Your space is quiet</p>
            <p className="mx-auto mt-3 max-w-md text-sm leading-7 text-[var(--muted-foreground)]">
              Add your first book — your Sage will read every word, and wait for your questions.
            </p>
          </div>
        ) : (
          <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5">
            {books.map((book) => (
              <BookCard key={book.id} book={book} onDelete={handleDelete} />
            ))}
          </div>
        )}

        {/* Discovery surface — calm, pull-discovered; only once the shelf has a
            book (an empty shelf shows the onboarding nudge instead).
            "For you" recommendations, the "Want to read" list, then the quiet
            "What I remember" honest entry point. */}
        {!loading && books.length > 0 && (
          <>
            <Recommendations onAdd={() => wantToReadRef.current?.refresh()} />
            <WantToReadList ref={wantToReadRef} />
            <MemoryNotes />
          </>
        )}
      </div>

      {/* Footer signature */}
      <footer className="mx-auto w-full max-w-6xl px-5 pb-4 pt-6 text-right text-xs text-[var(--soft-foreground)]/60 sm:px-8 lg:px-10">
        SageSpace · by W. Zhang
      </footer>
    </div>
  )
}
