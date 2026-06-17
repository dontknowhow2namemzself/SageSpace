'use client'
import { useState, useEffect, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { NotebookPen, ChevronDown, Pencil, Trash2, Check, X } from 'lucide-react'
import { getMemoryNotes, updateMemoryNote, deleteMemoryNote } from '@/lib/api'
import { MemoryNote } from '@/lib/types'

/**
 * The home "What I remember" panel (memory-system-design.md §A).
 *
 * Notes are captured silently and are normally invisible fuel for
 * recommendations -- but since they DO shape what the app suggests, this is the
 * honest, always-available entry point to see and correct them. It stays out of
 * the way: a small collapsed button (with count) that the user opens on demand;
 * writes elsewhere remain silent. Each note can be edited inline or deleted.
 */
export default function MemoryNotes() {
  const [notes, setNotes] = useState<MemoryNote[] | null>(null)
  const [open, setOpen] = useState(false)
  const [editingId, setEditingId] = useState<string | null>(null)
  const [draft, setDraft] = useState('')

  useEffect(() => {
    let alive = true
    getMemoryNotes()
      .then((data) => alive && setNotes(data))
      .catch(() => alive && setNotes([])) // backend down -> empty, stay quiet
    return () => {
      alive = false
    }
  }, [])

  const beginEdit = useCallback((note: MemoryNote) => {
    setEditingId(note.id)
    setDraft(note.text)
  }, [])

  const cancelEdit = useCallback(() => {
    setEditingId(null)
    setDraft('')
  }, [])

  const saveEdit = useCallback(
    async (id: string) => {
      const text = draft.trim()
      if (!text) return
      setNotes((cur) =>
        cur ? cur.map((n) => (n.id === id ? { ...n, text } : n)) : cur
      )
      setEditingId(null)
      try {
        await updateMemoryNote(id, text)
      } catch {
        // best-effort; a failed save will reconcile on next load
      }
    },
    [draft]
  )

  const remove = useCallback((id: string) => {
    setNotes((cur) => (cur ? cur.filter((n) => n.id !== id) : cur))
    deleteMemoryNote(id).catch(() => {})
  }, [])

  if (notes === null) return null // not loaded yet -> render nothing

  const count = notes.length

  return (
    <div className="mt-10">
      {/* Collapsed entry — small, quiet, honest */}
      <button
        onClick={() => setOpen((v) => !v)}
        className="inline-flex items-center gap-2 rounded-full border border-amber-200/15 bg-amber-200/[0.04] px-3.5 py-1.5 text-xs text-[var(--soft-foreground)] transition-colors hover:border-amber-200/25 hover:text-amber-100"
      >
        <NotebookPen size={13} strokeWidth={1.7} />
        What I remember
        <span className="rounded-full bg-amber-200/15 px-1.5 text-[10px] text-[var(--muted-foreground)]">
          {count}
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
            <div className="reading-surface mt-3 rounded-[1.25rem] p-4">
              <p className="mb-3 text-[11px] leading-5 text-[var(--soft-foreground)]/70">
                Notes I&apos;ve quietly kept from what you told me — they shape your
                recommendations. Edit or delete anything.
              </p>

              {count === 0 ? (
                <p className="py-4 text-center text-sm text-[var(--muted-foreground)]">
                  Nothing noted yet — I only keep what you explicitly tell me.
                </p>
              ) : (
                <ul className="space-y-2">
                  {notes.map((note) => (
                    <li
                      key={note.id}
                      className="group flex items-center gap-2 rounded-lg border border-white/5 bg-white/[0.02] px-3 py-2"
                    >
                      <span className="shrink-0 rounded-full border border-amber-200/15 bg-amber-200/[0.07] px-2 py-0.5 text-[10px] uppercase tracking-wide text-amber-100/80">
                        {note.type}
                      </span>

                      {editingId === note.id ? (
                        <>
                          <input
                            value={draft}
                            onChange={(e) => setDraft(e.target.value)}
                            onKeyDown={(e) => {
                              if (e.key === 'Enter') saveEdit(note.id)
                              if (e.key === 'Escape') cancelEdit()
                            }}
                            autoFocus
                            className="flex-1 rounded-md border border-white/10 bg-black/20 px-2 py-1 text-sm text-amber-50 focus:border-amber-300/40 focus:outline-none"
                          />
                          <button
                            onClick={() => saveEdit(note.id)}
                            aria-label="Save"
                            className="rounded-md p-1 text-amber-200/80 hover:text-amber-100"
                          >
                            <Check size={15} strokeWidth={2} />
                          </button>
                          <button
                            onClick={cancelEdit}
                            aria-label="Cancel"
                            className="rounded-md p-1 text-[var(--soft-foreground)] hover:text-amber-50"
                          >
                            <X size={15} strokeWidth={2} />
                          </button>
                        </>
                      ) : (
                        <>
                          <span className="flex-1 text-sm leading-6 text-[var(--foreground)]">
                            {note.text}
                          </span>
                          <button
                            onClick={() => beginEdit(note)}
                            aria-label="Edit note"
                            className="rounded-md p-1 text-[var(--soft-foreground)] opacity-0 transition-opacity hover:text-amber-100 focus:opacity-100 group-hover:opacity-100"
                          >
                            <Pencil size={14} strokeWidth={1.8} />
                          </button>
                          <button
                            onClick={() => remove(note.id)}
                            aria-label="Delete note"
                            className="rounded-md p-1 text-[var(--soft-foreground)] opacity-0 transition-opacity hover:text-red-300 focus:opacity-100 group-hover:opacity-100"
                          >
                            <Trash2 size={14} strokeWidth={1.8} />
                          </button>
                        </>
                      )}
                    </li>
                  ))}
                </ul>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}
