'use client'
import { useState, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { uploadBook } from '@/lib/api'

type Status = 'idle' | 'uploading' | 'done' | 'error'

export default function UploadZone({ onUploaded }: { onUploaded: () => void }) {
  const [dragging, setDragging] = useState(false)
  const [status, setStatus] = useState<Status>('idle')
  const [message, setMessage] = useState('')

  const handleFile = useCallback(
    async (file: File) => {
      if (!file.name.match(/\.(pdf|epub)$/i)) {
        setStatus('error')
        setMessage('Only PDF and ePub files are supported.')
        setTimeout(() => setStatus('idle'), 3000)
        return
      }
      setStatus('uploading')
      setMessage('Uploading...')
      try {
        await uploadBook(file)
        setStatus('done')
        setMessage('Your Sage is reading it now...')
        setTimeout(() => {
          setStatus('idle')
          onUploaded()
        }, 1500)
      } catch {
        setStatus('error')
        setMessage('Upload failed. Please try again.')
        setTimeout(() => setStatus('idle'), 3000)
      }
    },
    [onUploaded]
  )

  return (
    <motion.div
      className={`glass-panel rounded-[1.6rem] border border-dashed p-8 text-center transition-all duration-200 sm:p-10
        ${dragging ? 'border-amber-200/40 bg-amber-100/10 shadow-[0_20px_50px_rgba(84,53,24,0.2)]' : 'border-white/14 hover:border-white/20'}
        ${status === 'error' ? 'border-red-400/40' : ''}`}
      onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
      onDragLeave={() => setDragging(false)}
      onDrop={(e) => {
        e.preventDefault()
        setDragging(false)
        const f = e.dataTransfer.files[0]
        if (f) handleFile(f)
      }}
    >
      <AnimatePresence mode="wait">
        {status === 'idle' && (
          <motion.div
            key="idle"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
          >
            <div className="mb-3 text-4xl">📖</div>
            <p className="text-sm text-amber-50">Drop your PDF / ePub here</p>
            <p className="my-2 text-xs text-[var(--soft-foreground)]">or</p>
            <label className="inline-flex cursor-pointer items-center justify-center rounded-full border border-amber-100/15 bg-amber-100/10 px-4 py-2 text-sm text-amber-100 transition-colors hover:bg-amber-100/14">
              Choose a file
              <input
                type="file"
                accept=".pdf,.epub"
                className="hidden"
                onChange={(e) => {
                  const f = e.target.files?.[0]
                  if (f) handleFile(f)
                  e.target.value = ''
                }}
              />
            </label>
            <p className="mt-4 text-xs text-[var(--soft-foreground)]">Up to 50 MB</p>
          </motion.div>
        )}

        {(status === 'uploading' || status === 'done') && (
          <motion.div
            key="loading"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className={`text-sm ${status === 'done' ? 'text-amber-200' : 'text-[var(--muted-foreground)] animate-pulse'}`}
          >
            {status === 'done' && <span className="mr-2">✓</span>}
            {message}
          </motion.div>
        )}

        {status === 'error' && (
          <motion.div
            key="error"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="text-sm text-red-300"
          >
            ✕ {message}
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  )
}
