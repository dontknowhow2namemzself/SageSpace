'use client'
import { useState, useEffect } from 'react'
import { useParams, useRouter, useSearchParams } from 'next/navigation'
import { Suspense } from 'react'
import ProgressPanel from '@/components/dashboard/ProgressPanel'
import TokenStats from '@/components/dashboard/TokenStats'
import { getProgress, getBook } from '@/lib/api'
import { ProgressData, Book } from '@/lib/types'

function DashboardContent() {
  const { bookId } = useParams<{ bookId: string }>()
  const router = useRouter()
  const searchParams = useSearchParams()
  const sessionId = searchParams.get('session') || ''

  const [book, setBook] = useState<Book | null>(null)
  const [progress, setProgress] = useState<ProgressData | null>(null)

  useEffect(() => {
    getBook(bookId).then(setBook).catch(() => {})
    if (sessionId) {
      getProgress(bookId, sessionId).then(setProgress).catch(() => {})
    }
  }, [bookId, sessionId])

  return (
    <div className="min-h-screen bg-stone-950 p-8">
      <div className="max-w-3xl mx-auto">
        <div className="flex items-center gap-4 mb-8">
          <button
            onClick={() => router.back()}
            className="text-stone-500 hover:text-stone-300 text-sm transition-colors"
          >
            ← Back
          </button>
          <h1 className="font-serif text-2xl text-amber-100">
            {book?.title ? `${book.title} ` : ''}Dashboard
          </h1>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {progress ? (
            <>
              <ProgressPanel
                data={progress}
                readingMapHref={sessionId ? `/reading-map/${bookId}?session=${sessionId}` : undefined}
              />
              <TokenStats data={progress} />
            </>
          ) : (
            <div className="md:col-span-2 text-stone-600 text-sm text-center py-8">
              {sessionId
                ? 'Loading session progress...'
                : 'Open dashboard from chat to view current session data.'}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export default function DashboardPage() {
  return (
    <Suspense fallback={<div className="min-h-screen bg-stone-950" />}>
      <DashboardContent />
    </Suspense>
  )
}
