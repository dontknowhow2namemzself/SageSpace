'use client'
import { ProgressData } from '@/lib/types'
import Link from 'next/link'

export default function ProgressPanel({
  data,
  readingMapHref,
}: {
  data: ProgressData
  readingMapHref?: string
}) {
  return (
    <div className="bg-stone-900 rounded-xl p-5 border border-stone-800">
      <div className="mb-4 flex items-center justify-between gap-3">
        <h3 className="font-serif text-amber-200 text-base">Reading Progress</h3>
        {readingMapHref && (
          <Link
            href={readingMapHref}
            className="rounded-md border border-amber-700/40 px-2.5 py-1 text-[11px] uppercase tracking-wide text-amber-300/80 hover:text-amber-200 transition-colors"
          >
            Reading Map
          </Link>
        )}
      </div>
      <div className="space-y-4">
        <div>
          <div className="mb-2 flex items-end justify-between gap-3">
            <span className="text-stone-500 text-sm">Digested</span>
            <span className="text-amber-300 text-2xl font-bold font-mono">
              {data.digested_pct.toFixed(1)}%
            </span>
          </div>
          <div className="h-2 overflow-hidden rounded-full bg-stone-800">
            <div
              className="h-full rounded-full bg-amber-700"
              style={{ width: `${Math.min(data.digested_pct, 100)}%` }}
            />
          </div>
        </div>

        <div className="space-y-2 text-sm">
          <div className="flex justify-between gap-3 text-xs text-stone-500">
            <span>Explored chunks</span>
            <span>{data.cited_chunks} / {data.total_chunks}</span>
          </div>
        </div>
      </div>
    </div>
  )
}
