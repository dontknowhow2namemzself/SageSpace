'use client'
import { ChapterChunks, ChunkMapResponse } from '@/lib/types'

interface Props {
  data: ChunkMapResponse | null
  selectedChunkId: string | null
  onSelectChunk: (chunkId: string) => void
}

/**
 * Prefer the canonical section_label (PR8). Fall back to "Chapter N"
 * when the backend has not populated section_label (older books or an
 * older backend build).
 */
export function groupHeader(group: ChapterChunks): string {
  if (group.section_label && group.section_label.trim()) {
    return group.section_label
  }
  return `Chapter ${group.chapter}`
}

/**
 * Kind-based muting for front-matter / back-matter groups so the eye
 * can scan straight to body-matter (chapter / prologue / epilogue).
 */
function groupAccent(group: ChapterChunks): string {
  const kind = group.kind
  if (kind === 'chapter' || kind === 'prologue' || kind === 'epilogue') {
    return 'text-amber-200/80'
  }
  if (kind === 'appendix' || kind === 'afterword') {
    return 'text-amber-200/40'
  }
  // front_matter / cover / toc / preface / etc. — use the warm soft-
  // foreground token instead of stone so non-body groups read as
  // "quieter" rather than "different palette".
  return 'text-[var(--soft-foreground)]/70'
}

export default function ChunkMapPanel({
  data, selectedChunkId, onSelectChunk,
}: Props) {
  if (!data) return (
    <div className="p-4 text-sm text-[var(--soft-foreground)]/70">
      Loading chunk map…
    </div>
  )

  return (
    <div className="scrollbar-warm h-full space-y-4 overflow-y-auto p-4">
      <div className="text-xs text-[var(--muted-foreground)]">
        {data.total_chunks} chunks · grouped by section
      </div>
      {data.chapters.map((group) => (
        <div key={group.section_id || `legacy-${group.chapter}`}>
          <div className={`${groupAccent(group)} mb-1 font-mono text-xs`}>
            {groupHeader(group)} ({group.chunks.length})
          </div>
          <div className="flex flex-wrap gap-1">
            {group.chunks.map((chunk) => {
              const isSelected = chunk.chunk_id === selectedChunkId
              return (
                <button
                  key={chunk.chunk_id}
                  title={`${chunk.chunk_id} · p.${chunk.page}`}
                  onClick={() => onSelectChunk(chunk.chunk_id)}
                  className={[
                    'h-2.5 w-2.5 rounded-sm transition-all duration-300',
                    isSelected ? 'scale-125 ring-2 ring-amber-50' : '',
                    // Lit = saturated amber. Unlit = warm taupe/walnut:
                    // bright enough to see against the frosted dark
                    // backdrop, but warm-toned so the grid stays in the
                    // same palette family as everything else. Tune
                    // candidates: #5a4838 (current) / #6b5440 (warmer
                    // sepia) / #4a3a2c (deeper walnut).
                    chunk.is_lit ? 'bg-amber-500' : 'bg-[#5a4838]',
                  ].join(' ')}
                />
              )
            })}
          </div>
        </div>
      ))}
    </div>
  )
}
