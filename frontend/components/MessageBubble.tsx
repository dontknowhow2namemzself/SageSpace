'use client'
import { useState } from 'react'
import { motion } from 'framer-motion'
import { AlertCircle, BookOpen } from 'lucide-react'
import { Message } from '@/lib/types'
import CitationModal from '@/components/citation/CitationModal'
import MarkdownContent from '@/components/MarkdownContent'

interface Part {
  type: 'text' | 'citation' | 'commentary'
  content: string
  label?: string
  factId?: string
  chunkIds?: string[]
  eventIds?: string[]
}

export interface CitationInfo {
  label: string
  text: string
  chapter?: number
  page?: number
}

/**
 * Splits an assistant message into three kinds of parts:
 *   - text: plain factual narration (from <fact>...</fact> or untagged prose)
 *   - commentary: literary opinion / persona voice (from <commentary>...)
 *   - citation: the legacy [Chapter X · Page Y] block format, kept so older
 *     conversations that were persisted before the tag system still render.
 *
 * Tags are tolerant: case-insensitive, allow stray whitespace.
 */
function parseContent(content: string): Part[] {
  const parts: Part[] = []
  // 1) Pull out <fact>/<commentary> segments first; everything between
  //    matched tags becomes a typed part. Anything outside tags is
  //    treated as plain factual text (legacy answers / partial streams).
  const tagRe = /<(fact|commentary)([^>]*)>([\s\S]*?)<\/\1>/gi
  let cursor = 0
  let m: RegExpExecArray | null

  const parseAttrList = (rawAttrs: string) => {
    const attrs: Record<string, string> = {}
    const attrRe = /([\w-:]+)="([^"]*)"/g
    let attrMatch: RegExpExecArray | null
    while ((attrMatch = attrRe.exec(rawAttrs)) !== null) {
      attrs[attrMatch[1].toLowerCase()] = attrMatch[2]
    }
    return attrs
  }

  const splitIds = (raw?: string) =>
    (raw || '')
      .split(',')
      .map((value) => value.trim())
      .filter(Boolean)

  while ((m = tagRe.exec(content)) !== null) {
    if (m.index > cursor) {
      const between = content.slice(cursor, m.index)
      if (between.trim()) parts.push({ type: 'text', content: between })
    }
    const tag = m[1].toLowerCase()
    const attrs = parseAttrList(m[2] || '')
    parts.push({
      type: tag === 'commentary' ? 'commentary' : 'text',
      content: m[3],
      factId: attrs['data-fact-id'],
      chunkIds: splitIds(attrs['data-chunk-ids']),
      // data-raptor-ids is intentionally ignored: older persisted
      // answers still carry it, but it was the turn-level union of
      // retrieved summaries, not per-fact evidence (removed 2026-06-08).
      eventIds: splitIds(attrs['data-event-ids']),
    })
    cursor = m.index + m[0].length
  }
  if (cursor < content.length) {
    const tail = content.slice(cursor)
    if (tail.trim()) parts.push({ type: 'text', content: tail })
  }

  // 2) Legacy citation blocks live inside plain text parts that came from
  //    pre-tag-system answers (no fact_id, no chunk_ids). We must NOT
  //    touch fact-derived parts here — they look like plain text too
  //    (`type === 'text'`), but they carry attribution metadata (factId
  //    / chunkIds) that would be silently dropped if we ran them through
  //    the splitter. Gate on the presence of factId instead of type.
  const citationRe = /(\[(?:Chapter\s+\d+\s+·\s+(?:Page\s+\d+|Summary)|第\d+章[^\]]*)\])\n([\s\S]*?)(?=\n---|\n\[|$)/g
  const expanded: Part[] = []
  for (const p of parts) {
    const isAttributedFact = p.type === 'text' && (p.factId || (p.chunkIds && p.chunkIds.length > 0))
    if (p.type !== 'text' || isAttributedFact) {
      expanded.push(p)
      continue
    }
    let last = 0
    let cm: RegExpExecArray | null
    while ((cm = citationRe.exec(p.content)) !== null) {
      if (cm.index > last) {
        expanded.push({ type: 'text', content: p.content.slice(last, cm.index) })
      }
      expanded.push({ type: 'citation', label: cm[1], content: cm[2].trim() })
      last = cm.index + cm[0].length
    }
    if (last < p.content.length) {
      expanded.push({ type: 'text', content: p.content.slice(last) })
    }
    citationRe.lastIndex = 0
  }
  if (expanded.length === 0) expanded.push({ type: 'text', content })
  return expanded
}

function parseCitationLabel(label: string): Pick<CitationInfo, 'chapter' | 'page'> {
  const chapterMatch = label.match(/(?:第|Chapter\s+)(\d+)(?:章)?/i)
  const pageMatch = label.match(/(?:第|Page\s+)(\d+)(?:页)?/i)
  return {
    chapter: chapterMatch ? Number(chapterMatch[1]) : undefined,
    page: pageMatch ? Number(pageMatch[1]) : undefined,
  }
}

function CitationBlock({
  label,
  text,
  onClick,
}: {
  label: string
  text: string
  onClick?: (citation: CitationInfo) => void
}) {
  const [open, setOpen] = useState(false)
  const meta = parseCitationLabel(label)
  return (
    <div className="mt-2 border-l-2 border-amber-800/50 pl-3">
      <div className="flex items-center gap-3">
        <button
          onClick={() => setOpen((v) => !v)}
          className="text-amber-600/70 text-xs hover:text-amber-400 transition-colors"
        >
          {open ? '▼' : '▶'} {label}
        </button>
        {onClick && (
          <button
            onClick={() => onClick({ label, text, ...meta })}
            className="text-[11px] uppercase tracking-wide text-amber-500/70 hover:text-amber-300 transition-colors"
          >
            View Citation
          </button>
        )}
      </div>
      {open && (
        <motion.p
          initial={{ opacity: 0, height: 0 }}
          animate={{ opacity: 1, height: 'auto' }}
          className="text-stone-500 text-xs mt-1 italic leading-relaxed"
        >
          {text}
        </motion.p>
      )}
    </div>
  )
}

/**
 * Inline citation trigger(s) at the end of a <fact> paragraph.
 *
 * One icon per attributed source (the attribution mapper caps a fact at
 * two chunks), each opening CitationModal for ITS chunk — so a fact
 * supported by two passages exposes both, instead of advertising "2
 * references" while only ever opening the first.
 */
function FactCitationTrigger({
  bookId,
  sources,
}: {
  bookId: string
  sources: string[]
}) {
  const [activeId, setActiveId] = useState<string | null>(null)
  if (sources.length === 0) return null

  return (
    <>
      {sources.map((cid, i) => (
        <button
          key={cid}
          type="button"
          onClick={() => setActiveId(cid)}
          aria-label={`View source ${i + 1} of ${sources.length}`}
          title={
            sources.length === 1
              ? 'View source'
              : `View source ${i + 1} of ${sources.length}`
          }
          className="ml-1 inline-flex h-5 w-5 items-center justify-center align-middle rounded-full border border-amber-800/40 text-amber-300/80 hover:border-amber-500/60 hover:text-amber-200 transition-colors"
        >
          <BookOpen size={11} />
        </button>
      ))}
      {activeId && (
        <CitationModal
          bookId={bookId}
          chunkOrNodeId={activeId}
          onClose={() => setActiveId(null)}
        />
      )}
    </>
  )
}

export default function MessageBubble({
  msg,
  bookId,
  onCitationClick,
}: {
  msg: Message
  bookId?: string
  onCitationClick?: (citation: CitationInfo) => void
}) {
  const isUser = msg.role === 'user'

  // Pending state — assistant message has been requested but no tokens
  // have arrived yet. Italic placeholder with a soft pulse so the user
  // has a visible "the sage is working" affordance during the SSE
  // handshake + retrieval + first-token gap.
  if (!isUser && msg.pending) {
    return (
      <motion.div
        initial={{ opacity: 0, y: 6 }}
        animate={{ opacity: 1, y: 0 }}
        className="mb-5 flex justify-start"
      >
        <div className="max-w-[78%] font-serif text-base italic text-[var(--soft-foreground)] animate-pulse">
          Sage is turning the page…
        </div>
      </motion.div>
    )
  }

  const parts = parseContent(msg.content)

  // Body of the bubble — same parts rendering for both user and sage;
  // only the wrapping container differs (amber pill for user, walnut
  // sage-bubble plate for sage).
  const partsBody = parts.map((part, i) => {
    if (part.type === 'citation') {
      return (
        <CitationBlock
          key={i}
          label={part.label!}
          text={part.content}
          onClick={onCitationClick}
        />
      )
    }
    if (part.type === 'commentary') {
      return (
        <div key={i} className="mb-0.5 italic text-[var(--soft-foreground)]">
          <MarkdownContent content={part.content} />
        </div>
      )
    }
    // Fact / plain narration: text + an unobtrusive citation trigger
    // when the fact has matched raw chunks. Raw chunks point directly
    // at canonical blocks. (RAPTOR summary ids are no longer offered
    // as a fallback source — they were the turn-level union, not
    // per-fact evidence.)
    //
    // IMPORTANT: do NOT nest the trigger inside <p>. CitationModal
    // renders a fixed-position <div> via React portal-like pattern,
    // and the inner <button> + <div> would force the browser to
    // close the parent <p> prematurely (invalid HTML), making the
    // icon disappear from the rendered DOM. Wrap with <div> and
    // put the icon as a sibling block after the text instead.
    const sources = part.chunkIds ?? []
    // Showing the trigger whenever we have a bookId + sources, even
    // if the fact_id is missing — early answers were exported with
    // chunk_ids only, and we'd rather over-show the icon than have
    // it silently disappear.
    const showTrigger = Boolean(bookId) && sources.length > 0
    // Assistant prose renders as Markdown (GFM tables, lists, bold);
    // user messages stay verbatim — nobody wants their own `*`s eaten.
    // `[&>p:last-of-type]:inline` keeps the citation icon attached at
    // the end of the fact's final paragraph instead of dropping to its
    // own line. (last-of-type, NOT only-child: the trigger button is a
    // sibling of the <p>, so only-child would never match.)
    return (
      <div key={i} className="mb-0.5 [&>p:last-of-type]:inline">
        {isUser ? (
          <span className="whitespace-pre-wrap">{part.content}</span>
        ) : (
          <MarkdownContent content={part.content} />
        )}
        {showTrigger ? (
          <FactCitationTrigger bookId={bookId as string} sources={sources} />
        ) : null}
      </div>
    )
  })

  // ── User: amber pill with a deliberately mirrored asymmetric corner
  //    (squared at bottom-right, where the user "sits") that pairs with
  //    the sage's bottom-LEFT squared corner — they read as a matched
  //    pair pointing at each other. ──
  if (isUser) {
    return (
      <motion.div
        initial={{ opacity: 0, y: 6 }}
        animate={{ opacity: 1, y: 0 }}
        className="mb-5 flex justify-end"
      >
        <div className="max-w-[70%] rounded-2xl rounded-br-md border border-amber-300/30 bg-amber-900/30 px-4 py-3 text-sm text-amber-50">
          {partsBody}
        </div>
      </motion.div>
    )
  }

  // ── Sage: walnut "inscribed plate" bubble. Mirrored asymmetric corner
  //    (bottom-left squared off) pairs with the user bubble's bottom-
  //    right squared corner — they read as a matched pair pointing at
  //    each other. ──
  return (
    <motion.div
      initial={{ opacity: 0, y: 6 }}
      animate={{ opacity: 1, y: 0 }}
      className="mb-5 flex justify-start"
    >
      <div className="sage-bubble max-w-[80%] rounded-2xl rounded-bl-md px-5 py-4 font-serif text-base leading-relaxed text-[var(--foreground)]">
        {msg.notice && (
          <div className="mb-3 flex items-start gap-2 rounded-lg border border-amber-400/25 bg-amber-900/15 px-3 py-2 font-sans text-xs leading-relaxed text-amber-200/90">
            <AlertCircle className="mt-0.5 h-3.5 w-3.5 flex-shrink-0 text-amber-300/80" />
            <span>{msg.notice}</span>
          </div>
        )}
        {partsBody}
      </div>
    </motion.div>
  )
}
