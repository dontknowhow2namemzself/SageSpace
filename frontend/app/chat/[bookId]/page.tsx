'use client'
import { useState, useEffect, useRef, useCallback } from 'react'
import { useParams, useRouter } from 'next/navigation'
import { HelpCircle, Search } from 'lucide-react'
import MessageBubble from '@/components/MessageBubble'
import InsightCard from '@/components/dashboard/InsightCard'
import SessionHistory from '@/components/chat/SessionHistory'
import {
  createSession,
  getProgress,
  getBook,
  exportNotes,
  getSession,
} from '@/lib/api'
import { Message, ChatEvent, ProgressData, Book } from '@/lib/types'

const API = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

// Empty-state quick prompts. All three route cleanly through the intent
// classifier: summarize → book_overview, chapter 1 → chapter_summary,
// progress → reading_progress (deterministic, no LLM).
const EXAMPLE_PROMPTS = [
  'Summarize this book',
  "What's chapter 1 about?",
  'How much have I read?',
] as const

// Maps the agent's live tool name (PR2 ReAct retrieve + the deterministic
// chapter_summary node) to a short, human label for the activity indicator.
// Unknown tools fall back to the raw name with underscores spaced out, so a
// future tool still reads sensibly without a code change here.
const TOOL_ACTIVITY: Record<string, string> = {
  semantic_search: 'Searching by meaning',
  keyword_search: 'Searching by keyword',
  get_chapter: 'Pulling up the chapter',
  expand_neighbors: 'Widening the context',
  chapter_summary: 'Summarizing the chapter',
}

const toolActivityLabel = (tool: string) =>
  TOOL_ACTIVITY[tool] ?? tool.replace(/_/g, ' ')

export default function ChatPage() {
  const { bookId } = useParams<{ bookId: string }>()
  const router = useRouter()

  const [book, setBook] = useState<Book | null>(null)
  const [sessionId, setSessionId] = useState<string | null>(null)
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [streaming, setStreaming] = useState(false)
  const [progress, setProgress] = useState<ProgressData | null>(null)
  // The retrieval tool the agent is running right now: set on tool_start,
  // cleared on tool_end. Drives the live activity indicator above the
  // composer so the hybrid retrieval (semantic / keyword / chapter /
  // neighbor) is legible while a turn streams.
  const [activeTool, setActiveTool] = useState<string | null>(null)
  // PR3: when the clarify gate interrupts, the turn pauses and this holds
  // the question + options. Non-null => render the clarify panel; the next
  // answer resumes the turn (POST /chat/resume) instead of starting a new one.
  const [clarify, setClarify] = useState<{ prompt: string; options: string[]; multi: boolean } | null>(null)
  // PR4: when set, a compound question is being researched as N parallel
  // sub-question searches — render the coarse "researching N" panel.
  const [fanout, setFanout] = useState<{ subquestions: string[] } | null>(null)
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false)
  // Bumped whenever the session list may have changed (session created,
  // turn finished) so the history sidebar refetches its previews.
  const [historyRefresh, setHistoryRefresh] = useState(0)
  const bottomRef = useRef<HTMLDivElement>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  // Tracks whether we've already jumped to the bottom for the restored
  // conversation. Set once on the first messages.length > 0 transition;
  // subsequent token streams handle their own scrolling.
  const initialScrollDone = useRef(false)
  const sessionStorageKey = `sagespace.session.${bookId}`

  const initializeSession = useCallback(
    async (forceNew: boolean) => {
      const writeSession = (sid: string) => {
        setSessionId(sid)
        if (typeof window !== 'undefined') {
          window.localStorage.setItem(sessionStorageKey, sid)
        }
      }

      if (!forceNew && typeof window !== 'undefined') {
        const cached = window.localStorage.getItem(sessionStorageKey)
        if (cached) {
          try {
            const existing = await getSession(cached)
            if (existing.book_id === bookId) {
              writeSession(existing.session_id)
              setMessages(existing.conversation || [])
              return
            }
          } catch {
            // fall through to new session creation
          }
        }
      }

      const { session_id } = await createSession(bookId)
      writeSession(session_id)
      if (forceNew) {
        setMessages([])
      }
      setHistoryRefresh((n) => n + 1)
    },
    [bookId, sessionStorageKey]
  )

  // Open a past conversation from the history sidebar: swap the active
  // session + restore its messages. Progress re-polls via the sessionId
  // effect below.
  const openSession = useCallback(
    async (sid: string) => {
      if (streaming || sid === sessionId) return
      try {
        const existing = await getSession(sid)
        if (existing.book_id !== bookId) return
        setSessionId(existing.session_id)
        if (typeof window !== 'undefined') {
          window.localStorage.setItem(sessionStorageKey, existing.session_id)
        }
        setMessages(existing.conversation || [])
        setClarify(null)
        setFanout(null)
        setProgress(null)
        requestAnimationFrame(() => {
          bottomRef.current?.scrollIntoView({ behavior: 'auto', block: 'end' })
        })
      } catch {}
    },
    [bookId, sessionId, streaming, sessionStorageKey]
  )

  // After a history row is deleted: if it was the active conversation,
  // start a fresh one (the old thread is gone server-side).
  const handleSessionDeleted = useCallback(
    (sid: string) => {
      if (sid === sessionId) {
        setProgress(null)
        setMessages([])
        initializeSession(true).catch(() => {})
      }
    },
    [sessionId, initializeSession]
  )

  useEffect(() => {
    getBook(bookId).then(setBook).catch(() => {})
    initializeSession(false).catch(() => {})
  }, [bookId, initializeSession])
  const refreshProgress = useCallback(async () => {
    if (!sessionId) return
    try {
      setProgress(await getProgress(bookId, sessionId))
    } catch {}
  }, [bookId, sessionId])

  useEffect(() => {
    if (!sessionId) return
    refreshProgress()
    const interval = setInterval(refreshProgress, 15000)
    return () => clearInterval(interval)
  }, [sessionId, refreshProgress])

  // On the FIRST render where the restored conversation has content,
  // jump straight to the latest message — no smooth scroll, that would
  // be jarring at load time. Subsequent token streams handle their own
  // scrolling in the SSE handler below.
  useEffect(() => {
    if (!initialScrollDone.current && messages.length > 0) {
      initialScrollDone.current = true
      // Defer one frame so the message DOM nodes exist before we measure.
      requestAnimationFrame(() => {
        bottomRef.current?.scrollIntoView({ behavior: 'auto', block: 'end' })
      })
    }
  }, [messages.length])

  // Reads one SSE leg (a /chat or /chat/resume response) to completion,
  // updating the in-flight assistant bubble. Shared by sendMessage and
  // resumeMessage so a paused-then-resumed turn renders identically to a
  // single-shot one. Handles PR3's ask_user (pause -> clarify panel) and
  // notice (system banner) frames in addition to the streaming answer.
  const readStream = useCallback(
    async (res: Response) => {
      const reader = res.body!.getReader()
      const decoder = new TextDecoder()
      let fullText = ''
      let hadError = false
      // PR4: while a compound question fans out, suppress the per-branch
      // tool spinner (it would flicker between branches) — the coarse
      // fan-out panel shows instead.
      let inFanout = false

      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        const lines = decoder.decode(value).split('\n')
        for (const line of lines) {
          if (!line.startsWith('data: ')) continue
          try {
            const event: ChatEvent = JSON.parse(line.slice(6))
            if (event.type === 'token') {
              const piece = event.content ?? ''
              fullText += piece
              setMessages((prev) => {
                const updated = [...prev]
                const last = updated[updated.length - 1]
                updated[updated.length - 1] = {
                  ...last,
                  role: 'assistant',
                  content: fullText,
                  pending: fullText.length === 0 ? last?.pending : false,
                }
                return updated
              })
              bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
            } else if (event.type === 'error') {
              hadError = true
              const errMsg = event.content || 'The scholar ran into an issue. Please retry.'
              setMessages((prev) => {
                const updated = [...prev]
                updated[updated.length - 1] = { role: 'assistant', content: errMsg, pending: false }
                return updated
              })
            } else if (event.type === 'tool_start') {
              if (!inFanout) setActiveTool(event.tool || null)
            } else if (event.type === 'tool_end') {
              if (!inFanout) setActiveTool(null)
            } else if (event.type === 'fanout_start') {
              inFanout = true
              setActiveTool(null)
              setFanout({ subquestions: event.subquestions ?? [] })
            } else if (event.type === 'fanout_end') {
              inFanout = false
              setFanout(null)
            } else if (event.type === 'retrieval_update') {
              // No live merge into progress: the frame's counts are the
              // internal "fed to the synthesizer" ledger, while the
              // Insight panel speaks CITED — which only exists after
              // finalize. refreshProgress() on `done` picks it up.
            } else if (event.type === 'ask_user') {
              // Turn paused at a clarify interrupt: drop the empty pending
              // placeholder and surface the question via the clarify panel.
              setMessages((prev) => {
                const updated = [...prev]
                const last = updated[updated.length - 1]
                if (last?.role === 'assistant' && last.pending && !last.content) updated.pop()
                return updated
              })
              setClarify({
                prompt: event.prompt ?? '',
                options: event.options ?? [],
                multi: !!event.multi,
              })
            } else if (event.type === 'notice') {
              // clarify_expired -> banner above the (broad) answer that
              // follows; anything else (no_pending) -> the notice IS the reply.
              const expired = event.kind === 'clarify_expired'
              setMessages((prev) => {
                const updated = [...prev]
                const last = updated[updated.length - 1]
                if (last?.role === 'assistant') {
                  updated[updated.length - 1] = expired
                    ? { ...last, notice: event.message, pending: false }
                    : { ...last, content: event.message ?? '', pending: false }
                }
                return updated
              })
            } else if (event.type === 'done') {
              if (!hadError && fullText.length === 0) {
                setMessages((prev) => {
                  const updated = [...prev]
                  const last = updated[updated.length - 1]
                  // Don't clobber a notice-only bubble (no answer expected).
                  if (last?.role === 'assistant' && !last.notice) {
                    updated[updated.length - 1] = {
                      role: 'assistant',
                      content: 'Sorry, no response was received. Please try again.',
                      pending: false,
                    }
                  }
                  return updated
                })
              }
              refreshProgress()
            } else if (event.type === 'stream_end') {
              // No-op — the loop exits on its own once the reader closes.
            }
          } catch {}
        }
      }
    },
    [refreshProgress]
  )

  const sendMessage = useCallback(
    async (text: string) => {
      if (!text.trim() || !sessionId || streaming) return
      setClarify(null)
      setMessages((prev) => [...prev, { role: 'user', content: text }])
      setInput('')
      setStreaming(true)
      // pending=true renders the "Sage is turning the page…" placeholder
      // until the first real token lands.
      setMessages((prev) => [...prev, { role: 'assistant', content: '', pending: true }])
      requestAnimationFrame(() => {
        bottomRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' })
      })

      try {
        const res = await fetch(`${API}/api/chat`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ book_id: bookId, session_id: sessionId, message: text }),
        })
        if (!res.ok) {
          let detail = 'Request failed. Please try again.'
          try {
            const errJson = await res.json()
            if (errJson.detail) detail = errJson.detail
          } catch {}
          throw new Error(detail)
        }
        await readStream(res)
      } catch (err: unknown) {
        const msg = err instanceof Error ? err.message : 'Something went wrong. Please retry.'
        setMessages((prev) => {
          const updated = [...prev]
          updated[updated.length - 1] = { role: 'assistant', content: msg, pending: false }
          return updated
        })
      } finally {
        setStreaming(false)
        setActiveTool(null)
        setFanout(null)
        setHistoryRefresh((n) => n + 1)
      }
    },
    [bookId, sessionId, streaming, readStream]
  )

  // PR3: answer a clarify question -> resume the SAME turn (POST
  // /chat/resume). An empty answer means "skip" (proceed broad). Streams
  // the rest of the turn into a fresh assistant bubble.
  const resumeMessage = useCallback(
    async (answer: string) => {
      if (!sessionId || streaming) return
      setClarify(null)
      setInput('')
      setStreaming(true)
      setMessages((prev) => [...prev, { role: 'assistant', content: '', pending: true }])
      requestAnimationFrame(() => {
        bottomRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' })
      })

      try {
        const res = await fetch(`${API}/api/chat/resume`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ session_id: sessionId, answer }),
        })
        if (!res.ok) throw new Error('Resume failed. Please try again.')
        await readStream(res)
      } catch (err: unknown) {
        const msg = err instanceof Error ? err.message : 'Something went wrong. Please retry.'
        setMessages((prev) => {
          const updated = [...prev]
          updated[updated.length - 1] = { role: 'assistant', content: msg, pending: false }
          return updated
        })
      } finally {
        setStreaming(false)
        setActiveTool(null)
        setFanout(null)
        setHistoryRefresh((n) => n + 1)
      }
    },
    [sessionId, streaming, readStream]
  )

  const handleExport = async (format: 'pdf' | 'markdown') => {
    if (!sessionId) return
    try {
      const blob = await exportNotes(sessionId, format)
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `scholar_notes.${format === 'pdf' ? 'pdf' : 'md'}`
      a.click()
      URL.revokeObjectURL(url)
    } catch {
      alert('Export failed. Please try again.')
    }
  }

  const handleNewConversation = async () => {
    if (!confirm('Start a new conversation? Current chat context will be reset.')) return
    setProgress(null)
    await initializeSession(true)
  }

  return (
    <div className="chat-background flex h-screen flex-col overflow-hidden">
      {/* Top bar — chrome strip with explicit dark scrim so the title
          and back button are legible regardless of which patch of the
          background painting sits behind them. */}
      <div className="flex flex-shrink-0 items-center justify-between border-b border-white/10 bg-gradient-to-b from-black/55 to-black/35 px-5 py-3 backdrop-blur-md">
        <button
          onClick={() => router.push('/')}
          className="text-sm text-[var(--muted-foreground)] transition-colors hover:text-amber-100"
        >
          ← Back to Shelf
        </button>
        <span className="font-serif text-base tracking-wide text-amber-50 truncate max-w-xs">
          {book?.title || 'SageSpace'}
        </span>
        <div className="w-24" />
      </div>

      {/* Main area */}
      <div className="min-h-0 flex-1 p-3">
        <div className={`grid h-full min-h-0 gap-3 ${sidebarCollapsed ? 'grid-cols-[72px_minmax(0,1fr)]' : 'grid-cols-[minmax(260px,320px)_minmax(0,1fr)]'}`}>
          {/* Left column: session insights — collapsed rail or single InsightCard */}
          <section className="min-h-0">
            {sidebarCollapsed ? (
              <div className="glass-panel-soft flex h-full flex-col items-center justify-center rounded-2xl px-2 py-3">
                <button
                  type="button"
                  onClick={() => setSidebarCollapsed(false)}
                  className="group flex h-full w-full flex-col items-center justify-center rounded-2xl px-1 py-2 transition-colors hover:bg-white/[0.03]"
                >
                  <div className="mb-4 flex h-10 w-10 items-center justify-center rounded-full border border-amber-300/20 bg-amber-900/10 text-amber-200/85 transition-colors group-hover:text-amber-100">
                    ›
                  </div>
                  <div className="text-[11px] uppercase tracking-[0.22em] text-[var(--soft-foreground)] [writing-mode:vertical-rl] [text-orientation:mixed] group-hover:text-[var(--muted-foreground)]">
                    Insight Panel
                  </div>
                </button>
              </div>
            ) : (
              <div className="scrollbar-warm h-full min-h-0 space-y-3 overflow-y-auto">
                <InsightCard
                  bookId={bookId}
                  sessionId={sessionId ?? undefined}
                  progress={progress}
                  onHide={() => setSidebarCollapsed(true)}
                />
                <SessionHistory
                  bookId={bookId}
                  currentSessionId={sessionId}
                  refreshToken={historyRefresh}
                  onOpenSession={openSession}
                  onDeleted={handleSessionDeleted}
                />
              </div>
            )}
          </section>

          {/* Right: Chat area */}
          <section className="glass-panel flex min-h-0 flex-col rounded-2xl">
            <div className="flex items-center justify-between px-4 py-3">
              <div className="text-xs uppercase tracking-[0.22em] text-[var(--soft-foreground)]">
                Conversation
              </div>
              <div className="flex items-center gap-2">
                <button
                  onClick={() => handleExport('markdown')}
                  className="rounded-lg border border-amber-300/25 px-3 py-1.5 text-xs text-amber-100/90 transition-colors hover:border-amber-300/45 hover:text-amber-50"
                  title="Export Markdown notes"
                >
                  Export Notes
                </button>
                <button
                  onClick={handleNewConversation}
                  className="rounded-lg border border-white/10 px-3 py-1.5 text-xs text-[var(--muted-foreground)] transition-colors hover:border-white/20 hover:text-amber-100"
                  title="Start a new session"
                >
                  New Conversation
                </button>
              </div>
            </div>

            {/* Soft gradient divider — matches the BookCard top-edge
                vocabulary; reads as atmosphere rather than chrome. */}
            <div className="mx-4 h-px bg-gradient-to-r from-transparent via-white/12 to-transparent" />

            <div className="scrollbar-warm flex-1 overflow-y-auto px-6 py-4">
              {messages.length === 0 && (
                <div className="mt-20 flex flex-col items-center text-center">
                  <p className="font-serif text-2xl italic text-[var(--muted-foreground)]">
                    Where shall we begin?
                  </p>
                  <p className="mt-2 text-xs uppercase tracking-[0.22em] text-[var(--soft-foreground)]/70">
                    or try one of these
                  </p>
                  <div className="mt-5 flex max-w-md flex-wrap items-center justify-center gap-2">
                    {EXAMPLE_PROMPTS.map((prompt) => (
                      <button
                        key={prompt}
                        type="button"
                        onClick={() => sendMessage(prompt)}
                        disabled={streaming || !sessionId}
                        className="rounded-full border border-white/10 bg-black/15 px-4 py-1.5 text-sm text-[var(--muted-foreground)] transition-all duration-200 hover:-translate-y-0.5 hover:border-amber-300/30 hover:text-amber-100 disabled:cursor-not-allowed disabled:opacity-40 disabled:hover:translate-y-0"
                      >
                        {prompt}
                      </button>
                    ))}
                  </div>
                </div>
              )}
              {messages.map((msg, i) => (
                <MessageBubble key={i} msg={msg} bookId={bookId} />
              ))}
              {/* PR4: coarse fan-out panel while a compound question is
                  researched as N parallel sub-question searches. */}
              {fanout && (
                <div className="mb-5 flex justify-start">
                  <div className="sage-bubble max-w-[80%] rounded-2xl rounded-bl-md px-5 py-4">
                    <div className="mb-2 flex items-center gap-2 font-serif text-sm text-amber-100">
                      <Search className="h-4 w-4 flex-shrink-0 animate-pulse text-amber-300/80" />
                      <span>Researching {fanout.subquestions.length} sub-questions in parallel…</span>
                    </div>
                    <ul className="space-y-1">
                      {fanout.subquestions.map((sq, i) => (
                        <li key={i} className="flex items-start gap-2 text-xs leading-relaxed text-[var(--soft-foreground)]">
                          <span className="mt-0.5 text-amber-400/60">{i + 1}.</span>
                          <span>{sq}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                </div>
              )}
              <div ref={bottomRef} />
            </div>

            {/* Soft gradient divider matching the header */}
            <div className="mx-4 h-px bg-gradient-to-r from-transparent via-white/12 to-transparent" />

            {/* PR3 clarify panel — shown when the turn paused at a clarify
                interrupt. Picking an option (or typing below / Skip) resumes
                the same turn. */}
            {clarify && (
              <div className="mx-4 mb-1 rounded-xl border border-amber-300/30 bg-amber-900/15 px-4 py-3">
                <div className="mb-2 flex items-center gap-2 text-sm text-amber-100">
                  <HelpCircle className="h-4 w-4 flex-shrink-0 text-amber-300/80" />
                  <span className="font-serif">{clarify.prompt}</span>
                </div>
                <div className="flex flex-wrap gap-2">
                  {clarify.options.map((opt) => (
                    <button
                      key={opt}
                      type="button"
                      onClick={() => resumeMessage(opt)}
                      disabled={streaming}
                      className="rounded-full border border-amber-300/30 bg-black/15 px-3 py-1 text-sm text-amber-100 transition-all duration-200 hover:-translate-y-0.5 hover:border-amber-300/50 hover:text-amber-50 disabled:cursor-not-allowed disabled:opacity-40"
                    >
                      {opt}
                    </button>
                  ))}
                  <button
                    type="button"
                    onClick={() => resumeMessage('')}
                    disabled={streaming}
                    className="rounded-full border border-white/10 px-3 py-1 text-sm text-[var(--muted-foreground)] transition-colors hover:border-white/20 hover:text-amber-100 disabled:cursor-not-allowed disabled:opacity-40"
                  >
                    Skip
                  </button>
                </div>
                <div className="mt-2 text-[11px] text-[var(--soft-foreground)]/70">
                  Or type your answer below
                </div>
              </div>
            )}

            {/* Live retrieval activity — names the tool the agent is running
                right now so the hybrid retrieval reads as deliberate work
                rather than a stall. streaming and clarify are mutually
                exclusive, so this never stacks with the panel above. */}
            {streaming && activeTool && (
              <div className="mx-4 mb-1 flex items-center gap-2 px-1 text-xs text-[var(--soft-foreground)]">
                <Search className="h-3.5 w-3.5 flex-shrink-0 animate-pulse text-amber-300/70" />
                <span className="font-serif italic text-amber-100/75">
                  {toolActivityLabel(activeTool)}…
                </span>
              </div>
            )}

            <div className="flex flex-shrink-0 items-end gap-3 px-4 py-3">
              <textarea
                ref={textareaRef}
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault()
                    if (clarify) resumeMessage(input)
                    else sendMessage(input)
                  }
                }}
                placeholder={
                  clarify
                    ? 'Answer the clarification… (Enter to send)'
                    : 'Ask the sage… (Enter to send, Shift+Enter for new line)'
                }
                rows={1}
                className="scrollbar-warm flex-1 resize-none rounded-xl border border-white/10 bg-black/20 px-4 py-2
                  text-sm text-amber-50 placeholder-[var(--soft-foreground)]/60 transition-colors
                  focus:border-amber-300/35 focus:outline-none"
                style={{ maxHeight: '120px' }}
              />
              <button
                onClick={() => (clarify ? resumeMessage(input) : sendMessage(input))}
                disabled={streaming || !input.trim()}
                className="hero-button flex-shrink-0 rounded-xl px-5 py-2 text-sm text-amber-50 transition-transform duration-200 hover:-translate-y-0.5 disabled:cursor-not-allowed disabled:opacity-40 disabled:hover:translate-y-0"
              >
                {streaming ? '…' : clarify ? 'Answer' : 'Send'}
              </button>
            </div>
          </section>

        </div>
      </div>
    </div>
  )
}
