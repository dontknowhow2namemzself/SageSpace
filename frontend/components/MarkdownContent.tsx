'use client'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import remarkBreaks from 'remark-breaks'

/**
 * Renders sage-answer prose as Markdown in the design system's voice.
 *
 * - remark-gfm: pipe tables (chapter comparisons etc.) become real
 *   tables instead of raw `|` separators.
 * - remark-breaks: the model's single newlines stay visible line
 *   breaks, matching how the previous whitespace-pre-wrap text read.
 *
 * Elements are mapped onto the walnut/amber palette so a rendered
 * answer still reads as part of the sage bubble, not a webpage.
 */
export default function MarkdownContent({ content }: { content: string }) {
  return (
    <ReactMarkdown
      remarkPlugins={[remarkGfm, remarkBreaks]}
      components={{
        p: ({ children }) => <p className="mb-2 last:mb-0">{children}</p>,
        ul: ({ children }) => (
          <ul className="mb-2 list-disc space-y-1 pl-5">{children}</ul>
        ),
        ol: ({ children }) => (
          <ol className="mb-2 list-decimal space-y-1 pl-5">{children}</ol>
        ),
        li: ({ children }) => <li className="leading-relaxed">{children}</li>,
        strong: ({ children }) => (
          <strong className="font-semibold text-amber-100">{children}</strong>
        ),
        a: ({ children, href }) => (
          <a
            href={href}
            target="_blank"
            rel="noreferrer"
            className="text-amber-300 underline decoration-amber-300/40 underline-offset-2 transition-colors hover:text-amber-200"
          >
            {children}
          </a>
        ),
        h1: ({ children }) => (
          <h3 className="mb-2 mt-3 font-serif text-lg text-amber-100 first:mt-0">
            {children}
          </h3>
        ),
        h2: ({ children }) => (
          <h4 className="mb-2 mt-3 font-serif text-base text-amber-100 first:mt-0">
            {children}
          </h4>
        ),
        h3: ({ children }) => (
          <h5 className="mb-1.5 mt-2.5 font-serif text-base text-amber-100/90 first:mt-0">
            {children}
          </h5>
        ),
        code: ({ children }) => (
          <code className="rounded bg-black/30 px-1 py-0.5 font-mono text-[0.85em] text-amber-100/90">
            {children}
          </code>
        ),
        pre: ({ children }) => (
          <pre className="scrollbar-warm mb-2 overflow-x-auto rounded-lg bg-black/30 p-3 font-mono text-xs leading-relaxed [&>code]:bg-transparent [&>code]:p-0">
            {children}
          </pre>
        ),
        blockquote: ({ children }) => (
          <blockquote className="mb-2 border-l-2 border-amber-300/30 pl-3 italic text-[var(--soft-foreground)]">
            {children}
          </blockquote>
        ),
        hr: () => <hr className="my-3 border-amber-100/10" />,
        table: ({ children }) => (
          <div className="scrollbar-warm my-3 overflow-x-auto">
            <table className="w-full border-collapse text-sm">{children}</table>
          </div>
        ),
        th: ({ children }) => (
          <th className="border border-amber-100/15 bg-amber-900/25 px-3 py-2 text-left font-semibold text-amber-100">
            {children}
          </th>
        ),
        td: ({ children }) => (
          <td className="border border-amber-100/10 px-3 py-2 align-top leading-relaxed">
            {children}
          </td>
        ),
      }}
    >
      {content}
    </ReactMarkdown>
  )
}
