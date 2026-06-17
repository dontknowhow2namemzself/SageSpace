import { ProgressData } from '@/lib/types'

export default function TokenStats({
  data,
  collapsed = false,
  onToggle,
}: {
  data: ProgressData
  collapsed?: boolean
  onToggle?: () => void
}) {
  const { tokens_in, tokens_out, cost_usd } = data.token_stats
  return (
    <div className="bg-stone-900 rounded-xl p-5 border border-stone-800">
      <div className="mb-4 flex items-center justify-between gap-3">
        <h3 className="font-serif text-amber-200 text-base">Token Usage</h3>
        {onToggle && (
          <button
            type="button"
            onClick={onToggle}
            className="rounded-md border border-stone-700 px-2.5 py-1 text-[11px] uppercase tracking-wide text-stone-400 transition-colors hover:text-stone-200"
          >
            {collapsed ? 'Expand' : 'Collapse'}
          </button>
        )}
      </div>

      {collapsed ? (
        <div className="space-y-2 text-sm">
          <div className="flex justify-between gap-3 text-stone-400">
            <span className="text-stone-500">Total Tokens</span>
            <span className="font-mono">{(tokens_in + tokens_out).toLocaleString()}</span>
          </div>
          <div className="flex justify-between gap-3 text-stone-400">
            <span className="text-stone-500">Session Cost</span>
            <span className="text-amber-400 font-mono">${cost_usd.toFixed(4)}</span>
          </div>
        </div>
      ) : (
        <div className="space-y-3 text-sm">
          <div className="flex justify-between">
            <span className="text-stone-500">Input Tokens</span>
            <span className="text-stone-300 font-mono">{tokens_in.toLocaleString()}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-stone-500">Output Tokens</span>
            <span className="text-stone-300 font-mono">{tokens_out.toLocaleString()}</span>
          </div>
          <div className="h-px bg-stone-800" />
          <div className="flex justify-between">
            <span className="text-stone-500">Session Cost</span>
            <span className="text-amber-400 font-mono">${cost_usd.toFixed(4)}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-stone-500">Total Tokens</span>
            <span className="text-stone-400 font-mono">
              {(tokens_in + tokens_out).toLocaleString()}
            </span>
          </div>
        </div>
      )}
    </div>
  )
}
