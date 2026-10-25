function formatEur(value: number): string {
  if (value >= 1_000_000) return `€${(value / 1_000_000).toFixed(1)}M`
  if (value >= 1_000) return `€${(value / 1_000).toFixed(0)}K`
  return `€${value}`
}

function gapCopy(estimate: number, actual: number): string {
  const gap = estimate - actual
  const scale = actual > 0 ? Math.abs(gap) / actual : 1
  if (scale < 0.08) return 'In line with the Transfermarkt quote'
  const amount = formatEur(Math.abs(gap))
  return gap > 0 ? `${amount} above the Transfermarkt quote` : `${amount} under the Transfermarkt quote`
}

export function ValueCompare({
  estimate,
  actual,
}: {
  estimate: number
  actual: number | null
}) {
  const max = Math.max(estimate, actual ?? 0, 1)
  const estimateHeight = Math.max(8, Math.round((estimate / max) * 100))
  const actualHeight = actual == null ? 0 : Math.max(8, Math.round((actual / max) * 100))

  return (
    <div className="glass-panel p-5">
      <div className="flex items-end justify-between gap-4">
        <h3 className="font-display text-xl font-bold uppercase tracking-wide">
          Model vs market
        </h3>
        {actual != null && (
          <p className="text-sm text-lime">{gapCopy(estimate, actual)}</p>
        )}
      </div>
      <div className="value-scene">
        <div className="value-rig">
          <div className="value-floor" />
          <div className="value-col">
            <div className="value-bar value-bar-model" style={{ ['--h' as string]: `${estimateHeight}%` }}>
              <span>{formatEur(estimate)}</span>
            </div>
            <p>Model</p>
          </div>
          {actual != null && (
            <div className="value-col">
              <div className="value-bar value-bar-market" style={{ ['--h' as string]: `${actualHeight}%` }}>
                <span>{formatEur(actual)}</span>
              </div>
              <p>Market</p>
            </div>
          )}
        </div>
      </div>
      <p className="mt-2 text-xs leading-relaxed text-ink/55">
        {actual != null
          ? 'The market bar is the Transfermarkt quote nearest the end of the season those stats come from. The model bar is the estimate from those stats.'
          : 'This player has no matched squad quote, so only the model estimate is drawn.'}
      </p>
    </div>
  )
}
