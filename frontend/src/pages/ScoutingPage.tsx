import { useEffect, useMemo, useState } from 'react'
import {
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
  ZAxis,
} from 'recharts'
import { PlayerPortrait } from '../components/PlayerPortrait'
import {
  getClusters,
  getScoutingModelInfo,
  getSimilarPlayers,
  searchScoutPlayers,
} from '../api'
import type { ClustersResponse, ScoutPlayer, ScoutingModelInfo } from '../types'

function formatEur(value: number | null | undefined): string {
  if (value == null) return '—'
  if (value >= 1_000_000) return `€${(value / 1_000_000).toFixed(1)}M`
  if (value >= 1_000) return `€${(value / 1_000).toFixed(0)}K`
  return `€${value}`
}

const CLUSTER_COLORS = [
  '#0f6a4b',
  '#c8f542',
  '#1d4e89',
  '#d97706',
  '#7c3aed',
  '#be123c',
  '#0e7490',
  '#4d7c0f',
]

export function ScoutingPage() {
  const [query, setQuery] = useState('')
  const [results, setResults] = useState<ScoutPlayer[]>([])
  const [selected, setSelected] = useState<ScoutPlayer | null>(null)
  const [similar, setSimilar] = useState<ScoutPlayer[]>([])
  const [compareIds, setCompareIds] = useState<number[]>([])
  const [comparePlayers, setComparePlayers] = useState<ScoutPlayer[]>([])
  const [clusters, setClusters] = useState<ClustersResponse | null>(null)
  const [modelInfo, setModelInfo] = useState<ScoutingModelInfo | null>(null)
  const [positionFilter, setPositionFilter] = useState<string>('')
  const [maxAge, setMaxAge] = useState<string>('')
  const [maxValueM, setMaxValueM] = useState<string>('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [note, setNote] = useState<string>('')

  useEffect(() => {
    getClusters()
      .then(setClusters)
      .catch(() => setClusters(null))
    getScoutingModelInfo()
      .then(setModelInfo)
      .catch(() => setModelInfo(null))
  }, [])

  useEffect(() => {
    const handle = window.setTimeout(async () => {
      try {
        const players = await searchScoutPlayers(query)
        setResults(players)
      } catch {
        setResults([])
      }
    }, 250)
    return () => window.clearTimeout(handle)
  }, [query])

  async function runSimilar(player: ScoutPlayer) {
    setSelected(player)
    setLoading(true)
    setError(null)
    try {
      const res = await getSimilarPlayers(player.player_id, {
        limit: 8,
        position: positionFilter || undefined,
        maxAge: maxAge ? Number(maxAge) : undefined,
        maxValue: maxValueM ? Number(maxValueM) * 1_000_000 : undefined,
      })
      setSimilar(res.similar)
      setNote(res.dataset_note)
      setSelected(res.reference)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Similarity search failed')
      setSimilar([])
    } finally {
      setLoading(false)
    }
  }

  function toggleCompare(player: ScoutPlayer) {
    setCompareIds((prev) => {
      if (prev.includes(player.player_id)) {
        return prev.filter((id) => id !== player.player_id)
      }
      if (prev.length >= 4) return prev
      return [...prev, player.player_id]
    })
  }

  useEffect(() => {
    if (compareIds.length === 0) {
      setComparePlayers([])
      return
    }
    const pool = [selected, ...similar, ...results].filter(Boolean) as ScoutPlayer[]
    const byId = new Map(pool.map((p) => [p.player_id, p]))
    // Prefer already-loaded players; fill gaps from cluster map
    if (clusters) {
      for (const p of clusters.map_points) byId.set(p.player_id, p)
    }
    setComparePlayers(
      compareIds.map((id) => byId.get(id)).filter((p): p is ScoutPlayer => Boolean(p)),
    )
  }, [compareIds, selected, similar, results, clusters])

  const mapData = useMemo(() => {
    if (!clusters) return []
    return clusters.map_points.map((p) => ({
      ...p,
      fill: CLUSTER_COLORS[p.cluster_id % CLUSTER_COLORS.length],
    }))
  }, [clusters])

  return (
    <>
      <section className="animate-rise mb-10 max-w-2xl">
        <p className="mb-2 text-xs font-semibold uppercase tracking-[0.2em] text-pitch">
          Module 3 · Clustering
        </p>
        <h1 className="font-display text-5xl font-extrabold uppercase leading-none tracking-wide text-white md:text-6xl">
          Player scouting
        </h1>
        <p className="mt-4 max-w-xl text-base leading-relaxed text-white/65">
          Find statistically similar Premier League players, compare profiles side by side,
          and explore performance clusters from real season data.
        </p>
      </section>

      <div className="grid gap-8 lg:grid-cols-[1.05fr_0.95fr]">
        <section className="animate-rise-delay space-y-6">
          <div>
            <label className="mb-2 block text-sm font-semibold text-ink" htmlFor="scout-search">
              Reference player
            </label>
            <input
              id="scout-search"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Search name or club…"
              className="field-input"
            />
            <div className="mt-2 max-h-48 overflow-auto rounded-xl border border-white/10 bg-black/25">
              {results.map((player) => (
                <button
                  key={player.player_id}
                  type="button"
                  onClick={() => runSimilar(player)}
                  className={`flex w-full items-center justify-between gap-3 px-3 py-2 text-left text-sm transition hover:bg-white/10 ${
                    selected?.player_id === player.player_id ? 'bg-white/10' : ''
                  }`}
                >
                  <span className="flex items-center gap-3">
                    <PlayerPortrait name={player.player_name} photoUrl={player.photo_url} size="sm" />
                    <span>
                      <span className="font-semibold">{player.player_name}</span>
                      <span className="text-ink/50">
                        {' '}
                        · {player.club} · {player.cluster_label}
                      </span>
                    </span>
                  </span>
                  <span className="font-display text-lg text-pitch">{player.position}</span>
                </button>
              ))}
            </div>
          </div>

          <div className="grid gap-3 glass-panel p-4 sm:grid-cols-3">
            <label className="text-sm">
              <span className="mb-1 block font-medium text-ink/80">Position filter</span>
              <select
                value={positionFilter}
                onChange={(e) => setPositionFilter(e.target.value)}
                className="field-input"
              >
                <option value="">Any</option>
                {['GK', 'DEF', 'MID', 'FWD'].map((p) => (
                  <option key={p} value={p}>
                    {p}
                  </option>
                ))}
              </select>
            </label>
            <label className="text-sm">
              <span className="mb-1 block font-medium text-ink/80">Max age</span>
              <input
                type="text"
                inputMode="numeric"
                value={maxAge}
                onChange={(e) => {
                  const v = e.target.value
                  if (v === '' || /^\d+$/.test(v)) setMaxAge(v.replace(/^0+(?=\d)/, ''))
                }}
                placeholder="e.g. 26"
                className="field-input"
              />
            </label>
            <label className="text-sm">
              <span className="mb-1 block font-medium text-ink/80">Max value (€M)</span>
              <input
                type="text"
                inputMode="numeric"
                value={maxValueM}
                onChange={(e) => {
                  const v = e.target.value
                  if (v === '' || /^\d+$/.test(v)) setMaxValueM(v.replace(/^0+(?=\d)/, ''))
                }}
                placeholder="e.g. 40"
                className="field-input"
              />
            </label>
            {selected && (
              <button
                type="button"
                onClick={() => runSimilar(selected)}
                disabled={loading}
                className="sm:col-span-3 bg-pitch px-4 py-3 font-display text-lg font-bold uppercase tracking-wider text-white hover:bg-pitch-deep disabled:opacity-60"
              >
                {loading ? 'Finding…' : 'Refresh similar players'}
              </button>
            )}
          </div>

          {error && <p className="text-sm text-red-700">{error}</p>}

          {selected && (
            <div key={selected.player_id} className="player-enter glass-panel p-5">
              <div className="flex items-end justify-between gap-4">
                <div>
                  <h2 className="font-display text-3xl font-bold uppercase tracking-wide">
                    {selected.player_name}
                  </h2>
                  <p className="text-sm text-white/60">
                    {selected.club} · {selected.position} · {selected.age} · {selected.cluster_label}
                  </p>
                  <button
                    type="button"
                    onClick={() => toggleCompare(selected)}
                    className="mt-3 text-xs font-semibold uppercase tracking-wide text-lime"
                  >
                    {compareIds.includes(selected.player_id) ? 'Remove compare' : 'Add compare'}
                  </button>
                </div>
                <PlayerPortrait name={selected.player_name} photoUrl={selected.photo_url} />
              </div>
              <dl className="mt-5 grid grid-cols-2 gap-3 text-sm sm:grid-cols-4">
                <Stat label="Goals" value={String(selected.goals)} />
                <Stat label="Assists" value={String(selected.assists)} />
                <Stat label="Minutes" value={String(selected.minutes)} />
                <Stat label="Value" value={formatEur(selected.market_value_eur)} />
              </dl>
            </div>
          )}

          {similar.length > 0 && (
            <div className="glass-panel p-5">
              <h3 className="font-display text-xl font-bold uppercase tracking-wide">
                Similar players
              </h3>
              <div className="mt-4 grid grid-cols-2 gap-3">
                {similar.map((player, index) => (
                  <button
                    key={player.player_id}
                    type="button"
                    onClick={() => runSimilar(player)}
                    className="similar-card rounded-2xl bg-black/30 p-3 text-left"
                    style={{ animationDelay: `${index * 70}ms` }}
                  >
                    <PlayerPortrait name={player.player_name} photoUrl={player.photo_url} />
                    <p className="mt-3 font-semibold leading-tight">{player.player_name}</p>
                    <p className="text-xs text-white/50">
                      {player.club} · {player.position}
                    </p>
                    <p className="mt-2 font-display text-3xl text-lime">
                      {player.similarity?.toFixed(0)}%
                    </p>
                  </button>
                ))}
              </div>
              {note && <p className="mt-3 text-xs text-white/45">{note}</p>}
            </div>
          )}
        </section>

        <section className="animate-rise-delay-2 space-y-6">
          {comparePlayers.length > 0 && (
            <div className="glass-panel p-5">
              <h3 className="font-display text-xl font-bold uppercase tracking-wide">
                Side-by-side compare
              </h3>
              <div className="mt-4 overflow-x-auto">
                <table className="w-full min-w-[420px] text-left text-sm">
                  <thead>
                    <tr className="border-b border-pitch/15 text-xs uppercase tracking-wide text-ink/45">
                      <th className="py-2 pr-3">Metric</th>
                      {comparePlayers.map((p) => (
                        <th key={p.player_id} className="py-2 pr-3">
                          {p.player_name.split(' ').slice(-1)[0]}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {(
                      [
                        ['Club', (p: ScoutPlayer) => p.club],
                        ['Pos', (p: ScoutPlayer) => p.position],
                        ['Age', (p: ScoutPlayer) => String(p.age)],
                        ['Goals', (p: ScoutPlayer) => String(p.goals)],
                        ['Assists', (p: ScoutPlayer) => String(p.assists)],
                        ['Mins', (p: ScoutPlayer) => String(p.minutes)],
                        ['Value', (p: ScoutPlayer) => formatEur(p.market_value_eur)],
                        ['Cluster', (p: ScoutPlayer) => p.cluster_label],
                      ] as Array<[string, (p: ScoutPlayer) => string]>
                    ).map(([label, getter]) => (
                      <tr key={label} className="border-b border-pitch/10">
                        <td className="py-2 pr-3 text-ink/55">{label}</td>
                        {comparePlayers.map((p) => (
                          <td key={`${label}-${p.player_id}`} className="py-2 pr-3 font-semibold">
                            {getter(p)}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {clusters && (
            <div className="glass-panel p-5">
              <h3 className="font-display text-xl font-bold uppercase tracking-wide">
                Cluster map (PCA)
              </h3>
              <div className="mt-4 h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <ScatterChart margin={{ top: 8, right: 8, bottom: 8, left: 8 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#d9e5dc" />
                    <XAxis type="number" dataKey="pca_x" name="PC1" hide />
                    <YAxis type="number" dataKey="pca_y" name="PC2" hide />
                    <ZAxis range={[40, 40]} />
                    <Tooltip
                      cursor={{ strokeDasharray: '3 3' }}
                      content={({ payload }) => {
                        const p = payload?.[0]?.payload as ScoutPlayer | undefined
                        if (!p) return null
                        return (
                          <div className="border border-pitch/20 bg-white px-3 py-2 text-xs shadow-sm">
                            <p className="font-semibold">{p.player_name}</p>
                            <p className="text-ink/60">
                              {p.club} · {p.cluster_label}
                            </p>
                          </div>
                        )
                      }}
                    />
                    <Scatter
                      data={mapData}
                      onClick={(data) => {
                        const p = (data as { payload?: ScoutPlayer }).payload
                        if (p?.player_id) void runSimilar(p)
                      }}
                    >
                      {mapData.map((entry) => (
                        <Cell
                          key={`cell-${entry.player_id}`}
                          fill={CLUSTER_COLORS[entry.cluster_id % CLUSTER_COLORS.length]}
                        />
                      ))}
                    </Scatter>
                  </ScatterChart>
                </ResponsiveContainer>
              </div>
              <ul className="mt-4 space-y-2 text-sm">
                {clusters.clusters.map((c) => (
                  <li key={c.cluster_id} className="flex items-start gap-2">
                    <span
                      className="mt-1 inline-block h-2.5 w-2.5 rounded-full"
                      style={{
                        background: CLUSTER_COLORS[c.cluster_id % CLUSTER_COLORS.length],
                      }}
                    />
                    <span>
                      <span className="font-semibold">{c.label}</span>
                      <span className="text-ink/50">
                        {' '}
                        · {c.size} players · avg G/A {c.avg_goals}/{c.avg_assists}
                      </span>
                      {c.sample_players.length > 0 && (
                        <span className="block text-xs text-ink/45">
                          e.g. {c.sample_players.join(', ')}
                        </span>
                      )}
                    </span>
                  </li>
                ))}
              </ul>
              <p className="mt-3 text-xs text-ink/50">{clusters.dataset_note}</p>
            </div>
          )}

          {modelInfo && (
            <div className="glass-panel p-5">
              <h3 className="font-display text-xl font-bold uppercase tracking-wide">
                Model evaluation
              </h3>
              <dl className="mt-4 grid grid-cols-2 gap-3 text-sm">
                <div>
                  <dt className="text-xs uppercase tracking-wide text-ink/45">players</dt>
                  <dd className="font-semibold text-pitch-deep">
                    {String(modelInfo.metrics.n_players ?? '—')}
                  </dd>
                </div>
                <div>
                  <dt className="text-xs uppercase tracking-wide text-ink/45">clusters</dt>
                  <dd className="font-semibold text-pitch-deep">
                    {String(modelInfo.metrics.n_clusters ?? '—')}
                  </dd>
                </div>
                <div>
                  <dt className="text-xs uppercase tracking-wide text-ink/45">silhouette</dt>
                  <dd className="font-semibold text-pitch-deep">
                    {typeof modelInfo.metrics.silhouette_score === 'number'
                      ? modelInfo.metrics.silhouette_score.toFixed(3)
                      : '—'}
                  </dd>
                </div>
                <div>
                  <dt className="text-xs uppercase tracking-wide text-ink/45">model</dt>
                  <dd className="font-semibold text-pitch-deep text-xs leading-snug">
                    {modelInfo.model_name}
                  </dd>
                </div>
              </dl>
            </div>
          )}
        </section>
      </div>
    </>
  )
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <dt className="text-xs uppercase tracking-wide text-ink/45">{label}</dt>
      <dd className="font-semibold text-pitch-deep">{value}</dd>
    </div>
  )
}
