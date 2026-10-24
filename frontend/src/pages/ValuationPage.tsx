import { useEffect, useState } from 'react'
import { getModelInfo, predictValuation, searchPlayers } from '../api'
import { CountUp, PlayerPortrait } from '../components/PlayerPortrait'
import { ValueCompare } from '../components/ValueCompare'
import type { ModelInfoResponse, PlayerSummary, ValuationResponse } from '../types'

function formatEur(value: number | null | undefined): string {
  if (value == null) return '—'
  if (value >= 1_000_000) return `€${(value / 1_000_000).toFixed(1)}M`
  if (value >= 1_000) return `€${(value / 1_000).toFixed(0)}K`
  return `€${value}`
}

function formatMetric(key: string, value: number | string): string {
  if (typeof value === 'string') return value
  if (key === 'r2') return value.toFixed(3)
  if (key === 'mae' || key === 'rmse') return formatEur(Math.round(value))
  return String(value)
}

export function ValuationPage() {
  const [query, setQuery] = useState('')
  const [results, setResults] = useState<PlayerSummary[]>([])
  const [selected, setSelected] = useState<PlayerSummary | null>(null)
  const [prediction, setPrediction] = useState<ValuationResponse | null>(null)
  const [modelInfo, setModelInfo] = useState<ModelInfoResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [searching, setSearching] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    getModelInfo()
      .then(setModelInfo)
      .catch(() => setModelInfo(null))
  }, [])

  useEffect(() => {
    const handle = window.setTimeout(async () => {
      setSearching(true)
      try {
        setResults(await searchPlayers(query))
      } catch {
        setResults([])
      } finally {
        setSearching(false)
      }
    }, 250)
    return () => window.clearTimeout(handle)
  }, [query])

  async function applyPlayer(player: PlayerSummary) {
    setSelected(player)
    setPrediction(null)
    setError(null)
    setLoading(true)
    try {
      setPrediction(
        await predictValuation({
          age: player.age,
          position: player.position,
          appearances: player.appearances,
          goals: player.goals,
          assists: player.assists,
          minutes: player.minutes,
          player_name: player.player_name,
          club: player.club,
        }),
      )
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Prediction failed')
    } finally {
      setLoading(false)
    }
  }

  const quoted =
    selected?.market_value_source === 'transfermarkt' ? selected.market_value_eur : null
  const playedElsewhere =
    selected?.stats_club != null && selected.stats_club !== selected.club

  return (
    <>
      <section className="mb-10 max-w-2xl">
        <p className="mb-2 text-xs font-semibold uppercase tracking-[0.2em] text-pitch">
          Module 1 · Regression
        </p>
        <h1 className="font-display text-5xl font-extrabold uppercase leading-none tracking-wide text-white md:text-6xl">
          Transfer value predictor
        </h1>
        <p className="mt-4 max-w-xl text-base leading-relaxed text-white/65">
          Pick a current Premier League player. The model values them from last season’s
          finished stats, then sets that estimate beside the Transfermarkt quote.
        </p>
      </section>

      <div className="grid gap-8 lg:grid-cols-[1.05fr_0.95fr]">
        <section className="space-y-6">
          <div>
            <label className="mb-2 block text-sm font-semibold text-ink" htmlFor="search">
              Find a player
            </label>
            <input
              id="search"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Search name or club…"
              className="field-input"
            />
            <div className="mt-2 max-h-56 overflow-auto rounded-xl border border-white/10 bg-black/25">
              {searching && <p className="px-4 py-3 text-sm text-ink/50">Searching…</p>}
              {!searching && results.length === 0 && (
                <p className="px-4 py-3 text-sm text-ink/55">
                  No current player with a full previous Premier League season. Try another name.
                </p>
              )}
              {!searching &&
                results.map((player) => (
                  <button
                    key={player.player_id}
                    type="button"
                    onClick={() => applyPlayer(player)}
                    className={`flex w-full items-center justify-between gap-3 px-3 py-2 text-left text-sm transition hover:bg-white/10 ${
                      selected?.player_id === player.player_id ? 'bg-white/10' : ''
                    }`}
                  >
                    <span className="flex items-center gap-3">
                      <PlayerPortrait name={player.player_name} photoUrl={player.photo_url} size="sm" />
                      <span>
                        <span className="font-semibold">{player.player_name}</span>
                        <span className="text-ink/50"> · {player.club}</span>
                      </span>
                    </span>
                    <span className="font-display text-lg text-pitch">{player.position}</span>
                  </button>
                ))}
            </div>
            <p className="mt-2 text-xs text-ink/45">
              Squads are this season. The goals, assists and minutes are from the season before.
            </p>
          </div>

          {selected && (
            <div key={selected.player_id} className="player-enter glass-panel p-5">
              <div className="flex items-baseline justify-between gap-3">
                <h2 className="font-display text-2xl font-bold uppercase tracking-wide">
                  {selected.season} season
                </h2>
                <p className="text-xs uppercase tracking-wider text-ink/55">{selected.position}</p>
              </div>
              <p className="mt-1 text-sm text-ink/60">
                {playedElsewhere
                  ? `Now at ${selected.club}. These numbers are from ${selected.stats_club}.`
                  : selected.club}
              </p>
              <dl className="mt-4 grid grid-cols-2 gap-3 sm:grid-cols-3">
                {[
                  ['Age', selected.age],
                  ['Appearances', selected.appearances],
                  ['Minutes', selected.minutes],
                  ['Goals', selected.goals],
                  ['Assists', selected.assists],
                ].map(([label, value]) => (
                  <div key={String(label)} className="stat-pop rounded-xl bg-black/25 px-3 py-3">
                    <dt className="text-[10px] uppercase tracking-wider text-white/45">{label}</dt>
                    <dd className="font-display text-3xl leading-none">{value}</dd>
                  </div>
                ))}
              </dl>
            </div>
          )}
          {error && <p className="text-sm text-red-300">{error}</p>}
        </section>

        <section className="space-y-6">
          <div
            key={selected?.player_id ?? 'empty'}
            className="player-enter hero-slab relative overflow-hidden rounded-2xl px-6 py-8 text-chalk"
          >
            <div className="pointer-events-none absolute -right-8 top-0 h-40 w-40 rounded-full bg-lime/20 blur-2xl" />
            <div className="flex items-end justify-between gap-4">
              <div>
                <p className="text-xs font-semibold uppercase tracking-[0.18em] text-lime">
                  Estimated market value
                </p>
                <p className="mt-3 font-display text-6xl font-extrabold tracking-wide md:text-7xl">
                  {prediction ? (
                    <CountUp value={prediction.estimated_value_eur} format={formatEur} />
                  ) : (
                    '—'
                  )}
                </p>
                <p className="mt-3 text-sm text-chalk/70">
                  {loading
                    ? 'Reading last season…'
                    : selected
                      ? `${selected.player_name} · ${selected.club}`
                      : 'Select a player and the valuation arrives with them.'}
                </p>
              </div>
              {selected && (
                <PlayerPortrait name={selected.player_name} photoUrl={selected.photo_url} />
              )}
            </div>
            {quoted != null && (
              <p className="mt-4 text-sm text-chalk/80">
                Transfermarkt quote {formatEur(quoted)}
              </p>
            )}
          </div>

          {prediction && selected && (
            <ValueCompare estimate={prediction.estimated_value_eur} actual={quoted} />
          )}

          {modelInfo && (
            <div className="glass-panel p-5">
              <h3 className="font-display text-xl font-bold uppercase tracking-wide">
                Model evaluation
              </h3>
              <dl className="mt-4 grid grid-cols-2 gap-3 text-sm sm:grid-cols-3">
                {['mae', 'rmse', 'r2', 'n_train', 'n_test', 'model'].map((key) => {
                  const value = modelInfo.metrics[key]
                  if (value == null) return null
                  return (
                    <div key={key}>
                      <dt className="text-xs uppercase tracking-wide text-ink/45">{key}</dt>
                      <dd className="font-semibold text-lime">{formatMetric(key, value)}</dd>
                    </div>
                  )
                })}
              </dl>
            </div>
          )}
        </section>
      </div>
    </>
  )
}
