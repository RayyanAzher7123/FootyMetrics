import { useEffect, useState, type FormEvent } from 'react'
import { Tilt } from '../components/Tilt'
import { getMatchModelInfo, getTeams, predictMatch, previewMatch } from '../api'
import { clubIdentity } from '../data/clubs'
import type { MatchModelInfo, MatchPredictResponse, MatchPreview, TeamForm } from '../types'

export function MatchPage() {
  const [teams, setTeams] = useState<string[]>([])
  const [homeTeam, setHomeTeam] = useState('Arsenal')
  const [awayTeam, setAwayTeam] = useState('Liverpool')
  const [preview, setPreview] = useState<MatchPreview | null>(null)
  const [prediction, setPrediction] = useState<MatchPredictResponse | null>(null)
  const [modelInfo, setModelInfo] = useState<MatchModelInfo | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    getTeams()
      .then((list) => {
        setTeams(list)
        if (list.includes('Arsenal')) setHomeTeam('Arsenal')
        if (list.includes('Liverpool')) setAwayTeam('Liverpool')
      })
      .catch(() => setTeams([]))
    getMatchModelInfo()
      .then(setModelInfo)
      .catch(() => setModelInfo(null))
  }, [])

  useEffect(() => {
    if (!homeTeam || !awayTeam || homeTeam === awayTeam) {
      setPreview(null)
      return
    }
    const handle = window.setTimeout(() => {
      previewMatch(homeTeam, awayTeam)
        .then(setPreview)
        .catch(() => setPreview(null))
    }, 200)
    return () => window.clearTimeout(handle)
  }, [homeTeam, awayTeam])

  async function onSubmit(event: FormEvent) {
    event.preventDefault()
    setLoading(true)
    setError(null)
    try {
      const result = await predictMatch({ home_team: homeTeam, away_team: awayTeam })
      setPrediction(result)
      setPreview(result.preview)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Prediction failed')
      setPrediction(null)
    } finally {
      setLoading(false)
    }
  }

  function swapSides() {
    setHomeTeam(awayTeam)
    setAwayTeam(homeTeam)
    setPrediction(null)
  }

  const home = clubIdentity(homeTeam)
  const away = clubIdentity(awayTeam)
  const pick = prediction?.predicted_result

  return (
    <div className="space-y-6">
      <header className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <p className="text-xs font-semibold uppercase tracking-[0.22em] text-pitch">
            Match centre
          </p>
          <h1 className="font-display text-5xl font-extrabold uppercase leading-none tracking-wide text-white md:text-6xl">
            Who takes the points?
          </h1>
        </div>
        <p className="max-w-sm text-sm leading-relaxed text-white/60">
          Home win, draw, or away win from recent form, shots, and head-to-head history.
        </p>
      </header>

      <form onSubmit={onSubmit} className="space-y-4">
        <Tilt strength={12}>
        <div className="fixture-board overflow-hidden rounded-2xl">
          <div className="grid md:grid-cols-[1fr_88px_1fr]">
            <ClubPanel
              role="Home"
              team={homeTeam}
              teams={teams}
              identity={home}
              form={preview?.home}
              picked={pick === 'H'}
              align="left"
              onChange={(name) => {
                setHomeTeam(name)
                setPrediction(null)
              }}
            />
            <div className="flex flex-col items-center justify-center gap-3 bg-[#101820] px-2 py-6 text-white">
              <p className="text-[10px] font-semibold uppercase tracking-[0.25em] text-white/50">
                PL
              </p>
              <p className="font-display text-4xl font-extrabold tracking-wide">VS</p>
              <button
                type="button"
                onClick={swapSides}
                className="rounded-full border border-white/20 px-3 py-1 text-[10px] font-semibold uppercase tracking-wider text-white/80 hover:bg-white/10"
              >
                Swap
              </button>
            </div>
            <ClubPanel
              role="Away"
              team={awayTeam}
              teams={teams}
              identity={away}
              form={preview?.away}
              picked={pick === 'A'}
              align="right"
              onChange={(name) => {
                setAwayTeam(name)
                setPrediction(null)
              }}
            />
          </div>
        </div>
        </Tilt>

        <div className="flex flex-wrap items-center gap-3">
          <button
            type="submit"
            disabled={loading || homeTeam === awayTeam}
            className="rounded-xl bg-lime px-6 py-3 font-display text-xl font-bold uppercase tracking-wider text-pitch-deep shadow-[0_0_28px_rgba(214,255,74,0.35)] transition hover:brightness-110 disabled:opacity-60"
          >
            {loading ? 'Reading the game…' : 'Predict result'}
          </button>
          {homeTeam === awayTeam && (
            <p className="text-sm text-red-700">Pick two different clubs.</p>
          )}
          {error && <p className="text-sm text-red-700">{error}</p>}
        </div>
      </form>

      <div className="grid gap-4 lg:grid-cols-[1.15fr_0.85fr]">
        <section className="glass-panel p-5">
          <div className="flex items-baseline justify-between gap-3">
            <h2 className="font-display text-2xl font-bold uppercase tracking-wide">
              Probabilities
            </h2>
            {prediction && (
              <p className="font-display text-3xl font-extrabold uppercase text-lime">
                {prediction.predicted_label}
              </p>
            )}
          </div>
          {prediction ? (
            <ul className="mt-5 space-y-3">
              {prediction.probabilities.map((item) => {
                const winner = item.label === prediction.predicted_result
                const color =
                  item.label === 'H' ? home.primary : item.label === 'A' ? away.primary : '#8a928c'
                return (
                  <li key={item.label}>
                    <div className="mb-1 flex items-center justify-between text-sm">
                      <span className="font-semibold">{item.name}</span>
                      <span className="font-display text-2xl leading-none">{item.probability}%</span>
                    </div>
                    <div className="h-3 overflow-hidden rounded-full bg-mist">
                      <div
                        className="h-full rounded-full transition-all"
                        style={{
                          width: `${item.probability}%`,
                          background: color,
                          outline: winner ? `2px solid ${color}` : undefined,
                          outlineOffset: winner ? '2px' : undefined,
                        }}
                      />
                    </div>
                  </li>
                )
              })}
            </ul>
          ) : (
            <p className="mt-6 text-sm text-ink/55">
              Choose a fixture, then run a prediction. The call lands on the highest probability.
            </p>
          )}
          {prediction && (
            <p className="mt-5 text-xs leading-relaxed text-ink/45">{prediction.dataset_note}</p>
          )}
        </section>

        <section className="glass-panel p-5">
          <h2 className="font-display text-2xl font-bold uppercase tracking-wide">Head to head</h2>
          {preview ? (
            <>
              <div className="mt-4 grid grid-cols-3 text-center">
                <Record
                  label={`${home.short} wins`}
                  value={preview.head_to_head.home_wins}
                  color={home.primary}
                />
                <Record label="Draws" value={preview.head_to_head.draws} color="#8a928c" />
                <Record
                  label={`${away.short} wins`}
                  value={preview.head_to_head.away_wins}
                  color={away.primary}
                />
              </div>
              <ul className="mt-4 divide-y divide-pitch/10">
                {preview.head_to_head.recent.slice(0, 5).map((match) => (
                  <li
                    key={`${match.date}-${match.home_team}-${match.score}`}
                    className="grid grid-cols-[1fr_auto_1fr] items-center gap-2 py-2.5 text-sm"
                  >
                    <span className="truncate text-right font-medium">{match.home_team}</span>
                    <span className="rounded-md bg-[#101820] px-2 py-1 font-display text-lg tracking-wide text-white">
                      {match.score}
                    </span>
                    <span className="truncate font-medium">{match.away_team}</span>
                  </li>
                ))}
                {preview.head_to_head.recent.length === 0 && (
                  <li className="py-3 text-sm text-ink/45">No previous meetings in this dataset.</li>
                )}
              </ul>
              <p className="mt-2 text-xs text-ink/40">
                {preview.head_to_head.meetings} meetings on record
              </p>
            </>
          ) : (
            <p className="mt-4 text-sm text-ink/55">Head-to-head appears once both clubs are set.</p>
          )}
        </section>
      </div>

      {preview && (
        <section className="grid gap-4 sm:grid-cols-2">
          <StatStrip title="Home recent" form={preview.home} accent={home.primary} />
          <StatStrip title="Away recent" form={preview.away} accent={away.primary} />
        </section>
      )}

      {modelInfo && (
        <section className="glass-panel flex flex-wrap gap-x-6 gap-y-2 px-5 py-3 text-xs text-white/55">
          <span className="font-semibold uppercase tracking-wider text-ink/40">Model</span>
          <span>Accuracy {pct(modelInfo.metrics.accuracy)}</span>
          <span>Log loss {num(modelInfo.metrics.log_loss)}</span>
          <span>Matches {String(modelInfo.metrics.n_total ?? '—')}</span>
        </section>
      )}
    </div>
  )
}

function ClubPanel({
  role,
  team,
  teams,
  identity,
  form,
  picked,
  align,
  onChange,
}: {
  role: string
  team: string
  teams: string[]
  identity: ReturnType<typeof clubIdentity>
  form?: TeamForm
  picked: boolean
  align: 'left' | 'right'
  onChange: (name: string) => void
}) {
  return (
    <div
      className="relative min-h-64 px-5 py-6 md:px-7"
      style={{
        background: `linear-gradient(160deg, ${identity.primary} 0%, ${identity.secondary} 140%)`,
        color: identity.text,
      }}
    >
      <div className={`flex items-start justify-between gap-3 ${align === 'right' ? 'md:flex-row-reverse' : ''}`}>
        <div className={align === 'right' ? 'md:text-right' : ''}>
          <p className="text-[11px] font-semibold uppercase tracking-[0.22em] opacity-75">
        {role}
        {form?.form_season ? ` · ${form.form_season}` : ''}
      </p>
          <p className="mt-1 font-display text-4xl font-extrabold uppercase leading-none tracking-wide md:text-5xl">
            {identity.short}
          </p>
          <p className="mt-1 text-sm font-medium opacity-90">{team}</p>
        </div>
        <Crest src={identity.crest} name={team} />
      </div>

      {picked && (
        <p className="mt-4 inline-block rounded-full bg-white/20 px-3 py-1 text-[11px] font-bold uppercase tracking-wider">
          Model pick
        </p>
      )}

      <div className={`mt-5 flex ${align === 'right' ? 'md:justify-end' : ''}`}>
        <div className="inline-flex gap-1.5 rounded-full bg-black/40 px-2 py-1.5">
          {(form?.last_results ?? []).map((result, index) => (
            <span
              key={`${result}-${index}`}
              className="grid h-7 w-7 place-items-center rounded-full text-xs font-bold text-white"
              style={{ background: resultColor(result) }}
            >
              {result}
            </span>
          ))}
        </div>
      </div>

      <label className="mt-5 block">
        <span className="sr-only">{role} club</span>
        <select
          value={team}
          onChange={(event) => onChange(event.target.value)}
          className="w-full rounded-lg border border-white/25 bg-black/25 px-3 py-2 text-sm text-white outline-none backdrop-blur-sm"
        >
          {teams.map((name) => (
            <option key={`${role}-${name}`} value={name} className="text-ink">
              {name}
            </option>
          ))}
        </select>
      </label>
    </div>
  )
}

function Crest({ src, name }: { src: string; name: string }) {
  if (!src) {
    return (
      <div className="grid h-20 w-20 place-items-center rounded-full bg-white text-sm font-bold text-ink">
        {name.slice(0, 3).toUpperCase()}
      </div>
    )
  }
  return (
    <div className="crest-float grid h-20 w-20 place-items-center rounded-full bg-white shadow-[0_12px_30px_rgba(0,0,0,0.35)]">
      <img src={src} alt="" className="h-14 w-14 object-contain" />
    </div>
  )
}

function StatStrip({ title, form, accent }: { title: string; form: TeamForm; accent: string }) {
  return (
    <div className="glass-panel px-5 py-4">
      <p className="text-xs font-semibold uppercase tracking-[0.16em]" style={{ color: accent }}>
        {title}
        {form.form_season ? ` · ${form.form_season}` : ''}
      </p>
      <div className="mt-3 grid grid-cols-3 gap-3 text-sm">
        <div>
          <p className="text-xs uppercase text-ink/40">PPG</p>
          <p className="font-display text-3xl leading-none">{form.form_ppg.toFixed(2)}</p>
        </div>
        <div>
          <p className="text-xs uppercase text-ink/40">Goals for</p>
          <p className="font-display text-3xl leading-none">{form.goals_for_avg.toFixed(1)}</p>
        </div>
        <div>
          <p className="text-xs uppercase text-ink/40">Against</p>
          <p className="font-display text-3xl leading-none">{form.goals_against_avg.toFixed(1)}</p>
        </div>
      </div>
    </div>
  )
}

function Record({ label, value, color }: { label: string; value: number; color: string }) {
  return (
    <div>
      <p className="font-display text-4xl leading-none" style={{ color }}>
        {value}
      </p>
      <p className="mt-1 text-[11px] uppercase tracking-wide text-ink/45">{label}</p>
    </div>
  )
}

function resultColor(result: string) {
  if (result === 'W') return '#1f9d55'
  if (result === 'D') return '#d4a017'
  return '#9b1c1c'
}

function pct(value: unknown) {
  return typeof value === 'number' ? `${(value * 100).toFixed(1)}%` : '—'
}

function num(value: unknown) {
  return typeof value === 'number' ? value.toFixed(3) : '—'
}
