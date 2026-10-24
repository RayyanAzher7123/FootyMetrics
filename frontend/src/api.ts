import type {
  ClustersResponse,
  MatchModelInfo,
  MatchPredictRequest,
  MatchPredictResponse,
  MatchPreview,
  ModelInfoResponse,
  PlayerSummary,
  ScoutPlayer,
  ScoutingModelInfo,
  SimilarPlayersResponse,
  ValuationRequest,
  ValuationResponse,
} from './types'

const API_BASE = import.meta.env.VITE_API_BASE ?? ''

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: {
      'Content-Type': 'application/json',
      ...(init?.headers ?? {}),
    },
    ...init,
  })
  if (!res.ok) {
    const detail = await res.text()
    throw new Error(detail || `Request failed (${res.status})`)
  }
  return res.json() as Promise<T>
}

export function searchPlayers(q: string): Promise<PlayerSummary[]> {
  const params = new URLSearchParams()
  if (q.trim()) params.set('q', q.trim())
  params.set('limit', '50')
  return request(`/api/players?${params.toString()}`)
}

export function predictValuation(body: ValuationRequest): Promise<ValuationResponse> {
  return request('/api/valuation/predict', {
    method: 'POST',
    body: JSON.stringify(body),
  })
}

export function getModelInfo(): Promise<ModelInfoResponse> {
  return request('/api/valuation/model')
}

export function getTeams(): Promise<string[]> {
  return request('/api/matches/teams')
}

export function previewMatch(homeTeam: string, awayTeam: string): Promise<MatchPreview> {
  const params = new URLSearchParams({
    home_team: homeTeam,
    away_team: awayTeam,
  })
  return request(`/api/matches/preview?${params.toString()}`)
}

export function predictMatch(body: MatchPredictRequest): Promise<MatchPredictResponse> {
  return request('/api/matches/predict', {
    method: 'POST',
    body: JSON.stringify(body),
  })
}

export function getMatchModelInfo(): Promise<MatchModelInfo> {
  return request('/api/matches/model')
}

export function searchScoutPlayers(q: string): Promise<ScoutPlayer[]> {
  const params = new URLSearchParams()
  if (q.trim()) params.set('q', q.trim())
  params.set('limit', '40')
  return request(`/api/scouting/players?${params.toString()}`)
}

export function getSimilarPlayers(
  playerId: number,
  opts: { limit?: number; position?: string; maxAge?: number; maxValue?: number } = {},
): Promise<SimilarPlayersResponse> {
  const params = new URLSearchParams()
  params.set('limit', String(opts.limit ?? 8))
  if (opts.position) params.set('position', opts.position)
  if (opts.maxAge != null) params.set('max_age', String(opts.maxAge))
  if (opts.maxValue != null) params.set('max_value', String(opts.maxValue))
  return request(`/api/scouting/similar/${playerId}?${params.toString()}`)
}

export function compareScoutPlayers(ids: number[]): Promise<{ players: ScoutPlayer[]; dataset_note: string }> {
  return request(`/api/scouting/compare?ids=${ids.join(',')}`)
}

export function getClusters(): Promise<ClustersResponse> {
  return request('/api/scouting/clusters')
}

export function getScoutingModelInfo(): Promise<ScoutingModelInfo> {
  return request('/api/scouting/model')
}
