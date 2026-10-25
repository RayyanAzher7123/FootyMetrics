export type Position = 'GK' | 'DEF' | 'MID' | 'FWD'

export interface PlayerSummary {
  player_id: number
  player_name: string
  position: Position
  club: string
  age: number
  season: string
  appearances: number
  goals: number
  assists: number
  minutes: number
  goals_per90: number
  assists_per90: number
  market_value_eur: number | null
  market_value_source?: string | null
  stats_club?: string | null
  photo_url?: string | null
}

export interface ValuationRequest {
  age: number
  position: Position
  appearances: number
  goals: number
  assists: number
  minutes: number
  player_name?: string | null
  club?: string | null
}

export interface FeatureContribution {
  feature: string
  coefficient: number
}

export interface ValuationResponse {
  estimated_value_eur: number
  estimated_value_display: string
  inputs: ValuationRequest
  top_factors: FeatureContribution[]
  model_name: string
  dataset_note: string
  metrics: Record<string, number | string>
}

export interface ModelScore {
  model: string
  mae: number
  rmse: number
  r2: number
}

export interface ValueBandError {
  band: string
  n: number
  mae: number
  rmse: number
}

export interface ModelInfoResponse {
  model_name: string
  metrics: Record<string, number | string>
  top_factors: FeatureContribution[]
  numeric_features: string[]
  categorical_features: string[]
  dataset_note: string
  comparison?: ModelScore[]
  value_bands?: ValueBandError[]
}

export interface TeamForm {
  team: string
  form_ppg: number
  goals_for_avg: number
  goals_against_avg: number
  last_results: string[]
  form_season?: string
}

export interface MatchPreview {
  home: TeamForm
  away: TeamForm
  head_to_head: {
    meetings: number
    home_wins: number
    draws: number
    away_wins: number
    recent: Array<{
      date: string
      home_team: string
      away_team: string
      score: string
      result: string
    }>
  }
}

export interface MatchPredictRequest {
  home_team: string
  away_team: string
}

export interface MatchPredictResponse {
  home_team: string
  away_team: string
  probabilities: Array<{ label: string; name: string; probability: number }>
  predicted_result: string
  predicted_label: string
  confidence: number
  top_factors: Array<{ feature: string; importance: number }>
  model_name: string
  dataset_note: string
  metrics: Record<string, number | string | number[][] | string[]>
  preview: MatchPreview
}

export interface MatchModelInfo {
  model_name: string
  metrics: Record<string, number | string | number[][] | string[]>
  top_factors: Array<{ feature: string; importance: number }>
  features: string[]
  dataset_note: string
}

export interface ScoutPlayer {
  player_id: number
  player_name: string
  position: string
  club: string
  age: number
  season: string
  appearances: number
  goals: number
  assists: number
  minutes: number
  goals_per90: number
  assists_per90: number
  market_value_eur: number | null
  cluster_id: number
  cluster_label: string
  pca_x: number
  pca_y: number
  similarity?: number | null
  photo_url?: string | null
  qualified?: boolean
}

export interface SimilarPlayersResponse {
  reference: ScoutPlayer
  similar: ScoutPlayer[]
  dataset_note: string
}

export interface ClustersResponse {
  clusters: Array<{
    cluster_id: number
    label: string
    size: number
    avg_goals: number
    avg_assists: number
    avg_age: number
    dominant_position: string
    sample_players: string[]
  }>
  map_points: ScoutPlayer[]
  dataset_note: string
}

export interface ScoutingModelInfo {
  model_name: string
  metrics: Record<string, number | string | number[] | object>
  features: string[]
  dataset_note: string
}
