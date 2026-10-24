from pydantic import BaseModel, Field


class MatchPredictRequest(BaseModel):
    home_team: str = Field(..., min_length=2)
    away_team: str = Field(..., min_length=2)


class TeamForm(BaseModel):
    team: str
    form_ppg: float
    goals_for_avg: float
    goals_against_avg: float
    last_results: list[str]
    form_season: str = ""


class HeadToHeadSummary(BaseModel):
    meetings: int
    home_wins: int
    draws: int
    away_wins: int
    recent: list[dict]


class MatchPreviewResponse(BaseModel):
    home: TeamForm
    away: TeamForm
    head_to_head: HeadToHeadSummary


class OutcomeProbability(BaseModel):
    label: str
    name: str
    probability: float


class MatchPredictResponse(BaseModel):
    home_team: str
    away_team: str
    probabilities: list[OutcomeProbability]
    predicted_result: str
    predicted_label: str
    confidence: float
    top_factors: list[dict]
    model_name: str
    dataset_note: str
    metrics: dict
    preview: MatchPreviewResponse


class MatchModelInfoResponse(BaseModel):
    model_name: str
    metrics: dict
    top_factors: list[dict]
    features: list[str]
    dataset_note: str
