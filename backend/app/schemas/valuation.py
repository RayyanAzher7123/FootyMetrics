from pydantic import BaseModel, Field


class PlayerSummary(BaseModel):
    player_id: int
    player_name: str
    position: str
    club: str
    age: int
    season: str
    appearances: int
    goals: int
    assists: int
    minutes: int
    goals_per90: float
    assists_per90: float
    market_value_eur: int | None = None
    market_value_source: str = "fpl_price_proxy"
    stats_club: str | None = None
    photo_url: str | None = None


class ValuationRequest(BaseModel):
    age: int = Field(..., ge=15, le=45)
    position: str = Field(..., pattern="^(GK|DEF|MID|FWD)$")
    appearances: int = Field(..., ge=0, le=50)
    goals: int = Field(..., ge=0, le=60)
    assists: int = Field(..., ge=0, le=40)
    minutes: int = Field(..., ge=0, le=5000)
    player_name: str | None = None
    club: str | None = None


class FeatureContribution(BaseModel):
    feature: str
    coefficient: float


class ValuationResponse(BaseModel):
    estimated_value_eur: int
    estimated_value_display: str
    inputs: ValuationRequest
    top_factors: list[FeatureContribution]
    model_name: str
    dataset_note: str
    metrics: dict[str, float | int | str]


class ModelScore(BaseModel):
    model: str
    mae: float
    rmse: float
    r2: float


class ValueBandError(BaseModel):
    band: str
    n: int
    mae: float
    rmse: float


class ModelInfoResponse(BaseModel):
    model_name: str
    metrics: dict[str, float | int | str]
    top_factors: list[FeatureContribution]
    numeric_features: list[str]
    categorical_features: list[str]
    dataset_note: str
    comparison: list[ModelScore] = []
    value_bands: list[ValueBandError] = []
