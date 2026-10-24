from pydantic import BaseModel, Field


class ScoutPlayer(BaseModel):
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
    cluster_id: int
    cluster_label: str
    pca_x: float
    pca_y: float
    similarity: float | None = None
    photo_url: str | None = None


class SimilarPlayersResponse(BaseModel):
    reference: ScoutPlayer
    similar: list[ScoutPlayer]
    dataset_note: str


class CompareResponse(BaseModel):
    players: list[ScoutPlayer]
    dataset_note: str


class ClusterInfo(BaseModel):
    cluster_id: int
    label: str
    size: int
    avg_goals: float
    avg_assists: float
    avg_age: float
    dominant_position: str
    sample_players: list[str] = Field(default_factory=list)


class ClustersResponse(BaseModel):
    clusters: list[ClusterInfo]
    map_points: list[ScoutPlayer]
    dataset_note: str


class ScoutingModelInfo(BaseModel):
    model_name: str
    metrics: dict
    features: list[str]
    dataset_note: str
