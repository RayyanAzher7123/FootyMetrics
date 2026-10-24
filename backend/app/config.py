from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "FootyMetrics"
    database_url: str = "postgresql+psycopg://footymetrics:footymetrics@localhost:5432/footymetrics"
    model_dir: Path = ROOT / "ml" / "models"
    data_dir: Path = ROOT / "data"
    cors_origins: list[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:4173",
    ]

    @property
    def valuation_model_path(self) -> Path:
        return self.model_dir / "valuation_linear.joblib"

    @property
    def metrics_path(self) -> Path:
        return self.model_dir / "metrics.json"

    @property
    def feature_meta_path(self) -> Path:
        return self.model_dir / "feature_meta.json"

    @property
    def players_csv_path(self) -> Path:
        return self.data_dir / "raw" / "premier_league_valuations.csv"

    @property
    def matches_csv_path(self) -> Path:
        return self.data_dir / "raw" / "premier_league_matches.csv"

    @property
    def match_model_path(self) -> Path:
        return self.model_dir / "match_outcome_logreg.joblib"

    @property
    def match_metrics_path(self) -> Path:
        return self.model_dir / "match_metrics.json"

    @property
    def match_feature_meta_path(self) -> Path:
        return self.model_dir / "match_feature_meta.json"

    @property
    def scouting_bundle_path(self) -> Path:
        return self.model_dir / "scouting_bundle.joblib"

    @property
    def scouting_metrics_path(self) -> Path:
        return self.model_dir / "scouting_metrics.json"

    @property
    def scouting_players_path(self) -> Path:
        return self.data_dir / "processed" / "scouting_players.csv"


settings = Settings()
