from __future__ import annotations

from functools import lru_cache

import pandas as pd

from backend.app.config import settings
from backend.app.schemas.valuation import PlayerSummary
from backend.app.services.photos import photo_url


@lru_cache(maxsize=1)
def _load_frame() -> pd.DataFrame:
    path = settings.players_csv_path
    if not path.exists():
        raise FileNotFoundError(
            f"Player dataset not found at {path}. "
            "Run: python -m ml.training.generate_sample_data"
        )
    return pd.read_csv(path)


def reload_store() -> None:
    _load_frame.cache_clear()


def _valuation_players(df: pd.DataFrame) -> pd.DataFrame:
    """Current-squad players, valued from the previous Premier League season.

    Club and photo stay on the latest season so the squad is current.
    Goals, minutes and the rest come from the season before, which is a
    finished sample rather than a partial campaign.
    """
    seasons = sorted(str(season) for season in df["season"].dropna().unique())
    if not seasons:
        return df.iloc[0:0].copy()
    current = seasons[-1]
    previous = seasons[-2] if len(seasons) > 1 else current
    current_rows = (
        df[df["season"] == current]
        .drop_duplicates("player_id", keep="last")
        .loc[:, ["player_id", "club", "player_name", "photo_code"]]
        .rename(
            columns={
                "club": "current_club",
                "player_name": "current_name",
                "photo_code": "current_photo",
            }
        )
    )
    previous_rows = df[df["season"] == previous].drop_duplicates("player_id", keep="last")
    merged = previous_rows.merge(current_rows, on="player_id", how="inner")
    merged["stats_club"] = merged["club"]
    merged["club"] = merged["current_club"]
    merged["player_name"] = merged["current_name"]
    if "current_photo" in merged.columns:
        merged["photo_code"] = merged["current_photo"].where(
            merged["current_photo"].notna(), merged.get("photo_code")
        )
    return merged.drop(columns=["current_club", "current_name", "current_photo"], errors="ignore")


def list_latest_players(query: str | None = None, limit: int = 30) -> list[PlayerSummary]:
    latest = _valuation_players(_load_frame())
    if query:
        q = query.strip().lower()
        latest = latest[
            latest["player_name"].str.lower().str.contains(q, na=False)
            | latest["club"].str.lower().str.contains(q, na=False)
        ].copy()

        def _rank(row: pd.Series) -> tuple[int, str]:
            name = str(row["player_name"]).lower()
            club = str(row["club"]).lower()
            if name == q:
                return (0, name)
            if name.startswith(q) or any(part.startswith(q) for part in name.split()):
                return (1, name)
            if q in name.split() or f" {q}" in f" {name}":
                return (2, name)
            if q in name:
                return (3, name)
            if club.startswith(q) or q in club:
                return (4, name)
            return (5, name)

        latest["_rank"] = latest.apply(_rank, axis=1)
        latest = latest.sort_values(["_rank", "player_name"]).head(limit)
        return [_row_to_summary(row) for _, row in latest.iterrows()]

    latest = latest.sort_values("player_name").head(limit)
    return [_row_to_summary(row) for _, row in latest.iterrows()]


def get_player(player_id: int) -> PlayerSummary | None:
    rows = _valuation_players(_load_frame())
    rows = rows[rows["player_id"] == player_id]
    if rows.empty:
        return None
    return _row_to_summary(rows.iloc[0])


def _row_to_summary(row: pd.Series) -> PlayerSummary:
    return PlayerSummary(
        player_id=int(row["player_id"]),
        player_name=str(row["player_name"]),
        position=str(row["position"]),
        club=str(row["club"]),
        age=int(row["age"]),
        season=str(row["season"]),
        appearances=int(row["appearances"]),
        goals=int(row["goals"]),
        assists=int(row["assists"]),
        minutes=int(row["minutes"]),
        goals_per90=float(row["goals_per90"]),
        assists_per90=float(row["assists_per90"]),
        market_value_eur=int(row["market_value_eur"])
        if pd.notna(row["market_value_eur"])
        else None,
        market_value_source=str(row["value_source"])
        if pd.notna(row.get("value_source"))
        else "fpl_price_proxy",
        stats_club=str(row["stats_club"]) if pd.notna(row.get("stats_club")) else None,
        photo_url=photo_url(row),
    )
