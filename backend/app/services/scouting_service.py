from __future__ import annotations

import json
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_distances

from backend.app.config import settings
from backend.app.services.photos import photo_url
from backend.app.schemas.scouting import (
    ClusterInfo,
    ClustersResponse,
    CompareResponse,
    ScoutingModelInfo,
    ScoutPlayer,
    SimilarPlayersResponse,
)

DATASET_NOTE = (
    "Similar players share a position and a per-90 profile from the last finished season. "
    "Short appearances are left out of the comparison. Not a live recruitment database."
)


@lru_cache(maxsize=1)
def _load_bundle():
    path = settings.scouting_bundle_path
    if not path.exists():
        raise FileNotFoundError(
            f"Scouting model not found at {path}. Run: python -m ml.training.train_scouting"
        )
    return joblib.load(path)


@lru_cache(maxsize=1)
def _load_players() -> pd.DataFrame:
    path = settings.scouting_players_path
    if not path.exists():
        raise FileNotFoundError(
            f"Scouting player table not found at {path}. Run: python -m ml.training.train_scouting"
        )
    return pd.read_csv(path)


@lru_cache(maxsize=1)
def _load_metrics() -> dict:
    path = settings.scouting_metrics_path
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def reload_scouting() -> None:
    _load_bundle.cache_clear()
    _load_players.cache_clear()
    _load_metrics.cache_clear()


def _row_to_player(row: pd.Series, similarity: float | None = None) -> ScoutPlayer:
    return ScoutPlayer(
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
        if pd.notna(row.get("market_value_eur"))
        else None,
        cluster_id=int(row["cluster_id"]),
        cluster_label=str(row.get("cluster_label", f"Cluster {row['cluster_id']}")),
        pca_x=float(row["pca_x"]),
        pca_y=float(row["pca_y"]),
        similarity=similarity,
        photo_url=photo_url(row),
        qualified=_as_bool(row.get("qualified", True)),
    )


def _as_bool(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().lower() == "true"
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return False
    return bool(value)


def _floor() -> tuple[str, int]:
    bundle = _load_bundle()
    metrics = _load_metrics()
    season = str(bundle.get("stats_season") or metrics.get("stats_season") or "2025/26")
    minutes = int(bundle.get("min_minutes") or metrics.get("min_minutes") or 900)
    return season, minutes


def _note(*, qualified: bool, has_previous: bool) -> str:
    season, minutes = _floor()
    if not has_previous:
        return (
            f"Comparisons use {season} per-90 rates inside one position, "
            f"for players with at least {minutes} minutes. "
            "This player has no finished Premier League season on file."
        )
    if not qualified:
        return (
            f"Comparisons use {season} per-90 rates inside one position, "
            f"for players with at least {minutes} minutes. "
            "This player is under that minutes floor, so no one is recommended."
        )
    return (
        f"Same position, at least {minutes} minutes in {season}. "
        "Distance uses per-90 rates. Age and total minutes are not part of it."
    )


def search_players(query: str | None = None, limit: int = 40) -> list[ScoutPlayer]:
    df = _load_players()
    if query:
        q = query.strip().lower()
        df = df[
            df["player_name"].str.lower().str.contains(q, na=False)
            | df["club"].str.lower().str.contains(q, na=False)
        ].copy()

        def _rank(row: pd.Series) -> tuple[int, str]:
            name = str(row["player_name"]).lower()
            if name == q:
                return (0, name)
            if any(part.startswith(q) for part in name.split()):
                return (1, name)
            if q in name:
                return (2, name)
            return (3, name)

        df["_rank"] = df.apply(_rank, axis=1)
        df = df.sort_values(["_rank", "player_name"])
    else:
        df = df.sort_values("player_name")
    return [_row_to_player(row) for _, row in df.head(limit).iterrows()]


def get_player(player_id: int) -> ScoutPlayer | None:
    df = _load_players()
    rows = df[df["player_id"] == player_id]
    if rows.empty:
        return None
    return _row_to_player(rows.iloc[0])


def find_similar(
    player_id: int,
    limit: int = 8,
    position: str | None = None,
    same_position: bool = True,
    max_age: int | None = None,
    max_value: int | None = None,
) -> SimilarPlayersResponse:
    players = _load_players()
    rows = players[players["player_id"] == player_id]
    if rows.empty:
        raise ValueError("Player not found in scouting index")
    ref_row = rows.iloc[0]
    qualified = _as_bool(ref_row.get("qualified", True))
    has_previous = str(ref_row["season"]) == _floor()[0]
    if not qualified:
        return SimilarPlayersResponse(
            reference=_row_to_player(ref_row),
            similar=[],
            dataset_note=_note(qualified=False, has_previous=has_previous),
        )

    target = position or (str(ref_row["position"]) if same_position else None)
    spaces = _load_bundle().get("position_spaces") or {}
    space = spaces.get(target) if target else None
    if space is None and target is None:
        # Any-position request: score each position space with that position's scaler.
        similar = _similar_any_position(ref_row, limit, max_age, max_value)
        return SimilarPlayersResponse(
            reference=_row_to_player(ref_row),
            similar=similar,
            dataset_note=_note(qualified=True, has_previous=True),
        )
    if space is None:
        return SimilarPlayersResponse(
            reference=_row_to_player(ref_row),
            similar=[],
            dataset_note=_note(qualified=True, has_previous=True),
        )

    similar = _rank_space(ref_row, space, limit, max_age, max_value)
    return SimilarPlayersResponse(
        reference=_row_to_player(ref_row),
        similar=similar,
        dataset_note=_note(qualified=True, has_previous=True),
    )


def _reference_vector(ref_row: pd.Series, space: dict) -> np.ndarray:
    ids: list[int] = space["player_ids"]
    player_id = int(ref_row["player_id"])
    if player_id in ids and str(ref_row["position"]) == _space_position(space, player_id):
        idx = ids.index(player_id)
        return space["x_scaled"][idx : idx + 1]
    raw = np.array([[float(ref_row.get(col, 0.0)) for col in space["columns"]]])
    return space["scaler"].transform(raw)


def _space_position(space: dict, player_id: int) -> str:
    players = _load_players()
    match = players.loc[players["player_id"] == player_id, "position"]
    return str(match.iloc[0]) if not match.empty else ""


def _rank_space(
    ref_row: pd.Series,
    space: dict,
    limit: int,
    max_age: int | None,
    max_value: int | None,
) -> list[ScoutPlayer]:
    players = _load_players()
    ids: list[int] = space["player_ids"]
    vec = _reference_vector(ref_row, space)
    distances = cosine_distances(vec, space["x_scaled"])[0]
    order = np.argsort(distances)
    similar: list[ScoutPlayer] = []
    for j in order:
        pid = ids[int(j)]
        if pid == int(ref_row["player_id"]):
            continue
        row = players[players["player_id"] == pid]
        if row.empty:
            continue
        row = row.iloc[0]
        if max_age is not None and int(row["age"]) > max_age:
            continue
        if max_value is not None and pd.notna(row["market_value_eur"]):
            if int(row["market_value_eur"]) > max_value:
                continue
        sim = float(max(0.0, 1.0 - float(distances[j])) * 100.0)
        similar.append(_row_to_player(row, similarity=round(sim, 1)))
        if len(similar) >= limit:
            break
    return similar


def _similar_any_position(
    ref_row: pd.Series,
    limit: int,
    max_age: int | None,
    max_value: int | None,
) -> list[ScoutPlayer]:
    found: list[ScoutPlayer] = []
    for space in (_load_bundle().get("position_spaces") or {}).values():
        found.extend(_rank_space(ref_row, space, limit, max_age, max_value))
    found.sort(key=lambda player: player.similarity or 0, reverse=True)
    return found[:limit]


def compare_players(player_ids: list[int]) -> CompareResponse:
    if not player_ids:
        raise ValueError("Provide at least one player id")
    if len(player_ids) > 5:
        raise ValueError("Compare at most 5 players")
    players = []
    for pid in player_ids:
        player = get_player(pid)
        if player is None:
            raise ValueError(f"Player not found: {pid}")
        players.append(player)
    return CompareResponse(players=players, dataset_note=DATASET_NOTE)


def list_clusters(map_limit: int = 200) -> ClustersResponse:
    players = _load_players()
    players = players[players["cluster_id"] >= 0] if "cluster_id" in players.columns else players
    bundle = _load_bundle()
    meta = bundle.get("cluster_meta") or _load_metrics().get("clusters", [])
    clusters: list[ClusterInfo] = []
    for item in meta:
        cid = int(item["cluster_id"])
        dominant = str(item.get("dominant_position", ""))
        pool = players[players["cluster_id"] == cid]
        named = pool[pool["position"] == dominant] if dominant else pool
        if named.empty:
            named = pool
        sample = named.sort_values("goals", ascending=False)["player_name"].head(4).tolist()
        clusters.append(
            ClusterInfo(
                cluster_id=cid,
                label=str(item.get("label", f"Cluster {cid}")),
                size=int(item.get("size", 0)),
                avg_goals=float(item.get("avg_goals", 0)),
                avg_assists=float(item.get("avg_assists", 0)),
                avg_age=float(item.get("avg_age", 0)),
                dominant_position=str(item.get("dominant_position", "")),
                sample_players=sample,
            )
        )

    # Spread map points: sample across clusters
    points: list[ScoutPlayer] = []
    per = max(5, map_limit // max(len(clusters), 1))
    for cid in players["cluster_id"].unique():
        group = players[players["cluster_id"] == cid].head(per)
        points.extend(_row_to_player(row) for _, row in group.iterrows())
    points = points[:map_limit]

    return ClustersResponse(
        clusters=sorted(clusters, key=lambda c: c.cluster_id),
        map_points=points,
        dataset_note=_note(qualified=True, has_previous=True),
    )


def model_info() -> ScoutingModelInfo:
    metrics = _load_metrics()
    return ScoutingModelInfo(
        model_name=str(metrics.get("model", "KMeans+NearestNeighbors+PCA")),
        metrics=metrics,
        features=list(metrics.get("features", [])),
        dataset_note=_note(qualified=True, has_previous=True),
    )
