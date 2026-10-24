from __future__ import annotations

import json
from functools import lru_cache

import joblib
import pandas as pd

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
    "Scouting uses real Premier League season stats (FPL history). "
    "Similarity is cosine distance on scaled performance features; "
    "clusters from K-Means. Not a live recruitment database."
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
    max_age: int | None = None,
    max_value: int | None = None,
) -> SimilarPlayersResponse:
    bundle = _load_bundle()
    players = _load_players()
    ids: list[int] = bundle["player_ids"]
    if player_id not in ids:
        raise ValueError("Player not found in scouting index")

    idx = ids.index(player_id)
    x_scaled = bundle["x_scaled"]
    # Ask for extra neighbors so filters still leave enough results
    k = min(len(ids), max(limit * 5, 20))
    distances, indices = bundle["nn"].kneighbors(x_scaled[idx : idx + 1], n_neighbors=k)
    distances = distances[0]
    indices = indices[0]

    ref_row = players[players["player_id"] == player_id].iloc[0]
    similar: list[ScoutPlayer] = []
    for dist, j in zip(distances, indices, strict=True):
        pid = ids[j]
        if pid == player_id:
            continue
        row = players[players["player_id"] == pid].iloc[0]
        if position and str(row["position"]) != position:
            continue
        if max_age is not None and int(row["age"]) > max_age:
            continue
        if max_value is not None and pd.notna(row["market_value_eur"]):
            if int(row["market_value_eur"]) > max_value:
                continue
        # cosine distance -> similarity percent
        sim = float(max(0.0, 1.0 - dist) * 100.0)
        similar.append(_row_to_player(row, similarity=round(sim, 1)))
        if len(similar) >= limit:
            break

    return SimilarPlayersResponse(
        reference=_row_to_player(ref_row),
        similar=similar,
        dataset_note=DATASET_NOTE,
    )


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
    bundle = _load_bundle()
    meta = bundle.get("cluster_meta") or _load_metrics().get("clusters", [])
    clusters: list[ClusterInfo] = []
    for item in meta:
        cid = int(item["cluster_id"])
        sample = (
            players[players["cluster_id"] == cid]
            .sort_values("goals", ascending=False)["player_name"]
            .head(4)
            .tolist()
        )
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
        dataset_note=DATASET_NOTE,
    )


def model_info() -> ScoutingModelInfo:
    metrics = _load_metrics()
    return ScoutingModelInfo(
        model_name=str(metrics.get("model", "KMeans+NearestNeighbors+PCA")),
        metrics=metrics,
        features=list(metrics.get("features", [])),
        dataset_note=DATASET_NOTE,
    )
