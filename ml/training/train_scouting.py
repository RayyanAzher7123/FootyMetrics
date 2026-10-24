"""Train Module 3 scouting: K-Means clusters + nearest-neighbor similarity + PCA map."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = ROOT / "data" / "raw" / "premier_league_valuations.csv"
PROCESSED_PATH = ROOT / "data" / "processed" / "scouting_players.csv"
MODEL_DIR = ROOT / "ml" / "models"
ARTIFACT_PATH = MODEL_DIR / "scouting_bundle.joblib"
METRICS_PATH = MODEL_DIR / "scouting_metrics.json"

FEATURE_COLS = [
    "age",
    "appearances",
    "goals",
    "assists",
    "minutes",
    "goals_per90",
    "assists_per90",
]
N_CLUSTERS = 8
N_NEIGHBORS = 25


def load_latest_players(path: Path = RAW_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run: python -m ml.training.ingest_fpl_valuations"
        )
    df = pd.read_csv(path)
    required = FEATURE_COLS + [
        "player_id",
        "player_name",
        "position",
        "club",
        "season",
        "market_value_eur",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing columns: {missing}")

    df = df.dropna(subset=FEATURE_COLS + ["player_id", "player_name"]).copy()
    # Current Premier League squads only — last season we ingested.
    current_season = df["season"].max()
    latest = (
        df[df["season"] == current_season]
        .sort_values(["player_id"])
        .drop_duplicates("player_id", keep="last")
        .reset_index(drop=True)
    )
    return latest


def train(n_clusters: int = N_CLUSTERS, random_state: int = 42) -> dict:
    players = load_latest_players()
    x = players[FEATURE_COLS].astype(float).to_numpy()

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    k = min(n_clusters, max(2, len(players) // 15))
    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(x_scaled)

    pca = PCA(n_components=2, random_state=random_state)
    coords = pca.fit_transform(x_scaled)

    nn = NearestNeighbors(n_neighbors=min(N_NEIGHBORS, len(players)), metric="cosine")
    nn.fit(x_scaled)

    sil = float(silhouette_score(x_scaled, labels)) if len(set(labels)) > 1 else 0.0

    players = players.copy()
    players["cluster_id"] = labels.astype(int)
    players["pca_x"] = coords[:, 0]
    players["pca_y"] = coords[:, 1]

    # Human-readable cluster tags from dominant position + scoring profile
    cluster_meta: list[dict] = []
    for cid in sorted(players["cluster_id"].unique()):
        group = players[players["cluster_id"] == cid]
        top_pos = group["position"].value_counts().idxmax()
        avg_g = float(group["goals"].mean())
        avg_a = float(group["assists"].mean())
        if top_pos == "GK":
            tag = "Goalkeepers"
        elif top_pos == "DEF":
            tag = "Defenders"
        elif avg_g >= group["goals"].median() and top_pos == "FWD":
            tag = "Attacking finishers"
        elif avg_a >= avg_g and top_pos == "MID":
            tag = "Creative midfielders"
        elif top_pos == "MID":
            tag = "Midfield engines"
        else:
            tag = f"{top_pos} profiles"
        cluster_meta.append(
            {
                "cluster_id": int(cid),
                "label": tag,
                "size": int(len(group)),
                "avg_goals": round(avg_g, 2),
                "avg_assists": round(avg_a, 2),
                "avg_age": round(float(group["age"].mean()), 1),
                "dominant_position": top_pos,
            }
        )
        players.loc[players["cluster_id"] == cid, "cluster_label"] = tag

    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    players.to_csv(PROCESSED_PATH, index=False)

    bundle = {
        "scaler": scaler,
        "kmeans": kmeans,
        "pca": pca,
        "nn": nn,
        "feature_cols": FEATURE_COLS,
        "player_ids": players["player_id"].tolist(),
        "x_scaled": x_scaled,
        "cluster_meta": cluster_meta,
    }
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, ARTIFACT_PATH)

    metrics = {
        "model": "KMeans+NearestNeighbors(cosine)+PCA",
        "n_players": int(len(players)),
        "n_clusters": int(k),
        "silhouette_score": sil,
        "pca_explained_variance_ratio": [float(v) for v in pca.explained_variance_ratio_],
        "features": FEATURE_COLS,
        "dataset": str(RAW_PATH.relative_to(ROOT)),
        "data_source": "FPL season stats + Transfermarkt labels (latest season per player)",
        "clusters": cluster_meta,
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("Scouting model training complete")
    print(json.dumps({k: metrics[k] for k in ("n_players", "n_clusters", "silhouette_score")}, indent=2))
    print(f"Artifact -> {ARTIFACT_PATH}")
    return metrics


def main() -> None:
    train()


if __name__ == "__main__":
    main()
