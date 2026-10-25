"""Train Module 3 scouting on per-90 rates, within position, above a minutes floor.

Playing time and age are not similarity features. Neighbours are other players
in the same position who played at least MIN_MINUTES in the last finished
Premier League season. The current squad stays searchable; short samples are
flagged and left out of the neighbour index.
"""

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
FPL_DIR = ROOT / "data" / "raw" / "fpl"
PROCESSED_PATH = ROOT / "data" / "processed" / "scouting_players.csv"
MODEL_DIR = ROOT / "ml" / "models"
ARTIFACT_PATH = MODEL_DIR / "scouting_bundle.joblib"
METRICS_PATH = MODEL_DIR / "scouting_metrics.json"

MIN_MINUTES = 900
RATE_FEATURES = [
    "goals_per90",
    "assists_per90",
    "xg_per90",
    "xa_per90",
    "goals_conceded_per90",
    "saves_per90",
    "clean_sheets_per90",
    "defensive_contribution_per90",
]
FPL_RATE_MAP = {
    "expected_goals_per_90": "xg_per90",
    "expected_assists_per_90": "xa_per90",
    "goals_conceded_per_90": "goals_conceded_per90",
    "saves_per_90": "saves_per90",
    "clean_sheets_per_90": "clean_sheets_per90",
    "defensive_contribution_per_90": "defensive_contribution_per90",
}
STAT_COLS = [
    "age",
    "appearances",
    "goals",
    "assists",
    "minutes",
    "goals_per90",
    "assists_per90",
    "season",
]
N_CLUSTERS = 8


def _fpl_slug(season: str) -> str:
    start, end = str(season).split("/")
    return f"{start}-{end}"


def _attach_rates(frame: pd.DataFrame, season: str) -> pd.DataFrame:
    path = FPL_DIR / f"{_fpl_slug(season)}_players_raw.csv"
    out = frame.copy()
    for col in FPL_RATE_MAP.values():
        out[col] = np.nan
    if not path.exists():
        print(f"No FPL file for {season}; extra per-90 rates left blank")
        return out
    fpl = pd.read_csv(path)
    keep = ["code", *[c for c in FPL_RATE_MAP if c in fpl.columns]]
    fpl = fpl[keep].copy()
    fpl["photo_code"] = pd.to_numeric(fpl["code"], errors="coerce")
    fpl = fpl.dropna(subset=["photo_code"]).drop_duplicates("photo_code", keep="last")
    renamed = fpl.rename(columns=FPL_RATE_MAP)
    out["photo_code"] = pd.to_numeric(out["photo_code"], errors="coerce")
    merged = out.drop(columns=list(FPL_RATE_MAP.values())).merge(
        renamed[["photo_code", *FPL_RATE_MAP.values()]],
        on="photo_code",
        how="left",
    )
    matched = int(merged["xg_per90"].notna().sum())
    print(f"FPL per-90 rates matched for {matched}/{len(merged)} current-squad players")
    return merged


def load_squad_profiles(path: Path = RAW_PATH) -> pd.DataFrame:
    """Current squad, profiled with the previous finished season when it exists."""
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run: python -m ml.training.ingest_fpl_valuations"
        )
    df = pd.read_csv(path)
    seasons = sorted(str(s) for s in df["season"].dropna().unique())
    if len(seasons) < 2:
        raise ValueError("Need a current squad season and a previous finished season")
    current, previous = seasons[-1], seasons[-2]
    df["season"] = df["season"].astype(str)

    squad = (
        df[df["season"] == current]
        .drop_duplicates("player_id", keep="last")
        .reset_index(drop=True)
    )
    previous_rows = (
        df[df["season"] == previous]
        .drop_duplicates("player_id", keep="last")
        .loc[:, ["player_id", *STAT_COLS]]
    )
    merged = squad.merge(previous_rows, on="player_id", how="left", suffixes=("", "_prev"))
    has_previous = merged["minutes_prev"].notna()
    for col in STAT_COLS:
        merged[col] = merged[f"{col}_prev"].where(has_previous, merged[col])
    merged["qualified"] = has_previous & (merged["minutes_prev"] >= MIN_MINUTES)
    merged["stats_season"] = previous
    merged = merged.drop(columns=[f"{c}_prev" for c in STAT_COLS])
    merged = _attach_rates(merged, previous)
    for col in RATE_FEATURES:
        if col not in merged.columns:
            merged[col] = 0.0
        merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0.0)
    print(
        f"Squad {current}: {len(merged)} players, "
        f"{int(merged['qualified'].sum())} with {MIN_MINUTES}+ minutes in {previous}"
    )
    return merged


def _cluster_labels(group: pd.DataFrame) -> str:
    counts = group["position"].value_counts(normalize=True)
    top_pos = str(counts.index[0])
    avg_g90 = float(group["goals_per90"].mean())
    avg_a90 = float(group["assists_per90"].mean())
    # A slim majority should not name the whole group. 11 defenders and 9
    # midfielders is a mixed profile, not a defender group.
    if float(counts.iloc[0]) < 0.6:
        if avg_a90 >= avg_g90 and avg_a90 >= 0.15:
            return "Creative profiles"
        if avg_g90 >= 0.25:
            return "Attacking profiles"
        return "Mixed profiles"
    if top_pos == "GK":
        return "Goalkeepers"
    if top_pos == "DEF" and avg_g90 < 0.15:
        return "Defenders"
    if top_pos == "FWD" or avg_g90 >= 0.45:
        return "Attacking finishers"
    if top_pos == "MID" and avg_a90 >= avg_g90:
        return "Creative midfielders"
    if top_pos == "MID":
        return "Midfield engines"
    return f"{top_pos} profiles"


def _position_spaces(qualified: pd.DataFrame) -> dict:
    spaces: dict[str, dict] = {}
    for position, group in qualified.groupby("position"):
        cols = [c for c in RATE_FEATURES if float(group[c].std(ddof=0)) > 1e-6]
        if len(group) < 5 or not cols:
            continue
        scaler = StandardScaler()
        scaled = scaler.fit_transform(group[cols].astype(float).to_numpy())
        nn = NearestNeighbors(n_neighbors=len(group), metric="cosine")
        nn.fit(scaled)
        spaces[str(position)] = {
            "columns": cols,
            "scaler": scaler,
            "nn": nn,
            "player_ids": [int(pid) for pid in group["player_id"].tolist()],
            "x_scaled": scaled,
        }
        print(f"  {position}: {len(group)} players, rates {', '.join(cols)}")
    return spaces


def train(n_clusters: int = N_CLUSTERS, random_state: int = 42) -> dict:
    players = load_squad_profiles()
    qualified = players[players["qualified"]].reset_index(drop=True)
    if len(qualified) < 40:
        raise ValueError(f"Only {len(qualified)} players cleared the minutes floor")

    usable = [c for c in RATE_FEATURES if float(qualified[c].std(ddof=0)) > 1e-6]
    scaler = StandardScaler()
    x_rates = scaler.fit_transform(qualified[usable].astype(float).to_numpy())
    position_oh = pd.get_dummies(qualified["position"]).to_numpy(dtype=float)
    x_cluster = np.hstack([x_rates, position_oh])

    k = min(n_clusters, max(2, len(qualified) // 15))
    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(x_cluster)
    pca = PCA(n_components=2, random_state=random_state)
    coords = pca.fit_transform(x_cluster)
    sil = float(silhouette_score(x_cluster, labels)) if len(set(labels)) > 1 else 0.0

    qualified = qualified.copy()
    qualified["cluster_id"] = labels.astype(int)
    qualified["pca_x"] = coords[:, 0]
    qualified["pca_y"] = coords[:, 1]

    stats_season = str(qualified["stats_season"].iloc[0])
    cluster_meta: list[dict] = []
    for cid in sorted(qualified["cluster_id"].unique()):
        group = qualified[qualified["cluster_id"] == cid]
        tag = _cluster_labels(group)
        cluster_meta.append(
            {
                "cluster_id": int(cid),
                "label": tag,
                "size": int(len(group)),
                "avg_goals": round(float(group["goals"].mean()), 2),
                "avg_assists": round(float(group["assists"].mean()), 2),
                "avg_age": round(float(group["age"].mean()), 1),
                "dominant_position": str(group["position"].value_counts().idxmax()),
            }
        )
        qualified.loc[qualified["cluster_id"] == cid, "cluster_label"] = tag

    players = players.copy()
    players["cluster_id"] = -1
    players["cluster_label"] = f"Under {MIN_MINUTES} minutes"
    players["pca_x"] = 0.0
    players["pca_y"] = 0.0
    qualified_idx = players["player_id"].isin(qualified["player_id"])
    overlay = qualified.set_index("player_id")
    for col in ("cluster_id", "cluster_label", "pca_x", "pca_y"):
        players.loc[qualified_idx, col] = players.loc[qualified_idx, "player_id"].map(overlay[col])

    print("Position neighbour spaces:")
    spaces = _position_spaces(qualified)

    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    players.to_csv(PROCESSED_PATH, index=False)

    bundle = {
        "feature_cols": RATE_FEATURES,
        "min_minutes": MIN_MINUTES,
        "stats_season": stats_season,
        "position_spaces": spaces,
        "cluster_meta": cluster_meta,
        "kmeans": kmeans,
        "pca": pca,
    }
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, ARTIFACT_PATH)

    metrics = {
        "model": "Per90 cosine within position + KMeans",
        "n_players": int(len(qualified)),
        "n_squad": int(len(players)),
        "n_clusters": int(k),
        "silhouette_score": sil,
        "min_minutes": MIN_MINUTES,
        "stats_season": stats_season,
        "pca_explained_variance_ratio": [float(v) for v in pca.explained_variance_ratio_],
        "features": RATE_FEATURES,
        "dataset": str(RAW_PATH.relative_to(ROOT)),
        "data_source": (
            f"{stats_season} per-90 rates for current-squad players with at least {MIN_MINUTES} minutes"
        ),
        "clusters": cluster_meta,
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps({k: metrics[k] for k in ("n_players", "n_squad", "n_clusters", "silhouette_score", "stats_season")}, indent=2))
    print(f"Artifact -> {ARTIFACT_PATH}")
    return metrics


def main() -> None:
    train()


if __name__ == "__main__":
    main()
