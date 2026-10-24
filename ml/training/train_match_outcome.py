"""Train Module 2 match-outcome classifier on real football-data.co.uk results."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = ROOT / "data" / "raw" / "premier_league_matches.csv"
PROCESSED_PATH = ROOT / "data" / "processed" / "match_features.csv"
MODEL_DIR = ROOT / "ml" / "models"
MODEL_PATH = MODEL_DIR / "match_outcome_logreg.joblib"
METRICS_PATH = MODEL_DIR / "match_metrics.json"
FEATURE_META_PATH = MODEL_DIR / "match_feature_meta.json"

FEATURE_COLS = [
    "home_form_ppg",
    "away_form_ppg",
    "home_gf_avg",
    "home_ga_avg",
    "away_gf_avg",
    "away_ga_avg",
    "home_shots_avg",
    "away_shots_avg",
    "home_sot_avg",
    "away_sot_avg",
    "h2h_home_win_rate",
    "h2h_draw_rate",
]
TARGET = "result"
FORM_WINDOW = 5


def _points(result: str, home: bool) -> int:
    if result == "D":
        return 1
    if home:
        return 3 if result == "H" else 0
    return 3 if result == "A" else 0


def _get(row: object, name: str, default: float = 0.0) -> float:
    if hasattr(row, name):
        val = getattr(row, name)
        if pd.isna(val):
            return default
        return float(val)
    return default


def build_features(matches: pd.DataFrame) -> pd.DataFrame:
    df = matches.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    history: dict[str, list[dict]] = {
        t: [] for t in sorted(set(df["home_team"]) | set(df["away_team"]))
    }
    h2h: dict[tuple[str, str], list[str]] = {}
    rows: list[dict] = []

    for row in df.itertuples(index=False):
        home = row.home_team
        away = row.away_team
        key = (home, away)
        rev = (away, home)

        def team_form(team: str) -> dict[str, float]:
            past = history[team][-FORM_WINDOW:]
            if not past:
                return {
                    "ppg": 1.0,
                    "gf": 1.2,
                    "ga": 1.2,
                    "shots": 10.0,
                    "sot": 3.5,
                }
            return {
                "ppg": float(np.mean([p["points"] for p in past])),
                "gf": float(np.mean([p["gf"] for p in past])),
                "ga": float(np.mean([p["ga"] for p in past])),
                "shots": float(np.mean([p["shots"] for p in past])),
                "sot": float(np.mean([p["sot"] for p in past])),
            }

        hf = team_form(home)
        af = team_form(away)

        prior = h2h.get(key, []) + [
            ("H" if r == "A" else "A" if r == "H" else "D") for r in h2h.get(rev, [])
        ]
        if prior:
            h2h_home = prior.count("H") / len(prior)
            h2h_draw = prior.count("D") / len(prior)
        else:
            h2h_home, h2h_draw = 0.4, 0.25

        rows.append(
            {
                "match_id": row.match_id,
                "date": row.date,
                "season": row.season,
                "home_team": home,
                "away_team": away,
                "home_goals": row.home_goals,
                "away_goals": row.away_goals,
                "result": row.result,
                "home_form_ppg": hf["ppg"],
                "away_form_ppg": af["ppg"],
                "home_gf_avg": hf["gf"],
                "home_ga_avg": hf["ga"],
                "away_gf_avg": af["gf"],
                "away_ga_avg": af["ga"],
                "home_shots_avg": hf["shots"],
                "away_shots_avg": af["shots"],
                "home_sot_avg": hf["sot"],
                "away_sot_avg": af["sot"],
                "h2h_home_win_rate": h2h_home,
                "h2h_draw_rate": h2h_draw,
            }
        )

        history[home].append(
            {
                "points": _points(row.result, True),
                "gf": row.home_goals,
                "ga": row.away_goals,
                "shots": _get(row, "home_shots", 10.0),
                "sot": _get(row, "home_shots_on_target", 3.5),
            }
        )
        history[away].append(
            {
                "points": _points(row.result, False),
                "gf": row.away_goals,
                "ga": row.home_goals,
                "shots": _get(row, "away_shots", 10.0),
                "sot": _get(row, "away_shots_on_target", 3.5),
            }
        )
        h2h.setdefault(key, []).append(row.result)

    return pd.DataFrame(rows)


def train(test_size: float = 0.2, random_state: int = 42) -> dict:
    if not RAW_PATH.exists():
        raise FileNotFoundError(
            f"Missing {RAW_PATH}. Run: python -m ml.training.ingest_football_data"
        )

    matches = pd.read_csv(RAW_PATH)
    features = build_features(matches)
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(PROCESSED_PATH, index=False)

    # Time-aware split: train on earlier matches, test on later ones when possible
    features = features.sort_values("date")
    split_idx = int(len(features) * (1 - test_size))
    train_df = features.iloc[:split_idx]
    test_df = features.iloc[split_idx:]
    if test_df["result"].nunique() < 2:
        x_train, x_test, y_train, y_test = train_test_split(
            features[FEATURE_COLS],
            features[TARGET],
            test_size=test_size,
            random_state=random_state,
            stratify=features[TARGET],
        )
    else:
        x_train = train_df[FEATURE_COLS]
        x_test = test_df[FEATURE_COLS]
        y_train = train_df[TARGET]
        y_test = test_df[TARGET]

    pipeline = Pipeline(
        steps=[
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                ),
            ),
        ]
    )
    pipeline.fit(x_train, y_train)

    proba = pipeline.predict_proba(x_test)
    pred = pipeline.predict(x_test)
    classes = list(pipeline.named_steps["model"].classes_)

    y_onehot = np.zeros_like(proba)
    class_index = {c: i for i, c in enumerate(classes)}
    for i, label in enumerate(y_test):
        y_onehot[i, class_index[label]] = 1.0
    brier = float(np.mean(np.sum((proba - y_onehot) ** 2, axis=1)))

    metrics = {
        "model": "LogisticRegression",
        "accuracy": float(accuracy_score(y_test, pred)),
        "log_loss": float(log_loss(y_test, proba, labels=classes)),
        "brier_score": brier,
        "n_train": int(len(x_train)),
        "n_test": int(len(x_test)),
        "n_total": int(len(features)),
        "classes": classes,
        "confusion_matrix": confusion_matrix(y_test, pred, labels=classes).tolist(),
        "dataset": str(RAW_PATH.relative_to(ROOT)),
        "data_source": "football-data.co.uk Premier League (E0)",
        "split": "temporal",
        "features": FEATURE_COLS,
    }
    for i, label in enumerate(classes):
        metrics[f"brier_{label}"] = float(
            brier_score_loss((y_test == label).astype(int), proba[:, i])
        )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)

    coef = pipeline.named_steps["model"].coef_
    importance = np.mean(np.abs(coef), axis=0)
    ranked = sorted(
        zip(FEATURE_COLS, importance, strict=True),
        key=lambda x: x[1],
        reverse=True,
    )
    meta = {
        "features": FEATURE_COLS,
        "classes": classes,
        "feature_importance": [
            {"feature": name, "importance": float(score)} for name, score in ranked
        ],
        "data_source": "football-data.co.uk",
    }
    FEATURE_META_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("Match model training complete (real PL data)")
    print(
        json.dumps(
            {k: metrics[k] for k in ("accuracy", "log_loss", "brier_score", "n_total", "data_source")},
            indent=2,
        )
    )
    print(f"Model -> {MODEL_PATH}")
    return metrics


def main() -> None:
    train()


if __name__ == "__main__":
    main()
