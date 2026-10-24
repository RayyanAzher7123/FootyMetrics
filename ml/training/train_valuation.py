"""Train and evaluate the Module 1 transfer-value regression baseline."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = ROOT / "data" / "raw" / "premier_league_valuations.csv"
PROCESSED_PATH = ROOT / "data" / "processed" / "valuation_features.csv"
MODEL_DIR = ROOT / "ml" / "models"
MODEL_PATH = MODEL_DIR / "valuation_linear.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"
FEATURE_META_PATH = MODEL_DIR / "feature_meta.json"

NUMERIC_FEATURES = [
    "age",
    "appearances",
    "goals",
    "assists",
    "minutes",
    "goals_per90",
    "assists_per90",
]
CATEGORICAL_FEATURES = ["position"]
TARGET = "market_value_eur"


def load_dataset(path: Path = RAW_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run: python -m ml.training.ingest_fpl_valuations"
        )
    df = pd.read_csv(path)
    required = NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET, "player_name", "club", "season"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing columns: {missing}")
    df = df.dropna(subset=NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET]).copy()
    # Prefer real Transfermarkt labels when the ingest pipeline provides them
    if "value_source" in df.columns and (df["value_source"] == "transfermarkt").any():
        labelled = df[df["value_source"] == "transfermarkt"].copy()
        seasons = sorted(labelled["season"].unique())
        # The latest season is still in progress. Train on the finished one
        # before it, valued with the June 2026 squad quotes.
        if len(seasons) >= 2:
            finished = labelled[labelled["season"] == seasons[-2]]
            if len(finished) >= 80:
                labelled = finished
        if len(labelled) >= 80:
            print(f"Training on {len(labelled)} Transfermarkt-labelled rows "
                  f"(of {len(df)} total search rows)")
            return labelled
    return df


def build_pipeline() -> Pipeline:
    preprocess = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("model", LinearRegression()),
        ]
    )


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    return {"mae": mae, "rmse": rmse, "r2": r2}


def feature_coefficients(pipeline: Pipeline) -> list[dict]:
    preprocess: ColumnTransformer = pipeline.named_steps["preprocess"]
    model: LinearRegression = pipeline.named_steps["model"]
    feature_names = preprocess.get_feature_names_out()
    coefs = model.coef_
    pairs = sorted(
        zip(feature_names, coefs, strict=True),
        key=lambda x: abs(x[1]),
        reverse=True,
    )
    return [
        {"feature": name.replace("num__", "").replace("cat__", ""), "coefficient": float(c)}
        for name, c in pairs
    ]


def train(test_size: float = 0.2, random_state: int = 42) -> dict:
    df = load_dataset()
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_PATH, index=False)

    x = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET].astype(float)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state
    )

    pipeline = build_pipeline()
    pipeline.fit(x_train, y_train)
    y_pred = pipeline.predict(x_test)
    # Market values cannot be negative
    y_pred = np.clip(y_pred, 0, None)

    metrics = evaluate(y_test.to_numpy(), y_pred)
    metrics["n_train"] = int(len(x_train))
    metrics["n_test"] = int(len(x_test))
    metrics["n_total"] = int(len(df))
    metrics["model"] = "LinearRegression"
    metrics["target"] = TARGET
    metrics["dataset"] = str(RAW_PATH.relative_to(ROOT))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)

    meta = {
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "target": TARGET,
        "coefficients": feature_coefficients(pipeline),
    }
    FEATURE_META_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("Training complete")
    print(json.dumps(metrics, indent=2))
    print(f"Model -> {MODEL_PATH}")
    return metrics


def main() -> None:
    train()


if __name__ == "__main__":
    main()
