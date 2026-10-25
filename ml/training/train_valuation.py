"""Train Module 1 and score it on a later season, not a random split of the same one.

Labels are date-aligned Transfermarkt quotes. The comparison fits each
candidate on every finished season except the latest, scores that holdout,
and keeps the lowest holdout MAE. The saved model is then refit on all
finished seasons so the app can use the extra year. Reported error stays
the holdout figure.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
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
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

VALUE_BANDS = (
    ("Under €10M", 0, 10_000_000),
    ("€10–30M", 10_000_000, 30_000_000),
    ("€30–60M", 30_000_000, 60_000_000),
    ("Over €60M", 60_000_000, None),
)


def load_dataset(path: Path = RAW_PATH) -> pd.DataFrame:
    """Finished seasons whose quote date sits near the end of that season."""
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run: python -m ml.training.ingest_fpl_valuations"
        )
    df = pd.read_csv(path)
    required = FEATURES + [TARGET, "player_name", "club", "season", "value_source", "valuation_date"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing columns: {missing}")

    seasons = sorted(df["season"].astype(str).unique())
    if len(seasons) < 2:
        raise ValueError("Need at least one finished season and one later season")
    in_progress = seasons[-1]
    labelled = df[
        (df["value_source"] == "transfermarkt") & (df["season"].astype(str) != in_progress)
    ].copy()
    labelled = labelled.dropna(subset=FEATURES + [TARGET, "valuation_date"])
    if len(labelled) < 80:
        raise ValueError(
            f"Only {len(labelled)} date-aligned rows. Re-run ingest_fpl_valuations."
        )
    print(
        f"Finished labelled seasons: {sorted(labelled['season'].astype(str).unique())} "
        f"({len(labelled)} rows; {in_progress} held out of training as in progress)"
    )
    return labelled


def _estimator(name: str, random_state: int):
    alphas = np.logspace(-2, 3, 12)
    if name == "LinearRegression":
        return LinearRegression()
    if name == "Ridge":
        return RidgeCV(alphas=alphas)
    if name == "RidgeLog":
        return TransformedTargetRegressor(
            regressor=RidgeCV(alphas=alphas),
            func=np.log1p,
            inverse_func=np.expm1,
        )
    if name == "HistGradientBoosting":
        return HistGradientBoostingRegressor(
            max_depth=3,
            learning_rate=0.06,
            max_iter=400,
            min_samples_leaf=15,
            l2_regularization=1.0,
            random_state=random_state,
        )
    if name == "RandomForest":
        return RandomForestRegressor(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=5,
            random_state=random_state,
            n_jobs=1,
        )
    raise KeyError(name)


def build_pipeline(estimator) -> Pipeline:
    preprocess = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ]
    )
    return Pipeline(steps=[("preprocess", preprocess), ("model", estimator)])


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    return {"mae": mae, "rmse": rmse, "r2": r2}


def value_band_errors(y_true: np.ndarray, y_pred: np.ndarray) -> list[dict]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    rows: list[dict] = []
    for name, low, high in VALUE_BANDS:
        if high is None:
            mask = y_true >= low
        else:
            mask = (y_true >= low) & (y_true < high)
        count = int(mask.sum())
        if count == 0:
            continue
        scores = evaluate(y_true[mask], y_pred[mask])
        rows.append({"band": name, "n": count, "mae": scores["mae"], "rmse": scores["rmse"]})
    return rows


def _temporal_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    seasons = sorted(df["season"].astype(str).unique())
    holdout = seasons[-1]
    train_df = df[df["season"].astype(str) != holdout]
    test_df = df[df["season"].astype(str) == holdout]
    if len(train_df) < 80 or len(test_df) < 40:
        raise ValueError(
            f"Temporal split is too small (train={len(train_df)}, {holdout}={len(test_df)})"
        )
    return train_df, test_df, holdout


def _predict(pipeline: Pipeline, frame: pd.DataFrame) -> np.ndarray:
    return np.clip(pipeline.predict(frame[FEATURES]), 0, None)


def _feature_report(
    pipeline: Pipeline, x_eval: pd.DataFrame, y_eval: pd.Series
) -> tuple[list[dict], str]:
    model = pipeline.named_steps["model"]
    linear = model.regressor_ if hasattr(model, "regressor_") else model
    if hasattr(linear, "coef_"):
        names = pipeline.named_steps["preprocess"].get_feature_names_out()
        pairs = sorted(
            zip(names, np.ravel(linear.coef_), strict=True),
            key=lambda item: abs(item[1]),
            reverse=True,
        )
        kind = "coefficient_log_target" if hasattr(model, "regressor_") else "coefficient"
        return [
            {
                "feature": name.replace("num__", "").replace("cat__", ""),
                "coefficient": float(coef),
            }
            for name, coef in pairs
        ], kind

    result = permutation_importance(
        pipeline,
        x_eval,
        y_eval,
        n_repeats=5,
        random_state=42,
        scoring="neg_mean_absolute_error",
    )
    order = np.argsort(result.importances_mean)[::-1]
    columns = list(x_eval.columns)
    return [
        {"feature": columns[i], "coefficient": float(result.importances_mean[i])}
        for i in order
    ], "permutation_mae_increase"


def train(random_state: int = 42) -> dict:
    df = load_dataset()
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_PATH, index=False)

    train_df, test_df, holdout = _temporal_split(df)
    x_train = train_df[FEATURES]
    y_train = train_df[TARGET].astype(float)
    x_test = test_df[FEATURES]
    y_test = test_df[TARGET].astype(float)
    train_seasons = sorted(train_df["season"].astype(str).unique())

    comparison: list[dict] = []
    fitted: dict[str, Pipeline] = {}
    holdout_pred: dict[str, np.ndarray] = {}
    for name in ("LinearRegression", "Ridge", "RidgeLog", "HistGradientBoosting", "RandomForest"):
        pipeline = build_pipeline(_estimator(name, random_state))
        pipeline.fit(x_train, y_train)
        pred = _predict(pipeline, test_df)
        scores = evaluate(y_test.to_numpy(), pred)
        scores["model"] = name
        comparison.append(scores)
        fitted[name] = pipeline
        holdout_pred[name] = pred
        print(f"{name}: MAE €{scores['mae']/1e6:.2f}M  RMSE €{scores['rmse']/1e6:.2f}M  R2 {scores['r2']:.3f}")

    best = min(comparison, key=lambda row: (row["mae"], row["rmse"], row["model"]))
    best_name = str(best["model"])
    bands = value_band_errors(y_test.to_numpy(), holdout_pred[best_name])

    deployed = build_pipeline(_estimator(best_name, random_state))
    deployed.fit(df[FEATURES], df[TARGET].astype(float))

    note = (
        f"Labels are the Transfermarkt quote nearest 30 June at the end of each season. "
        f"The error is a {holdout} holdout after training on {train_seasons[0]}–{train_seasons[-1]}. "
        f"The saved model is refit on every finished season. Estimates are not live fees."
    )
    metrics: dict = {
        "mae": best["mae"],
        "rmse": best["rmse"],
        "r2": best["r2"],
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "n_total": int(len(df)),
        "model": best_name,
        "target": TARGET,
        "dataset": str(RAW_PATH.relative_to(ROOT)),
        "split": "temporal",
        "holdout_season": holdout,
        "train_seasons": train_seasons,
        "dataset_note": note,
        "comparison": comparison,
        "value_bands": bands,
    }

    factors, importance_kind = _feature_report(deployed, x_test, y_test)
    meta = {
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "target": TARGET,
        "importance_kind": importance_kind,
        "coefficients": factors,
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(deployed, MODEL_PATH)
    FEATURE_META_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Chosen {best_name} on the {holdout} holdout")
    for band in bands:
        print(f"  {band['band']}: n={band['n']} MAE €{band['mae']/1e6:.2f}M")
    print(f"Model -> {MODEL_PATH}")
    return metrics


def main() -> None:
    train()


if __name__ == "__main__":
    main()
