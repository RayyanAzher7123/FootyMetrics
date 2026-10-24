from __future__ import annotations

import json
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

from backend.app.config import settings
from backend.app.schemas.valuation import (
    FeatureContribution,
    ModelInfoResponse,
    ValuationRequest,
    ValuationResponse,
)

DATASET_NOTE = (
    "Trained on 2025/26 Premier League season stats. "
    "Market quotes are Transfermarkt squad values as of 12 June 2026, "
    "matched by player name. Estimates are model outputs, not live fees."
)


@lru_cache(maxsize=1)
def _load_model():
    path = settings.valuation_model_path
    if not path.exists():
        raise FileNotFoundError(
            f"Model not found at {path}. Run: python -m ml.training.train_valuation"
        )
    return joblib.load(path)


@lru_cache(maxsize=1)
def _load_metrics() -> dict:
    path = settings.metrics_path
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def _load_feature_meta() -> dict:
    path = settings.feature_meta_path
    if not path.exists():
        return {"coefficients": [], "numeric_features": [], "categorical_features": []}
    return json.loads(path.read_text(encoding="utf-8"))


def reload_model() -> None:
    _load_model.cache_clear()
    _load_metrics.cache_clear()
    _load_feature_meta.cache_clear()


def format_eur(value: int) -> str:
    if value >= 1_000_000:
        return f"€{value / 1_000_000:.1f}M"
    if value >= 1_000:
        return f"€{value / 1_000:.0f}K"
    return f"€{value}"


def predict_value(payload: ValuationRequest) -> ValuationResponse:
    model = _load_model()
    minutes = max(payload.minutes, 1)
    row = pd.DataFrame(
        [
            {
                "age": payload.age,
                "appearances": payload.appearances,
                "goals": payload.goals,
                "assists": payload.assists,
                "minutes": payload.minutes,
                "goals_per90": payload.goals / minutes * 90,
                "assists_per90": payload.assists / minutes * 90,
                "position": payload.position,
            }
        ]
    )
    raw = float(model.predict(row)[0])
    estimate = int(max(0, round(raw, -4)))

    meta = _load_feature_meta()
    top = [
        FeatureContribution(feature=item["feature"], coefficient=item["coefficient"])
        for item in meta.get("coefficients", [])[:6]
    ]
    metrics = _load_metrics()
    # Keep JSON-serializable primitives
    clean_metrics = {
        k: (float(v) if isinstance(v, (np.floating,)) else v) for k, v in metrics.items()
    }

    return ValuationResponse(
        estimated_value_eur=estimate,
        estimated_value_display=format_eur(estimate),
        inputs=payload,
        top_factors=top,
        model_name=str(clean_metrics.get("model", "LinearRegression")),
        dataset_note=DATASET_NOTE,
        metrics=clean_metrics,
    )


def model_info() -> ModelInfoResponse:
    meta = _load_feature_meta()
    metrics = _load_metrics()
    top = [
        FeatureContribution(feature=item["feature"], coefficient=item["coefficient"])
        for item in meta.get("coefficients", [])[:8]
    ]
    return ModelInfoResponse(
        model_name=str(metrics.get("model", "LinearRegression")),
        metrics=metrics,
        top_factors=top,
        numeric_features=list(meta.get("numeric_features", [])),
        categorical_features=list(meta.get("categorical_features", [])),
        dataset_note=DATASET_NOTE,
    )
