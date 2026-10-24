from __future__ import annotations

import json
from functools import lru_cache

import joblib
import pandas as pd

from backend.app.config import settings
from backend.app.schemas.matches import (
    MatchModelInfoResponse,
    MatchPredictRequest,
    MatchPredictResponse,
    OutcomeProbability,
)
from backend.app.services import match_store

DATASET_NOTE = (
    "Trained on real Premier League results from football-data.co.uk (E0), "
    "using pre-match form, shots, and head-to-head features only — no future leakage."
)

LABEL_NAMES = {"H": "Home win", "D": "Draw", "A": "Away win"}


@lru_cache(maxsize=1)
def _load_model():
    path = settings.match_model_path
    if not path.exists():
        raise FileNotFoundError(
            f"Match model not found at {path}. Run: python -m ml.training.train_match_outcome"
        )
    return joblib.load(path)


@lru_cache(maxsize=1)
def _load_metrics() -> dict:
    path = settings.match_metrics_path
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def _load_meta() -> dict:
    path = settings.match_feature_meta_path
    if not path.exists():
        return {"feature_importance": [], "features": []}
    return json.loads(path.read_text(encoding="utf-8"))


def reload_model() -> None:
    _load_model.cache_clear()
    _load_metrics.cache_clear()
    _load_meta.cache_clear()


def predict_match(payload: MatchPredictRequest) -> MatchPredictResponse:
    preview = match_store.preview_fixture(payload.home_team, payload.away_team)
    features = match_store.feature_row(payload.home_team, payload.away_team)
    model = _load_model()
    frame = pd.DataFrame([features])
    proba = model.predict_proba(frame)[0]
    classes = list(model.named_steps["model"].classes_)
    pairs = sorted(zip(classes, proba, strict=True), key=lambda x: x[1], reverse=True)
    predicted, confidence = pairs[0]

    probabilities = [
        OutcomeProbability(
            label=label,
            name=LABEL_NAMES.get(label, label),
            probability=round(float(p) * 100, 1),
        )
        for label, p in zip(classes, proba, strict=True)
    ]
    # Keep display order Home / Draw / Away
    order = {"H": 0, "D": 1, "A": 2}
    probabilities.sort(key=lambda x: order.get(x.label, 9))

    meta = _load_meta()
    top = meta.get("feature_importance", [])[:6]
    metrics = _load_metrics()

    return MatchPredictResponse(
        home_team=payload.home_team,
        away_team=payload.away_team,
        probabilities=probabilities,
        predicted_result=predicted,
        predicted_label=LABEL_NAMES.get(predicted, predicted),
        confidence=round(float(confidence) * 100, 1),
        top_factors=top,
        model_name=str(metrics.get("model", "LogisticRegression")),
        dataset_note=DATASET_NOTE,
        metrics=metrics,
        preview=preview,
    )


def model_info() -> MatchModelInfoResponse:
    metrics = _load_metrics()
    meta = _load_meta()
    return MatchModelInfoResponse(
        model_name=str(metrics.get("model", "LogisticRegression")),
        metrics=metrics,
        top_factors=meta.get("feature_importance", [])[:8],
        features=list(meta.get("features", [])),
        dataset_note=DATASET_NOTE,
    )
