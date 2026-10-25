"""Data-pipeline contracts: labels stay on the finished season, and match features cannot see the result they are predicting."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from backend.app.config import settings
from ml.training.ingest_fpl_valuations import quote_near_season_end, season_end
from ml.training.train_match_outcome import FEATURE_COLS, build_features
from ml.training.train_valuation import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET, load_dataset


def test_player_rows_have_consistent_rates_and_non_negative_quotes():
    df = pd.read_csv(settings.players_csv_path)
    played = df[df["minutes"] > 0]
    expected_goals = played["goals"] / played["minutes"] * 90
    expected_assists = played["assists"] / played["minutes"] * 90
    assert (played["goals_per90"] - expected_goals).abs().max() < 0.02
    assert (played["assists_per90"] - expected_assists).abs().max() < 0.02
    assert played["age"].between(15, 45).all()
    listed = df[df["market_value_eur"].notna()]
    assert (listed["market_value_eur"] >= 0).all()
    assert set(df["value_source"].dropna().unique()) <= {"transfermarkt", "unlisted"}


def test_quote_window_ignores_a_later_snapshot():
    dates = np.array(["2022-06-15", "2026-06-03"], dtype="datetime64[ns]")
    values = np.array([65_000_000, 110_000_000], dtype="float64")

    early_value, early_date = quote_near_season_end(dates, values, "2021/22")
    assert early_value == 65_000_000
    assert early_date == "2022-06-15"

    late_value, late_date = quote_near_season_end(dates, values, "2025/26")
    assert late_value == 110_000_000
    assert late_date == "2026-06-03"
    assert quote_near_season_end(dates, values, "2018/19") is None
    assert season_end("2025/26") == pd.Timestamp("2026-06-30")


def test_valuation_training_uses_finished_seasons_with_dated_quotes():
    labelled = load_dataset()
    raw = pd.read_csv(settings.players_csv_path)
    seasons = sorted(raw["season"].astype(str).unique())
    in_progress = seasons[-1]

    assert labelled["season"].astype(str).nunique() >= 2
    assert in_progress not in set(labelled["season"].astype(str))
    assert (labelled["value_source"] == "transfermarkt").all()
    assert labelled["market_value_eur"].notna().all()
    assert labelled["valuation_date"].notna().all()
    assert set(NUMERIC_FEATURES + CATEGORICAL_FEATURES).isdisjoint({TARGET, "season"})

    ends = labelled["season"].astype(str).map(season_end)
    dates = pd.to_datetime(labelled["valuation_date"])
    delta_days = (dates - ends).dt.days
    assert int(delta_days.max()) <= 45
    assert int(delta_days.min()) >= -400

    early = raw[(raw["season"].astype(str) == "2021/22") & (raw["value_source"] == "transfermarkt")]
    assert not early.empty
    assert pd.to_datetime(early["valuation_date"]).dt.year.max() <= 2022


def test_saved_valuation_metrics_use_a_later_season():
    metrics = json.loads(settings.metrics_path.read_text(encoding="utf-8"))
    assert metrics["split"] == "temporal"
    assert metrics["holdout_season"] not in metrics["train_seasons"]
    best = min(metrics["comparison"], key=lambda row: (row["mae"], row["rmse"], row["model"]))
    assert metrics["model"] == best["model"]
    assert metrics["n_test"] >= 40
    assert metrics["value_bands"]
    for band in metrics["value_bands"]:
        assert band["n"] > 0
        assert band["mae"] >= 0


def test_match_features_exclude_the_result_being_predicted():
    matches = pd.DataFrame(
        [
            {
                "match_id": 1,
                "date": "2024-08-01",
                "season": "2024/25",
                "home_team": "Alpha",
                "away_team": "Beta",
                "home_goals": 3,
                "away_goals": 0,
                "result": "H",
                "home_shots": 20,
                "home_shots_on_target": 8,
                "away_shots": 4,
                "away_shots_on_target": 1,
            },
            {
                "match_id": 2,
                "date": "2024-08-08",
                "season": "2024/25",
                "home_team": "Alpha",
                "away_team": "Gamma",
                "home_goals": 0,
                "away_goals": 2,
                "result": "A",
                "home_shots": 6,
                "home_shots_on_target": 1,
                "away_shots": 14,
                "away_shots_on_target": 6,
            },
        ]
    )
    features = build_features(matches)

    # The first match has no history. Its own 3-0 win must not become its form.
    first = features.iloc[0]
    assert first["home_form_ppg"] == 1.0
    assert first["home_gf_avg"] == 1.2

    # The second match may use the first result only. Including its own loss
    # would pull Alpha's points per game from 3.0 down to 1.5.
    second = features.iloc[1]
    assert second["home_form_ppg"] == 3.0
    assert second["home_gf_avg"] == 3.0
    assert second["home_ga_avg"] == 0.0

    leaked = {"result", "home_goals", "away_goals", "date"}
    assert leaked.isdisjoint(FEATURE_COLS)
