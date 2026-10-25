import pandas as pd
from fastapi.testclient import TestClient

from backend.app.config import settings
from backend.app.main import app

client = TestClient(app)


def _season_bounds() -> tuple[str, str]:
    df = pd.read_csv(settings.players_csv_path, usecols=["season"])
    seasons = sorted(str(season) for season in df["season"].dropna().unique())
    return seasons[-1], seasons[-2]


def test_health():
    res = client.get("/api/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"


def test_search_players_uses_current_club_and_previous_season_stats():
    """Search shows the latest squad, with numbers from the finished season before it."""
    current, previous = _season_bounds()
    raw = pd.read_csv(settings.players_csv_path)
    raw["season"] = raw["season"].astype(str)
    stats = raw[(raw["player_name"] == "Bukayo Saka") & (raw["season"] == previous)].iloc[0]
    squad = raw[(raw["player_name"] == "Bukayo Saka") & (raw["season"] == current)].iloc[0]

    res = client.get("/api/players", params={"q": "Saka"})
    assert res.status_code == 200
    data = res.json()
    assert len(data) >= 1
    player = data[0]
    assert player["player_name"] == "Bukayo Saka"
    assert player["season"] == previous
    assert player["season"] != current
    assert player["club"] == squad["club"]
    assert player["goals"] == int(stats["goals"])
    assert player["assists"] == int(stats["assists"])
    assert player["minutes"] == int(stats["minutes"])
    assert player["market_value_source"] == "transfermarkt"
    assert player["market_value_eur"] == int(stats["market_value_eur"])


def test_search_hides_players_without_a_previous_season():
    current, previous = _season_bounds()
    raw = pd.read_csv(settings.players_csv_path, usecols=["player_id", "player_name", "season"])
    raw["season"] = raw["season"].astype(str)
    current_ids = set(raw.loc[raw["season"] == current, "player_id"])
    previous_ids = set(raw.loc[raw["season"] == previous, "player_id"])
    only_current = current_ids - previous_ids
    assert only_current, "expected some current-squad players without a previous-season row"
    player_id = next(iter(only_current))
    name = raw.loc[
        (raw["player_id"] == player_id) & (raw["season"] == current), "player_name"
    ].iloc[0]

    res = client.get("/api/players", params={"q": name, "limit": 50})
    assert res.status_code == 200
    assert player_id not in {row["player_id"] for row in res.json()}


def test_blank_search_returns_previous_season_rows():
    _, previous = _season_bounds()
    res = client.get("/api/players", params={"limit": 20})
    assert res.status_code == 200
    rows = res.json()
    assert rows
    assert {row["season"] for row in rows} == {previous}


def test_unknown_player_is_not_found():
    res = client.get("/api/players/999999999")
    assert res.status_code == 404


def test_search_limit_is_bounded():
    assert client.get("/api/players", params={"limit": 0}).status_code == 422
    assert client.get("/api/players", params={"limit": 101}).status_code == 422


def test_valuation_rejects_invalid_inputs():
    base = {
        "age": 24,
        "position": "MID",
        "appearances": 30,
        "goals": 10,
        "assists": 8,
        "minutes": 2700,
    }
    cases = [
        {**base, "age": 14},
        {**base, "age": 46},
        {**base, "position": "ST"},
        {**base, "goals": -1},
        {**base, "assists": 41},
        {**base, "minutes": 5001},
        {**base, "appearances": 51},
    ]
    for payload in cases:
        res = client.post("/api/valuation/predict", json=payload)
        assert res.status_code == 422, payload


def test_valuation_accepts_zero_minutes_and_stays_non_negative():
    res = client.post(
        "/api/valuation/predict",
        json={
            "age": 18,
            "position": "GK",
            "appearances": 0,
            "goals": 0,
            "assists": 0,
            "minutes": 0,
        },
    )
    assert res.status_code == 200
    body = res.json()
    assert body["estimated_value_eur"] >= 0
    assert body["inputs"]["minutes"] == 0
    assert body["inputs"]["position"] == "GK"


def test_predict_valuation():
    payload = {
        "age": 24,
        "position": "MID",
        "appearances": 30,
        "goals": 10,
        "assists": 8,
        "minutes": 2700,
        "player_name": "Test Player",
        "club": "Arsenal",
    }
    res = client.post("/api/valuation/predict", json=payload)
    assert res.status_code == 200
    body = res.json()
    assert body["estimated_value_eur"] > 0
    assert body["estimated_value_display"].startswith("€")
    assert "mae" in body["metrics"]
