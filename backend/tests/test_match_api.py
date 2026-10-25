import pandas as pd
from fastapi.testclient import TestClient

from backend.app.config import settings
from backend.app.main import app

client = TestClient(app)


def test_list_teams():
    res = client.get("/api/matches/teams")
    assert res.status_code == 200
    teams = res.json()
    assert "Arsenal" in teams
    assert "Liverpool" in teams


def test_preview_and_predict():
    preview = client.get(
        "/api/matches/preview",
        params={"home_team": "Arsenal", "away_team": "Liverpool"},
    )
    assert preview.status_code == 200
    body = preview.json()
    assert body["home"]["team"] == "Arsenal"
    assert "form_ppg" in body["home"]

    pred = client.post(
        "/api/matches/predict",
        json={"home_team": "Arsenal", "away_team": "Liverpool"},
    )
    assert pred.status_code == 200
    result = pred.json()
    assert result["predicted_result"] in {"H", "D", "A"}
    assert len(result["probabilities"]) == 3
    assert abs(sum(p["probability"] for p in result["probabilities"]) - 100) < 1.5


def test_same_team_rejected():
    res = client.post(
        "/api/matches/predict",
        json={"home_team": "Arsenal", "away_team": "Arsenal"},
    )
    assert res.status_code == 400
    preview = client.get(
        "/api/matches/preview",
        params={"home_team": "Arsenal", "away_team": "Arsenal"},
    )
    assert preview.status_code == 400


def test_unknown_team_and_short_name_rejected():
    unknown = client.post(
        "/api/matches/predict",
        json={"home_team": "Arsenal", "away_team": "Not A Club"},
    )
    assert unknown.status_code == 400
    short = client.post(
        "/api/matches/predict",
        json={"home_team": "A", "away_team": "Arsenal"},
    )
    assert short.status_code == 422


def _result_letters(team: str, games: pd.DataFrame) -> list[str]:
    letters = []
    for row in games.itertuples(index=False):
        is_home = row.home_team == team
        if row.result == "D":
            letters.append("D")
        elif (is_home and row.result == "H") or (not is_home and row.result == "A"):
            letters.append("W")
        else:
            letters.append("L")
    return letters


def test_preview_form_uses_only_the_current_season():
    matches = pd.read_csv(settings.matches_csv_path)
    matches["date"] = pd.to_datetime(matches["date"])
    season = str(matches["season"].max())
    current = matches[matches["season"].astype(str) == season].sort_values("date")
    expected = _result_letters(
        "Arsenal",
        current[(current["home_team"] == "Arsenal") | (current["away_team"] == "Arsenal")].tail(5),
    )

    res = client.get(
        "/api/matches/preview",
        params={"home_team": "Arsenal", "away_team": "Liverpool"},
    )
    assert res.status_code == 200
    home = res.json()["home"]
    assert home["form_season"] == season
    assert home["last_results"] == expected
    assert 1 <= len(home["last_results"]) <= 5
    assert set(home["last_results"]) <= {"W", "D", "L"}
