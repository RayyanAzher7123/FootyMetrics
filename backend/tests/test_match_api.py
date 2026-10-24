from fastapi.testclient import TestClient

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
