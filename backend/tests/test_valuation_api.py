from fastapi.testclient import TestClient

from backend.app.main import app

client = TestClient(app)


def test_health():
    res = client.get("/api/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"


def test_search_players():
    res = client.get("/api/players", params={"q": "Saka"})
    assert res.status_code == 200
    data = res.json()
    assert len(data) >= 1
    assert data[0]["player_name"] == "Bukayo Saka"
    assert data[0]["season"] == "2025/26"
    assert data[0]["club"] == "Arsenal"
    assert data[0]["goals"] == 7


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
