from fastapi.testclient import TestClient

from backend.app.main import app

client = TestClient(app)


def test_scouting_search():
    res = client.get("/api/scouting/players", params={"q": "Saka"})
    assert res.status_code == 200
    data = res.json()
    assert len(data) >= 1
    assert "Saka" in data[0]["player_name"]
    assert "cluster_id" in data[0]


def test_similar_and_clusters():
    search = client.get("/api/scouting/players", params={"q": "Saka"}).json()
    player_id = search[0]["player_id"]

    similar = client.get(f"/api/scouting/similar/{player_id}", params={"limit": 5})
    assert similar.status_code == 200
    body = similar.json()
    assert body["reference"]["player_id"] == player_id
    assert len(body["similar"]) >= 1
    assert body["similar"][0]["similarity"] is not None

    clusters = client.get("/api/scouting/clusters")
    assert clusters.status_code == 200
    cbody = clusters.json()
    assert len(cbody["clusters"]) >= 2
    assert len(cbody["map_points"]) >= 10


def test_compare():
    search = client.get("/api/scouting/players", params={"limit": 3}).json()
    ids = ",".join(str(p["player_id"]) for p in search[:2])
    res = client.get("/api/scouting/compare", params={"ids": ids})
    assert res.status_code == 200
    assert len(res.json()["players"]) == 2
