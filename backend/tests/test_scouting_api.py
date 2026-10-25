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


def test_similar_unknown_player():
    res = client.get("/api/scouting/similar/999999999")
    assert res.status_code == 404


def test_compare_rejects_empty_unknown_and_oversized_sets():
    assert client.get("/api/scouting/compare", params={"ids": ""}).status_code == 400
    assert client.get("/api/scouting/compare", params={"ids": "abc"}).status_code == 400
    assert client.get("/api/scouting/compare", params={"ids": "999999999"}).status_code == 400

    search = client.get("/api/scouting/players", params={"limit": 1}).json()
    player_id = str(search[0]["player_id"])
    too_many = ",".join([player_id] * 6)
    assert client.get("/api/scouting/compare", params={"ids": too_many}).status_code == 400


def test_similar_rejects_invalid_filters():
    search = client.get("/api/scouting/players", params={"q": "Saka"}).json()
    player_id = search[0]["player_id"]
    bad_position = client.get(
        f"/api/scouting/similar/{player_id}",
        params={"position": "ST"},
    )
    assert bad_position.status_code == 422
    bad_limit = client.get(f"/api/scouting/similar/{player_id}", params={"limit": 0})
    assert bad_limit.status_code == 422
