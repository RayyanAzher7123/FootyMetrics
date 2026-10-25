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
    assert body["reference"]["qualified"] is True
    assert body["reference"]["minutes"] >= 900
    assert len(body["similar"]) >= 1
    assert body["similar"][0]["similarity"] is not None
    assert {row["position"] for row in body["similar"]} == {body["reference"]["position"]}
    assert min(row["minutes"] for row in body["similar"]) >= 900

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


def test_similar_can_target_another_position():
    search = client.get("/api/scouting/players", params={"q": "Saka"}).json()
    player_id = search[0]["player_id"]
    res = client.get(
        f"/api/scouting/similar/{player_id}",
        params={"limit": 5, "position": "FWD"},
    )
    assert res.status_code == 200
    similar = res.json()["similar"]
    assert similar
    assert {row["position"] for row in similar} == {"FWD"}
    assert min(row["minutes"] for row in similar) >= 900


def test_short_sample_is_not_recommended():
    import pandas as pd

    from backend.app.config import settings

    frame = pd.read_csv(settings.scouting_players_path)
    short = frame[frame["qualified"] == False]  # noqa: E712
    assert not short.empty
    player_id = int(short.iloc[0]["player_id"])
    res = client.get(f"/api/scouting/similar/{player_id}")
    assert res.status_code == 200
    body = res.json()
    assert body["similar"] == []
    assert "900" in body["dataset_note"]


def test_similarity_features_are_rates():
    res = client.get("/api/scouting/model")
    assert res.status_code == 200
    body = res.json()
    features = set(body["features"])
    assert "goals_per90" in features
    assert features.isdisjoint({"age", "minutes", "appearances", "goals", "assists"})
    assert body["metrics"]["min_minutes"] == 900
    assert body["metrics"]["stats_season"] == "2025/26"


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
