import zipfile
from pathlib import Path

from scripts.make_release import should_pack, write_release


def test_release_skips_environments_and_bulk_downloads():
    assert should_pack(Path("frontend/src/App.tsx"))
    assert should_pack(Path("data/raw/premier_league_valuations.csv"))
    assert should_pack(Path("ml/models/valuation_linear.joblib"))
    assert not should_pack(Path("frontend/node_modules/react/index.js"))
    assert not should_pack(Path(".venv/Scripts/python.exe"))
    assert not should_pack(Path("data/raw/transfermarkt/player_valuations.csv.gz"))
    assert not should_pack(Path("data/raw/fpl/2025-26_players_raw.csv"))
    assert not should_pack(Path("data/raw/football_data/2025-26_E0.csv"))
    assert not should_pack(Path("backend/app/__pycache__/main.cpython-311.pyc"))


def test_release_zip_contains_only_packed_files(tmp_path: Path):
    root = tmp_path / "project"
    (root / "frontend" / "node_modules" / "react").mkdir(parents=True)
    (root / "frontend" / "src").mkdir(parents=True)
    (root / ".venv" / "Lib").mkdir(parents=True)
    (root / "data" / "raw" / "transfermarkt").mkdir(parents=True)
    (root / "data" / "raw").joinpath("premier_league_valuations.csv").write_text("player\n", encoding="utf-8")
    (root / "frontend" / "node_modules" / "react" / "index.js").write_text("x", encoding="utf-8")
    (root / ".venv" / "Lib" / "site.py").write_text("x", encoding="utf-8")
    (root / "data" / "raw" / "transfermarkt" / "player_valuations.csv.gz").write_bytes(b"gz")
    (root / "frontend" / "src" / "App.tsx").write_text("export {}", encoding="utf-8")

    destination = tmp_path / "out" / "FootyMetrics.zip"
    write_release(root, destination)

    with zipfile.ZipFile(destination) as archive:
        names = set(archive.namelist())
    assert names == {
        "data/raw/premier_league_valuations.csv",
        "frontend/src/App.tsx",
    }
