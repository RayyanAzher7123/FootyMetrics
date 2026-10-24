"""Download and normalize real Premier League results from football-data.co.uk.

Source: https://www.football-data.co.uk/ (E0 = Premier League)
"""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw" / "football_data"
OUT_PATH = ROOT / "data" / "raw" / "premier_league_matches.csv"

# Season codes: 1516 = 2015/16 … 2526 = 2025/26
DEFAULT_SEASONS = [
    "1516",
    "1617",
    "1718",
    "1819",
    "1920",
    "2021",
    "2122",
    "2223",
    "2324",
    "2425",
    "2526",
    "2627",
]

TEAM_NAME_MAP = {
    "Man City": "Manchester City",
    "Man United": "Manchester United",
    "Nott'm Forest": "Nottingham Forest",
    "Newcastle": "Newcastle",
    "Wolves": "Wolves",
    "West Brom": "West Brom",
    "Sheffield United": "Sheffield United",
    "Sheffield Utd": "Sheffield United",
}


def season_label(code: str) -> str:
    a, b = int(code[:2]), int(code[2:])
    century_a = 2000 if a < 90 else 1900
    # handle 9900 -> 1999/00
    year_start = century_a + a if a >= 90 else 2000 + a
    if code == "9900":
        return "1999/00"
    year_end = year_start + 1
    return f"{year_start}/{str(year_end)[-2:].zfill(2)}"


def download_season(code: str, force: bool = False) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / f"{code}_E0.csv"
    if path.exists() and not force and path.stat().st_size > 1000:
        return path
    url = f"https://www.football-data.co.uk/mmz4281/{code}/E0.csv"
    print(f"Downloading {url}")
    with urllib.request.urlopen(url, timeout=60) as resp:
        data = resp.read()
    path.write_bytes(data)
    return path


def normalize_team(name: str) -> str:
    name = str(name).strip()
    return TEAM_NAME_MAP.get(name, name)


def parse_date(series: pd.Series) -> pd.Series:
    # football-data uses mixed dayfirst formats across seasons
    return pd.to_datetime(series, dayfirst=True, errors="coerce")


def load_season_file(path: Path, code: str) -> pd.DataFrame:
    # Some files have a trailing blank line / odd encoding
    text = path.read_text(encoding="utf-8", errors="replace")
    df = pd.read_csv(io.StringIO(text))
    required = {"Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} missing columns: {missing}")

    out = pd.DataFrame(
        {
            "date": parse_date(df["Date"]),
            "season": season_label(code),
            "season_code": code,
            "home_team": df["HomeTeam"].map(normalize_team),
            "away_team": df["AwayTeam"].map(normalize_team),
            "home_goals": pd.to_numeric(df["FTHG"], errors="coerce"),
            "away_goals": pd.to_numeric(df["FTAG"], errors="coerce"),
            "result": df["FTR"].astype(str).str.upper().str.strip(),
        }
    )

    # Optional richer match stats when present (from ~2000/01)
    optional_map = {
        "HS": "home_shots",
        "AS": "away_shots",
        "HST": "home_shots_on_target",
        "AST": "away_shots_on_target",
        "HC": "home_corners",
        "AC": "away_corners",
        "HF": "home_fouls",
        "AF": "away_fouls",
        "HY": "home_yellow",
        "AY": "away_yellow",
        "HR": "home_red",
        "AR": "away_red",
    }
    for src, dst in optional_map.items():
        if src in df.columns:
            out[dst] = pd.to_numeric(df[src], errors="coerce")

    out = out.dropna(subset=["date", "home_team", "away_team", "home_goals", "away_goals", "result"])
    out = out[out["result"].isin(["H", "D", "A"])]
    out["home_goals"] = out["home_goals"].astype(int)
    out["away_goals"] = out["away_goals"].astype(int)
    return out


def ingest(seasons: list[str] | None = None, force_download: bool = False) -> pd.DataFrame:
    seasons = seasons or DEFAULT_SEASONS
    frames: list[pd.DataFrame] = []
    for code in seasons:
        try:
            path = download_season(code, force=force_download)
            if path.stat().st_size < 500:
                print(f"Skipping empty/invalid file for {code}")
                continue
            frame = load_season_file(path, code)
            print(f"{code}: {len(frame)} matches")
            frames.append(frame)
        except Exception as exc:  # noqa: BLE001 - keep ingest resilient per season
            print(f"Failed {code}: {exc}")

    if not frames:
        raise RuntimeError("No match data ingested")

    matches = pd.concat(frames, ignore_index=True)
    matches = matches.sort_values("date").reset_index(drop=True)
    matches.insert(0, "match_id", range(1, len(matches) + 1))
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    matches.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(matches)} real matches -> {OUT_PATH}")
    print("Result mix:")
    print(matches["result"].value_counts(normalize=True).round(3))
    print("Teams:", sorted(set(matches["home_team"]) | set(matches["away_team"])))
    return matches


def main() -> None:
    ingest(force_download=False)


if __name__ == "__main__":
    main()
