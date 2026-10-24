"""Build Module 1 training data from real FPL season stats + Transfermarkt values.

Player performance: Fantasy Premier League historical season totals
(https://github.com/vaastav/Fantasy-Premier-League)

Market-value labels: Transfermarkt valuations from
https://github.com/eordo/transfermarkt-data (matched by player name when possible).
Rows without a TM match fall back to an FPL-price-scaled estimate and are flagged.
"""

from __future__ import annotations

import re
import unicodedata
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
FPL_DIR = ROOT / "data" / "raw" / "fpl"
TM_DIR = ROOT / "data" / "raw" / "transfermarkt"
OUT_PATH = ROOT / "data" / "raw" / "premier_league_valuations.csv"

SEASONS = ["2021-22", "2022-23", "2023-24", "2024-25", "2025-26", "2026-27"]
POSITION_MAP = {"GK": "GK", "DEF": "DEF", "MID": "MID", "FWD": "FWD", 1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}


_FOLD = str.maketrans(
    {
        "ø": "o",
        "ð": "d",
        "đ": "d",
        "ł": "l",
        "þ": "t",
        "Ø": "o",
        "Ð": "d",
        "Đ": "d",
        "Ł": "l",
    }
)
_PARTICLES = {"de", "da", "dos", "do", "van", "von", "di", "del", "bin", "al"}


def _normalize_name(name: str) -> str:
    name = (
        name.replace("æ", "ae")
        .replace("Æ", "ae")
        .replace("œ", "oe")
        .replace("ß", "ss")
        .translate(_FOLD)
    )
    name = unicodedata.normalize("NFKD", name)
    name = name.lower()
    name = name.encode("ascii", "ignore").decode("ascii")
    name = re.sub(r"[^a-z\s]", " ", name)
    name = re.sub(r"\s+", " ", name).strip()
    for token in (" jr", " junior"):
        if name.endswith(token):
            name = name[: -len(token)].strip()
    return name


def _significant_parts(name: str) -> list[str]:
    return [part for part in _normalize_name(name).split() if part not in _PARTICLES]


def _download(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 500:
        return
    print(f"Downloading {url}")
    with urllib.request.urlopen(url, timeout=60) as resp:
        path.write_bytes(resp.read())


def fetch_fpl_season(season: str) -> None:
    base = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"
    for name in ("players_raw.csv", "cleaned_players.csv", "teams.csv"):
        _download(f"{base}/{season}/{name}", FPL_DIR / f"{season}_{name}")


SQUAD_VALUES_URL = (
    "https://pub-e682421888d945d684bcae8890b0ec20.r2.dev/data/players.csv.gz"
)


def _age_at_season_end(birth, season: str) -> int | None:
    dob = pd.to_datetime(birth, errors="coerce")
    if pd.isna(dob):
        return None
    end_year = int(str(season)[:4]) + 1
    end = pd.Timestamp(year=end_year, month=6, day=30)
    years = end.year - dob.year - ((end.month, end.day) < (dob.month, dob.day))
    if years < 15 or years > 45:
        return None
    return int(years)


def load_squad_quotes() -> dict[str, dict[str, dict]]:
    """Full-squad Transfermarkt values, snapshot through 12 June 2026.

    The older premier_league/{year}.csv files only list players who moved,
    so most of the league had no quote. This table has a value for each
    player, including those who stayed at their club.
    """
    path = TM_DIR / "tm_players.csv.gz"
    _download(SQUAD_VALUES_URL, path)
    players = pd.read_csv(path)
    players = players.dropna(subset=["name", "market_value_in_eur"]).copy()
    last_season = pd.to_numeric(players["last_season"], errors="coerce").fillna(0)
    in_prem = players["current_club_domestic_competition_id"].eq("GB1")
    players["pref"] = in_prem.astype(int) * 2 + (last_season >= 2025).astype(int)
    players = players.sort_values(
        ["pref", "last_season", "market_value_in_eur"], ascending=False
    )

    exact: dict[str, dict] = {}
    single: dict[str, dict] = {}
    last_token_rows: dict[str, list[dict]] = {}

    for _, row in players.iterrows():
        info = {
            "value": float(row["market_value_in_eur"]),
            "tm_name": str(row["name"]),
            "pref": int(row["pref"]),
        }
        full = _normalize_name(str(row["name"]))
        parts = full.split()
        if full and full not in exact:
            exact[full] = info
        if len(parts) == 1 and info["pref"] >= 2 and parts[0] not in single:
            single[parts[0]] = info
        if parts and info["pref"] >= 3:
            last_token_rows.setdefault(parts[-1], []).append(info)

    last_token = {
        token: rows[0] for token, rows in last_token_rows.items() if len(rows) == 1
    }
    print(f"Squad quotes: {len(players)} valued players, {len(exact)} names")
    return {"exact": exact, "single": single, "last_token": last_token}


def match_quote(name: str, web_name: str, quotes: dict[str, dict[str, dict]]) -> dict | None:
    full = _normalize_name(name)
    if full in quotes["exact"]:
        return quotes["exact"][full]

    parts = _significant_parts(name)
    # Maternal surnames sit at the end ("David Raya Martín"), so try the
    # first two names before the first-and-last pair.
    if parts and parts[0] == "benjamin":
        hit = quotes["exact"].get(f"ben {parts[-1]}")
        if hit and hit["pref"] >= 2:
            return hit
    if len(parts) >= 3:
        given = f"{parts[0]} {parts[1]}"
        hit = quotes["exact"].get(given)
        if hit and hit["pref"] >= 2:
            return hit
    if len(parts) >= 2:
        pair = f"{parts[0]} {parts[-1]}"
        hit = quotes["exact"].get(pair)
        if hit and hit["pref"] >= 2:
            return hit

    web = _normalize_name(web_name or "")
    web_parts = web.split()
    if len(web_parts) == 2 and len(web_parts[0]) == 1:
        hit = quotes["last_token"].get(web_parts[1])
        if hit:
            return hit
    if web:
        for bucket in ("exact", "single", "last_token"):
            hit = quotes[bucket].get(web)
            if hit and hit["pref"] >= 2:
                return hit
    if len(parts) == 2:
        hit = quotes["single"].get(parts[0])
        if hit:
            return hit
    if parts:
        hit = quotes["last_token"].get(parts[-1])
        if hit:
            return hit
    return None


def fetch_transfermarkt(years: list[int] | None = None) -> pd.DataFrame:
    years = years or [2021, 2022, 2023, 2024, 2025, 2026]
    frames: list[pd.DataFrame] = []
    for year in years:
        path = TM_DIR / f"premier_league_{year}.csv"
        url = (
            "https://raw.githubusercontent.com/eordo/transfermarkt-data/master/"
            f"premier_league/{year}.csv"
        )
        try:
            _download(url, path)
            df = pd.read_csv(path)
            if "market_value" in df.columns and "player_name" in df.columns:
                frames.append(df)
                print(f"TM {year}: {len(df)} rows")
        except Exception as exc:  # noqa: BLE001
            print(f"TM {year} skipped: {exc}")
    if not frames:
        return pd.DataFrame(columns=["player_name", "market_value", "age", "pos"])
    tm = pd.concat(frames, ignore_index=True)
    tm["market_value"] = pd.to_numeric(tm["market_value"], errors="coerce")
    tm = tm.dropna(subset=["player_name", "market_value"])
    tm["name_key"] = tm["player_name"].map(_normalize_name)
    # latest / max observed value per normalized name
    agg = (
        tm.sort_values("market_value")
        .groupby("name_key", as_index=False)
        .agg(
            market_value_eur=("market_value", "max"),
            tm_age=("age", "last"),
            tm_pos=("pos", "last"),
            tm_name=("player_name", "last"),
        )
    )
    return agg


def _season_label(season: str) -> str:
    # 2023-24 -> 2023/24
    a, b = season.split("-")
    return f"{a}/{b[-2:]}"


def build_dataset() -> pd.DataFrame:
    for season in SEASONS:
        fetch_fpl_season(season)
    quotes = load_squad_quotes()

    rows: list[dict] = []
    player_id = 1
    id_by_key: dict[str, int] = {}

    for season in SEASONS:
        raw = pd.read_csv(FPL_DIR / f"{season}_players_raw.csv")
        teams = pd.read_csv(FPL_DIR / f"{season}_teams.csv")
        team_map = dict(zip(teams["id"], teams["name"], strict=False))

        # Prefer cleaned for position labels when available
        cleaned_path = FPL_DIR / f"{season}_cleaned_players.csv"
        cleaned = pd.read_csv(cleaned_path) if cleaned_path.exists() else None

        for _, r in raw.iterrows():
            first = str(r.get("first_name", "") or "")
            second = str(r.get("second_name", "") or "")
            web = str(r.get("web_name", "") or "")
            name = f"{first} {second}".strip() or web
            if not name:
                continue

            minutes = int(pd.to_numeric(r.get("minutes"), errors="coerce") or 0)
            # Historical rows need a real sample of minutes. The latest season
            # is kept in full so current clubs stay visible mid-campaign.
            if season != SEASONS[-1] and minutes < 90:
                continue

            goals = int(pd.to_numeric(r.get("goals_scored"), errors="coerce") or 0)
            assists = int(pd.to_numeric(r.get("assists"), errors="coerce") or 0)
            starts_raw = pd.to_numeric(r.get("starts"), errors="coerce")
            starts = int(starts_raw) if pd.notna(starts_raw) else 0
            appearances = starts if starts > 0 else max(1, minutes // 70)

            element_type = r.get("element_type")
            if cleaned is not None:
                match = cleaned[
                    (cleaned["first_name"] == first) & (cleaned["second_name"] == second)
                ]
                if not match.empty:
                    element_type = match.iloc[0]["element_type"]
            position = POSITION_MAP.get(element_type, "MID")
            if position not in {"GK", "DEF", "MID", "FWD"}:
                position = "MID"

            club = team_map.get(int(r["team"]), "Unknown") if pd.notna(r.get("team")) else "Unknown"
            key = _normalize_name(name)
            quote = match_quote(name, web, quotes)
            age = _age_at_season_end(r.get("birth_date"), season) or 25
            if quote:
                market_value = int(round(quote["value"], -3))
                value_source = "transfermarkt"
            else:
                # No squad quote. Leave it blank rather than pretending an FPL price is a fee.
                market_value = None
                value_source = "unlisted"

            if key not in id_by_key:
                id_by_key[key] = player_id
                player_id += 1

            rows.append(
                {
                    "player_id": id_by_key[key],
                    "player_name": name,
                    "season": _season_label(season),
                    "age": age,
                    "position": position,
                    "club": club,
                    "appearances": appearances,
                    "goals": goals,
                    "assists": assists,
                    "minutes": minutes,
                    "goals_per90": round(goals / minutes * 90, 3) if minutes else 0.0,
                    "assists_per90": round(assists / minutes * 90, 3) if minutes else 0.0,
                    "market_value_eur": market_value,
                    "valuation_date": f"{season[:4]}-06-01",
                    "value_source": value_source,
                    "photo_code": int(r["code"]) if pd.notna(r.get("code")) else None,
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No FPL player rows built")

    # Prefer Transfermarkt-labelled rows for training quality, but keep all for search
    print(
        "Value sources:",
        df.drop_duplicates(["player_id", "season"])["value_source"].value_counts().to_dict(),
    )
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(df)} rows ({df.player_id.nunique()} players) -> {OUT_PATH}")
    print(df.sort_values("market_value_eur", ascending=False).head(8)[
        ["player_name", "club", "season", "goals", "assists", "market_value_eur", "value_source"]
    ].to_string(index=False))
    return df


def main() -> None:
    build_dataset()


if __name__ == "__main__":
    main()
