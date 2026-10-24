"""Generate bootstrap Premier League match results for Module 2.

Replace with football-data.org (or similar) historical CSVs for real training.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT_PATH = ROOT / "data" / "raw" / "premier_league_matches.csv"

TEAMS = [
    "Arsenal",
    "Aston Villa",
    "Bournemouth",
    "Brentford",
    "Brighton",
    "Chelsea",
    "Crystal Palace",
    "Everton",
    "Fulham",
    "Liverpool",
    "Manchester City",
    "Manchester United",
    "Newcastle",
    "Nottingham Forest",
    "Tottenham",
    "West Ham",
    "Wolves",
    "Leicester",
    "Leeds",
    "Southampton",
]

# Relative strength priors for a more football-like outcome distribution
STRENGTH = {
    "Manchester City": 1.35,
    "Arsenal": 1.28,
    "Liverpool": 1.27,
    "Chelsea": 1.12,
    "Tottenham": 1.1,
    "Manchester United": 1.08,
    "Newcastle": 1.08,
    "Aston Villa": 1.05,
    "Brighton": 1.02,
    "West Ham": 0.98,
    "Fulham": 0.96,
    "Brentford": 0.95,
    "Crystal Palace": 0.94,
    "Bournemouth": 0.92,
    "Wolves": 0.92,
    "Everton": 0.9,
    "Nottingham Forest": 0.9,
    "Leicester": 0.88,
    "Leeds": 0.87,
    "Southampton": 0.85,
}


def _sample_goals(rng: np.random.Generator, attack: float, defense: float, home: bool) -> int:
    lam = 1.15 * attack / defense
    if home:
        lam *= 1.18
    return int(rng.poisson(max(0.2, lam)))


def generate(n_matches: int = 1800, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    seasons = ["2021/22", "2022/23", "2023/24", "2024/25"]
    rows: list[dict] = []
    match_id = 1

    # Roughly round-robin style sampling across seasons
    for season_idx, season in enumerate(seasons):
        year = 2021 + season_idx
        per_season = n_matches // len(seasons)
        for i in range(per_season):
            home, away = rng.choice(TEAMS, size=2, replace=False)
            hs = STRENGTH[home] * float(rng.lognormal(0, 0.05))
            as_ = STRENGTH[away] * float(rng.lognormal(0, 0.05))
            hg = _sample_goals(rng, hs, as_, home=True)
            ag = _sample_goals(rng, as_, hs, home=False)
            if hg > ag:
                result = "H"
            elif hg < ag:
                result = "A"
            else:
                result = "D"

            # Spread fixtures across the season
            month = 8 + (i * 9) // max(per_season, 1)
            if month > 12:
                month -= 12
                day_year = year + 1
            else:
                day_year = year
            day = int(rng.integers(1, 28))
            date = f"{day_year:04d}-{month:02d}-{day:02d}"

            rows.append(
                {
                    "match_id": match_id,
                    "date": date,
                    "season": season,
                    "home_team": home,
                    "away_team": away,
                    "home_goals": hg,
                    "away_goals": ag,
                    "result": result,
                }
            )
            match_id += 1

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df["match_id"] = np.arange(1, len(df) + 1)
    return df


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = generate()
    df.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(df)} matches -> {OUT_PATH}")
    print(df["result"].value_counts(normalize=True).round(3))
    print(df.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
