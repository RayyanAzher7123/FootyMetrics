# FootyMetrics Data Guide

## Module 2 — Match Outcome Predictor (real results)

| Path | Description |
|------|-------------|
| `data/raw/football_data/*_E0.csv` | Season files from [football-data.co.uk](https://www.football-data.co.uk/) |
| `data/raw/premier_league_matches.csv` | Normalized PL matches used for training + API |

```bash
python -m ml.training.ingest_football_data
python -m ml.training.train_match_outcome
```

Features are engineered chronologically (form, shots, H2H) so a fixture never sees its own result.

## Module 1 — Transfer Value Predictor (real stats + market labels)

| Path | Description |
|------|-------------|
| `data/raw/fpl/` | Fantasy Premier League season stats ([vaastav/FPL](https://github.com/vaastav/Fantasy-Premier-League)) |
| `data/raw/transfermarkt/` | Transfermarkt players and dated quotes through 12 June 2026 ([transfermarkt-datasets](https://github.com/dcaribou/transfermarkt-datasets) `players.csv.gz` and `player_valuations.csv.gz`). Each season is labelled with the quote nearest 30 June at the end of that season. Players with no name match, and seasons with no quote in that window, are left unlisted. |
| `data/raw/premier_league_valuations.csv` | Joined training/search table |

```bash
python -m ml.training.ingest_fpl_valuations
python -m ml.training.train_valuation
```

- **Performance features** come from real FPL season totals (minutes, goals, assists, etc.).
- **Market value labels** are dated Transfermarkt quotes matched by player name. A 2021/22 season uses the quote from around June 2022, not the June 2026 snapshot.
- Unmatched players stay searchable with `value_source=unlisted` and no market value. An FPL price is not used as a fee.
- The valuation model is compared on a later finished season (trained on earlier seasons only). The saved model is then refit on every finished season. The in-progress season is not used for training.
- Search shows the current squad with the previous season’s stats and that season’s end quote. Scouting shows the latest ingested season (currently 2026/27). Players who left the Premier League stay in the historical file but do not appear in search. Season stats for 2026/27 are season-to-date, not a full campaign.

Estimates are still model outputs — not live broker quotes.

## Module 3 — Player Scouting

Built from the same player table as Module 1 (`premier_league_valuations.csv`).

```bash
python -m ml.training.train_scouting
```

Outputs:
- `data/processed/scouting_players.csv` — latest profile per player + cluster + PCA coords  
- `ml/models/scouting_bundle.joblib` — scaler, K-Means, nearest neighbors, PCA  

Similarity is cosine distance on per-90 rates (goals, assists, xG, xA, goals conceded, saves, clean sheets, defensive contribution), compared inside one position. Age, appearances, and raw totals are not distance features. Only current-squad players with at least 900 minutes in the previous finished season enter the neighbour index. Shorter samples stay searchable and return no recommendation.

## Legacy bootstrap generators

`generate_sample_data.py` / `generate_match_data.py` remain available for offline demos without network access. Prefer the ingest scripts above for real-data mode.
