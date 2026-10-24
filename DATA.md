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
| `data/raw/transfermarkt/` | Full-squad Transfermarkt values through 12 June 2026 ([transfermarkt-datasets](https://github.com/dcaribou/transfermarkt-datasets) `players.csv.gz`). Players with no name match are left unlisted. |
| `data/raw/premier_league_valuations.csv` | Joined training/search table |

```bash
python -m ml.training.ingest_fpl_valuations
python -m ml.training.train_valuation
```

- **Performance features** come from real FPL season totals (minutes, goals, assists, etc.).
- **Market value labels** prefer Transfermarkt values matched by player name.
- Unmatched players stay searchable using an FPL-price proxy flag (`value_source`).
- Search and scouting show **only the latest ingested season** (currently 2026/27). Players who left the Premier League stay in the historical file for training but do not appear in the app. Season stats for 2026/27 are season-to-date, not a full campaign.

Estimates are still model outputs — not live broker quotes.

## Module 3 — Player Scouting

Built from the same player table as Module 1 (`premier_league_valuations.csv`).

```bash
python -m ml.training.train_scouting
```

Outputs:
- `data/processed/scouting_players.csv` — latest profile per player + cluster + PCA coords  
- `ml/models/scouting_bundle.joblib` — scaler, K-Means, nearest neighbors, PCA  

Similarity uses cosine distance on scaled season features. Filters (position, max age, max value) apply after neighbor search.

## Legacy bootstrap generators

`generate_sample_data.py` / `generate_match_data.py` remain available for offline demos without network access. Prefer the ingest scripts above for real-data mode.
