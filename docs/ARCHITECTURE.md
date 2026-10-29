# Architecture

FootyMetrics is one page with three modules. The browser talks to a FastAPI app, and that app reads CSV tables and joblib files. Nothing in the request path opens a database.

```text
Browser
  Value · Matches · Scouting
        |
        |  /api
        v
FastAPI
  players · valuation · matches · scouting
        |
        +-- data/raw/premier_league_valuations.csv
        +-- data/raw/premier_league_matches.csv
        +-- data/processed/scouting_players.csv
        +-- ml/models/*.joblib
```

In development, Vite on port 5173 proxies `/api` to the API on port 8000. In Docker, nginx in the web container does the same proxy and Compose waits until `/api/health` can see the CSV and model files. Health reports `"storage": "files"`.

## Modules

**Transfer value.** Search shows the current Premier League squad. The stats and the quote on the card come from the previous finished season. The saved model is a scikit-learn pipeline: one-hot position plus age, appearances, goals, assists, minutes, and per-90 rates, estimated with `HistGradientBoostingRegressor`. The quote beside the estimate is the Transfermarkt value nearest 30 June at the end of that season.

**Match odds.** The page builds current-season form, shots, and head-to-head rates for the two clubs, then a logistic regression predicts home win, draw, or away win. Those features are computed from matches that have already been played. A fixture does not see its own result.

**Scouting.** The current squad stays searchable. Neighbours are other players in the same position with at least 900 minutes in 2025/26. Distance is cosine similarity on per-90 rates only. A separate k-means model draws the map; it is not the neighbour search.

## Training and serving

Training scripts live in `ml/training/`. They write the CSV tables and the joblib bundles. The API loads those files on first use and keeps them in memory. After a retrain, the API process has to be restarted before it serves the new files.

| Piece | Role |
| --- | --- |
| `ml/training/ingest_fpl_valuations.py` | Join FPL season stats to a dated Transfermarkt quote |
| `ml/training/train_valuation.py` | Compare models on a later season, then refit the winner |
| `ml/training/ingest_football_data.py` | Normalise football-data.co.uk Premier League results |
| `ml/training/train_match_outcome.py` | Fit the logistic regression on earlier matches |
| `ml/training/train_scouting.py` | Build the per-position neighbour index and the cluster map |
| `backend/app/` | Read the saved files and serve `/api` |
| `frontend/` | One scrolling page for the three modules |

## What is reserved

`docker-compose.yml` includes PostgreSQL under the `postgres` profile. `docker compose up` does not start it, and the API has no database client. That service is there for a later persistence change.

Player photos are Premier League 110×140 cutouts, requested in the browser from `resources.premierleague.com`. They are not stored in this repo.
