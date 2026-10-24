# FootyMetrics

Football intelligence platform for the Premier League: transfer valuation, match outcome prediction, and player scouting — one app, three ML modules.

## Current status

All three modules are live:

1. **Valuation** — real FPL stats + Transfermarkt labels + Linear Regression  
2. **Match odds** — real football-data.co.uk results + Logistic Regression  
3. **Scouting** — similarity search, compare, K-Means clusters + PCA map  


## Stack

| Layer | Tech |
|-------|------|
| Frontend | React, TypeScript, Tailwind, Vite, Recharts |
| Backend | Python, FastAPI |
| ML | scikit-learn, pandas, NumPy, joblib |
| Database | PostgreSQL (Docker; optional for MVP file-backed mode) |

## Quick start (local, no Docker)

### 1. Python env + train model

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
# source .venv/bin/activate

pip install -r backend/requirements.txt

# Real data (recommended)
python -m ml.training.ingest_football_data
python -m ml.training.train_match_outcome
python -m ml.training.ingest_fpl_valuations
python -m ml.training.train_valuation
python -m ml.training.train_scouting
```

### 2. Backend

```bash
uvicorn backend.app.main:app --reload --port 8000
```

API docs: http://localhost:8000/docs

### 3. Frontend

```bash
cd frontend
npm install
npm run dev
```

App: http://localhost:5173

### Docker (optional)

```bash
python -m ml.training.generate_sample_data
python -m ml.training.train_valuation
docker compose up --build
```

## Project layout

```
backend/     FastAPI app
frontend/    React UI
ml/          Training scripts + saved models
data/        Raw / processed datasets
```

## Data note

See [DATA.md](DATA.md). Default mode uses real PL match results (football-data.co.uk) and real player season stats (FPL) with Transfermarkt value labels where names match.

## Roadmap

- [x] Phase 0 scaffold  
- [x] Module 1 — Transfer value  
- [x] Module 2 — Match outcome  
- [x] Module 3 — Scouting / similarity  
- [ ] Polish, more tests, deployment  
