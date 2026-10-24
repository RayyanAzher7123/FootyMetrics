from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.config import settings
from backend.app.routers import matches, players, scouting, valuation

app = FastAPI(
    title=settings.app_name,
    description=(
        "Football intelligence API. "
        "Module 1: transfer value. "
        "Module 2: match outcomes. "
        "Module 3: player scouting & similarity."
    ),
    version="0.3.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(players.router, prefix="/api")
app.include_router(valuation.router, prefix="/api")
app.include_router(matches.router, prefix="/api")
app.include_router(scouting.router, prefix="/api")


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": settings.app_name}
