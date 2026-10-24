from fastapi import APIRouter, HTTPException, Query

from backend.app.schemas.valuation import PlayerSummary
from backend.app.services import player_store

router = APIRouter(prefix="/players", tags=["players"])


@router.get("", response_model=list[PlayerSummary])
def search_players(
    q: str | None = Query(default=None, description="Search by name or club"),
    limit: int = Query(default=30, ge=1, le=100),
) -> list[PlayerSummary]:
    try:
        return player_store.list_latest_players(query=q, limit=limit)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/{player_id}", response_model=PlayerSummary)
def get_player(player_id: int) -> PlayerSummary:
    try:
        player = player_store.get_player(player_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if player is None:
        raise HTTPException(status_code=404, detail="Player not found")
    return player
