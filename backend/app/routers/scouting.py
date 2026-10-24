from fastapi import APIRouter, HTTPException, Query

from backend.app.schemas.scouting import (
    ClustersResponse,
    CompareResponse,
    ScoutingModelInfo,
    ScoutPlayer,
    SimilarPlayersResponse,
)
from backend.app.services import scouting_service

router = APIRouter(prefix="/scouting", tags=["scouting"])


@router.get("/players", response_model=list[ScoutPlayer])
def search_scouting_players(
    q: str | None = Query(default=None),
    limit: int = Query(default=40, ge=1, le=100),
) -> list[ScoutPlayer]:
    try:
        return scouting_service.search_players(query=q, limit=limit)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/similar/{player_id}", response_model=SimilarPlayersResponse)
def similar_players(
    player_id: int,
    limit: int = Query(default=8, ge=1, le=30),
    position: str | None = Query(default=None, pattern="^(GK|DEF|MID|FWD)$"),
    max_age: int | None = Query(default=None, ge=15, le=45),
    max_value: int | None = Query(default=None, ge=0),
) -> SimilarPlayersResponse:
    try:
        return scouting_service.find_similar(
            player_id=player_id,
            limit=limit,
            position=position,
            max_age=max_age,
            max_value=max_value,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/compare", response_model=CompareResponse)
def compare(ids: str = Query(..., description="Comma-separated player ids")) -> CompareResponse:
    try:
        player_ids = [int(x.strip()) for x in ids.split(",") if x.strip()]
        return scouting_service.compare_players(player_ids)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/clusters", response_model=ClustersResponse)
def clusters() -> ClustersResponse:
    try:
        return scouting_service.list_clusters()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/model", response_model=ScoutingModelInfo)
def get_model_info() -> ScoutingModelInfo:
    try:
        return scouting_service.model_info()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
