from fastapi import APIRouter, HTTPException, Query

from backend.app.schemas.matches import (
    MatchModelInfoResponse,
    MatchPredictRequest,
    MatchPredictResponse,
    MatchPreviewResponse,
)
from backend.app.services import match_model, match_store

router = APIRouter(prefix="/matches", tags=["matches"])


@router.get("/teams", response_model=list[str])
def get_teams() -> list[str]:
    try:
        return match_store.list_teams()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/preview", response_model=MatchPreviewResponse)
def preview(
    home_team: str = Query(...),
    away_team: str = Query(...),
) -> MatchPreviewResponse:
    try:
        return match_store.preview_fixture(home_team, away_team)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/predict", response_model=MatchPredictResponse)
def predict(payload: MatchPredictRequest) -> MatchPredictResponse:
    try:
        return match_model.predict_match(payload)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/model", response_model=MatchModelInfoResponse)
def get_model_info() -> MatchModelInfoResponse:
    try:
        return match_model.model_info()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
