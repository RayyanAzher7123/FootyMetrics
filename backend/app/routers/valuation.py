from fastapi import APIRouter, HTTPException

from backend.app.schemas.valuation import ModelInfoResponse, ValuationRequest, ValuationResponse
from backend.app.services import valuation_model

router = APIRouter(prefix="/valuation", tags=["valuation"])


@router.post("/predict", response_model=ValuationResponse)
def predict(payload: ValuationRequest) -> ValuationResponse:
    try:
        return valuation_model.predict_value(payload)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/model", response_model=ModelInfoResponse)
def get_model_info() -> ModelInfoResponse:
    try:
        return valuation_model.model_info()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
