# ─────────────────────────────────────────────
# Route — /risk
# Returns outbreak risk level for a country
# ─────────────────────────────────────────────
from fastapi import APIRouter, HTTPException
from api.schemas import RiskRequest, RiskResponse
from src.predict import predict_risk
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/risk", response_model=RiskResponse)
def outbreak_risk(request: RiskRequest):
    try:
        result = predict_risk(request.country)
        return RiskResponse(
            country=request.country,
            risk_level=result["risk_level"],
            confidence=result["confidence"],
            probabilities=result["probabilities"]
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Risk prediction error: {e}")
        raise HTTPException(status_code=500, detail="Risk prediction failed")