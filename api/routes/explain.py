from fastapi import APIRouter, HTTPException
from src.explainability import get_feature_importance
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/explain/{country}")
def explain_prediction(country: str):
    try:
        return get_feature_importance(country)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))