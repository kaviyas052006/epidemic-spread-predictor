# ─────────────────────────────────────────────
# Route — /train
# Triggers model training pipeline
# ─────────────────────────────────────────────
from fastapi import APIRouter, HTTPException
from api.schemas import TrainRequest, TrainResponse
from src.train_forecaster import train_forecaster
from src.train_classifier import train_classifier
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/train", response_model=TrainResponse)
def train_models(request: TrainRequest):
    try:
        logger.info("Training forecaster...")
        _, f_metrics = train_forecaster()

        logger.info("Training classifier...")
        _, c_metrics = train_classifier()

        return TrainResponse(
            status="✅ Models trained successfully",
            forecaster_metrics=f_metrics,
            classifier_metrics=c_metrics
        )
    except Exception as e:
        logger.error(f"Training error: {e}")
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")