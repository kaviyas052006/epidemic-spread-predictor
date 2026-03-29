# ─────────────────────────────────────────────
# Route — /metrics
# Returns saved model performance metrics
# ─────────────────────────────────────────────
import json
from fastapi import APIRouter, HTTPException
from api.schemas import MetricsResponse

router = APIRouter()

FORECASTER_METRICS  = "models/forecaster/metrics.json"
CLASSIFIER_METRICS  = "models/risk_classifier/metrics.json"

@router.get("/metrics", response_model=MetricsResponse)
def get_metrics():
    try:
        with open(FORECASTER_METRICS)  as f: f_metrics = json.load(f)
        with open(CLASSIFIER_METRICS) as f: c_metrics = json.load(f)
        return MetricsResponse(
            forecaster_metrics=f_metrics,
            classifier_metrics=c_metrics
        )
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail="Models not trained yet. Call /train first."
        )