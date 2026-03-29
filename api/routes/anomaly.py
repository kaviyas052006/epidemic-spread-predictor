from fastapi import APIRouter, HTTPException
from src.anomaly_detection import detect_anomalies, get_latest_alert
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/anomaly/{country}")
def anomaly_detection(country: str):
    try:
        result = detect_anomalies(country)
        alert  = get_latest_alert(country)
        result["alert_message"] = alert["alert_message"]
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))