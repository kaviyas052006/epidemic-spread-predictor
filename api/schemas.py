# ─────────────────────────────────────────────
# Pydantic Schemas — Input/Output Validation
# ─────────────────────────────────────────────
from pydantic import BaseModel
from typing import List, Optional, Dict

# ── Input Schemas ──────────────────────────
class PredictRequest(BaseModel):
    country: str
    days: Optional[int] = 30

class RiskRequest(BaseModel):
    country: str

class TrainRequest(BaseModel):
    country: Optional[str] = "all"

# ── Output Schemas ─────────────────────────
class PredictResponse(BaseModel):
    country: str
    days: int
    predicted_cases: List[float]
    alert: str

class RiskResponse(BaseModel):
    country: str
    risk_level: str
    confidence: float
    probabilities: Dict[str, float]

class MetricsResponse(BaseModel):
    forecaster_metrics: Dict[str, float]
    classifier_metrics: Dict[str, float]

class TrainResponse(BaseModel):
    status: str
    forecaster_metrics: Dict[str, float]
    classifier_metrics: Dict[str, float]

class AnomalyResponse(BaseModel):
    country: str
    total_anomalies: int
    anomaly_dates: List[str]
    warning: str
    alert_message: str

class ExplainResponse(BaseModel):
    country: str
    feature_importance: Dict[str, float]
    top_feature: str
    explanation: str