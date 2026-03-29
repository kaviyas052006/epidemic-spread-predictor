# ─────────────────────────────────────────────
# predict.py — Cluster-aware prediction
# ─────────────────────────────────────────────
from fastapi import APIRouter, HTTPException
from api.schemas import PredictRequest, PredictResponse
from src.predict import predict_cases
from src.anomaly_detection import get_latest_alert
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/predict", response_model=PredictResponse)
def forecast_cases(request: PredictRequest):
    try:
        predictions = predict_cases(request.country, request.days)
        alert_info  = get_latest_alert(request.country)
        return PredictResponse(
            country=request.country,
            days=request.days,
            predicted_cases=predictions,
            alert=alert_info["alert_message"]
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")
import joblib
import json
import numpy as np
import logging
from src.feature_engineering import prepare_dataset, get_feature_columns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CLUSTER_PATH    = "models/forecaster/country_clusters.json"
CLASSIFIER_PATH = "models/risk_classifier/lgb_classifier.pkl"
RISK_LABELS     = {0: "Low", 1: "Medium", 2: "High"}

def load_cluster_models():
    with open(CLUSTER_PATH) as f:
        cluster_map = json.load(f)
    models = {}
    for cid in set(cluster_map.values()):
        lgb_path = f"models/forecaster/lgb_cluster_{cid}.pkl"
        xgb_path = f"models/forecaster/xgb_cluster_{cid}.pkl"
        models[cid] = {
            'lgb': joblib.load(lgb_path),
            'xgb': joblib.load(xgb_path)
        }
    return cluster_map, models

def get_country_data(country: str):
    df         = prepare_dataset()
    country_df = df[df['Country/Region'].str.lower() == country.lower()]
    if country_df.empty:
        raise ValueError(f"Country '{country}' not found in dataset")
    return country_df

def predict_cases(country: str, days: int = 30):
    cluster_map, models = load_cluster_models()
    features            = get_feature_columns(for_classifier=False)
    country_df          = get_country_data(country)

    # Get cluster for this country
    cluster_id = cluster_map.get(country, 0)
    lgb_model  = models[cluster_id]['lgb']
    xgb_model  = models[cluster_id]['xgb']

    history  = country_df['New_Cases'].values.tolist()
    last_row = country_df[features].iloc[-1].copy()
    predictions = []

    for _ in range(days):
        row_input = last_row.values.reshape(1, -1)
        lgb_pred  = float(np.clip(np.expm1(lgb_model.predict(row_input)[0]), 0, None))
        xgb_pred  = float(np.clip(np.expm1(xgb_model.predict(row_input)[0]), 0, None))
        pred      = (lgb_pred + xgb_pred) / 2

        # Spike clipping
        rolling_avg = np.mean(history[-7:]) if len(history) >= 7 else pred
        pred        = min(pred, rolling_avg * 3 + 1000)
        pred        = max(0, round(pred, 2))
        predictions.append(pred)

        history.append(pred)
        last_row['lag_1']        = history[-1]
        last_row['lag_3']        = history[-3]  if len(history) >= 3  else pred
        last_row['lag_7']        = history[-7]  if len(history) >= 7  else pred
        last_row['lag_14']       = history[-14] if len(history) >= 14 else pred
        last_row['rolling_7']    = np.mean(history[-7:])
        last_row['rolling_14']   = np.mean(history[-14:])
        last_row['rolling_21']   = np.mean(history[-21:])
        last_row['growth_rate']  = (pred - history[-2]) / history[-2] if len(history) >= 2 and history[-2] > 0 else 0
        last_row['pct_change_7'] = (pred - history[-7]) / history[-7] if len(history) >= 7 and history[-7] > 0 else 0
        last_row['trend']        = 1 if last_row['rolling_7'] > last_row['rolling_14'] else -1

    return predictions

def predict_risk(country: str):
    classifier = joblib.load(CLASSIFIER_PATH)
    features   = get_feature_columns(for_classifier=True)
    country_df = get_country_data(country)
    last_row   = country_df[features].iloc[-1].values.reshape(1, -1)

    risk_index    = classifier.predict(last_row)[0]
    probabilities = classifier.predict_proba(last_row)[0]

    return {
        "risk_level"   : RISK_LABELS[risk_index],
        "confidence"   : round(float(np.max(probabilities)), 4),
        "probabilities": {
            "Low"   : round(float(probabilities[0]), 4),
            "Medium": round(float(probabilities[1]), 4),
            "High"  : round(float(probabilities[2]), 4)
        }
    }