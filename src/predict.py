# ─────────────────────────────────────────────
# Prediction Logic — LightGBM Version
# ─────────────────────────────────────────────
import joblib
import numpy as np
import logging
from src.feature_engineering import prepare_dataset, get_feature_columns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FORECASTER_PATH = "models/forecaster/lgb_model.pkl"
CLASSIFIER_PATH = "models/risk_classifier/lgb_classifier.pkl"
RISK_LABELS     = {0: "Low", 1: "Medium", 2: "High"}

def load_models():
    return joblib.load(FORECASTER_PATH), joblib.load(CLASSIFIER_PATH)

def get_country_data(country: str):
    df         = prepare_dataset()
    country_df = df[df['Country/Region'].str.lower() == country.lower()]
    if country_df.empty:
        raise ValueError(f"Country '{country}' not found in dataset")
    return country_df

def predict_cases(country: str, days: int = 30):
    forecaster, _ = load_models()
    features      = get_feature_columns(for_classifier=False)

    country_df = get_country_data(country)
    history    = country_df['New_Cases'].values.tolist()
    last_row   = country_df[features].iloc[-1].copy()

    predictions = []

    for _ in range(days):
        row_input = last_row.values.reshape(1, -1)
        pred      = float(np.clip(forecaster.predict(row_input)[0], 0, None))
        predictions.append(round(pred, 2))

        # ✅ Update all lag and rolling features properly
        history.append(pred)

        last_row['lag_1']  = history[-1]
        last_row['lag_3']  = history[-3]  if len(history) >= 3  else pred
        last_row['lag_7']  = history[-7]  if len(history) >= 7  else pred
        last_row['lag_14'] = history[-14] if len(history) >= 14 else pred

        last_row['rolling_7']  = np.mean(history[-7:])
        last_row['rolling_14'] = np.mean(history[-14:])
        last_row['rolling_21'] = np.mean(history[-21:])

        last_row['growth_rate'] = (
            (pred - history[-2]) / history[-2]
            if len(history) >= 2 and history[-2] > 0 else 0
        )

        last_row['pct_change_7'] = (
            (pred - history[-7]) / history[-7]
            if len(history) >= 7 and history[-7] > 0 else 0
        )

        last_row['trend'] = (
            1 if last_row['rolling_7'] > last_row['rolling_14']
            else (-1 if last_row['rolling_7'] < last_row['rolling_14'] else 0)
        )

        last_row['acceleration'] = (
            last_row['growth_rate'] - (
                (history[-2] - history[-3]) / history[-3]
                if len(history) >= 3 and history[-3] > 0 else 0
            )
        )

    logger.info(f"Generated {days}-day forecast for {country}")
    return predictions

def predict_risk(country: str):
    _, classifier = load_models()
    features      = get_feature_columns(for_classifier=True)

    country_df    = get_country_data(country)
    last_row      = country_df[features].iloc[-1].values.reshape(1, -1)

    risk_index    = classifier.predict(last_row)[0]
    probabilities = classifier.predict_proba(last_row)[0]
    confidence    = round(float(np.max(probabilities)), 4)
    risk_label    = RISK_LABELS[risk_index]

    return {
        "risk_level":    risk_label,
        "confidence":    confidence,
        "probabilities": {
            "Low":    round(float(probabilities[0]), 4),
            "Medium": round(float(probabilities[1]), 4),
            "High":   round(float(probabilities[2]), 4)
        }
    }