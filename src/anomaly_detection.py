# ─────────────────────────────────────────────
# Anomaly Detection — Early Warning System
# Detects sudden spikes in case counts
# ─────────────────────────────────────────────
import numpy as np
import pandas as pd
import logging
from sklearn.ensemble import IsolationForest
from src.feature_engineering import prepare_dataset, get_feature_columns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def detect_anomalies(country: str):
    """
    Detect anomalous spikes in case data for a country
    Uses Isolation Forest — unsupervised anomaly detection
    Returns list of anomaly dates and scores
    """
    df         = prepare_dataset()
    country_df = df[df['Country/Region'].str.lower() == country.lower()].copy()

    if country_df.empty:
        raise ValueError(f"Country '{country}' not found")

    features = ['New_Cases', 'growth_rate', 'New_Cases_7Day_Avg']
    X        = country_df[features].fillna(0)

    # Train Isolation Forest
    model = IsolationForest(
        contamination=0.05,  # Expect ~5% anomalies
        random_state=42,
        n_estimators=100
    )

    country_df['anomaly_score'] = model.fit_predict(X)
    country_df['anomaly_flag']  = country_df['anomaly_score'] == -1

    # Extract anomaly records
    anomalies = country_df[country_df['anomaly_flag']][
        ['Date', 'New_Cases', 'growth_rate', 'anomaly_flag']
    ].copy()

    anomalies['Date'] = anomalies['Date'].astype(str)

    logger.info(f"Detected {len(anomalies)} anomalies for {country}")

    return {
        "country": country,
        "total_anomalies": len(anomalies),
        "anomaly_dates": anomalies['Date'].tolist(),
        "anomaly_cases": anomalies['New_Cases'].tolist(),
        "warning": "⚠️ Sudden spike detected!" if len(anomalies) > 0 else "✅ No anomalies"
    }

def get_latest_alert(country: str):
    """
    Check if latest data point is an anomaly
    Used for real-time early warning
    """
    result = detect_anomalies(country)

    df         = prepare_dataset()
    country_df = df[df['Country/Region'].str.lower() == country.lower()]
    latest_date = str(country_df['Date'].iloc[-1])

    is_alert = latest_date in result['anomaly_dates']

    return {
        "country": country,
        "latest_date": latest_date,
        "is_alert": is_alert,
        "alert_message": "🚨 EARLY WARNING: Spike detected in latest data!" if is_alert else "✅ Normal activity"
    }