# ─────────────────────────────────────────────
# Explainability — SHAP Feature Importance
# Explains WHY a prediction was made
# ─────────────────────────────────────────────
import shap
import joblib
import numpy as np
import pandas as pd
import logging
from src.feature_engineering import prepare_dataset, get_feature_columns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FORECASTER_PATH = "models/forecaster/xgb_model.pkl"

def get_feature_importance(country: str):
    """
    Returns SHAP-based feature importance for a country
    Explains which features drove the prediction
    """
    model    = joblib.load(FORECASTER_PATH)
    features = get_feature_columns()

    df         = prepare_dataset()
    country_df = df[df['Country/Region'].str.lower() == country.lower()]

    if country_df.empty:
        raise ValueError(f"Country '{country}' not found")

    # Use last 30 rows for SHAP explanation
    X = country_df[features].tail(30)

    # SHAP explainer for XGBoost
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # Mean absolute SHAP value per feature
    importance = np.abs(shap_values).mean(axis=0)
    feature_importance = dict(zip(features, importance.tolist()))

    # Sort by importance
    sorted_importance = dict(
        sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    )

    logger.info(f"SHAP explanation generated for {country}")

    return {
        "country": country,
        "feature_importance": sorted_importance,
        "top_feature": list(sorted_importance.keys())[0],
        "explanation": f"The most influential factor for {country} is '{list(sorted_importance.keys())[0]}'"
    }