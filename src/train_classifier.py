# ──────────────────────────────────────────────────────
# Train Risk Classifier — LightGBM, Zero Leakage
# ──────────────────────────────────────────────────────
import os
import json
import joblib
import logging
import lightgbm as lgb
from sklearn.metrics import classification_report, f1_score
from src.feature_engineering import prepare_dataset, get_feature_columns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_PATH   = "models/risk_classifier/lgb_classifier.pkl"
METRICS_PATH = "models/risk_classifier/metrics.json"

def train_classifier():
    df       = prepare_dataset()
    features = get_feature_columns(for_classifier=True)
    target   = 'risk_label'

    # ✅ Chronological split
    split_date = df['Date'].quantile(0.8)
    train_df   = df[df['Date'] <= split_date]
    test_df    = df[df['Date'] >  split_date]

    X_train, y_train = train_df[features], train_df[target]
    X_test,  y_test  = test_df[features],  test_df[target]

    logger.info(f"Train size: {len(X_train)} | Test size: {len(X_test)}")
    logger.info(f"Label distribution (test):\n{y_test.value_counts()}")

    model = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=8,
        num_leaves=63,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=100)
        ]
    )

    preds  = model.predict(X_test)
    f1     = f1_score(y_test, preds, average='weighted')
    report = classification_report(
        y_test, preds,
        target_names=["Low", "Medium", "High"]
    )

    logger.info(f"\n📊 Classification Report:\n{report}")
    logger.info(f"✅ Real F1 Score: {f1:.4f}")

    os.makedirs("models/risk_classifier", exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    metrics = {"F1_Score": round(f1, 4)}
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f)

    logger.info(f"Model saved → {MODEL_PATH}")
    return model, metrics

if __name__ == "__main__":
    train_classifier()