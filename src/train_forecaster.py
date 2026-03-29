# ─────────────────────────────────────────────
# Train Forecaster — Stable Restored Version
# LightGBM + XGBoost ensemble only
# No over-engineering
# ─────────────────────────────────────────────
import os
import json
import joblib
import logging
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, mean_absolute_error
from src.feature_engineering import prepare_dataset, get_feature_columns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CLUSTER_PATH = "models/forecaster/country_clusters.json"
METRICS_PATH = "models/forecaster/metrics.json"

def compute_mape(actual, predicted, threshold=50):
    a    = np.array(actual, dtype=float)
    p    = np.array(predicted, dtype=float)
    mask = a > threshold
    if mask.sum() == 0:
        return 999.0
    eps  = 1e-8
    return float(np.mean(np.abs((a[mask] - p[mask]) / (a[mask] + eps))) * 100)

def compute_worst_case(actual, predicted, threshold=50, top_n=10):
    a    = np.array(actual, dtype=float)
    p    = np.array(predicted, dtype=float)
    mask = a > threshold
    if mask.sum() == 0:
        return 999.0
    eps    = 1e-8
    errors = np.abs((a[mask] - p[mask]) / (a[mask] + eps)) * 100
    top_n  = min(top_n, len(errors))
    return float(np.mean(np.sort(errors)[-top_n:]))

def compute_p95(actual, predicted, threshold=50):
    a    = np.array(actual, dtype=float)
    p    = np.array(predicted, dtype=float)
    mask = a > threshold
    if mask.sum() == 0:
        return 999.0
    eps    = 1e-8
    errors = np.abs((a[mask] - p[mask]) / (a[mask] + eps)) * 100
    return float(np.percentile(errors, 95))

def smooth_spikes(df, column='New_Cases', threshold=5.0):
    """Smooth only extreme spikes — conservative threshold"""
    df          = df.copy()
    df[column]  = df[column].astype(float)
    rolling_med = (
        df.groupby('Country/Region')[column]
        .transform(lambda x: x.rolling(7, min_periods=1).median())
    )
    spike_mask = df[column] > (threshold * rolling_med + 1)
    df.loc[spike_mask, column] = rolling_med[spike_mask].values
    logger.info(f"   Smoothed {spike_mask.sum()} spike rows")
    return df

def cluster_countries(df, n_clusters=3):
    country_stats = (
        df.groupby('Country/Region')['New_Cases']
        .agg(['mean', 'std', 'max'])
        .fillna(0)
    )
    stats_log = np.log1p(country_stats)
    km        = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    country_stats['cluster'] = km.fit_predict(stats_log)
    cluster_map = country_stats['cluster'].to_dict()

    for c in range(n_clusters):
        avg = country_stats.loc[country_stats['cluster'] == c, 'mean'].mean()
        cnt = sum(1 for v in cluster_map.values() if v == c)
        logger.info(f"   Cluster {c}: {cnt} countries | avg: {avg:.0f}")

    return cluster_map

def train_single_cluster(train_df, test_df, features, target, cluster_id):
    X_train    = train_df[features]
    X_test     = test_df[features]
    y_train    = np.log1p(train_df[target].astype(float))
    y_test     = test_df[target].astype(float)
    y_test_log = np.log1p(y_test)

    if len(X_train) < 100:
        logger.warning(f"Cluster {cluster_id}: insufficient data")
        return None, None, 999.0

    # ✅ LightGBM only
    lgb_model = lgb.LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=8,
        num_leaves=63,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    lgb_model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test_log)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=999)
        ]
    )

    # ✅ XGBoost only
    xgb_model = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
        early_stopping_rounds=50
    )
    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test_log)],
        verbose=False
    )

    # ✅ Predictions
    lgb_preds = np.clip(np.expm1(lgb_model.predict(X_test)), 0, None)
    xgb_preds = np.clip(np.expm1(xgb_model.predict(X_test)), 0, None)

    lgb_mape  = compute_mape(y_test.values, lgb_preds)
    xgb_mape  = compute_mape(y_test.values, xgb_preds)

    # ✅ Weighted ensemble — better model gets more weight
    total     = lgb_mape + xgb_mape + 1e-6
    lgb_w     = 1 - (lgb_mape / total)
    xgb_w     = 1 - (xgb_mape / total)
    ens_preds = (lgb_w * lgb_preds + xgb_w * xgb_preds) / (lgb_w + xgb_w)

    # ✅ Conservative clipping — cap at 3x rolling mean
    cap       = test_df['rolling_7'].values * 3 + 1000
    ens_preds = np.minimum(ens_preds, cap)
    ens_preds = np.clip(ens_preds, 0, None)

    mape  = compute_mape(y_test.values, ens_preds)
    worst = compute_worst_case(y_test.values, ens_preds)
    p95   = compute_p95(y_test.values, ens_preds)

    logger.info(
        f"   Cluster {cluster_id}: "
        f"LGB={lgb_mape:.1f}% | XGB={xgb_mape:.1f}% | "
        f"Ens={mape:.1f}% | Worst={worst:.1f}% | P95={p95:.1f}%"
    )

    return lgb_model, xgb_model, mape

def walk_forward_validate(df, features, target, n_splits=3):
    logger.info(f"\n📊 WALK-FORWARD VALIDATION ({n_splits} splits):")
    dates = df['Date'].sort_values().unique()
    chunk = len(dates) // (n_splits + 1)
    mapes = []

    for i in range(1, n_splits + 1):
        cutoff   = dates[chunk * i]
        test_end = dates[min(chunk * (i + 1), len(dates) - 1)]
        tr       = df[df['Date'] <= cutoff]
        te       = df[(df['Date'] > cutoff) & (df['Date'] <= test_end)]

        if len(te) == 0:
            continue

        m = lgb.LGBMRegressor(
            n_estimators=500, learning_rate=0.05,
            max_depth=8, num_leaves=63,
            random_state=42, n_jobs=-1, verbose=-1
        )
        m.fit(tr[features], np.log1p(tr[target].astype(float)))
        preds = np.clip(np.expm1(m.predict(te[features])), 0, None)
        cap   = te['rolling_7'].values * 3 + 1000
        preds = np.minimum(preds, cap)

        mape  = compute_mape(te[target].values, preds)
        worst = compute_worst_case(te[target].values, preds)
        mapes.append(mape)
        logger.info(f"   Split {i}: cutoff={str(cutoff)[:10]} | MAPE={mape:.2f}% | Worst={worst:.1f}%")

    avg = np.mean(mapes) if mapes else 999.0
    logger.info(f"   Avg Walk-Forward MAPE: {avg:.2f}%")
    return avg

def train_forecaster():
    df       = prepare_dataset()
    features = get_feature_columns(for_classifier=False)
    target   = 'New_Cases'

    # ✅ Active countries
    active_countries = (
        df.groupby('Country/Region')['New_Cases']
        .mean().loc[lambda x: x > 50].index
    )
    df = df[df['Country/Region'].isin(active_countries)].copy()
    logger.info(f"Active countries: {len(active_countries)}")

    # ✅ Spike smoothing
    logger.info("Smoothing spikes...")
    df = smooth_spikes(df, column='New_Cases', threshold=5.0)

    # ✅ Cluster
    logger.info("\nClustering countries:")
    cluster_map   = cluster_countries(df, n_clusters=3)
    df['cluster'] = df['Country/Region'].map(cluster_map)

    # ✅ Chronological split
    split_date = df['Date'].quantile(0.8)
    train_df   = df[df['Date'] <= split_date]
    test_df    = df[df['Date'] >  split_date]

    # ✅ Train per cluster
    all_actuals    = []
    all_preds      = []
    cluster_models = {}

    logger.info("\nTraining per cluster:")
    for cluster_id in sorted(df['cluster'].unique()):
        tr = train_df[train_df['cluster'] == cluster_id]
        te = test_df[test_df['cluster']  == cluster_id]

        lgb_m, xgb_m, c_mape = train_single_cluster(
            tr, te, features, target, cluster_id
        )

        if lgb_m is not None:
            cluster_models[cluster_id] = {'lgb': lgb_m, 'xgb': xgb_m}

            lgb_p = np.clip(np.expm1(lgb_m.predict(te[features])), 0, None)
            xgb_p = np.clip(np.expm1(xgb_m.predict(te[features])), 0, None)
            preds = (lgb_p + xgb_p) / 2
            cap   = te['rolling_7'].values * 3 + 1000
            preds = np.minimum(preds, cap)
            preds = np.clip(preds, 0, None)

            all_actuals.extend(te[target].values)
            all_preds.extend(preds)

    # ✅ Metrics
    all_actuals = np.array(all_actuals, dtype=float)
    all_preds   = np.array(all_preds,   dtype=float)
    rmse        = np.sqrt(mean_squared_error(all_actuals, all_preds))
    mae         = mean_absolute_error(all_actuals, all_preds)
    mape        = compute_mape(all_actuals, all_preds)
    worst_case  = compute_worst_case(all_actuals, all_preds)
    p95         = compute_p95(all_actuals, all_preds)

    logger.info(f"\n{'='*55}")
    logger.info(f"✅ FINAL METRICS:")
    logger.info(f"   RMSE       : {rmse:.2f}")
    logger.info(f"   MAE        : {mae:.2f}")
    logger.info(f"   MAPE       : {mape:.2f}%")
    logger.info(f"   Worst-Case : {worst_case:.2f}%")
    logger.info(f"   P95 Error  : {p95:.2f}%")
    logger.info(f"{'='*55}")

    # ✅ Baselines
    y_test_all = test_df[target].astype(float)
    naive_mape = compute_mape(y_test_all.values, y_test_all.shift(1).fillna(0).values)
    mavg_mape  = compute_mape(y_test_all.values, test_df['rolling_7'].values)

    logger.info(f"\n📊 BASELINE COMPARISON:")
    logger.info(f"   Naive MAPE      : {naive_mape:.2f}%")
    logger.info(f"   Moving Avg MAPE : {mavg_mape:.2f}%")
    logger.info(f"   Our Model MAPE  : {mape:.2f}%")
    logger.info("✅ Beats BOTH" if mape < min(naive_mape, mavg_mape) else "⚠️ Check baselines")

    # ✅ Walk-forward
    wf_mape = walk_forward_validate(df, features, target)

    # ✅ Verdict
    logger.info(f"\n{'='*55}")
    if mape < 10:
        logger.info("🏆 EXCELLENT — MAPE below 10%")
    elif mape < 15:
        logger.info("🏆 WINNING LEVEL — MAPE below 15%")
    elif mape < 25:
        logger.info("✅ GOOD — MAPE below 25%")
    else:
        logger.info("❌ Needs improvement")
    logger.info(f"{'='*55}")

    # ✅ Save
    os.makedirs("models/forecaster", exist_ok=True)
    for cid, mdict in cluster_models.items():
        joblib.dump(mdict['lgb'], f"models/forecaster/lgb_cluster_{cid}.pkl")
        joblib.dump(mdict['xgb'], f"models/forecaster/xgb_cluster_{cid}.pkl")

    with open(CLUSTER_PATH, "w") as f:
        json.dump({str(k): int(v) for k, v in cluster_map.items()}, f)

    metrics = {
        "RMSE"        : round(rmse, 2),
        "MAE"         : round(mae, 2),
        "MAPE"        : round(mape, 2),
        "Worst_Case"  : round(worst_case, 2),
        "P95_Error"   : round(p95, 2),
        "Naive_MAPE"  : round(naive_mape, 2),
        "MovAvg_MAPE" : round(mavg_mape, 2),
        "WalkFwd_MAPE": round(wf_mape, 2),
        "Beats_Naive" : bool(mape < naive_mape)
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info("Models saved ✅")
    return cluster_models, metrics

if __name__ == "__main__":
    train_forecaster()