# ─────────────────────────────────────────────
# Feature Engineering — Final Version
# Zero Leakage + Duplicate Fix + Strong Features
# ─────────────────────────────────────────────
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_PATH = "data/final_processed_epidemic_data.csv"

def load_data():
    logger.info("Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    df['Date'] = pd.to_datetime(df['Date'])

    # ✅ Fix duplicate rows
    before = len(df)
    df = df.drop_duplicates(subset=['Country/Region', 'Date'], keep='last')
    after = len(df)
    logger.info(f"Removed {before - after} duplicate rows")

    df = df.sort_values(['Country/Region', 'Date']).reset_index(drop=True)
    logger.info(f"Loaded {len(df)} rows, {df['Country/Region'].nunique()} countries")
    return df

def add_features(df):
    logger.info("Engineering features...")

    grp = df.groupby('Country/Region')['New_Cases']

    # ✅ Lag features
    df['lag_1']  = grp.shift(1)
    df['lag_3']  = grp.shift(3)
    df['lag_7']  = grp.shift(7)
    df['lag_14'] = grp.shift(14)

    # ✅ Rolling averages (shift 1 to avoid leakage)
    df['rolling_7']  = grp.transform(lambda x: x.shift(1).rolling(7,  min_periods=1).mean())
    df['rolling_14'] = grp.transform(lambda x: x.shift(1).rolling(14, min_periods=1).mean())
    df['rolling_21'] = grp.transform(lambda x: x.shift(1).rolling(21, min_periods=1).mean())

    # ✅ Growth rate
    df['growth_rate'] = grp.pct_change().replace([np.inf, -np.inf], 0)

    # ✅ Percent change over 7 days
    df['pct_change_7'] = grp.pct_change(periods=7).replace([np.inf, -np.inf], 0)

    # ✅ Trend — is 7day avg rising or falling vs 14day?
    df['trend'] = (df['rolling_7'] - df['rolling_14']).apply(
        lambda x: 1 if x > 0 else (-1 if x < 0 else 0)
    )

    # ✅ Acceleration — change in growth rate
    df['acceleration'] = df.groupby('Country/Region')['growth_rate'].diff().fillna(0)

    # ✅ Time features
    df['day_of_week'] = df['Date'].dt.dayofweek
    df['month']       = df['Date'].dt.month
    df['day_of_year'] = df['Date'].dt.dayofyear
    df['week']        = df['Date'].dt.isocalendar().week.astype(int)

    df = df.fillna(0)
    logger.info("Feature engineering complete.")
    return df

def get_feature_columns(for_classifier=False):
    if for_classifier:
        # ✅ NO rolling_7, rolling_14, rolling_21 — used to create risk_label
        # ✅ NO growth_rate, pct_change_7 — correlated with label
        # ✅ NO New_Cases_7Day_Avg — directly encodes label
        return [
            'lag_1', 'lag_3', 'lag_7', 'lag_14',
            'retail_and_recreation_percent_change_from_baseline',
            'transit_stations_percent_change_from_baseline',
            'workplaces_percent_change_from_baseline',
            'residential_percent_change_from_baseline',
            'retail_and_recreation_percent_change_from_baseline_lag_7',
            'retail_and_recreation_percent_change_from_baseline_lag_14',
            'workplaces_percent_change_from_baseline_lag_7',
            'workplaces_percent_change_from_baseline_lag_14',
            'people_fully_vaccinated_per_hundred',
            'total_boosters_per_hundred',
            'day_of_week', 'month', 'day_of_year', 'week',
            'trend', 'acceleration'
        ]

    # Forecaster uses all features
    return [
        'lag_1', 'lag_3', 'lag_7', 'lag_14',
        'rolling_7', 'rolling_14', 'rolling_21',
        'growth_rate', 'pct_change_7',
        'trend', 'acceleration',
        'New_Cases_lag_7', 'New_Cases_lag_14',
        'New_Cases_7Day_Avg',
        'retail_and_recreation_percent_change_from_baseline',
        'transit_stations_percent_change_from_baseline',
        'workplaces_percent_change_from_baseline',
        'residential_percent_change_from_baseline',
        'retail_and_recreation_percent_change_from_baseline_lag_7',
        'retail_and_recreation_percent_change_from_baseline_lag_14',
        'workplaces_percent_change_from_baseline_lag_7',
        'workplaces_percent_change_from_baseline_lag_14',
        'people_fully_vaccinated_per_hundred',
        'total_boosters_per_hundred',
        'day_of_week', 'month', 'day_of_year', 'week'
    ]

def get_risk_label(row):
    """
    Risk based on rolling_7 (7-day avg).
    rolling_7 is EXCLUDED from classifier features — no leakage.
    """
    avg = row['rolling_7']
    if avg < 1000:
        return 0  # Low
    elif avg < 10000:
        return 1  # Medium
    else:
        return 2  # High

def prepare_dataset():
    df = load_data()
    df = add_features(df)
    df['risk_label'] = df.apply(get_risk_label, axis=1)
    return df