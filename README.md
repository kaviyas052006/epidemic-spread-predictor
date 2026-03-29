# 🦠 Epidemic Spread Predictor
### CodeCure Biohackathon — Track C

An AI-powered epidemic forecasting system that predicts COVID-19 case counts and outbreak risk levels across 122 countries.

---

## 🏆 Model Performance

| Metric | Value |
|--------|-------|
| Risk Classification F1 | **92.49%** |
| Case Forecasting MAPE | **14.04%** |
| Walk-Forward MAPE | **14.55%** |
| Beats Naive Baseline | 347% → 14% |
| Beats Moving Average | 43% → 14% |
| Data Leakage | None ✅ |

---

## 🎯 What It Does

- **Predicts future case counts** for any country (next N days)
- **Classifies outbreak risk** as Low / Medium / High
- **Detects anomalies** — sudden spikes and early warnings
- **Explains predictions** using SHAP feature importance

---

## 🧠 ML Architecture

### Case Forecasting
- **Models**: LightGBM + XGBoost weighted ensemble
- **Strategy**: Countries clustered into 3 groups by case scale
- **Target**: Log-transformed (log1p) to handle scale differences
- **Split**: Chronological date-based (no random split)
- **Validation**: Walk-forward validation across 3 time splits

### Risk Classification
- **Model**: LightGBM Classifier
- **Labels**: Low / Medium / High (based on 7-day rolling average)
- **F1 Score**: 92.49% (no data leakage)

### Anomaly Detection
- **Model**: Isolation Forest
- **Purpose**: Early warning system for sudden spikes

### Explainability
- **Method**: SHAP values
- **Output**: Feature importance per country

---

## 📊 Dataset

| Source | Description |
|--------|-------------|
| Johns Hopkins COVID-19 | Daily confirmed cases, 122 countries |
| Our World in Data | Vaccination, testing, hospitalization |
| Google Mobility | Workplace, retail, residential mobility |

- **Total rows**: 115,818 (after duplicate removal)
- **Countries**: 122
- **Date range**: 2020 — 2022

---

## ⚙️ Backend API

Built with **FastAPI** — 6 endpoints:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict` | POST | Forecast future cases |
| `/risk` | POST | Get outbreak risk level |
| `/metrics` | GET | Model performance metrics |
| `/train` | POST | Trigger model retraining |
| `/anomaly/{country}` | GET | Detect case spikes |
| `/explain/{country}` | GET | SHAP feature importance |

---

## 🗂️ Project Structure
```
epidemic-spread-predictor/
├── data/
│   └── final_processed_epidemic_data.csv
├── models/
│   ├── forecaster/
│   └── risk_classifier/
├── src/
│   ├── feature_engineering.py
│   ├── train_forecaster.py
│   ├── train_classifier.py
│   ├── predict.py
│   ├── anomaly_detection.py
│   └── explainability.py
├── api/
│   ├── main.py
│   ├── schemas.py
│   └── routes/
│       ├── predict.py
│       ├── risk.py
│       ├── metrics.py
│       ├── train.py
│       ├── anomaly.py
│       └── explain.py
├── notebooks/
│   └── TrackC_Data_Preprocessing.ipynb
├── requirements.txt
└── README.md
```

---

## 🚀 How To Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train models
```bash
python -m src.train_forecaster
python -m src.train_classifier
```

### 3. Start API server
```bash
uvicorn api.main:app --reload
```

### 4. Test API
```
http://127.0.0.1:8000/docs
```

---

## 🔬 Feature Engineering

| Feature | Description |
|---------|-------------|
| lag_1, lag_3, lag_7, lag_14 | Past case counts |
| rolling_7, rolling_14, rolling_21 | Rolling averages |
| growth_rate | Daily % change |
| pct_change_7 | 7-day % change |
| trend | Rising or falling indicator |
| acceleration | Change in growth rate |
| Mobility features | Google mobility data |
| Vaccination features | People fully vaccinated |

---

## 👥 Team

| Role | Responsibility |
|------|---------------|
| Person 1 | Data Preprocessing |
| Person 2 (Madhu) | ML Model + Backend API |
| Person 3 | Frontend UI |
| Person 4 | Git Management |

---

## 📝 Honest Notes

- Worst-case errors occur on countries with **irregular reporting patterns** (bulk case dumps after reporting gaps). These are data artifacts, not model failures.
- **95% of predictions** are within 54% accuracy (P95 error).
- All metrics are computed with **chronological split** — no future data leaks into training.

---

## 🛠️ Tech Stack

`Python` `FastAPI` `LightGBM` `XGBoost` `Scikit-learn` `SHAP` `Pandas` `NumPy`