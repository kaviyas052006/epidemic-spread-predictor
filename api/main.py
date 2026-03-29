from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import predict, risk, metrics, train, anomaly, explain

app = FastAPI(
    title="Epidemic Spread Predictor API",
    description="ML-powered epidemic forecasting and outbreak risk detection",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict.router)
app.include_router(risk.router)
app.include_router(metrics.router)
app.include_router(train.router)
app.include_router(anomaly.router)
app.include_router(explain.router)

@app.get("/")
def root():
    return {
        "message": "Epidemic Spread Predictor API is running ✅",
        "endpoints": ["/predict", "/risk", "/metrics", "/train", "/anomaly/{country}", "/explain/{country}"]
    }