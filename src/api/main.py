from __future__ import annotations

from collections import Counter
from pathlib import Path
import logging
import sys
import time

import joblib
import pandas as pd
from fastapi import FastAPI, Request
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.models.package_model import FINAL_MODEL_PATH

LOG_DIR = PROJECT_ROOT / "artifacts" / "monitoring"
LOG_FILE = LOG_DIR / "api_requests.log"

LOG_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("heart_disease_api")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

app = FastAPI(
    title="Heart Disease Prediction API",
    version="1.0.0",
    description="FastAPI service for heart disease risk prediction.",
)

REQUEST_METRICS = {
    "total_requests": 0,
    "predict_requests": 0,
    "health_requests": 0,
    "root_requests": 0,
    "total_prediction_time_seconds": 0.0,
    "response_status_counts": Counter(),
}


class PredictionRequest(BaseModel):
    age: float
    sex: float
    cp: float
    trestbps: float
    chol: float
    fbs: float
    restecg: float
    thalach: float
    exang: float
    oldpeak: float
    slope: float
    ca: float
    thal: float


def load_model():
    if not FINAL_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Packaged model not found at: {FINAL_MODEL_PATH}. Run package_model.py first."
        )
    return joblib.load(FINAL_MODEL_PATH)


MODEL = load_model()


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.perf_counter()
    response = await call_next(request)
    duration = time.perf_counter() - start_time

    REQUEST_METRICS["total_requests"] += 1
    REQUEST_METRICS["response_status_counts"][response.status_code] += 1

    if request.url.path == "/predict":
        REQUEST_METRICS["predict_requests"] += 1
        REQUEST_METRICS["total_prediction_time_seconds"] += duration
    elif request.url.path == "/health":
        REQUEST_METRICS["health_requests"] += 1
    elif request.url.path == "/":
        REQUEST_METRICS["root_requests"] += 1

    logger.info(
        "path=%s method=%s status_code=%s duration_seconds=%.6f client=%s",
        request.url.path,
        request.method,
        response.status_code,
        duration,
        request.client.host if request.client else "unknown",
    )
    return response


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "Heart Disease Prediction API is running."}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/metrics")
def metrics() -> dict[str, object]:
    predict_requests = REQUEST_METRICS["predict_requests"]
    average_prediction_time = 0.0
    if predict_requests:
        average_prediction_time = (
            REQUEST_METRICS["total_prediction_time_seconds"] / predict_requests
        )

    return {
        "total_requests": REQUEST_METRICS["total_requests"],
        "predict_requests": predict_requests,
        "health_requests": REQUEST_METRICS["health_requests"],
        "root_requests": REQUEST_METRICS["root_requests"],
        "average_prediction_time_seconds": round(average_prediction_time, 6),
        "response_status_counts": dict(REQUEST_METRICS["response_status_counts"]),
        "log_file": str(LOG_FILE),
    }


@app.post("/predict")
def predict(request: PredictionRequest) -> dict[str, float | int]:
    input_df = pd.DataFrame([request.model_dump()])
    prediction = int(MODEL.predict(input_df)[0])
    confidence = float(MODEL.predict_proba(input_df)[0][1])

    return {
        "prediction": prediction,
        "confidence": round(confidence, 6),
    }
