
"""Fraud-scoring FastAPI service.

Versioning contract: MODEL_VERSION = Git SHA that produced the artifact.
In CI, GitHub Actions injects it automatically. Locally it defaults to "dev".
"""
import os
import time

import joblib
from fastapi import FastAPI, HTTPException

from app.pipeline import featurize
from app.schemas import FraudScore, Transaction

MODEL_PATH = os.getenv("MODEL_PATH", "models/model.pkl")
MODEL_VERSION = os.getenv("MODEL_VERSION", "dev")

app = FastAPI(title="fraud-demo scoring API", version=MODEL_VERSION)
model = joblib.load(MODEL_PATH)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_version": MODEL_VERSION}


@app.post("/score", response_model=FraudScore)
def score(tx: Transaction) -> FraudScore:
    start = time.perf_counter()
    try:
        proba = float(model.predict_proba(featurize(tx.model_dump()))[0, 1])
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    latency_ms = (time.perf_counter() - start) * 1000
    print(f"score latency_ms={latency_ms:.2f}")
    return FraudScore(is_fraud=proba >= 0.5, fraud_probability=round(proba, 4),
                      model_version=MODEL_VERSION)
