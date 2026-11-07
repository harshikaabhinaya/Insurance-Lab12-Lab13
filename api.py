from datetime import datetime, timezone
from pathlib import Path
import json
import time
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from src.common import MODEL_PATH, LOGS

app = FastAPI(title="Insurance Charges Prediction API", version="1.0.0")
model = None

class InsuranceInput(BaseModel):
    age: int = Field(ge=18, le=100)
    sex: str
    bmi: float = Field(gt=0, le=100)
    children: int = Field(ge=0, le=20)
    smoker: str
    region: str

@app.on_event("startup")
def load_model():
    global model
    if MODEL_PATH.exists():
        model = joblib.load(MODEL_PATH)

@app.middleware("http")
async def record_latency(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    latency = (time.perf_counter() - start) * 1000
    LOGS.mkdir(parents=True, exist_ok=True)
    with (LOGS / "api_requests.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({"timestamp": datetime.now(timezone.utc).isoformat(),
                            "path": request.url.path, "status_code": response.status_code,
                            "latency_ms": round(latency, 3)}) + "\n")
    return response

@app.get("/")
def home():
    return {"message": "Insurance prediction API is running", "docs": "/docs"}

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict")
def predict(payload: InsuranceInput):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not found. Run python 13_end_to_end.py first.")
    row = pd.DataFrame([payload.model_dump()])
    prediction = float(model.predict(row)[0])
    return {"predicted_insurance_charges": round(prediction, 2), "currency": "dataset units"}
