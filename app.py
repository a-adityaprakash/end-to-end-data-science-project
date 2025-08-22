# app.py
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Dict, Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from schema import PredictRequest, PredictResponse

APP_DIR = Path(__file__).parent.resolve()
ARTIFACT_PATH = APP_DIR / "artifacts" / "model.joblib"

app = FastAPI(
    title="Task 3 — Diabetes Progression Regressor",
    description=(
        "Full data science project: data collection, preprocessing, "
        "model training, and deployment via FastAPI."
    ),
    version="1.0.0",
)

templates = Jinja2Templates(directory=str(APP_DIR / "templates"))


def _load_model():
    if not ARTIFACT_PATH.exists():
        raise FileNotFoundError(
            f"Model artifact not found at {ARTIFACT_PATH}. "
            f"Run `python train.py` first."
        )
    obj = joblib.load(ARTIFACT_PATH)
    return obj["pipeline"], obj


PIPELINE, META = _load_model()


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": True}


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest):
    try:
        df = pd.DataFrame(payload.records)
        # Ensure training-time feature order
        expected = META["feature_names"]
        missing = [c for c in expected if c not in df.columns]
        extra = [c for c in df.columns if c not in expected]
        if missing:
            raise HTTPException(status_code=400, detail=f"Missing features: {missing}")
        if extra:
            # Keep only known features; ignore extras
            df = df[expected]

        preds = PIPELINE.predict(df)
        return PredictResponse(
            predictions=[float(x) for x in preds],
            model_info={
                "features": expected,
                "target": META["target_name"],
                "sklearn": META["sklearn_version"],
            },
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ----- Minimal Web UI to demo the model -----

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "features": META["feature_names"]})


@app.post("/web-predict", response_class=HTMLResponse)
def web_predict(request: Request, **form_data):
    try:
        features = META["feature_names"]
        row = {k: float(form_data.get(k, "0")) for k in features}
        df = pd.DataFrame([row])
        pred = float(PIPELINE.predict(df)[0])
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "features": features, "last_input": row, "prediction": pred},
        )
    except Exception as e:
        return HTMLResponse(f"<h3>Error: {e}</h3>", status_code=500)
