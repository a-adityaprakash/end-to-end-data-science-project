# Task 3 — End-to-End Data Science Project with FastAPI Deployment

## Objective
Develop a full data science project covering **data collection, preprocessing, model training**, and **deployment** as an API/web app using **FastAPI**. The deliverable is a running service that exposes the model for real-time predictions and a minimal web interface to showcase functionality.

## Overview
- **Dataset**: `sklearn.datasets.load_diabetes` (10 numeric features; offline, reproducible).
- **Problem Type**: Regression (disease progression prediction).
- **Preprocessing**: Median imputation, standard scaling (scikit-learn `Pipeline` + `ColumnTransformer`).
- **Model**: Random Forest Regressor (300 trees).
- **Artifacts**: A single serialized pipeline (`artifacts/model.joblib`) containing both preprocessing and the model.
- **Deployment**: FastAPI app with:
  - `POST /predict` for batch JSON inference.
  - `GET /` minimal web form to demo the model.
  - `GET /docs` interactive OpenAPI/Swagger UI.

## Project Structure
task3/
├─ README.md
├─ requirements.txt
├─ train.py
├─ app.py
├─ schema.py
├─ utils.py
├─ templates/
│ └─ index.html
└─ artifacts/
└─ model.joblib # created by train.py


## Setup
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt

## Train the Model
python train.py --out_dir artifacts

Outputs:

artifacts/model.joblib — serialized preprocessing+model pipeline

artifacts/metrics.json — train/val/test metrics (R², MAE, RMSE)

## Run the API/Web App
uvicorn app:app --reload --host 0.0.0.0 --port 8000

Open http://localhost:8000/
 for the simple web demo.

Open http://localhost:8000/docs
 for interactive Swagger UI.

## Example Inference (REST)

Request:

curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
        "records": [
          {"age": 0.05, "sex": -0.04, "bmi": 0.03, "bp": 0.02,
           "s1": -0.02, "s2": 0.01, "s3": -0.01, "s4": 0.03, "s5": 0.04, "s6": -0.05}
        ]
      }'


Response:

{
  "predictions": [158.42],
  "model_info": {
    "features": ["age","sex","bmi","bp","s1","s2","s3","s4","s5","s6"],
    "target": "disease_progression",
    "sklearn": "1.5.x"
  }
}

## Academic Notes

Demonstrates the full ML lifecycle: collection → preprocessing → training → evaluation → packaging → deployment.

Uses a single scikit-learn Pipeline to ensure training-inference parity and prevent data leakage.

Provides reproducibility (fixed random seed) and separation of concerns (schema for I/O, utilities for data).

Deployment includes both API endpoints and a minimal web UI for non-technical stakeholders.

Optional: Docker (if required by host)

Create Dockerfile:

FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# Ensure model artifact exists (or run train at build time if acceptable)
# RUN python train.py --out_dir artifacts
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]


Build and run:

docker build -t task3-api .
docker run -p 8000:8000 task3-api

## Notes on Inputs

The diabetes dataset uses standardized/centered feature values. For production use with raw clinical data, add a feature engineering step to map raw inputs to the standardized format used in training.


---

## How to use
1) Put these files under `intern-cert/task3/`.  
2) `pip install -r requirements.txt`  
3) `python train.py` (creates `artifacts/model.joblib`)  
4) `uvicorn app:app --reload` and open `http://localhost:8000/` or test `/docs`.
