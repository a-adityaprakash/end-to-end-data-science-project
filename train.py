# train.py
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

from utils import collect_data, save_json, rmse


def build_pipeline(numeric_cols):
    """
    Preprocessing + Model as a single sklearn Pipeline.
    """
    num_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    pre = ColumnTransformer(
        transformers=[("num", num_pipe, numeric_cols)],
        remainder="drop",
    )

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
    )

    pipe = Pipeline(steps=[
        ("preprocess", pre),
        ("model", model),
    ])
    return pipe


def main(test_size=0.2, val_size=0.1, random_state=42, out_dir="artifacts"):
    # -------- Data collection
    X, y = collect_data()
    numeric_cols = X.columns.tolist()

    # -------- Split: train/val/test
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # From train_full carve out validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=val_size, random_state=random_state
    )

    # -------- Pipeline
    pipe = build_pipeline(numeric_cols)

    # -------- Train
    pipe.fit(X_train, y_train)

    # -------- Evaluate
    def eval_split(Xs, ys):
        preds = pipe.predict(Xs)
        return {
            "r2": r2_score(ys, preds),
            "mae": float(mean_absolute_error(ys, preds)),
            "rmse": rmse(np.asarray(ys), preds),
        }

    metrics = {
        "train": eval_split(X_train, y_train),
        "val": eval_split(X_val, y_val),
        "test": eval_split(X_test, y_test),
    }

    # -------- Persist model and metadata
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    model_path = out_path / "model.joblib"
    joblib.dump({
        "pipeline": pipe,
        "feature_names": numeric_cols,
        "target_name": "disease_progression",
        "sklearn_version": "1.5.x"
    }, model_path)

    save_json(metrics, out_path / "metrics.json")

    print("[INFO] Model saved:", model_path.resolve())
    print("[INFO] Metrics:", metrics)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--val_size", type=float, default=0.1)
    ap.add_argument("--out_dir", type=str, default="artifacts")
    args = ap.parse_args()

    main(test_size=args.test_size, val_size=args.val_size, out_dir=args.out_dir)
