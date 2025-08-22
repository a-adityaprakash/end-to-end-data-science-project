# utils.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes


def collect_data() -> Tuple[pd.DataFrame, pd.Series]:
    """
    Data collection step.
    Uses sklearn's Diabetes regression dataset (10 numeric features, offline).
    Returns X (DataFrame) and y (Series).
    """
    ds = load_diabetes(as_frame=True)
    X: pd.DataFrame = ds.data.copy()
    y: pd.Series = ds.target.copy()
    return X, y


def save_json(obj, path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
