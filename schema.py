# schema.py
from __future__ import annotations

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    # Accept either a single record or a batch
    records: List[Dict[str, Any]] = Field(
        ..., description="List of feature dicts matching the training schema."
    )


class PredictResponse(BaseModel):
    predictions: List[float]
    model_info: dict
