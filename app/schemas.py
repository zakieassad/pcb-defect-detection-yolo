from __future__ import annotations

from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class Detection(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    bbox: BoundingBox


class PredictionResponse(BaseModel):
    filename: str

    image_width: int
    image_height: int

    detection_count: int
    detections: list[Detection]

    backend: str
    model_name: str
    model_version: str

