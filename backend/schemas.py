from pydantic import BaseModel
from typing import Optional
import time


class PredictionResponse(BaseModel):
    report: str
    confidence: float
    processing_time_ms: int
    model_version: str


class DicomPredictionResponse(BaseModel):
    report: str
    confidence: float
    processing_time_ms: int
    model_version: str
    window_applied: str
    patient_info: dict


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str
    uptime_seconds: float


class MetadataResponse(BaseModel):
    model_version: str
    vocab_size: int
    training_dataset: str
    embed_dim: int
    num_heads: int
    num_layers: int
    max_len: int


class ErrorResponse(BaseModel):
    error: str
    detail: str
    request_id: str
