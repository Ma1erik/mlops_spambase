from typing import List, Optional, Dict
from pydantic import BaseModel, Field, field_validator


class SpambaseItem(BaseModel):
    features: List[float] = Field(..., min_length=57, max_length=57)

    @field_validator("features")
    @classmethod
    def validate_features_non_negative(cls, v: List[float]) -> List[float]:
        for idx, val in enumerate(v):
            if val < 0:
                raise ValueError(
                    f"Feature at index {idx} must be non-negative (got {val})"
                )
        return v

    model_config = {
        "json_schema_extra": {"example": {"features": [0.0] * 54 + [1.0, 1.0, 1.0]}}
    }


class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    model_version: str
    status: str = "ok"


class ModelVersionInfo(BaseModel):
    version: str
    model_type: str
    path: str
    sha256: str
    accuracy: float
    f1_score: float
    status: str


class RegistryState(BaseModel):
    active_version: str
    previous_version: Optional[str] = None
    models: Dict[str, ModelVersionInfo]


class SwitchVersionRequest(BaseModel):
    version: Optional[str] = None
    reason: str


class HealthResponse(BaseModel):
    status: str
    active_version: str
    uptime_seconds: float
    loaded_models_count: int


class MetricsResponse(BaseModel):
    requests_total: int
    predictions_total: int
    errors_total: int
    active_version: str
    latency_p50_ms: float
    latency_p95_ms: float
    memory_rss_mb: float
    status: str
