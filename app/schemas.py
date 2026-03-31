from typing import Annotated

from pydantic import BaseModel, Field, field_validator


class PredictRequest(BaseModel):
    features: Annotated[list[float], Field(min_length=30, max_length=30)]

    @field_validator("features")
    @classmethod
    def validate_range(cls, v: list[float]) -> list[float]:
        for i, val in enumerate(v):
            if not (-10.0 <= val <= 10.0):
                raise ValueError(
                    f"Feature at index {i} has value {val} which is out of allowed range [-10.0, 10.0]"
                )
        return v


class PredictResponse(BaseModel):
    prediction: str
    confidence: float
    probabilities: dict[str, float]
    model_version: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    db_connected: bool


class VersionResponse(BaseModel):
    model_version: str
    api_version: str
