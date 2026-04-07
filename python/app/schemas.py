from pydantic import BaseModel, field_validator
from datetime import datetime
from typing import List, Optional, Dict, Any

class ForecastRequest(BaseModel):
    dataset_id: str
    fh: List[int]
    correlation_id: str
    frequency: Optional[str] = None

    @field_validator('fh')
    @classmethod
    def fh_must_be_positive(cls, v: List[int]) -> List[int]:
        if any(h <= 0 for h in v):
            raise ValueError('all horizon values must be positive integers')
        return v

class ForecastResponse(BaseModel):
    dataset_id: str
    predictions: List[float]
    prediction_intervals: Optional[Dict[str, List[float]]] = None
    model_version: str
    model_class: str
    model_status: str
    drift_score: Optional[float] = None
    drift_method: Optional[str] = None
    warning: Optional[str] = None
    llm_rationale: Optional[str] = None
    cache_hit: bool
    correlation_id: str

class RetrainJob(BaseModel):
    dataset_id: str
    reason: str
    triggered_at: datetime

class DataProfile(BaseModel):
    dataset_id: str
    length: int
    frequency: Optional[str]
    has_seasonality: bool
    is_stationary: bool
    missing_rate: float
    variance: float

    def to_natural_language(self) -> str:
        return f"Time series with {self.length} observations, freq={self.frequency}, missing rate={self.missing_rate:.1%}"\n