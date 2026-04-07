import os

files = {
    "python/requirements.txt": """sktime
pandas>=2.0
numpy>=1.26
mlflow
mlflow-sktime
redis>=4.2
fastapi
uvicorn
pydantic>=2.0
pydantic-settings
anthropic
pytest
pytest-asyncio
alibi-detect
river
""",
    "python/app/core/config.py": """from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    valkey_url: str = "redis://localhost:6379"
    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_artifact_uri: str = "s3://sktime-agentic-models"
    s3_bucket: str = "sktime-agentic-models"
    s3_region: str = "us-east-1"

    min_history_length: int = 10
    incremental_update_wait_seconds: int = 10

    retrain_lock_ttl_seconds: int = 1800
    max_training_time_seconds: int = 3600

    llm_provider: str = "anthropic"
    llm_api_key: str = "dummy"
    llm_model: str = "claude-sonnet-4-20250514"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
""",
    "python/app/schemas/api.py": """from pydantic import BaseModel, field_validator
from datetime import datetime

class ForecastRequest(BaseModel):
    dataset_id: str
    fh: list[int]
    correlation_id: str
    frequency: str | None = None

    @field_validator("fh")
    def fh_must_be_positive(cls, v):
        if any(h <= 0 for h in v):
            raise ValueError("all horizon values must be positive integers")
        return v

class ForecastResponse(BaseModel):
    dataset_id: str
    predictions: list[float]
    prediction_intervals: dict | None = None
    model_version: str
    model_class: str
    model_status: str
    drift_score: float | None = None
    drift_method: str | None = None
    warning: str | None = None
    cache_hit: bool
    correlation_id: str
""",
    "python/app/orchestrator.py": """from app.schemas.api import ForecastRequest, ForecastResponse
import asyncio

class Orchestrator:
    def __init__(self, valkey, mlflow_client, settings):
        self.valkey = valkey
        self.mlflow = mlflow_client
        self.settings = settings
        self.model_cache = {}
        
    async def handle_job(self, job: ForecastRequest) -> ForecastResponse:
        # Check if model promotion happened since last request
        await self._maybe_reload_model(job.dataset_id)
        # TODO: call agents based on model existence
        pass

    async def _maybe_reload_model(self, dataset_id: str) -> None:
        signal = await self.valkey.get(f"model_updated:{dataset_id}")
        if signal:
            pass

    async def _cleanup_orphaned_locks(self) -> None:
        async for key in self.valkey.scan_iter("model_lock:*"):
            ttl = await self.valkey.ttl(key)
            if ttl < 0:
                await self.valkey.delete(key)
""",
    "go/cmd/server/main.go": """package main

import (
\t"log"
\t"github.com/gofiber/fiber/v2"
)

func main() {
\tapp := fiber.New()

\tapp.Get("/health", func(c *fiber.Ctx) error {
\t\treturn c.SendString("OK")
\t})

\tapp.Get("/ready", func(c *fiber.Ctx) error {
\t\treturn c.SendString("READY")
\t})

\tlog.Fatal(app.Listen(":8080"))
}
""",
    "go/go.mod": """module sktime-agentic

go 1.23

require (
\tgithub.com/go-playground/validator/v10 v10.17.0
\tgithub.com/gofiber/fiber/v2 v2.52.0
\tgithub.com/google/uuid v1.6.0
\tgithub.com/redis/go-redis/v9 v9.4.0
)
""",
    "docker-compose.yml": """services:
  valkey:
    image: valkey/valkey:8
    ports:
      - "6379:6379"

  mlflow:
    image: ghcr.io/mlflow/mlflow:latest
    ports:
      - "5000:5000"
    command: >
      mlflow server
      --host 0.0.0.0
      --backend-store-uri sqlite:///mlflow.db
      --default-artifact-root s3://sktime-agentic-models

  python-worker:
    build:
      context: ./python
    depends_on:
      - valkey
      - mlflow

  go-gateway:
    build:
      context: ./go
    ports:
      - "8080:8080"
    depends_on:
      - valkey
"""
}

for path, content in files.items():
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

print("Files populated successfully.")
