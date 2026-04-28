from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import time

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi import status
from fastapi.responses import PlainTextResponse
import mlflow
from mlflow.tracking import MlflowClient
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
import redis.asyncio as redis

from app.agents.watchdog import Watchdog
from app.agents.chat_router import ChatRouterAgent
from app.config import Settings
from app.contracts import AgentMemoryProtocol, WatchdogProtocol
from app.data.local_loader import build_data_loader
from app.mcp.client import MCPClient
from app.memory.memory import AgentMemory
from app.orchestrator import Orchestrator
from app.registry.data_registry import DataRegistry
from app.schemas import (
    AdminModelResponse,
    AdminRetrainRequest,
    AdminRetrainResponse,
    ChatRequest,
    FrequencyForecastResponse,
    ForecastRequest,
    ForecastResponse,
    MultiFrequencyForecastResponse,
)

app = FastAPI(title="sktime-agentic")
settings = Settings()

FORECAST_REQUESTS_TOTAL = Counter(
    "forecast_requests_total",
    "Total number of forecast requests served by Python API",
)
FORECAST_ERRORS_TOTAL = Counter(
    "forecast_errors_total",
    "Total number of forecast requests that failed",
)
FORECAST_LATENCY_SECONDS = Histogram(
    "forecast_latency_seconds",
    "End-to-end latency for /forecast endpoint",
)
WORKER_READY = Gauge(
    "python_worker_ready",
    "Readiness state of Python worker (1 ready, 0 not ready)",
)


def _decode_redis_value(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, str):
        return value
    return str(value)


def _require_admin_token(
    x_admin_token: str | None = Header(default=None, alias="X-Admin-Token"),
) -> None:
    required_token = str(getattr(settings, "admin_api_token", "") or "").strip()
    if not required_token:
        return

    if x_admin_token != required_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid admin token",
        )


@app.on_event("startup")
async def startup_event() -> None:
    valkey = redis.from_url(settings.valkey_url, decode_responses=False)

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow_client = MlflowClient(tracking_uri=settings.mlflow_tracking_uri)

    data_loader = getattr(settings, "data_loader", None)
    if data_loader is None:
        local_dataset_dir = str(getattr(settings, "local_dataset_dir", "") or "").strip()
        data_loader = build_data_loader(local_dataset_dir or None)
        settings.data_loader = data_loader
    memory_loader = getattr(settings, "memory_loader", None)
    mcp_client = MCPClient(data_loader=data_loader, memory_loader=memory_loader)
    agent_memory: AgentMemoryProtocol = AgentMemory(valkey)
    watchdog: WatchdogProtocol = Watchdog(valkey, settings)

    orchestrator = Orchestrator(
        valkey,
        mlflow_client,
        mcp_client,
        settings,
        agent_memory=agent_memory,
        watchdog=watchdog,
    )
    await orchestrator.startup_cleanup()
    await orchestrator.start_stream_workers(
        include_forecast=bool(getattr(settings, "enable_forecast_worker", True)),
        include_retrain=bool(getattr(settings, "enable_retrain_worker", False)),
    )

    app.state.valkey = valkey
    app.state.mlflow_client = mlflow_client
    app.state.mcp_client = mcp_client
    app.state.agent_memory = agent_memory
    app.state.watchdog = watchdog
    app.state.orchestrator = orchestrator
    
    app.state.data_registry = DataRegistry(valkey)

    chat_router = ChatRouterAgent(settings, data_loader)
    app.state.chat_router = chat_router


@app.on_event("shutdown")
async def shutdown_event() -> None:
    orchestrator = getattr(app.state, "orchestrator", None)
    if orchestrator is not None:
        await orchestrator.stop_stream_workers()

    valkey = getattr(app.state, "valkey", None)
    if valkey is not None:
        if hasattr(valkey, "aclose"):
            await valkey.aclose()
        else:
            await valkey.close()


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.get("/ready")
async def ready_check():
    valkey = getattr(app.state, "valkey", None)
    if valkey is None:
        WORKER_READY.set(0)
        raise HTTPException(status_code=503, detail="Valkey client not initialized")

    try:
        await valkey.ping()
    except Exception as exc:
        WORKER_READY.set(0)
        raise HTTPException(status_code=503, detail=f"Valkey not ready: {exc}") from exc

    WORKER_READY.set(1)
    return {"status": "ready"}


@app.get("/metrics")
def metrics() -> PlainTextResponse:
    payload = generate_latest().decode("utf-8")
    return PlainTextResponse(content=payload, media_type=CONTENT_TYPE_LATEST)


@app.post("/forecast", response_model=ForecastResponse)
async def forecast(request: ForecastRequest) -> ForecastResponse:
    orchestrator: Orchestrator = app.state.orchestrator
    FORECAST_REQUESTS_TOTAL.inc()
    started = time.perf_counter()
    try:
        return await orchestrator.handle_job(request)
    except Exception as exc:
        FORECAST_ERRORS_TOTAL.inc()
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        FORECAST_LATENCY_SECONDS.observe(max(0.0, time.perf_counter() - started))


@app.post("/chat", response_model=ForecastResponse | MultiFrequencyForecastResponse)
async def chat_interaction(request: ChatRequest) -> ForecastResponse | MultiFrequencyForecastResponse:
    chat_router: ChatRouterAgent = app.state.chat_router
    orchestrator: Orchestrator = app.state.orchestrator
    data_registry: DataRegistry = app.state.data_registry

    # 1. Discover available datasets dynamically from Valkey registry.
    dataset_records = await data_registry.get_all_records()
    
    # Optional fallback for testing without adding any streams
    if not dataset_records:
        dataset_records = {
            "airline": {
                "description": "Monthly totals of international airline passengers. (Built-in Fallback)",
                "frequency": "M",
            }
        }
                
    # 2. Parse request via Agent
    try:
        forecast_requests = await chat_router.route_request(request.query, dataset_records)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if len(forecast_requests) == 1:
        forecast_request = forecast_requests[0]
        FORECAST_REQUESTS_TOTAL.inc()
        started = time.perf_counter()
        try:
            return await orchestrator.handle_job(forecast_request)
        except Exception as exc:
            FORECAST_ERRORS_TOTAL.inc()
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        finally:
            FORECAST_LATENCY_SECONDS.observe(max(0.0, time.perf_counter() - started))
        
    # 3. Multi-frequency chat request: run one forecast per selected dataset.
    FORECAST_REQUESTS_TOTAL.inc(len(forecast_requests))
    started = time.perf_counter()

    try:
        results = await asyncio.gather(
            *(orchestrator.handle_job(job) for job in forecast_requests),
            return_exceptions=True,
        )

        forecasts: list[FrequencyForecastResponse] = []
        for job, result in zip(forecast_requests, results):
            if isinstance(result, Exception):
                FORECAST_ERRORS_TOTAL.inc()
                forecasts.append(
                    FrequencyForecastResponse(
                        dataset_id=job.dataset_id,
                        frequency=job.frequency,
                        error=str(result),
                    )
                )
            else:
                forecasts.append(
                    FrequencyForecastResponse(
                        dataset_id=job.dataset_id,
                        frequency=job.frequency,
                        forecast=result,
                    )
                )

        if not any(item.forecast is not None for item in forecasts):
            first_error = forecasts[0].error if forecasts else "all frequency forecasts failed"
            raise HTTPException(status_code=500, detail=str(first_error))

        return MultiFrequencyForecastResponse(forecasts=forecasts)
    except HTTPException:
        raise
    except Exception as exc:
        FORECAST_ERRORS_TOTAL.inc()
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        FORECAST_LATENCY_SECONDS.observe(max(0.0, time.perf_counter() - started))


@app.post("/admin/retrain", response_model=AdminRetrainResponse)
async def admin_retrain(
    body: AdminRetrainRequest,
    _auth: None = Depends(_require_admin_token),
) -> AdminRetrainResponse:
    valkey = app.state.valkey
    lock_key = f"retrain_lock:{body.dataset_id}"

    if await valkey.exists(lock_key):
        return AdminRetrainResponse(
            dataset_id=body.dataset_id,
            reason=body.reason,
            queued=False,
            stream_id=None,
        )

    await valkey.setex(lock_key, settings.retrain_lock_ttl_seconds, "1")
    try:
        stream_id = await valkey.xadd(
            "retrain:jobs",
            {
                "dataset_id": body.dataset_id,
                "reason": body.reason,
                "triggered_at": datetime.now(tz=timezone.utc).isoformat(),
                "trigger": "admin",
            },
        )
    except Exception:
        await valkey.delete(lock_key)
        raise

    return AdminRetrainResponse(
        dataset_id=body.dataset_id,
        reason=body.reason,
        queued=True,
        stream_id=_decode_redis_value(stream_id),
    )


@app.get("/admin/model/{dataset_id}", response_model=AdminModelResponse)
async def admin_model_info(
    dataset_id: str,
    _auth: None = Depends(_require_admin_token),
) -> AdminModelResponse:
    valkey = app.state.valkey
    mlflow_client: MlflowClient = app.state.mlflow_client
    agent_memory: AgentMemoryProtocol | None = getattr(app.state, "agent_memory", None)

    version_raw = await valkey.get(f"model_version:{dataset_id}")
    if not version_raw:
        version_raw = await valkey.get(f"model:version:{dataset_id}")
    model_version = _decode_redis_value(version_raw) if version_raw else None

    class_raw = await valkey.get(f"model:class:{dataset_id}")
    model_class = _decode_redis_value(class_raw) if class_raw else None

    cv_score = None
    promoted_at = None
    drift_reason = None

    if agent_memory is not None:
        try:
            memory = await agent_memory.get_dataset_memory(dataset_id)
            model_history = memory.get("model_history") or []
            drift_events = memory.get("drift_events") or []
            if model_history:
                latest_model = model_history[-1]
                if latest_model.get("val_mae") is not None:
                    cv_score = float(latest_model["val_mae"])
                promoted_at = latest_model.get("promoted_at")
            if drift_events:
                drift_reason = drift_events[-1].get("method")
        except Exception:
            pass

    if cv_score is None and model_version is not None:
        try:
            model_name = f"ts-forecaster-{dataset_id}"
            versions = mlflow_client.get_latest_versions(
                model_name,
                stages=["Production", "Staging", "None"],
            )
            if versions:
                latest = sorted(versions, key=lambda v: int(v.version), reverse=True)[0]
                run = mlflow_client.get_run(latest.run_id)
                val_mae = run.data.metrics.get("val_mae")
                if val_mae is not None:
                    cv_score = float(val_mae)
        except Exception:
            pass

    return AdminModelResponse(
        dataset_id=dataset_id,
        model_version=model_version,
        model_class=model_class,
        cv_score=cv_score,
        promoted_at=promoted_at,
        drift_reason=drift_reason,
    )