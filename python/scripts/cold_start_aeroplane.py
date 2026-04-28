import asyncio
import json
import sys
from pathlib import Path

from sktime.datasets import load_airline

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

import redis.asyncio as redis
from mlflow.tracking import MlflowClient

from app.config import Settings
from app.mcp.client import MCPClient
from app.memory.memory import AgentMemory
from app.agents.watchdog import Watchdog
from app.agents.model_selector import ModelSelectorAgent
from app.orchestrator import Orchestrator
from app.schemas import ForecastRequest


def load_dataset(dataset_id: str):
    if dataset_id == "airline":
        return load_airline()
    raise ValueError(f"Unsupported dataset_id: {dataset_id}")

async def main():
    root = Path(__file__).resolve().parents[1]
    settings = Settings(
        valkey_url='redis://valkey:6379',
        mlflow_tracking_uri='http://mlflow:5000',
        enable_forecast_worker=True,
        enable_retrain_worker=False,
    )

    valkey = redis.from_url(settings.valkey_url, decode_responses=False)
    mlflow_client = MlflowClient(tracking_uri=settings.mlflow_tracking_uri)
    mcp_client = MCPClient(data_loader=load_dataset, memory_loader=None)
    agent_memory = AgentMemory(valkey)
    watchdog = Watchdog(valkey, settings)
    orchestrator = Orchestrator(valkey, mlflow_client, mcp_client, settings, agent_memory=agent_memory, watchdog=watchdog)

    request = ForecastRequest(dataset_id='airline', fh=[1, 2, 3, 4, 5, 6], correlation_id='docker-cold-start')
    print('ForecastRequest:', request.model_dump())
    result = await orchestrator.handle_job(request)
    print('ForecastResponse:', json.dumps(result.model_dump(), indent=2))
    await valkey.close()

if __name__ == '__main__':
    asyncio.run(main())
