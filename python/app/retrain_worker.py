from __future__ import annotations

import asyncio
import contextlib
import logging
import signal

import mlflow
from mlflow.tracking import MlflowClient
import redis.asyncio as redis

from app.agents.watchdog import Watchdog
from app.config import Settings
from app.contracts import AgentMemoryProtocol, WatchdogProtocol
from app.data.local_loader import build_data_loader
from app.mcp.client import MCPClient
from app.memory.memory import AgentMemory
from app.orchestrator import Orchestrator

logger = logging.getLogger(__name__)


async def _run() -> None:
    settings = Settings()
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
    await orchestrator.start_stream_workers(include_forecast=False, include_retrain=True)

    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()

    def _stop() -> None:
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(sig, _stop)

    logger.info("Retrain worker started and listening on retrain stream")
    await stop_event.wait()

    await orchestrator.stop_stream_workers()
    if hasattr(valkey, "aclose"):
        await valkey.aclose()
    else:
        await valkey.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(_run())
