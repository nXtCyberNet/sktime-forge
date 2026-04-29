import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
import sys

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    
logging.basicConfig(level=logging.INFO)

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

import mlflow
import pandas as pd
import numpy as np
import redis.asyncio as redis
from mlflow.tracking import MlflowClient
from sktime.datasets import load_airline

from app.config import Settings
from app.mcp.client import MCPClient
from app.memory.memory import AgentMemory
from app.agents.watchdog import Watchdog
from app.orchestrator import Orchestrator
from app.schemas import ForecastRequest


# --- New generic data loader wired into settings ---
def data_loader(dataset_id: str) -> np.ndarray:
    """Generic loader - extend this dict as you add datasets."""
    loaders = {
        "airline": load_airline,
        # "m4_monthly": lambda: load_m4_weekly()["y"],
        # "electricity": load_electricity,
    }
    if dataset_id not in loaders:
        raise ValueError(
            f"Unknown dataset_id: {dataset_id}. Register it in data_loader."
        )
    return np.asarray(loaders[dataset_id](), dtype=float)


def build_dataset_loader(local_dataset_dir: Path):
    def load_dataset(dataset_id: str):
        if dataset_id == "airline":
            return load_airline()

        csv_file = local_dataset_dir / dataset_id
        if csv_file.is_file():
            df = pd.read_csv(csv_file)
            if "value" in df.columns:
                return df["value"]
            if df.shape[1] == 1:
                return df.iloc[:, 0]
            raise ValueError(
                "CSV must contain a single value column or a column named 'value'"
            )

        raise ValueError(
            f"Unsupported dataset_id '{dataset_id}'. Use 'airline' or a CSV filename in {local_dataset_dir}"
        )

    return load_dataset


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a local sktime-agentic cold-start demo"
    )
    parser.add_argument(
        "--dataset_id",
        default="airline",
        help="Dataset id or CSV filename in sample_datasets",
    )
    parser.add_argument(
        "--fh",
        nargs="+",
        type=int,
        default=[1, 2, 3, 4, 5, 6],
        help="Forecast horizon steps",
    )
    parser.add_argument(
        "--valkey_url",
        default=None,
        help="Valkey/Redis URL (default from python/.env or redis://localhost:6379)",
    )
    parser.add_argument(
        "--mlflow_tracking_uri",
        default=None,
        help="MLflow tracking URI (default from python/.env or http://localhost:5000)",
    )
    parser.add_argument(
        "--local_dataset_dir",
        default=None,
        help="Local sample dataset folder path",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    env_file = root / ".env"
    if not env_file.exists():
        env_file = root.parent / ".env"
    settings = Settings(_env_file=env_file)

    if args.valkey_url:
        settings.valkey_url = args.valkey_url
    if args.mlflow_tracking_uri:
        settings.mlflow_tracking_uri = args.mlflow_tracking_uri
    if args.local_dataset_dir:
        settings.local_dataset_dir = args.local_dataset_dir

    # --- Inject the new loader into settings ---
    settings.data_loader = data_loader

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)

    print("Using Valkey URL:", settings.valkey_url)
    print("Using MLflow tracking URI:", settings.mlflow_tracking_uri)

    dataset_dir = Path(
        settings.local_dataset_dir
        or root / "tests" / "fixtures" / "sample_datasets"
    )

    mcp_client = MCPClient(
        data_loader=build_dataset_loader(dataset_dir),
        memory_loader=None,
    )
    valkey = redis.from_url(settings.valkey_url, decode_responses=False)
    mlflow_client = MlflowClient(tracking_uri=settings.mlflow_tracking_uri)
    agent_memory = AgentMemory(valkey)
    watchdog = Watchdog(valkey, settings)
    orchestrator = Orchestrator(
        valkey,
        mlflow_client,
        mcp_client,
        settings,
        agent_memory=agent_memory,
        watchdog=watchdog,
    )

    request = ForecastRequest(
        dataset_id=args.dataset_id,
        fh=args.fh,
        correlation_id="demo-run",
    )

    print("Running demo for dataset:", args.dataset_id)
    print("Forecast horizon:", args.fh)

    result = await orchestrator.handle_job(request)
    print("Forecast result:")
    print(json.dumps(result.model_dump(), indent=2, default=str))

    await valkey.close()


if __name__ == "__main__":
    asyncio.run(main())