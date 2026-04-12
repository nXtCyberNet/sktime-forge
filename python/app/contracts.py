from __future__ import annotations

from typing import Any, Protocol


class AgentMemoryProtocol(Protocol):
    async def get_dataset_memory(self, dataset_id: str) -> dict[str, Any]:
        ...

    async def update_dataset_memory(self, dataset_id: str, updates: dict[str, Any]) -> None:
        ...

    async def record_model_promotion(
        self,
        dataset_id: str,
        estimator: str,
        version: str,
        val_mae: float,
    ) -> None:
        ...

    async def record_model_failure(
        self,
        dataset_id: str,
        estimator: str,
        reason: str,
    ) -> None:
        ...

    async def record_drift_event(
        self,
        dataset_id: str,
        method: str,
        level: str,
        score: float,
    ) -> None:
        ...


class WatchdogProtocol(Protocol):
    async def monitor_post_promotion(
        self,
        dataset_id: str,
        baseline_score: float,
        model_version: str | None = None,
    ) -> None:
        ...

    async def record_residual(
        self,
        dataset_id: str,
        model_version: str,
        predicted: float,
        actual: float,
    ) -> None:
        ...
