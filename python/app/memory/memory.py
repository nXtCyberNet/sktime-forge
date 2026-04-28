from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

_MEMORY_KEY = "memory:{dataset_id}"
_MEMORY_TTL = 7 * 86_400   # 7 days — survives weekend gaps in batch pipelines

# Maximum number of model_history and drift_events entries to keep.
# Older entries are trimmed on every write to bound the key size.
_MAX_MODEL_HISTORY  = 20
_MAX_DRIFT_EVENTS   = 50


class AgentMemory:
    """
    Parameters
    ----------
    valkey : async Valkey/Redis client (redis.asyncio compatible)
    """

    def __init__(self, valkey):
        self.valkey = valkey

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    async def get_dataset_memory(self, dataset_id: str) -> dict[str, Any]:
        """
        Load the full memory dict for *dataset_id*.

        Returns an empty (but fully-structured) dict on cache miss or parse
        error so callers never have to guard against None.
        """
        key = _MEMORY_KEY.format(dataset_id=dataset_id)
        try:
            raw = await self.valkey.get(key)
            if not raw:
                return _empty_memory()
            data = json.loads(raw)
            if not isinstance(data, dict):
                return _empty_memory()
            # Back-fill any missing keys so callers can rely on structure
            return _backfill(data)
        except Exception as exc:
            logger.warning(
                "AgentMemory.get_dataset_memory: read failed for %s: %s",
                dataset_id, exc,
            )
            return _empty_memory()

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    async def update_dataset_memory(
        self,
        dataset_id: str,
        updates: dict[str, Any],
    ) -> None:
        """
        Merge *updates* into the existing memory for *dataset_id* and persist.

        Recognised top-level update keys
        ---------------------------------
        append_model_history  : dict — appended to model_history list (trimmed)
        append_drift_event    : dict — appended to drift_events list (trimmed)
        data_characteristics  : dict — merged (shallow) into data_characteristics

        Any unrecognised keys are merged directly into the top-level dict so
        this method stays forward-compatible.
        """
        current = await self.get_dataset_memory(dataset_id)

        # ---- Append model history entry ----
        model_entry: dict | None = updates.pop("append_model_history", None)
        if model_entry is not None:
            if "promoted_at" not in model_entry:
                model_entry["promoted_at"] = datetime.now(tz=timezone.utc).isoformat()
            current["model_history"].append(model_entry)
            current["model_history"] = current["model_history"][-_MAX_MODEL_HISTORY:]

        # ---- Append drift event ----
        drift_entry: dict | None = updates.pop("append_drift_event", None)
        if drift_entry is not None:
            if "triggered_at" not in drift_entry:
                drift_entry["triggered_at"] = datetime.now(tz=timezone.utc).isoformat()
            current["drift_events"].append(drift_entry)
            current["drift_events"] = current["drift_events"][-_MAX_DRIFT_EVENTS:]

        # ---- Shallow-merge data_characteristics ----
        chars_update: dict | None = updates.pop("data_characteristics", None)
        if chars_update is not None:
            current["data_characteristics"].update(chars_update)

        # ---- Merge any remaining keys directly ----
        current.update(updates)

        # ---- Persist ----
        key = _MEMORY_KEY.format(dataset_id=dataset_id)
        try:
            await self.valkey.setex(key, _MEMORY_TTL, json.dumps(current))
        except Exception as exc:
            logger.error(
                "AgentMemory.update_dataset_memory: write failed for %s: %s",
                dataset_id, exc,
            )

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    async def record_model_promotion(
        self,
        dataset_id: str,
        estimator: str,
        version: str,
        val_mae: float,
    ) -> None:
        """Shorthand to record a successful model promotion."""
        await self.update_dataset_memory(
            dataset_id,
            {
                "append_model_history": {
                    "estimator":      estimator,
                    "version":        version,
                    "val_mae":        val_mae,
                    "failure_reason": None,
                }
            },
        )

    async def record_model_failure(
        self,
        dataset_id: str,
        estimator: str,
        reason: str,
    ) -> None:
        """Shorthand to record a model fit/predict failure."""
        await self.update_dataset_memory(
            dataset_id,
            {
                "append_model_history": {
                    "estimator":      estimator,
                    "version":        None,
                    "val_mae":        None,
                    "failure_reason": reason,
                }
            },
        )

    async def record_drift_event(
        self,
        dataset_id: str,
        method: str,
        level: str,
        score: float,
    ) -> None:
        """Shorthand to record a drift event."""
        await self.update_dataset_memory(
            dataset_id,
            {
                "append_drift_event": {
                    "method": method,
                    "level":  level,
                    "score":  round(score, 4),
                }
            },
        )

def _empty_memory() -> dict[str, Any]:
    return {
        "model_history":      [],
        "drift_events":       [],
        "data_characteristics": {},
    }


def _backfill(data: dict[str, Any]) -> dict[str, Any]:
    """Ensure all expected top-level keys exist (forward/backward compat)."""
    defaults = _empty_memory()
    for key, default in defaults.items():
        data.setdefault(key, default)
    return data