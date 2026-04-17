from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from app.schemas import DataProfile

logger = logging.getLogger(__name__)

_PROFILE_KEY = "profile:{dataset_id}"
_PROFILE_TTL = 3600  # 1 hour — long enough for a full training run


class PipelineArchitectAgent:
    """
    Parameters
    ----------
    valkey     : async Valkey/Redis client (redis.asyncio compatible)
    mcp_client : app.mcp.client.MCPClient
    settings   : app.config.Settings
    """

    def __init__(self, valkey, mcp_client, settings):
        self.valkey   = valkey
        self.mcp      = mcp_client
        self.settings = settings

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    async def construct_pipeline(self, dataset_id: str) -> DataProfile:
        """
        Build and cache the full DataProfile for *dataset_id*.

        All synchronous MCP calls are dispatched via run_in_executor so they
        do not block the asyncio event loop.

        Returns
        -------
        DataProfile
            Ready to pass directly to ModelSelectorAgent.
        """
        logger.info("PipelineArchitectAgent.construct_pipeline: starting for %s", dataset_id)
        loop = asyncio.get_running_loop()

        # ---- 1. Full diagnostic profile (single synchronous MCP call, off-thread) ----
        freq = self._infer_freq(dataset_id)
        profile_raw: dict[str, Any] = await loop.run_in_executor(
            None, lambda: self.mcp.profile_dataset(dataset_id, freq=freq)
        )
        logger.debug(
            "PipelineArchitectAgent: profile_dataset for %s: %s",
            dataset_id, profile_raw.get("narrative"),
        )

        # ---- 2. Production history (off-thread) ----
        history: dict[str, Any] = await loop.run_in_executor(
            None, lambda: self.mcp.get_dataset_history(dataset_id)
        )
        logger.debug(
            "PipelineArchitectAgent: history status=%s for %s",
            history.get("status"), dataset_id,
        )

        # ---- 3. Targeted drill-down when profile signals ambiguity ----
        stationarity     = profile_raw["stationarity"]
        structural_break = profile_raw["structural_break"]

        ambiguous_stationarity = stationarity["conclusion"] in (
            "trend_stationary", "difference_stationary"
        )
        borderline_break = (
            structural_break["break_detected"]
            and structural_break.get("confidence", 0.0) < 0.3
        )

        if ambiguous_stationarity and borderline_break:
            logger.info(
                "PipelineArchitectAgent: running targeted structural-break drill-down for %s",
                dataset_id,
            )
            refined_break: dict[str, Any] = await loop.run_in_executor(
                None, lambda: self.mcp.check_structural_break(dataset_id)
            )
            # Only override if the refined result disagrees with the profile result
            if refined_break["break_detected"] != structural_break["break_detected"]:
                profile_raw["structural_break"] = refined_break

        # ---- 4. Training-cost estimates for all permitted models (off-thread) ----
        complexity_budget  = profile_raw["complexity_budget"]
        permitted_models   = complexity_budget.get("permitted_models", [])
        seasonality_period = profile_raw["seasonality"].get("period") or 1
        training_costs     = await self._estimate_costs(
            loop, dataset_id, permitted_models, seasonality_period
        )

        # ---- 5. Assemble DataProfile ----
        profile = DataProfile(
            dataset_id        = dataset_id,
            n_observations    = profile_raw["n_observations"],
            variance          = profile_raw.get("variance", 0.0),
            narrative         = profile_raw["narrative"],
            stationarity      = profile_raw["stationarity"],
            seasonality       = profile_raw["seasonality"],
            structural_break  = profile_raw["structural_break"],
            complexity_budget = profile_raw["complexity_budget"],
            dataset_history   = history,
            training_costs    = training_costs,
        )

        # ---- 6. Persist to Valkey ----
        key = _PROFILE_KEY.format(dataset_id=dataset_id)
        await self.valkey.setex(key, _PROFILE_TTL, profile.model_dump_json())
        logger.info(
            "PipelineArchitectAgent: DataProfile cached at key=%s (TTL=%ds)",
            key, _PROFILE_TTL,
        )

        return profile

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _infer_freq(self, dataset_id: str) -> str | None:
        """
        Infer sampling frequency from the dataset_id naming convention.

        Examples
        --------
        "sales_weekly_store42" → "W"
        "iot_hourly_sensor7"   → "H"
        anything else          → None  (let MCP auto-detect)
        """
        lowered = dataset_id.lower()
        freq_hints = {
            "hourly":  "H",
            "daily":   "D",
            "weekly":  "W",
            "monthly": "M",
        }
        for hint, freq in freq_hints.items():
            if hint in lowered:
                return freq
        return None

    async def _estimate_costs(
        self,
        loop: asyncio.AbstractEventLoop,
        dataset_id: str,
        permitted_models: list[str],
        seasonality_period: int,
    ) -> dict[str, dict[str, Any]]:
        """
        Call estimate_training_cost for every permitted model, concurrently,
        each in its own executor thread so none block the loop.

        Returns a dict keyed by model class name, e.g.::

            {
                "AutoARIMA": {"estimated_minutes": 1.2, "estimated_cost_usd": 0.00060, …},
                "Prophet":   {"estimated_minutes": 0.4, "estimated_cost_usd": 0.00020, …},
            }

        Failures are logged and skipped gracefully.
        """
        async def _one(model_class: str) -> tuple[str, dict[str, Any] | None]:
            try:
                result = await loop.run_in_executor(
                    None,
                    lambda: self.mcp.estimate_training_cost(
                        dataset_id=dataset_id,
                        model_class=model_class,
                        seasonality_period=seasonality_period,
                    ),
                )
                return model_class, result
            except Exception as exc:
                logger.warning(
                    "PipelineArchitectAgent: cost estimation failed for %s / %s: %s",
                    dataset_id, model_class, exc,
                )
                return model_class, None

        results = await asyncio.gather(*[_one(m) for m in permitted_models])
        return {name: cost for name, cost in results if cost is not None}