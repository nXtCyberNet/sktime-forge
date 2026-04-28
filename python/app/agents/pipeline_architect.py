from __future__ import annotations

import asyncio
import logging
from typing import Any

from app.schemas import DataProfile

logger = logging.getLogger(__name__)

_PROFILE_KEY = "profile:{dataset_id}"
_PROFILE_TTL = 3600  # 1 hour — long enough for a full training run


class PipelineArchitectAgent:
    def __init__(self, valkey, mcp_client, settings):
        self.valkey   = valkey
        self.mcp      = mcp_client
        self.settings = settings


    async def construct_pipeline(
        self,
        dataset_id: str,
        frequency_hint: str | None = None,
    ) -> DataProfile:

        logger.info("PipelineArchitectAgent.construct_pipeline: starting for %s", dataset_id)
        loop = asyncio.get_running_loop()

        freq = self._clean_frequency_hint(frequency_hint)
        profile_raw: dict[str, Any] = await loop.run_in_executor(
            None, lambda: self.mcp.profile_dataset(dataset_id, freq=freq)
        )
        logger.debug(
            "PipelineArchitectAgent: profile_dataset for %s: %s",
            dataset_id, profile_raw.get("narrative"),
        )

        history: dict[str, Any] = await loop.run_in_executor(
            None, lambda: self.mcp.get_dataset_history(dataset_id)
        )
        logger.debug(
            "PipelineArchitectAgent: history status=%s for %s",
            history.get("status"), dataset_id,
        )

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
            
            if refined_break["break_detected"] != structural_break["break_detected"]:
                profile_raw["structural_break"] = refined_break

        
        complexity_budget  = profile_raw["complexity_budget"]
        permitted_models   = complexity_budget.get("permitted_models", [])
        seasonality_period = profile_raw["seasonality"].get("period") or 1
        training_costs     = await self._estimate_costs(
            loop, dataset_id, permitted_models, seasonality_period
        )

        
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

        
        key = _PROFILE_KEY.format(dataset_id=dataset_id)
        await self.valkey.setex(key, _PROFILE_TTL, profile.model_dump_json())
        logger.info(
            "PipelineArchitectAgent: DataProfile cached at key=%s (TTL=%ds)",
            key, _PROFILE_TTL,
        )

        return profile

    @staticmethod
    def _clean_frequency_hint(value: str | None) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        if text.lower() == "unknown":
            return None
        return text

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