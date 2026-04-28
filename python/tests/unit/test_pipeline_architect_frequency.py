from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.agents.pipeline_architect import PipelineArchitectAgent


def _profile_payload() -> dict:
    return {
        "n_observations": 120,
        "variance": 1.25,
        "narrative": "profile narrative",
        "stationarity": {
            "conclusion": "stationary",
            "is_stationary": True,
            "adf_pvalue": 0.01,
            "kpss_pvalue": 0.2,
        },
        "seasonality": {
            "period": 12,
            "seasonality_class": "strong",
            "strength": 0.7,
            "confidence": "high",
        },
        "structural_break": {
            "break_detected": False,
            "confidence": 0.9,
            "location": None,
            "location_fraction": 0.0,
        },
        "complexity_budget": {
            "permitted_models": [],
            "forbidden_models": [],
        },
    }


@pytest.mark.asyncio
async def test_construct_pipeline_uses_explicit_frequency_hint() -> None:
    valkey = AsyncMock()
    valkey.setex = AsyncMock(return_value=True)

    mcp = MagicMock()
    mcp.profile_dataset = MagicMock(return_value=_profile_payload())
    mcp.get_dataset_history = MagicMock(return_value={"status": "cold_start"})

    agent = PipelineArchitectAgent(valkey, mcp, settings=object())
    await agent.construct_pipeline("sales_weekly", frequency_hint="W")

    mcp.profile_dataset.assert_called_once_with("sales_weekly", freq="W")


@pytest.mark.asyncio
async def test_construct_pipeline_ignores_unknown_frequency_hint() -> None:
    valkey = AsyncMock()
    valkey.setex = AsyncMock(return_value=True)

    mcp = MagicMock()
    mcp.profile_dataset = MagicMock(return_value=_profile_payload())
    mcp.get_dataset_history = MagicMock(return_value={"status": "cold_start"})

    agent = PipelineArchitectAgent(valkey, mcp, settings=object())
    await agent.construct_pipeline("sales_unknown", frequency_hint="unknown")

    mcp.profile_dataset.assert_called_once_with("sales_unknown", freq=None)


@pytest.mark.asyncio
async def test_construct_pipeline_allows_raw_frequency_text_from_metadata() -> None:
    valkey = AsyncMock()
    valkey.setex = AsyncMock(return_value=True)

    mcp = MagicMock()
    mcp.profile_dataset = MagicMock(return_value=_profile_payload())
    mcp.get_dataset_history = MagicMock(return_value={"status": "cold_start"})

    agent = PipelineArchitectAgent(valkey, mcp, settings=object())
    await agent.construct_pipeline("crypto_ticks", frequency_hint="5m")

    mcp.profile_dataset.assert_called_once_with("crypto_ticks", freq="5m")