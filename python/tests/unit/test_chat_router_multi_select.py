from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.agents.chat_router import ChatRouterAgent


def _records() -> dict[str, dict[str, str]]:
    return {
        "sales_weekly": {
            "description": "Weekly sales",
            "frequency": "W",
        },
        "sales_monthly": {
            "description": "Monthly sales",
            "frequency": "M",
        },
    }


@pytest.mark.asyncio
async def test_route_request_returns_multiple_requests_from_dataset_ids() -> None:
    agent = ChatRouterAgent(settings=SimpleNamespace())
    agent._request_llm_text = AsyncMock(
        return_value='{"dataset_ids": ["sales_weekly", "sales_monthly"], "fh": [1, 2, 3]}'
    )

    requests = await agent.route_request("forecast sales", _records())

    assert len(requests) == 2
    assert requests[0].dataset_id == "sales_weekly"
    assert requests[0].frequency == "W"
    assert requests[1].dataset_id == "sales_monthly"
    assert requests[1].frequency == "M"


@pytest.mark.asyncio
async def test_route_request_supports_legacy_single_dataset_id_key() -> None:
    agent = ChatRouterAgent(settings=SimpleNamespace())
    agent._request_llm_text = AsyncMock(
        return_value='{"dataset_id": "sales_weekly", "fh": [1]}'
    )

    requests = await agent.route_request("weekly forecast", _records())

    assert len(requests) == 1
    assert requests[0].dataset_id == "sales_weekly"
    assert requests[0].frequency == "W"