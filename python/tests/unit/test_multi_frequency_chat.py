from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app import main
from app.schemas import (
    ChatRequest,
    ForecastRequest,
    ForecastResponse,
    MultiFrequencyForecastResponse,
)


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


def _build_forecast(job: ForecastRequest) -> ForecastResponse:
    return ForecastResponse(
        dataset_id=job.dataset_id,
        predictions=[1.0, 2.0, 3.0],
        model_version="1",
        model_class="NaiveForecaster",
        model_status="ok",
        cache_hit=False,
        correlation_id=job.correlation_id,
    )


@pytest.mark.asyncio
async def test_chat_returns_multi_frequency_payload_for_generic_query() -> None:
    chat_router = SimpleNamespace(
        route_request=AsyncMock(
            return_value=[
                ForecastRequest(
                    dataset_id="sales_weekly",
                    fh=[1, 2, 3],
                    correlation_id="c-generic-1",
                    frequency="W",
                ),
                ForecastRequest(
                    dataset_id="sales_monthly",
                    fh=[1, 2, 3],
                    correlation_id="c-generic-2",
                    frequency="M",
                ),
            ]
        )
    )

    async def fake_handle_job(job: ForecastRequest) -> ForecastResponse:
        return _build_forecast(job)

    orchestrator = SimpleNamespace(handle_job=AsyncMock(side_effect=fake_handle_job))

    registry = SimpleNamespace(get_all_records=AsyncMock(return_value=_records()))

    main.app.state.chat_router = chat_router
    main.app.state.orchestrator = orchestrator
    main.app.state.data_registry = registry

    response = await main.chat_interaction(ChatRequest(query="forecast sales for next 3 steps"))

    assert isinstance(response, MultiFrequencyForecastResponse)
    assert len(response.forecasts) == 2
    assert orchestrator.handle_job.await_count == 2

    called_dataset_ids = {
        call.args[0].dataset_id
        for call in orchestrator.handle_job.await_args_list
    }
    assert called_dataset_ids == {"sales_weekly", "sales_monthly"}


@pytest.mark.asyncio
async def test_chat_returns_single_forecast_when_query_is_frequency_specific() -> None:
    chat_router = SimpleNamespace(
        route_request=AsyncMock(
            return_value=[
                ForecastRequest(
                    dataset_id="sales_weekly",
                    fh=[1, 2, 3],
                    correlation_id="c-weekly",
                    frequency="W",
                )
            ]
        )
    )

    async def fake_handle_job(job: ForecastRequest) -> ForecastResponse:
        return _build_forecast(job)

    orchestrator = SimpleNamespace(handle_job=AsyncMock(side_effect=fake_handle_job))

    registry = SimpleNamespace(get_all_records=AsyncMock(return_value=_records()))

    main.app.state.chat_router = chat_router
    main.app.state.orchestrator = orchestrator
    main.app.state.data_registry = registry

    response = await main.chat_interaction(ChatRequest(query="weekly forecast for sales"))

    assert isinstance(response, ForecastResponse)
    assert response.dataset_id == "sales_weekly"
    assert orchestrator.handle_job.await_count == 1