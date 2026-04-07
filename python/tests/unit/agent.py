"""
Unit tests for the sktime Agentic Python layer.

Run with:
    pytest tests/ -v

These tests use only in-process mocks — no Valkey, no MLflow, no S3 needed.
"""
from __future__ import annotations

import asyncio
import json
from collections import deque
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def settings():
    from app.core.settings import Settings
    return Settings(
        valkey_url="redis://localhost:6379/0",
        llm_api_key="",          # no LLM in tests
        min_history_length=10,
        drift_check_every_n=5,
        drift_check_every_t_minutes=60,
        no_drift_threshold=0.2,
        minor_drift_threshold=0.5,
        major_drift_threshold=0.5,
        retrain_lock_ttl_seconds=30,
    )


@pytest.fixture
def short_series() -> pd.Series:
    idx = pd.date_range("2020-01-01", periods=5, freq="D")
    return pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=idx, name="y")


@pytest.fixture
def long_series() -> pd.Series:
    rng = np.random.default_rng(42)
    idx = pd.date_range("2020-01-01", periods=200, freq="D")
    values = np.linspace(0, 10, 200) + rng.normal(0, 0.5, 200)
    return pd.Series(values, index=idx, name="y")


@pytest.fixture
def mock_valkey():
    v = AsyncMock()
    v.get = AsyncMock(return_value=None)
    v.set = AsyncMock(return_value=True)
    v.setex = AsyncMock(return_value=True)
    v.exists = AsyncMock(return_value=False)
    v.delete = AsyncMock(return_value=1)
    v.xadd = AsyncMock(return_value=b"1-0")
    v.xreadgroup = AsyncMock(return_value=[])
    v.xack = AsyncMock(return_value=1)
    v.ping = AsyncMock(return_value=True)
    v.ttl = AsyncMock(return_value=100)
    v.scan_iter = MagicMock(return_value=_async_iter([]))
    return v


@pytest.fixture
def mock_mlflow_client():
    client = MagicMock()
    client.get_latest_versions = MagicMock(return_value=[])
    client.transition_model_version_stage = MagicMock()
    client.search_model_versions = MagicMock(return_value=[])
    return client


def _async_iter(items):
    """Helper to create an async iterator from a list."""
    async def _inner():
        for item in items:
            yield item
    return _inner()


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class TestSchemas:
    def test_forecast_request_valid(self):
        from app.schemas import ForecastRequest
        req = ForecastRequest(
            dataset_id="ds-1",
            fh=[1, 2, 3],
            correlation_id="abc-123",
            frequency="D",
        )
        assert req.fh == [1, 2, 3]

    def test_forecast_request_invalid_fh(self):
        from app.schemas import ForecastRequest
        import pydantic
        with pytest.raises(pydantic.ValidationError):
            ForecastRequest(dataset_id="ds-1", fh=[0, -1], correlation_id="x")

    def test_data_profile_natural_language(self):
        from app.schemas import DataProfile
        p = DataProfile(
            dataset_id="ds-1",
            length=100,
            frequency="D",
            has_seasonality=True,
            is_stationary=False,
            missing_rate=0.05,
            variance=1.23,
        )
        nl = p.to_natural_language()
        assert "100 observations" in nl
        assert "seasonality=yes" in nl
        assert "stationary=no" in nl


# ---------------------------------------------------------------------------
# ModelSelectorAgent — rule-based path
# ---------------------------------------------------------------------------

class TestRuleBasedSelection:
    def setup_method(self):
        # Patch ALLOWED_ESTIMATORS so tests don't need sktime installed
        import app.agents.model_selector as ms
        ms.ALLOWED_ESTIMATORS.clear()
        ms.ALLOWED_ESTIMATORS.extend([
            "AutoARIMA", "Prophet", "NaiveForecaster",
            "ExponentialSmoothing", "ThetaForecaster", "BATS", "TBATS",
        ])

    def test_short_series_prefers_arima(self):
        from app.schemas import DataProfile
        from app.agents.model_selector import _rule_based_select
        profile = DataProfile(
            dataset_id="x", length=50, frequency="D",
            has_seasonality=False, is_stationary=True,
            missing_rate=0.0, variance=1.0,
        )
        result = _rule_based_select(profile)
        assert result is not None
        assert result[0] == "AutoARIMA"

    def test_unknown_frequency_returns_naive(self):
        from app.schemas import DataProfile
        from app.agents.model_selector import _rule_based_select
        profile = DataProfile(
            dataset_id="x", length=200, frequency=None,
            has_seasonality=False, is_stationary=True,
            missing_rate=0.0, variance=1.0,
        )
        result = _rule_based_select(profile)
        assert result == ["NaiveForecaster"]

    def test_long_seasonal_prefers_prophet(self):
        from app.schemas import DataProfile
        from app.agents.model_selector import _rule_based_select
        profile = DataProfile(
            dataset_id="x", length=300, frequency="D",
            has_seasonality=True, is_stationary=False,
            missing_rate=0.0, variance=1.0,
        )
        result = _rule_based_select(profile)
        assert result is not None
        assert "Prophet" in result

    def test_ambiguous_returns_none(self):
        from app.schemas import DataProfile
        from app.agents.model_selector import _rule_based_select
        profile = DataProfile(
            dataset_id="x", length=150, frequency="D",
            has_seasonality=False, is_stationary=True,
            missing_rate=0.0, variance=1.0,
        )
        # 150 obs, no seasonality, stationary → ambiguous → None
        result = _rule_based_select(profile)
        assert result is None


# ---------------------------------------------------------------------------
# ModelSelectorAgent — LLM validation whitelist
# ---------------------------------------------------------------------------

class TestLLMWhitelistValidation:
    def test_llm_result_filtered_against_whitelist(self, mock_valkey, mock_mlflow_client, settings):
        import app.agents.model_selector as ms
        ms.ALLOWED_ESTIMATORS.clear()
        ms.ALLOWED_ESTIMATORS.extend(["AutoARIMA", "NaiveForecaster"])

        agent = ms.ModelSelectorAgent(mock_valkey, mock_mlflow_client, settings)

        # Simulate LLM returning something valid + something hallucinated
        fake_response_text = json.dumps({"ranked": ["AutoARIMA", "HALLUCINATED_MODEL", "NaiveForecaster"]})

        mock_msg = MagicMock()
        mock_msg.content = [MagicMock(text=fake_response_text)]
        mock_client = MagicMock()
        mock_client.messages.create = MagicMock(return_value=mock_msg)
        agent._llm_client = mock_client
        agent.settings.llm_provider = "anthropic"

        from app.schemas import DataProfile
        profile = DataProfile(
            dataset_id="x", length=200, frequency="D",
            has_seasonality=False, is_stationary=True,
            missing_rate=0.0, variance=1.0,
        )
        result = agent._llm_select(profile)
        assert "HALLUCINATED_MODEL" not in result
        assert "AutoARIMA" in result
        assert "NaiveForecaster" in result

    def test_llm_failure_raises_llm_selection_error(self, mock_valkey, mock_mlflow_client, settings):
        import app.agents.model_selector as ms
        from app.core.exceptions import LLMSelectionError

        agent = ms.ModelSelectorAgent(mock_valkey, mock_mlflow_client, settings)
        mock_client = MagicMock()
        mock_client.messages.create = MagicMock(side_effect=Exception("network error"))
        agent._llm_client = mock_client
        agent.settings.llm_provider = "anthropic"

        from app.schemas import DataProfile
        profile = DataProfile(
            dataset_id="x", length=200, frequency="D",
            has_seasonality=False, is_stationary=True,
            missing_rate=0.0, variance=1.0,
        )
        with pytest.raises(LLMSelectionError):
            agent._llm_select(profile)


# ---------------------------------------------------------------------------
# PredictionAgent
# ---------------------------------------------------------------------------

class TestPredictionAgent:
    @pytest.mark.asyncio
    async def test_insufficient_history_raises(self, mock_valkey, mock_mlflow_client, settings, short_series):
        from app.agents.prediction_agent import PredictionAgent
        from app.core.exceptions import InsufficientHistoryError
        from app.schemas import ForecastRequest

        agent = PredictionAgent(mock_valkey, mock_mlflow_client, settings)
        job = ForecastRequest(dataset_id="ds-1", fh=[1, 2, 3], correlation_id="x")

        with pytest.raises(InsufficientHistoryError):
            await agent.predict(job, "1", {}, short_series)

    @pytest.mark.asyncio
    async def test_valkey_cache_hit_returns_immediately(self, mock_valkey, mock_mlflow_client, settings, long_series):
        from app.agents.prediction_agent import PredictionAgent
        from app.schemas import ForecastRequest

        cached_payload = json.dumps({
            "predictions": [1.0, 2.0, 3.0],
            "prediction_intervals": None,
            "model_class": "AutoARIMA",
        })
        mock_valkey.get = AsyncMock(return_value=cached_payload)

        agent = PredictionAgent(mock_valkey, mock_mlflow_client, settings)
        job = ForecastRequest(dataset_id="ds-1", fh=[1, 2, 3], correlation_id="corr-1")
        result = await agent.predict(job, "v3", {}, long_series)

        assert result.cache_hit is True
        assert result.predictions == [1.0, 2.0, 3.0]
        assert result.model_class == "AutoARIMA"

    @pytest.mark.asyncio
    async def test_cache_key_includes_model_version(self, mock_valkey, mock_mlflow_client, settings):
        from app.agents.prediction_agent import PredictionAgent, _fh_hash
        agent = PredictionAgent(mock_valkey, mock_mlflow_client, settings)
        key = agent._cache_key("v5", "sensor-42", [1, 2, 3])
        assert "v5" in key
        assert "sensor-42" in key

    @pytest.mark.asyncio
    async def test_different_model_versions_produce_different_cache_keys(self, mock_valkey, mock_mlflow_client, settings):
        from app.agents.prediction_agent import PredictionAgent
        agent = PredictionAgent(mock_valkey, mock_mlflow_client, settings)
        k1 = agent._cache_key("v1", "ds", [1])
        k2 = agent._cache_key("v2", "ds", [1])
        assert k1 != k2


# ---------------------------------------------------------------------------
# DriftMonitor
# ---------------------------------------------------------------------------

class TestDriftMonitor:
    @pytest.mark.asyncio
    async def test_no_retrain_when_no_drift(self, mock_valkey, settings):
        from app.agents.drift_monitor import DriftMonitor
        from app.schemas import ForecastRequest, ForecastResponse

        monitor = DriftMonitor(mock_valkey, settings)
        # Feed stable residuals
        for _ in range(100):
            monitor._residuals.setdefault("ds-1", deque(maxlen=100))
            monitor._residuals["ds-1"].append(0.01)  # negligible residual

        monitor._prediction_counts["ds-1"] = 50
        monitor._last_check_times["ds-1"] = datetime(2020, 1, 1, tzinfo=timezone.utc)

        await monitor._run_detection("ds-1", "v1")
        mock_valkey.xadd.assert_not_called()

    @pytest.mark.asyncio
    async def test_major_drift_triggers_retrain(self, mock_valkey, settings):
        from app.agents.drift_monitor import DriftMonitor

        mock_valkey.exists = AsyncMock(return_value=False)
        monitor = DriftMonitor(mock_valkey, settings)

        # Feed diverging residuals to force major drift
        residuals = deque(maxlen=100)
        for i in range(50):
            residuals.append(float(i * 2))    # rapidly increasing = major drift
        monitor._residuals["ds-1"] = residuals
        monitor._prediction_counts["ds-1"] = 50
        monitor._last_check_times["ds-1"] = datetime(2020, 1, 1, tzinfo=timezone.utc)

        await monitor._run_detection("ds-1", "v1")
        mock_valkey.xadd.assert_called_once()
        call_args = mock_valkey.xadd.call_args
        assert "retrain" in str(call_args)

    @pytest.mark.asyncio
    async def test_duplicate_retrain_deduplicated(self, mock_valkey, settings):
        from app.agents.drift_monitor import DriftMonitor

        # Simulate retrain_lock already set
        mock_valkey.exists = AsyncMock(return_value=True)
        monitor = DriftMonitor(mock_valkey, settings)

        await monitor._maybe_trigger_retrain("ds-1", "CUSUM_major", "full")
        mock_valkey.xadd.assert_not_called()

    @pytest.mark.asyncio
    async def test_check_increments_prediction_count(self, mock_valkey, settings):
        from app.agents.drift_monitor import DriftMonitor
        from app.schemas import ForecastRequest, ForecastResponse

        monitor = DriftMonitor(mock_valkey, settings)
        job = ForecastRequest(dataset_id="ds-2", fh=[1], correlation_id="c1")
        resp = ForecastResponse(
            dataset_id="ds-2",
            predictions=[1.0],
            model_version="1",
            model_class="NaiveForecaster",
            model_status="stable",
            cache_hit=False,
            correlation_id="c1",
        )
        await monitor.check(job, resp)
        assert monitor._prediction_counts["ds-2"] == 1


# ---------------------------------------------------------------------------
# Orchestrator — routing logic
# ---------------------------------------------------------------------------

class TestOrchestrator:
    @pytest.mark.asyncio
    async def test_orphaned_lock_cleaned_on_startup(self, mock_valkey, mock_mlflow_client, settings):
        from app.agents.orchestrator import Orchestrator

        # Simulate one orphaned lock (ttl = -1)
        orphaned_key = b"model_lock:ds-stale"
        mock_valkey.scan_iter = MagicMock(return_value=_async_iter([orphaned_key]))
        mock_valkey.ttl = AsyncMock(return_value=-1)
        mock_valkey.xgroup_create = AsyncMock(side_effect=Exception("BUSYGROUP"))

        orch = Orchestrator(
            valkey=mock_valkey,
            mlflow_client=mock_mlflow_client,
            settings=settings,
            data_loader=AsyncMock(),
        )
        await orch._cleanup_orphaned_locks()
        mock_valkey.delete.assert_called_with(orphaned_key)

    @pytest.mark.asyncio
    async def test_model_selector_called_when_no_production_model(self, mock_valkey, mock_mlflow_client, settings, long_series):
        from app.agents.orchestrator import Orchestrator
        from app.schemas import ForecastRequest

        mock_mlflow_client.get_latest_versions = MagicMock(return_value=[])

        orch = Orchestrator(
            valkey=mock_valkey,
            mlflow_client=mock_mlflow_client,
            settings=settings,
            data_loader=AsyncMock(return_value=long_series),
        )
        orch.model_selector.select = AsyncMock(return_value="1")
        orch.prediction_agent.predict = AsyncMock(return_value=MagicMock(
            dataset_id="ds-1",
            predictions=[1.0],
            model_version="1",
            model_class="NaiveForecaster",
            model_status="stable",
            cache_hit=False,
            correlation_id="c1",
            prediction_intervals=None,
            drift_score=None,
            drift_method=None,
            warning=None,
        ))

        job = ForecastRequest(dataset_id="ds-1", fh=[1], correlation_id="c1")
        await orch._dispatch(job)
        orch.model_selector.select.assert_called_once()

    @pytest.mark.asyncio
    async def test_model_selector_not_called_when_production_model_exists(self, mock_valkey, mock_mlflow_client, settings, long_series):
        from app.agents.orchestrator import Orchestrator
        from app.schemas import ForecastRequest

        version_mock = MagicMock()
        version_mock.version = "3"
        mock_mlflow_client.get_latest_versions = MagicMock(return_value=[version_mock])

        orch = Orchestrator(
            valkey=mock_valkey,
            mlflow_client=mock_mlflow_client,
            settings=settings,
            data_loader=AsyncMock(return_value=long_series),
        )
        orch.model_selector.select = AsyncMock()
        orch.prediction_agent.predict = AsyncMock(return_value=MagicMock(
            dataset_id="ds-1", predictions=[1.0], model_version="3",
            model_class="AutoARIMA", model_status="stable", cache_hit=False,
            correlation_id="c1", prediction_intervals=None,
            drift_score=None, drift_method=None, warning=None,
        ))

        job = ForecastRequest(dataset_id="ds-1", fh=[1], correlation_id="c1")
        await orch._dispatch(job)
        orch.model_selector.select.assert_not_called()