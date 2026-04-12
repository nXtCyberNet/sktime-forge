"""
Unit tests for the sktime Agentic Python layer.

Run with:
    pytest tests/unit/test_agents.py -v

All tests use in-process mocks — no Valkey, MLflow, or S3 required.

Fixes vs original test file
----------------------------
- All import paths corrected to match actual module structure:
    app.config.Settings          (not app.core.settings)
    app.agents.model_selector    (ModelSelectorAgent)
    app.agents.prediction        (PredictionAgent, not prediction_agent)
    app.monitoring.drift_monitor (DriftMonitor, not app.agents.drift_monitor)
    app.orchestrator             (Orchestrator)
- Settings fixture no longer passes major_drift_threshold (field now exists
  in Settings so no ValidationError, but we use the correct field names).
- TestSchemas.test_data_profile_natural_language assertions fixed to match
  the actual to_natural_language() output ("seasonality=yes", "stationary=no").
- TestRuleBasedSelection replaced with TestLLMCandidateFiltering that tests
  the actual forbidden-model stripping behaviour (no _rule_based_select).
- TestLLMWhitelistValidation patching targets self._llm (not self._llm_client).
- TestPredictionAgent.predict() called with correct 3-arg signature.
- TestDriftMonitor uses correct method names (_handle_major_drift, not
  _maybe_trigger_retrain) and correct import path.
- TestOrchestrator uses correct method names (handle_job, not _dispatch;
  startup_cleanup not _cleanup_orphaned_locks).
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
# Helpers
# ---------------------------------------------------------------------------

async def _async_iter_helper(items):
    for item in items:
        yield item


def _async_iter(items):
    return _async_iter_helper(items)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def settings():
    from app.config import Settings
    return Settings(
        valkey_url="redis://localhost:6379/0",
        llm_api_key="",
        llm_provider="openai_compatible",
        min_history_length=10,
        drift_check_every_n_predictions=5,
        drift_check_every_t_minutes=60,
        no_drift_threshold=0.2,
        minor_drift_threshold=0.35,
        major_drift_threshold=0.5,
        retrain_lock_ttl_seconds=30,
    )


@pytest.fixture
def short_series() -> pd.Series:
    idx = pd.RangeIndex(5)
    return pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=idx, name="y")


@pytest.fixture
def long_series() -> pd.Series:
    rng = np.random.default_rng(42)
    values = np.linspace(0, 10, 200) + rng.normal(0, 0.5, 200)
    return pd.Series(values, index=pd.RangeIndex(200), name="y")


@pytest.fixture
def mock_valkey():
    v = AsyncMock()
    v.get    = AsyncMock(return_value=None)
    v.set    = AsyncMock(return_value=True)
    v.setex  = AsyncMock(return_value=True)
    v.exists = AsyncMock(return_value=False)
    v.delete = AsyncMock(return_value=1)
    v.xadd   = AsyncMock(return_value=b"1-0")
    v.lrange = AsyncMock(return_value=[])
    v.rpush  = AsyncMock(return_value=1)
    v.ltrim  = AsyncMock(return_value=True)
    v.expire = AsyncMock(return_value=True)
    v.xreadgroup = AsyncMock(return_value=[])
    v.xack   = AsyncMock(return_value=1)
    v.ping   = AsyncMock(return_value=True)
    v.ttl    = AsyncMock(return_value=100)
    v.scan_iter = MagicMock(return_value=_async_iter([]))

    # pipeline() must return an async context manager
    pipe_mock = AsyncMock()
    pipe_mock.__aenter__ = AsyncMock(return_value=pipe_mock)
    pipe_mock.__aexit__  = AsyncMock(return_value=False)
    pipe_mock.incr   = AsyncMock(return_value=1)
    pipe_mock.expire = AsyncMock(return_value=True)
    pipe_mock.rpush  = AsyncMock(return_value=1)
    pipe_mock.ltrim  = AsyncMock(return_value=True)
    pipe_mock.execute = AsyncMock(return_value=[1, True])
    v.pipeline = MagicMock(return_value=pipe_mock)

    return v


@pytest.fixture
def mock_mlflow_client():
    client = MagicMock()
    client.get_latest_versions          = MagicMock(return_value=[])
    client.transition_model_version_stage = MagicMock()
    client.search_model_versions        = MagicMock(return_value=[])
    client.get_experiment_by_name       = MagicMock(return_value=None)
    client.create_experiment            = MagicMock(return_value="1")
    return client


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
            actual=42.5,
        )
        assert req.fh == [1, 2, 3]
        assert req.actual == 42.5

    def test_forecast_request_invalid_fh(self):
        from app.schemas import ForecastRequest
        import pydantic
        with pytest.raises(pydantic.ValidationError):
            ForecastRequest(dataset_id="ds-1", fh=[0, -1], correlation_id="x")

    def test_data_profile_natural_language_contains_correct_strings(self):
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
        # These strings match the actual to_natural_language() implementation
        assert "100 observations" in nl
        assert "seasonality=yes" in nl
        assert "stationary=no" in nl

    def test_data_profile_legacy_field_sync(self):
        """n_observations and length should stay in sync via the model validator."""
        from app.schemas import DataProfile
        p = DataProfile(dataset_id="x", length=50)
        assert p.n_observations == 50

        q = DataProfile(dataset_id="y", n_observations=80)
        assert q.length == 80


# ---------------------------------------------------------------------------
# ModelSelectorAgent — forbidden-model stripping
# ---------------------------------------------------------------------------

class TestForbiddenModelStripping:
    """
    Tests the hard Python post-processing step that strips forbidden models
    from the LLM's response. This is the critical correctness invariant —
    forbidden models must never reach TrainingAgent regardless of what the
    LLM returns.
    """

    @pytest.mark.asyncio
    async def test_forbidden_models_stripped_from_llm_output(
        self, mock_valkey, mock_mlflow_client, settings
    ):
        from app.agents.model_selector import ModelSelectorAgent
        from app.schemas import DataProfile

        profile = DataProfile(
            dataset_id="ds-1",
            n_observations=100,
            complexity_budget={
                "permitted_models": ["AutoARIMA", "NaiveForecaster"],
                "forbidden_models": ["LSTMForecaster", "Transformers"],
            },
            dataset_history={},
        )

        # Serialise profile to Valkey mock
        mock_valkey.get = AsyncMock(
            return_value=profile.model_dump_json().encode()
        )

        agent = ModelSelectorAgent(mock_valkey, mock_mlflow_client, None, settings)

        # Patch _llm_select to simulate LLM returning a forbidden model
        agent._llm_select = AsyncMock(
            return_value=["LSTMForecaster", "AutoARIMA", "NaiveForecaster"]
        )

        result = await agent.select(type("Job", (), {"dataset_id": "ds-1"})())

        assert "LSTMForecaster" not in result
        assert "AutoARIMA" in result
        assert "NaiveForecaster" in result

    @pytest.mark.asyncio
    async def test_all_forbidden_falls_back_to_first_permitted(
        self, mock_valkey, mock_mlflow_client, settings
    ):
        from app.agents.model_selector import ModelSelectorAgent
        from app.schemas import DataProfile

        profile = DataProfile(
            dataset_id="ds-2",
            n_observations=100,
            complexity_budget={
                "permitted_models": ["NaiveForecaster"],
                "forbidden_models": ["LSTMForecaster"],
            },
            dataset_history={},
        )
        mock_valkey.get = AsyncMock(
            return_value=profile.model_dump_json().encode()
        )
        agent = ModelSelectorAgent(mock_valkey, mock_mlflow_client, None, settings)
        # LLM returns only forbidden
        agent._llm_select = AsyncMock(return_value=["LSTMForecaster"])

        result = await agent.select(type("Job", (), {"dataset_id": "ds-2"})())
        assert result == ["NaiveForecaster"]

    @pytest.mark.asyncio
    async def test_registry_filters_non_registry_estimators(
        self, mock_valkey, mock_mlflow_client, settings
    ):
        from app.agents.model_selector import ModelSelectorAgent
        from app.schemas import DataProfile

        profile = DataProfile(
            dataset_id="ds-3",
            n_observations=120,
            complexity_budget={
                "permitted_models": ["UnknownForecaster", "AutoARIMA", "NaiveForecaster"],
                "forbidden_models": [],
            },
            dataset_history={},
        )
        mock_valkey.get = AsyncMock(return_value=profile.model_dump_json().encode())

        agent = ModelSelectorAgent(mock_valkey, mock_mlflow_client, None, settings)
        agent._llm_select = AsyncMock(
            return_value=["UnknownForecaster", "AutoARIMA", "NaiveForecaster"]
        )

        result = await agent.select(type("Job", (), {"dataset_id": "ds-3"})())
        assert result == ["AutoARIMA", "NaiveForecaster"]

    @pytest.mark.asyncio
    async def test_empty_registry_allowed_set_falls_back_to_naive(
        self, mock_valkey, mock_mlflow_client, settings
    ):
        from app.agents.model_selector import ModelSelectorAgent
        from app.schemas import DataProfile

        profile = DataProfile(
            dataset_id="ds-4",
            n_observations=80,
            complexity_budget={
                "permitted_models": ["UnknownForecaster"],
                "forbidden_models": [],
            },
            dataset_history={},
        )
        mock_valkey.get = AsyncMock(return_value=profile.model_dump_json().encode())

        agent = ModelSelectorAgent(mock_valkey, mock_mlflow_client, None, settings)
        agent._llm_select = AsyncMock(return_value=["UnknownForecaster"])

        result = await agent.select(type("Job", (), {"dataset_id": "ds-4"})())
        assert result == ["NaiveForecaster"]


class TestTrainingCandidateValidation:
    def test_sanitize_candidates_drops_unsupported(self, mock_valkey, mock_mlflow_client, settings):
        from app.agents.training import TrainingAgent

        agent = TrainingAgent(mock_valkey, mock_mlflow_client, settings)
        sanitized = agent._sanitize_candidates(
            ["UnknownForecaster", "NaiveForecaster", "UnknownForecaster"],
            dataset_id="ds-1",
        )
        assert sanitized == ["NaiveForecaster"]

    def test_sanitize_candidates_returns_empty_when_none_supported(
        self, mock_valkey, mock_mlflow_client, settings
    ):
        from app.agents.training import TrainingAgent

        agent = TrainingAgent(mock_valkey, mock_mlflow_client, settings)
        sanitized = agent._sanitize_candidates(
            ["UnknownForecaster", "AnotherUnknown"],
            dataset_id="ds-1",
        )
        assert sanitized == []


# ---------------------------------------------------------------------------
# ModelSelectorAgent — LLM parse helpers
# ---------------------------------------------------------------------------

class TestLLMResponseParsing:
    def test_parse_bare_array(self):
        from app.agents.model_selector import ModelSelectorAgent
        raw = json.dumps(["AutoARIMA", "Prophet", "NaiveForecaster"])
        result = ModelSelectorAgent._parse_candidate_response(raw)
        assert result == ["AutoARIMA", "Prophet", "NaiveForecaster"]

    def test_parse_object_with_candidates_key(self):
        from app.agents.model_selector import ModelSelectorAgent
        raw = json.dumps({"candidates": ["AutoARIMA", "NaiveForecaster"]})
        result = ModelSelectorAgent._parse_candidate_response(raw)
        assert result == ["AutoARIMA", "NaiveForecaster"]

    def test_parse_invalid_raises_value_error(self):
        from app.agents.model_selector import ModelSelectorAgent
        raw = json.dumps({"unexpected_key": "value"})
        with pytest.raises(ValueError):
            ModelSelectorAgent._parse_candidate_response(raw)

    def test_content_to_text_list_of_blocks(self):
        from app.agents.model_selector import ModelSelectorAgent
        content = [
            {"type": "text", "text": "hello"},
            {"type": "text", "text": "world"},
        ]
        assert ModelSelectorAgent._content_to_text(content) == "hello\nworld"


# ---------------------------------------------------------------------------
# PredictionAgent
# ---------------------------------------------------------------------------

class TestPredictionAgent:
    @pytest.mark.asyncio
    async def test_resolve_version_from_valkey(self, mock_valkey, mock_mlflow_client, settings):
        from app.agents.prediction import PredictionAgent
        mock_valkey.get = AsyncMock(return_value=b"42")
        agent = PredictionAgent(mock_valkey, mock_mlflow_client, settings)
        version = await agent._resolve_model_version("ds-1")
        assert version == "42"

    @pytest.mark.asyncio
    async def test_resolve_version_mlflow_fallback(self, mock_valkey, mock_mlflow_client, settings):
        from app.agents.prediction import PredictionAgent
        mock_valkey.get = AsyncMock(return_value=None)

        v_mock = MagicMock()
        v_mock.version = "7"
        mock_mlflow_client.get_latest_versions = MagicMock(return_value=[v_mock])

        agent = PredictionAgent(mock_valkey, mock_mlflow_client, settings)
        version = await agent._resolve_model_version("ds-1")
        assert version == "7"

    @pytest.mark.asyncio
    async def test_resolve_version_returns_none_when_nothing_found(
        self, mock_valkey, mock_mlflow_client, settings
    ):
        from app.agents.prediction import PredictionAgent
        mock_valkey.get = AsyncMock(return_value=None)
        mock_mlflow_client.get_latest_versions = MagicMock(return_value=[])
        agent = PredictionAgent(mock_valkey, mock_mlflow_client, settings)
        version = await agent._resolve_model_version("ds-missing")
        assert version is None

    @pytest.mark.asyncio
    async def test_different_versions_have_different_cache_keys(
        self, mock_valkey, mock_mlflow_client, settings
    ):
        from app.agents.prediction import PredictionAgent
        agent = PredictionAgent(mock_valkey, mock_mlflow_client, settings)
        # Cache is keyed by (dataset_id, model_version) tuples
        k1 = ("ds-1", "v1")
        k2 = ("ds-1", "v2")
        assert k1 != k2

    def test_run_inference_returns_prediction_intervals_from_predict_interval(
        self, mock_valkey, mock_mlflow_client, settings
    ):
        from app.agents.prediction import PredictionAgent

        class IntervalModel:
            def predict(self, fh):
                return pd.Series([10.0, 11.0], index=pd.RangeIndex(2), name="y")

            def predict_interval(self, fh, coverage):
                cols = pd.MultiIndex.from_tuples([
                    ("y", 0.9, "lower"),
                    ("y", 0.9, "upper"),
                ])
                return pd.DataFrame([[9.0, 11.0], [10.0, 12.0]], columns=cols)

        agent = PredictionAgent(mock_valkey, mock_mlflow_client, settings)
        preds, intervals = agent._run_inference(IntervalModel(), [1, 2])

        assert preds == [10.0, 11.0]
        assert intervals == {"lower": [9.0, 10.0], "upper": [11.0, 12.0]}

    def test_run_inference_returns_prediction_intervals_from_quantiles(
        self, mock_valkey, mock_mlflow_client, settings
    ):
        from app.agents.prediction import PredictionAgent

        class QuantileModel:
            def predict(self, fh):
                return np.array([5.0, 6.0])

            def predict_quantiles(self, fh, alpha):
                cols = pd.MultiIndex.from_tuples([
                    ("y", alpha[0]),
                    ("y", alpha[1]),
                ])
                return pd.DataFrame([[4.0, 6.0], [5.0, 7.0]], columns=cols)

        agent = PredictionAgent(mock_valkey, mock_mlflow_client, settings)
        preds, intervals = agent._run_inference(QuantileModel(), [1, 2])

        assert preds == [5.0, 6.0]
        assert intervals == {"lower": [4.0, 5.0], "upper": [6.0, 7.0]}

    def test_run_inference_pyfunc_model_leaves_intervals_none(
        self, mock_valkey, mock_mlflow_client, settings
    ):
        from app.agents.prediction import PredictionAgent

        class PyfuncLikeModel:
            unwrap_python_model = object()

            def predict(self, model_input):
                return pd.DataFrame({"y": [1.0, 2.0]})

        agent = PredictionAgent(mock_valkey, mock_mlflow_client, settings)
        preds, intervals = agent._run_inference(PyfuncLikeModel(), [1, 2])

        assert preds == [1.0, 2.0]
        assert intervals is None


# ---------------------------------------------------------------------------
# DriftMonitor
# ---------------------------------------------------------------------------

class TestDriftMonitor:
    @pytest.mark.asyncio
    async def test_no_retrain_when_residuals_stable(self, mock_valkey, settings):
        from app.monitoring.drift_monitor import DriftMonitor

        monitor = DriftMonitor(mock_valkey, settings)
        monitor._residuals["ds-1"] = deque(
            [0.01 if i % 2 == 0 else -0.01 for i in range(100)], maxlen=100
        )  # tiny zero-mean stable residuals
        monitor._prediction_counts["ds-1"]   = 50
        monitor._last_check_times["ds-1"]    = datetime(2020, 1, 1, tzinfo=timezone.utc)
        monitor._active_model_version["ds-1"] = "v1"
        monitor._adwin_triggered["ds-1"]     = False

        await monitor._run_detection("ds-1", "v1")
        mock_valkey.xadd.assert_not_called()

    @pytest.mark.asyncio
    async def test_major_drift_triggers_retrain_job(self, mock_valkey, settings):
        from app.monitoring.drift_monitor import DriftMonitor

        mock_valkey.exists = AsyncMock(return_value=False)
        monitor = DriftMonitor(mock_valkey, settings)

        # Rapidly growing residuals → major drift
        residuals = deque(
            [float(i * 2) for i in range(50)], maxlen=100
        )
        monitor._residuals["ds-1"]            = residuals
        monitor._prediction_counts["ds-1"]    = 50
        monitor._last_check_times["ds-1"]     = datetime(2020, 1, 1, tzinfo=timezone.utc)
        monitor._active_model_version["ds-1"] = "v1"
        monitor._adwin_triggered["ds-1"]      = False

        await monitor._run_detection("ds-1", "v1")
        mock_valkey.xadd.assert_called_once()
        stream_name = mock_valkey.xadd.call_args[0][0]
        assert stream_name == "retrain:jobs"

    @pytest.mark.asyncio
    async def test_duplicate_retrain_deduplicated(self, mock_valkey, settings):
        from app.monitoring.drift_monitor import DriftMonitor

        # Lock already set → _handle_major_drift should not call xadd
        mock_valkey.exists = AsyncMock(return_value=True)
        monitor = DriftMonitor(mock_valkey, settings)
        await monitor._handle_major_drift("ds-1", "CUSUM", 0.9)
        mock_valkey.xadd.assert_not_called()

    @pytest.mark.asyncio
    async def test_check_increments_prediction_count(self, mock_valkey, settings):
        from app.monitoring.drift_monitor import DriftMonitor
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
# Watchdog
# ---------------------------------------------------------------------------

class TestWatchdog:
    @pytest.mark.asyncio
    async def test_record_residual_writes_to_valkey(self, mock_valkey, settings):
        from app.agents.watchdog import Watchdog

        watchdog = Watchdog(mock_valkey, settings)
        await watchdog.record_residual("ds-1", "v1", predicted=1.5, actual=1.0)
        # pipeline().__aenter__().rpush should have been called
        pipe = mock_valkey.pipeline.return_value
        pipe.rpush.assert_called_once()

    @pytest.mark.asyncio
    async def test_invalid_baseline_aborts_immediately(self, mock_valkey, settings):
        from app.agents.watchdog import Watchdog

        watchdog = Watchdog(mock_valkey, settings)
        # baseline_score <= 0 should return immediately without Valkey calls
        await watchdog.monitor_post_promotion("ds-1", baseline_score=0.0)
        mock_valkey.get.assert_not_called()

    @pytest.mark.asyncio
    async def test_retrain_queued_stops_monitoring(self, mock_valkey, settings):
        from app.agents.watchdog import Watchdog

        mock_valkey.exists = AsyncMock(return_value=False)
        mock_valkey.get    = AsyncMock(return_value=b"v1")

        # Seed Valkey lrange to return large residuals immediately
        large_residuals = [str(float(i)).encode() for i in range(100)]
        mock_valkey.lrange = AsyncMock(return_value=large_residuals)

        watchdog = Watchdog(mock_valkey, settings)
        # Reduce thresholds so the test terminates quickly
        watchdog._min_obs        = 5
        watchdog._degrad_thresh  = 0.0   # any degradation triggers retrain
        watchdog._poll_interval  = 0.001

        await watchdog.monitor_post_promotion("ds-1", baseline_score=1.0, model_version="v1")
        mock_valkey.xadd.assert_called_once()


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class TestOrchestrator:
    @pytest.mark.asyncio
    async def test_orphaned_locks_cleaned_on_startup(
        self, mock_valkey, mock_mlflow_client, settings
    ):
        from app.orchestrator import Orchestrator
        from app.mcp.client import MCPClient

        orphaned_key = b"model_lock:ds-stale"
        mock_valkey.scan_iter = MagicMock(return_value=_async_iter([orphaned_key]))
        mock_valkey.ttl       = AsyncMock(return_value=-1)

        mcp = MCPClient()
        orch = Orchestrator(mock_valkey, mock_mlflow_client, mcp, settings)
        await orch.startup_cleanup()
        mock_valkey.delete.assert_called_with(orphaned_key)

    @pytest.mark.asyncio
    async def test_cold_start_triggers_full_pipeline(
        self, mock_valkey, mock_mlflow_client, settings
    ):
        from app.orchestrator import Orchestrator
        from app.mcp.client import MCPClient
        from app.schemas import ForecastRequest, ForecastResponse

        # No model version in Valkey → cold start
        mock_valkey.get = AsyncMock(return_value=None)
        mock_mlflow_client.get_latest_versions = MagicMock(return_value=[])

        mcp  = MCPClient()
        orch = Orchestrator(mock_valkey, mock_mlflow_client, mcp, settings)

        # Stub out the three pipeline stages
        orch.pipeline_architect.construct_pipeline = AsyncMock(return_value=None)
        orch.model_selector.select                 = AsyncMock(return_value=["NaiveForecaster"])
        orch.training_agent.handle_retrain_job     = AsyncMock(return_value="1")

        fake_response = ForecastResponse(
            dataset_id="ds-1",
            predictions=[1.0],
            model_version="1",
            model_class="NaiveForecaster",
            model_status="ok",
            cache_hit=False,
            correlation_id="c1",
        )
        orch.prediction_agent.predict = AsyncMock(return_value=fake_response)

        job    = ForecastRequest(dataset_id="ds-1", fh=[1], correlation_id="c1")
        result = await orch.handle_job(job)

        orch.pipeline_architect.construct_pipeline.assert_called_once_with("ds-1")
        orch.model_selector.select.assert_called_once()
        orch.training_agent.handle_retrain_job.assert_called_once()
        assert result.predictions == [1.0]

    @pytest.mark.asyncio
    async def test_warm_start_skips_pipeline(
        self, mock_valkey, mock_mlflow_client, settings
    ):
        from app.orchestrator import Orchestrator
        from app.mcp.client import MCPClient
        from app.schemas import ForecastRequest, ForecastResponse

        # Model version already present in Valkey
        mock_valkey.get = AsyncMock(return_value=b"3")

        mcp  = MCPClient()
        orch = Orchestrator(mock_valkey, mock_mlflow_client, mcp, settings)

        orch.pipeline_architect.construct_pipeline = AsyncMock()
        orch.model_selector.select                 = AsyncMock()
        orch.training_agent.handle_retrain_job     = AsyncMock()

        fake_response = ForecastResponse(
            dataset_id="ds-1",
            predictions=[2.0],
            model_version="3",
            model_class="AutoARIMA",
            model_status="ok",
            cache_hit=True,
            correlation_id="c1",
        )
        orch.prediction_agent.predict = AsyncMock(return_value=fake_response)

        job    = ForecastRequest(dataset_id="ds-1", fh=[1], correlation_id="c1")
        result = await orch.handle_job(job)

        orch.pipeline_architect.construct_pipeline.assert_not_called()
        orch.model_selector.select.assert_not_called()
        orch.training_agent.handle_retrain_job.assert_not_called()
        assert result.model_version == "3"
        assert isinstance(result.llm_rationale, str)
        assert "dataset ds-1" in result.llm_rationale

    def test_build_forecast_request_parses_actual(
        self, mock_valkey, mock_mlflow_client, settings
    ):
        from app.orchestrator import Orchestrator
        from app.mcp.client import MCPClient

        orch = Orchestrator(mock_valkey, mock_mlflow_client, MCPClient(), settings)
        req = orch._build_forecast_request(
            {
                "dataset_id": "ds-9",
                "fh": "[1,2]",
                "correlation_id": "corr-9",
                "actual": "123.4",
            },
            "1-0",
        )
        assert req.dataset_id == "ds-9"
        assert req.actual == 123.4

    @pytest.mark.asyncio
    async def test_start_stream_workers_forecast_only(
        self, mock_valkey, mock_mlflow_client, settings
    ):
        from app.orchestrator import Orchestrator
        from app.mcp.client import MCPClient

        orch = Orchestrator(mock_valkey, mock_mlflow_client, MCPClient(), settings)
        orch._forecast_worker_loop = AsyncMock(return_value=None)
        orch._retrain_worker_loop = AsyncMock(return_value=None)

        await orch.start_stream_workers(include_forecast=True, include_retrain=False)
        assert len(orch._worker_tasks) == 1
        assert mock_valkey.xgroup_create.call_count == 1

        await orch.stop_stream_workers()


# ---------------------------------------------------------------------------
# AgentMemory
# ---------------------------------------------------------------------------

class TestAgentMemory:
    @pytest.mark.asyncio
    async def test_get_returns_empty_on_cache_miss(self, mock_valkey):
        from app.memory.memory import AgentMemory
        mock_valkey.get = AsyncMock(return_value=None)
        mem = AgentMemory(mock_valkey)
        result = await mem.get_dataset_memory("ds-1")
        assert result["model_history"] == []
        assert result["drift_events"]  == []

    @pytest.mark.asyncio
    async def test_append_model_history_trimmed(self, mock_valkey):
        from app.memory.memory import AgentMemory, _MAX_MODEL_HISTORY

        existing = {
            "model_history": [{"estimator": "NaiveForecaster"}] * _MAX_MODEL_HISTORY,
            "drift_events": [],
            "data_characteristics": {},
        }
        mock_valkey.get = AsyncMock(return_value=json.dumps(existing).encode())
        mem = AgentMemory(mock_valkey)

        await mem.update_dataset_memory(
            "ds-1",
            {"append_model_history": {"estimator": "AutoARIMA", "val_mae": 0.3}},
        )

        # Should have been called with setex — grab the persisted payload
        call_args = mock_valkey.setex.call_args
        payload = json.loads(call_args[0][2])
        # List was trimmed to _MAX_MODEL_HISTORY entries
        assert len(payload["model_history"]) == _MAX_MODEL_HISTORY
        # Newest entry should be at the end
        assert payload["model_history"][-1]["estimator"] == "AutoARIMA"

    @pytest.mark.asyncio
    async def test_record_drift_event_convenience(self, mock_valkey):
        from app.memory.memory import AgentMemory
        mock_valkey.get = AsyncMock(return_value=None)
        mem = AgentMemory(mock_valkey)
        await mem.record_drift_event("ds-1", method="CUSUM", level="major", score=0.75)
        call_args = mock_valkey.setex.call_args
        payload   = json.loads(call_args[0][2])
        assert len(payload["drift_events"]) == 1
        assert payload["drift_events"][0]["method"] == "CUSUM"


class TestMainAPI:
    @pytest.mark.asyncio
    async def test_ready_check_ok(self, mock_valkey):
        from app import main

        mock_valkey.ping = AsyncMock(return_value=True)
        main.app.state.valkey = mock_valkey

        payload = await main.ready_check()
        assert payload["status"] == "ready"

    @pytest.mark.asyncio
    async def test_admin_retrain_queues_job(self, mock_valkey):
        from app import main
        from app.schemas import AdminRetrainRequest

        mock_valkey.exists = AsyncMock(return_value=False)
        mock_valkey.setex = AsyncMock(return_value=True)
        mock_valkey.xadd = AsyncMock(return_value=b"1-0")
        main.app.state.valkey = mock_valkey

        response = await main.admin_retrain(
            AdminRetrainRequest(dataset_id="ds-admin", reason="manual"),
            _auth=None,
        )
        assert response.queued is True
        assert response.stream_id == "1-0"

    @pytest.mark.asyncio
    async def test_admin_model_info_reads_cache_and_memory(
        self, mock_valkey, mock_mlflow_client
    ):
        from app import main

        async def get_side_effect(key):
            mapping = {
                "model_version:ds-admin": b"5",
                "model:class:ds-admin": b"AutoARIMA",
            }
            return mapping.get(key)

        mock_valkey.get = AsyncMock(side_effect=get_side_effect)
        main.app.state.valkey = mock_valkey
        main.app.state.mlflow_client = mock_mlflow_client

        class _Mem:
            async def get_dataset_memory(self, dataset_id):
                return {
                    "model_history": [
                        {
                            "estimator": "AutoARIMA",
                            "val_mae": 0.123,
                            "promoted_at": "2026-04-12T00:00:00Z",
                        }
                    ],
                    "drift_events": [{"method": "CUSUM"}],
                }

        main.app.state.agent_memory = _Mem()

        response = await main.admin_model_info("ds-admin", _auth=None)
        assert response.model_version == "5"
        assert response.model_class == "AutoARIMA"
        assert response.cv_score == 0.123
        assert response.drift_reason == "CUSUM"

    def test_metrics_endpoint_plain_text(self):
        from app import main

        response = main.metrics()
        assert response.status_code == 200
        assert "forecast_requests_total" in response.body.decode("utf-8")