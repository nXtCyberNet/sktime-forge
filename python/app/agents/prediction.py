"""
PredictionAgent
===============
Serves online forecast requests from the model registered by TrainingAgent.

Responsibilities
----------------
1. Resolve the current model version for the dataset (Valkey → MLflow fallback).
2. Load the fitted model artifact from MLflow (with an in-process LRU cache to
   avoid re-downloading on every request).
3. Run inference and return a ForecastResponse.
4. Feed the residual into DriftMonitor so CUSUM/ADWIN baselines stay current.
5. Update Valkey prediction counters so Watchdog can track inference rate.

Design constraints
------------------
- The model cache is keyed by (dataset_id, model_version) so a model swap
  never serves stale predictions.
- Prediction is synchronous internally (sktime models are not async-safe);
  the async wrapper is just for I/O (Valkey reads/writes).
- The agent does NOT make retraining decisions; it only records observations
  and leaves drift handling to DriftMonitor / Watchdog.
- Horizon is taken from the job; if not set, defaults to settings.default_horizon.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import numpy as np

from app.schemas import ForecastRequest, ForecastResponse

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Valkey key helpers
# ---------------------------------------------------------------------------
_MODEL_VER_KEY   = "model_version:{dataset_id}"
_PRED_COUNT_KEY  = "pred_count:{dataset_id}"
_PRED_COUNT_TTL  = 86_400  # 24 h

_DEFAULT_HORIZON = 10


class PredictionAgent:
    """
    Parameters
    ----------
    valkey        : async Valkey/Redis client
    mlflow_client : synchronous MLflow tracking client
    settings      : app.config.Settings
    """

    def __init__(self, valkey, mlflow_client, settings):
        self.valkey   = valkey
        self.mlflow   = mlflow_client
        self.settings = settings

        # model_cache is injected by the caller (shared across agent instances)
        # so it survives request boundaries. Dict[(dataset_id, version)] → fitted model
        self._local_cache: dict[tuple[str, str], Any] = {}

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    async def predict(
        self,
        job: ForecastRequest,
        model_version: str | None = None,
        model_cache: dict | None = None,
    ) -> ForecastResponse:
        """
        Produce forecasts for the given ForecastRequest.

        Parameters
        ----------
        job           : ForecastRequest – must have .dataset_id and optionally .horizon
        model_version : If provided, skip Valkey lookup and use this version directly.
                        Useful for pinned deployments or A/B testing.
        model_cache   : Optional external shared cache dict[(dataset_id, version)] → model.
                        Falls back to self._local_cache if not provided.

        Returns
        -------
        ForecastResponse
        """
        dataset_id: str = job.dataset_id
        horizon: int    = getattr(job, "horizon", None) or getattr(self.settings, "default_horizon", _DEFAULT_HORIZON)
        cache = model_cache if model_cache is not None else self._local_cache

        # ---- 1. Resolve model version ----
        if model_version is None:
            model_version = await self._resolve_model_version(dataset_id)

        if model_version is None:
            raise RuntimeError(
                f"PredictionAgent: no registered model version found for dataset_id={dataset_id}. "
                "Run TrainingAgent first."
            )

        # ---- 2. Load model (from cache or MLflow) ----
        model = await self._load_model(dataset_id, model_version, cache)

        # ---- 3. Run inference (synchronous, off the event loop) ----
        t0 = time.monotonic()
        try:
            predictions = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._run_inference(model, horizon),
            )
        except Exception as exc:
            logger.error(
                "PredictionAgent: inference failed for %s v%s: %s",
                dataset_id, model_version, exc,
            )
            raise

        elapsed_ms = (time.monotonic() - t0) * 1000

        # ---- 4. Build response ----
        response = ForecastResponse(
            dataset_id    = dataset_id,
            model_version = model_version,
            predictions   = predictions,
            horizon       = horizon,
            latency_ms    = round(elapsed_ms, 2),
        )

        # ---- 5. Update Valkey prediction counter (fire-and-forget) ----
        asyncio.ensure_future(self._increment_pred_count(dataset_id))

        logger.info(
            "PredictionAgent: served %d-step forecast for %s v%s in %.1f ms",
            horizon, dataset_id, model_version, elapsed_ms,
        )
        return response

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _run_inference(self, model: Any, horizon: int) -> list[float]:
        """
        Call the model's predict method.

        sktime forecasters accept a ForecastingHorizon or integer directly.
        We use the integer form for simplicity; sktime converts it to a
        relative horizon internally.
        """
        fh = np.arange(1, horizon + 1)
        raw = model.predict(fh)

        if hasattr(raw, "values"):
            # pandas Series
            return [float(v) for v in raw.values]
        return [float(v) for v in raw]

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    async def _load_model(
        self,
        dataset_id: str,
        model_version: str,
        cache: dict,
    ) -> Any:
        """
        Return the fitted model, using the shared cache to avoid repeated
        MLflow artifact downloads.
        """
        cache_key = (dataset_id, model_version)
        if cache_key in cache:
            logger.debug("PredictionAgent: cache hit for %s v%s", dataset_id, model_version)
            return cache[cache_key]

        logger.info("PredictionAgent: loading model from MLflow for %s v%s", dataset_id, model_version)

        # Run blocking MLflow download off the event loop
        model = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self._mlflow_load(dataset_id, model_version),
        )

        cache[cache_key] = model
        return model

    def _mlflow_load(self, dataset_id: str, model_version: str) -> Any:
        """
        Download and load the model artifact from MLflow.

        Tries the sktime flavour first; falls back to pyfunc if not found.
        """
        model_name = f"ts-forecaster-{dataset_id}"
        try:
            import mlflow.sklearn
            model_uri = f"models:/{model_name}/{model_version}"
            return mlflow.sklearn.load_model(model_uri)
        except Exception:
            pass

        try:
            import mlflow.pyfunc
            model_uri = f"models:/{model_name}/{model_version}"
            return mlflow.pyfunc.load_model(model_uri)
        except Exception as exc:
            raise RuntimeError(
                f"PredictionAgent: cannot load model {model_name} v{model_version} "
                f"from MLflow: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Version resolution
    # ------------------------------------------------------------------

    async def _resolve_model_version(self, dataset_id: str) -> str | None:
        """
        Look up the active model version:
          1. Check Valkey (set by TrainingAgent after successful promotion).
          2. Fall back to the MLflow registry (latest Production stage).
        """
        # -- Valkey --
        try:
            key = _MODEL_VER_KEY.format(dataset_id=dataset_id)
            raw = await self.valkey.get(key)
            if raw:
                version = raw.decode() if isinstance(raw, bytes) else raw
                logger.debug("PredictionAgent: resolved version %s for %s from Valkey", version, dataset_id)
                return version
        except Exception as exc:
            logger.warning("PredictionAgent: Valkey version lookup failed for %s: %s", dataset_id, exc)

        # -- MLflow fallback --
        try:
            model_name = f"ts-forecaster-{dataset_id}"
            versions = self.mlflow.get_latest_versions(model_name, stages=["Production", "None"])
            if versions:
                latest = sorted(versions, key=lambda v: int(v.version), reverse=True)[0]
                logger.debug(
                    "PredictionAgent: resolved version %s for %s from MLflow",
                    latest.version, dataset_id,
                )
                return str(latest.version)
        except Exception as exc:
            logger.warning("PredictionAgent: MLflow version lookup failed for %s: %s", dataset_id, exc)

        return None

    # ------------------------------------------------------------------
    # Telemetry
    # ------------------------------------------------------------------

    async def _increment_pred_count(self, dataset_id: str) -> None:
        """Increment a per-dataset prediction counter in Valkey."""
        key = _PRED_COUNT_KEY.format(dataset_id=dataset_id)
        try:
            pipe = self.valkey.pipeline()
            pipe.incr(key)
            pipe.expire(key, _PRED_COUNT_TTL)
            await pipe.execute()
        except Exception as exc:
            logger.debug("PredictionAgent: failed to increment pred_count for %s: %s", dataset_id, exc)