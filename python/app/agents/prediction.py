from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import mlflow.pyfunc
import mlflow.sklearn
import numpy as np
import pandas as pd
from sktime.forecasting.base import ForecastingHorizon

from app.schemas import ForecastRequest, ForecastResponse

logger = logging.getLogger(__name__)

_MODEL_VER_KEY  = "model_version:{dataset_id}"
_PRED_COUNT_KEY = "pred_count:{dataset_id}"
_PRED_COUNT_TTL = 86_400   # 24 h
_DEFAULT_HORIZON = 10
_DEFAULT_INTERVAL_COVERAGE = 0.9


class PredictionAgent:
    """
    Parameters
    ----------
    valkey        : async Valkey/Redis client
    mlflow_client : synchronous MlflowClient (for registry lookups)
    settings      : app.config.Settings
    """

    def __init__(self, valkey, mlflow_client, settings):
        self.valkey   = valkey
        self.mlflow   = mlflow_client
        self.settings = settings

        self._local_cache: dict[tuple[str, str], Any] = {}

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
        job           : ForecastRequest — must have .dataset_id and .fh
        model_version : If provided, skip Valkey lookup (pinned deployment /
                        A/B testing).
        model_cache   : Optional shared cache dict[(dataset_id, version)] → model.
                        Falls back to self._local_cache if not provided.

        Returns
        -------
        ForecastResponse
        """
        dataset_id: str  = job.dataset_id
        fh_values: list[int] = list(getattr(job, "fh", []) or [])
        if not fh_values:
            default_horizon = int(
                getattr(self.settings, "default_horizon", _DEFAULT_HORIZON)
            )
            fh_values = list(range(1, default_horizon + 1))

        cache = model_cache if model_cache is not None else self._local_cache

        
        if model_version is None:
            model_version = await self._resolve_model_version(dataset_id)

        if model_version is None:
            raise RuntimeError(
                f"PredictionAgent: no registered model version found for "
                f"dataset_id={dataset_id}. Run TrainingAgent first."
            )

        
        cache_key = (dataset_id, model_version)
        cache_hit = cache_key in cache
        model     = await self._load_model(dataset_id, model_version, cache)

       
        t0 = time.monotonic()
        try:
            predictions, prediction_intervals = await asyncio.get_running_loop().run_in_executor(
                None,
                lambda: self._run_inference(model, fh_values),
            )
        except Exception as exc:
            logger.error(
                "PredictionAgent: inference failed for %s v%s: %s",
                dataset_id, model_version, exc,
            )
            raise

        elapsed_ms = (time.monotonic() - t0) * 1000

        response = ForecastResponse(
            dataset_id    = dataset_id,
            predictions   = predictions,
            prediction_intervals = prediction_intervals,
            model_version = model_version,
            model_class   = type(model).__name__,
            model_status  = "ok",
            drift_score   = None,
            drift_method  = None,
            warning       = None,
            llm_rationale = None,
            cache_hit     = cache_hit,
            correlation_id= job.correlation_id,
        )

        asyncio.ensure_future(self._increment_pred_count(dataset_id))

        logger.info(
            "PredictionAgent: served %d-step forecast for %s v%s "
            "in %.1f ms (cache_hit=%s)",
            len(fh_values), dataset_id, model_version, elapsed_ms, cache_hit,
        )
        return response

    def _run_inference(
        self,
        model: Any,
        fh_values: list[int],
    ) -> tuple[list[float], dict[str, list[float]] | None]:
    
        try:
            fh = ForecastingHorizon(fh_values, is_relative=True)
            raw = model.predict(fh)
            predictions = self._to_float_list(raw)
            intervals = self._try_predict_intervals(model, fh)
            return predictions, intervals
        except Exception as exc:
           
            logger.debug(
                "PredictionAgent: native ForecastingHorizon predict failed: %s; "
                "trying pyfunc DataFrame approach",
                exc,
            )
            try:
                raw = model.predict(fh=list(fh_values))
                predictions = self._to_float_list(raw)
                return predictions, None
            except Exception as pyfunc_exc:
                logger.error(
                    "PredictionAgent: both native and pyfunc predict failed: %s / %s",
                    exc, pyfunc_exc,
                )
                raise exc  

    @staticmethod
    def _to_float_list(raw: Any) -> list[float]:
        if isinstance(raw, pd.DataFrame):
            if raw.shape[1] == 0:
                return []
            return [float(v) for v in raw.iloc[:, 0].tolist()]

        if hasattr(raw, "values"):
            return [float(v) for v in np.ravel(raw.values)]

        return [float(v) for v in np.ravel(raw)]

    def _try_predict_intervals(
        self,
        model: Any,
        fh: ForecastingHorizon,
    ) -> dict[str, list[float]] | None:
        coverage = float(
            getattr(self.settings, "prediction_interval_coverage", _DEFAULT_INTERVAL_COVERAGE)
        )
        coverage = min(max(coverage, 1e-6), 0.999999)

        if hasattr(model, "predict_interval"):
            interval_df = self._call_predict_interval(model, fh, coverage)
            parsed = self._extract_interval_bounds(interval_df)
            if parsed is not None:
                return parsed

        if hasattr(model, "predict_quantiles"):
            quant_df = self._call_predict_quantiles(model, fh, coverage)
            parsed = self._extract_quantile_bounds(quant_df)
            if parsed is not None:
                return parsed

        return None

    @staticmethod
    def _call_predict_interval(
        model: Any,
        fh: ForecastingHorizon,
        coverage: float,
    ) -> Any | None:
        try:
            return model.predict_interval(fh=fh, coverage=[coverage])
        except TypeError:
            try:
                return model.predict_interval(fh=fh, coverage=coverage)
            except Exception:
                return None
        except Exception:
            return None

    @staticmethod
    def _call_predict_quantiles(
        model: Any,
        fh: ForecastingHorizon,
        coverage: float,
    ) -> Any | None:
        alpha = (1.0 - coverage) / 2.0
        upper_alpha = 1.0 - alpha
        try:
            return model.predict_quantiles(fh=fh, alpha=[alpha, upper_alpha])
        except Exception:
            return None

    @staticmethod
    def _extract_interval_bounds(data: Any) -> dict[str, list[float]] | None:
        if not isinstance(data, pd.DataFrame) or data.empty:
            return None

        if isinstance(data.columns, pd.MultiIndex):
            lower_cols = [c for c in data.columns if str(c[-1]).lower() == "lower"]
            upper_cols = [c for c in data.columns if str(c[-1]).lower() == "upper"]
            if lower_cols and upper_cols:
                lower = [float(v) for v in data[lower_cols[0]].tolist()]
                upper = [float(v) for v in data[upper_cols[0]].tolist()]
                if len(lower) == len(upper):
                    return {"lower": lower, "upper": upper}
            return None

        lower_col = next((c for c in data.columns if str(c).lower() == "lower"), None)
        upper_col = next((c for c in data.columns if str(c).lower() == "upper"), None)
        if lower_col is None or upper_col is None:
            return None

        lower = [float(v) for v in data[lower_col].tolist()]
        upper = [float(v) for v in data[upper_col].tolist()]
        if len(lower) != len(upper):
            return None
        return {"lower": lower, "upper": upper}

    @staticmethod
    def _extract_quantile_bounds(data: Any) -> dict[str, list[float]] | None:
        if not isinstance(data, pd.DataFrame) or data.empty:
            return None

        if isinstance(data.columns, pd.MultiIndex):
            cols = list(data.columns)
            sorted_cols = sorted(cols, key=lambda c: float(c[-1]))
            lower_col = sorted_cols[0]
            upper_col = sorted_cols[-1]
            lower = [float(v) for v in data[lower_col].tolist()]
            upper = [float(v) for v in data[upper_col].tolist()]
            if len(lower) == len(upper):
                return {"lower": lower, "upper": upper}
            return None

        if len(data.columns) < 2:
            return None
        sorted_cols = sorted(data.columns, key=lambda c: float(c))
        lower = [float(v) for v in data[sorted_cols[0]].tolist()]
        upper = [float(v) for v in data[sorted_cols[-1]].tolist()]
        if len(lower) != len(upper):
            return None
        return {"lower": lower, "upper": upper}

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
            logger.debug(
                "PredictionAgent: cache hit for %s v%s", dataset_id, model_version
            )
            return cache[cache_key]

        logger.info(
            "PredictionAgent: loading model from MLflow for %s v%s",
            dataset_id, model_version,
        )

        model = await asyncio.get_running_loop().run_in_executor(
            None,
            lambda: self._mlflow_load(dataset_id, model_version),
        )
        cache[cache_key] = model
        return model

    def _mlflow_load(self, dataset_id: str, model_version: str) -> Any:
        """
        Download and return the fitted model artifact from MLflow.

        Prioritizes pyfunc (which preserves full sktime pipelines with transforms),
        falls back to sklearn for simple estimators.
        """
        model_name = f"ts-forecaster-{dataset_id}"
        model_uri  = f"models:/{model_name}/{model_version}"

        # Try pyfunc first (handles full pipelines with transforms)
        try:
            model = mlflow.sklearn.load_model(model_uri)
            logger.debug(
                "PredictionAgent: loaded %s v%s as pyfunc model",
                model_name, model_version,
            )
            return model
        except Exception as pyfunc_exc:
            logger.debug(
                "PredictionAgent: pyfunc load failed for %s v%s: %s",
                model_name, model_version, pyfunc_exc,
            )

        # Fall back to sklearn for backward compatibility
        try:
            model = mlflow.sklearn.load_model(model_uri)
            logger.debug(
                "PredictionAgent: loaded %s v%s as sklearn model",
                model_name, model_version,
            )
            return model
        except Exception as sklearn_exc:
            logger.debug(
                "PredictionAgent: sklearn load failed for %s v%s: %s",
                model_name, model_version, sklearn_exc,
            )

        raise RuntimeError(
            f"PredictionAgent: cannot load model {model_name} v{model_version} "
            f"from MLflow (tried pyfunc and sklearn)"
        )

    async def _resolve_model_version(self, dataset_id: str) -> str | None:
        """
        Resolve the active model version:
        1. Valkey (set by TrainingAgent after promotion — fastest path).
        2. MLflow registry fallback (latest version in any stage).
        """
       
        try:
            key = _MODEL_VER_KEY.format(dataset_id=dataset_id)
            raw = await self.valkey.get(key)
            if raw:
                version = raw.decode() if isinstance(raw, bytes) else raw
                logger.debug(
                    "PredictionAgent: resolved version %s for %s from Valkey",
                    version, dataset_id,
                )
                return version
        except Exception as exc:
            logger.warning(
                "PredictionAgent: Valkey version lookup failed for %s: %s",
                dataset_id, exc,
            )

    
        try:
            model_name = f"ts-forecaster-{dataset_id}"
            versions   = self.mlflow.get_latest_versions(
                model_name, stages=["Production", "Staging", "None"]
            )
            if versions:
                latest = sorted(
                    versions, key=lambda v: int(v.version), reverse=True
                )[0]
                logger.debug(
                    "PredictionAgent: resolved version %s for %s from MLflow",
                    latest.version, dataset_id,
                )
                return str(latest.version)
        except Exception as exc:
            logger.warning(
                "PredictionAgent: MLflow version lookup failed for %s: %s",
                dataset_id, exc,
            )

        return None

    async def _increment_pred_count(self, dataset_id: str) -> None:
        """Increment the per-dataset prediction counter in Valkey."""
        key = _PRED_COUNT_KEY.format(dataset_id=dataset_id)
        try:
            async with self.valkey.pipeline(transaction=False) as pipe:
                await pipe.incr(key)
                await pipe.expire(key, _PRED_COUNT_TTL)
                await pipe.execute()
        except Exception as exc:
            logger.debug(
                "PredictionAgent: failed to increment pred_count for %s: %s",
                dataset_id, exc,
            )