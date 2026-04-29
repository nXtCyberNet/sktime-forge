from __future__ import annotations

import asyncio
import hashlib
import importlib
import json
import logging
import time
import traceback
from typing import Any

import mlflow
import mlflow.pyfunc
import mlflow.sklearn
import numpy as np
import pandas as pd
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.compose import TransformedTargetForecaster
from sktime.transformations.series.difference import Differencer
from sktime.transformations.series.detrend import Deseasonalizer
from sktime.performance_metrics.forecasting import (
    MeanAbsoluteError,
    MeanAbsolutePercentageError,
    MeanSquaredError,
)

from app.registry.registry import validate_pipeline_spec

logger = logging.getLogger(__name__)

_CANDIDATE_KEY   = "candidates:{dataset_id}"
_MODEL_VER_KEY   = "model_version:{dataset_id}"
_MODEL_VER_TTL   = 86_400   # 24 h
_VALIDATION_FRAC = 0.2      # last 20 % used for evaluation

# Pre-instantiate metric objects once (they are stateless)
_MAE  = MeanAbsoluteError()
_MAPE = MeanAbsolutePercentageError()
_RMSE = MeanSquaredError(square_root=True)

_ESTIMATOR_MAP: dict[str, tuple[str, str, dict[str, Any]]] = {
    "NaiveForecaster": (
        "sktime.forecasting.naive", "NaiveForecaster",
        {"strategy": "last"},
    ),
    "PolynomialTrendForecaster": (
        "sktime.forecasting.trend", "PolynomialTrendForecaster",
        {"degree": 1},
    ),
    "ThetaForecaster": (
        "sktime.forecasting.theta", "ThetaForecaster",
        {},
    ),
    "ExponentialSmoothing": (
        "sktime.forecasting.exp_smoothing", "ExponentialSmoothing",
        {"trend": "add", "damped_trend": True},
    ),
    "AutoARIMA": (
        "sktime.forecasting.arima", "AutoARIMA",
        {"sp": 1, "suppress_warnings": True, "error_action": "ignore"},
    ),
    "AutoETS": (
        "sktime.forecasting.ets", "AutoETS",
        {"auto": True, "information_criterion": "aic"},
    ),
    "Prophet": (
        "sktime.forecasting.fbprophet", "Prophet",
        {"seasonality_mode": "additive"},
    ),
    "BATS": (
        "sktime.forecasting.bats", "BATS",
        {"use_box_cox": None, "use_trend": True},
    ),
    "TBATS": (
        "sktime.forecasting.tbats", "TBATS",
        {"use_box_cox": None, "use_trend": True},
    ),
}


class _SktimePyfuncModel(mlflow.pyfunc.PythonModel):
    """PyFunc wrapper that serves a fitted sktime estimator by relative fh."""

    def __init__(self, estimator: Any):
        self._estimator = estimator

    def predict(self, context, model_input):
        if isinstance(model_input, pd.DataFrame):
            if "fh" in model_input.columns:
                fh_values = [int(v) for v in model_input["fh"].tolist()]
            else:
                fh_values = [int(v) for v in model_input.iloc[:, 0].tolist()]
        elif isinstance(model_input, (list, tuple, np.ndarray, pd.Series)):
            fh_values = [int(v) for v in list(model_input)]
        else:
            raise ValueError("Unsupported model_input for pyfunc forecast wrapper")

        fh = ForecastingHorizon(fh_values, is_relative=True)
        preds = self._estimator.predict(fh)

        if hasattr(preds, "to_numpy"):
            return preds.to_numpy()
        return np.asarray(preds)


class TrainingAgent:
    """
    Parameters
    ----------
    valkey        : async Valkey/Redis client
    mlflow_client : synchronous MlflowClient (used for registry queries only)
    settings      : app.config.Settings
    """

    def __init__(self, valkey, mlflow_client, settings):
        self.valkey   = valkey
        self.mlflow   = mlflow_client   # MlflowClient — queries only
        self.settings = settings
        self._last_training_summary: dict[str, dict[str, Any]] = {}

    async def handle_retrain_job(self, job) -> str | None:
        """
        Train all candidate models and promote the winner to the registry.

        Parameters
        ----------
        job : object or dict with .dataset_id / ["dataset_id"]

        Returns
        -------
        str | None
            MLflow model version string for the promoted model, or None on
            total failure.
        """
        dataset_id: str = (
            job.dataset_id if hasattr(job, "dataset_id") else job["dataset_id"]
        )
        reason: str = getattr(
            job, "reason",
            job.get("reason", "scheduled") if isinstance(job, dict) else "scheduled",
        )
        logger.info(
            "TrainingAgent.handle_retrain_job: dataset_id=%s reason=%s",
            dataset_id, reason,
        )

        candidates: list[str] = await self._load_candidates(dataset_id)
        candidates = self._sanitize_candidates(candidates, dataset_id)
        if not candidates:
            logger.error("TrainingAgent: no candidates found for %s – aborting", dataset_id)
            return None

        y_train, y_val = self._load_data(dataset_id)
        if len(y_train) < 5:
            logger.error(
                "TrainingAgent: insufficient training data for %s (%d obs)",
                dataset_id, len(y_train),
            )
            return None

        sp = 1
        try:
            profile_json = await self.valkey.get(f"profile:{dataset_id}")
            if profile_json:
                prof = json.loads(profile_json)
                sp = int(prof.get("seasonality", {}).get("period", 1) or 1)
        except Exception as exc:
            logger.warning("TrainingAgent: failed to load profile to get sp: %s", exc)

        experiment_name = f"ts-{dataset_id}"
        experiment_id   = self._ensure_experiment(experiment_name, dataset_id)

        results: list[dict[str, Any]] = []
        loop = asyncio.get_running_loop()

        for estimator_name in candidates:
            result = await loop.run_in_executor(
                None,
                lambda name=estimator_name: self._train_one(
                    dataset_id=dataset_id,
                    estimator_name=name,
                    y_train=y_train,
                    y_val=y_val,
                    experiment_id=experiment_id,
                    reason=reason,
                    sp=sp,
                ),
            )
            if result is not None:
                results.append(result)
                early_stop_mae = getattr(self.settings, "early_stop_mae", None)
                if early_stop_mae and result["val_mae"] <= float(early_stop_mae):
                    logger.info(
                        "TrainingAgent: early stop triggered for %s "
                        "(mae=%.4f ≤ threshold=%.4f)",
                        estimator_name, result["val_mae"], early_stop_mae,
                    )
                    break

        if not results:
            logger.error("TrainingAgent: all candidates failed for %s", dataset_id)
            return None
        best = min(results, key=lambda r: r["val_mae"])
        logger.info(
            "TrainingAgent: best model for %s is %s (val_mae=%.4f)",
            dataset_id, best["estimator_name"], best["val_mae"],
        )

        self._last_training_summary[dataset_id] = {
            "dataset_id": dataset_id,
            "estimator_name": str(best["estimator_name"]),
            "val_mae": float(best["val_mae"]),
            "model_version": None,
        }

        model_version = await loop.run_in_executor(
            None, lambda: self._register_model(dataset_id, best)
        )
        if model_version is None:
            return None

        key = _MODEL_VER_KEY.format(dataset_id=dataset_id)
        try:
            await self.valkey.setex(key, _MODEL_VER_TTL, model_version)
        except Exception as exc:
            logger.warning(
                "TrainingAgent: Valkey cache write failed (non-fatal): %s", exc
            )

        await self.valkey.set(f"model:version:{dataset_id}", model_version)
        await self.valkey.set(f"model:class:{dataset_id}", best["estimator_name"])

        await self.valkey.setex(f"model_updated:{dataset_id}", 300, "1")

        logger.info(
            "TrainingAgent: promoted model version %s for %s",
            model_version, dataset_id,
        )

        self._last_training_summary[dataset_id]["model_version"] = str(model_version)
        return model_version

    def get_last_training_summary(self, dataset_id: str) -> dict[str, Any] | None:
        return self._last_training_summary.get(dataset_id)

    def _train_one(
        self,
        dataset_id: str,
        estimator_name: str,
        y_train: np.ndarray,
        y_val: np.ndarray,
        experiment_id: str | None,
        reason: str,
        sp: int = 1,
    ) -> dict[str, Any] | None:
        """
        Fit a single estimator, evaluate it, and log an MLflow run.

        Returns None on any unrecoverable error so the caller can skip ahead
        to the next candidate.

        sktime API notes
        ----------------
        - fit()     requires a pd.Series (not np.ndarray).
        - predict() requires a ForecastingHorizon (not np.arange).
        - Metrics   require pd.Series for both y_true and y_pred.
        """
        logger.info("TrainingAgent: fitting %s for %s", estimator_name, dataset_id)
        t0 = time.monotonic()

        y_train_s = pd.Series(y_train, index=pd.RangeIndex(len(y_train)), name="y")
        y_val_s   = pd.Series(
            y_val,
            index=pd.RangeIndex(len(y_train), len(y_train) + len(y_val)),
            name="y",
        )

        try:
            estimator = self._instantiate_estimator(estimator_name, sp=sp)
        except (ImportError, ValueError) as exc:
            logger.error(
                "TrainingAgent: cannot instantiate %s: %s", estimator_name, exc
            )
            return None

        profile_json = None
        try:
            profile_json = asyncio.run(self.valkey.get(f"profile:{dataset_id}"))
        except Exception as exc:
            logger.warning(
                "TrainingAgent: failed to load profile for %s: %s", dataset_id, exc
            )

        pipeline = self._build_training_pipeline(
            estimator=estimator,
            profile_json=profile_json,
            sp=sp,
        )

        try:
            pipeline.fit(y_train_s)
        except Exception as exc:
            logger.error(
                "TrainingAgent: fit failed for %s on %s: %s",
                estimator_name, dataset_id, exc,
            )
            return None

        elapsed_fit = time.monotonic() - t0

        try:
            fh = ForecastingHorizon(
                list(range(1, len(y_val_s) + 1)), is_relative=True
            )
            preds: pd.Series = pipeline.predict(fh)

            preds.index = y_val_s.index

            val_mae  = float(_MAE(y_val_s,  preds))
            val_mape = float(_MAPE(y_val_s, preds))
            val_rmse = float(_RMSE(y_val_s, preds))
        except Exception as exc:
            logger.error(
                "TrainingAgent: predict/eval failed for %s on %s: %s",
                estimator_name, dataset_id, exc,
            )
            return None

        run_id: str | None = None
        try:
            with mlflow.start_run(experiment_id=experiment_id) as run:
                run_id = run.info.run_id
                mlflow.log_params({
                    "estimator":       estimator_name,
                    "dataset_id":      dataset_id,
                    "n_train":         len(y_train_s),
                    "n_val":           len(y_val_s),
                    "retrain_reason":  reason,
                })
                mlflow.log_metrics({
                    "val_mae":      val_mae,
                    "val_mape":     val_mape,
                    "val_rmse":     val_rmse,
                    "fit_seconds":  elapsed_fit,
                })
                mlflow.set_tags({
                    "estimator":  estimator_name,
                    "dataset_id": dataset_id,
                })

                model_logged = self._log_model_artifact(pipeline, estimator_name)
                if not model_logged:
                    logger.error(
                        "TrainingAgent: model artifact logging failed for %s; "
                        "skipping this candidate",
                        estimator_name,
                    )
                    return None
        except Exception as exc:
            logger.warning(
                "TrainingAgent: MLflow run logging failed for %s: %s",
                estimator_name, exc,
            )

        logger.info(
            "TrainingAgent: %s → val_mae=%.4f val_rmse=%.4f fit_seconds=%.1f",
            estimator_name, val_mae, val_rmse, elapsed_fit,
        )

        return {
            "estimator_name": estimator_name,
            "estimator_obj":  pipeline,
            "val_mae":        val_mae,
            "val_mape":       val_mape,
            "val_rmse":       val_rmse,
            "fit_seconds":    elapsed_fit,
            "run_id":         run_id,
        }

    def _instantiate_estimator(self, name: str, sp: int = 1) -> Any:
        """
        Lazily import and instantiate a sktime-compatible estimator by name.

        Raises
        ------
        ImportError  — optional package not installed (e.g. prophet)
        ValueError   — unknown estimator name
        """
        if name not in _ESTIMATOR_MAP:
            raise ValueError(
                f"Unknown estimator: {name!r}. "
                f"Known names: {sorted(_ESTIMATOR_MAP)}"
            )
        module_path, class_name, default_kwargs = _ESTIMATOR_MAP[name]
        
        # Inject 'sp' for seasonal models
        kwargs = dict(default_kwargs)
        if name in ("AutoETS", "AutoARIMA", "ExponentialSmoothing", "TBATS", "BATS"):
            kwargs["sp"] = sp
            if name == "ExponentialSmoothing":
                if sp > 1:
                    kwargs["seasonal"] = "add"
                else:
                    kwargs.pop("seasonal", None)

        mod = importlib.import_module(module_path)
        cls = getattr(mod, class_name)
        return cls(**kwargs)

    def _build_training_pipeline(
        self,
        estimator: Any,
        profile_json: str | None,
        sp: int = 1,
    ) -> Any:
        """
        Wrap the selected forecaster in a TransformedTargetForecaster pipeline.

        This ensures any target transformation step is persisted together
        with the fitted estimator so MLflow can restore the full pipeline
        and perform inverse_transform automatically.
        """
        steps: list[tuple[str, Any]] = []
        profile: dict[str, Any] | None = None

        if profile_json:
            try:
                profile = json.loads(profile_json)
            except Exception:
                profile = None

        if profile is not None:
            seasonality = profile.get("seasonality", {}) or {}
            stationarity = profile.get("stationarity", {}) or {}

            if seasonality.get("seasonality_class") not in (None, "none"):
                seasonality_class = str(seasonality.get("seasonality_class", "")).lower()
                model = "multiplicative" if "multiplicative" in seasonality_class else "additive"
                steps.append(("deseasonalizer", Deseasonalizer(model=model, sp=sp)))

            if stationarity.get("conclusion") in ("difference_stationary", "trend_stationary"):
                steps.append(("differencer", Differencer()))

        steps.append(("forecaster", estimator))
        return TransformedTargetForecaster(steps=steps)

    def _log_model_artifact(self, estimator: Any, estimator_name: str) -> bool:
        """
        Log fitted model artifact under "model" in the active MLflow run.

        For sktime TransformedTargetForecaster pipelines, we use pyfunc wrapper
        to ensure the full pipeline (including inverse_transform) is serialized.
        """
        from sktime.forecasting.compose import TransformedTargetForecaster
        
        # Always use pyfunc for pipelines to preserve full transform chain
        if isinstance(estimator, TransformedTargetForecaster):
            try:
                mlflow.sklearn.log_model(
                    estimator,
                    artifact_path="model",                   
                    serialization_format="cloudpickle"
                )
                logger.info(
                "TrainingAgent: logged %s as sklearn pipeline", estimator_name
            )
                return True
            except Exception as exc:
                logger.error(
                    "TrainingAgent: sklearn logging failed for pipeline %s: %s , %s ",
                    estimator_name, exc , traceback.format_exc()
                )
            return False
        
        # Try sklearn for simple estimators
        try:
            mlflow.sklearn.log_model(estimator, artifact_path="model")
            logger.info("TrainingAgent: logged %s as sklearn model", estimator_name)
            return True
        except Exception as log_exc:
            logger.warning(
                "TrainingAgent: mlflow.sklearn.log_model failed for %s (%s); "
                "trying pyfunc fallback",
                estimator_name,
                log_exc,
            )

        try:
            pyfunc_model = _SktimePyfuncModel(estimator)
            mlflow.pyfunc.log_model(
                artifact_path="model",
                python_model=pyfunc_model,
                input_example=pd.DataFrame({"fh": [1, 2, 3]}),
            )
            logger.info("TrainingAgent: logged %s as pyfunc (fallback)", estimator_name)
            return True
        except Exception as pyfunc_exc:
            logger.error(
                "TrainingAgent: mlflow.pyfunc.log_model fallback failed for %s: %s",
                estimator_name,
                pyfunc_exc,
            )
            return False

    def _ensure_experiment(
        self, experiment_name: str, dataset_id: str
    ) -> str | None:
        """Create the MLflow experiment if it does not already exist."""
        try:
            exp = self.mlflow.get_experiment_by_name(experiment_name)
            if exp is None:
                return self.mlflow.create_experiment(
                    experiment_name, tags={"dataset_id": dataset_id}
                )
            if getattr(exp, "lifecycle_stage", "active") == "deleted":
                try:
                    self.mlflow.restore_experiment(exp.experiment_id)
                    logger.info(
                        "TrainingAgent: restored deleted MLflow experiment %s (%s)",
                        experiment_name,
                        exp.experiment_id,
                    )
                    return exp.experiment_id
                except Exception as restore_exc:
                    logger.warning(
                        "TrainingAgent: failed to restore deleted experiment %s: %s",
                        experiment_name,
                        restore_exc,
                    )
                    return self.mlflow.create_experiment(
                        experiment_name, tags={"dataset_id": dataset_id}
                    )
            return exp.experiment_id
        except Exception as exc:
            logger.warning(
                "TrainingAgent: MLflow experiment setup warning for %s: %s",
                experiment_name, exc,
            )
            return None

    def _register_model(
        self, dataset_id: str, best: dict[str, Any]
    ) -> str | None:
        model_name = f"ts-forecaster-{dataset_id}"
        run_id     = best.get("run_id")

        if not run_id:
            logger.warning("TrainingAgent: no run_id for best model – skipping registry")
            return None

        try:
            model_uri = f"runs:/{run_id}/model"
            mv = mlflow.register_model(model_uri, model_name)

            self.mlflow.update_registered_model(
                model_name,
                description=(
                    f"Best forecaster for dataset '{dataset_id}' "
                    f"(estimator={best['estimator_name']}, "
                    f"val_mae={best['val_mae']:.4f})"
                ),
            )
            self.mlflow.set_registered_model_tag(model_name, "dataset_id", dataset_id)
            self.mlflow.set_registered_model_tag(
                model_name, "estimator", best["estimator_name"]
            )
            return str(mv.version)

        except Exception as exc:
            logger.error(
                "TrainingAgent: MLflow registration failed for %s: %s",
                dataset_id, exc,
            )
            return None

    def _load_data(
        self, dataset_id: str
    ) -> tuple[np.ndarray, np.ndarray]:
        loader = getattr(self.settings, "data_loader", None)
        if loader is not None:
            y = np.asarray(loader(dataset_id), dtype=float)
        else:
            digest = hashlib.sha256(dataset_id.encode("utf-8")).digest()
            seed = int.from_bytes(digest[:8], byteorder="big", signed=False) % (2**31)
            rng = np.random.default_rng(seed=seed)
            y   = rng.standard_normal(200).cumsum()

        split = max(5, int(len(y) * (1 - _VALIDATION_FRAC)))
        return y[:split], y[split:]

    async def _load_candidates(self, dataset_id: str) -> list[str]:
        """Read the ranked candidate list written by ModelSelectorAgent."""
        key = _CANDIDATE_KEY.format(dataset_id=dataset_id)
        raw = await self.valkey.get(key)
        if not raw:
            logger.warning(
                "TrainingAgent: no candidates key in Valkey for %s", dataset_id
            )
            return []
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            logger.error(
                "TrainingAgent: corrupt candidates payload for %s: %s",
                dataset_id, exc,
            )
            return []

    def _sanitize_candidates(self, candidates: list[str], dataset_id: str) -> list[str]:
        """Validate and filter candidate estimators against runtime-supported registry."""
        deduped = list(dict.fromkeys(str(c) for c in candidates))
        runtime_registry = list(_ESTIMATOR_MAP.keys())

        spec = {"estimators": deduped}
        if validate_pipeline_spec(spec, registry=runtime_registry):
            return deduped

        filtered = [name for name in deduped if name in _ESTIMATOR_MAP]
        dropped = [name for name in deduped if name not in _ESTIMATOR_MAP]
        if dropped:
            logger.warning(
                "TrainingAgent: dropping unsupported estimators for %s: %s",
                dataset_id,
                dropped,
            )

        spec = {"estimators": filtered}
        if validate_pipeline_spec(spec, registry=runtime_registry):
            return filtered
        return []