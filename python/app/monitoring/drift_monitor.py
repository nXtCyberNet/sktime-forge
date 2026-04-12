from __future__ import annotations

import json
import logging
from collections import deque
from datetime import datetime, timezone
from importlib import import_module
from typing import Any, Tuple

import numpy as np

from app.config import Settings
from app.contracts import AgentMemoryProtocol
from app.schemas import ForecastRequest, ForecastResponse

logger = logging.getLogger(__name__)

def _cusum_score(residuals: deque[float], k: float, h: float) -> float:
    if len(residuals) < 5:
        return 0.0

    arr = np.array([float(x) for x in residuals if np.isfinite(x)], dtype=float)
    if len(arr) < 5:
        return 0.0

    mu = 0.0
    s_pos = 0.0
    s_neg = 0.0
    for residual in arr:
        s_pos = max(0.0, s_pos + (residual - mu) - k)
        s_neg = max(0.0, s_neg - (residual - mu) - k)

    peak = max(s_pos, s_neg)
    return float(max(0.0, min(peak / max(float(h), 1e-8), 1.0)))

def _build_adwin() -> Any | None:
    """Create a River ADWIN instance when the package is available."""
    try:
        module = import_module("river.drift")
        return module.ADWIN()
    except Exception:
        return None

def _adwin_severity_from_window(values: deque[float]) -> float:
    if len(values) < 20:
        return 0.0

    arr = np.array([float(x) for x in values if np.isfinite(x)], dtype=float)
    if len(arr) < 20:
        return 0.0

    mid = len(arr) // 2
    old, new = arr[:mid], arr[mid:]
    pooled_std = max((float(old.std()) + float(new.std())) / 2.0, 1e-8)
    effect_size = abs(float(old.mean()) - float(new.mean())) / pooled_std

    return float(max(0.0, min(1.0, 1.0 - np.exp(-effect_size))))

class DriftMonitor:
    def __init__(
        self,
        valkey: Any,
        settings: Settings,
        agent_memory: AgentMemoryProtocol | None = None,
        window_size: int = 100,
    ) -> None:
        self.valkey = valkey
        self.settings = settings
        self.agent_memory = agent_memory
        self.window_size = window_size

        self._residuals: dict[str, deque[float]] = {}
        self._prediction_counts: dict[str, int] = {}
        self._last_check_times: dict[str, datetime] = {}
        self._active_model_version: dict[str, str] = {}
        self._adwin_detectors: dict[str, Any | None] = {}
        self._adwin_triggered: dict[str, bool] = {}

        self._baseline_params: dict[str, dict[str, float]] = {}
        self._baseline_bootstrap: dict[str, deque[float]] = {}

        # Missing config defaults handled gracefully
        self.cusum_baseline_min_samples = 30
        self.cusum_k_sigma = 0.5
        self.cusum_h_sigma = 5.0

    def triage(self, dataset_id: str, model_version: str) -> str:
        """Stage 3: Fast, synchronous triage. No network calls."""
        self._ensure_dataset_state(dataset_id)
        self._ensure_model_version_state(dataset_id, model_version)
        
        method, score = self._get_current_drift_score(dataset_id, model_version)

        no_drift_threshold = float(self.settings.no_drift_threshold)
        major_drift_threshold = float(
            getattr(self.settings, "major_drift_threshold", self.settings.minor_drift_threshold)
        )
        
        if score < no_drift_threshold:
            return "none"
        if score < major_drift_threshold:
            return "minor"
        return "major"

    def _get_current_drift_score(self, dataset_id: str, model_version: str) -> Tuple[str, float]:
        residuals = self._residuals.get(dataset_id)
        if not residuals or len(residuals) < 10:
            return "none", 0.0

        arr = np.array([float(x) for x in residuals if np.isfinite(x)], dtype=float)
        if len(arr) < 10:
            return "none", 0.0
            
        residual_mean_score = min(abs(float(arr.mean())) / (float(arr.std()) or 1.0), 1.0)

        # CUSUM - synchronous memory check only per Stage 3 constraints
        key = self._baseline_key(dataset_id, model_version)
        baseline = self._baseline_params.get(key)
        if baseline is None:
            cusum = 0.0
        else:
            cusum = _cusum_score(residuals, k=float(baseline["k"]), h=float(baseline["h"]))

        # ADWIN
        adwin = self._adwin_score(dataset_id, residuals)

        method_scores = {
            "residual": residual_mean_score,
            "CUSUM": cusum,
            "ADWIN": adwin,
        }
        method = max(method_scores, key=method_scores.get)
        score = float(method_scores[method])
        return method, score

    async def check(
        self,
        job: ForecastRequest,
        result: ForecastResponse,
        actual: float | None = None,
    ) -> None:
        dataset_id = job.dataset_id
        model_version = result.model_version

        self._ensure_dataset_state(dataset_id)
        self._ensure_model_version_state(dataset_id, model_version)
        self._prediction_counts[dataset_id] += 1

        if actual is not None and result.predictions:
            residual = float(result.predictions[0]) - actual
            self._residuals[dataset_id].append(residual)
            self._update_adwin(dataset_id, residual)
            self._collect_baseline_sample(dataset_id, model_version, residual)

        if not self._should_check(dataset_id):
            return

        await self._run_detection(dataset_id, model_version)

    def _should_check(self, dataset_id: str) -> bool:
        count = self._prediction_counts[dataset_id]
        last = self._last_check_times[dataset_id]
        elapsed_minutes = (datetime.now(tz=timezone.utc) - last).total_seconds() / 60
        check_every_n = max(1, int(self.settings.drift_check_every_n_predictions))

        return count % check_every_n == 0 or elapsed_minutes >= self.settings.drift_check_every_t_minutes

    async def _run_detection(self, dataset_id: str, model_version: str) -> None:
        self._last_check_times[dataset_id] = datetime.now(tz=timezone.utc)

        no_drift_threshold = float(self.settings.no_drift_threshold)
        major_drift_threshold = float(
            getattr(self.settings, "major_drift_threshold", self.settings.minor_drift_threshold)
        )

        # Baseline must be populated asynchronously here in Stage 5, not Stage 3.
        await self._get_or_init_baseline(dataset_id, model_version)
        
        method, score = self._get_current_drift_score(dataset_id, model_version)

        logger.info("DriftMonitor %s: drift score=%.3f (%s)", dataset_id, score, method)

        if score >= major_drift_threshold:
            await self._record_drift_event(dataset_id, method, "major", score)
            logger.warning("DriftMonitor: MAJOR drift for %s (score=%.3f)", dataset_id, score)
            await self._handle_major_drift(dataset_id, method, score)
        elif score >= no_drift_threshold:
            await self._record_drift_event(dataset_id, method, "minor", score)
            logger.info("DriftMonitor: MINOR drift for %s (score=%.3f)", dataset_id, score)
            # Minor drift does not trigger background retrain queue in the new architecture.
            # It just tags the state for inline update on the next triage.

    async def _record_drift_event(
        self,
        dataset_id: str,
        method: str,
        level: str,
        score: float,
    ) -> None:
        if self.agent_memory is None:
            return

        try:
            await self.agent_memory.record_drift_event(
                dataset_id=dataset_id,
                method=method,
                level=level,
                score=score,
            )
        except Exception as exc:
            logger.warning(
                "DriftMonitor: AgentMemory drift event write failed for %s: %s",
                dataset_id,
                exc,
            )

    def _ensure_dataset_state(self, dataset_id: str) -> None:
        if dataset_id not in self._residuals:
            self._residuals[dataset_id] = deque(maxlen=self.window_size)
            self._prediction_counts[dataset_id] = 0
            self._last_check_times[dataset_id] = datetime.now(tz=timezone.utc)
            self._adwin_detectors[dataset_id] = _build_adwin()
            self._adwin_triggered[dataset_id] = False

    def _ensure_model_version_state(self, dataset_id: str, model_version: str) -> None:
        previous_version = self._active_model_version.get(dataset_id)
        if previous_version == model_version:
            return

        if previous_version is not None:
            logger.info("DriftMonitor: resetting state for %s (%s -> %s)", dataset_id, previous_version, model_version)

        self._active_model_version[dataset_id] = model_version
        self._residuals[dataset_id] = deque(maxlen=self.window_size)
        self._prediction_counts[dataset_id] = 0
        self._last_check_times[dataset_id] = datetime.now(tz=timezone.utc)
        self._adwin_detectors[dataset_id] = _build_adwin()
        self._adwin_triggered[dataset_id] = False

        key = self._baseline_key(dataset_id, model_version)
        self._baseline_bootstrap[key] = deque(maxlen=self.cusum_baseline_min_samples)

    def _update_adwin(self, dataset_id: str, residual: float) -> None:
        if not np.isfinite(residual):
            return
        detector = self._adwin_detectors.setdefault(dataset_id, _build_adwin())
        if detector:
            detector.update(float(residual))
            if bool(getattr(detector, "drift_detected", False)):
                self._adwin_triggered[dataset_id] = True

    def _adwin_score(self, dataset_id: str, residuals: deque[float]) -> float:
        detector = self._adwin_detectors.get(dataset_id)
        if not detector:
            return 0.0

        drift_now = bool(getattr(detector, "drift_detected", False))
        drift_latched = self._adwin_triggered.get(dataset_id, False)
        if not drift_now and not drift_latched:
            return 0.0

        self._adwin_triggered[dataset_id] = False
        return _adwin_severity_from_window(residuals)

    def _baseline_key(self, dataset_id: str, model_version: str) -> str:
        return f"{dataset_id}:{model_version}"

    def _baseline_store_key(self, dataset_id: str, model_version: str) -> str:
        return f"cusum:baseline:{dataset_id}:{model_version}"

    def _collect_baseline_sample(self, dataset_id: str, model_version: str, residual: float) -> None:
        key = self._baseline_key(dataset_id, model_version)
        if key in self._baseline_params:
            return

        bootstrap = self._baseline_bootstrap.setdefault(key, deque(maxlen=self.cusum_baseline_min_samples))
        if np.isfinite(residual):
            bootstrap.append(float(residual))

    async def _get_or_init_baseline(self, dataset_id: str, model_version: str) -> dict[str, float] | None:
        key = self._baseline_key(dataset_id, model_version)
        cached = self._baseline_params.get(key)
        if cached is not None:
            return cached

        store_key = self._baseline_store_key(dataset_id, model_version)
        try:
            raw = await self.valkey.get(store_key)
            if raw:
                parsed = json.loads(raw)
                if parsed.get("k", 0.0) > 0 and parsed.get("sigma", 0.0) > 0:
                    self._baseline_params[key] = parsed
                    return parsed
        except Exception:
            pass

        bootstrap = self._baseline_bootstrap.setdefault(key, deque(maxlen=self.cusum_baseline_min_samples))
        if len(bootstrap) < self.cusum_baseline_min_samples:
            return None

        arr = np.array([float(x) for x in bootstrap if np.isfinite(x)], dtype=float)
        sigma = max(float(arr.std()), 1e-8)
        params = {
            "k": sigma * self.cusum_k_sigma,
            "h": max(sigma * self.cusum_h_sigma, 1e-8),
            "sigma": sigma
        }

        self._baseline_params[key] = params
        try:
            await self.valkey.set(store_key, json.dumps(params))
        except Exception as exc:
            logger.warning("DriftMonitor: failed to persist baseline: %s", exc)

        return params

    async def _handle_major_drift(self, dataset_id: str, method: str, score: float) -> None:
        lock_key = f"retrain_lock:{dataset_id}"
        if await self.valkey.exists(lock_key):
            logger.debug("DriftMonitor: retrain already queued for %s", dataset_id)
            return

        await self.valkey.setex(lock_key, self.settings.retrain_lock_ttl_seconds, "1")
        
        # In final arch, always pushing to retrain:jobs stream for background agent
        await self.valkey.xadd(
            "retrain:jobs",
            {
                "dataset_id": dataset_id,
                "reason": f"{method}_major",
                "triggered_at": datetime.now(tz=timezone.utc).isoformat(),
            },
        )
        logger.info("DriftMonitor: published background retrain job for %s", dataset_id)
