from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_DEFAULT_POLL_INTERVAL_S    = 30
_DEFAULT_OBSERVATION_WINDOW = 50
_DEFAULT_MONITOR_TTL_S      = 3600
_DEFAULT_DEGRADATION_THRESH = 0.25
_RETRAIN_LOCK_TTL_S         = 600

_RESIDUALS_KEY  = "watchdog:residuals:{dataset_id}:{model_version}"
_RETRAIN_LOCK   = "retrain_lock:{dataset_id}"
_RETRAIN_STREAM = "retrain:jobs"


class Watchdog:
    """
    Parameters
    ----------
    valkey   : async Valkey/Redis client
    settings : app.config.Settings
    """

    def __init__(self, valkey, settings):
        self.valkey   = valkey
        self.settings = settings

        self._poll_interval = getattr(settings, "watchdog_poll_interval_s",   _DEFAULT_POLL_INTERVAL_S)
        self._min_obs       = getattr(settings, "watchdog_min_observations",  _DEFAULT_OBSERVATION_WINDOW)
        self._ttl           = getattr(settings, "watchdog_monitor_ttl_s",     _DEFAULT_MONITOR_TTL_S)
        self._degrad_thresh = getattr(settings, "watchdog_degradation_thresh", _DEFAULT_DEGRADATION_THRESH)
        self._lock_ttl      = getattr(settings, "retrain_lock_ttl_seconds",   _RETRAIN_LOCK_TTL_S)

    async def monitor_post_promotion(
        self,
        dataset_id: str,
        baseline_score: float,
        model_version: str | None = None,
    ) -> None:
        """
        Begin post-promotion monitoring for *dataset_id*.

        Parameters
        ----------
        dataset_id     : Dataset being monitored.
        baseline_score : Validation MAE from TrainingAgent at promotion time.
        model_version  : Active model version. Resolved from Valkey if None.
        """
        if baseline_score <= 0:
            logger.error(
                "Watchdog: invalid baseline_score=%.4f for %s – monitoring aborted",
                baseline_score, dataset_id,
            )
            return

        if model_version is None:
            model_version = await self._resolve_model_version(dataset_id)
        if model_version is None:
            logger.warning(
                "Watchdog: cannot resolve model version for %s – monitoring aborted",
                dataset_id,
            )
            return

        logger.info(
            "Watchdog: starting post-promotion monitoring for %s v%s "
            "(baseline_mae=%.4f, ttl=%ds)",
            dataset_id, model_version, baseline_score, self._ttl,
        )

        loop       = asyncio.get_running_loop()
        start_time = loop.time()
        should_stop = False

        while not should_stop:
            # ---- TTL guard ----
            elapsed = loop.time() - start_time
            if elapsed >= self._ttl:
                logger.info(
                    "Watchdog: TTL expired for %s v%s after %.0f s – stopping",
                    dataset_id, model_version, elapsed,
                )
                break

            residuals = await self._fetch_residuals(dataset_id, model_version)
            n_obs     = len(residuals)

            if n_obs < self._min_obs:
                logger.debug(
                    "Watchdog: %s v%s — waiting for minimum observations (%d/%d)",
                    dataset_id, model_version, n_obs, self._min_obs,
                )
                await asyncio.sleep(self._poll_interval)
                continue

            live_mae           = float(np.mean(np.abs(residuals)))
            degradation_ratio  = (live_mae - baseline_score) / max(baseline_score, 1e-8)

            logger.info(
                "Watchdog: %s v%s — live_mae=%.4f baseline_mae=%.4f "
                "degradation=%.1f%% (n=%d)",
                dataset_id, model_version,
                live_mae, baseline_score,
                degradation_ratio * 100,
                n_obs,
            )

            if degradation_ratio > self._degrad_thresh:
                logger.warning(
                    "Watchdog: DEGRADATION DETECTED for %s v%s — "
                    "live_mae=%.4f exceeds baseline=%.4f by %.1f%% "
                    "(threshold=%.1f%%)",
                    dataset_id, model_version,
                    live_mae, baseline_score,
                    degradation_ratio * 100,
                    self._degrad_thresh * 100,
                )
                queued = await self._queue_retrain(
                    dataset_id=dataset_id,
                    model_version=model_version,
                    live_mae=live_mae,
                    baseline_score=baseline_score,
                    degradation_ratio=degradation_ratio,
                )
                if queued:
                    should_stop = True
                    continue
            else:
                logger.debug(
                    "Watchdog: %s v%s — accuracy within threshold "
                    "(degradation=%.1f%%)",
                    dataset_id, model_version, degradation_ratio * 100,
                )

            await asyncio.sleep(self._poll_interval)

        logger.info(
            "Watchdog: monitoring complete for %s v%s", dataset_id, model_version
        )

    async def record_residual(
        self,
        dataset_id: str,
        model_version: str,
        predicted: float,
        actual: float,
    ) -> None:
        """
        Append a single residual (predicted − actual) to the Watchdog's
        Valkey list.

        Called by the orchestrator each time an actual value becomes
        available for the most recent prediction.

        The list is capped at 500 entries to bound memory usage.
        """
        residual = float(predicted) - float(actual)
        key      = _RESIDUALS_KEY.format(
            dataset_id=dataset_id, model_version=model_version
        )
        try:
            async with self.valkey.pipeline(transaction=False) as pipe:
                await pipe.rpush(key, str(residual))
                await pipe.ltrim(key, -500, -1)
                await pipe.expire(key, self._ttl)
                await pipe.execute()
        except Exception as exc:
            logger.warning(
                "Watchdog.record_residual: Valkey write failed for %s: %s",
                dataset_id, exc,
            )

    async def _fetch_residuals(
        self, dataset_id: str, model_version: str
    ) -> np.ndarray:
        """Retrieve all stored residuals for this (dataset_id, model_version)."""
        key = _RESIDUALS_KEY.format(
            dataset_id=dataset_id, model_version=model_version
        )
        try:
            raw_list = await self.valkey.lrange(key, 0, -1)
            if not raw_list:
                return np.array([], dtype=float)

            values: list[float] = []
            for item in raw_list:
                try:
                    v = float(item.decode() if isinstance(item, bytes) else item)
                    if np.isfinite(v):
                        values.append(v)
                except (ValueError, TypeError):
                    continue
            return np.array(values, dtype=float)

        except Exception as exc:
            logger.warning(
                "Watchdog: Valkey residual fetch failed for %s: %s",
                dataset_id, exc,
            )
            return np.array([], dtype=float)

    async def _queue_retrain(
        self,
        dataset_id: str,
        model_version: str,
        live_mae: float,
        baseline_score: float,
        degradation_ratio: float,
    ) -> bool:
        """
        Publish a retrain job to the shared "retrain:jobs" stream.

        Uses the same retrain_lock key as DriftMonitor to prevent
        double-queuing from the two monitors.

        Returns
        -------
        True  — retrain is queued or already in progress
        False — queue publication failed; monitoring should continue
        """
        lock_key = _RETRAIN_LOCK.format(dataset_id=dataset_id)

        try:
            already_locked = await self.valkey.exists(lock_key)
            if already_locked:
                logger.info(
                    "Watchdog: retrain already queued for %s (lock present) – "
                    "stopping monitoring",
                    dataset_id,
                )
                return True

            await self.valkey.setex(lock_key, self._lock_ttl, "1")

            await self.valkey.xadd(
                _RETRAIN_STREAM,
                {
                    "dataset_id":        dataset_id,
                    "model_version":     model_version,
                    "reason":            "watchdog_degradation",
                    "live_mae":          str(round(live_mae, 6)),
                    "baseline_mae":      str(round(baseline_score, 6)),
                    "degradation_ratio": str(round(degradation_ratio, 4)),
                    "triggered_at":      datetime.now(tz=timezone.utc).isoformat(),
                },
            )
            logger.info(
                "Watchdog: queued retrain job for %s "
                "(live_mae=%.4f vs baseline=%.4f)",
                dataset_id, live_mae, baseline_score,
            )
            return True

        except Exception as exc:
            logger.error(
                "Watchdog: failed to queue retrain for %s: %s", dataset_id, exc
            )
            return False

    async def _resolve_model_version(self, dataset_id: str) -> str | None:
        """Read the active model version set by TrainingAgent in Valkey."""
        key = f"model_version:{dataset_id}"
        try:
            raw = await self.valkey.get(key)
            if raw:
                return raw.decode() if isinstance(raw, bytes) else raw
        except Exception as exc:
            logger.warning(
                "Watchdog: cannot read model version for %s: %s",
                dataset_id, exc,
            )
        return None