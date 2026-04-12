import asyncio
import json
import logging
import os
import socket
from datetime import datetime, timezone
from typing import Any

from app.agents.model_selector import ModelSelectorAgent
from app.agents.pipeline_architect import PipelineArchitectAgent
from app.agents.prediction import PredictionAgent
from app.agents.training import TrainingAgent
from app.agents.watchdog import Watchdog
from app.contracts import AgentMemoryProtocol, WatchdogProtocol
from app.memory.memory import AgentMemory
from app.monitoring.drift_monitor import DriftMonitor
from app.schemas import ForecastRequest, ForecastResponse

logger = logging.getLogger(__name__)


def _decode_redis_value(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, str):
        return value
    return str(value)


class Orchestrator:
    def __init__(
        self,
        valkey,
        mlflow_client,
        mcp_client,
        settings,
        agent_memory: AgentMemoryProtocol | None = None,
        watchdog: WatchdogProtocol | None = None,
    ):
        self.valkey = valkey
        self.mlflow = mlflow_client
        self.mcp = mcp_client
        self.settings = settings
        self.model_cache = {}

        self.agent_memory: AgentMemoryProtocol = agent_memory or AgentMemory(valkey)
        self.watchdog: WatchdogProtocol = watchdog or Watchdog(valkey, settings)

        self._forecast_stream = str(
            getattr(settings, "forecast_stream_name", "forecast:jobs")
        )
        self._retrain_stream = str(
            getattr(settings, "retrain_stream_name", "retrain:jobs")
        )
        self._forecast_group = str(
            getattr(settings, "forecast_consumer_group", "python-forecast-workers")
        )
        self._retrain_group = str(
            getattr(settings, "retrain_consumer_group", "python-retrain-workers")
        )
        self._consumer_name = str(
            getattr(
                settings,
                "stream_consumer_name",
                f"py-{socket.gethostname()}-{os.getpid()}",
            )
        )
        self._stream_read_count = int(getattr(settings, "stream_read_count", 1))
        self._stream_block_ms = int(getattr(settings, "stream_block_ms", 5000))
        self._result_key_prefix = str(getattr(settings, "result_key_prefix", "result:"))
        self._result_ttl_seconds = int(getattr(settings, "result_ttl_seconds", 60))
        self._enable_stream_workers = bool(getattr(settings, "enable_stream_workers", True))

        self._stop_event = asyncio.Event()
        self._worker_tasks: list[asyncio.Task] = []

        self.pipeline_architect = PipelineArchitectAgent(valkey, mcp_client, settings)
        self.model_selector = ModelSelectorAgent(valkey, mlflow_client, mcp_client, settings)
        self.training_agent = TrainingAgent(valkey, mlflow_client, settings)
        self.prediction_agent = PredictionAgent(valkey, mlflow_client, settings)
        self.drift_monitor = DriftMonitor(
            valkey,
            settings,
            agent_memory=self.agent_memory,
        )

    async def handle_job(self, job: ForecastRequest) -> ForecastResponse:
        dataset_id = job.dataset_id

        await self._maybe_reload_model(dataset_id)

        model_version = await self._get_cached_model_version(dataset_id)
        if not model_version:
            logger.info("Orchestrator: cold start flow for dataset_id=%s", dataset_id)
            model_version = await self._run_training_cycle(
                dataset_id=dataset_id,
                reason="cold_start",
                selector_input=job,
            )

        if not model_version:
            raise RuntimeError(
                f"Orchestrator: unable to resolve/train model version for dataset_id={dataset_id}"
            )

        result = await self.prediction_agent.predict(
            job,
            model_version=model_version,
            model_cache=self.model_cache,
        )

        await self._enrich_response_rationale(job, result)

        await self._record_prediction_memory(job, result)

        actual_value = getattr(job, "actual", None)
        if actual_value is not None and result.predictions:
            asyncio.create_task(
                self._safe_watchdog_residual(
                    dataset_id=dataset_id,
                    model_version=result.model_version,
                    predicted=float(result.predictions[0]),
                    actual=float(actual_value),
                )
            )

        # Non-blocking drift check; prediction response should not wait on this.
        asyncio.create_task(
            self.drift_monitor.check(
                job,
                result,
                actual=actual_value,
            )
        )
        return result

    async def _enrich_response_rationale(
        self,
        job: ForecastRequest,
        result: ForecastResponse,
    ) -> None:
        fallback = self._build_deterministic_rationale(job, result)
        result.llm_rationale = fallback

        if not bool(getattr(self.settings, "enable_llm_rationale", True)):
            return

        has_api_key = bool(
            str(getattr(self.settings, "llm_api_key", "") or "").strip()
            or str(getattr(self.settings, "anthropic_api_key", "") or "").strip()
        )
        if not has_api_key:
            return

        try:
            memory = await self.agent_memory.get_dataset_memory(job.dataset_id)
        except Exception:
            memory = {}

        system_prompt = "\n".join([
            "You are a forecasting assistant.",
            "Write a concise rationale in 2-4 sentences.",
            "Explain model choice confidence, expected behavior, and any risk flags.",
            "Do not output markdown or JSON.",
        ])

        evidence = {
            "dataset_id": job.dataset_id,
            "model_class": result.model_class,
            "model_version": result.model_version,
            "horizon_steps": len(result.predictions),
            "prediction_head": result.predictions[:3],
            "prediction_intervals_available": result.prediction_intervals is not None,
            "drift_method": result.drift_method,
            "drift_score": result.drift_score,
            "warning": result.warning,
            "memory_summary": {
                "last_model_history_entry": (memory.get("model_history") or [None])[-1],
                "last_drift_event": (memory.get("drift_events") or [None])[-1],
            },
        }
        user_prompt = (
            "Provide a user-facing rationale for this forecast:\n"
            f"{json.dumps(evidence, ensure_ascii=True)}"
        )

        try:
            rationale_timeout = float(
                getattr(self.settings, "llm_rationale_timeout_seconds", 6.0)
            )
            rich = await self.model_selector._request_llm_text(
                system_prompt,
                user_prompt,
                timeout_seconds=rationale_timeout,
            )
            if rich and rich.strip():
                result.llm_rationale = rich.strip()
        except Exception as exc:
            logger.debug(
                "Orchestrator: LLM rationale generation failed for %s: %s",
                job.dataset_id,
                exc,
            )

    @staticmethod
    def _build_deterministic_rationale(
        job: ForecastRequest,
        result: ForecastResponse,
    ) -> str:
        head = result.predictions[:3]
        prediction_preview = ", ".join(f"{v:.3f}" for v in head) if head else "n/a"
        interval_note = (
            "Prediction intervals are included to show forecast uncertainty."
            if result.prediction_intervals
            else "Prediction intervals are unavailable for this model artifact."
        )
        drift_note = (
            f"Drift monitor signal: {result.drift_method}={result.drift_score:.3f}."
            if result.drift_score is not None and result.drift_method
            else "No active drift signal is attached to this response."
        )
        return (
            f"Forecast generated for dataset {job.dataset_id} using {result.model_class} "
            f"(version {result.model_version}) over {len(result.predictions)} horizon steps. "
            f"First predictions: {prediction_preview}. {interval_note} {drift_note}"
        )

    async def _run_training_cycle(
        self,
        dataset_id: str,
        reason: str,
        selector_input: Any,
    ) -> str | None:
        try:
            await self.pipeline_architect.construct_pipeline(dataset_id)
            await self.model_selector.select(selector_input)
            model_version = await self.training_agent.handle_retrain_job(
                {"dataset_id": dataset_id, "reason": reason}
            )
        except Exception as exc:
            await self._safe_record_model_failure(
                dataset_id=dataset_id,
                estimator="unknown",
                reason=f"{reason}: {exc}",
            )
            raise

        if not model_version:
            await self._safe_record_model_failure(
                dataset_id=dataset_id,
                estimator="unknown",
                reason=f"{reason}: no_model_version",
            )
            return None

        await self._post_promotion_hooks(dataset_id, model_version)
        return model_version

    async def _post_promotion_hooks(self, dataset_id: str, model_version: str) -> None:
        summary = self.training_agent.get_last_training_summary(dataset_id) or {}
        estimator_name = str(summary.get("estimator_name", "unknown"))

        val_mae_raw = summary.get("val_mae")
        summary_version = summary.get("model_version") or model_version

        await self._safe_update_dataset_memory(
            dataset_id,
            {
                "last_promoted_version": str(summary_version),
                "last_retrain_reason": "promotion",
                "last_retrain_at": datetime.now(tz=timezone.utc).isoformat(),
            },
        )

        if val_mae_raw is None:
            return

        try:
            baseline_score = float(val_mae_raw)
        except (TypeError, ValueError):
            return

        await self._safe_record_model_promotion(
            dataset_id=dataset_id,
            estimator=estimator_name,
            version=str(summary_version),
            val_mae=baseline_score,
        )

        asyncio.create_task(
            self._safe_watchdog_monitor(
                dataset_id=dataset_id,
                baseline_score=baseline_score,
                model_version=str(summary_version),
            )
        )

    async def _record_prediction_memory(
        self,
        job: ForecastRequest,
        result: ForecastResponse,
    ) -> None:
        prediction_head = result.predictions[0] if result.predictions else None
        await self._safe_update_dataset_memory(
            job.dataset_id,
            {
                "last_forecast_at": datetime.now(tz=timezone.utc).isoformat(),
                "last_model_version": result.model_version,
                "last_correlation_id": result.correlation_id,
                "last_horizon": len(result.predictions),
                "last_prediction": float(prediction_head) if prediction_head is not None else None,
            },
        )

    async def _safe_update_dataset_memory(
        self,
        dataset_id: str,
        updates: dict[str, Any],
    ) -> None:
        try:
            await self.agent_memory.update_dataset_memory(dataset_id, updates)
        except Exception as exc:
            logger.warning("Orchestrator: AgentMemory update failed for %s: %s", dataset_id, exc)

    async def _safe_record_model_promotion(
        self,
        dataset_id: str,
        estimator: str,
        version: str,
        val_mae: float,
    ) -> None:
        try:
            await self.agent_memory.record_model_promotion(
                dataset_id=dataset_id,
                estimator=estimator,
                version=version,
                val_mae=val_mae,
            )
        except Exception as exc:
            logger.warning("Orchestrator: AgentMemory promotion write failed for %s: %s", dataset_id, exc)

    async def _safe_record_model_failure(
        self,
        dataset_id: str,
        estimator: str,
        reason: str,
    ) -> None:
        try:
            await self.agent_memory.record_model_failure(
                dataset_id=dataset_id,
                estimator=estimator,
                reason=reason,
            )
        except Exception as exc:
            logger.warning("Orchestrator: AgentMemory failure write failed for %s: %s", dataset_id, exc)

    async def _safe_watchdog_monitor(
        self,
        dataset_id: str,
        baseline_score: float,
        model_version: str | None = None,
    ) -> None:
        try:
            await self.watchdog.monitor_post_promotion(
                dataset_id=dataset_id,
                baseline_score=baseline_score,
                model_version=model_version,
            )
        except Exception as exc:
            logger.warning("Orchestrator: Watchdog monitor failed for %s: %s", dataset_id, exc)

    async def _safe_watchdog_residual(
        self,
        dataset_id: str,
        model_version: str,
        predicted: float,
        actual: float,
    ) -> None:
        try:
            await self.watchdog.record_residual(
                dataset_id=dataset_id,
                model_version=model_version,
                predicted=predicted,
                actual=actual,
            )
        except Exception as exc:
            logger.warning("Orchestrator: Watchdog residual write failed for %s: %s", dataset_id, exc)

    async def _maybe_reload_model(self, dataset_id: str) -> None:
        signal_key = f"model_updated:{dataset_id}"
        try:
            signal = await self.valkey.get(signal_key)
            if not signal:
                return

            stale_keys = [key for key in self.model_cache if key[0] == dataset_id]
            for key in stale_keys:
                self.model_cache.pop(key, None)

            await self.valkey.delete(signal_key)
            logger.info(
                "Orchestrator: model_updated signal processed for %s (invalidated %d cached entries)",
                dataset_id,
                len(stale_keys),
            )
        except Exception as exc:
            logger.warning("Orchestrator: failed to process model_updated signal for %s: %s", dataset_id, exc)

    async def _get_cached_model_version(self, dataset_id: str) -> str | None:
        key = f"model_version:{dataset_id}"
        try:
            raw = await self.valkey.get(key)
            if raw:
                return raw.decode() if isinstance(raw, bytes) else str(raw)
        except Exception as exc:
            logger.warning("Orchestrator: failed reading model version for %s: %s", dataset_id, exc)

        # Fall back to MLflow registry when the Valkey key is missing.
        try:
            model_name = f"ts-forecaster-{dataset_id}"
            versions = self.mlflow.get_latest_versions(
                model_name,
                stages=["Production", "Staging", "None"],
            )
            if versions:
                latest = sorted(versions, key=lambda v: int(v.version), reverse=True)[0]
                return str(latest.version)
        except Exception as exc:
            logger.warning("Orchestrator: failed MLflow version fallback for %s: %s", dataset_id, exc)

        return None

    async def startup_cleanup(self) -> None:
        patterns = ("model_lock:*", "retrain_lock:*")

        for pattern in patterns:
            try:
                async for key in self.valkey.scan_iter(pattern):
                    ttl = await self.valkey.ttl(key)
                    if ttl is not None and int(ttl) < 0:
                        await self.valkey.delete(key)
                        logger.info("Orchestrator: removed orphaned lock %s", key)
            except Exception as exc:
                logger.warning("Orchestrator: startup cleanup failed for pattern %s: %s", pattern, exc)

    # ------------------------------------------------------------------
    # Stream workers (forecast:jobs / retrain:jobs)
    # ------------------------------------------------------------------

    async def start_stream_workers(
        self,
        include_forecast: bool = True,
        include_retrain: bool = True,
    ) -> None:
        if not self._enable_stream_workers:
            logger.info("Orchestrator: stream workers are disabled via settings")
            return

        if self._worker_tasks:
            return

        if not include_forecast and not include_retrain:
            logger.info("Orchestrator: no stream workers requested to start")
            return

        if include_forecast:
            await self._ensure_consumer_group(self._forecast_stream, self._forecast_group)
        if include_retrain:
            await self._ensure_consumer_group(self._retrain_stream, self._retrain_group)

        self._stop_event.clear()
        tasks: list[asyncio.Task] = []
        if include_forecast:
            tasks.append(
                asyncio.create_task(
                    self._forecast_worker_loop(),
                    name="forecast-stream-worker",
                )
            )
        if include_retrain:
            tasks.append(
                asyncio.create_task(
                    self._retrain_worker_loop(),
                    name="retrain-stream-worker",
                )
            )
        self._worker_tasks = tasks
        logger.info(
            "Orchestrator: started stream workers (consumer=%s, forecast=%s, retrain=%s)",
            self._consumer_name,
            include_forecast,
            include_retrain,
        )

    async def stop_stream_workers(self) -> None:
        if not self._worker_tasks:
            return

        self._stop_event.set()
        for task in self._worker_tasks:
            task.cancel()

        await asyncio.gather(*self._worker_tasks, return_exceptions=True)
        self._worker_tasks.clear()
        logger.info("Orchestrator: stream workers stopped")

    async def _ensure_consumer_group(self, stream: str, group: str) -> None:
        try:
            await self.valkey.xgroup_create(
                name=stream,
                groupname=group,
                id="0",
                mkstream=True,
            )
            logger.info("Orchestrator: created consumer group %s for stream %s", group, stream)
        except Exception as exc:
            # BUSYGROUP means the group already exists.
            if "BUSYGROUP" not in str(exc):
                raise

    async def _forecast_worker_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                batches = await self.valkey.xreadgroup(
                    groupname=self._forecast_group,
                    consumername=self._consumer_name,
                    streams={self._forecast_stream: ">"},
                    count=self._stream_read_count,
                    block=self._stream_block_ms,
                )
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.warning("Orchestrator: forecast stream read failed: %s", exc)
                await asyncio.sleep(1)
                continue

            if not batches:
                continue

            for _, messages in batches:
                for message_id, fields in messages:
                    correlation_id: str | None = None
                    try:
                        payload = {
                            _decode_redis_value(k): _decode_redis_value(v)
                            for k, v in fields.items()
                        }
                        correlation_id = payload.get("correlation_id") or payload.get("job_id")

                        request = self._build_forecast_request(payload, message_id)
                        response = await self.handle_job(request)

                        await self._publish_result(
                            request.correlation_id,
                            response.model_dump_json(),
                        )
                    except Exception as exc:
                        logger.exception(
                            "Orchestrator: failed to process forecast stream message %s: %s",
                            message_id,
                            exc,
                        )
                        if correlation_id:
                            error_payload = {
                                "error": str(exc),
                                "correlation_id": correlation_id,
                            }
                            await self._publish_result(
                                correlation_id,
                                json.dumps(error_payload),
                            )
                    finally:
                        try:
                            await self.valkey.xack(
                                self._forecast_stream,
                                self._forecast_group,
                                message_id,
                            )
                        except Exception as ack_exc:
                            logger.warning(
                                "Orchestrator: forecast xack failed for %s: %s",
                                message_id,
                                ack_exc,
                            )

    async def _retrain_worker_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                batches = await self.valkey.xreadgroup(
                    groupname=self._retrain_group,
                    consumername=self._consumer_name,
                    streams={self._retrain_stream: ">"},
                    count=self._stream_read_count,
                    block=self._stream_block_ms,
                )
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.warning("Orchestrator: retrain stream read failed: %s", exc)
                await asyncio.sleep(1)
                continue

            if not batches:
                continue

            for _, messages in batches:
                for message_id, fields in messages:
                    try:
                        payload = {
                            _decode_redis_value(k): _decode_redis_value(v)
                            for k, v in fields.items()
                        }
                        dataset_id = payload.get("dataset_id")
                        reason = payload.get("reason", "scheduled")

                        if not dataset_id:
                            raise ValueError("retrain message missing dataset_id")

                        await self._run_training_cycle(
                            dataset_id=dataset_id,
                            reason=reason,
                            selector_input={"dataset_id": dataset_id},
                        )
                    except Exception as exc:
                        logger.exception(
                            "Orchestrator: failed to process retrain stream message %s: %s",
                            message_id,
                            exc,
                        )
                    finally:
                        try:
                            await self.valkey.xack(
                                self._retrain_stream,
                                self._retrain_group,
                                message_id,
                            )
                        except Exception as ack_exc:
                            logger.warning(
                                "Orchestrator: retrain xack failed for %s: %s",
                                message_id,
                                ack_exc,
                            )

    def _build_forecast_request(self, payload: dict[str, str], message_id: str) -> ForecastRequest:
        dataset_id = payload.get("dataset_id")
        if not dataset_id:
            raise ValueError("forecast message missing dataset_id")

        correlation_id = payload.get("correlation_id") or payload.get("job_id") or _decode_redis_value(message_id)
        frequency = payload.get("frequency") or None
        fh_values = self._parse_fh(payload.get("fh", ""))
        actual_raw = payload.get("actual")
        actual_value: float | None = None
        if actual_raw not in (None, ""):
            actual_value = float(actual_raw)

        return ForecastRequest(
            dataset_id=dataset_id,
            fh=fh_values,
            correlation_id=correlation_id,
            frequency=frequency,
            actual=actual_value,
        )

    @staticmethod
    def _parse_fh(raw_fh: str) -> list[int]:
        text = raw_fh.strip()
        if not text:
            return [1]

        # Accept JSON array form or comma-separated values.
        if text.startswith("["):
            parsed = json.loads(text)
            if not isinstance(parsed, list):
                raise ValueError("fh must be a list")
            return [int(v) for v in parsed]

        values = [part.strip() for part in text.split(",") if part.strip()]
        if not values:
            raise ValueError("fh is empty")
        return [int(v) for v in values]

    async def _publish_result(self, correlation_id: str, payload: str) -> None:
        key = f"{self._result_key_prefix}{correlation_id}"
        async with self.valkey.pipeline(transaction=False) as pipe:
            await pipe.rpush(key, payload)
            await pipe.expire(key, self._result_ttl_seconds)
            await pipe.execute()