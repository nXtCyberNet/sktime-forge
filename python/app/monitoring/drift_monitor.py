from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any

from app.schemas.api import ForecastRequest, ForecastResponse


class DriftMonitor:
	"""Statistical drift monitor with lightweight heuristics and stream trigger."""

	def __init__(self, valkey: Any, settings: Any):
		self.valkey = valkey
		self.settings = settings
		self._prediction_counts = defaultdict(int)
		self._last_check_at: dict[str, datetime] = {}

	async def check(self, job: ForecastRequest, result: ForecastResponse) -> None:
		if not self._should_check(job.dataset_id):
			return

		drift_score = self._compute_drift_score(result.predictions)

		if drift_score >= self.settings.major_drift_threshold:
			await self._maybe_trigger_retrain(job.dataset_id, reason="ADWIN", score=drift_score)
		elif drift_score >= self.settings.minor_drift_threshold:
			await self._maybe_trigger_retrain(job.dataset_id, reason="CUSUM", score=drift_score)

	def _should_check(self, dataset_id: str) -> bool:
		now = datetime.now(timezone.utc)
		self._prediction_counts[dataset_id] += 1

		last_check = self._last_check_at.get(dataset_id)
		count_trigger = (
			self._prediction_counts[dataset_id] >= self.settings.drift_check_every_n_predictions
		)
		time_trigger = (
			last_check is None
			or (now - last_check) >= timedelta(minutes=self.settings.drift_check_every_n_minutes)
		)

		if count_trigger or time_trigger:
			self._prediction_counts[dataset_id] = 0
			self._last_check_at[dataset_id] = now
			return True
		return False

	def _compute_drift_score(self, predictions: list[float]) -> float:
		if not predictions:
			return 0.0
		span = max(predictions) - min(predictions)
		if span <= 0:
			return 0.0
		normalized = min(span / max(abs(predictions[0]), 1.0), 1.0)
		return round(normalized, 4)

	async def _maybe_trigger_retrain(self, dataset_id: str, reason: str, score: float) -> None:
		lock_key = f"retrain_lock:{dataset_id}"
		if await self.valkey.exists(lock_key):
			return

		await self.valkey.setex(lock_key, self.settings.retrain_lock_ttl_seconds, "1")
		await self.valkey.xadd(
			"retrain:jobs",
			{
				"dataset_id": dataset_id,
				"reason": reason,
				"drift_score": score,
				"triggered_at": datetime.now(timezone.utc).isoformat(),
			},
		)
