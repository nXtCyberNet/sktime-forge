from __future__ import annotations

import json
import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


class DataRegistry:
    """
    Manages available datastream metadata using Valkey.
    This replaces hardcoded or filesystem-derived dataset discovery.
    """

    def __init__(self, valkey_client):
        self.valkey = valkey_client
        self.prefix = "dataset:meta:"
        self._unknown_frequency = "unknown"

    async def register_dataset(
        self,
        dataset_id: str,
        description: str,
        frequency: str = "unknown",
    ) -> bool:
        """Register or update one dataset metadata record."""
        key = f"{self.prefix}{dataset_id}"
        payload: dict[str, str] = {
            "description": description,
            "frequency": frequency,
        }

        try:
            await self.valkey.set(key, json.dumps(payload))
            logger.info("Registered dataset %s in Valkey registry", dataset_id)
            return True
        except Exception as exc:
            logger.error("Failed to register dataset %s: %s", dataset_id, exc)
            return False

    async def get_all_metadata(self) -> Dict[str, str]:
        """
        Return dataset_id -> description for LLM routing prompts.

        Frequency is appended when available so intent parsing can align user
        language with available sources.
        """
        records = await self.get_all_records()
        datasets_info: dict[str, str] = {}

        for dataset_id, record in records.items():
            description = str(record.get("description") or "No description provided.")
            frequency = str(record.get("frequency") or self._unknown_frequency)
            if frequency.lower() != self._unknown_frequency:
                datasets_info[dataset_id] = f"{description} [frequency={frequency}]"
            else:
                datasets_info[dataset_id] = description
        return datasets_info

    async def get_all_records(self) -> Dict[str, Dict[str, str]]:
        """Return dataset_id -> metadata record with description + frequency."""
        raw_records = await self._scan_registry_payloads()
        records: dict[str, dict[str, str]] = {}

        for dataset_id, payload in raw_records.items():
            description = self._to_text(payload.get("description"), "No description provided.")
            frequency = self._to_text(payload.get("frequency"), self._unknown_frequency)
            records[dataset_id] = {
                "description": description,
                "frequency": frequency,
            }

        return records

    async def _scan_registry_payloads(self) -> Dict[str, Dict[str, Any]]:
        if self.valkey is None:
            return {}

        records: dict[str, dict[str, Any]] = {}
        cursor: int | str | bytes = 0
        try:
            while True:
                cursor, keys = await self.valkey.scan(
                    cursor=cursor,
                    match=f"{self.prefix}*",
                )
                for key in keys:
                    key_text = key.decode("utf-8") if isinstance(key, bytes) else str(key)
                    dataset_id = key_text[len(self.prefix):]
                    data_raw = await self.valkey.get(key)
                    if not data_raw:
                        continue

                    data_text = data_raw.decode("utf-8") if isinstance(data_raw, bytes) else str(data_raw)
                    parsed = json.loads(data_text)
                    if isinstance(parsed, dict):
                        records[dataset_id] = parsed

                if cursor in (0, "0", b"0"):
                    break
        except Exception as exc:
            logger.error("Error fetching dataset metadata from Valkey: %s", exc)
            return {}

        return records

    @staticmethod
    def _to_text(value: Any, default: str) -> str:
        if value is None:
            return default
        text = str(value).strip()
        return text if text else default
