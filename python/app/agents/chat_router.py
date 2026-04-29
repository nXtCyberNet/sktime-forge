from __future__ import annotations

import json
import logging
from typing import Any
import uuid

import httpx

from app.schemas import ForecastRequest

logger = logging.getLogger(__name__)

class ChatRouterAgent:
    def __init__(self, settings, data_loader=None):
        self.settings = settings
        self.data_loader = data_loader

    async def route_request(
        self,
        user_query: str,
        available_datasets: dict[str, dict[str, str]],
    ) -> list[ForecastRequest]:
        logger.info("ChatRouterAgent: Translating natural language for prompt='%s'", user_query)

        datasets_formatted = "\n".join(
            [
                (
                    f"- {name}: {meta.get('description', 'No description provided.')} "
                    f"(frequency={meta.get('frequency', 'unknown')})"
                )
                for name, meta in available_datasets.items()
            ]
        )
        dataset_ids = ", ".join(sorted(available_datasets.keys()))
        
        system_prompt = (
            "You are an expert Data Scientist embedded in an automated Agentic Pipeline.\n"
            "Your task is to take a user's natural English request and map it to a structured JSON forecasting configuration.\n"
            "You must map the user's intent to one or more datasets available in the system.\n\n"
            f"AVAILABLE DATASETS IN SYSTEM:\n{datasets_formatted}\n\n"
            f"VALID DATASET IDS: {dataset_ids}\n\n"
            "INSTRUCTIONS:\n"
            "1. Output NOTHING but a valid JSON object. No markdown wrappers or backticks.\n"
            "2. Required JSON Keys:\n"
            '   - "dataset_ids": (array of strings) Pick one or more dataset ids from VALID DATASET IDS.\n'
            '   - "fh": (array of integers) The forecast horizon in steps. E.g., if they ask for "next 3 months", use [1, 2, 3].\n'
            "3. If user request is generic and multiple frequencies are relevant, include all relevant dataset ids in dataset_ids.\n"
            "4. Never invent a dataset id that is not in VALID DATASET IDS.\n"
        )

        try:
            raw_text = await self._request_llm_text(system_prompt, user_query)
            if raw_text.startswith("```json"): raw_text = raw_text[7:]
            if raw_text.startswith("```"): raw_text = raw_text[3:]
            if raw_text.endswith("```"): raw_text = raw_text[:-3]
            
            parsed = json.loads(raw_text.strip())
            
            logger.info("ChatRouterAgent: LLM parsed parameters -> %s", parsed)

            fh = parsed.get("fh")
            if not isinstance(fh, list):
                raise ValueError("LLM response must include fh as an array of positive integers")

            selected_ids_raw = parsed.get("dataset_ids")
            if selected_ids_raw is None and parsed.get("dataset_id") is not None:
                selected_ids_raw = [parsed["dataset_id"]]

            if not isinstance(selected_ids_raw, list) or not selected_ids_raw:
                raise ValueError("LLM response must include dataset_ids as a non-empty array")

            selected_ids = self._sanitize_dataset_ids(
                selected_ids_raw,
                available_datasets,
            )
            if not selected_ids:
                raise ValueError("LLM selected dataset ids that are not present in metadata")

            requests: list[ForecastRequest] = []
            for dataset_id in selected_ids:
                metadata = available_datasets.get(dataset_id, {})
                frequency = metadata.get("frequency")
                frequency_hint = str(frequency).strip() if frequency is not None else None
                if frequency_hint and frequency_hint.lower() == "unknown":
                    frequency_hint = None

                requests.append(
                    ForecastRequest(
                        dataset_id=dataset_id,
                        fh=fh,
                        correlation_id=str(uuid.uuid4()),
                        frequency=frequency_hint,
                    )
                )

            return requests
            
        except Exception as exc:
            logger.error("ChatRouterAgent failed to translate request: %s", exc)
            raise ValueError(f"ChatRouter Agent failed to understand the request based on available datasets: {exc}")

    @staticmethod
    def _sanitize_dataset_ids(
        selected_ids_raw: list[Any],
        available_datasets: dict[str, dict[str, str]],
    ) -> list[str]:
        seen: set[str] = set()
        selected: list[str] = []
        valid_ids = set(available_datasets.keys())

        for raw in selected_ids_raw:
            dataset_id = str(raw).strip()
            if not dataset_id:
                continue
            if dataset_id not in valid_ids:
                continue
            if dataset_id in seen:
                continue
            seen.add(dataset_id)
            selected.append(dataset_id)
        return selected

    async def _request_llm_text(
        self,
        system_prompt: str,
        user_message: str,
    ) -> str:
        """Call the LLM using the application's global configuration."""
        url, headers, payload = self._build_llm_request(system_prompt, user_message)
        timeout = float(getattr(self.settings, "llm_timeout_seconds", 30.0))

        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            body = response.json()

        return self._extract_text_from_response(body).strip()

    def _build_llm_request(self, system_prompt: str, user_message: str) -> tuple[str, dict[str, str], dict[str, Any]]:
        provider = str(getattr(self.settings, "llm_provider", "openai_compatible")).strip().lower()
        model = str(getattr(self.settings, "llm_model", "gpt-4o-mini")).strip()
        max_tokens = int(getattr(self.settings, "llm_max_tokens", 256))
        api_key = str(
            getattr(self.settings, "llm_api_key", "")
            or getattr(self.settings, "anthropic_api_key", "")
        ).strip()

        if provider == "anthropic":
            url = str(getattr(self.settings, "llm_api_url", "")).strip() or "https://api.anthropic.com/v1/messages"
            version = str(getattr(self.settings, "llm_api_version", "2023-06-01")).strip()
            headers = {"Content-Type": "application/json", "anthropic-version": version}
            if api_key:
                headers["x-api-key"] = api_key

            payload = {
                "model": model,
                "max_tokens": max_tokens,
                "system": system_prompt,
                "messages": [{"role": "user", "content": user_message}],
            }
            return url, headers, payload

        # OpenAI compatible fallback
        url = str(getattr(self.settings, "llm_api_url", "")).strip() or "https://api.openai.com/v1/chat/completions"
        headers = {"Content-Type": "application/json"}
        auth_header = str(getattr(self.settings, "llm_auth_header", "Authorization")).strip()
        auth_scheme = str(getattr(self.settings, "llm_auth_scheme", "Bearer")).strip()

        if api_key and auth_header:
            auth_value = f"{auth_scheme} {api_key}".strip() if auth_scheme else api_key
            headers[auth_header] = auth_value

        payload = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
        }
        return url, headers, payload

    def _extract_text_from_response(self, body: dict[str, Any]) -> str:
        choices = body.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0]
            if isinstance(first, dict):
                message = first.get("message")
                if isinstance(message, dict):
                    content = message.get("content")
                    if isinstance(content, str):
                        return content

        content = body.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    return block.get("text", "")
        
        for key in ("output_text", "text", "content"):
            raw = body.get(key)
            if isinstance(raw, str):
                return raw
                
        raise ValueError("Unsupported format, no text content found.")
