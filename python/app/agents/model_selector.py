from __future__ import annotations

import json
import logging
from typing import Any

import httpx

from app.schemas import DataProfile

logger = logging.getLogger(__name__)


# Valkey key helpers
_PROFILE_KEY   = "profile:{dataset_id}"
_CANDIDATE_KEY = "candidates:{dataset_id}"
_CANDIDATE_TTL = 3600  # seconds


class ModelSelectorAgent:
    def __init__(self, valkey, mlflow_client, mcp_client, settings):
        self.valkey   = valkey
        self.mlflow   = mlflow_client
        self.mcp      = mcp_client
        self.settings = settings

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    async def select(self, job) -> list[str]:
        dataset_id: str = job.dataset_id if hasattr(job, "dataset_id") else job["dataset_id"]
        logger.info("ModelSelectorAgent.select: starting for dataset_id=%s", dataset_id)

        # ---- 1. Load DataProfile from Valkey ----
        profile: DataProfile = await self._load_profile(dataset_id)

        # ---- 2. Enrich with MLflow history ----
        mlflow_context = self._fetch_mlflow_context(dataset_id)

        # ---- 3. Ask the LLM ----
        raw_candidates: list[str] = await self._llm_select(profile, mlflow_context)

        # ---- 4. Strip forbidden models (hard constraint – never delegated to LLM) ----
        forbidden: set[str] = set(profile.complexity_budget.get("forbidden_models", []))
        candidates = [m for m in raw_candidates if m not in forbidden]

        if not candidates:
            # Absolute fallback: cheapest permitted model
            permitted = profile.complexity_budget.get("permitted_models", ["NaiveForecaster"])
            candidates = permitted[:1]
            logger.warning(
                "ModelSelectorAgent: LLM returned only forbidden models for %s; "
                "falling back to %s",
                dataset_id, candidates,
            )

        # ---- 5. Persist to Valkey ----
        key = _CANDIDATE_KEY.format(dataset_id=dataset_id)
        await self.valkey.setex(key, _CANDIDATE_TTL, json.dumps(candidates))
        logger.info("ModelSelectorAgent: wrote %d candidates for %s → %s", len(candidates), dataset_id, candidates)

        return candidates

    # ------------------------------------------------------------------
    # LLM selection
    # ------------------------------------------------------------------

    async def _llm_select(
        self,
        profile: DataProfile,
        mlflow_context: dict[str, Any] | None = None,
    ) -> list[str]:
        system_prompt = (
            "You are a time-series model selection expert embedded in an automated ML pipeline. "
            "Your sole output must be a JSON array of estimator class names, ranked from most "
            "preferred to least preferred. Do not include any explanation, markdown, or text "
            "outside the JSON array."
            "Rules you must follow:"
            "1. Only recommend estimators from the permitted_models list in the complexity budget."
            "2. Never recommend an estimator that appears in failed_estimators with failure_count > 1 "
            "   unless all permitted alternatives have also failed."
            "3. If the series has a structural break (break_detected=true), prefer models that handle "
            "   changepoints natively (Prophet) or are robust to level shifts (NaiveForecaster, "
            "   ExponentialSmoothing) over ARIMA-family models."
            "4. If the series is non-stationary (is_stationary=false), prefer models that do not "
            "   require stationarity (Prophet, ExponentialSmoothing, NaiveForecaster) unless "
            "   AutoARIMA is permitted and no structural break is present."
            "5. If seasonality is strong (seasonality_class=strong), prefer models that model "
            "   seasonality explicitly (Prophet, TBATS, ExponentialSmoothing with seasonal_periods, "
            "   AutoARIMA with seasonal=True)."
            "6. Always include at least one simple baseline (NaiveForecaster or "
            "   PolynomialTrendForecaster) at the end of the list as a last-resort fallback."
            "7. Return between 2 and 5 estimators."
        )

        # Build a single, self-contained user message with all evidence
        evidence = {
            "dataset_id":        profile.dataset_id,
            "n_observations":    profile.n_observations,
            "narrative":         profile.narrative,
            "stationarity":      profile.stationarity,
            "seasonality":       profile.seasonality,
            "structural_break":  profile.structural_break,
            "complexity_budget": profile.complexity_budget,
            "dataset_history":   profile.dataset_history,
            "mlflow_context":    mlflow_context or {},
        }

        user_message = (
            "Select and rank estimators for the following dataset profile.\n\n"
            f"{json.dumps(evidence, indent=2)}\n\n"
            "Return ONLY a JSON array of estimator class names."
        )

        try:
            raw_text = await self._request_llm_text(system_prompt, user_message)
        except Exception as exc:
            logger.error(
                "ModelSelectorAgent._llm_select: LLM HTTP request failed for %s: %s",
                profile.dataset_id,
                exc,
            )
            return profile.complexity_budget.get("permitted_models", ["NaiveForecaster"])

        try:
            return self._parse_candidate_response(raw_text)
        except (json.JSONDecodeError, ValueError) as exc:
            logger.error(
                "ModelSelectorAgent._llm_select: failed to parse LLM response for %s: %s\n"
                "Raw response: %s",
                profile.dataset_id, exc, raw_text,
            )
            # Graceful degradation: return the full permitted list in complexity order
            return profile.complexity_budget.get("permitted_models", ["NaiveForecaster"])

    async def _request_llm_text(self, system_prompt: str, user_message: str) -> str:
        url, headers, payload = self._build_llm_request(system_prompt, user_message)
        timeout_seconds = float(getattr(self.settings, "llm_timeout_seconds", 30.0))

        async with httpx.AsyncClient(timeout=timeout_seconds) as client:
            response = await client.post(url, headers=headers, json=payload)

        response.raise_for_status()
        body = response.json()
        raw_text = self._extract_text_from_response(body)
        return raw_text.strip()

    def _build_llm_request(
        self,
        system_prompt: str,
        user_message: str,
    ) -> tuple[str, dict[str, str], dict[str, Any]]:
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
            headers = {
                "Content-Type": "application/json",
                "anthropic-version": version,
            }
            if api_key:
                headers["x-api-key"] = api_key

            payload = {
                "model": model,
                "max_tokens": max_tokens,
                "system": system_prompt,
                "messages": [{"role": "user", "content": user_message}],
            }
            return url, headers, payload

        # Default to OpenAI-compatible Chat Completions payload used by many providers.
        url = str(getattr(self.settings, "llm_api_url", "")).strip() or "https://api.openai.com/v1/chat/completions"
        headers = {"Content-Type": "application/json"}
        auth_header = str(getattr(self.settings, "llm_auth_header", "Authorization")).strip()
        auth_scheme = str(getattr(self.settings, "llm_auth_scheme", "Bearer")).strip()

        if api_key and auth_header:
            auth_value = f"{auth_scheme} {api_key}".strip() if auth_scheme else api_key
            headers[auth_header] = auth_value

        extra_headers_json = str(getattr(self.settings, "llm_extra_headers_json", "")).strip()
        if extra_headers_json:
            try:
                extra_headers = json.loads(extra_headers_json)
                if isinstance(extra_headers, dict):
                    headers.update({str(k): str(v) for k, v in extra_headers.items()})
            except json.JSONDecodeError:
                logger.warning("ModelSelectorAgent: invalid llm_extra_headers_json; ignoring")

        payload = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
        }
        return url, headers, payload

    def _extract_text_from_response(self, body: dict[str, Any]) -> str:
        # OpenAI-compatible: {"choices": [{"message": {"content": "..."}}]}
        choices = body.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0]
            if isinstance(first, dict):
                message = first.get("message")
                if isinstance(message, dict):
                    text = self._content_to_text(message.get("content"))
                    if text:
                        return text
                text = first.get("text")
                if isinstance(text, str) and text.strip():
                    return text

        # Anthropic-style: {"content": [{"type": "text", "text": "..."}]}
        text = self._content_to_text(body.get("content"))
        if text:
            return text

        output_text = body.get("output_text")
        if isinstance(output_text, str) and output_text.strip():
            return output_text

        raw_text = body.get("text")
        if isinstance(raw_text, str) and raw_text.strip():
            return raw_text

        raise ValueError("Unsupported LLM response format: no text content found")

    @staticmethod
    def _content_to_text(content: Any) -> str:
        if isinstance(content, str):
            return content

        if isinstance(content, list):
            parts: list[str] = []
            for block in content:
                if isinstance(block, str):
                    parts.append(block)
                    continue
                if isinstance(block, dict):
                    text = block.get("text")
                    if isinstance(text, str):
                        parts.append(text)
            return "\n".join(part for part in parts if part)

        return ""

    @staticmethod
    def _parse_candidate_response(raw_text: str) -> list[str]:
        parsed = json.loads(raw_text)

        if isinstance(parsed, list):
            return [str(candidate) for candidate in parsed]

        if isinstance(parsed, dict):
            for key in ("candidates", "ranked", "estimators", "models"):
                value = parsed.get(key)
                if isinstance(value, list):
                    return [str(candidate) for candidate in value]

        raise ValueError("Expected a JSON array of estimator class names")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _load_profile(self, dataset_id: str) -> DataProfile:
        """Load DataProfile written by PipelineArchitectAgent from Valkey."""
        key = _PROFILE_KEY.format(dataset_id=dataset_id)
        raw = await self.valkey.get(key)
        if not raw:
            raise RuntimeError(
                f"ModelSelectorAgent: no profile found in Valkey for dataset_id={dataset_id}. "
                "Ensure PipelineArchitectAgent ran successfully first."
            )
        data = json.loads(raw)
        return DataProfile(**data)

    def _fetch_mlflow_context(self, dataset_id: str) -> dict[str, Any]:
        try:
            versions = self.mlflow.search_model_versions(f"tags.dataset_id='{dataset_id}'")
            parsed = []
            best_mae = None

            for v in versions:
                run = self.mlflow.get_run(v.run_id)
                mae = run.data.metrics.get("val_mae")
                parsed.append({
                    "version":    v.version,
                    "run_id":     v.run_id,
                    "estimator":  run.data.tags.get("estimator", "unknown"),
                    "val_mae":    mae,
                    "status":     v.status,
                })
                if mae is not None and (best_mae is None or mae < best_mae):
                    best_mae = mae

            return {"registered_versions": parsed, "best_mae": best_mae}

        except Exception as exc:
            logger.warning("ModelSelectorAgent: MLflow query failed for %s: %s", dataset_id, exc)
            return {"registered_versions": [], "best_mae": None}