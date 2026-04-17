from __future__ import annotations

import json
import logging
from typing import Any

import httpx

from app.registry.registry import allowed_for_profile, validate_pipeline_spec
from app.schemas import DataProfile

logger = logging.getLogger(__name__)

_PROFILE_KEY   = "profile:{dataset_id}"
_CANDIDATE_KEY = "candidates:{dataset_id}"
_CANDIDATE_TTL = 3600  # seconds


class ModelSelectorAgent:
    """
    Parameters
    ----------
    valkey        : async Valkey/Redis client
    mlflow_client : synchronous MlflowClient (for querying only)
    mcp_client    : app.mcp.client.MCPClient
    settings      : app.config.Settings
    """

    def __init__(self, valkey, mlflow_client, mcp_client, settings):
        self.valkey   = valkey
        self.mlflow   = mlflow_client
        self.mcp      = mcp_client
        self.settings = settings

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    async def select(self, job) -> list[str]:
        """
        Main entry point called by the orchestrator.

        Parameters
        ----------
        job : ForecastRequest or dict — must have a .dataset_id attribute
              or ["dataset_id"] key.

        Returns
        -------
        List of estimator class names in preference order, e.g.
        ["AutoARIMA", "ExponentialSmoothing", "NaiveForecaster"]
        """
        dataset_id: str = (
            job.dataset_id if hasattr(job, "dataset_id") else job["dataset_id"]
        )
        logger.info("ModelSelectorAgent.select: starting for dataset_id=%s", dataset_id)

        # ---- 1. Load DataProfile from Valkey ----
        profile: DataProfile = await self._load_profile(dataset_id)
        profile_allowed = allowed_for_profile(profile)
        if not profile_allowed:
            profile_allowed = ["NaiveForecaster"]

        # ---- 2. Enrich with MLflow history ----
        mlflow_context = self._fetch_mlflow_context(dataset_id)

        # ---- 3. Ask the LLM ----
        raw_candidates: list[str] = await self._llm_select(profile, mlflow_context)

        # ---- 4. Constrain and validate against registry + profile budget ----
        allowed_set = set(profile_allowed)
        candidates: list[str] = []
        for estimator in raw_candidates:
            if estimator in allowed_set and estimator not in candidates:
                candidates.append(estimator)

        if not candidates:
            candidates = profile_allowed[:1]
            logger.warning(
                "ModelSelectorAgent: LLM returned no registry-valid candidates for %s; "
                "falling back to %s",
                dataset_id, candidates,
            )

        spec = {"estimators": candidates}
        if not validate_pipeline_spec(spec, registry=profile_allowed):
            logger.warning(
                "ModelSelectorAgent: candidate spec failed validation for %s; "
                "falling back to %s",
                dataset_id,
                profile_allowed[:1],
            )
            candidates = profile_allowed[:1]

        # ---- 5. Persist to Valkey ----
        key = _CANDIDATE_KEY.format(dataset_id=dataset_id)
        await self.valkey.setex(key, _CANDIDATE_TTL, json.dumps(candidates))
        logger.info(
            "ModelSelectorAgent: wrote %d candidates for %s → %s",
            len(candidates), dataset_id, candidates,
        )

        return candidates

    # ------------------------------------------------------------------
    # LLM selection
    # ------------------------------------------------------------------

    async def _llm_select(
        self,
        profile: DataProfile,
        mlflow_context: dict[str, Any] | None = None,
    ) -> list[str]:
        """
        Send the full DataProfile to the configured LLM and parse a ranked
        list of estimator class names from the response.

        On any error (transport, parse, etc.) falls back gracefully to the
        full permitted list from the complexity budget.
        """
        system_prompt = "\n".join([
            "You are a time-series model selection expert embedded in an automated ML pipeline.",
            "Your sole output must be a JSON array of estimator class names, ranked from most",
            "preferred to least preferred. Do not include any explanation, markdown, or text",
            "outside the JSON array.",
            "",
            "Rules you must follow:",
            "1. Only recommend estimators from the permitted_models list in the complexity budget.",
            "2. Never recommend an estimator that appears in failed_estimators with failure_count > 1",
            "   unless all permitted alternatives have also failed.",
            "3. If the series has a structural break (break_detected=true), prefer models that handle",
            "   changepoints natively (Prophet) or are robust to level shifts (NaiveForecaster,",
            "   ExponentialSmoothing) over ARIMA-family models.",
            "4. If the series is non-stationary (is_stationary=false), prefer models that do not",
            "   require stationarity (Prophet, ExponentialSmoothing, NaiveForecaster) unless",
            "   AutoARIMA is permitted and no structural break is present.",
            "5. If seasonality is strong (seasonality_class=strong), prefer models that model",
            "   seasonality explicitly (Prophet, TBATS, ExponentialSmoothing with seasonal_periods,",
            "   AutoARIMA with seasonal=True).",
            "6. Always include at least one simple baseline (NaiveForecaster or",
            "   PolynomialTrendForecaster) at the end of the list as a last-resort fallback.",
            "7. Return between 2 and 5 estimators.",
        ])

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
                profile.dataset_id, exc,
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
            return profile.complexity_budget.get("permitted_models", ["NaiveForecaster"])

    async def _request_llm_text(
        self,
        system_prompt: str,
        user_message: str,
        timeout_seconds: float | None = None,
    ) -> str:
        """Fire the HTTP request and return the raw text content from the response."""
        url, headers, payload = self._build_llm_request(system_prompt, user_message)
        resolved_timeout = float(
            timeout_seconds
            if timeout_seconds is not None
            else getattr(self.settings, "llm_timeout_seconds", 30.0)
        )

        async with httpx.AsyncClient(timeout=resolved_timeout) as client:
            response = await client.post(url, headers=headers, json=payload)
            # raise_for_status() must be called INSIDE the async-with block
            # while the response object is still open.
            response.raise_for_status()
            body = response.json()

        return self._extract_text_from_response(body).strip()

    def _build_llm_request(
        self,
        system_prompt: str,
        user_message: str,
    ) -> tuple[str, dict[str, str], dict[str, Any]]:
        """
        Build (url, headers, payload) for the configured LLM provider.

        Supports two provider modes:
        - "anthropic"        → Anthropic Messages API format
        - "openai_compatible" (default) → OpenAI Chat Completions format
        """
        provider   = str(getattr(self.settings, "llm_provider",   "openai_compatible")).strip().lower()
        model      = str(getattr(self.settings, "llm_model",       "gpt-4o-mini")).strip()
        max_tokens = int(getattr(self.settings, "llm_max_tokens",  256))
        api_key    = str(
            getattr(self.settings, "llm_api_key", "")
            or getattr(self.settings, "anthropic_api_key", "")
        ).strip()

        if provider == "anthropic":
            url = (
                str(getattr(self.settings, "llm_api_url", "")).strip()
                or "https://api.anthropic.com/v1/messages"
            )
            version = str(getattr(self.settings, "llm_api_version", "2023-06-01")).strip()
            headers: dict[str, str] = {
                "Content-Type":      "application/json",
                "anthropic-version": version,
            }
            if api_key:
                headers["x-api-key"] = api_key

            payload: dict[str, Any] = {
                "model":      model,
                "max_tokens": max_tokens,
                "system":     system_prompt,
                "messages":   [{"role": "user", "content": user_message}],
            }
            return url, headers, payload

        # ---- OpenAI-compatible (default) ----
        url = (
            str(getattr(self.settings, "llm_api_url", "")).strip()
            or "https://api.openai.com/v1/chat/completions"
        )
        headers = {"Content-Type": "application/json"}
        auth_header = str(getattr(self.settings, "llm_auth_header", "Authorization")).strip()
        auth_scheme = str(getattr(self.settings, "llm_auth_scheme", "Bearer")).strip()

        if api_key and auth_header:
            auth_value = f"{auth_scheme} {api_key}".strip() if auth_scheme else api_key
            headers[auth_header] = auth_value

        extra_headers_json = str(getattr(self.settings, "llm_extra_headers_json", "")).strip()
        if extra_headers_json:
            try:
                extra = json.loads(extra_headers_json)
                if isinstance(extra, dict):
                    headers.update({str(k): str(v) for k, v in extra.items()})
            except json.JSONDecodeError:
                logger.warning("ModelSelectorAgent: invalid llm_extra_headers_json; ignoring")

        payload = {
            "model":       model,
            "max_tokens":  max_tokens,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_message},
            ],
        }
        return url, headers, payload

    def _extract_text_from_response(self, body: dict[str, Any]) -> str:
        """
        Extract the text string from a provider response body.

        Handles three shapes:
        - OpenAI:    {"choices": [{"message": {"content": "..."}}]}
        - Anthropic: {"content": [{"type": "text", "text": "..."}]}
        - Fallback:  {"output_text": "..."} or {"text": "..."}
        """
        logger.error(f"DEBUG: LLM raw body: {json.dumps(body)}")
        # OpenAI-compatible
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
                    
        # If we failed to parse but choices was present, log everything
        logger.error(f"DEBUG LLM choices block failed. choices={choices}")

        # Anthropic-style
        text = self._content_to_text(body.get("content"))
        if text:
            return text

        # Generic fallbacks
        for key in ("output_text", "text"):
            raw = body.get(key)
            if isinstance(raw, str) and raw.strip():
                return raw

        raise ValueError(f"Unsupported LLM response format: no text content found in body keys {list(body)}")

    @staticmethod
    def _content_to_text(content: Any) -> str:
        """Collapse a content field (str or list-of-blocks) to a plain string."""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for block in content:
                if isinstance(block, str):
                    parts.append(block)
                elif isinstance(block, dict):
                    text = block.get("text")
                    if isinstance(text, str):
                        parts.append(text)
            return "\n".join(p for p in parts if p)
        return ""

    @staticmethod
    def _parse_candidate_response(raw_text: str) -> list[str]:
        """
        Parse the LLM response into a list of estimator class name strings.

        Accepts:
        - A bare JSON array:  ["AutoARIMA", "Prophet", "NaiveForecaster"]
        - A JSON object with a known list key (candidates / ranked / estimators / models)
        """
        parsed = json.loads(raw_text)

        if isinstance(parsed, list):
            return [str(c) for c in parsed]

        if isinstance(parsed, dict):
            for key in ("candidates", "ranked", "estimators", "models"):
                value = parsed.get(key)
                if isinstance(value, list):
                    return [str(c) for c in value]

        raise ValueError(
            f"Expected a JSON array of estimator class names, got: {type(parsed).__name__}"
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _load_profile(self, dataset_id: str) -> DataProfile:
        """Load the DataProfile written by PipelineArchitectAgent from Valkey."""
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
        """
        Query the MLflow registry for previously registered model versions on
        this dataset.

        Returns
        -------
        dict with keys:
            registered_versions : list of {version, run_id, estimator, val_mae, status}
            best_mae            : float | None  — best validation MAE seen so far
        """
        try:
            model_name = f"ts-forecaster-{dataset_id}"
            versions = self.mlflow.search_model_versions(f"name='{model_name}'")
            if not versions:
                # Backward-compatible fallback for older registries that relied on tags.
                versions = self.mlflow.search_model_versions(
                    f"tags.dataset_id='{dataset_id}'"
                )

            parsed: list[dict[str, Any]] = []
            best_mae: float | None = None

            for v in versions:
                run  = self.mlflow.get_run(v.run_id)
                mae  = run.data.metrics.get("val_mae")
                parsed.append({
                    "version":   v.version,
                    "run_id":    v.run_id,
                    "estimator": run.data.tags.get("estimator", "unknown"),
                    "val_mae":   mae,
                    "status":    v.status,
                })
                if mae is not None and (best_mae is None or mae < best_mae):
                    best_mae = mae

            return {"registered_versions": parsed, "best_mae": best_mae}

        except Exception as exc:
            logger.warning(
                "ModelSelectorAgent: MLflow query failed for %s: %s", dataset_id, exc
            )
            return {"registered_versions": [], "best_mae": None}