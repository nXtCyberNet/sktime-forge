from __future__ import annotations

import json
import logging
from typing import Any

import httpx

from app.registry.registry import allowed_for_profile, validate_pipeline_spec
from app.schemas import DataProfile
from app.mcp.tools import MCP_TOOL_SCHEMAS, dispatch_tool


logger = logging.getLogger(__name__)

_PROFILE_KEY   = "profile:{dataset_id}"
_CANDIDATE_KEY = "candidates:{dataset_id}"
_CANDIDATE_TTL = 3600  # seconds


class ModelSelectorAgent:
    """
    Parameters
    ----------
    valkey         : async Valkey/Redis client
    mlflow_client : synchronous MlflowClient (for querying only)
    mcp_client    : app.mcp.client.MCPClient
    settings      : app.config.Settings
    """

    def __init__(self, valkey, mlflow_client, mcp_client, settings):
        self.valkey   = valkey
        self.mlflow   = mlflow_client
        self.mcp      = mcp_client
        self.settings = settings

    async def select(self, job) -> list[str]:
        dataset_id: str = (
            job.dataset_id if hasattr(job, "dataset_id") else job["dataset_id"]
        )
        logger.info("ModelSelectorAgent.select: starting for dataset_id=%s", dataset_id)

        profile: DataProfile = await self._load_profile(dataset_id)
        profile_allowed = allowed_for_profile(profile)
        if not profile_allowed:
            profile_allowed = ["NaiveForecaster"]

        mlflow_context = self._fetch_mlflow_context(dataset_id)

        raw_candidates: list[str] = await self._llm_select(profile, mlflow_context)

        from sktime.registry import all_estimators
        valid_sktime_forecasters = {name for name, _ in all_estimators(estimator_types="forecaster")}

        allowed_set = set(profile_allowed)
        candidates: list[str] = []
        for estimator in raw_candidates:
            if estimator in allowed_set and estimator in valid_sktime_forecasters and estimator not in candidates:
                candidates.append(estimator)

        if not candidates:
            candidates = profile_allowed[:1]
            logger.warning(
                "ModelSelectorAgent: LLM returned no registry-valid candidates for %s; fallback to %s",
                dataset_id, candidates,
            )

        spec = {"estimators": candidates}
        if not validate_pipeline_spec(spec, registry=profile_allowed):
            logger.warning("ModelSelectorAgent: candidate spec failed validation for %s", dataset_id)
            candidates = profile_allowed[:1]

        key = _CANDIDATE_KEY.format(dataset_id=dataset_id)
        await self.valkey.setex(key, _CANDIDATE_TTL, json.dumps(candidates))
        return candidates

    async def _llm_select(
        self,
        profile: DataProfile,
        mlflow_context: dict[str, Any] | None = None,
    ) -> list[str]:
        system_prompt = "\n".join([
            "You are a time-series model selection expert. You have access to tools via MCP.",
            "Your final output MUST be a JSON array of estimator class names, ranked by preference.",
            "Rules:",
            "1. Only recommend models from the complexity budget.",
            "2. Use tools to investigate dataset history or specifics if needed.",
            "3. Always include a simple baseline (NaiveForecaster) as a last resort.",
            "4. Return ONLY the JSON array in your final response (no explanation)."
        ])

        evidence = {
            "dataset_id":        profile.dataset_id,
            "n_observations":    profile.n_observations,
            "narrative":         profile.narrative,
            "stationarity":      profile.stationarity,
            "seasonality":       profile.seasonality,
            "structural_break":  profile.structural_break,
            "complexity_budget": profile.complexity_budget,
            "mlflow_context":    mlflow_context or {},
        }

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Select and rank estimators:\n{json.dumps(evidence)}"}
        ]

        try:
            
            for _ in range(5):
                response_data = await self._request_llm_raw(messages, use_tools=True)
                
                tool_calls = response_data.get("tool_calls")
                
                if not tool_calls:
                    content = response_data.get("content") or ""
                    return self._parse_candidate_response(content)

                messages.append({"role": "assistant", "tool_calls": tool_calls})
                
                for tc in tool_calls:
                    func_name = tc["function"]["name"]
                    func_args = json.loads(tc["function"]["arguments"])
                    
                    logger.info("ModelSelectorAgent: executing MCP tool %s", func_name)
                    result = dispatch_tool(self.mcp, func_name, func_args)
                    
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "name": func_name,
                        "content": json.dumps(result)
                    })
            
            logger.error("ModelSelectorAgent: MCP loop exceeded max iterations")
            return profile.complexity_budget.get("permitted_models", ["NaiveForecaster"])

        except Exception as exc:
            logger.error("ModelSelectorAgent._llm_select failure: %s", exc)
            return profile.complexity_budget.get("permitted_models", ["NaiveForecaster"])

    async def _request_llm_raw(
        self, 
        messages: list[dict[str, Any]], 
        use_tools: bool = False
    ) -> dict[str, Any]:
        url, headers, payload = self._build_llm_request_v2(messages, use_tools)
        timeout = float(getattr(self.settings, "llm_timeout_seconds", 30.0))

        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            body = response.json()
            
        
        return self._extract_message_object(body)

    def _build_llm_request_v2(
        self, 
        messages: list[dict[str, Any]], 
        use_tools: bool
    ) -> tuple[str, dict[str, str], dict[str, Any]]:
        """Modified builder to support tool definitions and multi-turn messages."""
        provider = str(getattr(self.settings, "llm_provider", "openai_compatible")).strip().lower()
        api_key  = str(getattr(self.settings, "llm_api_key", "")).strip()
        
        url = str(getattr(self.settings, "llm_api_url", "")).strip() or "https://api.openai.com/v1/chat/completions"
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
        
        payload = {
            "model": getattr(self.settings, "llm_model", "gpt-4o-mini"),
            "messages": messages,
            "temperature": 0,
        }
        
        if use_tools:
            payload["tools"] = MCP_TOOL_SCHEMAS
            payload["tool_choice"] = "auto"

        return url, headers, payload

    def _extract_message_object(self, body: dict[str, Any]) -> dict[str, Any]:
        """Extracts the 'message' part of the response, including tool_calls."""
        choices = body.get("choices")
        if choices and isinstance(choices, list):
            return choices[0].get("message", {})
        raise ValueError(f"Unexpected LLM response format: {body}")

    @staticmethod
    def _parse_candidate_response(raw_text: str) -> list[str]:
        # Cleanup potential markdown ticks
        clean_text = raw_text.replace("```json", "").replace("```", "").strip()
        try:
            parsed = json.loads(clean_text)
            if isinstance(parsed, list):
                return [str(c) for c in parsed]
            if isinstance(parsed, dict):
                for key in ("candidates", "ranked", "estimators", "models"):
                    if isinstance(parsed.get(key), list):
                        return [str(c) for c in parsed[key]]
        except Exception:
            pass
        raise ValueError(f"Could not parse estimators from: {raw_text[:100]}")

    async def _load_profile(self, dataset_id: str) -> DataProfile:
        key = _PROFILE_KEY.format(dataset_id=dataset_id)
        raw = await self.valkey.get(key)
        if not raw:
            raise RuntimeError(f"No profile found for {dataset_id}")
        return DataProfile(**json.loads(raw))

    def _fetch_mlflow_context(self, dataset_id: str) -> dict[str, Any]:
        try:
            model_name = f"ts-forecaster-{dataset_id}"
            versions = self.mlflow.search_model_versions(f"name='{model_name}'")
            parsed = []
            best_mae = None
            for v in versions:
                run = self.mlflow.get_run(v.run_id)
                mae = run.data.metrics.get("val_mae")
                parsed.append({
                    "version": v.version,
                    "estimator": run.data.tags.get("estimator", "unknown"),
                    "val_mae": mae,
                })
                if mae is not None and (best_mae is None or mae < best_mae):
                    best_mae = mae
            return {"registered_versions": parsed, "best_mae": best_mae}
        except Exception as exc:
            logger.warning("MLflow query failed: %s", exc)
            return {"registered_versions": [], "best_mae": None}