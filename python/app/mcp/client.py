import numpy as np
from typing import Dict, Any, List, Optional

from .check_structural_break import check_structural_break_tool
from .run_stationarity_test import run_stationarity_test_tool
from .detect_seasonality import detect_seasonality_tool
from .get_model_complexity_budget import get_model_complexity_budget_tool
from .estimate_training_cost import estimate_training_cost_tool
from .get_dataset_history import get_dataset_history_tool


class MCPClient:
    def __init__(self, data_loader=None, memory_loader=None):
        # data_loader: callable (dataset_id: str) -> np.ndarray
        self.data_loader = data_loader
        # memory_loader: callable (dataset_id: str) -> Dict[str, Any]
        self.memory_loader = memory_loader

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_data(self, dataset_id: str) -> np.ndarray:
        if self.data_loader:
            return self.data_loader(dataset_id)
        # Safe deterministic mock for unit tests — not random so tests are repeatable
        rng = np.random.default_rng(seed=42)
        return rng.standard_normal(100)

    def _get_memory(self, dataset_id: str) -> Dict[str, Any]:
        if self.memory_loader:
            return self.memory_loader(dataset_id)
        return {}

    # ------------------------------------------------------------------
    # profile_dataset — full diagnostic dashboard, single call
    #
    # The PipelineArchitectAgent calls this first to get the complete
    # picture in one shot. Individual tools exist for targeted drill-down
    # when the agent needs to investigate a specific dimension more deeply.
    # ------------------------------------------------------------------

    def profile_dataset(self, dataset_id: str, freq: Optional[str] = None) -> Dict[str, Any]:
        y = self._get_data(dataset_id)

        stationarity = run_stationarity_test_tool(dataset_id, y)
        seasonality = detect_seasonality_tool(dataset_id, y, freq=freq)
        structural_break = check_structural_break_tool(dataset_id, y)
        complexity_budget = get_model_complexity_budget_tool(dataset_id, y)

        # Build a single plain-English narrative the LLM can reason from directly
        narrative_parts = [
            f"Dataset '{dataset_id}' contains {len(y)} observations.",
            f"Stationarity: {stationarity['conclusion']} "
            f"(ADF p={stationarity['adf_pvalue']}, KPSS p={stationarity['kpss_pvalue']}).",
            f"Seasonality: {seasonality['seasonality_class']} "
            f"(period={seasonality['period']}, strength={seasonality['strength']}, "
            f"confidence={seasonality['confidence']}).",
            (
                f"Structural break: DETECTED at observation {structural_break['location']} "
                f"({structural_break['location_fraction']:.0%} into series, "
                f"confidence={structural_break['confidence']})."
                if structural_break["break_detected"]
                else "Structural break: NOT detected."
            ),
            f"Permitted model tiers for this dataset size: {', '.join(complexity_budget['permitted_models'])}.",
        ]

        return {
            "dataset_id": dataset_id,
            "n_observations": len(y),
            "narrative": " ".join(narrative_parts),
            # Full sub-results preserved — agent can inspect any field
            "stationarity": stationarity,
            "seasonality": seasonality,
            "structural_break": structural_break,
            "complexity_budget": complexity_budget,
            # No suggested_next_tools here — after profile_dataset the LLM has the
            # full picture and should decide autonomously what to investigate further.
        }

    # ------------------------------------------------------------------
    # Individual tools — for targeted drill-down after profile_dataset
    # ------------------------------------------------------------------

    def get_dataset_history(self, dataset_id: str) -> Dict[str, Any]:
        memory_dict = self._get_memory(dataset_id)
        return get_dataset_history_tool(dataset_id, memory_dict)

    def run_stationarity_test(self, dataset_id: str) -> Dict[str, Any]:
        y = self._get_data(dataset_id)
        return run_stationarity_test_tool(dataset_id, y)

    def check_structural_break(self, dataset_id: str) -> Dict[str, Any]:
        y = self._get_data(dataset_id)
        return check_structural_break_tool(dataset_id, y)

    def detect_seasonality(self, dataset_id: str, freq: Optional[str] = None) -> Dict[str, Any]:
        y = self._get_data(dataset_id)
        return detect_seasonality_tool(dataset_id, y, freq=freq)

    def get_model_complexity_budget(self, dataset_id: str) -> Dict[str, Any]:
        y = self._get_data(dataset_id)
        return get_model_complexity_budget_tool(dataset_id, y)

    def estimate_training_cost(
        self,
        dataset_id: str,
        model_class: str,
        seasonality_period: int = 1,
    ) -> Dict[str, Any]:
        y = self._get_data(dataset_id)
        return estimate_training_cost_tool(dataset_id, y, model_class, seasonality_period)