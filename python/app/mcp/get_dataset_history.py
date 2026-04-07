from typing import Dict, Any, List


def get_dataset_history_tool(dataset_id: str, memory_dict: Dict[str, Any]) -> Dict[str, Any]:
    if not memory_dict:
        return {
            "status": "cold_start",
            "production_memory_summary": (
                f"No previous production history exists for dataset '{dataset_id}'. "
                "This is a first-time run. All model choices are exploratory."
            ),
            "model_history": [],
            "drift_events": [],
            "data_characteristics": {},
            "failed_estimators": {},
        }

    model_history: List[Dict] = memory_dict.get("model_history", [])
    drift_events: List[Dict] = memory_dict.get("drift_events", [])
    data_characteristics: Dict = memory_dict.get("data_characteristics", {})

    # ---------- Summarise failures ----------
    failures = [m for m in model_history if m.get("failure_reason")]
    failed_estimators: Dict[str, int] = {}
    for f in failures:
        est = f.get("estimator", "Unknown")
        failed_estimators[est] = failed_estimators.get(est, 0) + 1

    # ---------- Build a narrative summary for LLM context ----------
    summary_parts = []

    if failed_estimators:
        for est, count in failed_estimators.items():
            noun = "failure" if count == 1 else "failures"
            summary_parts.append(f"{est} has {count} recorded {noun} on this dataset.")
    else:
        summary_parts.append("No model failures recorded — production history is stable.")

    if drift_events:
        recent = drift_events[-1]
        summary_parts.append(
            f"Most recent drift event: method={recent.get('method', 'unknown')}, "
            f"level={recent.get('level', 'unknown')}, "
            f"score={recent.get('score', 'N/A')}."
        )
    else:
        summary_parts.append("No drift events recorded.")

    return {
        "status": "history_retrieved",
        "production_memory_summary": " ".join(summary_parts),
        "model_history": model_history,
        "drift_events": drift_events,
        "data_characteristics": data_characteristics,
        # Structured failure index — more useful to the LLM than a prose list
        "failed_estimators": failed_estimators,
        # No next_action_hint — the LLM decides what evidence gap remains after
        # reading production history. It may already have enough to compose a pipeline.
    }