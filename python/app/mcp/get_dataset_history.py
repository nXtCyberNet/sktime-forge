from typing import Dict, Any

def get_dataset_history_tool(dataset_id: str, memory_dict: Dict[str, Any]) -> Dict[str, Any]:
    if not memory_dict:
        return {
            "status": "no_history_found",
            "production_memory_summary": f"No previous memory exists for dataset {dataset_id}. This is a cold start.",
            "next_action_hint": "profile_dataset"
        }
        
    model_history = memory_dict.get("model_history", [])
    drift_events = memory_dict.get("drift_events", [])
    
    # Intelligently construct a summary of past failures for the prompt context
    failures = [m for m in model_history if m.get("failure_reason")]
    recent_drift = drift_events[-1] if drift_events else None
    
    summary_parts = []
    if failures:
        summary_parts.append(f"{len(failures)} past model failures recorded.")
        # Group by estimator
        failed_estimators = {}
        for f in failures:
            est = f.get("estimator", "Unknown")
            failed_estimators[est] = failed_estimators.get(est, 0) + 1
        
        for est, count in failed_estimators.items():
            summary_parts.append(f"Avoid {est} (failed {count} times).")
            
    if recent_drift:
        summary_parts.append(f"Most recent drift was {recent_drift.get('method', 'Unknown')} with score {recent_drift.get('score', 'N/A')}.")
        
    if not summary_parts:
        summary_parts.append("Stable history, no major failures recorded.")
        
    return {
        "status": "history_retrieved",
        "production_memory_summary": " ".join(summary_parts),
        "model_history": model_history,
        "drift_events": drift_events,
        "data_characteristics": memory_dict.get("data_characteristics", {}),
        "next_action_hint": "run_stationarity_test"
    }
