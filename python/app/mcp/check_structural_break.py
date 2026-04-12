import numpy as np
from typing import Dict, Any, List, Optional


def check_structural_break_tool(dataset_id: str, y: np.ndarray) -> Dict[str, Any]:
    n = len(y)

    if n < 10:
        return {
            "break_detected": False,
            "location": None,
            "location_fraction": None,
            "confidence": 0.0,
            "test_stat": None,
            "critical_value": None,
            "interpretation": "Fewer than 10 observations — CUSUM test is unreliable at this sample size.",
            "suggested_next_tools": ["get_model_complexity_budget"],
        }

    mean_y = np.mean(y)
    std_y = np.std(y) or 1.0

    # CUSUM: cumulative sum of mean-centered residuals
    s = np.cumsum(y - mean_y)
    max_abs_s = float(np.max(np.abs(s)))
    location = int(np.argmax(np.abs(s)))

    test_stat = max_abs_s / (std_y * np.sqrt(n))
    # Critical value scaled to penalise noise on short series
    critical_value = 1.358 * (1 + 0.1 * np.log(n))

    break_detected = bool(test_stat > critical_value)
    location_fraction = round(location / n, 3)  # e.g. 0.73 = 73% into the series

    if break_detected:
        # excess above threshold, clipped to [0, 1]
        confidence = float(min(1.0, (test_stat - critical_value) / critical_value))
        interpretation = (
            f"Structural break detected at observation {location} "
            f"({location_fraction:.0%} into the series, confidence={confidence:.3f}). "
            "A level shift here may cause ARIMA order selection to underestimate integration. "
            "Consider whether training data should be split at this point, "
            "or whether a model that handles changepoints natively (e.g. Prophet) is appropriate."
        )
    else:
        confidence = float(max(0.0, test_stat / critical_value))
        interpretation = (
            f"No structural break detected (test_stat={test_stat:.3f} < critical={critical_value:.3f}). "
            "The level of the series appears stable. Non-break non-stationarity "
            "(trend or unit root) is more likely if the stationarity test flagged an issue."
        )

    return {
        "break_detected": break_detected,
        "location": location if break_detected else None,
        "location_fraction": location_fraction if break_detected else None,
        "confidence": round(confidence, 3),
        "test_stat": round(test_stat, 3),
        "critical_value": round(critical_value, 3),
        "interpretation": interpretation,
        # Both are legitimate next steps regardless of break outcome
        "suggested_next_tools": ["get_dataset_history", "get_model_complexity_budget"],
    }