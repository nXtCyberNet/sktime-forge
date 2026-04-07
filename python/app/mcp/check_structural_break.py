import numpy as np
from typing import Dict, Any


def check_structural_break_tool(dataset_id: str, y: np.ndarray) -> Dict[str, Any]:
    if len(y) < 10:
        return {
            "break_detected": False,
            "location": None,
            "confidence": 0.0,
            "next_action_hint": "profile_dataset"
        }

    # ---------- 1. Use original data (No differencing) ----------
    # Differencing destroys level shifts. We want to detect changes in the mean level.
    n = len(y)
    mean_y = np.mean(y)
    std_y = np.std(y) or 1.0

    # ---------- 2. CUSUM (Cumulative Sum of mean-centered data) ----------
    s = np.cumsum(y - mean_y)

    # The location of the structural break is where the cumulative sum is furthest from 0
    max_abs_s = np.max(np.abs(s))
    location = int(np.argmax(np.abs(s)))

    # ---------- 3. Test statistic ----------
    test_stat = max_abs_s / (std_y * np.sqrt(n))

    # ---------- 4. Threshold (h) ----------
    # 'h_threshold' acts as our 'h' boundary. 1.358 corresponds to 95% confidence bounds 
    # of a standard Brownian bridge. We scale it with N to prevent noise triggering.
    h_threshold = 1.358 * (1 + 0.1 * np.log(n))

    break_detected = bool(test_stat > h_threshold)

    # ---------- 5. Confidence ----------
    if break_detected:
        confidence = min(1.0, (test_stat - h_threshold) / h_threshold)
        next_action = "get_dataset_history"
    else:
        confidence = max(0.0, test_stat / h_threshold)
        next_action = "detect_seasonality"

    return {
        "break_detected": break_detected,
        "location": location,
        "confidence": round(float(confidence), 3),
        "test_stat": round(float(test_stat), 3),
        "next_action_hint": next_action
    }
