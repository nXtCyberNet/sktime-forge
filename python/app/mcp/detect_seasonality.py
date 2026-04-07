import numpy as np
from typing import Dict, Any, List, Optional


def _empty_result() -> Dict[str, Any]:
    return {
        "period": 1,
        "strength": 0.0,
        "seasonality_class": "none",
        "confidence": "low",
        "candidates": [],
        "method": "autocorrelation_detrended_fft",
        "next_action_hint": "run_stationarity_test"
    }


def detect_seasonality_tool(
    dataset_id: str,
    y: np.ndarray,
    freq: Optional[str] = None  # e.g. "D", "W", "M"
) -> Dict[str, Any]:
    """
    Detect seasonality using FFT-based ACF with:
    - Adaptive detrending
    - Statistical significance filtering
    - Dynamic thresholds
    - Optional frequency awareness
    """

    n = len(y)
    if n < 5:
        return _empty_result()

    # ---------- 1. Adaptive detrending ----------
    y_diff = np.diff(y)

    # choose version with lower variance (less trend influence)
    if len(y_diff) > 0 and np.var(y_diff) < np.var(y):
        y_used = y_diff
    else:
        y_used = y

    n_used = len(y_used)
    variance = np.var(y_used)

    if variance == 0:
        return _empty_result()

    # normalize
    y_norm = y_used - np.mean(y_used)

    # ---------- 2. FFT-based ACF ----------
    f = np.fft.fft(y_norm, n=2 * n_used)
    acf_full = np.fft.ifft(f * np.conjugate(f))[:n_used].real
    acf = acf_full / acf_full[0]

    # ---------- 3. Statistical threshold ----------
    conf = 1.96 / np.sqrt(n_used)
    threshold = max(0.2, conf)

    # ---------- 4. Lag constraints ----------
    min_lag = 3
    max_lag = n_used // 2

    peaks = []

    for i in range(min_lag, max_lag):
        if acf[i] > threshold:
            if acf[i] > acf[i - 1] and acf[i] > acf[i + 1]:
                peaks.append((i, acf[i]))

    if not peaks:
        return _empty_result()

    # ---------- 5. Frequency awareness ----------
    # bias known seasonal periods
    freq_map = {
        "D": [7, 30],
        "W": [52],
        "M": [12],
        "H": [24]
    }

    if freq in freq_map:
        preferred = freq_map[freq]

        def score_peak(p):
            lag, corr = p
            bonus = 0.1 if lag in preferred else 0
            return corr + bonus

        peaks = sorted(peaks, key=score_peak, reverse=True)
    else:
        peaks = sorted(peaks, key=lambda x: x[1], reverse=True)

    best_lag, best_corr = peaks[0]

    # ---------- 6. Candidates ----------
    candidates = [
        {"period": int(lp), "correlation": round(float(cp), 3)}
        for lp, cp in peaks[:3]
    ]

    # ---------- 7. Strength ----------
    strength = float(np.clip(best_corr, 0.0, 1.0))

    # ---------- 8. Confidence ----------
    if len(peaks) >= 3 and strength > 0.6:
        confidence = "high"
    elif strength > 0.4:
        confidence = "medium"
    else:
        confidence = "low"

    # ---------- 9. Classification + next action ----------
    if strength > 0.6:
        seasonality_class = "strong"
        next_action = "deseasonalize_then_stationarity"
    elif strength > 0.3:
        seasonality_class = "moderate"
        next_action = "run_stationarity_test"
    else:
        seasonality_class = "weak"
        next_action = "check_structural_break"

    return {
        "period": int(best_lag),
        "strength": round(strength, 3),
        "seasonality_class": seasonality_class,
        "confidence": confidence,
        "candidates": candidates,
        "method": "autocorrelation_detrended_fft",
        "next_action_hint": next_action
    }