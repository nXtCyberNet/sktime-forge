import numpy as np
from typing import Dict, Any, List, Optional


def _empty_result() -> Dict[str, Any]:
    return {
        "period": None,
        "strength": 0.0,
        "seasonality_class": "none",
        "confidence": "low",
        "candidates": [],
        "method": "autocorrelation_detrended_fft",
        "interpretation": "No significant seasonal pattern detected at any lag.",
        "suggested_next_tools": ["check_structural_break", "run_stationarity_test"],
    }


def detect_seasonality_tool(
    dataset_id: str,
    y: np.ndarray,
    freq: Optional[str] = None,  # e.g. "D", "W", "M", "H"
) -> Dict[str, Any]:
    n = len(y)
    if n < 5:
        return _empty_result()

    # ---------- 1. Adaptive detrending ----------
    # Use differenced series only if differencing actually reduces variance
    y_diff = np.diff(y)
    if len(y_diff) > 0 and np.var(y_diff) < np.var(y):
        y_used = y_diff
    else:
        y_used = y

    n_used = len(y_used)
    if np.var(y_used) == 0:
        return _empty_result()

    y_norm = y_used - np.mean(y_used)

    # ---------- 2. FFT-based ACF ----------
    f = np.fft.fft(y_norm, n=2 * n_used)
    acf_full = np.fft.ifft(f * np.conjugate(f))[:n_used].real
    acf = acf_full / acf_full[0]

    # ---------- 3. Statistical significance threshold ----------
    conf = 1.96 / np.sqrt(n_used)
    threshold = max(0.2, conf)

    # ---------- 4. Peak detection (local maxima above threshold) ----------
    min_lag = 3
    max_lag = n_used // 2
    peaks = []

    for i in range(min_lag, max_lag):
        if acf[i] > threshold and acf[i] > acf[i - 1] and acf[i] > acf[i + 1]:
            peaks.append((i, acf[i]))

    if not peaks:
        return _empty_result()

    # ---------- 5. Frequency-aware scoring (bias toward known periods) ----------
    freq_map = {"D": [7, 30], "W": [52], "M": [12], "H": [24]}
    if freq in freq_map:
        preferred = set(freq_map[freq])
        peaks = sorted(peaks, key=lambda p: p[1] + (0.1 if p[0] in preferred else 0), reverse=True)
    else:
        peaks = sorted(peaks, key=lambda p: p[1], reverse=True)

    best_lag, best_corr = peaks[0]
    strength = float(np.clip(best_corr, 0.0, 1.0))

    # ---------- 6. Confidence ----------
    if len(peaks) >= 3 and strength > 0.6:
        confidence = "high"
    elif strength > 0.4:
        confidence = "medium"
    else:
        confidence = "low"

    # ---------- 7. Classification ----------
    if strength > 0.6:
        seasonality_class = "strong"
        interpretation = (
            f"Strong seasonal pattern detected at period={best_lag} (strength={strength:.3f}). "
            "Deseasonalization is likely beneficial before fitting. "
            "Consider whether the break (if any) predates or postdates the seasonal pattern."
        )
    elif strength > 0.3:
        seasonality_class = "weak"
        interpretation = (
            f"Weak seasonal signal at period={best_lag} (strength={strength:.3f}). "
            "Seasonality-aware models (e.g., Prophet, ExponentialSmoothing with seasonal_periods) "
            "may help, but deseasonalization may not be necessary."
        )
    else:
        seasonality_class = "none"
        interpretation = (
            f"No meaningful seasonal pattern found (best strength={strength:.3f}). "
            "Non-seasonal models are appropriate unless domain knowledge suggests otherwise."
        )

    candidates = [
        {"period": int(lag), "correlation": round(float(corr), 3)}
        for lag, corr in peaks[:3]
    ]

    return {
        "period": int(best_lag),
        "strength": round(strength, 3),
        "seasonality_class": seasonality_class,
        "confidence": confidence,
        "candidates": candidates,
        "method": "autocorrelation_detrended_fft",
        "interpretation": interpretation,
        # Two equally valid follow-ups — LLM chooses based on what it already knows
        "suggested_next_tools": ["check_structural_break", "run_stationarity_test"],
    }