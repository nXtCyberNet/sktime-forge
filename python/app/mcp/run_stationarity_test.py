import warnings
import numpy as np
from typing import Dict, Any, List
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tools.sm_exceptions import InterpolationWarning


def run_stationarity_test_tool(dataset_id: str, y: np.ndarray) -> Dict[str, Any]:
    if len(y) < 10:
        return {
            "adf_pvalue": None,
            "kpss_pvalue": None,
            "is_stationary": False,
            "conclusion": "insufficient_data",
            "interpretation": "Fewer than 10 observations — ADF and KPSS are unreliable at this sample size.",
            "suggested_next_tools": ["get_model_complexity_budget"],
        }

    # ADF: H0 = non-stationary (unit root). p < 0.05 → reject H0 → stationary.
    try:
        adf_result = adfuller(y, autolag="AIC")
        adf_pval = float(adf_result[1])
    except Exception:
        adf_pval = 1.0  # conservative: assume non-stationary on failure

    # KPSS: H0 = stationary. p < 0.05 → reject H0 → non-stationary.
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", InterpolationWarning)
            kpss_result = kpss(y, regression="c", nlags="auto")
            kpss_pval = float(kpss_result[1])
    except Exception:
        kpss_pval = 0.0  # conservative: assume non-stationary on failure

    adf_stationary = adf_pval < 0.05
    kpss_stationary = kpss_pval >= 0.05

    if adf_stationary and kpss_stationary:
        conclusion = "strictly_stationary"
        is_stationary = True
        interpretation = (
            "Both ADF and KPSS agree the series is stationary. "
            "No differencing or detrending is required."
        )
        # Structurally stationary → seasonality is the natural next investigation;
        # but structural breaks can still exist even in a stationary series.
        suggested_next_tools = ["detect_seasonality", "check_structural_break"]

    elif not adf_stationary and not kpss_stationary:
        conclusion = "strictly_non_stationary"
        is_stationary = False
        interpretation = (
            "Both ADF and KPSS agree the series is non-stationary. "
            "Differencing is likely needed, but a structural break can mimic a unit root — "
            "consider checking for breaks before differencing blindly."
        )
        suggested_next_tools = ["check_structural_break", "detect_seasonality"]

    elif not adf_stationary and kpss_stationary:
        conclusion = "trend_stationary"
        is_stationary = False
        interpretation = (
            "KPSS does not reject stationarity, but ADF does. "
            "The series may be stationary around a deterministic trend. "
            "Detrending (not differencing) is the typical remedy."
        )
        suggested_next_tools = ["detect_seasonality", "check_structural_break"]

    else:  # adf_stationary and not kpss_stationary
        conclusion = "difference_stationary"
        is_stationary = False
        interpretation = (
            "ADF rejects a unit root, but KPSS rejects stationarity. "
            "Classic difference-stationary signal — one round of differencing should suffice. "
            "Verify no structural break is masking the result."
        )
        suggested_next_tools = ["check_structural_break", "detect_seasonality"]

    return {
        "adf_pvalue": round(adf_pval, 4),
        "kpss_pvalue": round(kpss_pval, 4),
        "is_stationary": is_stationary,
        "conclusion": conclusion,
        "interpretation": interpretation,
        "suggested_next_tools": suggested_next_tools,
    }