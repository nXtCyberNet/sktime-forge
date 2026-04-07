import numpy as np
from typing import Dict, Any


def estimate_training_cost_tool(
    dataset_id: str,
    y: np.ndarray,
    model_class: str,
    seasonality_period: int = 1,
) -> Dict[str, Any]:
    n = len(y)
    base_overhead_minutes = 0.01  # container startup + import cost
    cost_per_minute_usd = 0.0005

    # ---------- Seasonality penalty factor ----------
    sp_penalty = 1.0
    seasonal_model = "Naive" not in model_class and "Polynomial" not in model_class

    if seasonality_period > 1 and seasonal_model:
        if "ARIMA" in model_class:
            sp_penalty = 4.0 + (seasonality_period * 0.05)
        elif "ETS" in model_class:
            sp_penalty = 3.0 + (seasonality_period * 0.04)
        elif "Exponential" in model_class:
            sp_penalty = 2.0 + (seasonality_period * 0.02)
        elif "Prophet" in model_class or "BATS" in model_class or "Theta" in model_class:
            sp_penalty = 1.5 + (seasonality_period * 0.01)
        else:
            sp_penalty = 1.2

    # ---------- Compute minutes ----------
    if "Naive" in model_class or "Polynomial" in model_class:
        fit_minutes = n * 0.00001  # sp_penalty intentionally excluded

    elif "Theta" in model_class:
        fit_minutes = (n * 0.00005) * sp_penalty

    elif "Exponential" in model_class:
        fit_minutes = (n * 0.00005) * sp_penalty

    elif "Prophet" in model_class or "BATS" in model_class:
        fit_minutes = (n * np.log(n + 1) * 0.0001) * sp_penalty

    elif "ARIMA" in model_class or "ETS" in model_class:
        fit_minutes = ((n / 100) ** 3) * 0.0015 * sp_penalty
        fit_minutes = max(fit_minutes, 0.1)  # floor: likelihood maximisation always costs something

    else:
        fit_minutes = (n * 0.001) * sp_penalty

    total_minutes = base_overhead_minutes + fit_minutes
    cost_usd = total_minutes * cost_per_minute_usd

    return {
        "estimated_minutes": round(float(total_minutes), 3),
        "estimated_cost_usd": round(float(cost_usd), 5),
        "model_class": model_class,
        "dataset_size": n,
        "seasonality_period": seasonality_period,
        "seasonality_penalty_factor": round(sp_penalty, 3),
        # No recommendation, no hint. The SLA rule lives in the agent system prompt.
        # e.g. "If estimated_minutes > 60, you must select a simpler model."
    }