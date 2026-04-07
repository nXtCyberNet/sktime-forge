import numpy as np
from typing import Dict, Any, List


# Model complexity tiers by algorithmic scaling
_COMPLEXITY_TIERS: Dict[str, List[str]] = {
    "O(1)":      ["NaiveForecaster"],
    "O(N)":      ["ThetaForecaster", "ExponentialSmoothing", "PolynomialTrendForecaster"],
    "O(N log N)": ["Prophet", "TBATS", "BATS"],
    "O(N^3)":    ["AutoARIMA", "AutoETS"],
    "DL":        ["LSTMForecaster", "Transformers"],
}


def get_model_complexity_budget_tool(dataset_id: str, y: np.ndarray) -> Dict[str, Any]:
    n = len(y)

    o1 = _COMPLEXITY_TIERS["O(1)"]
    on = _COMPLEXITY_TIERS["O(N)"]
    on_log_n = _COMPLEXITY_TIERS["O(N log N)"]
    on3 = _COMPLEXITY_TIERS["O(N^3)"]
    dl = _COMPLEXITY_TIERS["DL"]

    if n < 30:
        permitted = o1 + on
        forbidden = on_log_n + on3 + dl
        reason = (
            f"n={n} is severely constrained. Models with large parameter spaces "
            "will overfit with near-certainty. Only O(1) and O(N) models are safe."
        )

    elif n < 200:
        permitted = o1 + on + on_log_n + on3
        forbidden = dl
        reason = (
            f"n={n} is sufficient for classical statistical models. "
            "Deep learning models require far more data to generalise — "
            "they are likely to memorise noise at this scale."
        )

    elif n < 5000:
        permitted = o1 + on + on_log_n + on3 + dl
        forbidden = []
        reason = (
            f"n={n} falls in the optimal band for all model tiers. "
            "No complexity restrictions apply."
        )

    else:
        permitted = o1 + on + on_log_n + dl
        forbidden = on3
        reason = (
            f"n={n} is too large for O(N^3) exact solvers like AutoARIMA. "
            "Likelihood maximisation at this scale will exceed compute budgets. "
            "Use O(N log N) models or deep learning."
        )

    return {
        "dataset_size": n,
        "permitted_models": permitted,
        "forbidden_models": forbidden,
        "complexity_tiers": {tier: models for tier, models in _COMPLEXITY_TIERS.items()},
        "reason": reason,
        # No next_action_hint — the LLM decides whether to call estimate_training_cost
        # on one model, all permitted models, or proceed directly to pipeline composition.
    }