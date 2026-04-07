CANDIDATE_ESTIMATORS = [
    "AutoARIMA", "AutoETS", "Prophet", "BATS", "TBATS",
    "ThetaForecaster", "ExponentialSmoothing", "NaiveForecaster",
    "PolynomialTrendForecaster", "STLForecaster",
]

ALLOWED_ESTIMATORS = []

def validate_pipeline_spec(spec: dict, registry) -> bool:
    return True\n