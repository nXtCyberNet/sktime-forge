# sktime-forge Test Log

This document tracks the validation and testing of the sktime agentic architecture.

## ✅ Completed Tests

### 1. Variance Telemetry & M4 Monthly Test
- **Objective:** Fix the `0.0` variance bug in the `profile_dataset` MCP tool and validate LLM model selection logic.
- **Dataset:** `m4_monthly_subset_real` / `m4_monthly_variance_fix`
- **Result:** **Passed**. Detected that the MCP node was failing to calculate variance. Post-fix, the system surfaced a variance of `~3,199,452`, allowing the agent to break out of its baseline extrapolation and choose more sophisticated estimators (e.g., `AutoETS`). 

### 2. Seasonal Parameter (`sp`) Injection
- **Objective:** Fix crashes where native sktime periodic models (`AutoARIMA`, `AutoETS`, `ExponentialSmoothing`, `TBATS`) were instantiated without their required `sp` (seasonal periods) argument, causing default-fallback flatlines or crashes.
- **Dataset:** M4 Monthly sets.
- **Result:** **Passed**. Wrote injection logic in `TrainingAgent._instantiate_estimator` to extract the `seasonality.period` from the DataProfile and dynamically inject it into the `kwargs` of the target estimators. Models now train natively and successfully compete for the lowest CV score.

### 3. Graceful Degradation on Sparse/Zero-Inflated Data
- **Objective:** Expose the pipeline to noisy, zero-inflated single-SKU data where periodic models traditionally overfit or crash mathematically.
- **Dataset:** `m5_raw_test` (Kaggle Walmart M5 `HOBBIES_1_001_CA_1` SKU)
- **Result:** **Passed**. The system evaluated the DataProfile's sparseness and successfully avoided unstable seasonal estimators, degrading gracefully to the `PolynomialTrendForecaster` as its primary fallback.

### 4. Structural Break & Non-Stationarity Handling
- **Objective:** Determine if the pipeline natively detects and reroutes around major structural anomalies (like the March 2020 COVID drop) using LLM reasoning.
- **Dataset:** `m5_ca1_aggregated` (Smoothed compilation of all CA_1 store items).
- **Result:** **Passed**. Telemetry successfully flagged "structural break detected". The LLM strictly adhered to Rule 3 and Rule 4: it specifically avoided `AutoARIMA` and prioritized `ExponentialSmoothing` due to its robustness to sudden level shifts.

---

### 7. Hard History Limits & Edge Cases
- **Objective:** Test that the infrastructure correctly handles severely constrained data (`n < 30`) without throwing internal `sktime` limit errors or overfitting complex DL/ARIMA models.
- **Dataset:** `short_series.csv` (truncated to 14 rows).
- **Result:** **Passed**. The Pipeline Architect identified `n=14` and injected rigid complexity bounds limiting permitted models strictly to `O(1)` and `O(N)` algorithmic tiers. The Model Selector correctly obeyed this limit by choosing the `PolynomialTrendForecaster`, generating valid extrapolated forecasts without instability.

## 🔜 Pending Tests

1. **Go Gateway & Valkey Caching Validation**

---
## ✅ Completed Tests (Phase 2)

### 5. Drift Monitoring & Background Retraining
- **Objective:** Test that the CUSUM/ADWIN background drift monitor correctly detects major structural shifts in residuals, queues the `retrain_worker`, and flags it in the next triage cycle.
- **Dataset:** `yahoo_s5_like_drift.csv`
- **Result:** **Passed**. A simulated script walked the dataset through ~35 stable forecasts and then forced a +50 deviation. The monitor correctly spiked the Drift Score, flagged the state as `drift_major`, and published to the `retrain:jobs` Valkey queue. 

### 6. High-Frequency & Complexity Budgets
- **Objective:** Validate that high-frequency data (`freq=H`) correctly routes into polynomial or computationally viable paths by penalizing the seasonality computational multiplier, preventing timeouts.
- **Dataset:** `etth1_like_hourly.csv`
- **Result:** **Passed**. High-frequency seasonality constraints forced the pipeline architect and model selectors to prioritize `PolynomialTrendForecaster`, generating valid `fh=[1,2,3]` outputs in sub-second limits rather than queuing four-hour `AutoARIMA` jobs.