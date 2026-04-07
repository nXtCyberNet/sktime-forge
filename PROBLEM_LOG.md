# Problem Log

This file tracks issues found during implementation, along with root cause and fix.
It should be updated at each development step.

## Session: 2026-04-05

| Step | Layer | Problem | Root Cause | Fix Applied | Status |
|---|---|---|---|---|---|
| 1 | Scaffolding | Python one-liner file population failed with unterminated string syntax error | Nested quotes in inline command broke string boundaries | Switched to a dedicated script file and executed it | Resolved |
| 2 | Deploy | Kubernetes deployment YAML showed invalid indentation and duplicate map errors | Tabs were used instead of spaces in YAML | Rewrote manifests using space indentation and valid list structure | Resolved |
| 3 | Python imports | Import ..core.settings could not be resolved | Code expected settings.py but only config.py existed | Added compatibility module app/core/settings.py and exported symbols | Resolved |
| 4 | Python imports | Import mlflow.sktime could not be resolved | Static analyzer could not resolve optional mlflow submodule in current environment | Added lazy loader app/core/mlflow_compat.py and replaced direct imports | Resolved |
| 5 | Python imports | Multiple relative imports in orchestrator and main failed | Import paths referenced moved/nonexistent modules | Corrected import paths and added compatibility alias module app/agents/orchestrator.py | Resolved |
| 6 | Python syntax | model_selector.py had unexpected indentation | Indentation drift during patching | Corrected indentation in validate_serialization block | Resolved |
| 7 | Test environment | pytest import unresolved in tests | Python environment in editor does not currently expose pytest package | Keep requirements entry; resolve by selecting/installing workspace environment packages | Open |
| 8 | Drift detection | ADWIN helper uses midpoint split heuristic, not true adaptive ADWIN test | Simplified implementation omitted adaptive windows and statistical confidence bounds | Flagged for v1 improvement: add robust shift test with sample-size-aware thresholding (or use river ADWIN) | Open |
| 9 | Drift detection | River ADWIN helper returned no numeric value and printed to stdout | Function switched to boolean drift flag usage without mapping to expected float score | Updated helper to return [0,1] score: 0 when no drift, effect-size severity when ADWIN detects drift | Resolved |
| 10 | Drift detection | Circular CUSUM tuning risk when k/h are recomputed from the same residual window used for detection | Drift in the current window inflates sigma, which inflates k and h and suppresses drift sensitivity | Implemented fixed baseline per model version in DriftMonitor: bootstrap baseline from first N residuals, lock k=0.5*sigma and h=4.0*sigma, persist/reuse via Valkey key cusum:baseline:{dataset}:{version}, reset window on model version change | Resolved |
| 11 | Drift detection | Hard ADWIN import caused environment-dependent import errors | river may be missing in editor/runtime environments | Switched to lazy import via importlib in DriftMonitor helper so module remains import-safe and ADWIN scoring degrades to 0 when unavailable | Resolved |
| 12 | Drift detection | CUSUM k/h were built inline in async baseline initializer, reducing clarity | Parameter construction logic mixed with I/O and bootstrap orchestration | Refactored to dedicated helper methods: _build_cusum_params and _is_valid_cusum_params; async function now orchestrates flow only | Resolved |
| 13 | Drift detection | ADWIN detector was rebuilt from scratch on each detection call | Batch replay of residual window lost true streaming semantics and wasted compute | Refactored to per-dataset persistent ADWIN detectors updated once per residual; detection reads stored state and resets detector on model version changes | Resolved |

## Update Rule

When a new issue appears, append one new row with:

- Step number
- Layer or area
- Exact problem symptom
- Root cause
- Fix applied
- Status (Resolved or Open)

## Design Note: CUSUM Baseline Locking

Decision summary:

- ADWIN from River is primarily a drift flag, not a severity percentage.
- If severity is needed, compute it separately.
- CUSUM thresholds must not be estimated from the same window used for detection.

Approved approach:

1. On model promotion, collect first N in-control residuals.
2. Compute baseline sigma from that clean window.
3. Lock parameters for this model version:
	 - k = 0.5 * sigma
	 - h = 4.0 * sigma
4. Store baseline params keyed by dataset and model version.
5. Run future CUSUM checks with fixed k and h.
6. On next model promotion, reset baseline and repeat.

Implementation hints:

- Keep baseline params separate from rolling residual window state.
- Use mu = 0.0 for CUSUM center on residuals of a well-performing model.
- Add settings:
	- cusum_baseline_min_samples = 50
	- cusum_k_sigma = 0.5
	- cusum_h_sigma = 4.0
