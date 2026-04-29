# sktime-forge Debugging Journey
> A full log of every bug hit, diagnosed, and fixed during the April 2026 dev session.
> Final state: **working end-to-end** — AutoETS selected by agent tournament, real airline passenger predictions (300–500 range), pipeline saved/loaded correctly from MLflow.

---

## Bug 1 — Predictions Were Identical and Negative
**Symptom:**
```json
"predictions": [-11.249, -11.249, -11.249, -11.249, -11.249, -11.249]
```
**Initial hypothesis:** Transform applied to data but never inverse-transformed before serving.  
**Actual root cause:** `_load_data` had no `settings.data_loader` registered, so it silently fell back to a seeded `rng.standard_normal(200).cumsum()` — fake data whose last value happened to be `~-11.25`. The forecaster was fitting on random noise, not the airline dataset.  
**Fix:** Wire a generic `data_loader` dict into `settings` in `run_demo.py`, keyed by `dataset_id`. Unknown datasets raise an explicit `ValueError` instead of silently returning garbage.

---

## Bug 2 — MLflow Artifacts Were Always Empty
**Symptom:**
```
artifacts: []   # on every run, every model
```
**Diagnosis path:**
1. Assumed `log_model` was outside the `with mlflow.start_run()` block → checked code → it was inside, correct.
2. Assumed nested `start_run()` inside `_log_model_artifact` → checked → no nesting.
3. Tested `mlflow.log_artifact` directly → worked fine, artifacts saved correctly.
4. Concluded `mlflow.sklearn.log_model` was throwing a **silent exception** swallowed by a bare `except`.

**Fix:** Added `traceback.format_exc()` to the except block to surface the real error. Root cause was `_log_model_artifact` originally used `mlflow.pyfunc.log_model` which failed silently on sktime pipelines.

---

## Bug 3 — Model Logged as pyfunc, Loaded as PyFuncModel
**Symptom:**
```json
"model_class": "PyFuncModel"
```
and
```
RuntimeError: Invalid fh. Found type <class 'pandas.core.frame.DataFrame'>
```
**Root cause:** `_log_model_artifact` used `mlflow.pyfunc.log_model` thinking it preserved the sktime transform chain. But pyfunc wraps the model and strips sktime's native `.predict(fh=...)` API. The pyfunc wrapper expects a DataFrame input, sktime expects a list/array.  
**Fix:** Switch to `mlflow.sklearn.log_model(..., artifact_path="model", serialization_format="cloudpickle")` for all `TransformedTargetForecaster` pipelines. Load with `mlflow.sklearn.load_model(model_uri)` — returns the native sktime object directly.

---

## Bug 4 — Estimator Saved Without Pipeline (No Inverse Transform)
**Symptom:** Predictions were in transformed space, never converted back to original scale.  
**Root cause:** Training code was fitting the estimator on pre-transformed `y` and saving only the estimator, not the full `TransformedTargetForecaster` pipeline.  
**Fix:** Wrap every estimator in `TransformedTargetForecaster` via `_build_training_pipeline()` before fitting, and save the full pipeline to MLflow. `pipeline.predict(fh)` then auto-calls `inverse_transform()` — no manual inversion needed.

---

## Bug 5 — Stale MLflow Model Kept Being Loaded (Cold Start Never Triggered)
**Symptom:** Even after fixing training code, the old broken model kept serving.  
**Root cause:** The orchestrator checks MLflow registry **before** Valkey. A registered model with empty artifacts was still blocking cold start. Additionally, Valkey had a cached `model_version` key that survived container restarts because the MLflow server runs locally (not in Docker) and persists on disk.  
**Fix:**
```python
# Delete all versions first, then the registered model
versions = client.search_model_versions("name='ts-forecaster-airline'")
for v in versions:
    client.delete_model_version('ts-forecaster-airline', v.version)
client.delete_registered_model('ts-forecaster-airline')
```
Also flush Valkey: `docker exec -it sktime-forge-valkey-1 valkey-cli FLUSHALL`

---

## Bug 6 — Windows asyncio ProactorEventLoop Crash
**Symptom:**
```
AttributeError: 'NoneType' object has no attribute 'send'
RuntimeError: Event loop is closed
asyncio.exceptions.InvalidStateError: invalid state
```
**Root cause:** Python 3.13 + Windows uses `ProactorEventLoop` by default. The training agent called `asyncio.run()` (to read Valkey profile) inside an already-running async context — illegal in Python 3.10+. This caused the event loop to enter an invalid state after the blocking call returned, crashing the subsequent Valkey cache write.  
**Fix 1:** Add at top of `run_demo.py`:
```python
import asyncio, sys
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
```
**Fix 2:** Replace `asyncio.run(self.valkey.get(...))` in `training.py` with a synchronous redis client:
```python
import redis as sync_redis
r = sync_redis.from_url(self.valkey_url)
raw = r.get(f"profile:{dataset_id}")
profile_json = raw.decode() if raw else None
```
**Fix 3:** Wrap the post-training Valkey cache write in try/except — it's non-fatal since MLflow is the source of truth.

---

## Bug 7 — MLflow Artifact Path Warning (`artifact_path` deprecated)
**Symptom:**
```
WARNING mlflow.models.model: `artifact_path` is deprecated. Please use `name` instead.
```
**Finding:** Despite the deprecation warning, `artifact_path=` actually stores artifacts correctly in this MLflow version. Using `name=` instead caused the artifact to be stored at a different path, breaking the `models:/model-name/version` URI lookup.  
**Status:** Keep `artifact_path="model"` for now and suppress the warning. Migrate to `name=` only after MLflow registry URI resolution is confirmed working with the new parameter.

---

## Bug 8 — `get_latest_versions` Deprecated
**Symptom:**
```
FutureWarning: MlflowClient.get_latest_versions is deprecated since 2.9.0
```
**Fix (pending):**
```python
# Replace:
versions = self.mlflow.get_latest_versions(model_name)
# With:
versions = self.mlflow.search_model_versions(f"name='{model_name}'")
```

---

## Bug 9 — LLM Response Truncated Mid-JSON
**Symptom:**
```
ERROR: failed to parse LLM response: Unterminated string starting at line 1 column 46
Raw response: ["Prophet", "ExponentialSmoothing", "TBATS", "NaiveForecaster
```
**Root cause:** `max_tokens` too low on the model selector LLM call — the response was cut off before the closing `"]"`.  
**Fix (pending):** Set `max_tokens=200` on the selector call. A 5-element JSON array never needs more than ~60 tokens.

---

## Bug 10 — Missing Optional Dependencies Not Caught at Startup
**Symptom:**
```
ERROR: cannot instantiate Prophet: requires package 'prophet'
ERROR: cannot instantiate TBATS: requires package 'tbats'
ERROR: cannot instantiate AutoARIMA: requires package 'pmdarima'
```
**Root cause:** LLM nominates models that aren't installed. Failure only surfaces at fit time, wasting training time and polluting logs.  
**Fix (pending):** Pre-filter LLM candidates using `sktime.utils.dependencies._check_soft_dependencies` before writing to Valkey. The LLM should only ever receive candidates that can actually be instantiated.

---

## Bug 11 — Valkey `aclose()` Deprecation
**Symptom:**
```
DeprecationWarning: Call to deprecated close. Use aclose() instead -- Deprecated since version 5.0.1
```
**Fix (pending):** `await valkey.aclose()` everywhere instead of `await valkey.close()`.

---

## Bug 12 — Profile Load Fails → No Transforms in Pipeline
**Symptom:**
```
WARNING: failed to load profile for airline: Task got Future attached to a different loop
WARNING: failed to load profile for airline: Event loop is closed
```
**Consequence:** `_build_training_pipeline` receives `profile_json=None`, so no `Deseasonalizer` or `Differencer` is added. The pipeline is just a bare forecaster with no transforms — meaning the LLM's reasoning about seasonality and stationarity never actually influences the pipeline structure.  
**Fix (pending):** Use sync redis client for profile reads inside the thread-executor context (see Bug 6 Fix 2).

---

## Final Working Output
```json
{
  "dataset_id": "airline",
  "predictions": [483.75, 429.04, 373.51, 326.01, 370.04, 375.17],
  "model_version": "1",
  "model_class": "TransformedTargetForecaster",
  "model_status": "ok",
  "cache_hit": false
}
```
**Winner:** AutoETS (`val_mae=25.15`) beat ThetaForecaster (43.17), ExponentialSmoothing (43.49), PolynomialTrendForecaster (34.55), NaiveForecaster (81.44).

---

## Remaining Work (Post-Break)
| Priority | Fix |
|----------|-----|
| 🔴 High | Profile load → sync redis client (Bug 12) |
| 🟡 Med | LLM rationale using template instead of real output |
| 🟡 Med | max_tokens fix on selector call (Bug 9) |
| 🟡 Med | Pre-filter unavailable candidates (Bug 10) |
| 🟢 Low | `aclose()` deprecation (Bug 11) |
| 🟢 Low | `get_latest_versions` deprecation (Bug 8) |