(.venv) PS C:\Users\cyber\oss\sktime-forge> python python/scripts/run_demo.py --dataset_id airline --valkey_url redis://localhost:6379                                                                      
C:\Users\cyber\oss\sktime-forge\.venv\Lib\site-packages\mlflow\pyfunc\utils\data_validation.py:187: UserWarning: Add type hints to the `predict` method to enable data validation and automatic signature inference during model logging. Check https://mlflow.org/docs/latest/model/python_model.html#type-hint-usage-in-pythonmodel for more details.                                     
  color_warning(                                                          
Using Valkey URL: redis://localhost:6379                      
Using MLflow tracking URI: http://localhost:5000
Running demo for dataset: airline
Forecast horizon: [1, 2, 3, 4, 5, 6]
C:\Users\cyber\oss\sktime-forge\python\app\orchestrator.py:425: FutureWarning: ``mlflow.tracking.client.MlflowClient.get_latest_versions`` is deprecated since 2.9.0. Model registry stages will be removed in a future major release. To learn more about the deprecation of model registry stages, see our migration guide here: https://mlflow.org/docs/latest/model-registry.html#migrating-from-stages
  versions = self.mlflow.get_latest_versions(
WARNING:app.orchestrator:Orchestrator: failed MLflow version fallback for airline: RESOURCE_DOES_NOT_EXIST: Registered Model with name=ts-forecaster-airline not found
INFO:app.orchestrator:Orchestrator: cold start flow for dataset_id=airline
INFO:app.agents.pipeline_architect:PipelineArchitectAgent.construct_pipeline: starting for airline
INFO:app.agents.pipeline_architect:PipelineArchitectAgent: DataProfile cached at key=profile:airline (TTL=3600s)
INFO:app.agents.model_selector:ModelSelectorAgent.select: starting for dataset_id=airline
INFO:httpx:HTTP Request: POST https://ai.hackclub.com/proxy/v1/chat/completions "HTTP/1.1 200 OK"
ERROR:app.agents.model_selector:DEBUG: LLM raw body: {"id": "gen-1777440880-OLcyrOeqSF6YR4mjdYHG", "object": "chat.completion", "created": 1777440880, "model": "openai/gpt-oss-120b", "provider": "Google", "system_fingerprint": null, "choices": [{"index": 0, "logprobs": null, "finish_reason": "stop", "native_finish_reason": "stop", "message": {"role": "assistant", "content": "[\"Prophet\", \"ExponentialSmoothing\", \"TBATS\", \"NaiveForecaster", "refusal": null, "reasoning": "We need to select and rank estimators (2-5) based on rules.\n\nDataset: non-stationary, strong seasonality, structural break detected. Permitted models list includes many.\n\nRules:\n\n1. Only from permitted_models (list includes all we can use).\n\n2. No failed_estimators with failure_count >1 (none).\n\n3. If structural break, prefer models that handle changepoints natively (Prophet) or are robust to level shifts (NaiveForecaster, ExponentialSmoothing) over ARIMA-family models.\n\n4. If non-stationary, prefer models that do not require stationarity (Prophet, ExponentialSmoothing, NaiveForecaster) unless AutoARIMA is permitted and no structural break is present. Since break present, we should avoid AutoARIMA unless needed.\n\n5. If seasonality strong, prefer models that model seasonality explicitly (Prophet, TBATS, ExponentialSmoothing with seasonal_periods, AutoARIMA with seasonal=True). So ExponentialSmoothing, Prophet, TBATS are good.\n\n6. Always include at least one simple baseline (NaiveForecaster or PolynomialTrendForecaster) at the end of the list as a last-resort fallback.\n\n7. Return between 2 and 5 estimators.\n\nThus we need to rank from most preferred to least.\n\nGiven break and non-stationarity, best is Prophet (handles changepoints, seasonality). Next maybe ExponentialSmoothing (handles seasonality, robust to level shifts). Next maybe TBATS (handles multiple seasonality, robust). Then maybe ThetaForecaster (simple but may not handle break). Then baseline NaiveForecaster at end.\n\nBut we need at least one baseline at the end. So list could be: Prophet, ExponentialSmoothing, TBATS, NaiveForecaster. That's 4 estimators, baseline at end.\n\nCheck rule: baseline at the end of the list as a last-resort fallback. So NaiveForecaster should be last.\n\nThus order: \"Prophet\", \"ExponentialSmoothing\", \"TBATS\", \"NaiveForecaster\". Could also include ThetaForecaster before baseline, but we have 4 already. That's fine.\n\nMake sure all are permitted: yes.\n\nReturn JSON array.\n\n", "reasoning_details": [{"type": "reasoning.text", "text": "We need to select and rank estimators (2-5) based on rules.\n\nDataset: non-stationary, strong seasonality, structural break detected. Permitted models list includes many.\n\nRules:\n\n1. Only from permitted_models (list includes all we can use).\n\n2. No failed_estimators with failure_count >1 (none).\n\n3. If structural break, prefer models that handle changepoints natively (Prophet) or are robust to level shifts (NaiveForecaster, ExponentialSmoothing) over ARIMA-family models.\n\n4. If non-stationary, prefer models that do not require stationarity (Prophet, ExponentialSmoothing, NaiveForecaster) unless AutoARIMA is permitted and no structural break is present. Since break present, we should avoid AutoARIMA unless needed.\n\n5. If seasonality strong, prefer models that model seasonality explicitly (Prophet, TBATS, ExponentialSmoothing with seasonal_periods, AutoARIMA with seasonal=True). So ExponentialSmoothing, Prophet, TBATS are good.\n\n6. Always include at least one simple baseline (NaiveForecaster or PolynomialTrendForecaster) at the end of the list as a last-resort fallback.\n\n7. Return between 2 and 5 estimators.\n\nThus we need to rank from most preferred to least.\n\nGiven break and non-stationarity, best is Prophet (handles changepoints, seasonality). Next maybe ExponentialSmoothing (handles seasonality, robust to level shifts). Next maybe TBATS (handles multiple seasonality, robust). Then maybe ThetaForecaster (simple but may not handle break). Then baseline NaiveForecaster at end.\n\nBut we need at least one baseline at the end. So list could be: Prophet, ExponentialSmoothing, TBATS, NaiveForecaster. That's 4 estimators, baseline at end.\n\nCheck rule: baseline at the end of the list as a last-resort fallback. So NaiveForecaster should be last.\n\nThus order: \"Prophet\", \"ExponentialSmoothing\", \"TBATS\", \"NaiveForecaster\". Could also include ThetaForecaster before baseline, but we have 4 already. That's fine.\n\nMake sure all are permitted: yes.\n\nReturn JSON array.\n\n", "format": "unknown", "index": 0}]}}], "usage": {"prompt_tokens": 1366, "completion_tokens": 494, "total_tokens": 1860, "cost": 0.00030078, "is_byok": false, "prompt_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0, "audio_tokens": 0, "video_tokens": 0}, "cost_details": {"upstream_inference_cost": 0.00030078, "upstream_inference_prompt_cost": 0.00012294, "upstream_inference_completions_cost": 0.00017784}, "completion_tokens_details": {"reasoning_tokens": 522, "image_tokens": 0, "audio_tokens": 0}}}
ERROR:app.agents.model_selector:ModelSelectorAgent._llm_select: failed to parse LLM response for airline: Unterminated string starting at: line 1 column 46 (char 45)
Raw response: ["Prophet", "ExponentialSmoothing", "TBATS", "NaiveForecaster
INFO:app.agents.model_selector:ModelSelectorAgent: wrote 9 candidates for airline → ['NaiveForecaster', 'ThetaForecaster', 'ExponentialSmoothing', 'PolynomialTrendForecaster', 'Prophet', 'TBATS', 'BATS', 'AutoARIMA', 'AutoETS']
INFO:app.agents.training:TrainingAgent.handle_retrain_job: dataset_id=airline reason=cold_start
INFO:app.agents.training:TrainingAgent: fitting NaiveForecaster for airline
WARNING:app.agents.training:TrainingAgent: failed to load profile for airline: Task <Task pending name='Task-11' coro=<Redis.execute_command() running at C:\Users\cyber\oss\sktime-forge\.venv\Lib\site-packages\redis\asyncio\client.py:781> cb=[_run_until_complete_cb() at C:\Users\cyber\AppData\Local\Programs\Python\Python313\Lib\asyncio\base_events.py:181]> got Future <Future pending> attached to a different loop
2026/04/29 11:04:47 WARNING mlflow.models.model: `artifact_path` is deprecated. Please use `name` instead.
2026/04/29 11:04:47 WARNING mlflow.sklearn: Saving scikit-learn models in the pickle or cloudpickle format requires exercising caution because these formats rely on Python's object serialization mechanism, which can execute arbitrary code during deserialization. The recommended safe alternative is the 'skops' format. For more information, see: https://scikit-learn.org/stable/model_persistence.html
INFO:app.agents.training:TrainingAgent: logged NaiveForecaster as sklearn pipeline
🏃 View run puzzled-snipe-845 at: http://localhost:5000/#/experiments/1/runs/fa2b4694cf3f4d9a823451f583f7c0f3
🧪 View experiment at: http://localhost:5000/#/experiments/1
INFO:app.agents.training:TrainingAgent: NaiveForecaster → val_mae=81.4483 val_rmse=93.1339 fit_seconds=0.0
INFO:app.agents.training:TrainingAgent: fitting ThetaForecaster for airline
2026/04/29 11:05:00 WARNING mlflow.models.model: `artifact_path` is deprecated. Please use `name` instead.
2026/04/29 11:05:00 WARNING mlflow.sklearn: Saving scikit-learn models in the pickle or cloudpickle format requires exercising caution because these formats rely on Python's object serialization mechanism, which can execute arbitrary code during deserialization. The recommended safe alternative is the 'skops' format. For more information, see: https://scikit-learn.org/stable/model_persistence.html
INFO:app.agents.training:TrainingAgent: logged ThetaForecaster as sklearn pipeline
🏃 View run thundering-sloth-45 at: http://localhost:5000/#/experiments/1/runs/ae81bd157c7e46779e04f7a4a128a9f0
🧪 View experiment at: http://localhost:5000/#/experiments/1
INFO:app.agents.training:TrainingAgent: ThetaForecaster → val_mae=43.1737 val_rmse=50.8658 fit_seconds=0.3
INFO:app.agents.training:TrainingAgent: fitting ExponentialSmoothing for airline
WARNING:app.agents.training:TrainingAgent: failed to load profile for airline: Event loop is closed
2026/04/29 11:05:10 WARNING mlflow.models.model: `artifact_path` is deprecated. Please use `name` instead.
2026/04/29 11:05:10 WARNING mlflow.sklearn: Saving scikit-learn models in the pickle or cloudpickle format requires exercising caution because these formats rely on Python's object serialization mechanism, which can execute arbitrary code during deserialization. The recommended safe alternative is the 'skops' format. For more information, see: https://scikit-learn.org/stable/model_persistence.html
INFO:app.agents.training:TrainingAgent: logged ExponentialSmoothing as sklearn pipeline
🏃 View run calm-eel-402 at: http://localhost:5000/#/experiments/1/runs/95b86c7bf2fc45e3b46a738f91633d6a
🧪 View experiment at: http://localhost:5000/#/experiments/1
INFO:app.agents.training:TrainingAgent: ExponentialSmoothing → val_mae=43.4993 val_rmse=51.9193 fit_seconds=0.1
INFO:app.agents.training:TrainingAgent: fitting PolynomialTrendForecaster for airline
2026/04/29 11:05:17 WARNING mlflow.models.model: `artifact_path` is deprecated. Please use `name` instead.
2026/04/29 11:05:17 WARNING mlflow.sklearn: Saving scikit-learn models in the pickle or cloudpickle format requires exercising caution because these formats rely on Python's object serialization mechanism, which can execute arbitrary code during deserialization. The recommended safe alternative is the 'skops' format. For more information, see: https://scikit-learn.org/stable/model_persistence.html
INFO:app.agents.training:TrainingAgent: logged PolynomialTrendForecaster as sklearn pipeline
🏃 View run omniscient-wren-779 at: http://localhost:5000/#/experiments/1/runs/cd4d942f11344bd984bbbb843dbf370d
🧪 View experiment at: http://localhost:5000/#/experiments/1
INFO:app.agents.training:TrainingAgent: PolynomialTrendForecaster → val_mae=34.5551 val_rmse=48.1882 fit_seconds=0.0
INFO:app.agents.training:TrainingAgent: fitting Prophet for airline
ERROR:app.agents.training:TrainingAgent: cannot instantiate Prophet: Prophet requires package 'prophet' to be present in the python environment, but 'prophet' was not found. 'prophet' is a dependency of Prophet and required to construct it. To install the requirement 'prophet', please run: `pip install prophet` 
INFO:app.agents.training:TrainingAgent: fitting TBATS for airline
ERROR:app.agents.training:TrainingAgent: cannot instantiate TBATS: TBATS requires package 'tbats' to be present in the python environment, but 'tbats' was not found. 'tbats' is a dependency of TBATS and required to construct it. To install the requirement 'tbats', please run: `pip install tbats` 
INFO:app.agents.training:TrainingAgent: fitting BATS for airline
ERROR:app.agents.training:TrainingAgent: cannot instantiate BATS: BATS requires package 'tbats' to be present in the python environment, but 'tbats' was not found. 'tbats' is a dependency of BATS and required to construct it. To install the requirement 'tbats', please run: `pip install tbats` 
INFO:app.agents.training:TrainingAgent: fitting AutoARIMA for airline
ERROR:app.agents.training:TrainingAgent: cannot instantiate AutoARIMA: AutoARIMA requires package 'pmdarima' to be present in the python environment, but 'pmdarima' was not found. 'pmdarima' is a dependency of AutoARIMA and required to construct it. To install the requirement 'pmdarima', please run: `pip install pmdarima` 
INFO:app.agents.training:TrainingAgent: fitting AutoETS for airline
WARNING:app.agents.training:TrainingAgent: failed to load profile for airline: Event loop is closed
2026/04/29 11:05:30 WARNING mlflow.models.model: `artifact_path` is deprecated. Please use `name` instead.
2026/04/29 11:05:30 WARNING mlflow.sklearn: Saving scikit-learn models in the pickle or cloudpickle format requires exercising caution because these formats rely on Python's object serialization mechanism, which can execute arbitrary code during deserialization. The recommended safe alternative is the 'skops' format. For more information, see: https://scikit-learn.org/stable/model_persistence.html
INFO:app.agents.training:TrainingAgent: logged AutoETS as sklearn pipeline
🏃 View run bald-shrew-708 at: http://localhost:5000/#/experiments/1/runs/bc7b0db1ff484a74b80f396d98055bdf
🧪 View experiment at: http://localhost:5000/#/experiments/1
INFO:app.agents.training:TrainingAgent: AutoETS → val_mae=25.1520 val_rmse=29.7411 fit_seconds=3.3
INFO:app.agents.training:TrainingAgent: best model for airline is AutoETS (val_mae=25.1520)
Successfully registered model 'ts-forecaster-airline'.
2026/04/29 11:05:39 WARNING mlflow.tracking._model_registry.fluent: Run with id bc7b0db1ff484a74b80f396d98055bdf has no artifacts at artifact path 'model', registering model based on models:/m-a81e0d33f6594ce28cbfe2bdd4102f51 instead
2026/04/29 11:05:39 INFO mlflow.store.model_registry.abstract_store: Waiting up to 300 seconds for model version to finish creation. Model name: ts-forecaster-airline, version 1
Created version '1' of model 'ts-forecaster-airline'.
INFO:app.agents.training:TrainingAgent: promoted model version 1 for airline
INFO:app.agents.prediction:PredictionAgent: loading model from MLflow for airline v1
INFO:app.agents.watchdog:Watchdog: starting post-promotion monitoring for airline v1 (baseline_mae=25.1520, ttl=3600s)
INFO:app.agents.prediction:PredictionAgent: served 6-step forecast for airline v1 in 40.0 ms (cache_hit=False)
Forecast result:
{
  "dataset_id": "airline",
  "predictions": [
    483.7564244857241,
    429.041125509465,
    373.51584265216655,
    326.0116381435012,
    370.04631685386715,
    375.1702170511556
  ],
  "prediction_intervals": {
    "lower": [
      457.0147321424251,
      398.723502699071,
      341.65241702109716,
      295.0724385872573,
      331.7973240048838,
      333.45957539721593
    ],
    "upper": [
      511.5389965929269,
      460.50101616910973,
      402.7546789616854,
      356.76794889811816,
      407.5204474454512,
      417.36596121453573
    ]
  },
  "model_version": "1",
  "model_class": "TransformedTargetForecaster",
  "model_status": "ok",
  "drift_score": null,
  "drift_method": null,
  "warning": null,
  "llm_rationale": "Forecast generated for dataset airline using TransformedTargetForecaster (version 1) over 6 horizon steps. First predictions: 483.756, 429.041, 373.516. Prediction intervals are included to show forecast uncertainty. No active drift signal is attached to this response.",
  "cache_hit": false,
  "correlation_id": "demo-run"
}
C:\Users\cyber\oss\sktime-forge\python\scripts\run_demo.py:160: DeprecationWarning: Call to deprecated close. (Use aclose() instead) -- Deprecated since version 5.0.1.
  await valkey.close()




   python python/scripts/run_demo.py --dataset_id airline --valkey_url redis://localhost:6379
C:\Users\cyber\oss\sktime-forge\.venv\Lib\site-packages\mlflow\pyfunc\utils\data_validation.py:187: UserWarning: Add type hints to the `predict` method to enable data validation and automatic signature inference during model logging. Check https://mlflow.org/docs/latest/model/python_model.html#type-hint-usage-in-pythonmodel for more details.
  color_warning(
Using Valkey URL: redis://localhost:6379
Using MLflow tracking URI: http://localhost:5000
Running demo for dataset: airline
Forecast horizon: [1, 2, 3, 4, 5, 6]
C:\Users\cyber\oss\sktime-forge\python\app\orchestrator.py:425: FutureWarning: ``mlflow.tracking.client.MlflowClient.get_latest_versions`` is deprecated since 2.9.0. Model registry stages will be removed in a future major release. To learn more about the deprecation of model registry stages, see our migration guide here: https://mlflow.org/docs/latest/model-registry.html#migrating-from-stages
  versions = self.mlflow.get_latest_versions(
WARNING:app.orchestrator:Orchestrator: failed MLflow version fallback for airline: RESOURCE_DOES_NOT_EXIST: Registered Model with name=ts-forecaster-airline not found
INFO:app.orchestrator:Orchestrator: cold start flow for dataset_id=airline
INFO:app.agents.pipeline_architect:PipelineArchitectAgent.construct_pipeline: starting for airline
INFO:app.agents.pipeline_architect:PipelineArchitectAgent: DataProfile cached at key=profile:airline (TTL=3600s)
INFO:app.agents.model_selector:ModelSelectorAgent.select: starting for dataset_id=airline
INFO:httpx:HTTP Request: POST https://ai.hackclub.com/proxy/v1/chat/completions "HTTP/1.1 200 OK"
INFO:app.agents.training:TrainingAgent.handle_retrain_job: dataset_id=airline reason=cold_start
INFO:app.agents.training:TrainingAgent: fitting Prophet for airline
ERROR:app.agents.training:TrainingAgent: cannot instantiate Prophet: Prophet requires package 'prophet' to be present in the python environment, but 'prophet' was not found. 'prophet' is a dependency of Prophet and required to construct it. To install the requirement 'prophet', please run: `pip install prophet` 
INFO:app.agents.training:TrainingAgent: fitting TBATS for airline
ERROR:app.agents.training:TrainingAgent: cannot instantiate TBATS: TBATS requires package 'tbats' to be present in the python environment, but 'tbats' was not found. 'tbats' is a dependency of TBATS and required to construct it. To install the requirement 'tbats', please run: `pip install tbats` 
INFO:app.agents.training:TrainingAgent: fitting BATS for airline
ERROR:app.agents.training:TrainingAgent: cannot instantiate BATS: BATS requires package 'tbats' to be present in the python environment, but 'tbats' was not found. 'tbats' is a dependency of BATS and required to construct it. To install the requirement 'tbats', please run: `pip install tbats` 
INFO:app.agents.training:TrainingAgent: fitting AutoETS for airline
WARNING:app.agents.training:TrainingAgent: failed to load profile for airline: Task <Task pending name='Task-11' coro=<Redis.execute_command() running at C:\Users\cyber\oss\sktime-forge\.venv\Lib\site-packages\redis\asyncio\client.py:781> cb=[_run_until_complete_cb() at C:\Users\cyber\AppData\Local\Programs\Python\Python313\Lib\asyncio\base_events.py:181]> got Future <Future pending> attached to a different loop
2026/04/29 11:46:46 WARNING mlflow.models.model: `artifact_path` is deprecated. Please use `name` instead.
2026/04/29 11:46:46 WARNING mlflow.sklearn: Saving scikit-learn models in the pickle or cloudpickle format requires exercising caution because these formats rely on Python's object serialization mechanism, which can execute arbitrary code during deserialization. The recommended safe alternative is the 'skops' format. For more information, see: https://scikit-learn.org/stable/model_persistence.html
INFO:app.agents.training:TrainingAgent: logged AutoETS as sklearn pipeline
🏃 View run thoughtful-sow-779 at: http://localhost:5000/#/experiments/1/runs/b641e32f86f24fd09ddd36ebf870f590
🧪 View experiment at: http://localhost:5000/#/experiments/1
INFO:app.agents.training:TrainingAgent: AutoETS → val_mae=25.1520 val_rmse=29.7411 fit_seconds=2.1
INFO:app.agents.training:TrainingAgent: fitting AutoARIMA for airline
ERROR:app.agents.training:TrainingAgent: cannot instantiate AutoARIMA: AutoARIMA requires package 'pmdarima' to be present in the python environment, but 'pmdarima' was not found. 'pmdarima' is a dependency of AutoARIMA and required to construct it. To install the requirement 'pmdarima', please run: `pip install pmdarima` 
INFO:app.agents.training:TrainingAgent: fitting ExponentialSmoothing for airline
2026/04/29 11:46:54 WARNING mlflow.models.model: `artifact_path` is deprecated. Please use `name` instead.
2026/04/29 11:46:54 WARNING mlflow.sklearn: Saving scikit-learn models in the pickle or cloudpickle format requires exercising caution because these formats rely on Python's object serialization mechanism, which can execute arbitrary code during deserialization. The recommended safe alternative is the 'skops' format. For more information, see: https://scikit-learn.org/stable/model_persistence.html
INFO:app.agents.training:TrainingAgent: logged ExponentialSmoothing as sklearn pipeline
🏃 View run monumental-mare-275 at: http://localhost:5000/#/experiments/1/runs/1d500b73f61643bb93747a900e424fec
🧪 View experiment at: http://localhost:5000/#/experiments/1
INFO:app.agents.training:TrainingAgent: ExponentialSmoothing → val_mae=43.4961 val_rmse=51.9158 fit_seconds=0.2
INFO:app.agents.training:TrainingAgent: fitting ThetaForecaster for airline
WARNING:app.agents.training:TrainingAgent: failed to load profile for airline: Event loop is closed
2026/04/29 11:46:58 WARNING mlflow.models.model: `artifact_path` is deprecated. Please use `name` instead.
2026/04/29 11:46:59 WARNING mlflow.sklearn: Saving scikit-learn models in the pickle or cloudpickle format requires exercising caution because these formats rely on Python's object serialization mechanism, which can execute arbitrary code during deserialization. The recommended safe alternative is the 'skops' format. For more information, see: https://scikit-learn.org/stable/model_persistence.html
INFO:app.agents.training:TrainingAgent: logged ThetaForecaster as sklearn pipeline
🏃 View run colorful-horse-931 at: http://localhost:5000/#/experiments/1/runs/6e63ee0ae5a047068d09b605945da47a
🧪 View experiment at: http://localhost:5000/#/experiments/1
INFO:app.agents.training:TrainingAgent: ThetaForecaster → val_mae=91.3702 val_rmse=102.4195 fit_seconds=0.0
INFO:app.agents.training:TrainingAgent: fitting PolynomialTrendForecaster for airline
2026/04/29 11:47:03 WARNING mlflow.models.model: `artifact_path` is deprecated. Please use `name` instead.
2026/04/29 11:47:04 WARNING mlflow.sklearn: Saving scikit-learn models in the pickle or cloudpickle format requires exercising caution because these formats rely on Python's object serialization mechanism, which can execute arbitrary code during deserialization. The recommended safe alternative is the 'skops' format. For more information, see: https://scikit-learn.org/stable/model_persistence.html
INFO:app.agents.training:TrainingAgent: logged PolynomialTrendForecaster as sklearn pipeline
🏃 View run brawny-whale-483 at: http://localhost:5000/#/experiments/1/runs/6f47b1cbd2694856a624354d1be5a21b
🧪 View experiment at: http://localhost:5000/#/experiments/1
INFO:app.agents.training:TrainingAgent: PolynomialTrendForecaster → val_mae=34.5551 val_rmse=48.1882 fit_seconds=0.0
INFO:app.agents.training:TrainingAgent: fitting NaiveForecaster for airline
WARNING:app.agents.training:TrainingAgent: failed to load profile for airline: Event loop is closed
2026/04/29 11:47:12 WARNING mlflow.models.model: `artifact_path` is deprecated. Please use `name` instead.
2026/04/29 11:47:12 WARNING mlflow.sklearn: Saving scikit-learn models in the pickle or cloudpickle format requires exercising caution because these formats rely on Python's object serialization mechanism, which can execute arbitrary code during deserialization. The recommended safe alternative is the 'skops' format. For more information, see: https://scikit-learn.org/stable/model_persistence.html
INFO:app.agents.training:TrainingAgent: logged NaiveForecaster as sklearn pipeline
🏃 View run skillful-shoat-878 at: http://localhost:5000/#/experiments/1/runs/d008500d2e704357bfec096232199f71
🧪 View experiment at: http://localhost:5000/#/experiments/1
INFO:app.agents.training:TrainingAgent: NaiveForecaster → val_mae=81.4483 val_rmse=93.1339 fit_seconds=0.0
INFO:app.agents.training:TrainingAgent: best model for airline is AutoETS (val_mae=25.1520)
Successfully registered model 'ts-forecaster-airline'.
2026/04/29 11:47:16 WARNING mlflow.tracking._model_registry.fluent: Run with id b641e32f86f24fd09ddd36ebf870f590 has no artifacts at artifact path 'model', registering model based on models:/m-cd05f96f29a747ac82060574a5d21c51 instead
2026/04/29 11:47:17 INFO mlflow.store.model_registry.abstract_store: Waiting up to 300 seconds for model version to finish creation. Model name: ts-forecaster-airline, version 1
Created version '1' of model 'ts-forecaster-airline'.
INFO:app.agents.training:TrainingAgent: promoted model version 1 for airline
INFO:app.agents.prediction:PredictionAgent: loading model from MLflow for airline v1
INFO:app.agents.watchdog:Watchdog: starting post-promotion monitoring for airline v1 (baseline_mae=25.1520, ttl=3600s)
INFO:app.agents.prediction:PredictionAgent: served 6-step forecast for airline v1 in 25.8 ms (cache_hit=False)
Forecast result:
{
  "dataset_id": "airline",
  "predictions": [
    483.7564244857241,
    429.041125509465,
    373.51584265216655,
    326.0116381435012,
    370.04631685386715,
    375.1702170511556
  ],
  "prediction_intervals": {
    "lower": [
      457.3183478997306,
      399.3248085996499,
      344.0507669451156,
      293.40917665636346,
      330.63026342005463,
      333.3896399070918
    ],
    "upper": [
      510.6893239968434,
      458.5687141934848,
      405.01485118160423,
      357.596458855407,
      411.4126654856508,
      417.6872264454395
    ]
  },
  "model_version": "1",
  "model_class": "TransformedTargetForecaster",
  "model_status": "ok",
  "drift_score": null,
  "drift_method": null,
  "warning": null,
  "llm_rationale": "Forecast generated for dataset airline using TransformedTargetForecaster (version 1) over 6 horizon steps. First predictions: 483.756, 429.041, 373.516. Prediction intervals are included to show forecast uncertainty. No active drift signal is attached to this response.",
  "cache_hit": false,
  "correlation_id": "demo-run"
}
C:\Users\cyber\oss\sktime-forge\python\scripts\run_demo.py:160: DeprecationWarning: Call to deprecated close. (Use aclose() instead) -- Deprecated since version 5.0.1.
  await valkey.close()

  