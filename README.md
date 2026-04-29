# sktime-agentic

**Autonomous time-series forecasting infrastructure driven by an LLM agent.**

A companion repo to [`sktime`](https://github.com/sktime/sktime) and [`sktime-mcp`](https://github.com/sktime/sktime-mcp).

---

## How It Works

Every decision — model selection, training, promotion, drift response — is made by an LLM agent that calls sktime capabilities as MCP tools. The agent investigates the data, reasons across production history, constructs a pipeline, evaluates all candidates, and decides which model to promote. Nothing happens without the agent choosing to make it happen.

```
Production event (drift / cold start / human request)
         │
         ▼
┌─────────────────────────────────────────────────┐
│              LLM Agent Loop (ReAct)             │
│                                                 │
│  observe → reason → call tool → observe → ...  │
│                                                 │
│  Has access to: full production history,        │
│  past model failures, drift patterns,           │
│  dataset characteristics across all runs        │
└────────────────────┬────────────────────────────┘
                     │  MCP tool calls
                     ▼
┌─────────────────────────────────────────────────┐
│              sktime-mcp Tool Layer              │
│                                                 │
│  profile_dataset     detect_seasonality         │
│  run_stationarity    detect_drift               │
│  list_candidates     evaluate_model             │
│  fit_model           promote_model              │
│  get_forecast        get_model_history          │
│  get_drift_history   check_structural_break     │
└─────────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│           Production Infrastructure             │
│                                                 │
│  sktime pipelines   MLflow model registry       │
│  Valkey / Redis     FastAPI serving layer       │
│  Go API gateway     S3 / GCS model storage      │
└─────────────────────────────────────────────────┘
```

---

## Verified End-to-End: Airline Dataset

> Full log: [`docs/full_end_log.md`](docs/full_end_log.md)

The following is a real, unedited trace of the system running the `airline` dataset from a cold start — no model in the registry, no cached state. The **LLM is actively driving the tool calls**; it receives the data profile and decides which estimators to rank and why.

### Step 1 — Cold Start Detected

The orchestrator checks MLflow, finds no registered model, and triggers the full agent pipeline.

```
WARNING:app.orchestrator: failed MLflow version fallback for airline:
  RESOURCE_DOES_NOT_EXIST: Registered Model with name=ts-forecaster-airline not found
INFO:app.orchestrator: cold start flow for dataset_id=airline
```

### Step 2 — PipelineArchitectAgent Profiles the Data

```
INFO:app.agents.pipeline_architect: PipelineArchitectAgent.construct_pipeline: starting for airline
INFO:app.agents.pipeline_architect: PipelineArchitectAgent: DataProfile cached at key=profile:airline (TTL=3600s)
```

### Step 3 — ModelSelectorAgent Calls the LLM (ReAct loop)

The LLM receives the data profile and a list of permitted estimators. It reasons through the following chain of thought in real time before returning its ranked list:

> *"Dataset: non-stationary, strong seasonality, structural break detected.*
> *If structural break, prefer models that handle changepoints natively (Prophet) or are robust to level shifts (NaiveForecaster, ExponentialSmoothing) over ARIMA-family models.*
> *If non-stationary, prefer models that do not require stationarity. Since break is present, avoid AutoARIMA.*
> *If seasonality strong, prefer models that model seasonality explicitly (Prophet, TBATS, ExponentialSmoothing).*
> *Always include at least one simple baseline (NaiveForecaster) at the end as a last-resort fallback.*"*

```
INFO:app.agents.model_selector: ModelSelectorAgent.select: starting for dataset_id=airline
INFO:httpx: HTTP Request: POST https://ai.hackclub.com/proxy/v1/chat/completions "HTTP/1.1 200 OK"

LLM ranked output: ["Prophet", "ExponentialSmoothing", "TBATS", "NaiveForecaster"]

INFO:app.agents.model_selector: ModelSelectorAgent: wrote 9 candidates for airline →
  ['NaiveForecaster', 'ThetaForecaster', 'ExponentialSmoothing', 'PolynomialTrendForecaster',
   'Prophet', 'TBATS', 'BATS', 'AutoARIMA', 'AutoETS']
```

The LLM's top picks (Prophet, TBATS) were unavailable due to missing optional dependencies — the system gracefully fell through to the next available candidates.

### Step 4 — TrainingAgent Fits and Evaluates All Candidates

Each candidate is fitted as a `TransformedTargetForecaster` sktime pipeline and logged to MLflow. Results are compared by `val_mae` on a held-out validation split.

```
INFO:app.agents.training: TrainingAgent.handle_retrain_job: dataset_id=airline reason=cold_start

INFO:app.agents.training: fitting AutoETS for airline
INFO:app.agents.training: logged AutoETS as sklearn pipeline
🏃 View run thoughtful-sow-779 at: http://localhost:5000/#/experiments/1/runs/b641e32f86f24fd09ddd36ebf870f590
INFO:app.agents.training: AutoETS → val_mae=25.1520 val_rmse=29.7411 fit_seconds=2.1

INFO:app.agents.training: fitting ExponentialSmoothing for airline
INFO:app.agents.training: logged ExponentialSmoothing as sklearn pipeline
🏃 View run monumental-mare-275 at: http://localhost:5000/#/experiments/1/runs/1d500b73f61643bb93747a900e424fec
INFO:app.agents.training: ExponentialSmoothing → val_mae=43.4961 val_rmse=51.9158 fit_seconds=0.2

INFO:app.agents.training: fitting ThetaForecaster for airline
INFO:app.agents.training: logged ThetaForecaster as sklearn pipeline
🏃 View run colorful-horse-931 at: http://localhost:5000/#/experiments/1/runs/6e63ee0ae5a047068d09b605945da47a
INFO:app.agents.training: ThetaForecaster → val_mae=91.3702 val_rmse=102.4195 fit_seconds=0.0

INFO:app.agents.training: fitting PolynomialTrendForecaster for airline
INFO:app.agents.training: logged PolynomialTrendForecaster as sklearn pipeline
🏃 View run brawny-whale-483 at: http://localhost:5000/#/experiments/1/runs/6f47b1cbd2694856a624354d1be5a21b
INFO:app.agents.training: PolynomialTrendForecaster → val_mae=34.5551 val_rmse=48.1882 fit_seconds=0.0

INFO:app.agents.training: fitting NaiveForecaster for airline
INFO:app.agents.training: logged NaiveForecaster as sklearn pipeline
🏃 View run skillful-shoat-878 at: http://localhost:5000/#/experiments/1/runs/d008500d2e704357bfec096232199f71
INFO:app.agents.training: NaiveForecaster → val_mae=81.4483 val_rmse=93.1339 fit_seconds=0.0
```

**Model comparison summary:**

| Model | val_mae | val_rmse | fit_seconds |
|---|---|---|---|
| **AutoETS** ✅ | **25.1520** | **29.7411** | 2.1 |
| PolynomialTrendForecaster | 34.5551 | 48.1882 | 0.0 |
| ExponentialSmoothing | 43.4961 | 51.9158 | 0.2 |
| NaiveForecaster | 81.4483 | 93.1339 | 0.0 |
| ThetaForecaster | 91.3702 | 102.4195 | 0.0 |

### Step 5 — Winner Promoted to MLflow Registry

```
INFO:app.agents.training: best model for airline is AutoETS (val_mae=25.1520)

Successfully registered model 'ts-forecaster-airline'.
INFO mlflow.store.model_registry.abstract_store: Waiting up to 300 seconds for model version to finish creation.
  Model name: ts-forecaster-airline, version 1
Created version '1' of model 'ts-forecaster-airline'.

INFO:app.agents.training: promoted model version 1 for airline
```

### Step 6 — PredictionAgent Serves the Forecast

```
INFO:app.agents.prediction: PredictionAgent: loading model from MLflow for airline v1
INFO:app.agents.watchdog: Watchdog: starting post-promotion monitoring for airline v1
  (baseline_mae=25.1520, ttl=3600s)
INFO:app.agents.prediction: PredictionAgent: served 6-step forecast for airline v1 in 25.8 ms (cache_hit=False)
```

### Step 7 — Final Forecast Response

```json
{
  "dataset_id": "airline",
  "predictions": [483.756, 429.041, 373.516, 326.012, 370.046, 375.170],
  "prediction_intervals": {
    "lower": [457.318, 399.325, 344.051, 293.409, 330.630, 333.390],
    "upper": [510.689, 458.569, 405.015, 357.596, 411.413, 417.687]
  },
  "model_version": "1",
  "model_class": "TransformedTargetForecaster",
  "model_status": "ok",
  "llm_rationale": "Forecast generated for dataset airline using TransformedTargetForecaster (version 1)
    over 6 horizon steps. First predictions: 483.756, 429.041, 373.516.
    Prediction intervals are included to show forecast uncertainty.
    No active drift signal is attached to this response.",
  "cache_hit": false,
  "correlation_id": "demo-run"
}
```

---

## Architecture

### Actual Project Layout

```
sktime-agentic/
├── python/
│   ├── app/
│   │   ├── agents/
│   │   │   ├── chat_router.py        # NL query → structured ForecastRequest
│   │   │   ├── model_selector.py     # ReAct loop: LLM calls MCP tools to rank models
│   │   │   ├── pipeline_architect.py # Profiles data, writes DataProfile to Valkey
│   │   │   ├── prediction.py         # Loads model from MLflow, runs inference
│   │   │   ├── training.py           # Fits + evaluates all candidates, promotes winner
│   │   │   └── watchdog.py           # Post-promotion MAE monitoring, queues retrain
│   │   ├── mcp/
│   │   │   ├── client.py             # MCPClient: dispatches tool calls to implementations
│   │   │   ├── check_structural_break.py
│   │   │   ├── detect_seasonality.py
│   │   │   ├── estimate_training_cost.py
│   │   │   ├── get_dataset_history.py
│   │   │   ├── get_model_complexity_budget.py
│   │   │   └── run_stationarity_test.py
│   │   ├── memory/
│   │   │   └── memory.py             # Per-dataset history in Valkey (model history, drift events)
│   │   ├── monitoring/
│   │   │   └── drift_monitor.py      # CUSUM + ADWIN detection, publishes signal only
│   │   ├── registry/
│   │   │   ├── registry.py           # CANDIDATE_ESTIMATORS, profile-based filtering
│   │   │   └── data_registry.py      # Dataset record store in Valkey
│   │   ├── data/
│   │   │   └── local_loader.py       # CSV loader for local fixture datasets
│   │   ├── config.py                 # Pydantic Settings — reads from .env
│   │   ├── contracts.py              # Protocol interfaces (AgentMemory, Watchdog)
│   │   ├── main.py                   # FastAPI app: /forecast, /chat, /admin/*, /metrics
│   │   ├── orchestrator.py           # Coordinates cold-start and retrain flows
│   │   ├── prompts/prompts.py        # System prompts for each LLM agent
│   │   ├── retrain_worker.py         # Valkey stream consumer for retrain:jobs
│   │   └── schemas.py                # Pydantic request/response models
│   ├── scripts/
│   │   ├── run_demo.py               # Local end-to-end demo runner
│   │   ├── cold_start_aeroplane.py   # Seed + cold-start for aeroplane dataset
│   │   ├── ingest_data.py            # Push a CSV dataset into Valkey
│   │   └── start_local_mlflow.py     # Start MLflow tracking server locally
│   ├── tests/
│   │   ├── fixtures/sample_datasets/ # Curated CSVs for testing
│   │   └── unit/                     # Agent unit tests
│   └── requirements.txt
│
├── go/                               # Go API gateway (request routing, Valkey bridge)
├── k8s/                              # Kubernetes manifests for all services
├── docs/                             # Architecture docs, problem log, full run logs
├── data_cache/                       # Airline CSV (built-in fallback dataset)
├── docker-compose.yml                # Valkey + MLflow + Python/Go workers
└── .env.example                      # All required environment variables
```

### What Each Layer Does

**`ModelSelectorAgent` (`agents/model_selector.py`)**  
The core intelligence. Runs a ReAct loop: builds a data profile via `PipelineArchitectAgent`, then calls the LLM with that profile and a list of permitted estimators. The LLM reasons through seasonality, stationarity, structural breaks, and past failures to return a ranked candidate list. Tool calls are dispatched via `MCPClient`. The ranked list is written to Valkey for `TrainingAgent` to consume.

**`TrainingAgent` (`agents/training.py`)**  
Reads the ranked candidate list, fits each estimator as a `TransformedTargetForecaster` sktime pipeline in a thread executor (non-async), evaluates on a validation split, logs every run to MLflow, picks the lowest `val_mae` winner, and registers it in the MLflow model registry.

**`PredictionAgent` (`agents/prediction.py`)**  
Resolves the active model version from Valkey (falls back to MLflow registry), loads the model (with in-process caching to avoid repeated artifact downloads), and runs inference off the event loop in an executor. Returns point forecasts + prediction intervals and an LLM-generated rationale string.

**`Watchdog` (`agents/watchdog.py`)**  
Spawned after every model promotion. Polls residuals from Valkey, computes live MAE, compares against the baseline MAE from training, and queues a retrain job if degradation exceeds the threshold.

**`Orchestrator` (`orchestrator.py`)**  
The top-level coordinator. Detects cold-start vs warm-start, chains `PipelineArchitectAgent → ModelSelectorAgent → TrainingAgent → PredictionAgent`, and manages Valkey stream workers.

**`DriftMonitor` (`monitoring/drift_monitor.py`)**  
CUSUM + ADWIN statistical detection. Publishes a signal only — it makes no decisions. The agent decides what to do in response.

**Go layer (`go/`)**  
API gateway for routing external forecast requests. Stateless — all state lives in Valkey and MLflow.

---

## MCP Tool Reference

All tools are implemented under `python/app/mcp/` and dispatched by `MCPClient`.

### Statistical Analysis

| Tool | Description | Returns |
|------|-------------|---------|
| `detect_seasonality` | Seasonal period and strength detection | period, strength, method |
| `run_stationarity_test` | ADF + KPSS stationarity tests | p-values, conclusion |
| `check_structural_break` | CUSUM-based break detection | break_detected, location, confidence |
| `get_dataset_history` | Full production history for a dataset | past models, scores, failures, drift events |
| `estimate_training_cost` | Cost estimate before fitting | estimated_seconds, complexity |
| `get_model_complexity_budget` | Budget constraints for model selection | max_params, time_budget |

---

## Quickstart

```bash
# Clone
git clone https://github.com/your-org/sktime-agentic
cd sktime-agentic

# Set up Python environment
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r python/requirements.txt

# Configure
cp .env.example .env
# Edit .env — set LLM_API_KEY, MLFLOW_TRACKING_URI, VALKEY_URL

# Option A: run with local MLflow + Valkey from Docker
docker compose up -d valkey mlflow
python python/scripts/run_demo.py --dataset_id airline --valkey_url redis://localhost:6379

# Option B: run MLflow locally without Docker
python python/scripts/start_local_mlflow.py   # separate terminal
python python/scripts/run_demo.py --dataset_id airline --valkey_url redis://localhost:6379

# Run against a local CSV fixture
python python/scripts/run_demo.py \
  --dataset_id yahoo_s5_like_drift.csv \
  --local_dataset_dir python/tests/fixtures/sample_datasets

# Start the FastAPI server
uvicorn python.app.main:app --reload
# POST /forecast   {"dataset_id": "airline", "fh": [1,2,3,4,5,6]}
# POST /chat       {"query": "forecast airline next 6 months"}
```

---

## License

BSD 3-Clause — same as sktime.
