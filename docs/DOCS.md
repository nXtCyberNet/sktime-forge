# sktime Agentic Forecasting System
### A Production-Grade Companion Layer for Autonomous Time Series Forecasting
**ESoC 2026 — GC.OS Track — sktime Agentic**

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Design Principles](#2-design-principles)
3. [High-Level Architecture](#3-high-level-architecture)
4. [Data Flow — Step by Step](#4-data-flow--step-by-step)
5. [Python Layer — The Agentic Brain](#5-python-layer--the-agentic-brain)
   - [Orchestrator](#51-orchestrator)
   - [ModelSelectorAgent](#52-modelselectoragent)
   - [PredictionAgent](#53-predictionagent)
   - [DriftMonitor](#54-driftmonitor)
   - [TrainingAgent](#55-trainingagent)
6. [Go Layer — Production Wrapper](#6-go-layer--production-wrapper)
7. [Infrastructure Layer](#7-infrastructure-layer)
   - [Valkey Streams](#71-valkey-streams)
   - [MLflow + Cloud Storage](#72-mlflow--cloud-storage)
   - [Kubernetes](#73-kubernetes)
8. [API Reference](#8-api-reference)
9. [Schemas](#9-schemas)
10. [Configuration](#10-configuration)
11. [Tech Stack](#11-tech-stack)
12. [Constraints and Tradeoffs](#12-constraints-and-tradeoffs)
13. [Error Handling and Failure Modes](#13-error-handling-and-failure-modes)
14. [Local Development Setup](#14-local-development-setup)
15. [Testing Strategy](#15-testing-strategy)
16. [Milestone Table](#16-milestone-table)
17. [v1 vs v2 Roadmap](#17-v1-vs-v2-roadmap)
18. [Why This Architecture](#18-why-this-architecture)

---

## 1. Project Overview

### One-Sentence Summary

A smart autonomous forecasting system that uses sktime to decide how to forecast, automatically retrains itself when data changes, and serves fast reliable predictions through a clean API — without the user ever waiting for a model decision.

### What This Is

The sktime Agentic Forecasting System is a **companion repository** to sktime — same philosophy as `sktime-mcp`, but focused on production-grade autonomous deployment rather than MCP tooling. It does not modify or fork the sktime core. It wraps sktime's intelligence in an infrastructure layer that makes it self-managing at scale.

The system doesn't just predict. It **decides how to predict**:

- Automatically selects the best sktime forecaster for each dataset
- Detects when data distribution shifts (drift)
- Retrains itself in the background without interrupting users
- Serves predictions immediately via a fast Go API gateway
- Warns users explicitly when predictions may be less reliable

### What This Is Not

- Not a fork of sktime
- Not a replacement for sktime-mcp
- Not a research notebook
- Not a monolith — Python handles ML intelligence, Go handles API serving

### Relation to sktime Ecosystem

```
sktime (core ML library)
    ↑ uses
sktime-mcp (MCP tool layer)          sktime-agentic (this project)
    ↑ used by                              ↑ uses
AI coding agents                      Production applications
```

---

## 2. Design Principles

Every architectural decision in this system flows from six principles. When in doubt, these are the tiebreakers.

| Principle | What It Means in Practice |
|---|---|
| **User never waits** | Prediction always returns immediately, even during retraining |
| **Agents only where needed** | Simple math stays simple code. LLM overhead only for genuinely complex decisions |
| **No message loss** | Every job persists in Valkey Streams until a worker explicitly confirms it via XACK |
| **Open source clean** | Every dependency is OSI-approved. No legal ambiguity for GC.OS review |
| **Locally testable** | Full system runs on one laptop with Docker Compose. No cloud required for dev |
| **Explainable** | Every design decision has a one-line reason. No magic, no black boxes |

---

## 3. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      User / Application                      │
└───────────────────────────┬─────────────────────────────────┘
                            │ HTTP / gRPC
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                     Go API Gateway                           │
│  - Request validation                                        │
│  - Correlation ID generation                                 │
│  - Direct Valkey cache check (cache hits never hit Python)   │
│  - BLPOP result:{correlation_id} (blocking, no polling)      │
└───────────────────────────┬─────────────────────────────────┘
                            │ XADD forecast:jobs
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                   Valkey Streams                             │
│  forecast:jobs     → prediction job queue                    │
│  retrain:jobs      → background retrain queue                │
│  result:{uuid}     → response delivery                       │
│  pred:{ver}:{id}   → prediction cache                        │
│  retrain_lock:{id} → deduplication lock                      │
│  model_lock:{id}   → model update lock                       │
│  model_updated:{id}→ promotion signal                        │
└───────────────────────────┬─────────────────────────────────┘
                            │ XREADGROUP
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              Python Worker (stateless container)             │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │                    Orchestrator                       │   │
│  │  - Routes requests to correct agent                  │   │
│  │  - Manages Valkey locks                              │   │
│  │  - Holds in-memory model cache                       │   │
│  └──────┬──────────────┬───────────────┬────────────────┘   │
│         │              │               │                     │
│         ▼              ▼               ▼                     │
│  ┌────────────┐ ┌──────────────┐ ┌───────────────┐          │
│  │  Model     │ │  Prediction  │ │     Drift     │          │
│  │  Selector  │ │    Agent     │ │    Monitor    │          │
│  │  Agent     │ │              │ │               │          │
│  └────────────┘ └──────┬───────┘ └───────┬───────┘          │
│                        │                 │                   │
│                        ▼                 ▼                   │
│               Valkey Cache        XADD retrain:jobs          │
└────────────────────────┬─────────────────────────────────────┘
                         │ XACK + write result:{uuid}
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                 Background: TrainingAgent                    │
│  - Triggered only by DriftMonitor                           │
│  - Incremental update or full retrain                       │
│  - Saves to MLflow + S3/GCS                                 │
│  - Promotes model to Production stage                       │
└─────────────────────────────────────────────────────────────┘
```

---

## 4. Data Flow — Step by Step

### Normal Prediction Request (Cache Miss)

**Step 1 — Request arrives**
User sends `POST /forecast` to Go gateway with `dataset_id`, `fh` (forecasting horizon), and optional metadata. Go validates the request struct.

**Step 2 — Cache check in Go**
Go constructs cache key `pred:{current_model_version}:{dataset_id}:{fh_hash}` and checks Valkey directly. If hit → return immediately. No Python involved. ~1ms latency.

**Step 3 — Job published**
On cache miss, Go generates a UUID `correlation_id` and does:
```
XADD forecast:jobs * job_id {uuid} dataset_id {id} fh {horizon}
```
Go then does `BLPOP result:{uuid} 30` — blocks for up to 30 seconds. No polling, no busy loop.

**Step 4 — Worker picks up**
Python worker in consumer group does:
```
XREADGROUP GROUP workers worker-1 COUNT 1 STREAMS forecast:jobs >
```
If the worker crashes before XACK, Valkey holds the message. Another worker claims it via `XCLAIM` after a configurable timeout (default 60s).

**Step 5 — Orchestrator routes**
Orchestrator checks: is there a Production model in MLflow for this `dataset_id`?
- Yes → go directly to PredictionAgent
- No → ModelSelectorAgent first, then PredictionAgent

**Step 6 — Prediction**
PredictionAgent checks in-memory model cache. If current model version is loaded → run `predict()` directly. If not → fetch from MLflow (one-time per version), load into memory, run `predict()`. Cache result in Valkey.

**Step 7 — Drift check (non-blocking)**
After prediction, DriftMonitor runs as a fire-and-forget async task:
```python
asyncio.create_task(self.drift_monitor.check(job, result))
```
User response is never held for this.

**Step 8 — Result delivery**
Python worker writes to `result:{uuid}` in Valkey (TTL 60s), sends `XACK`. Go's `BLPOP` unblocks, returns JSON to user.

---

### Drift Detected — Tiered Response

Drift handling is tiered based on severity. The correct response depends on how long the fix takes.

```
Drift Detected
      │
      ├── MINOR drift (residual creep, small CUSUM score)
      │   → Incremental update: forecaster.update(y_new)
      │   → Expected time: 2–15 seconds
      │   → Hold request for up to incremental_update_wait_seconds (default: 10s)
      │   → If update finishes in time: serve fresh prediction, status="updated"
      │   → If timeout hit: serve old prediction, status="drift_minor" + warning
      │
      └── MAJOR drift (ADWIN trigger, large score)
          → Full retrain required
          → Expected time: minutes to hours
          → NEVER hold the request
          → Serve immediately with status="drift_major" + warning
          → XADD retrain:jobs (background)
```

**Why tiered?** A 10-second wait for an incremental update is often better than serving a systematically wrong prediction. A 15-minute wait for a full retrain is never acceptable. The threshold between minor and major is configurable per deployment.

---

### Background Retrain Flow

1. DriftMonitor publishes to `retrain:jobs` stream
2. Before publishing: checks `retrain_lock:{dataset_id}` — if exists, retrain already queued/running, skip (prevents duplicate retrains from multiple drift triggers)
3. TrainingAgent picks up job via `XREADGROUP`
4. Acquires `model_lock:{dataset_id}` with TTL = `max_expected_training_time * 1.5`
5. Determines strategy: cold start / incremental `.update()` / full retrain
6. Validates new model CV score against current Production model score
7. If better: promotes to Production in MLflow, sets `model_updated:{dataset_id}` signal in Valkey
8. Releases lock, sends `XACK`
9. Next PredictionAgent call sees `model_updated` signal, reloads model into memory
10. Old cache entries become unreachable automatically (model version baked into cache key)

---

## 5. Python Layer — The Agentic Brain

### 5.1 Orchestrator

**What it is:** A coordinator class, not an agent. Plain Python logic (~80 lines).

**Responsibilities:**
- Listens to `forecast:jobs` Valkey Stream via `XREADGROUP`
- Decides flow per request (model selection needed? prediction only?)
- Acquires and releases Valkey locks before any model state change
- Holds in-memory model store (`dict[str, BaseForecaster]`)
- On startup: checks for orphaned locks from previous crashes and releases them

**Lock TTL design:**
```
model_lock TTL = max_expected_training_time * 1.5
```
If TrainingAgent crashes mid-retrain, the lock expires automatically. On worker restart, Orchestrator scans for stale locks (age > TTL) and cleans them up.

**Why not an agent:** Coordination logic is deterministic. Every decision is an if/else, not a reasoning problem. Adding LLM overhead here would slow every single request and make failures unpredictable and hard to debug.

```python
class Orchestrator:
    def __init__(
        self,
        valkey: redis.asyncio.Redis,
        mlflow_client: MlflowClient,
        settings: Settings
    ):
        self.valkey = valkey
        self.mlflow = mlflow_client
        self.settings = settings
        self.model_cache: dict[str, BaseForecaster] = {}
        self.model_selector = ModelSelectorAgent(valkey, mlflow_client, settings)
        self.prediction_agent = PredictionAgent(valkey, mlflow_client, settings)
        self.drift_monitor = DriftMonitor(valkey, settings)

    async def handle_job(self, job: ForecastRequest) -> ForecastResponse:
        # Check if model promotion happened since last request
        await self._maybe_reload_model(job.dataset_id)

        # Check if production model exists
        model_version = self._get_production_version(job.dataset_id)

        if model_version is None:
            # First ever request for this dataset
            model_version = await self.model_selector.select(job)

        result = await self.prediction_agent.predict(job, model_version, self.model_cache)

        # Fire-and-forget drift check — never blocks user
        asyncio.create_task(self.drift_monitor.check(job, result))

        return result

    async def _maybe_reload_model(self, dataset_id: str) -> None:
        signal = await self.valkey.get(f"model_updated:{dataset_id}")
        if signal:
            new_version = self._get_production_version(dataset_id)
            model = mlflow.sktime.load_model(f"models:/{dataset_id}/Production")
            self.model_cache[dataset_id] = model
            await self.valkey.delete(f"model_updated:{dataset_id}")

    async def _cleanup_orphaned_locks(self) -> None:
        # Called on worker startup
        async for key in self.valkey.scan_iter("model_lock:*"):
            ttl = await self.valkey.ttl(key)
            if ttl < 0:  # no TTL set — orphaned
                await self.valkey.delete(key)
```

---

### 5.2 ModelSelectorAgent

**What it is:** The only true "agent" in the Python layer. Uses one focused LLM call for ambiguous selection decisions.

**Selection pipeline (in order):**

```
1. Profile the dataset
   → length, frequency, seasonality, missingness, variance, stationarity

2. Rule-based fast path (no LLM)
   → short series (<100 obs)    : ARIMA, ETS
   → long seasonal              : Prophet, BATS, TBATS
   → multivariate               : VAR, VARMAX
   → irregular frequency        : NaiveForecaster + interpolation

3. If ambiguous → one LLM API call
   → describe data profile in natural language
   → ask for ranked list from ALLOWED_ESTIMATORS whitelist
   → parse JSON response
   → validate all returned estimators are in whitelist

4. If LLM call fails → MultiplexForecaster + ForecastingGridSearchCV

5. Ultimate fallback → NaiveForecaster(strategy="last")
   (system never crashes, always returns something)
```

**Why direct SDK, not LangChain:**

| | Direct Anthropic/OpenAI SDK | LangChain |
|---|---|---|
| Dependencies | 1 package | 200+ transitive |
| Container image size | Small | Large |
| Explainability | 5 lines, obvious | AgentExecutor chain, opaque |
| Debugging | Stack trace is clear | Hard to trace through abstractions |
| Interview explanation | "It calls the API directly" | "It goes through LangChain's..." |

**LLM call:**
```python
async def _llm_select(self, profile: DataProfile) -> list[str]:
    response = self.llm_client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=500,
        messages=[{
            "role": "user",
            "content": (
                f"Given this time series profile:\n{profile.to_natural_language()}\n\n"
                f"Rank these sktime estimators by expected performance:\n"
                f"{ALLOWED_ESTIMATORS}\n\n"
                f"Return JSON only: {{\"ranked\": [\"estimator1\", \"estimator2\"]}}"
            )
        }]
    )
    raw = response.content[0].text
    parsed = json.loads(raw)
    # Validate against whitelist — never trust LLM output directly
    return [e for e in parsed["ranked"] if e in ALLOWED_ESTIMATORS]
```

**Serialization safety — Week 1 task:**
Not all sktime estimators survive MLflow serialization. Some use C extensions that break pickle. The `ALLOWED_ESTIMATORS` whitelist must be validated against `mlflow.sktime.save_model()` in the first week before building anything else. Any estimator that fails serialization is removed from the whitelist permanently.

**sktime-mcp integration (stretch goal):**
ModelSelectorAgent can optionally use `sktime-mcp` tools to query sktime's own documentation for estimator recommendations, making the selection genuinely grounded in sktime's API surface rather than just LLM training data.

---

### 5.3 PredictionAgent

**Responsibilities:**
- Checks in-memory model cache (fastest path)
- Checks Valkey prediction cache (second path)
- Loads from MLflow if neither is available (one-time per version)
- Runs `predict()`, `predict_interval()`, or `predict_proba()` depending on request
- Writes result to Valkey cache with version-baked key
- Always includes `model_status`, `drift_score`, and `warning` in response

**Cache key format:**
```
pred:{model_version}:{dataset_id}:{fh_hash}
```
Version is baked into the key. When a new model is promoted, old keys become unreachable automatically — no explicit flush needed, no race condition on promotion.

**Cache TTL strategy (configurable):**
```python
# High-frequency data (hourly, sub-hourly)
cache_ttl = 900   # 15 minutes

# Daily data
cache_ttl = 21600  # 6 hours

# Weekly/monthly data
cache_ttl = 86400  # 24 hours
```

**Data never goes in Valkey:**
Raw time series data travels as `dataset_id` only. Python worker fetches actual data directly from S3/GCS using the ID. Putting time series in Valkey/Redis would be huge, lossy, and wrong.

**Hard minimum history check:**
```python
if len(y) < self.settings.min_history_length:
    raise InsufficientHistoryError(
        f"Dataset {job.dataset_id} has {len(y)} observations. "
        f"Minimum required: {self.settings.min_history_length}."
    )
```
Returns HTTP 422 — clear error, never attempts a meaningless prediction.

**Response always includes model metadata:**
```python
return ForecastResponse(
    predictions=[...],
    model_version=current_version,
    model_class="AutoARIMA",
    model_status="stable",      # or "drift_minor" / "drift_major" / "updated"
    drift_score=None,
    warning=None,
    cache_hit=False,
    correlation_id=job.correlation_id
)
```

---

### 5.4 DriftMonitor

**What it is:** A statistical check function. Normal Python logic, not an agent.

**Trigger conditions:**
- Every N predictions (configurable, default: 50)
- Every T minutes (configurable, default: 10)
- Whichever comes first

**Detection methods (in order of compute cost):**

| Method | When Used | Cost |
|---|---|---|
| Residual error tracking | Always (baseline) | Negligible |
| CUSUM | When residuals trend consistently in one direction | Low |
| ADWIN | When distribution shift is suspected | Medium |

**Drift severity thresholds:**
```python
# Below this: no action
NO_DRIFT_THRESHOLD = 0.2

# Between these: minor drift → incremental update
MINOR_DRIFT_THRESHOLD = 0.5

# Above this: major drift → full retrain
MAJOR_DRIFT_THRESHOLD = 0.5
```

**Deduplication — prevents multiple retrain jobs:**
```python
async def _maybe_trigger_retrain(self, dataset_id: str, reason: str) -> None:
    lock_key = f"retrain_lock:{dataset_id}"
    already_queued = await self.valkey.exists(lock_key)
    if already_queued:
        return  # retrain already running or queued — skip
    await self.valkey.setex(lock_key, self.settings.retrain_lock_ttl_seconds, "1")
    await self.valkey.xadd("retrain:jobs", {
        "dataset_id": dataset_id,
        "reason": reason,
        "triggered_at": datetime.utcnow().isoformat()
    })
```

**Why not a full agent:** Drift detection is pure statistics. There is no ambiguity, no natural language reasoning, no need for LLM involvement. Making it an agent would slow every post-prediction step and introduce failure modes with no benefit.

---

### 5.5 TrainingAgent

**Triggered by:** DriftMonitor via `retrain:jobs` stream only. Never triggered directly by user requests.

**Training strategy selection:**
```
Is this the first ever train?
├── Yes → cold start: full fit on all available history
└── No →
    Minor drift?
    ├── Yes → forecaster.update(y_new)  [incremental, fast]
    └── No  → full retrain from scratch
```

**MLflow run tagging:**
Every training run is tagged with:
```python
mlflow.set_tags({
    "drift_reason": job.reason,          # CUSUM / ADWIN / residual / manual
    "dataset_id": job.dataset_id,
    "previous_version": previous_version,
    "estimator_class": type(forecaster).__name__,
    "sktime_version": sktime.__version__,
})
mlflow.log_metrics({
    "cv_score": cv_score,
    "training_duration_seconds": duration,
})
```

**Promotion gate:** New model only promoted to Production if its CV score strictly beats the current Production model's CV score. Prevents a retrain from actually making things worse.

**Model lifecycle stages:**
```
Staging   → TrainingAgent just finished, awaiting validation
Production → validated, PredictionAgent uses this
Archived  → previous Production, kept for rollback
```

**Lock behavior:**
```python
lock_ttl = int(self.settings.max_training_time_seconds * 1.5)
acquired = await self.valkey.set(
    f"model_lock:{dataset_id}",
    "1",
    nx=True,    # only set if not exists
    ex=lock_ttl
)
if not acquired:
    return  # another worker is already training this dataset
```

---

## 6. Go Layer — Production Wrapper

### Why Go, Not Python for the Gateway

Python is the right language for time series intelligence. Go is the right language for a production API gateway — high concurrency with goroutines, low memory overhead, single binary deployment, and microsecond-level routing. Using both is the standard industry pattern for ML systems at scale. The entire agentic intelligence lives in Python and is fully usable as a standalone service without Go.

### Go API Gateway — Responsibilities

- Fast HTTP/REST and gRPC endpoints
- Request struct validation (Go validator library)
- Direct Valkey cache check — cache hits never touch Python (~1ms vs ~50ms)
- Correlation ID generation (UUID v4)
- `XADD` to `forecast:jobs` stream
- `BLPOP result:{uuid}` with configurable timeout (default 30s)
- Timeout handling — returns HTTP 503 with `Retry-After` header
- Health and readiness endpoints for Kubernetes probes
- Prometheus metrics endpoint

### Endpoints

```
POST /forecast              → main prediction endpoint
GET  /health                → liveness probe (always 200 if process is alive)
GET  /ready                 → readiness probe (checks Valkey connection)
GET  /metrics               → Prometheus metrics
POST /admin/retrain         → manual retrain trigger (authenticated, admin only)
GET  /admin/model/{dataset} → current Production model info for a dataset
```

### Request Handling — Go Code Sketch

```go
func (h *Handler) Forecast(c *fiber.Ctx) error {
    var req ForecastRequest
    if err := c.BodyParser(&req); err != nil {
        return c.Status(400).JSON(ErrorResponse{Error: "invalid request"})
    }
    if err := validate.Struct(req); err != nil {
        return c.Status(422).JSON(ErrorResponse{Error: err.Error()})
    }

    // Direct cache check in Go — bypass Python entirely
    cacheKey := fmt.Sprintf("pred:%s:%s:%s", currentModelVersion, req.DatasetID, hashFH(req.FH))
    cached, err := h.valkey.Get(ctx, cacheKey).Result()
    if err == nil {
        return c.JSON(cached) // cache hit — done
    }

    // Cache miss — send to Python
    correlationID := uuid.New().String()
    h.valkey.XAdd(ctx, &redis.XAddArgs{
        Stream: "forecast:jobs",
        Values: map[string]interface{}{
            "job_id":         correlationID,
            "dataset_id":     req.DatasetID,
            "fh":             req.FH,
            "correlation_id": correlationID,
        },
    })

    // Block until Python writes result — no polling
    result, err := h.valkey.BLPop(ctx, 30*time.Second, "result:"+correlationID).Result()
    if err != nil {
        return c.Status(503).
            Set("Retry-After", "5").
            JSON(ErrorResponse{Error: "prediction timeout, please retry"})
    }

    return c.JSON(result[1])
}
```

---

## 7. Infrastructure Layer

### 7.1 Valkey Streams

**Why Valkey, not Redis:**
Redis changed its license in March 2024 from BSD-3 to RSALv2 + SSPLv1. Neither license is OSI-approved. SSPL specifically targets service providers — anyone offering Redis as a service must open-source their entire stack. For an ESoC project submitted to GC.OS (a strictly open source organisation), a non-OSI-approved dependency is a real compliance risk.

Valkey is a Linux Foundation fork of Redis created immediately after the license change. It remains on BSD-3-Clause, is fully API-compatible with Redis, and is maintained by contributors from AWS, Google, and Oracle. The `redis-py` Python client and `go-redis` client both work with Valkey without any code changes.

**Why Streams, not Pub/Sub:**
Redis/Valkey Pub/Sub is fire-and-forget. If a worker is down for even one second when a message arrives, that message is permanently lost with no retry. Streams persist messages until a worker explicitly acknowledges them via `XACK`. If a worker crashes mid-job, another worker claims the message via `XCLAIM` after a timeout.

**Stream and key layout:**

| Key | Type | Purpose | TTL |
|---|---|---|---|
| `forecast:jobs` | Stream | Prediction job queue | No TTL (XACK deletes) |
| `retrain:jobs` | Stream | Background retrain queue | No TTL (XACK deletes) |
| `result:{uuid}` | String | Response delivery to Go | 60 seconds |
| `pred:{ver}:{id}:{fh}` | String | Prediction cache | Configurable (15min–24h) |
| `retrain_lock:{id}` | String | Retrain deduplication | 30 minutes |
| `model_lock:{id}` | String | Model update lock | `max_train_time * 1.5` |
| `model_updated:{id}` | String | Promotion signal to workers | 5 minutes |

**Consumer group setup:**
```bash
XGROUP CREATE forecast:jobs workers $ MKSTREAM
XGROUP CREATE retrain:jobs trainers $ MKSTREAM
```

---

### 7.2 MLflow + Cloud Storage

**Single source of truth for all models and datasets.**

**Local development:** MinIO (S3-compatible) — full stack runs locally without AWS.

**Production:** AWS S3 or GCS, configurable via `MLFLOW_ARTIFACT_URI` environment variable.

**Model registry flow:**
```
mlflow.sktime.log_model()   → saves to Staging
validate CV score           → if better than Production:
mlflow.transition_stage()   → Staging → Production
                            → old Production → Archived
```

**Serialization whitelist validation (Week 1 priority):**
Not every sktime estimator survives `mlflow.sktime.save_model()`. Some use C extensions that break pickle serialization. Run this validation before building any agent logic:

```python
CANDIDATE_ESTIMATORS = [
    "AutoARIMA", "AutoETS", "Prophet", "BATS", "TBATS",
    "ThetaForecaster", "ExponentialSmoothing", "NaiveForecaster",
    "PolynomialTrendForecaster", "STLForecaster",
]

ALLOWED_ESTIMATORS = []
for name in CANDIDATE_ESTIMATORS:
    try:
        estimator = registry[name]()
        estimator.fit(y_test)
        with tempfile.TemporaryDirectory() as tmp:
            mlflow.sktime.save_model(estimator, tmp)
        ALLOWED_ESTIMATORS.append(name)
    except Exception as e:
        print(f"EXCLUDED {name}: {e}")
```

Only estimators that pass this test go into the whitelist. Any LLM suggestion outside the whitelist is rejected.

---

### 7.3 Kubernetes

**v1 — Basic Deployments (achievable in one summer):**

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: python-worker
spec:
  replicas: 2
  template:
    spec:
      containers:
        - name: worker
          image: sktime-agentic-worker:latest
          envFrom:
            - secretRef:
                name: sktime-agentic-secrets
          livenessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 10
            periodSeconds: 30
          readinessProbe:
            httpGet:
              path: /ready
              port: 8000
            initialDelaySeconds: 5
            periodSeconds: 10
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: go-gateway
spec:
  replicas: 2
  template:
    spec:
      containers:
        - name: gateway
          image: sktime-agentic-gateway:latest
          ports:
            - containerPort: 8080
          livenessProbe:
            httpGet:
              path: /health
              port: 8080
          readinessProbe:
            httpGet:
              path: /ready
              port: 8080
```

**Why liveness and readiness probes matter:**
Without probes, Kubernetes cannot distinguish between a worker stuck in a long training job and a worker that has actually crashed. The readiness probe checks Valkey connectivity — if Valkey is unreachable, the pod is marked not-ready and removed from the load balancer automatically.

**v2 stretch goal — KEDA autoscaling:**
Scale Python workers based on `forecast:jobs` stream length. Zero workers when no jobs, scales up under load. Not included in v1 to keep scope achievable.

---

## 8. API Reference

### POST /forecast

**Request:**
```json
{
  "dataset_id": "sensor-42-temperature",
  "fh": [1, 2, 3, 4, 5],
  "frequency": "H"
}
```

**Response (stable):**
```json
{
  "dataset_id": "sensor-42-temperature",
  "predictions": [22.4, 22.7, 23.1, 23.0, 22.8],
  "prediction_intervals": {
    "lower": [21.1, 21.3, 21.6, 21.4, 21.2],
    "upper": [23.7, 24.1, 24.6, 24.6, 24.4]
  },
  "model_version": "3",
  "model_class": "AutoARIMA",
  "model_status": "stable",
  "drift_score": null,
  "warning": null,
  "cache_hit": false,
  "correlation_id": "f47ac10b-58cc-4372-a567-0e02b2c3d479"
}
```

**Response (drift detected — minor):**
```json
{
  "predictions": [22.4, 22.7, 23.1, 23.0, 22.8],
  "model_status": "drift_minor",
  "drift_score": 0.38,
  "warning": "Minor drift detected (CUSUM score: 0.38). Incremental update in progress. Predictions may be slightly less accurate until update completes.",
  "cache_hit": false
}
```

**Response (drift detected — major):**
```json
{
  "predictions": [22.4, 22.7, 23.1, 23.0, 22.8],
  "model_status": "drift_major",
  "drift_score": 0.81,
  "warning": "Significant drift detected (ADWIN, score: 0.81). Full model retraining has been triggered in the background. Current predictions are from the previous model version and may be less accurate.",
  "cache_hit": false
}
```

**Response (updated — incremental update completed before timeout):**
```json
{
  "predictions": [22.1, 22.4, 22.9, 22.7, 22.5],
  "model_version": "4",
  "model_status": "updated",
  "drift_score": 0.38,
  "warning": null,
  "cache_hit": false
}
```

### GET /health
Returns `200 OK` if the process is alive. Always returns 200 — used for liveness probe only.

### GET /ready
Returns `200 OK` if Valkey connection is healthy. Returns `503` if Valkey is unreachable. Used for readiness probe — pod is removed from load balancer on 503.

### POST /admin/retrain
Manually triggers a retrain for a dataset. Requires authentication header.
```json
{ "dataset_id": "sensor-42-temperature", "reason": "manual" }
```

### GET /admin/model/{dataset_id}
Returns current Production model info.
```json
{
  "dataset_id": "sensor-42-temperature",
  "model_version": "3",
  "model_class": "AutoARIMA",
  "cv_score": 0.94,
  "promoted_at": "2026-07-14T10:23:00Z",
  "drift_reason": "CUSUM"
}
```

---

## 9. Schemas

### Python — Pydantic v2

```python
from pydantic import BaseModel, field_validator
from pydantic_settings import BaseSettings
from datetime import datetime

class ForecastRequest(BaseModel):
    dataset_id: str
    fh: list[int]           # e.g. [1, 2, 3] for next 3 steps
    correlation_id: str     # generated by Go, passed through
    frequency: str | None = None  # "H", "D", "W", "M" — optional hint

    @field_validator("fh")
    def fh_must_be_positive(cls, v):
        if any(h <= 0 for h in v):
            raise ValueError("all horizon values must be positive integers")
        return v

class ForecastResponse(BaseModel):
    dataset_id: str
    predictions: list[float]
    prediction_intervals: dict | None = None
    model_version: str
    model_class: str
    model_status: str       # "stable" | "updated" | "drift_minor" | "drift_major" | "retraining"
    drift_score: float | None = None
    drift_method: str | None = None   # "CUSUM" | "ADWIN" | "residual"
    warning: str | None = None
    cache_hit: bool
    correlation_id: str

class RetrainJob(BaseModel):
    dataset_id: str
    reason: str             # "CUSUM" | "ADWIN" | "residual" | "manual"
    triggered_at: datetime

class DataProfile(BaseModel):
    dataset_id: str
    length: int
    frequency: str | None
    has_seasonality: bool
    is_stationary: bool
    missing_rate: float
    variance: float

    def to_natural_language(self) -> str:
        return (
            f"Time series with {self.length} observations, "
            f"frequency={self.frequency}, "
            f"seasonality={'yes' if self.has_seasonality else 'no'}, "
            f"stationary={'yes' if self.is_stationary else 'no'}, "
            f"missing rate={self.missing_rate:.1%}, "
            f"variance={self.variance:.4f}"
        )
```

---

## 10. Configuration

All configuration via environment variables. Never hardcoded. Never logged (especially API keys).

```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Valkey
    valkey_url: str = "redis://localhost:6379"

    # MLflow
    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_artifact_uri: str = "s3://sktime-agentic-models"

    # Storage
    s3_bucket: str
    s3_region: str = "us-east-1"

    # Prediction behaviour
    min_history_length: int = 10
    prediction_timeout_seconds: int = 30
    incremental_update_wait_seconds: int = 10

    # Cache TTL
    cache_ttl_high_frequency_seconds: int = 900   # 15 min
    cache_ttl_daily_seconds: int = 21600           # 6 hours
    cache_ttl_low_frequency_seconds: int = 86400   # 24 hours

    # Drift thresholds
    drift_check_every_n_predictions: int = 50
    drift_check_every_n_minutes: int = 10
    minor_drift_threshold: float = 0.3
    major_drift_threshold: float = 0.5

    # Locks
    retrain_lock_ttl_seconds: int = 1800           # 30 min
    max_training_time_seconds: int = 3600          # 1 hour
    # model_lock TTL = max_training_time * 1.5 (computed)

    # LLM
    llm_provider: str = "anthropic"                # "anthropic" | "openai"
    llm_api_key: str                               # from env, never logged
    llm_model: str = "claude-sonnet-4-20250514"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
```

---

## 11. Tech Stack

### Python Layer

| Package | Version | Purpose |
|---|---|---|
| `sktime` | latest | Core forecasting library |
| `pandas` | >=2.0 | Data manipulation |
| `numpy` | >=1.26 | Numerical operations |
| `mlflow` | latest | Model registry and tracking |
| `mlflow-sktime` | latest | sktime MLflow flavour |
| `redis` | >=4.2 | Valkey client (`redis.asyncio` — NOT aioredis, which is deprecated and unmaintained) |
| `fastapi` | latest | Health and debug endpoints only |
| `pydantic` | >=2.0 | Request/response schemas |
| `pydantic-settings` | latest | Environment variable configuration |
| `anthropic` | latest | LLM API — direct SDK, no LangChain |
| `alibi-detect` or `river` | latest | ADWIN drift detection |
| `pytest` | latest | Testing |
| `pytest-asyncio` | latest | Async test support |

### Go Layer

| Package | Purpose |
|---|---|
| `Go 1.23+` | Language |
| `gofiber/fiber/v2` | HTTP framework (lower memory than Gin) |
| `go-redis/redis/v9` | Valkey client |
| `go-playground/validator/v10` | Request struct validation |
| `google/uuid` | Correlation ID generation |

### Infrastructure

| Component | Choice | License |
|---|---|---|
| Async queue | Valkey Streams | BSD-3-Clause |
| Cache + locks | Valkey | BSD-3-Clause |
| Model registry | MLflow | Apache 2.0 |
| Cloud storage | S3 / GCS / MinIO (local) | Apache 2.0 (MinIO AGPL — local dev only) |
| Containerisation | Docker multi-stage builds | Apache 2.0 |
| Orchestration | Kubernetes | Apache 2.0 |
| Core ML | sktime | BSD-3-Clause |

**All production dependencies are OSI-approved open source licenses.**

---

## 12. Constraints and Tradeoffs

| Decision | What We Chose | What We Gave Up | Why |
|---|---|---|---|
| Async queue | Valkey Streams + XACK | Full Kafka | Kafka requires Zookeeper/KRaft — separate operational burden. Streams give persistence and consumer groups with zero extra infrastructure. Kafka is a v2 upgrade path. |
| Valkey not Redis | Valkey (BSD-3-Clause) | Redis (RSALv2 + SSPLv1) | OSI compliance for GC.OS open source submission. Valkey is API-compatible — zero code changes. |
| Pub/Sub → Streams | XADD / XREADGROUP / XACK | Simpler Pub/Sub | Pub/Sub loses messages if worker is down for even one second. Streams persist until XACK — no message loss. |
| Go checks cache directly | Go reads Valkey directly | All traffic through Python | Eliminates entire Python round-trip for cache hits. ~50x latency improvement for repeated identical requests. |
| Correlation ID + BLPOP | Blocking pop on `result:{uuid}` | Polling loop | No busy loop, no race condition. Go blocks until its exact result arrives. Clean and correct. |
| No LangChain | Direct Anthropic/OpenAI SDK | LangChain ecosystem | One focused LLM call needs 5 lines, not a framework. 200+ fewer transitive dependencies. Easier to test, debug, explain in interview. |
| DriftMonitor as plain logic | Normal Python function | Full DriftAgent | Drift detection is statistics, not reasoning. No LLM needed. Agent overhead would slow every post-prediction step. |
| In-memory model | Model loaded in worker process | Fresh MLflow fetch per request | MLflow + S3 fetch costs 2–5 seconds. Unacceptable for a prediction API. Reload only on model version change. |
| Cache key includes version | `pred:{ver}:{id}:{fh}` | Explicit cache flush on retrain | Old cache entries become unreachable automatically after model promotion. No flush needed, no race condition. |
| Tiered drift response | Wait for minor, background for major | Single policy for all drift | A 10-second wait for incremental update beats a wrong prediction. A 15-minute wait for full retrain never acceptable. |
| Warning in output | `model_status` + `warning` in every response | Clean minimal response | User deserves to know prediction reliability. Silent wrong answer is worst outcome in any forecasting system. |
| NaiveForecaster fallback | Always have a fallback estimator | Refusing on no model | System never crashes. Returns something reasonable while real model trains. |
| Hard minimum history | HTTP 422 if `len(y) < min_history_length` | Attempting prediction on tiny data | NaiveForecaster on 3 data points is meaningless. Clear error is better than silent nonsense. |
| K8s basic Deployments | Static replica count v1 | KEDA event-driven autoscaling | KEDA adds operational complexity. Manual scaling is sufficient for ESoC. KEDA is clearly documented as v2. |
| MLflow model staging | Staging → Production → Archived | Simple file storage | Full audit trail, rollback capability, CV score gating before promotion. |
| No data in Valkey | `dataset_id` reference only | Convenience of passing data directly | Raw time series in JSON is large and lossy. S3/GCS is the correct store for binary blobs. |
| Python layer standalone | Works with FastAPI alone, no Go required | Go is mandatory | Lowers barrier to adoption. Pure Python users can deploy without Go. Go adds production scale. |

**Biggest constraint overall:** 12-week ESoC project, one person. Everything in v1 is scoped so the Python agentic layer works end-to-end even if Go/K8s parts are placeholders. The intelligence is in Python — the infra is the wrapper.

---

## 13. Error Handling and Failure Modes

| Failure | What Happens | Recovery |
|---|---|---|
| Worker crashes mid-prediction | Message stays in Valkey Stream. Another worker claims it via `XCLAIM` after timeout. | Automatic — no data loss. |
| Worker crashes mid-retrain | `model_lock` TTL expires. Next retrain trigger sees no lock and starts fresh. | Automatic — lock TTL prevents permanent deadlock. |
| MLflow unreachable | PredictionAgent uses last known in-memory model. Returns `model_status: "stale_registry"` warning. | Graceful degradation — predictions continue. |
| LLM API call fails | ModelSelectorAgent falls back to `MultiplexForecaster + GridSearchCV`, then `NaiveForecaster`. | Automatic cascade — system never blocks on LLM availability. |
| Go timeout (30s) | Returns HTTP 503 with `Retry-After: 5` header. | Client retries. Worker continues and writes result — next request will cache hit. |
| Valkey unreachable | Go readiness probe returns 503. Pod removed from load balancer by K8s. | K8s handles routing away automatically. |
| New model performs worse than old | Promotion gate rejects it. Old Production model stays. MLflow run tagged "rejected". | Automatic — no regression possible through retraining. |
| True cold start (no history at all) | HTTP 422 with clear message: "insufficient history". | User must provide data before predictions are possible. |
| Orphaned lock on startup | Orchestrator startup routine scans for locks with no TTL and deletes them. | Automatic on every worker restart. |
| Duplicate retrain triggers | `retrain_lock:{dataset_id}` prevents publishing multiple retrain jobs. | Automatic deduplication. |

---

## 14. Local Development Setup

Full system runs on one laptop with Docker Compose. No cloud account required.

```yaml
# docker-compose.yml
services:

  valkey:
    image: valkey/valkey:8
    ports:
      - "6379:6379"
    volumes:
      - valkey-data:/data

  mlflow:
    image: ghcr.io/mlflow/mlflow:latest
    ports:
      - "5000:5000"
    command: >
      mlflow server
      --host 0.0.0.0
      --backend-store-uri sqlite:///mlflow.db
      --default-artifact-root s3://sktime-agentic-models

  minio:
    image: minio/minio:latest
    ports:
      - "9000:9000"
      - "9001:9001"   # MinIO console
    environment:
      MINIO_ROOT_USER: minioadmin
      MINIO_ROOT_PASSWORD: minioadmin
    command: server /data --console-address ":9001"
    volumes:
      - minio-data:/data

  python-worker:
    build:
      context: ./python
      dockerfile: Dockerfile
    env_file: .env
    depends_on:
      - valkey
      - mlflow
    deploy:
      replicas: 2

  go-gateway:
    build:
      context: ./go
      dockerfile: Dockerfile
    ports:
      - "8080:8080"
    env_file: .env
    depends_on:
      - valkey

volumes:
  valkey-data:
  minio-data:
```

**Start everything:**
```bash
docker compose up --build
```

**Send a test request:**
```bash
curl -X POST http://localhost:8080/forecast \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_id": "test-dataset-1",
    "fh": [1, 2, 3],
    "frequency": "D"
  }'
```

**Python-only mode (no Go):**
The Python layer exposes its own FastAPI endpoints for development and testing without Go:
```bash
cd python
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

---

## 15. Testing Strategy

```
tests/
├── unit/
│   ├── test_model_selector.py      # Rule-based selection logic
│   ├── test_drift_monitor.py       # CUSUM, ADWIN, residual checks
│   ├── test_prediction_agent.py    # Cache hit/miss, stale entry
│   ├── test_orchestrator.py        # Routing logic, lock behaviour
│   └── test_schemas.py             # Pydantic validation
├── integration/
│   ├── test_full_flow.py           # Request → prediction → XACK
│   ├── test_drift_retrain_cycle.py # Drift → retrain → promotion → new prediction
│   └── test_cache_invalidation.py  # Old version unreachable after promotion
└── fixtures/
    ├── sample_datasets/            # Small time series for testing
    └── mock_mlflow/                # MLflow fixtures
```

**Key test cases:**
- Duplicate retrain lock: inject drift twice simultaneously, verify only one retrain job published
- Orphaned lock recovery: kill worker mid-retrain, restart, verify lock is cleaned and retrain can run
- Cache key staleness: promote new model, verify old cache key is unreachable, new key is populated
- LLM fallback: mock LLM API to fail, verify cascade to GridSearchCV then NaiveForecaster
- Minimum history: send dataset with 5 observations, verify HTTP 422 with correct message
- Tiered drift: inject minor drift, verify wait behaviour and `updated` status; inject major drift, verify immediate response with warning

---

## 16. Milestone Table

| Phase | Weeks | Deliverable |
|---|---|---|
| **Pre-start** | Before June 9 | ModelSelectorAgent + PredictionAgent core working locally |
| **Pre-start** | Before June 9 | Orchestrator + Valkey Streams wiring — end-to-end job flow |
| **Pre-start** | Before June 9 | TrainingAgent + MLflow save/load/promote working |
| **Week 1–2** | June 9–20 | DriftMonitor + full background retrain cycle tested |
| **Week 3–4** | June 23 – July 4 | Go gateway + Correlation ID pattern + full round-trip |
| **Week 5–6** | July 7–18 | Cache layer + version-baked invalidation verified |
| **Week 7–8** | July 21 – Aug 1 | Docker Compose full stack + basic K8s Deployments |
| **Week 9–10** | Aug 4–15 | sktime-mcp integration in ModelSelectorAgent |
| **Week 11** | Aug 18–22 | Tutorial notebook + demo video |
| **Week 12** | Aug 25–29 | Final docs, cleanup, companion repo PR to sktime ecosystem |

---

## 17. v1 vs v2 Roadmap

| Feature | v1 (ESoC 2026) | v2 (Future) |
|---|---|---|
| Async queue | Valkey Streams | Apache Kafka |
| Drift detection | CUSUM + ADWIN (statistical) | Full DriftAgent with LLM-generated explanation and root cause |
| K8s scaling | Basic Deployments (manual replicas) | KEDA event-driven autoscaling on stream length |
| Valkey topology | Single instance | Valkey Cluster |
| Model selection | Rule-based + one LLM call | Multi-round agentic selection with self-critique |
| Monitoring | Basic logs + FastAPI metrics | Prometheus + Grafana dashboard with drift history |
| Multi-tenancy | Single namespace | Full tenant isolation per dataset namespace |
| Go → Python protocol | Valkey Streams | Direct gRPC for lower latency |
| LLM integration | Anthropic/OpenAI SDK direct | sktime-mcp tools for grounded estimator selection |

---

## 18. Why This Architecture

### For sktime Reviewers

This system is a true companion to sktime — same philosophy as `sktime-mcp`, but focused on production-grade autonomous deployment. Every sktime API is used as intended: `fit()`, `update()`, `predict()`, `predict_interval()`, `MultiplexForecaster`, `ForecastingGridSearchCV`, MLflow sktime flavour. Nothing is monkey-patched or worked around.

The Go gateway is not a departure from sktime's Python identity. Python handles all forecasting intelligence. Go handles what Python should not: high-concurrency HTTP routing, microsecond-level request handling, and single-binary deployment. The Python layer is fully standalone without Go.

### For Systems Reviewers

Every design decision in this document has a one-line justification. The tradeoffs table does not hide compromises — it names them explicitly and gives the reason for each choice. This is not a wishlist document. v1 is scoped so one person can deliver a working, production-feeling system in 12 weeks. v2 is clearly labelled as future work.

### For the ESoC Agentic Track

The system is fully agentic where agency adds value and uses plain logic where it does not. The ModelSelectorAgent uses LLM reasoning because model selection on an unfamiliar dataset is genuinely ambiguous. The DriftMonitor uses statistics because drift detection is pure math. The Orchestrator uses if/else because coordination logic is deterministic. This distinction — agent where needed, code where sufficient — is the right engineering judgement.

---

*sktime Agentic Forecasting System — ESoC 2026 — GC.OS Track*
*Companion repository to sktime. Not affiliated with the sktime core team.*
*All dependencies OSI-approved. BSD-3-Clause licence.*
