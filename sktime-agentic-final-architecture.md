# sktime-agentic — Final Architecture Design Document
### ESoC 2026 · sktime Agentic Track · Own Idea Category · v3.0 (Merged Final)

---

> **How to read this document.**
> This is your single source of truth. Architecture 1 contributed the agentic brain (ReAct loop, pipeline architect, MCP-native action surface, cross-run memory, hallucination mitigation). Architecture 2 contributed the delivery body (Go gateway, Valkey Streams, deployment layer, milestone table, testing strategy, honest v1/v2 scope). Both are merged here. Nothing is contradicted or duplicated — this is the one document you follow.

---

## Table of Contents

1. [Project Identity](#1-project-identity)
2. [Design Principles](#2-design-principles)
3. [What Makes It Genuinely Agentic](#3-what-makes-it-genuinely-agentic)
4. [Relation to sktime Ecosystem](#4-relation-to-sktime-ecosystem)
5. [High-Level System Architecture](#5-high-level-system-architecture)
6. [Request Lifecycle — End to End](#6-request-lifecycle--end-to-end)
7. [LLM Call Inventory](#7-llm-call-inventory)
8. [Python Layer — The Agentic Brain](#8-python-layer--the-agentic-brain)
   - [Orchestrator](#81-orchestrator)
   - [ModelSelectorAgent](#82-modelselectoragent)
   - [PipelineArchitectAgent — The Agentic Core](#83-pipelinearchitectagent--the-agentic-core)
   - [PredictionAgent](#84-predictionagent)
   - [DriftMonitor](#85-driftmonitor)
   - [TrainingAgent](#86-trainingagent)
   - [Watchdog](#87-watchdog)
9. [Agent Memory](#9-agent-memory)
10. [Hallucination Mitigation](#10-hallucination-mitigation)
11. [MCP Tool Surface](#11-mcp-tool-surface)
12. [Go Layer — Production Gateway](#12-go-layer--production-gateway)
13. [Infrastructure Layer](#13-infrastructure-layer)
    - [Valkey Streams](#131-valkey-streams)
    - [MLflow + Cloud Storage](#132-mlflow--cloud-storage)
    - [Kubernetes Deployment](#133-kubernetes-deployment)
    - [Docker Compose — Local Development](#134-docker-compose--local-development)
14. [API Reference](#14-api-reference)
15. [Schemas](#15-schemas)
16. [Configuration](#16-configuration)
17. [Error Handling and Failure Modes](#17-error-handling-and-failure-modes)
18. [Testing Strategy](#18-testing-strategy)
19. [Upstream Contributions to sktime-mcp](#19-upstream-contributions-to-sktime-mcp)
20. [Milestone Table](#20-milestone-table)
21. [v1 vs v2 Roadmap](#21-v1-vs-v2-roadmap)
22. [Repository Layout](#22-repository-layout)

---

## 1. Project Identity

### One-Sentence Summary

An LLM-driven autonomous time-series forecasting system where an agent constructs novel sktime pipelines, retains cross-run production memory, and makes every lifecycle decision — served through a production-grade Go API gateway — without any human intervention.

### What This Is

`sktime-agentic` is a **companion repository** to `sktime` and `sktime-mcp` — same positioning as `sktime-mcp` but focused on production-grade autonomous deployment rather than MCP tooling. It does not modify or fork the `sktime` core. It wraps sktime's capabilities in an agentic intelligence layer and a deployment layer that make it self-managing at scale.

The system does not just predict. It **decides how to predict, constructs the pipeline that does it, remembers every decision it has ever made, and corrects itself when things go wrong** — without any human in the loop.

### What This Is Not

- Not a fork or modification of `sktime`
- Not a replacement for `sktime-mcp` — it depends on it
- Not a LangChain wrapper — direct Anthropic SDK, zero framework magic
- Not a research notebook — the Python layer is fully standalone and production-deployable
- Not a monolith — Python owns ML intelligence, Go owns HTTP serving

---

## 2. Design Principles

Every architectural decision flows from these principles. When there is ambiguity, these are the tiebreakers.

| Principle | What It Means in Practice |
|---|---|
| **LLM where reasoning is needed** | Ambiguous model selection and novel pipeline composition are genuinely hard reasoning problems. LLM overhead only there. Never for drift detection, routing, or coordination — those are pure math or pure if/else. |
| **User never waits** | Prediction always returns within the SLA, even during retraining. Background agent fires after the response is already sent. |
| **MCP-native action surface** | The agent never imports sktime directly. All sktime capabilities are accessed via MCP tool calls. This demonstrates the sktime-mcp pattern at production scale and makes the agent genuinely composable. |
| **Memory over statelessness** | A stateless LLM call sees one snapshot. The memory-equipped agent knows AutoARIMA failed twice on this dataset and avoids it without being told. |
| **No message loss** | Every job persists in Valkey Streams until a worker explicitly confirms it via XACK. Worker crash = message stays and is reclaimed. |
| **Open source clean** | Every dependency is OSI-approved. BSD-3-Clause licence throughout. No legal ambiguity for GC.OS review. |
| **Locally testable** | Full stack runs on one laptop with Docker Compose. No cloud account required for development or CI. |
| **Honest scope** | v1 is scoped for one person to deliver in 12 weeks. v2 is clearly labelled as future work. Nothing is promised that will not be delivered. |

---

## 3. What Makes It Genuinely Agentic

The distinction between an automation system and an agentic system is precise. Three conditions must hold simultaneously:

```
1. The LLM decides what to do next — not a rule, not a fixed pipeline
2. The LLM uses tools to investigate before deciding — multi-step, not single-shot
3. The LLM's decision changes based on what it discovers — memory + reasoning, not pattern matching
```

All three conditions are met in `sktime-agentic`. The `PipelineArchitectAgent` receives statistical evidence from MCP tool calls, reasons across production memory, and constructs a `ForecastingPipeline` composition that a grid search cannot produce — because it combines transformers and forecasters in novel sequences based on what the data reveals, not from a pre-defined candidate list.

The `ModelSelectorAgent` meets all three conditions for ambiguous datasets: it profiles the data via tools, reasons about which estimators suit the characteristics, and changes its output based on what the profile reveals.

Everything else — drift detection, request routing, lock management — is plain Python. Not because those components are unimportant, but because they are deterministic. Making them agents would add failure modes with no benefit.

---

## 4. Relation to sktime Ecosystem

```
sktime (core ML library)
    ↑ uses
sktime-mcp (MCP tool layer)          sktime-agentic (this project)
    ↑ used by                              ↑ depends on sktime-mcp
AI coding agents                      ↑ wraps sktime via MCP
                                      Production applications
```

| | sktime-mcp | sktime-agentic |
|---|---|---|
| Role | Tool provider | Agent orchestrator |
| Knows | How to execute sktime ops | Why, when, in what order |
| Memory | None — stateless | Full production history per dataset |
| Decisions | None | All of them |
| LLM | Not required | Core component |
| Dependency | sktime | sktime-mcp |

**Upstream contribution:** 6 new tools contributed to `sktime-mcp` as part of this project. See Section 19.

---

## 5. High-Level System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        User / Application                        │
└────────────────────────────┬────────────────────────────────────┘
                             │ HTTP / REST
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Go API Gateway                              │
│  - Request validation + correlation ID generation                │
│  - Direct Valkey cache check (hits never touch Python)           │
│  - XADD forecast:jobs on cache miss                              │
│  - BLPOP result:{uuid} — blocking wait, no polling              │
│  - /health  /ready  /metrics  /admin endpoints                   │
└────────────────────────────┬────────────────────────────────────┘
                             │ XADD forecast:jobs
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                       Valkey Streams                             │
│  forecast:jobs      → prediction job queue                       │
│  retrain:jobs       → background retrain queue                   │
│  result:{uuid}      → response delivery channel                  │
│  pred:{ver}:{id}    → prediction cache (version-baked key)       │
│  retrain_lock:{id}  → deduplication lock                         │
│  model_lock:{id}    → model update lock                          │
│  model_updated:{id} → promotion signal to workers               │
│  memory:{id}        → persistent agent memory (JSON)            │
└────────────────────────────┬────────────────────────────────────┘
                             │ XREADGROUP
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│            Python Worker (stateless container, ×2 replicas)      │
│                                                                   │
│  ┌───────────────────────────────────────────────────────────┐   │
│  │                      Orchestrator                          │   │
│  │  - Routes jobs to correct agent path                      │   │
│  │  - Manages Valkey locks                                    │   │
│  │  - Holds in-memory model cache                            │   │
│  │  - Cleans orphaned locks on startup                       │   │
│  └────────┬────────────────┬──────────────┬──────────────────┘   │
│           │                │              │                       │
│           ▼                ▼              ▼                       │
│  ┌──────────────┐ ┌──────────────┐ ┌─────────────────┐           │
│  │ ModelSelector│ │  Prediction  │ │   DriftMonitor   │           │
│  │    Agent     │ │    Agent     │ │  (stats only,    │           │
│  │  (LLM call   │ │              │ │   no LLM)        │           │
│  │  if ambiguous│ │              │ │                  │           │
│  └──────┬───────┘ └──────┬───────┘ └────────┬─────────┘           │
│         │                │                   │                    │
│         ▼                ▼                   ▼                    │
│  ┌──────────────────┐  Valkey Cache    XADD retrain:jobs          │
│  │ PipelineArchitect│  (version-baked)                            │
│  │     Agent        │                                             │
│  │ (ReAct loop,     │                                             │
│  │  MCP tools,      │                                             │
│  │  memory-aware,   │                                             │
│  │  major drift     │                                             │
│  │  only)           │                                             │
│  └──────────────────┘                                             │
└───────────────────────────┬─────────────────────────────────────┘
                            │ XACK + write result:{uuid}
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│              Background: TrainingAgent + Watchdog                │
│  - Triggered only by DriftMonitor via retrain:jobs stream        │
│  - Incremental update (minor drift) or full retrain (major)      │
│  - Saves to MLflow + MinIO / S3 / GCS                            │
│  - CV promotion gate — only promotes if better than Production   │
│  - Watchdog monitors post-promotion and auto-reverts on regression│
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    MLflow Model Registry                          │
│                + S3 / GCS / MinIO Artifact Store                 │
│                                                                   │
│  Staging    → just trained, awaiting CV validation               │
│  Production → active model, PredictionAgent uses this            │
│  Archived   → previous Production, kept for rollback             │
└─────────────────────────────────────────────────────────────────┘
```

---

## 6. Request Lifecycle — End to End

Every user request follows five stages. Stages 1–4 complete within the 10-second SLA and return the response. Stage 5 fires in the background after the response is already sent — the user never waits for it.

### Stage 1 — Go Gateway (~0.1s)

Request arrives at Go gateway. Go validates the request struct, generates a UUID correlation ID, and checks the Valkey prediction cache directly. A cache hit returns in ~1ms — Python is never contacted. On a cache miss, Go publishes to `forecast:jobs` stream via `XADD` and blocks on `BLPOP result:{uuid}` with a 30-second timeout. Health and readiness probes are answered by the gateway itself — Python is never contacted for infrastructure checks.

### Stage 2 — LLM Intent Parse (~0.3s)

A single small-model LLM call extracts `dataset_id`, forecast horizon `fh`, and optional metadata from the user's natural language request, returning strict JSON. This is structured extraction only — no reasoning. If the request is already structured JSON, this stage is skipped entirely.

### Stage 3 — Cache Lookup + Drift Triage (parallel, ~1.0s)

Both run concurrently via `asyncio.gather`. Cache lookup is a Valkey MGET against `pred:{model_version}:{dataset_id}:{fh_hash}`. Drift triage is pure Python statistics on the stored residual window — CUSUM is a single pass over a deque, ADWIN result is stored state not recomputed. Neither makes a network call. Drift triage produces one of three outputs: `none`, `minor`, or `major`.

### Stage 4 — Forecast + LLM Response Assembly (~4.1s combined)

If no drift or minor drift: production model loads from in-memory cache (populated at startup, invalidated on promotion via `model_updated` signal). `predict()` and `predict_interval()` run synchronously. The LLM response assembly call receives approximately 300 tokens — summary statistics, forecast values, drift level, model metadata. Raw historical rows bypass the LLM entirely, eliminating numerical hallucination risk. Result is written to `result:{uuid}` in Valkey, `XACK` sent, Go's `BLPOP` unblocks, response returned.

**Minor drift path:** Incremental `forecaster.update(y_new)` is attempted. If it completes within `incremental_update_wait_seconds` (default 10s), the fresh prediction is served with `model_status: "updated"`. If timeout is hit, the old prediction is served with `model_status: "drift_minor"` and a warning.

### Stage 5 — Background Pipeline Architect Agent (major drift only, async)

Only activates on major drift — approximately 10–20% of retrain events in a healthy system. Fires after the response is already sent. The `PipelineArchitectAgent` runs the ReAct loop: calls MCP tools to gather statistical evidence, reads production memory, constructs a novel `ForecastingPipeline` spec, fits it in a thread executor, evaluates against the current production model via CV, and promotes or rejects. The `Watchdog` monitors live predictions post-promotion and auto-reverts if regression exceeds configured tolerance.

### Latency Budget

| Stage | Target | Notes |
|---|---|---|
| Go gateway + cache check | ~0.1s | Stateless — scales horizontally |
| LLM intent parse | ~0.3s | Small model, structured output only |
| Cache lookup + drift triage | ~1.0s | Parallel asyncio.gather |
| Forecast (in-memory model) | ~0.5s | No MLflow call on hot path |
| LLM response assembly | ~2.5s | ~300 token input, single call |
| **Total user-facing** | **~4.5s** | **5.5s headroom within 10s SLA** |
| Background agent (async) | Minutes | User never waits — already responded |

---

## 7. LLM Call Inventory

The system makes exactly **three LLM calls** per complete workflow. Two are in the user-facing critical path. One fires in the background only when a major drift event requires a full pipeline reconstruction.

| Call | Agent | When | Input tokens | Output | Risk |
|---|---|---|---|---|---|
| Intent parse | Orchestrator | Every request | ~150 | Structured JSON | Near zero |
| Model selection | ModelSelectorAgent | First request per dataset, or when ambiguous | ~400 | Ranked estimator list JSON | Low — whitelist validated |
| Pipeline architect | PipelineArchitectAgent | Major drift only | ~600 | Pipeline spec JSON | Mitigated — 3-layer defence |
| Response assembly | Orchestrator | Every request | ~300 | Natural language narration | Low |

**Total LLM calls on a warm, stable system:** 2 per request (intent parse + response assembly). Model selection and pipeline architect are one-time or rare events.

---

## 8. Python Layer — The Agentic Brain

### 8.1 Orchestrator

**What it is:** A coordinator class, not an agent. Plain Python logic (~100 lines).

**Responsibilities:**
- Listens to `forecast:jobs` stream via `XREADGROUP`
- Routes jobs: model selection needed? prediction only? pipeline reconstruction needed?
- Acquires and releases Valkey locks before any model state change
- Holds in-memory model store (`dict[str, BaseForecaster]`)
- On startup: scans for orphaned locks from previous crashes and releases them

**Why not an agent:** Coordination logic is deterministic. Every decision is an `if/else`. Adding LLM overhead would slow every request and make failures unpredictable.

```python
class Orchestrator:
    def __init__(self, valkey, mlflow_client, mcp_client, settings):
        self.valkey = valkey
        self.mlflow = mlflow_client
        self.mcp = mcp_client
        self.settings = settings
        self.model_cache: dict[str, BaseForecaster] = {}
        self.model_selector = ModelSelectorAgent(valkey, mlflow_client, mcp_client, settings)
        self.prediction_agent = PredictionAgent(valkey, mlflow_client, settings)
        self.drift_monitor = DriftMonitor(valkey, settings)
        self.pipeline_architect = PipelineArchitectAgent(valkey, mcp_client, settings)

    async def handle_job(self, job: ForecastRequest) -> ForecastResponse:
        await self._maybe_reload_model(job.dataset_id)
        model_version = self._get_production_version(job.dataset_id)

        if model_version is None:
            model_version = await self.model_selector.select(job)

        result = await self.prediction_agent.predict(job, model_version, self.model_cache)

        # Fire-and-forget — user response never held for this
        asyncio.create_task(self.drift_monitor.check(job, result))

        return result

    async def _maybe_reload_model(self, dataset_id: str) -> None:
        signal = await self.valkey.get(f"model_updated:{dataset_id}")
        if signal:
            model = mlflow.sktime.load_model(f"models:/{dataset_id}/Production")
            self.model_cache[dataset_id] = model
            await self.valkey.delete(f"model_updated:{dataset_id}")

    async def startup_cleanup(self) -> None:
        async for key in self.valkey.scan_iter("model_lock:*"):
            if await self.valkey.ttl(key) < 0:
                await self.valkey.delete(key)
```

---

### 8.2 ModelSelectorAgent

**What it is:** The first agent in the flow. One focused LLM call for genuinely ambiguous model selection. Rule-based fast path for clear-cut cases.

**Selection pipeline:**

```
1. Profile the dataset via MCP tool call
   → length, frequency, seasonality, missingness, variance, stationarity

2. Rule-based fast path (no LLM, instant)
   → short series (<100 obs)    : ARIMA, ETS
   → long seasonal               : Prophet, BATS, TBATS
   → multivariate                : VAR, VARMAX
   → irregular frequency         : NaiveForecaster + interpolation

3. If ambiguous → one LLM call
   → describe data profile in natural language
   → ask for ranked list from ALLOWED_ESTIMATORS whitelist
   → parse JSON response
   → validate all returned estimators against whitelist

4. If LLM call fails → MultiplexForecaster + ForecastingGridSearchCV

5. Ultimate fallback → NaiveForecaster(strategy="last")
   (system never crashes, always returns something)
```

**LLM call — direct Anthropic SDK (no LangChain):**
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
    parsed = json.loads(response.content[0].text)
    return [e for e in parsed["ranked"] if e in ALLOWED_ESTIMATORS]
```

**Week 1 priority — serialization whitelist validation:**
Run this before building any agent logic. Not every sktime estimator survives `mlflow.sktime.save_model()`. Any estimator that fails is permanently excluded from `ALLOWED_ESTIMATORS`.

```python
CANDIDATE_ESTIMATORS = [
    "AutoARIMA", "AutoETS", "Prophet", "BATS", "TBATS",
    "ThetaForecaster", "ExponentialSmoothing", "NaiveForecaster",
    "PolynomialTrendForecaster", "STLForecaster",
]
ALLOWED_ESTIMATORS = []
for name in CANDIDATE_ESTIMATORS:
    try:
        est = registry[name](); est.fit(y_test)
        with tempfile.TemporaryDirectory() as tmp:
            mlflow.sktime.save_model(est, tmp)
        ALLOWED_ESTIMATORS.append(name)
    except Exception as e:
        print(f"EXCLUDED {name}: {e}")
```

---

### 8.3 PipelineArchitectAgent — The Agentic Core

**This is the component that makes `sktime-agentic` genuinely novel.** It activates only on major drift events, runs entirely in the background, and constructs a `ForecastingPipeline` specification — not a model name selected from a list, but a composed sequence of transformers and a forecaster tailored to what the data reveals.

**Why this is not a grid search:**
Grid search selects from a pre-defined candidate list. The architect agent composes from the full transformer + forecaster registry in novel sequences — a sequence no grid search would enumerate because it was derived from statistical evidence and production failure history, not a pre-defined search space.

**ReAct loop:**
```
Thought → Action (MCP tool call) → Observation → Thought → Action → … → Pipeline spec
```

**Input — evidence dict assembled by pure Python statistical tests:**
```json
{
  "n_obs": 312,
  "frequency": "W",
  "stationarity": "non-stationary",
  "adf_pvalue": 0.31,
  "seasonality_period": 52,
  "seasonality_strength": 0.81,
  "structural_break": true,
  "structural_break_location": 289,
  "past_failures": [
    "AutoARIMA v3 degraded in 28 days",
    "AutoARIMA v5 degraded in 31 days"
  ],
  "production_memory_summary": "Two AutoARIMA failures on this dataset in 60 days. Drift consistently follows promotional events. Break at t=289 predates both failed models."
}
```

**Example LLM output — a pipeline composition no grid search would produce:**
```json
{
  "pipeline_steps": [
    {"step": "Deseasonalizer", "params": {"sp": 52}},
    {"step": "Prophet", "params": {"changepoint_prior_scale": 0.1}}
  ],
  "rationale": "Two AutoARIMA failures on a series with a confirmed structural break at t=289. The break implies a regime change that ARIMA order selection cannot adapt to. Prophet handles changepoints natively. Deseasonalizer applied first because Prophet additive seasonality conflicts with sktime seasonal handling at sp=52.",
  "confidence": "high",
  "estimated_fit_minutes": 1.8,
  "fallback_pipeline": [
    {"step": "ThetaForecaster", "params": {"sp": 52}}
  ]
}
```

**MCP tools called during the ReAct loop:**
- `profile_dataset(dataset_id)` — statistical profile
- `run_stationarity_test(dataset_id)` — ADF + KPSS
- `detect_seasonality(dataset_id)` — period and strength
- `check_structural_break(dataset_id)` — CUSUM-based change point
- `get_dataset_history(dataset_id)` — full production history
- `get_model_complexity_budget(dataset_id)` — size-aware model filtering
- `estimate_training_cost(dataset_id, model)` — pre-fit cost estimation

**Tool result `next_action_hint`:**
Every MCP tool result includes a `next_action_hint` field that suggests the most likely next tool for standard investigation sequences. The agent follows hints for routine sequences and overrides for novel situations — reducing unnecessary reasoning steps.

```json
{
  "stationarity": "non-stationary",
  "adf_pvalue": 0.31,
  "kpss_pvalue": 0.02,
  "conclusion": "differencing or detrending recommended",
  "next_action_hint": "detect_seasonality"
}
```

---

### 8.4 PredictionAgent

**Responsibilities:**
- Checks in-memory model cache (fastest path — microseconds)
- Checks Valkey prediction cache (second path — ~1ms)
- Loads from MLflow if neither available (one-time per version)
- Runs `predict()` and `predict_interval()`
- Writes result to Valkey cache with version-baked key

**Cache key format — version baked in, no explicit flush needed:**
```
pred:{model_version}:{dataset_id}:{fh_hash}
```
When a new model is promoted, old keys become unreachable automatically. No race condition on promotion.

**Cache TTL by frequency:**
```python
cache_ttl = {
    "H": 900,    # hourly  → 15 minutes
    "D": 21600,  # daily   → 6 hours
    "W": 86400,  # weekly  → 24 hours
    "M": 86400,  # monthly → 24 hours
}
```

**Hard minimum history check:**
```python
if len(y) < self.settings.min_history_length:
    raise InsufficientHistoryError(
        f"Dataset {job.dataset_id} has {len(y)} observations. "
        f"Minimum required: {self.settings.min_history_length}."
    )  # Returns HTTP 422 — never attempts a meaningless prediction
```

---

### 8.5 DriftMonitor

**What it is:** A statistical check function. Plain Python, not an agent. Runs as a fire-and-forget async task after every prediction — user response is never held for it.

**Detection methods (in order of compute cost):**

| Method | Trigger | Cost |
|---|---|---|
| Residual error tracking | Always (baseline) | Negligible |
| CUSUM | Residuals trending consistently in one direction | Low |
| ADWIN | Distribution shift suspected | Medium |

**Tiered drift response:**

```
Drift Detected
      │
      ├── MINOR drift (residual creep, small CUSUM score < 0.5)
      │   → Incremental update: forecaster.update(y_new)
      │   → Expected time: 2–15 seconds
      │   → Hold request up to incremental_update_wait_seconds (default: 10s)
      │   → If update finishes: serve fresh prediction, status="updated"
      │   → If timeout: serve old prediction, status="drift_minor" + warning
      │
      └── MAJOR drift (ADWIN trigger, score ≥ 0.5)
          → Full retrain + pipeline reconstruction required
          → NEVER hold the request
          → Serve immediately with status="drift_major" + warning
          → XADD retrain:jobs → PipelineArchitectAgent fires in background
```

**Deduplication — prevents multiple retrain jobs:**
```python
async def _maybe_trigger_retrain(self, dataset_id: str, reason: str) -> None:
    lock_key = f"retrain_lock:{dataset_id}"
    if await self.valkey.exists(lock_key):
        return  # retrain already queued — skip
    await self.valkey.setex(lock_key, self.settings.retrain_lock_ttl_seconds, "1")
    await self.valkey.xadd("retrain:jobs", {
        "dataset_id": dataset_id,
        "reason": reason,
        "triggered_at": datetime.utcnow().isoformat()
    })
```

---

### 8.6 TrainingAgent

**Triggered by:** `DriftMonitor` via `retrain:jobs` stream. Never triggered directly by user requests.

**Training strategy selection:**
```
Is this the first ever train?
├── Yes → cold start: full fit on all available history
└── No →
    Minor drift?
    ├── Yes → forecaster.update(y_new)  [incremental — seconds]
    └── No  →
        PipelineArchitectAgent constructs new spec
        ├── Valid spec + fits well → use it
        └── Spec invalid or fits poorly → ForecastingGridSearchCV fallback
```

**CV promotion gate:**
New model is only promoted to Production if its CV score strictly beats the current Production model. A confident but wrong LLM recommendation that produces a worse model never reaches production.

**MLflow lifecycle:**
```
mlflow.sktime.log_model()    → saves to Staging
validate CV score             → if better than Production:
mlflow.transition_stage()     → Staging → Production
                              → old Production → Archived
valkey.set("model_updated:{id}", ...)  → signal to workers to reload
```

**Lock behaviour:**
```python
lock_ttl = int(self.settings.max_training_time_seconds * 1.5)
acquired = await self.valkey.set(
    f"model_lock:{dataset_id}", "1", nx=True, ex=lock_ttl
)
if not acquired:
    return  # another worker is already training this dataset
```

---

### 8.7 Watchdog

A lightweight background process that monitors live predictions after every model promotion. If post-promotion prediction error exceeds `watchdog_regression_tolerance` (default: 15% worse than pre-promotion baseline), it automatically reverts Production to the previous Archived model and signals workers to reload.

```python
class Watchdog:
    async def monitor_post_promotion(self, dataset_id: str, baseline_score: float):
        await asyncio.sleep(self.settings.watchdog_observation_window_seconds)
        live_score = self._compute_live_error(dataset_id)
        if live_score > baseline_score * (1 + self.settings.watchdog_regression_tolerance):
            await self._revert_to_archived(dataset_id)
            logger.warning(f"Watchdog reverted {dataset_id} — live score {live_score:.4f} exceeded threshold")
```

---

## 9. Agent Memory

Persistent memory is what separates reactive decisions from genuinely intelligent ones. A stateless LLM call sees one snapshot. The memory-equipped agent knows AutoARIMA failed on this dataset twice in 60 days and avoids it without being told.

**Schema — stored in Valkey as JSON, keyed by `dataset_id`:**
```json
{
  "model_history": [
    {
      "version": "v3",
      "estimator": "AutoARIMA",
      "promoted_at": "2026-06-15T10:00:00Z",
      "cv_score": -198.7,
      "survived_days": 28,
      "failure_reason": "drift",
      "pipeline_steps": null
    },
    {
      "version": "v6",
      "estimator": "ForecastingPipeline",
      "promoted_at": "2026-07-20T14:23:00Z",
      "cv_score": -154.2,
      "survived_days": null,
      "failure_reason": null,
      "pipeline_steps": [
        {"step": "Deseasonalizer", "params": {"sp": 52}},
        {"step": "Prophet", "params": {"changepoint_prior_scale": 0.1}}
      ]
    }
  ],
  "drift_events": [
    {
      "timestamp": "2026-07-18T09:15:00Z",
      "score": 0.87,
      "method": "ADWIN",
      "response": "full_retrain",
      "outcome": "new_model_promoted"
    }
  ],
  "pipeline_specs": [
    {
      "version": "v6",
      "steps": [...],
      "rationale": "Two AutoARIMA failures + structural break at t=289...",
      "constructed_by": "llm",
      "confidence": "high"
    }
  ],
  "data_characteristics": {
    "frequency": "W",
    "n_obs": 312,
    "last_structural_break": 289,
    "last_updated": "2026-07-20T14:23:00Z"
  }
}
```

**What memory enables:**
- Agent knows AutoARIMA failed and avoids it without being told
- Agent knows the structural break entered at t=289 and reasons that pre-break model assumptions are invalid
- Agent can identify that drift events cluster around certain calendar periods and adjust response
- Every promoted model has its full LLM rationale attached in MLflow — debugging a failure has a paper trail
- Watchdog knows the pre-promotion baseline score without recomputing it

---

## 10. Hallucination Mitigation

Three layered defences eliminate the risk of incorrect LLM output reaching production:

**Layer 1 — Component registry constraint.**
The LLM is given an explicit registry of valid sktime class names and their permitted parameters. It cannot invent a class name that does not exist — it can only select and parameterise from the registry. This is enforced at prompt construction time, not post-hoc.

**Layer 2 — Structural validation.**
Every pipeline spec is validated against the registry before any fitting occurs. Unknown class names or invalid parameter keys are caught immediately. Validation failure silently falls back to `ForecastingGridSearchCV` — the user never sees an error.

**Layer 3 — CV promotion gate.**
Even a valid, well-formed pipeline must beat the current Production model's CV score to be promoted. A confident but wrong LLM recommendation that produces a worse model never reaches production.

```python
def validate_pipeline_spec(spec: dict, registry: ComponentRegistry) -> bool:
    for step in spec["pipeline_steps"]:
        if step["step"] not in registry.valid_transformers + registry.valid_forecasters:
            return False
        invalid_params = set(step["params"]) - registry.valid_params[step["step"]]
        if invalid_params:
            return False
    return True
```

---

## 11. MCP Tool Surface

All agent actions go through `sktime-mcp`. The agent never imports sktime directly.

### Dataset Intelligence Tools
| Tool | Description | Returns |
|---|---|---|
| `profile_dataset(dataset_id)` | Statistical profile | `n_obs, freq, missing_pct, mean, std, CV` |
| `get_dataset_history(dataset_id)` | Full production history | Past models, scores, failures, drift events |
| `get_drift_history(dataset_id)` | Drift event log | Timestamps, scores, methods, responses |
| `get_last_n_rows(dataset_id)` | Frequency-sized recent actuals | Date, actual, predicted columns |

### Statistical Test Tools (contributed upstream to sktime-mcp)
| Tool | Description | Returns |
|---|---|---|
| `run_stationarity_test(dataset_id)` | ADF + KPSS tests | `p-values, conclusion, next_action_hint` |
| `detect_seasonality(dataset_id)` | Seasonal period and strength | `period, strength, method` |
| `check_structural_break(dataset_id)` | CUSUM-based break detection | `break_detected, location, confidence` |
| `detect_drift(dataset_id)` | Current drift scores | `CUSUM, ADWIN, residual scores` |

### Training and Evaluation Tools (contributed upstream to sktime-mcp)
| Tool | Description | Returns |
|---|---|---|
| `get_model_complexity_budget(dataset_id)` | Permitted models for dataset size | `permitted, forbidden, reason` |
| `estimate_training_cost(dataset_id, model)` | Cost before committing | `minutes, cost_usd, recommendation` |
| `evaluate_model(dataset_id, model, params)` | CV on 20% subsample | `cv_score, fold_scores` |
| `fit_model(dataset_id, model, params)` | Full fit and stage | `version, staging_uri` |
| `promote_model(dataset_id, version)` | Staging → Production | `confirmation` |
| `reject_model(dataset_id, version)` | Archive a staged model | `confirmation` |

---

## 12. Go Layer — Production Gateway

**Why Go, not Python for the gateway:** Python is the right language for time series intelligence. Go is the right language for a production API gateway — high concurrency with goroutines, low memory overhead, single binary deployment, and microsecond-level routing. The Python layer is fully standalone without Go — useful for development and CI without the full stack.

### Responsibilities

- HTTP/REST endpoint at `/forecast`
- Request struct validation (Go `validator` library)
- Direct Valkey cache check — cache hits never touch Python (~1ms vs ~50ms)
- UUID correlation ID generation
- `XADD` to `forecast:jobs` stream
- `BLPOP result:{uuid}` — blocking wait with 30s timeout, no polling, no busy loop
- HTTP 503 + `Retry-After: 5` on timeout
- Prometheus metrics endpoint
- Health and readiness probes for Kubernetes

### Endpoints

```
POST /forecast                  → main prediction endpoint
GET  /health                    → liveness probe (200 if process alive)
GET  /ready                     → readiness probe (checks Valkey connection)
GET  /metrics                   → Prometheus metrics
POST /admin/retrain             → manual retrain trigger (authenticated)
GET  /admin/model/{dataset_id}  → current Production model info
```

### Request Handler Sketch

```go
func (h *Handler) Forecast(c *fiber.Ctx) error {
    var req ForecastRequest
    if err := c.BodyParser(&req); err != nil {
        return c.Status(400).JSON(ErrorResponse{Error: "invalid request"})
    }
    if err := validate.Struct(req); err != nil {
        return c.Status(422).JSON(ErrorResponse{Error: err.Error()})
    }

    // Direct cache check — bypass Python entirely on hit
    cacheKey := fmt.Sprintf("pred:%s:%s:%s", currentModelVersion, req.DatasetID, hashFH(req.FH))
    if cached, err := h.valkey.Get(ctx, cacheKey).Result(); err == nil {
        return c.JSON(cached)
    }

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

## 13. Infrastructure Layer

### 13.1 Valkey Streams

**Why Valkey, not Redis:** Redis changed its licence in March 2024 from BSD-3 to RSALv2 + SSPLv1 — neither is OSI-approved. For a GC.OS submission, a non-OSI-approved dependency is a genuine compliance risk. Valkey is the Linux Foundation fork of Redis, BSD-3-Clause, API-compatible, maintained by contributors from AWS, Google, and Oracle. The `redis-py` Python client and `go-redis` client both work with Valkey without any code changes.

**Why Streams, not Pub/Sub:** Pub/Sub is fire-and-forget. If a worker is down for even one second, that message is permanently lost. Streams persist messages until explicitly acknowledged via `XACK`. Worker crash = message held. Another worker claims it via `XCLAIM` after a configurable timeout.

**Stream and key layout:**

| Key | Type | Purpose | TTL |
|---|---|---|---|
| `forecast:jobs` | Stream | Prediction job queue | No TTL — XACK deletes |
| `retrain:jobs` | Stream | Background retrain queue | No TTL — XACK deletes |
| `result:{uuid}` | String | Response delivery to Go | 60 seconds |
| `pred:{ver}:{id}:{fh}` | String | Prediction cache | 15 min – 24 h (by frequency) |
| `retrain_lock:{id}` | String | Retrain deduplication | 30 minutes |
| `model_lock:{id}` | String | Model update lock | `max_train_time × 1.5` |
| `model_updated:{id}` | String | Promotion signal to workers | 5 minutes |
| `memory:{id}` | String (JSON) | Agent persistent memory | No TTL — updated in place |

**Consumer group setup:**
```bash
XGROUP CREATE forecast:jobs workers  $ MKSTREAM
XGROUP CREATE retrain:jobs  trainers $ MKSTREAM
```

---

### 13.2 MLflow + Cloud Storage

**Local development:** MinIO (S3-compatible, runs in Docker). No AWS account required.
**Production:** AWS S3 or GCS, configured via `MLFLOW_ARTIFACT_URI`.

**Model lifecycle stages:**
```
Staging    → TrainingAgent just finished, awaiting CV validation
Production → validated, PredictionAgent uses this
Archived   → previous Production, kept for Watchdog rollback
```

**Every MLflow run tagged with:**
```python
mlflow.set_tags({
    "drift_reason":       job.reason,
    "dataset_id":         job.dataset_id,
    "previous_version":   previous_version,
    "estimator_class":    type(forecaster).__name__,
    "pipeline_spec":      json.dumps(spec["pipeline_steps"]),
    "llm_rationale":      spec.get("rationale", "rule-based"),
    "sktime_version":     sktime.__version__,
})
mlflow.log_metrics({
    "cv_score":                cv_score,
    "training_duration_seconds": duration,
    "beats_production_by":     production_score - cv_score,
})
```

---

### 13.3 Kubernetes Deployment

**v1 — Basic Deployments (achievable in one summer):**

```yaml
# python-worker Deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: python-worker
spec:
  replicas: 2
  selector:
    matchLabels:
      app: python-worker
  template:
    metadata:
      labels:
        app: python-worker
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
          resources:
            requests:
              memory: "512Mi"
              cpu: "500m"
            limits:
              memory: "2Gi"
              cpu: "2000m"
---
# go-gateway Deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: go-gateway
spec:
  replicas: 2
  selector:
    matchLabels:
      app: go-gateway
  template:
    metadata:
      labels:
        app: go-gateway
    spec:
      containers:
        - name: gateway
          image: sktime-agentic-gateway:latest
          ports:
            - containerPort: 8080
          envFrom:
            - secretRef:
                name: sktime-agentic-secrets
          livenessProbe:
            httpGet:
              path: /health
              port: 8080
          readinessProbe:
            httpGet:
              path: /ready
              port: 8080
          resources:
            requests:
              memory: "64Mi"
              cpu: "100m"
            limits:
              memory: "256Mi"
              cpu: "500m"
---
# go-gateway Service
apiVersion: v1
kind: Service
metadata:
  name: go-gateway-svc
spec:
  selector:
    app: go-gateway
  ports:
    - port: 80
      targetPort: 8080
  type: LoadBalancer
---
# Valkey StatefulSet
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: valkey
spec:
  serviceName: valkey
  replicas: 1
  selector:
    matchLabels:
      app: valkey
  template:
    metadata:
      labels:
        app: valkey
    spec:
      containers:
        - name: valkey
          image: valkey/valkey:8
          ports:
            - containerPort: 6379
          volumeMounts:
            - name: valkey-data
              mountPath: /data
  volumeClaimTemplates:
    - metadata:
        name: valkey-data
      spec:
        accessModes: ["ReadWriteOnce"]
        resources:
          requests:
            storage: 5Gi
```

**Probe rationale:** Without probes, Kubernetes cannot distinguish between a worker stuck in a long training job and one that has crashed. The readiness probe checks Valkey connectivity — if Valkey is unreachable, the pod is marked not-ready and removed from the load balancer automatically.

**v2 stretch goal — KEDA autoscaling:** Scale Python workers based on `forecast:jobs` stream length. Zero workers when idle, scales up under load. Labelled clearly as future work.

---

### 13.4 Docker Compose — Local Development

Full stack runs on one laptop. No cloud account, no Kubernetes required for development or CI.

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
    environment:
      MLFLOW_S3_ENDPOINT_URL: http://minio:9000
      AWS_ACCESS_KEY_ID: minioadmin
      AWS_SECRET_ACCESS_KEY: minioadmin

  minio:
    image: minio/minio:latest
    ports:
      - "9000:9000"
      - "9001:9001"
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
  -d '{"dataset_id": "test-dataset-1", "fh": [1, 2, 3], "frequency": "D"}'
```

**Python-only mode (no Go):**
```bash
cd python && uvicorn app.main:app --host 0.0.0.0 --port 8000
```

---

## 14. API Reference

### POST /forecast

**Request:**
```json
{
  "dataset_id": "sensor-42-temperature",
  "fh": [1, 2, 3, 4, 5],
  "frequency": "H"
}
```

**Response — stable:**
```json
{
  "dataset_id": "sensor-42-temperature",
  "predictions": [22.4, 22.7, 23.1, 23.0, 22.8],
  "prediction_intervals": {
    "lower": [21.1, 21.3, 21.6, 21.4, 21.2],
    "upper": [23.7, 24.1, 24.6, 24.6, 24.4]
  },
  "model_version": "6",
  "model_class": "ForecastingPipeline(Deseasonalizer+Prophet)",
  "model_status": "stable",
  "drift_score": null,
  "warning": null,
  "cache_hit": false,
  "correlation_id": "f47ac10b-58cc-4372-a567-0e02b2c3d479"
}
```

**Response — drift major:**
```json
{
  "predictions": [22.4, 22.7, 23.1, 23.0, 22.8],
  "model_status": "drift_major",
  "drift_score": 0.81,
  "drift_method": "ADWIN",
  "warning": "Significant drift detected (ADWIN, score: 0.81). Full pipeline reconstruction triggered in background. Current predictions are from previous model version.",
  "cache_hit": false
}
```

**Response — pipeline updated:**
```json
{
  "predictions": [22.1, 22.4, 22.9, 22.7, 22.5],
  "model_version": "7",
  "model_class": "ForecastingPipeline(Deseasonalizer+ThetaForecaster)",
  "model_status": "updated",
  "llm_rationale": "Prophet fallback triggered — Prophet fit exceeded complexity budget for dataset length 89.",
  "cache_hit": false
}
```

---

## 15. Schemas

```python
from pydantic import BaseModel, field_validator
from datetime import datetime

class ForecastRequest(BaseModel):
    dataset_id: str
    fh: list[int]
    correlation_id: str
    frequency: str | None = None

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
    model_status: str   # "stable"|"updated"|"drift_minor"|"drift_major"|"retraining"
    drift_score: float | None = None
    drift_method: str | None = None  # "CUSUM"|"ADWIN"|"residual"
    warning: str | None = None
    llm_rationale: str | None = None
    cache_hit: bool
    correlation_id: str

class RetrainJob(BaseModel):
    dataset_id: str
    reason: str  # "CUSUM"|"ADWIN"|"residual"|"manual"
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
            f"missing rate={self.missing_rate:.1%}"
        )
```

---

## 16. Configuration

```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # LLM
    anthropic_api_key: str
    llm_model: str = "claude-sonnet-4-20250514"

    # Valkey
    valkey_url: str = "redis://localhost:6379"
    retrain_lock_ttl_seconds: int = 1800    # 30 minutes
    max_training_time_seconds: int = 3600   # 1 hour
    result_ttl_seconds: int = 60

    # Drift
    no_drift_threshold: float = 0.2
    minor_drift_threshold: float = 0.5
    drift_check_every_n_predictions: int = 50
    drift_check_every_t_minutes: int = 10
    incremental_update_wait_seconds: int = 10

    # Watchdog
    watchdog_observation_window_seconds: int = 3600
    watchdog_regression_tolerance: float = 0.15

    # Prediction
    min_history_length: int = 30

    # MLflow
    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_artifact_uri: str = "s3://sktime-agentic-models"

    class Config:
        env_file = ".env"
```

---

## 17. Error Handling and Failure Modes

| Failure | Response | How Handled |
|---|---|---|
| LLM API call fails (model selection) | Cascade: `MultiplexForecaster + GridSearchCV` → `NaiveForecaster` | Automatic — system never blocks on LLM availability |
| LLM API call fails (pipeline architect) | Cascade: `ForecastingGridSearchCV` with ALLOWED_ESTIMATORS | Automatic — background retrain continues without LLM |
| LLM output fails registry validation | Silent fallback to `GridSearchCV` | Automatic — user never sees error |
| New model CV worse than Production | Promotion gate rejects. Staged model archived. Old Production stays. | Automatic — regression impossible through retraining |
| Watchdog detects post-promotion regression | Auto-revert to Archived model. Signal workers to reload. | Automatic |
| Go timeout (30s) | HTTP 503 + `Retry-After: 5`. Worker continues and writes result — next request hits cache. | Client retries cleanly |
| Valkey unreachable | Go readiness probe returns 503. Pod removed from load balancer by K8s. | Automatic via K8s probes |
| Orphaned lock on startup | Orchestrator startup routine scans locks with no TTL and deletes them. | Automatic on every worker restart |
| Duplicate retrain triggers | `retrain_lock:{dataset_id}` prevents publishing multiple jobs. | Automatic deduplication |
| Worker crashes mid-job | Valkey holds message — another worker claims via `XCLAIM` after timeout. | Automatic stream recovery |
| Insufficient history | HTTP 422: `"Dataset X has N observations. Minimum: M."` | Clear user-facing error, no prediction attempted |

---

## 18. Testing Strategy

```
tests/
├── unit/
│   ├── test_model_selector.py        # Rule-based selection, LLM fallback, whitelist validation
│   ├── test_pipeline_architect.py    # ReAct loop, spec validation, hallucination defence
│   ├── test_drift_monitor.py         # CUSUM, ADWIN, tiered response
│   ├── test_prediction_agent.py      # Cache hit/miss, version-baked key, stale entry
│   ├── test_orchestrator.py          # Routing, lock behaviour, orphan cleanup
│   ├── test_watchdog.py              # Regression detection, auto-revert
│   └── test_schemas.py               # Pydantic validation, fh validator
├── integration/
│   ├── test_full_flow.py             # Request → prediction → XACK → response
│   ├── test_drift_retrain_cycle.py   # Drift → retrain → promotion → new prediction
│   ├── test_pipeline_architect_e2e.py # Major drift → ReAct loop → spec → fit → promote
│   ├── test_cache_invalidation.py    # Old version unreachable after promotion
│   └── test_memory_persistence.py   # Memory persists across worker restarts
└── fixtures/
    ├── sample_datasets/              # Small time series for testing
    ├── mock_mlflow/                  # MLflow fixtures
    └── mock_mcp/                     # Mock MCP tool responses for unit tests
```

**Key test cases:**
- Duplicate retrain lock: inject drift twice simultaneously — verify only one retrain job published
- Orphaned lock recovery: kill worker mid-retrain, restart — verify lock cleaned and retrain can run
- Cache key staleness: promote new model — verify old cache key unreachable, new key populated
- LLM fallback cascade: mock LLM API to fail — verify cascade to `GridSearchCV` then `NaiveForecaster`
- Hallucination defence: inject invalid class name in LLM output — verify registry rejects, fallback triggers
- Watchdog regression: mock post-promotion error exceeding tolerance — verify auto-revert and reload
- Minimum history: send dataset with 5 observations — verify HTTP 422 with correct message
- Tiered drift: inject minor drift → verify wait + `updated` status; inject major → verify immediate response + warning

---

## 19. Upstream Contributions to sktime-mcp

The following tools are contributed upstream to the `sktime-mcp` repository as part of this project. This gives the submission dual credit: own idea category AND direct sktime-mcp contribution.

| Tool | Type | Description |
|---|---|---|
| `run_stationarity_test` | Statistical | ADF + KPSS with structured output + `next_action_hint` |
| `detect_seasonality` | Statistical | Period and strength detection |
| `check_structural_break` | Statistical | CUSUM-based change point detection |
| `get_model_complexity_budget` | Training | Dataset-size-aware model filtering |
| `estimate_training_cost` | Training | Pre-fit cost estimation |
| `get_dataset_history` | Memory | Production history resource |

---

## 20. Milestone Table

| Phase | Weeks | Deliverable | Priority |
|---|---|---|---|
| **Pre-start** | Before June 9 | Serialization whitelist validated — all `ALLOWED_ESTIMATORS` pass `mlflow.sktime.save_model()` | **Critical — do first** |
| **Pre-start** | Before June 9 | `ModelSelectorAgent` + `PredictionAgent` core working locally | High |
| **Pre-start** | Before June 9 | `Orchestrator` + Valkey Streams wiring — end-to-end job flow | High |
| **Pre-start** | Before June 9 | `TrainingAgent` + MLflow save/load/promote working | High |
| **Week 1–2** | June 9–20 | `DriftMonitor` + full background retrain cycle tested | High |
| **Week 3–4** | June 23 – July 4 | Go gateway + Correlation ID pattern + full round-trip | Medium |
| **Week 5–6** | July 7–18 | `PipelineArchitectAgent` ReAct loop + MCP tool calls working | **High — core differentiator** |
| **Week 7–8** | July 21 – Aug 1 | Agent memory schema + cross-run reasoning demonstrated | High |
| **Week 9** | Aug 4–8 | `Watchdog` + auto-revert | Medium |
| **Week 10** | Aug 11–15 | Docker Compose full stack + Kubernetes Deployments | Medium |
| **Week 11** | Aug 18–22 | Tutorial notebook + demo video showing pipeline composition | **High — required for submission** |
| **Week 12** | Aug 25–29 | Final docs, upstream MCP tool PRs, companion repo PR to sktime ecosystem | High |

---

## 21. v1 vs v2 Roadmap

| Feature | v1 (ESoC 2026 — this submission) | v2 (Future) |
|---|---|---|
| Async queue | Valkey Streams | Apache Kafka |
| LLM framework | Direct Anthropic SDK | Same — LangChain explicitly excluded |
| Drift detection | CUSUM + ADWIN (statistical) | Full DriftAgent with LLM-generated root cause explanation |
| Pipeline architect | ReAct loop, MCP-native, 1 LLM call | Multi-round self-critique + confidence scoring |
| Model selection | Rule-based + 1 LLM call | Fully agentic multi-tool investigation |
| K8s scaling | Basic Deployments (manual replicas) | KEDA event-driven autoscaling on stream length |
| Valkey topology | Single instance | Valkey Cluster |
| Monitoring | Basic logs + FastAPI metrics | Prometheus + Grafana with drift history dashboard |
| Go → Python protocol | Valkey Streams | Direct gRPC for lower latency |
| Multi-tenancy | Single namespace | Full tenant isolation per dataset namespace |
| Memory storage | Valkey JSON | Dedicated vector store for semantic memory search |

---

## 22. Repository Layout

```
sktime-agentic/
├── python/
│   ├── app/
│   │   ├── main.py                  # FastAPI app — standalone without Go
│   │   ├── orchestrator.py          # Job routing + lock management
│   │   ├── agents/
│   │   │   ├── model_selector.py    # ModelSelectorAgent
│   │   │   ├── pipeline_architect.py# PipelineArchitectAgent — ReAct loop
│   │   │   ├── prediction.py        # PredictionAgent
│   │   │   ├── training.py          # TrainingAgent
│   │   │   └── watchdog.py          # Watchdog — post-promotion monitor
│   │   ├── monitoring/
│   │   │   └── drift_monitor.py     # DriftMonitor — CUSUM + ADWIN
│   │   ├── memory/
│   │   │   └── memory.py            # Agent memory read/write
│   │   ├── mcp/
│   │   │   └── client.py            # MCP client — all sktime access goes here
│   │   ├── registry/
│   │   │   └── registry.py          # Component registry — whitelist + validation
│   │   ├── prompts/
│   │   │   └── prompts.py           # System prompts + context assembly
│   │   ├── schemas.py               # Pydantic models
│   │   └── config.py                # Settings
│   ├── Dockerfile
│   └── requirements.txt
├── go/
│   ├── gateway/
│   │   ├── main.go                  # Entry point
│   │   ├── handler.go               # /forecast, /health, /ready, /admin
│   │   └── valkey.go                # Valkey client helpers
│   ├── go.mod
│   └── Dockerfile
├── k8s/
│   ├── python-worker.yaml           # Deployment + resource limits
│   ├── go-gateway.yaml              # Deployment + Service + LoadBalancer
│   └── valkey.yaml                  # StatefulSet + PVC
├── tests/
│   ├── unit/
│   ├── integration/
│   └── fixtures/
├── notebooks/
│   └── tutorial.ipynb               # Required for ESoC submission
├── docker-compose.yml
├── .env.example
├── LICENSE                          # BSD-3-Clause
└── README.md
```

---

*sktime-agentic — ESoC 2026 — GC.OS Track — sktime Agentic — Own Idea Category*
*Companion repository to sktime. Not affiliated with the sktime core team.*
*All dependencies OSI-approved. BSD-3-Clause licence.*
*Upstream contributions: 6 new tools to sktime-mcp.*
*Deadline: April 30, 2026 — 18:00 UTC*
