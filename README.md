# sktime-agentic

**Autonomous time-series forecasting infrastructure driven by an LLM agent.**

A companion repo to [`sktime`](https://github.com/sktime/sktime) and [`sktime-mcp`](https://github.com/sktime/sktime-mcp), built for ESoC 2026.

---

## What This Is

Most production forecasting systems are deterministic pipelines:
drift detected → retrain → promote. A rule fires, a script runs, a model swaps.
The system is automated but not intelligent — it cannot reason about *why* drift
happened, *whether* retraining is the right response, or *which* model family
actually suits the data after a structural break.

`sktime-agentic` flips this. Every decision — model selection, retraining,
promotion, drift response — is made by an LLM agent that calls sktime capabilities
as MCP tools. The agent investigates, reasons across production history, constructs
a pipeline, evaluates it, and decides whether to promote. Nothing happens without
the agent choosing to make it happen.

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
│  Kafka job queue    S3 / GCS model storage      │
└─────────────────────────────────────────────────┘
```

---

## Why MCP

[`sktime-mcp`](https://github.com/sktime/sktime-mcp) exposes sktime operations
as callable tools. `sktime-agentic` uses those tools as the action surface for
an LLM agent loop — the agent decides which tools to call, in what order, based
on accumulated production context.

`sktime-mcp` knows how to *execute* sktime operations.
`sktime-agentic` knows *why*, *when*, and *in what sequence* to execute them.

This is not a wrapper. sktime-mcp has no memory across runs, no awareness of
production history, no ability to reason about past model failures or drift
patterns. The agent holds all of that and uses sktime-mcp as its hands.

---

## Agent Behaviour — A Concrete Example

**Trigger:** drift detected on `sales_weekly`, score = 0.89 (major)

**Without sktime-agentic** (deterministic):
```
drift score > threshold → xadd retrain:jobs → grid search → promote
```

**With sktime-agentic** (LLM agent loop):
```
Agent receives:  drift signal + dataset_id + production context

Agent thinks:    "Third retrain this month. Last two runs selected AutoARIMA.
                 Both degraded within 3 weeks. Let me investigate before
                 repeating the same mistake."

Agent calls:     profile_dataset("sales_weekly")
                 → n_obs=312, freq=W, missing_pct=0.0

Agent calls:     run_stationarity_test("sales_weekly")
                 → non-stationary, ADF p=0.31

Agent calls:     check_structural_break("sales_weekly")
                 → structural break detected at t=289 (6 weeks ago)

Agent calls:     detect_seasonality("sales_weekly")
                 → period=52 confirmed, strength=0.81

Agent reasons:   "Structural break 6 weeks ago — likely a regime change,
                 not parameter drift. AutoARIMA is fitting to pre-break data.
                 Prophet handles trend changepoints natively. Try it."

Agent calls:     evaluate_model("sales_weekly", "Prophet", {"changepoint_prior_scale": 0.1})
                 → cv_score = -142.3 (vs AutoARIMA -198.7)

Agent calls:     fit_model("sales_weekly", "Prophet", {"changepoint_prior_scale": 0.1})
                 → version = "v7", staged

Agent calls:     promote_model("sales_weekly", "v7")
                 → promoted to Production

Agent writes:    audit log with full reasoning chain
```

The agent reached a conclusion a rule-based system structurally cannot: that the
*same* retrain trigger, on the *same* dataset, should produce a *different*
model family this time, because of accumulated failure history.

---

## MCP Tool Reference

All tools are exposed via `sktime-mcp`. The agent calls these via the MCP
protocol — no direct Python imports in the agent loop.

### Data & Profiling

| Tool | Description | Returns |
|------|-------------|---------|
| `profile_dataset(dataset_id)` | Statistical profile of the series | n_obs, freq, missing_pct, mean, std, CV |
| `get_dataset_history(dataset_id)` | Full production history | past models, scores, failures, drift events |
| `get_drift_history(dataset_id)` | Drift event log | timestamps, scores, methods, responses |

### Statistical Tests

| Tool | Description | Returns |
|------|-------------|---------|
| `run_stationarity_test(dataset_id)` | ADF + KPSS stationarity tests | p-values, conclusion |
| `detect_seasonality(dataset_id)` | Seasonal period and strength | period, strength, method |
| `check_structural_break(dataset_id)` | CUSUM-based break detection | break_detected, location, confidence |
| `detect_drift(dataset_id)` | Current drift scores | CUSUM score, ADWIN score, residual score |

### Training & Evaluation

| Tool | Description | Returns |
|------|-------------|---------|
| `list_candidate_models()` | Available sktime estimators | model names + metadata |
| `evaluate_model(dataset_id, model, params)` | CV evaluation without fitting | cv_score, fold_scores, duration |
| `fit_model(dataset_id, model, params)` | Fit and stage a model | version, staging_uri |
| `cross_validate_model(dataset_id, model, params, n_splits)` | Full CV with fold detail | per-fold scores, mean, std |
| `promote_model(dataset_id, version)` | Promote staged → Production | confirmation |
| `reject_model(dataset_id, version)` | Archive a staged model | confirmation |
| `update_model(dataset_id)` | Incremental update (no full retrain) | updated version |

### Serving & State

| Tool | Description | Returns |
|------|-------------|---------|
| `get_forecast(dataset_id, fh)` | Get prediction from Production model | predictions, model_version, cached |
| `get_production_model(dataset_id)` | Current Production model info | version, estimator, cv_score, promoted_at |
| `get_model_history(dataset_id)` | All model versions + outcomes | versions, scores, promotion history |
| `get_retrain_queue()` | Pending retrain jobs | queue contents |

---

## Architecture

### Components

```
sktime-agentic/
├── agent/
│   ├── loop.py              # ReAct agent loop — the main entry point
│   ├── memory.py            # Persistent production memory across runs
│   ├── prompts.py           # System prompt + context assembly
│   └── tools.py             # MCP tool call wrappers for the agent
│
├── mcp/
│   └── server.py            # FastMCP server — exposes all tools above
│
├── infrastructure/
│   ├── drift_monitor.py     # CUSUM + ADWIN — publishes signals, no decisions
│   ├── prediction_cache.py  # 3-tier: in-memory → Valkey → MLflow
│   ├── model_registry.py    # MLflow lifecycle helpers
│   └── data_loader.py       # Dataset fetching from S3/GCS
│
├── api/
│   └── main.py              # FastAPI: /forecast, /health, /admin
│
├── go/
│   ├── gateway/             # API gateway + request routing
│   └── orchestrator/        # Kafka consumer → agent trigger
│
└── tests/
    ├── test_agent_loop.py   # Agent reasoning tests with mocked tools
    ├── test_mcp_tools.py    # Tool correctness tests
    └── test_drift.py        # Drift detection tests
```

### What Each Layer Does

**Agent loop (`agent/loop.py`)**
The LLM runs a ReAct loop: observe the trigger + production context, reason,
call an MCP tool, observe the result, reason again, repeat until a terminal
action (promote, reject, update, escalate). Every step is logged.

**Persistent memory (`agent/memory.py`)**
The agent's context is not just the current trigger. It includes the full
history of every model ever trained on this dataset — CV scores, failure reasons,
drift patterns, how long each model survived in production. This is what allows
the agent to avoid repeating past mistakes. Stored in Valkey, keyed by
`dataset_id`.

**MCP server (`mcp/server.py`)**
FastMCP server exposing all tools listed above. Wraps sktime operations,
statistical tests, and MLflow lifecycle calls. This is what the agent calls —
the agent never imports sktime directly.

**Drift monitor (`infrastructure/drift_monitor.py`)**
CUSUM + ADWIN statistical detection. Publishes a signal to the agent trigger
queue when drift is detected. Makes no decisions — that is the agent's job.
Previously this triggered retraining directly; now it just notifies the agent.

**Go layer (`go/`)**
API gateway for routing external forecast requests.
Kafka consumer for converting production events into agent triggers.
Stateless — all state lives in Valkey and MLflow.

---

## Key Design Decisions

**Why is the LLM the orchestrator and not an advisor?**

Previous designs used the LLM as a post-hoc auditor — the model was already
selected and trained, then the LLM reviewed the outcome. This is advisory and
replaceable with a rule. In `sktime-agentic`, the LLM makes the decision before
any training happens. If the LLM decides not to retrain, nothing retrains.

**Why does the agent need memory across runs?**

A stateless LLM call sees one snapshot. A stateful agent sees that AutoARIMA
failed on this dataset twice in two months, that drift always follows a pattern
of 3–4 weeks after a promotional event, that the series has a structural break
that wasn't present six months ago. That accumulated context is what makes the
decision genuinely intelligent rather than reflexive.

**Why not just use sktime-mcp directly?**

You can. sktime-mcp exposes individual operations. `sktime-agentic` adds the
reasoning layer that decides which operations to call, in what order, with what
parameters, and whether to act on the results. sktime-mcp is the toolbox.
`sktime-agentic` is the engineer who knows when to use which tool.

**Why keep the deterministic drift monitor?**

CUSUM and ADWIN are better at detecting drift than an LLM polling a dataset.
The monitor does what statistics does well: signal detection. The agent does
what LLMs do well: reasoning about what to do in response.

---

## Relation to sktime-mcp

| | sktime-mcp | sktime-agentic |
|---|---|---|
| **Role** | Tool provider | Agent orchestrator |
| **Knows** | How to execute sktime operations | Why, when, in what order to call tools |
| **Memory** | None (stateless) | Full production history per dataset |
| **Decisions** | None | All of them |
| **LLM** | Not required | Core component |
| **Dependency** | — | Depends on sktime-mcp for tools |

---

## Quickstart

```bash
# Clone
git clone https://github.com/your-org/sktime-agentic
cd sktime-agentic

# Install
pip install -e ".[all]"

# Configure
cp .env.example .env
# set ANTHROPIC_API_KEY, MLFLOW_TRACKING_URI, VALKEY_URL

# Start the local services
# Option A: use local Python MLflow server instead of Docker MLflow
# Open a separate terminal and run:
python python/scripts/start_local_mlflow.py

# Then run the demo directly from your host Python environment:
python python/scripts/run_demo.py --dataset_id airline

# Option B: use Docker for Valkey and MLflow
# In one terminal:
docker compose up -d valkey mlflow

# In another terminal, run the demo inside the python-worker container:
docker compose run --rm python-worker python scripts/run_demo.py --dataset_id airline
```

## Local demo using sample CSV data

You can also run the demo against a sample CSV file from `python/tests/fixtures/sample_datasets`:

```bash
docker compose run --rm python-worker python scripts/run_demo.py \
  --dataset_id m4_monthly_subset_like.csv
```

```bash
python python/scripts/run_demo.py --dataset_id airline \
  --valkey_url redis://localhost:6379 \
  --mlflow_tracking_uri http://localhost:5000
```

```bash
python python/scripts/run_demo.py --dataset_id yahoo_s5_like_drift.csv \
  --local_dataset_dir python/tests/fixtures/sample_datasets
```

```bash
# Run the agent on a dataset with the original loop entrypoint
python -m sktime_agentic.agent.loop \
  --trigger cold_start \
  --dataset_id my_sales_series \
  --data path/to/series.csv
```

Or connect the MCP server to Claude Desktop and drive everything interactively:

```json
{
  "mcpServers": {
    "sktime-agentic": {
      "command": "python",
      "args": ["-m", "sktime_agentic.mcp.server"]
    }
  }
}
```

---

## ESoC 2026

This project is submitted under the **sktime agentic — own idea** track of
[ESoC 2026](https://github.com/european-summer-of-code/esoc2026), hosted by
the German Center for Open Source AI.

It contributes to the agentic sktime ecosystem by:
- Providing a production-grade agent loop that uses sktime-mcp as its tool surface
- Adding new MCP tools to the sktime-mcp ecosystem (statistical tests, production
  memory, drift history, structural break detection)
- Demonstrating a pattern for LLM-driven ML infrastructure that goes beyond
  single-shot model selection to multi-step, history-aware autonomous operation

Contributions to sktime-mcp made during this project are upstreamed directly
to the sktime-mcp repository.

---

## License

BSD 3-Clause — same as sktime.
