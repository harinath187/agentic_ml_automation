# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

An agentic ML pipeline (`Agentic_ML_Pipeline_Implementation_Plan.txt`, Phases 1-10). Given a
dataset + a business problem description, it plans, cleans, engineers features, trains models
via AutoGluon (plus classical candidates), evaluates, and produces an HTML recommendation
report. The LLM only ever sees column names/dtypes/stats/metrics/logs - **never raw rows** (see
`tests/test_llm_safety.py`). It is a local, single-user tool - no auth, no hosted/multi-tenant
deployment.

## Setup

```
py -3.11 -m venv .venv
.venv\Scripts\pip install -r requirements.txt
copy .env.example .env
```

AutoGluon requires Python <=3.11; this project targets 3.11 for that reason.

- `.env` holds `GROQ_API_KEY` (and optional Phase-9 tuning vars documented in `.env.example`).
  **Do not read, print, or edit `.env`** - it is gitignored and holds a live API key. Work from
  `.env.example` instead when you need to know what variables exist.
- Never commit `.env`, `reports/app.db`, `reports/experiments.jsonl`, `uploads/`, or
  `AutogluonModels/` - all are gitignored generated/local state.

## Commands

Run (CLI):
```
.venv\Scripts\python main.py --data sample_data\churn_classification.csv ^
    --description "Predict which customers will churn next month" ^
    --sensitive-columns customer_name
```

Run (Web UI - FastAPI backend + React frontend):
```
.venv\Scripts\uvicorn api.main:app --reload --port 8000
```
```
cd frontend && npm install   # first time only
npm run dev
```
Vite dev server (http://localhost:5173) proxies `/api/*` to the FastAPI backend on port 8000.

Tests: **do not run pytest yourself** - if a change should be verified with tests, tell the user
which test file/command to run and let them run it manually (e.g.
`.venv\Scripts\pytest tests\test_cleaning.py -k some_case`). No API key is needed for tests - the
Planner/Evaluator/Reporter LLM calls are stubbed. `tests/test_pipeline_integration.py` and
`tests/test_phase10_e2e_validation.py` additionally require `langgraph` and
`autogluon.tabular`/`autogluon.timeseries` and are skipped automatically otherwise.

## Architecture

The pipeline is a LangGraph `StateGraph` (`orchestration/graph.py`, built by `build_graph()` and
driven by `run_pipeline()`) over a single `PipelineState` dict threaded node-to-node. The
DataFrame itself lives only in this process's local state and is never serialized to an LLM -
only `schema_summary`/`eda_summary`/`metrics` are.

**Deterministic Data Intelligence (no LLM call), always first:**
```
ingest -> profile_data -> quality_analysis -> detect_problem
```
Pandas-only (`tools/profiling.py`, `tools/problem_detection.py`); produces the `DatasetProfile`,
`TargetAnalysis`, `DataQualityReport`, `ProblemDefinition` the Planner reasons over.

**Planning:**
```
planner_agent -> validate_plan
```
`planner_agent` (`agents/planner.py`) is the only LLM call before training - produces an
`ExperimentPlan` (candidate shortlist, not the final model choice) and the run's
`scope_strategy`. `validate_plan` (`tools/plan_validation.py`) is deterministic: repairs
obviously-fixable issues (hallucinated columns, missing defaults) and only falls back to
`needs_clarification` (routes straight to `END`) when a repair isn't safe.

**Scope-dependent branch**, chosen by `route_after_validate_plan` on
`ExperimentPlan.scope_strategy`:

| scope_strategy | route |
|---|---|
| `pooled` / none | `eda -> clean -> feature_engineer -> split -> train` |
| `single_entity` | `filter_entity` (subset to one entity, drop entity column) -> rejoins at `eda` |
| `per_entity` | `per_entity_pipeline` - loops clean/engineer/split/train per entity, in-process, capped at 25 entities, skips entities with <10 rows |
| `hierarchical` | `hierarchical_train` - AutoGluon `TimeSeriesPredictor` across all entities via `item_id` |

**Evaluation, retry, finish:**
```
evaluate -> (retry: back to clean / per_entity_pipeline / hierarchical_train)
         -> recommend -> report -> END
```
`evaluate` (`agents/evaluator.py`) decides `retry`/`accept`/`stop` up to `max_retries` (default
2). `recommend` (`agents/recommender.py`) explains the deterministic evaluation engine's
already-decided winner for a business audience - **it cannot change which model won**. `report`
(`agents/reporter.py`) renders the HTML report; every number comes from deterministic pipeline
objects, the LLM supplies only executive-summary/approach-narrative prose.

**Cross-cutting concerns**, applied uniformly across nodes:
- **Cancellation**: every node wrapped by `_cancellable()`, checks `state["cancel_event"]`
  before running, raises `PipelineCancelled` - checked only at node boundaries (a node already
  inside a blocking call like AutoGluon's `.fit()` finishes that call first).
- **Progress reporting**: `run_pipeline()` uses `app.stream(..., stream_mode="updates")` so
  `api/run_store.py` can persist which step a long-running job is on via `progress_callback`.
- **Experiment tracking**: `run_pipeline()` records an `ExperimentRecord` in a `finally` block
  regardless of outcome (complete/needs_clarification/cancelled/raised).

### Entity/grouping-aware planning

The Planner detects entity/group-identifier columns (store, customer, machine, etc. - inferred
from cardinality/repetition, not column names) and chooses `scope_strategy`. `pooled` keeps
lag/rolling features and chronological splits grouped per entity so nothing leaks across
entities (`tools/feature_engineering.py`, `tools/splitting.py`). If the Planner can't confidently
pick a strategy, it asks a clarification question instead of guessing.

### Experiment tracking

Every `run_pipeline()` call is recorded (`tools/experiment_tracking.py`) via an
`ExperimentStore` interface - `JSONLinesExperimentStore` (default, appends to
`reports/experiments.jsonl`), `SQLiteExperimentStore`, or `InMemoryExperimentStore` (tests).
Recording never affects pipeline behavior - a tracking failure is swallowed, never raised.
Reproducibility: `set_global_seeds()` plus fixed `random_state` on every model candidate
(`tools/model_registry.py`) and on the train/test split (`tools/splitting.py`).

### Production readiness (Phase 9) - `api/`

Moved off in-memory dicts and one-thread-per-run onto persistent storage + a bounded worker
pool, still single-machine:
- `api/db.py` - SQLite (WAL mode) for upload metadata and run history (`reports/app.db`,
  `APP_DB_PATH`).
- `api/job_queue.py` - bounded `ThreadPoolExecutor` (`PIPELINE_MAX_WORKERS`, default 2); jobs
  beyond that queue (`PIPELINE_MAX_QUEUED_RUNS`, default 20, 429 past that).
- Job status: `queued -> running -> completed|failed|cancelled|needs_clarification` (note this
  differs from the pre-Phase-9 `running`/`complete`/`error` vocabulary).
- Cancellation: `POST /api/runs/{run_id}/cancel` - cooperative via `PipelineCancelled`, not
  instant.
- Report filenames keyed by `run_id` (not timestamp) to stay safe under concurrent runs;
  `tools/report_charts.py` builds matplotlib `Figure`s directly (not via `pyplot`, whose global
  current-figure state isn't thread-safe).
- Structured JSON logging (`tools/logging_config.py`, `configure_logging()`).
- `recover_interrupted_runs()` marks runs still `running` at process start as `failed` (a
  crashed process can't resume an in-flight run - see its docstring).
- Resource limits: `MAX_UPLOAD_MB` (default 200), `PIPELINE_MAX_RUNTIME_S` (default 3600) on top
  of each run's own `time_limit_s`.
- `tools/file_cleanup.py` - background loop (`CLEANUP_INTERVAL_S`, default 6h) deletes
  uploads/reports/AutoGluon model dirs older than `FILE_RETENTION_HOURS` (default 7 days), never
  anything touched in the last hour.
- Uploaded files: size-capped streaming writes, stored filename is always a fresh uuid4 (never
  derived from the client filename - no path-traversal surface), failed parses are rejected with
  a 400 and the partial file deleted.

## Known limitations (v1)

- Forecasting defaults to regression over lag/rolling/date features (pooled/single_entity/
  per_entity); `autogluon.timeseries` is only pulled in for `hierarchical`.
- `hierarchical` forecasting is univariate (target + time + entity only) - no other covariates.
- `per_entity` capped at 25 entities/run; skipped entities are reported, not silently dropped.
- PII handling is a manual column allowlist/denylist supplied at run time - no automated PII
  detection.
- Cancellation is cooperative: a run inside a blocking call (`.fit()`, an LLM request) finishes
  that call before a cancel takes effect.
- Report output is HTML only (no PDF/DOCX, to avoid native deps like WeasyPrint's GTK on
  Windows).
- `GET /api/runs` lists recent runs but there's no dedicated UI for it yet.
