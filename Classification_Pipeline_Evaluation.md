# Agentic ML Pipeline — Classification Evaluation

Scope: this document covers **classification only** (regression and forecasting are out of scope).

## 1. Project Overview

The Agentic ML Pipeline is a local, single-user tool that takes a dataset plus a plain-English
business problem description and automatically plans, cleans, engineers features, trains
multiple candidate models, evaluates them, and produces an HTML recommendation report. An LLM
(via Groq) is used only for planning, evaluation narrative, business recommendation, and report
prose — it never sees raw data rows, only column names/dtypes/aggregated stats/metrics
(`tests/test_llm_safety.py`). All model selection is deterministic; the LLM cannot change which
model wins.

For classification specifically, there are two related code paths:

- The main Phase 1-10 LangGraph pipeline (`orchestration/graph.py`), which plans + trains + reports
  end-to-end for any problem type, including classification.
- A separate, deterministic, LLM-free **classification improvement-cycle workflow**
  (`tools/classification_cycle.py`), which runs up to 3 controlled training cycles specifically for
  classification problems and is independent of the LangGraph pipeline.

## 2. Current Workflow (Upload → Preprocessing → Training → Evaluation)

Main pipeline sequence (`orchestration/graph.py`):

```
ingest → profile_data → quality_analysis → detect_problem
      → planner_agent → validate_plan
      → feature_selection → eda → clean → feature_engineer → split → tune_models → train
      → evaluate → (retry back to clean, up to max_retries) or recommend → report → END
```

- `ingest → profile_data → quality_analysis → detect_problem`: deterministic, pandas-only
  (`tools/profiling.py`, `tools/problem_detection.py`); builds the `DatasetProfile`,
  `TargetAnalysis`, `DataQualityReport`, `ProblemDefinition` used by the planner.
- `planner_agent`: the only LLM call before training; produces an `ExperimentPlan` (candidate
  model shortlist, feature columns, evaluation metrics) — a shortlist, not the final choice.
- `validate_plan`: deterministic repair of the plan (hallucinated columns, missing defaults); may
  route straight to `END` with `needs_clarification` if a repair isn't safe.
- `feature_selection → clean → feature_engineer → split → train → evaluate → recommend → report`:
  described in Sections 6–10 below.

Via the API (`api/main.py`): upload a dataset (`POST /api/datasets`), start a run
(`POST /api/runs`), poll status (`GET /api/runs/{id}`), optionally cancel
(`POST /api/runs/{id}/cancel`), and fetch the final HTML report
(`GET /api/runs/{id}/report`). Runs execute on a bounded worker pool (`api/job_queue.py`) and
persist to SQLite (`api/db.py`).

## 3. Supported Problem Types

This document covers **classification** only. (Regression and time-series forecasting are also
supported by the pipeline but out of scope here.)

## 4. Implemented Models (Classification)

10 classification candidates, defined in `tools/model_registry.py` (`_classification_registry`,
lines ~402-425), all wrapped with a common train/predict/predict_proba interface:

| # | Candidate | Algorithm / config |
|---|---|---|
| 1 | `baseline` | `DummyClassifier(strategy="most_frequent")` |
| 2 | `logistic_regression` | `LogisticRegression(max_iter=1000)`; tuning: `C ∈ {0.1,1,10}`, `class_weight ∈ {None, "balanced"}` |
| 3 | `random_forest` | `RandomForestClassifier(n_estimators=100, random_state=42)`; tuning: `n_estimators`, `max_depth`, `min_samples_split` |
| 4 | `xgboost` | `XGBClassifier(eval_metric="logloss", random_state=42)` (optional dependency) |
| 5 | `lightgbm` | `LGBMClassifier(verbosity=-1, random_state=42)` (optional dependency) |
| 6 | `decision_tree` | `DecisionTreeClassifier(random_state=42)` |
| 7 | `svm` | `SVC(probability=True, random_state=42)`, features scaled |
| 8 | `knn` | `KNeighborsClassifier()`, features scaled |
| 9 | `naive_bayes` | `GaussianNB()` |
| 10 | `neural_network` | `MLPClassifier(random_state=42, max_iter=500)`, features scaled |

`resolve_candidates()` / `default_candidates()` filter the registry against the planner's
shortlist; unmatched names are tolerated, and the set falls back to `("baseline",)` if nothing
matches. A fixed `random_state=42` is set on every model that supports it, plus global seeding,
for reproducibility.

Note: `baseline` (`DummyClassifier`) is deliberately excluded from improvement cycles 2 and 3 in
`tools/classification_cycle.py` since it has no hyperparameters to vary.

## 5. Dataset Details

Example classification dataset shipped in the repo: `sample_data/churn_classification.csv`
— 300 rows, 6 columns: `customer_name`, `tenure_months`, `monthly_charge`, `contract_type`,
`support_calls`, `churn` (target, binary).

Note: `customer_name` is included as a raw feature column with no evident PII detection/exclusion
filter anywhere in the pipeline (see Section 11).

## 6. Preprocessing

**Feature selection** (`tools/feature_selection.py`, runs before cleaning): deterministic,
pandas-only. Keeps target/time/entity columns plus the planner's `feature_columns`; drops
everything else with a reason (constant, near-constant, id-like, possible leakage from
`DataQualityReport`, or "not selected by plan"). By design, this step does **not** check the
quality of features created later by cleaning/feature-engineering (e.g. one-hot columns) — a
documented scope boundary, not a bug.

**Cleaning** (`tools/cleaning.py`):
- Drops duplicate rows.
- Converts messy numeric-as-text/range columns (e.g. `"2100 - 2850"` → midpoint) if ≥80% of
  values parse as numeric.
- Missing-value imputation: numeric columns use median (or forward/back-fill if a time column is
  present); categorical columns use mode.
- Optional IQR-based outlier capping (only if explicitly enabled — off by default).
- Categorical encoding: one-hot encoding (`pd.get_dummies`, drop_first) for columns with ≤15
  unique values; label-encoding (`pd.factorize`) otherwise.
- Target/time/entity columns are explicitly excluded from text/categorical handling (a past
  regression where date/entity columns were mis-encoded before feature-engineering could parse
  them was fixed by adding this exclusion).

**Feature engineering / scaling** (`tools/feature_engineering.py`): for classification,
`StandardScaler` is applied to all numeric feature columns (target excluded). No new features or
categorical encoding happen here for classification — that's handled by cleaning. (Date parts,
lag, and rolling-window features in this module are forecasting-only.)

**Splitting** (`tools/splitting.py`):
- Main pipeline (`split_data`): random `train_test_split(shuffle=True)`, default `test_size=0.2`;
  class distribution is logged.
- Classification improvement-cycle workflow (`split_train_val_test`): a one-time **stratified**
  3-way `TRAIN`/`VALIDATION`/`TEST` split — default `val_size=test_size=0.2` of the original data
  (so roughly 60/20/20). Falls back to a plain random split if a class is too small to stratify at
  that ratio. `TEST` is carved off once and never touched again until the final evaluation step.

## 7. Model Training

**Main pipeline**: `tune_models` (optional hyperparameter tuning) then `train` fits each
candidate from the planner's shortlist on the training split, in-process, with fixed
`random_state` values for reproducibility.

**Classification improvement cycles** (`tools/classification_cycle.py`), entry point
`run_classification_cycles(df, target_column, feature_columns, config=None)`:
- Up to `max_cycles` (default 3, configurable via `CLASSIFICATION_CYCLE_MAX_CYCLES`).
- Cycle 1: trains every available classification candidate with default hyperparameters.
- Cycles 2–3: a deterministic, data-driven adjustment — if the target is imbalanced (minority
  class share < 15%), applies class-weight balancing (cycle 2) and/or minority oversampling
  (cycle 3) for models that support it; otherwise applies a small fixed hyperparameter variant to
  the top-3 performers so far (or all models, if configured). Never a blind identical retrain.
- Stops early the moment any candidate's validation metrics satisfy every configured acceptance
  threshold (AND semantics); otherwise runs all `max_cycles` and reports
  `status: "threshold_not_met"`.
- Best model is selected by the configured `selection_metric` (default `"f1"`) on **validation**
  metrics across *all* cycles, not just the last, then evaluated exactly once, separately, on the
  fixed `TEST` set — the test set never feeds back into model selection.

**Technical retries** (`tools/technical_retry.py`): transient failures (`TimeoutError`,
`ConnectionError`, `OSError`) are retried with exponential backoff (default up to 3 attempts),
completely separate from and never advancing the improvement-cycle counter.
`PermanentTrainingError` (e.g. missing target column, invalid/insufficient data, unsupported
model) is never retried.

## 8. Evaluation Metrics

Computed in `tools/evaluation.py`:
- Always: accuracy, weighted precision, weighted recall, weighted F1.
- ROC-AUC: computed when the problem is binary and predicted probabilities are available; for
  multiclass, `compute_classification_metrics_detailed` computes macro-averaged ROC-AUC via
  `roc_auc_score(..., multi_class=strategy, average="macro")`, with `strategy` configurable
  between one-vs-rest (`"ovr"`, default) and one-vs-one (`"ovo"`).
- Additional detail (used by the improvement-cycle workflow): full confusion matrix, per-class
  precision/recall/F1, and PR-AUC (`average_precision_score`).
- All metric computation is wrapped so a single metric failure never crashes a candidate's
  evaluation.
- Acceptance criteria in the improvement-cycle workflow default to
  `accuracy ≥ 0.70`, `roc_auc ≥ 0.70`, `f1 ≥ 0.65` (AND semantics).

## 9. Model Comparison

`select_primary_metric` picks the ranking metric: it honors the planner's requested evaluation
metric if every successful candidate reports it, otherwise defaults to ROC-AUC (if every
candidate has it) or accuracy. `build_model_comparison` is the sole ranking authority — the
winner is simply the top of the ranked list; failed or metric-missing candidates are kept in the
comparison table but left unranked.

Downstream LLM steps cannot override this:
- `agents/evaluator.py`: the LLM only decides retry vs. proceed and supplies narrative reasoning;
  `best_model` is unconditionally overwritten with the deterministic winner. If the retry cap
  (`max_retries`, default 2) is reached, the pipeline auto-proceeds without even asking the LLM.
- `agents/recommender.py`: turns the deterministic winner into business-facing narrative;
  `recommended_model` is forcibly overwritten with the deterministic winner, and any
  LLM-cited metrics/features are validated against actual results and dropped if fabricated.
- `agents/reporter.py`: renders the final HTML report from a fixed template; the LLM supplies only
  the executive-summary and approach-narrative prose — every number, table, and chart comes from
  deterministic pipeline objects.

## 10. Error Handling

- **Transient/technical failures** (timeout, connection, I/O errors) are retried automatically
  with exponential backoff, independent of model-quality retries (`tools/technical_retry.py`).
- **Permanent failures** (`PermanentTrainingError` — missing target column, invalid/insufficient
  data, unsupported model) are never retried; they surface immediately.
- **Model-quality retries** in the main pipeline: `evaluate` can decide `retry` (back to
  cleaning), `accept`, or `stop`, up to `max_retries` (default 2); retrying is intended only when
  every candidate performed poorly, not merely to chase a preferred alternative.
- **Per-metric robustness**: metric computation in `tools/evaluation.py` is wrapped in
  try/except, so one metric failing (e.g. ROC-AUC undefined for a degenerate class split) doesn't
  crash the whole evaluation for that candidate.
- **Cancellation**: every pipeline node is wrapped to check a cancel event before running
  (cooperative — a node already inside a blocking `.fit()` call finishes it first).
- **Experiment tracking**: every run is recorded in a `finally` block regardless of outcome
  (complete / needs_clarification / cancelled / raised); a tracking failure itself is swallowed,
  never raised.

## 11. Limitations

- **No PII exclusion mechanism.** No PII detection/exclusion exists anywhere in cleaning,
  feature selection, or profiling. All column names/dtypes/stats reach the LLM, and all columns
  (e.g. `customer_name` in the sample churn dataset) reach model training unless the planner
  itself chooses to drop them. A previous manual denylist was removed because it silently dropped
  columns from training entirely, which could hurt model quality without the user realizing why.
- **Feature-selection doesn't validate engineered features.** `feature_selection` runs before
  cleaning/encoding, so it can't evaluate the quality of one-hot/derived columns created
  afterward — a documented scope boundary.
- **Cooperative cancellation only** — a run inside a blocking `.fit()` call or LLM request
  finishes that call before a cancel takes effect.
- **No dedicated run-history UI** — `GET /api/runs` exists but has no frontend view yet.
- **Report output is HTML only** (no PDF/DOCX), to avoid native rendering dependencies.
- **Improvement-cycle adjustments are limited to two knobs** (class-weight/oversampling for
  imbalance, or a small fixed hyperparameter variant) — not a general hyperparameter search.

## 12. Future Improvements

- Broader hyperparameter tuning/search beyond the current fixed-variant approach in the
  improvement-cycle workflow.
- Automated PII detection with an explicit, visible warning to the user rather than silent
  dropping or no filtering at all.
- Deeper feature-quality checks after encoding/engineering, not just before.
- Model explainability (e.g. SHAP/feature importance) surfaced in the report.
- A dedicated run-history/monitoring UI.
- Non-cooperative (harder) cancellation for long-running `.fit()` calls.
