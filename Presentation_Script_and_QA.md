# Agentic ML Pipeline — Presentation Script & Q&A Prep

## How to Start

Open with the problem, not the tech:

> "Right now, if a business person has a dataset and a question — like 'which customers will
> churn' — they need a data scientist to clean the data, pick a model, tune it, and explain the
> results. This project automates that whole chain: you upload a dataset, describe the business
> problem in plain English, and the system plans, cleans, trains multiple models, evaluates them,
> and gives you back an HTML report explaining which model to trust and why."

Then one sentence on what makes it "agentic":

> "It uses an LLM for the parts that need judgment — planning the approach, deciding whether to
> retry, and writing the business explanation — but every number, every metric, every model
> ranking is computed deterministically in code. The LLM never sees raw data, and it can't
> override which model actually wins."

That second sentence is the single most important line in the whole pitch — lead with it, repeat
it if asked about safety/trust.

## Script Outline (5–7 minutes)

1. **The problem** (30s) — manual ML workflow is slow and requires expertise.
2. **The workflow, end to end** (90s) — walk the pipeline stages in order:
   - Upload dataset + describe the business problem.
   - Deterministic data profiling (column types, quality issues, target detection) — no LLM yet.
   - Planner (LLM) proposes a shortlist of candidate models and a strategy.
   - Deterministic cleaning, feature engineering, train/test split.
   - Train up to 10 classification models.
   - Deterministic evaluation ranks them on accuracy/precision/recall/F1/ROC-AUC.
   - Evaluator (LLM) decides retry or proceed — but can't change the winner.
   - Recommender (LLM) explains the winner in business terms.
   - Reporter renders the final HTML report.
3. **Why it's trustworthy** (60s) — LLM-safety boundary: raw rows never reach the LLM; model
   selection is 100% deterministic code, not LLM judgment; every claim in the report is validated
   against real metrics before being shown.
4. **Live demo** (2–3 min) — upload `sample_data/churn_classification.csv`, run it, show the
   report: model comparison table, winning model, business explanation.
5. **Where it's headed** (30s) — hyperparameter search depth, explainability (SHAP), PII
   handling, more problem types.

## What to Highlight

- **Deterministic core, LLM at the edges.** This is the single strongest technical claim — say
  it more than once. The LLM plans and narrates; code decides and computes.
- **Multi-model comparison, not a single black box.** 10 classification algorithms trained and
  ranked automatically (baseline, logistic regression, random forest, XGBoost, LightGBM,
  decision tree, SVM, KNN, naive Bayes, neural network).
- **Self-improving cycles.** For classification specifically, there's a dedicated 3-cycle
  improvement loop: cycle 1 tries everything with defaults, cycles 2–3 adapt automatically (class
  balancing for imbalanced targets, or hyperparameter variants otherwise) and stop early the
  moment a model clears the acceptance bar.
- **Held-out test discipline.** The test set is carved off once, up front, and never touched
  again until the very last, single evaluation — so the reported number isn't optimistic.
- **Retry logic that's actually two different mechanisms.** Transient technical failures
  (timeouts, connection errors) retry automatically without cost to model quality attempts;
  permanent failures (bad data, missing column) fail fast and clearly instead of retrying
  uselessly.
- **No-raw-data-to-LLM guarantee**, backed by an actual test (`tests/test_llm_safety.py`) rather
  than just a policy statement.
- **Local, single-user, no auth.** Be upfront about this — it's a deliberate scope choice, not
  an oversight, if asked.
- **Known gaps, stated honestly.** Have Section 11 (Limitations) of the evaluation doc ready:
  no PII exclusion, cooperative-only cancellation, no run-history UI, outlier handling is
  optional/off by default.

## Likely Questions & Answers

**Q: Does the LLM ever see my actual data?**
A: No. It only ever sees column names, data types, aggregated statistics, and computed metrics —
never raw rows. This is enforced and covered by a dedicated test (`tests/test_llm_safety.py`).

**Q: Can the AI just make up a good-sounding recommendation that isn't true?**
A: No — the winning model is chosen purely by deterministic code comparing metrics
(`build_model_comparison`). The LLM's job downstream is only to explain that already-decided
winner in business language, and any metric or feature it cites in that explanation is checked
against the real results and dropped if it doesn't match.

**Q: What models does it try, and how does it pick the best one?**
A: For classification: baseline, logistic regression, random forest, XGBoost, LightGBM, decision
tree, SVM, KNN, naive Bayes, and a neural network — 10 candidates. They're ranked by whichever
metric the plan calls for, defaulting to ROC-AUC or accuracy, and the top of that ranking is the
winner. No human or LLM judgment enters that ranking step.

**Q: How does it handle missing values / messy data?**
A: Numeric columns are median-imputed (or forward/back-filled for time-based data); categorical
columns use the mode. Text-like numeric ranges (e.g. "2100 - 2850") are parsed to their midpoint
when most of a column matches that pattern. Categorical columns are one-hot or label-encoded
depending on cardinality.

**Q: What about outliers?**
A: Optional IQR-based capping exists but is off by default — it doesn't auto-detect and doesn't
remove rows, only clips extreme values when explicitly enabled. Tree-based models tolerate
outliers reasonably well regardless; distance- and gradient-sensitive models (SVM, KNN, logistic
regression, neural net) are more exposed if it's left off. This is a known area for improvement.

**Q: How do you avoid overfitting / an inflated accuracy number?**
A: The classification workflow carves off a stratified test set once, up front. Every improvement
cycle only ever sees train/validation data; the test set is touched exactly once, at the very
end, on the already-selected best model.

**Q: What happens if a model fails to train?**
A: Transient/technical failures (timeouts, connection issues) are retried automatically with
backoff, without costing a "real" improvement attempt. Permanent failures (missing target column,
invalid data, unsupported model) fail immediately and are never retried. Metric computation
failures for one candidate don't crash the run — that candidate is just marked without that
metric.

**Q: Does it work for imbalanced classes (e.g. rare fraud/churn cases)?**
A: Yes — the improvement-cycle workflow specifically detects imbalance (minority class under
15%) and automatically applies class-weight balancing and/or minority oversampling in later
cycles for models that support it.

**Q: Is my data private / sent anywhere?**
A: It's a local, single-user tool — no hosted multi-tenant deployment, no auth layer. The only
external call is the LLM API (Groq), and only for schema/metric-level summaries, never raw rows.

**Q: What's not handled well yet?**
A: Be honest here — there's no PII detection or exclusion (all columns, including something like
a name column, are used unless the planner drops them); cancellation is cooperative so a run
already inside a training call finishes before stopping; there's no dedicated UI for run history
yet; and outlier handling isn't automatic.

**Q: Why use an LLM at all if the core decisions are deterministic?**
A: Because the hard part for a non-technical user isn't computing metrics — it's knowing what
approach to take and what the numbers mean. The LLM handles judgment calls (which modeling
strategy fits this data, whether results are good enough to ship, how to explain a ROC-AUC number
to a business stakeholder) while code handles anything that must be trustworthy and reproducible.

**Q: Can this scale to production / many users?**
A: Not today — it's explicitly scoped as local and single-user, no auth. That's a stated
boundary, not a bug.
