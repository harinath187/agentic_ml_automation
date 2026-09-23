"""LangGraph orchestration wiring together all agents and tools.

Standard flow (scope_strategy is pooled or null):
    Ingest -> Profile Data -> Quality Analysis -> Detect Problem ->
    Planner Agent -> Validate Plan -> Feature Selection -> EDA -> Clean ->
    Feature Engineer -> Split -> Train -> Evaluate ->
    (retry loop back to Clean, or) -> Recommend -> Report -> END.

Feature Selection (tools/feature_selection.py) is deterministic, pandas-only:
subsets the dataframe to the validated ExperimentPlan's feature_columns plus
target/time/entity columns, so the DataQualityReport exclusion lists
(id-like, near-duplicate/leakage, constant/near-constant columns) and the
plan's own feature list actually determine what reaches training, not just
advisory context. Runs before Clean/Feature Engineer (which one-hot encode
and create lag/rolling columns) so it never has to reconcile its exclusion
lists against expanded/engineered column names - see that module's docstring
for the resulting scope boundary (engineered-feature quality isn't checked).
Skipped for hierarchical scope (already univariate by design); applied once
for per_entity scope (node_per_entity_pipeline), globally across all entities
rather than per entity.

Recommend (Phase 5) is the Recommendation Agent (agents/recommender.py): it
explains the deterministic evaluation engine's already-decided winner
(tools/evaluation.py's ModelComparison, produced in Train/node_train) for a
business audience. It cannot change which model won - see
agents/recommender.py's validate_recommendation.

Profile Data / Quality Analysis / Detect Problem are the deterministic Data
Intelligence layer (tools/profiling.py, tools/problem_detection.py) -
pandas-only, no LLM call. They run before the Planner so it reasons over
structured facts (DatasetProfile, TargetAnalysis, DataQualityReport,
ProblemDefinition) rather than inferring everything itself from a thin
schema summary. Validate Plan is likewise deterministic: it repairs
obviously-fixable issues in the Planner's ExperimentPlan (tools/plan_validation.py)
and only falls back to needs_clarification when a repair isn't safe -
never raises.

Entity-aware branches, chosen by the Planner via ExperimentPlan.scope_strategy:
    single_entity  -> Validate Plan -> filter_entity (subset to one entity,
                       drop the entity column) -> re-joins the standard flow
                       at EDA.
    per_entity     -> Validate Plan -> per_entity_pipeline (loops clean/
                       engineer/split/train once per entity value, in-process,
                       never touching an LLM per entity) -> Evaluate.
    hierarchical   -> Validate Plan -> hierarchical_train (AutoGluon
                       TimeSeriesPredictor across all entities via item_id
                       grouping) -> Evaluate.

The DataFrame lives only in this process's local state (never serialized to
an LLM); only `schema_summary` / `eda_summary` / `metrics` are ever passed to
agents.planner / agents.evaluator / agents.reporter.
"""
from __future__ import annotations

import functools
import threading
import time
import uuid
from collections import defaultdict
from typing import Any, Callable, Optional, TypedDict

from langgraph.graph import END, StateGraph

from agents import evaluator, planner, recommender, reporter
from agents.schemas import (
    DataQualityReport,
    DatasetProfile,
    EvaluatorDecision,
    ExperimentPlan,
    ProblemDefinition,
    ProblemType,
    RecommendationOutput,
    ScopeStrategy,
    TargetAnalysis,
)
from data_ingestion.loader import extract_schema, load_dataset
from tools import (
    automl_training,
    cleaning,
    eda,
    experiment_tracking,
    feature_engineering,
    feature_selection,
    failure_analysis,
    hyperparameter_tuning,
    model_registry,
    model_runner,
    plan_validation,
    problem_detection,
    profiling,
    report_charts,
    splitting,
    text_vectorization,
)
from tools.logging_config import get_logger

logger = get_logger(__name__)

MAX_RETRIES_DEFAULT = 2

# per_entity scope safeguards: cap total entities trained (each entity is a
# full AutoML fit) and require a minimum row count per entity to bother.
MAX_PER_ENTITY = 25
MIN_ROWS_PER_ENTITY = 10


class PipelineCancelled(RuntimeError):
    """Raised between pipeline steps when the run's cancel_event has been
    set (Phase 9 job cancellation - see api/job_queue.py). Caught in
    run_pipeline() and recorded as an ExperimentRecord with status
    "cancelled" rather than "error".

    Cancellation is checked at node boundaries (see _cancellable() below),
    and at cooperative checkpoints inside the classical tuning loop. A run
    already partway through a blocking model fit finishes that fit before
    cancellation takes effect. There is no way to preempt a single blocking
    call mid-flight without a subprocess-based worker, which Phase 9
    deliberately avoids (see this module's scope notes).
    """


class PipelineState(TypedDict, total=False):
    run_id: str
    cancel_event: Any  # Optional[threading.Event] - process-local only, never serialized/sent to an LLM
    file_path: str
    business_description: str
    max_retries: int
    progress_callback: Optional[Callable[[str, dict], None]]

    df: Any
    schema_summary: dict
    dataset_profile: DatasetProfile
    data_quality_report: DataQualityReport
    target_analysis: TargetAnalysis
    problem_definition: ProblemDefinition
    plan: ExperimentPlan
    needs_clarification: bool

    feature_selection_log: dict  # absent for hierarchical scope, which trains univariate (no selection applies)
    eda_summary: dict
    cleaned_df: Any
    cleaning_log: dict
    engineered_df: Any
    feature_log: dict
    train_df: Any
    test_df: Any
    split_log: dict
    text_vectorizers: dict
    text_vectorization_log: dict
    tuned_model_params: dict
    tuning_results: dict

    metrics: dict
    chart_data: dict
    retry_count: int
    decision: EvaluatorDecision
    recommendation: RecommendationOutput

    report_path: str
    error: Optional[str]

    experiment_record: "experiment_tracking.ExperimentRecord"


def node_ingest(state: PipelineState) -> PipelineState:
    df = load_dataset(state["file_path"])
    schema_summary = extract_schema(df)
    return {"df": df, "schema_summary": schema_summary}


def node_profile_data(state: PipelineState) -> PipelineState:
    """Deterministic Data Intelligence, step 1: dataset-level profiling."""
    dataset_profile = profiling.profile_dataset(state["df"])
    return {"dataset_profile": dataset_profile}


def node_quality_analysis(state: PipelineState) -> PipelineState:
    """Deterministic Data Intelligence, step 2: target signals + quality report.

    Both are pandas-only (no LLM call) and depend only on node_profile_data's
    output, so they run once per pipeline invocation regardless of scope
    strategy or retries.
    """
    dataset_profile = state["dataset_profile"]
    target_analysis = profiling.analyze_target(state["df"], dataset_profile)
    data_quality_report = profiling.analyze_data_quality(
        state["df"], dataset_profile,
        likely_target_column=target_analysis.recommended_target,
    )
    return {"target_analysis": target_analysis, "data_quality_report": data_quality_report}


def node_detect_problem(state: PipelineState) -> PipelineState:
    """Deterministic Data Intelligence, step 3: problem-type detection.

    Still no LLM call - narrows problem_type/target_column candidates (and,
    when a datetime column exists, time-series signals: frequency, missing
    periods, seasonality, trend) before the Planner runs.
    """
    problem_definition = problem_detection.detect_problem(
        state["df"],
        state["dataset_profile"],
        state["target_analysis"],
        state["data_quality_report"],
    )
    return {"problem_definition": problem_definition}


def node_planner_agent(state: PipelineState) -> PipelineState:
    """The only LLM call before training: produces an ExperimentPlan - a
    shortlist of candidate approaches to test, not the final model choice."""
    plan = planner.build_plan(
        state["business_description"],
        state["schema_summary"],
        dataset_profile=state.get("dataset_profile"),
        quality_report=state.get("data_quality_report"),
        target_analysis=state.get("target_analysis"),
        problem_definition=state.get("problem_definition"),
    )
    return {"plan": plan, "needs_clarification": plan.needs_clarification}


def node_validate_plan(state: PipelineState) -> PipelineState:
    """Deterministic validation/repair of the Planner's ExperimentPlan - no
    LLM call. Repairs what it safely can (hallucinated columns, missing
    defaults) using the Data Intelligence report; anything else surfaces via
    needs_clarification, the same mechanism the Planner itself uses.
    """
    validated_plan = plan_validation.validate_plan(
        state["plan"],
        state["dataset_profile"],
        state["target_analysis"],
        state["data_quality_report"],
    )
    return {"plan": validated_plan, "needs_clarification": validated_plan.needs_clarification}


def node_filter_entity(state: PipelineState) -> PipelineState:
    """single_entity scope: subset to one entity, drop the now-constant entity column."""
    plan = state["plan"]
    if not plan.entity_column or plan.entity_filter_value is None:
        raise ValueError(
            "scope_strategy is single_entity but entity_column/entity_filter_value "
            "was not set by the Planner."
        )
    df = state["df"]
    filtered = df[df[plan.entity_column].astype(str) == str(plan.entity_filter_value)]
    if filtered.empty:
        raise ValueError(
            f"No rows found for {plan.entity_column!r} == {plan.entity_filter_value!r}"
        )
    return {"df": filtered.drop(columns=[plan.entity_column])}


def node_feature_selection(state: PipelineState) -> PipelineState:
    """Deterministic, pandas-only: subsets state["df"] to plan.feature_columns
    plus target/time/entity columns before cleaning/feature-engineering ever
    runs, so the ExperimentPlan/DataQualityReport exclusion lists actually
    determine what gets trained on instead of just being advisory context.
    Not run for hierarchical scope (see tools/feature_selection.py docstring
    and route_after_validate_plan) - state["feature_selection_log"] is
    correspondingly absent for that scope, which downstream consumers
    (agents/reporter.py, tools/experiment_tracking.py) must treat as valid.
    """
    selected_df, log = feature_selection.apply_feature_selection(
        state["df"], state["plan"], state["data_quality_report"]
    )
    return {"df": selected_df, "feature_selection_log": log}


def node_eda(state: PipelineState) -> PipelineState:
    summary = eda.run_eda(state["df"])
    return {"eda_summary": summary}


def node_cleaning(state: PipelineState) -> PipelineState:
    plan = state["plan"]
    # A retry always re-cleans aggressively; a first pass does too if the
    # Planner's preprocessing_requirements explicitly asked for it - previously
    # aggressive_outlier_handling ignored the plan entirely on a first attempt.
    aggressive = state.get("retry_count", 0) > 0 or "cap_outliers" in plan.preprocessing_requirements
    # entity_column is only threaded through for pooled forecasting, same
    # condition node_feature_engineering/node_split already use - see
    # tools/cleaning.py's docstring for why it must never be one-hot/label
    # encoded away before feature engineering/splitting can group by it.
    entity_column = plan.entity_column if plan.scope_strategy == ScopeStrategy.POOLED else None
    text_columns = [c for c in state.get("dataset_profile", DatasetProfile(row_count=0, column_count=0)).text_columns
                    if c in plan.feature_columns]
    df = state["df"]
    problem_type = plan.problem_type.value if plan.problem_type else "regression"

    # Leakage-safe cleaning (TabularCleaner, tools/cleaning.py): node_split
    # below re-runs splitting.split_data() on the (row-count/row-order
    # preserving) engineered_df with these exact same arguments and gets a
    # deterministic partition (fixed random_state for shuffle-splits, a
    # position-based chronological cutoff for forecasting) - fixed_state
    # here reproduces that same partition early, purely to fit imputation/
    # outlier statistics on TRAIN rows only, then applies those stored
    # statistics to every row (train and test alike) via transform(). This
    # avoids computing medians/modes/IQR-bounds over train+test combined,
    # which the previous clean-before-split ordering did silently.
    train_for_fit, _test_for_fit, _ = splitting.split_data(
        df,
        problem_type=problem_type,
        time_column=plan.time_column,
        entity_column=entity_column,
        target_column=plan.target_column,
    )
    cleaner = cleaning.TabularCleaner(aggressive_outlier_handling=aggressive)
    cleaner.fit(
        train_for_fit,
        target_column=plan.target_column,
        time_column=plan.time_column,
        entity_column=entity_column,
        text_columns=text_columns,
    )
    cleaned_df, log = cleaner.transform(df)
    return {"cleaned_df": cleaned_df, "cleaning_log": log}


def node_feature_engineering(state: PipelineState) -> PipelineState:
    plan = state["plan"]
    # entity_column is only threaded through for pooled forecasting, so lag/
    # rolling features stay scoped per entity instead of leaking across them.
    entity_column = plan.entity_column if plan.scope_strategy == ScopeStrategy.POOLED else None
    text_columns = [c for c in state.get("dataset_profile", DatasetProfile(row_count=0, column_count=0)).text_columns
                    if c in plan.feature_columns]
    engineered_df, log = feature_engineering.engineer_features(
        state["cleaned_df"],
        problem_type=plan.problem_type.value if plan.problem_type else "regression",
        target_column=plan.target_column,
        time_column=plan.time_column,
        entity_column=entity_column,
        text_columns=text_columns,
    )
    return {"engineered_df": engineered_df, "feature_log": log}


def node_split(state: PipelineState) -> PipelineState:
    plan = state["plan"]
    entity_column = plan.entity_column if plan.scope_strategy == ScopeStrategy.POOLED else None
    train_df, test_df, log = splitting.split_data(
        state["engineered_df"],
        problem_type=plan.problem_type.value if plan.problem_type else "regression",
        time_column=plan.time_column,
        entity_column=entity_column,
        target_column=plan.target_column,
    )
    return {"train_df": train_df, "test_df": test_df, "split_log": log}


def node_vectorize_text(state: PipelineState) -> PipelineState:
    plan = state["plan"]
    profile = state.get("dataset_profile")
    text_columns = [c for c in profile.text_columns if c in plan.feature_columns] if profile else []
    vectorizers, log = text_vectorization.fit_tfidf_vectorizers(
        state["train_df"], text_columns
    )
    return {
        "train_df": text_vectorization.apply_tfidf_vectorizers(
            state["train_df"], vectorizers, text_columns
        ),
        "test_df": text_vectorization.apply_tfidf_vectorizers(
            state["test_df"], vectorizers, text_columns
        ),
        "text_vectorizers": vectorizers,
        "text_vectorization_log": log,
    }


def node_train(state: PipelineState) -> PipelineState:
    """Runs candidates for this pooled/single_entity path - classical
    sklearn/xgboost/lightgbm models only. AutoGluon is NOT in this registry
    (see tools/model_registry.py's module docstring) and this path never
    calls it directly; AutoGluon only runs for the per_entity
    (tools/automl_training.train_models, called per entity) and hierarchical
    (train_hierarchical_timeseries) scope strategies below.

    For classification and regression, every registered model for the
    problem type is trained (not just the Planner's shortlist) - the
    Planner's candidate_model_families remains informative context for the
    plan narrative/report, but doesn't gate which models actually run. A
    model that fails (missing dependency, unsupported validation strategy,
    training/prediction/scoring error) is still reported per-model via
    failure_analysis.build_failure_summary()/explain_failure_summary() below
    rather than silently dropped. Other problem types keep the plan-driven
    resolve_candidates()/default_candidates() fallback so a run never trains
    zero models."""
    plan = state["plan"]
    problem_type = plan.problem_type or ProblemType.REGRESSION
    if problem_type in (ProblemType.CLASSIFICATION, ProblemType.REGRESSION):
        candidates = model_registry.get_registry_for_problem_type(problem_type)
    else:
        candidates, _unmatched = model_registry.resolve_candidates(problem_type, plan.candidate_model_families)
    if not candidates:
        candidates = model_registry.default_candidates(problem_type)

    train_df, test_df = state["train_df"], state["test_df"]
    # entity_column survives cleaning/feature-engineering/split for pooled
    # scope (needed for groupby-scoped lag/rolling and per-entity chronological
    # splits - see node_cleaning/node_feature_engineering/node_split), but it's
    # not a real feature. Drop it here, once, before it reaches any candidate -
    # classical models, AutoGluon, and explainability (tools/model_runner.py's
    # _compute_explainability) all reuse these same train_df/test_df objects,
    # so this single drop covers every downstream consumer without threading a
    # new parameter through tools/model_registry.py, tools/automl_training.py,
    # and tools/explainability.py's function signatures.
    if plan.scope_strategy == ScopeStrategy.POOLED and plan.entity_column:
        train_df = train_df.drop(columns=[plan.entity_column], errors="ignore")
        test_df = test_df.drop(columns=[plan.entity_column], errors="ignore")

    metrics, chart_data = model_runner.run_candidates(
        candidates,
        train_df,
        test_df,
        target_column=plan.target_column,
        time_column=plan.time_column,
        problem_type=problem_type,
        validation_strategy=plan.validation_strategy.strategy_type if plan.validation_strategy else None,
        validation_folds=plan.validation_strategy.folds if plan.validation_strategy else None,
        evaluation_metrics=plan.evaluation_metrics,
        tuned_model_params=state.get("tuned_model_params"),
    )
    metrics["text_feature_tokens"] = state.get("text_vectorization_log", {}).get("feature_tokens", {})
    metrics["tuning_results"] = state.get("tuning_results", {})
    failure_summary = failure_analysis.build_failure_summary(
        metrics,
        state.get("dataset_profile"),
        state.get("data_quality_report"),
        plan,
    )
    metrics["failure_summary"] = failure_summary
    metrics["failure_explanation"] = failure_analysis.explain_failure_summary(failure_summary)
    return {"metrics": metrics, "chart_data": chart_data}


def node_tune_models(state: PipelineState) -> PipelineState:
    """Tune declared classical model spaces on train_df only.

    The final test frame is intentionally not passed to this node. Forecasting,
    hierarchical, and AutoGluon per-entity paths do not use this registry tuner.
    """
    plan = state["plan"]
    problem_type = plan.problem_type or ProblemType.REGRESSION
    if problem_type in (ProblemType.CLASSIFICATION, ProblemType.REGRESSION):
        candidates = model_registry.get_registry_for_problem_type(problem_type)
    else:
        candidates, _unmatched = model_registry.resolve_candidates(
            problem_type, plan.candidate_model_families
        )
    if not candidates:
        candidates = model_registry.default_candidates(problem_type)
    if problem_type not in (ProblemType.CLASSIFICATION, ProblemType.REGRESSION):
        return {
            "tuned_model_params": {},
            "tuning_results": {"status": "skipped", "reason": "unsupported problem type"},
        }

    try:
        tuned_params, results = hyperparameter_tuning.tune_models(
            candidates,
            state["train_df"],
            target_column=plan.target_column,
            time_column=plan.time_column,
            problem_type=problem_type,
            validation_strategy=(
                plan.validation_strategy.strategy_type if plan.validation_strategy else None
            ),
            metric=plan.evaluation_metrics[0] if plan.evaluation_metrics else None,
            cancel_check=lambda: bool(
                state.get("cancel_event") and state["cancel_event"].is_set()
            ),
        )
    except hyperparameter_tuning.TuningCancelled as exc:
        raise PipelineCancelled(str(exc)) from exc
    return {"tuned_model_params": tuned_params, "tuning_results": results}


def node_per_entity_pipeline(state: PipelineState) -> PipelineState:
    """per_entity scope: run clean -> engineer -> split -> train independently
    for each entity value, entirely in local process/tool code (no LLM call
    per entity). Produces a combined metrics dict with a per-entity breakdown
    plus an averaged 'models' view the Evaluator/Reporter can reason about.
    """
    plan = state["plan"]
    # Selection uses one GLOBAL exclusion set (quality_report was computed once,
    # before any entity split exists) applied identically to every entity - a
    # column constant only within one entity but not others is kept/dropped the
    # same way everywhere. entity_column itself is always kept here (it's
    # structural - each entity's subset is carved out from it below) even
    # though it's never in plan.feature_columns.
    df, feature_selection_log = feature_selection.apply_feature_selection(
        state["df"], plan, state["data_quality_report"]
    )
    problem_type = plan.problem_type.value if plan.problem_type else "regression"
    aggressive = state.get("retry_count", 0) > 0

    entity_values = sorted(df[plan.entity_column].dropna().unique().tolist(), key=str)
    truncated = len(entity_values) > MAX_PER_ENTITY
    if truncated:
        entity_values = entity_values[:MAX_PER_ENTITY]

    aggressive = aggressive or "cap_outliers" in plan.preprocessing_requirements

    per_entity_metrics: dict[str, dict] = {}
    text_vectorizers: dict = {}
    text_vectorization_log: dict = {}
    skipped: list[dict] = []

    for value in entity_values:
        subset = df[df[plan.entity_column] == value].drop(columns=[plan.entity_column])
        if len(subset) < MIN_ROWS_PER_ENTITY:
            skipped.append({"entity": str(value), "reason": f"only {len(subset)} rows (< {MIN_ROWS_PER_ENTITY})"})
            continue
        try:
            text_columns = [c for c in state.get("dataset_profile", DatasetProfile(row_count=0, column_count=0)).text_columns
                            if c in plan.feature_columns]
            # Same leakage-safe fit/transform split as node_cleaning: fit
            # imputation/outlier stats on this entity's TRAIN rows only
            # (identified via the same deterministic split re-run below on
            # engineered data), then apply those stats to the whole subset.
            train_for_fit, _test_for_fit, _ = splitting.split_data(
                subset, problem_type=problem_type, time_column=plan.time_column
            )
            entity_cleaner = cleaning.TabularCleaner(aggressive_outlier_handling=aggressive)
            entity_cleaner.fit(
                train_for_fit,
                target_column=plan.target_column,
                time_column=plan.time_column,
                text_columns=text_columns,
            )
            cleaned, _ = entity_cleaner.transform(subset)
            engineered, _ = feature_engineering.engineer_features(
                cleaned,
                problem_type=problem_type,
                target_column=plan.target_column,
                time_column=plan.time_column,
                text_columns=text_columns,
            )
            train_df, test_df, _ = splitting.split_data(
                engineered, problem_type=problem_type, time_column=plan.time_column
            )
            if len(train_df) < 5 or len(test_df) < 1:
                skipped.append({"entity": str(value), "reason": "not enough rows after split"})
                continue
            entity_vectorizers, entity_log = text_vectorization.fit_tfidf_vectorizers(
                train_df, text_columns
            )
            train_df = text_vectorization.apply_tfidf_vectorizers(train_df, entity_vectorizers, text_columns)
            test_df = text_vectorization.apply_tfidf_vectorizers(test_df, entity_vectorizers, text_columns)
            text_vectorizers[value] = entity_vectorizers
            text_vectorization_log[value] = entity_log
            per_entity_metrics[str(value)] = automl_training.train_models(
                train_df,
                test_df,
                target_column=plan.target_column,
                problem_type=problem_type,
                time_column=plan.time_column,
            )
        except Exception as exc:  # noqa: BLE001 - one bad entity shouldn't kill the whole run
            skipped.append({"entity": str(value), "reason": str(exc)})

    metrics = {
        "eval_metric": next((m["eval_metric"] for m in per_entity_metrics.values()), None),
        "models": _aggregate_per_entity_models(per_entity_metrics),
        "per_entity": per_entity_metrics,
        "entities_trained": len(per_entity_metrics),
        "entities_skipped": skipped,
        "entities_truncated": truncated,
    }
    return {
        "metrics": metrics,
        "feature_selection_log": feature_selection_log,
        "text_vectorizers": text_vectorizers,
        "text_vectorization_log": text_vectorization_log,
    }


def _aggregate_per_entity_models(per_entity_metrics: dict[str, dict]) -> dict:
    """Average each model family's test score across entities that trained it,
    so the Evaluator has a single 'models' view alongside the full breakdown."""
    sums: dict[str, dict] = defaultdict(lambda: {"score_test_sum": 0.0, "fit_time_sum": 0.0, "count": 0})
    for entity_metrics in per_entity_metrics.values():
        for model_name, scores in entity_metrics.get("models", {}).items():
            bucket = sums[model_name]
            bucket["score_test_sum"] += scores.get("score_test") or 0.0
            bucket["fit_time_sum"] += scores.get("fit_time_s") or 0.0
            bucket["count"] += 1

    aggregated = {}
    for model_name, bucket in sums.items():
        if bucket["count"] == 0:
            continue
        aggregated[model_name] = {
            "score_test": round(bucket["score_test_sum"] / bucket["count"], 4),
            "score_val": None,
            "fit_time_s": round(bucket["fit_time_sum"] / bucket["count"], 3),
        }
    return aggregated


def node_hierarchical_train(state: PipelineState) -> PipelineState:
    """hierarchical scope: joint forecasting across all entities via
    AutoGluon TimeSeriesPredictor (item_id = entity_column). Bypasses the
    standard clean/feature_engineer/split tools - TimeSeriesPredictor handles
    missing values and its own feature engineering internally."""
    plan = state["plan"]
    metrics = automl_training.train_hierarchical_timeseries(
        state["df"],
        entity_column=plan.entity_column,
        target_column=plan.target_column,
        time_column=plan.time_column,
    )
    return {"metrics": metrics}


def node_evaluate(state: PipelineState) -> PipelineState:
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", MAX_RETRIES_DEFAULT)
    decision = evaluator.evaluate_results(state["metrics"], state["plan"], retry_count, max_retries)
    next_retry_count = retry_count + 1 if decision.decision == "retry" else retry_count
    return {"decision": decision, "retry_count": next_retry_count}


def node_recommend(state: PipelineState) -> PipelineState:
    """The Recommendation Agent (Phase 5): explains the deterministic
    evaluation engine's winner for a business audience - reason,
    performance summary, comparison to alternatives, business
    interpretation, limitations, confidence. Cannot change the ranking; see
    agents/recommender.py's validate_recommendation for the enforcement.
    """
    plan = state["plan"]
    # Reuses agents/evaluator.py's fallback (per_entity/hierarchical scopes
    # bypass tools/model_runner.py and only have the old flat score_test
    # shape) rather than duplicating the same logic here.
    comparison = evaluator._get_or_build_comparison(state["metrics"], plan)
    recommendation = recommender.generate_recommendation(
        dataset_profile=state.get("dataset_profile"),
        problem_definition=state.get("problem_definition"),
        plan=plan,
        model_comparison=comparison,
        decision=state["decision"],
        validation_strategy=plan.validation_strategy,
    )
    return {"recommendation": recommendation}


def node_report(state: PipelineState) -> PipelineState:
    """Renders the ML analysis report (Phase 6): every number comes from
    deterministic pipeline objects (dataset_profile, data_quality_report,
    model_comparison, chart_data, ...); the LLM (ReportContent) supplies
    only the executive summary and approach narrative prose - see
    agents/reporter.py.
    """
    plan = state["plan"]
    metrics = state["metrics"]
    problem_type_value = plan.problem_type.value if plan.problem_type else None
    charts = report_charts.build_report_charts(
        problem_type_value, metrics.get("model_comparison"), state.get("chart_data")
    )

    # Phase 9: name the report after this run's id rather than a
    # datetime-only timestamp. Two concurrent runs (the bounded worker pool
    # now allows more than one at a time) finishing within the same second
    # would otherwise collide and overwrite each other's report file.
    run_id = state.get("run_id")
    output_path = str(reporter.OUTPUT_DIR / f"report_{run_id}.html") if run_id else None

    path = reporter.generate_report(
        business_description=state["business_description"],
        plan=plan,
        metrics=metrics,
        decision=state["decision"],
        dataset_profile=state.get("dataset_profile"),
        data_quality_report=state.get("data_quality_report"),
        problem_definition=state.get("problem_definition"),
        recommendation=state.get("recommendation"),
        cleaning_log=state.get("cleaning_log"),
        feature_log=state.get("feature_log"),
        feature_selection_log=state.get("feature_selection_log"),
        split_log=state.get("split_log"),
        charts=charts,
        output_path=output_path,
    )
    return {"report_path": path}


def route_after_validate_plan(state: PipelineState) -> str:
    if state.get("needs_clarification"):
        return "end_clarification"
    scope = state["plan"].scope_strategy
    if scope == ScopeStrategy.SINGLE_ENTITY:
        return "filter_entity"
    if scope == ScopeStrategy.PER_ENTITY:
        return "per_entity"
    if scope == ScopeStrategy.HIERARCHICAL:
        return "hierarchical"
    return "eda"  # pooled, or no grouping structure


def route_after_evaluate(state: PipelineState) -> str:
    if state["decision"].decision != "retry":
        return "recommend"
    scope = state["plan"].scope_strategy
    if scope == ScopeStrategy.PER_ENTITY:
        return "retry_per_entity"
    if scope == ScopeStrategy.HIERARCHICAL:
        return "retry_hierarchical"
    return "retry_standard"


def _cancellable(node_fn):
    """Wraps a node function so every node checks state["cancel_event"]
    before running (Phase 9). Applied uniformly in build_graph() rather than
    editing every node body - a run with no cancel_event (CLI usage, tests)
    pays for one None check and behaves exactly as before.
    """

    @functools.wraps(node_fn)
    def wrapper(state: PipelineState) -> PipelineState:
        event = state.get("cancel_event")
        if event is not None and event.is_set():
            raise PipelineCancelled("Run was cancelled.")
        progress_callback = state.get("progress_callback")
        if progress_callback is not None:
            node_name = node_fn.__name__
            if node_name.startswith("node_"):
                node_name = node_name[5:]
            progress_callback(node_name, {"progress_status": "started"})
        return node_fn(state)

    return wrapper


def build_graph():
    graph = StateGraph(PipelineState)

    graph.add_node("ingest", _cancellable(node_ingest))
    graph.add_node("profile_data", _cancellable(node_profile_data))
    graph.add_node("quality_analysis", _cancellable(node_quality_analysis))
    graph.add_node("detect_problem", _cancellable(node_detect_problem))
    graph.add_node("planner_agent", _cancellable(node_planner_agent))
    graph.add_node("validate_plan", _cancellable(node_validate_plan))
    graph.add_node("filter_entity", _cancellable(node_filter_entity))
    graph.add_node("feature_selection", _cancellable(node_feature_selection))
    graph.add_node("eda", _cancellable(node_eda))
    graph.add_node("clean", _cancellable(node_cleaning))
    graph.add_node("feature_engineer", _cancellable(node_feature_engineering))
    graph.add_node("split", _cancellable(node_split))
    graph.add_node("vectorize_text", _cancellable(node_vectorize_text))
    graph.add_node("tune_models", _cancellable(node_tune_models))
    graph.add_node("train", _cancellable(node_train))
    graph.add_node("per_entity_pipeline", _cancellable(node_per_entity_pipeline))
    graph.add_node("hierarchical_train", _cancellable(node_hierarchical_train))
    graph.add_node("evaluate", _cancellable(node_evaluate))
    graph.add_node("recommend", _cancellable(node_recommend))
    graph.add_node("report", _cancellable(node_report))

    graph.set_entry_point("ingest")
    graph.add_edge("ingest", "profile_data")
    graph.add_edge("profile_data", "quality_analysis")
    graph.add_edge("quality_analysis", "detect_problem")
    graph.add_edge("detect_problem", "planner_agent")
    graph.add_edge("planner_agent", "validate_plan")
    graph.add_conditional_edges(
        "validate_plan",
        route_after_validate_plan,
        {
            "filter_entity": "filter_entity",
            "per_entity": "per_entity_pipeline",
            "hierarchical": "hierarchical_train",
            "eda": "feature_selection",
            "end_clarification": END,
        },
    )
    graph.add_edge("filter_entity", "feature_selection")
    graph.add_edge("feature_selection", "eda")
    graph.add_edge("eda", "clean")
    graph.add_edge("clean", "feature_engineer")
    graph.add_edge("feature_engineer", "split")
    graph.add_edge("split", "vectorize_text")
    graph.add_edge("vectorize_text", "tune_models")
    graph.add_edge("tune_models", "train")
    graph.add_edge("train", "evaluate")
    graph.add_edge("per_entity_pipeline", "evaluate")
    graph.add_edge("hierarchical_train", "evaluate")
    graph.add_conditional_edges(
        "evaluate",
        route_after_evaluate,
        {
            "retry_standard": "clean",
            "retry_per_entity": "per_entity_pipeline",
            "retry_hierarchical": "hierarchical_train",
            "recommend": "recommend",
        },
    )
    graph.add_edge("recommend", "report")
    graph.add_edge("report", END)

    return graph.compile()


def run_pipeline(
    file_path: str,
    business_description: str,
    max_retries: int = MAX_RETRIES_DEFAULT,
    run_id: Optional[str] = None,
    cancel_event: Optional[threading.Event] = None,
    progress_callback: Optional[Callable[[str, dict], None]] = None,
) -> PipelineState:
    """Runs the graph, then records an ExperimentRecord (Phase 8) for the run
    - reproducibility/audit trail, entirely separate from the pipeline's own
    return value. Recording happens whether the run completes, needs
    clarification, is cancelled, or raises, so every outcome is tracked; it
    can never itself fail or alter the pipeline's result (see
    experiment_tracking.py).

    run_id (Phase 9): callers that already track this run under an id of
    their own (api/run_store.py's job/DB row id) should pass it here, so the
    same id is used for the job record, the ExperimentRecord, and the
    generated report's filename - previously the API layer's run id and this
    function's own internally-generated experiment run id were two different
    values, and the report filename didn't include either (see node_report).
    Auto-generated when omitted (CLI usage, tests).

    cancel_event (Phase 9): an optional threading.Event a caller can set to
    cooperatively cancel a running job - checked at the start of every node
    (see _cancellable()/PipelineCancelled above). Unset for CLI usage.

    progress_callback: an optional `fn(node_name, updates)` invoked as each
    node finishes (e.g. "clean", "train", "evaluate") - lets a caller (see
    api/run_store.py) persist which step a long-running job is currently on,
    so a client polling GET /api/runs/{run_id} mid-run sees more than a bare
    "running" status. `updates` is exactly the PipelineState dict that node
    returned (e.g. node_validate_plan's `{"plan": ..., "needs_clarification":
    ...}`) - not the full accumulated state - so a caller can react to a
    specific field becoming available (e.g. persisting the ExperimentPlan as
    soon as it's decided, well before training finishes) without re-deriving
    it from the whole state. Uses app.stream(..., stream_mode="updates")
    instead of app.invoke() to get a callback point between nodes; the
    accumulated per-node updates are merged into the same PipelineState shape
    invoke() would have returned, so callers otherwise see no difference.
    """
    experiment_tracking.set_global_seeds()

    run_id = run_id or uuid.uuid4().hex
    started_at_iso = experiment_tracking.now_iso()
    started_perf = time.perf_counter()
    logger.info("pipeline_started", extra={"run_id": run_id, "file_path": file_path})

    app = build_graph()
    initial_state: PipelineState = {
        "run_id": run_id,
        "cancel_event": cancel_event,
        "file_path": file_path,
        "business_description": business_description,
        "max_retries": max_retries,
        "progress_callback": progress_callback,
        "retry_count": 0,
    }

    result: PipelineState = dict(initial_state)  # type: ignore[assignment]
    exception: Optional[BaseException] = None
    status_override: Optional[str] = None
    try:
        for step in app.stream(initial_state, config={"recursion_limit": 50}, stream_mode="updates"):
            for node_name, updates in step.items():
                result.update(updates)
                if progress_callback is not None:
                    progress_callback(node_name, updates)
    except PipelineCancelled as exc:
        exception = exc
        status_override = "cancelled"
        logger.info("pipeline_cancelled", extra={"run_id": run_id})
        raise
    except Exception as exc:  # noqa: BLE001 - record the failure, then re-raise unchanged
        exception = exc
        logger.error("pipeline_failed", extra={"run_id": run_id, "error": str(exc)})
        raise
    else:
        logger.info(
            "pipeline_finished",
            extra={
                "run_id": run_id,
                "needs_clarification": bool(result.get("needs_clarification")),
                "runtime_seconds": round(time.perf_counter() - started_perf, 3),
            },
        )
    finally:
        record = experiment_tracking.build_record(
            run_id=run_id,
            started_at_iso=started_at_iso,
            started_perf=started_perf,
            file_path=file_path,
            business_description=business_description,
            max_retries=max_retries,
            state=result,
            exception=exception,
            status_override=status_override,
        )
        experiment_tracking.get_default_store().save(record)
        result["experiment_record"] = record

    return result
