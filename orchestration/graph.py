"""LangGraph orchestration wiring together all agents and tools.

Standard flow (scope_strategy is pooled or null):
    Ingest -> Profile Data -> Quality Analysis -> Plan -> EDA -> Clean ->
    Feature Engineer -> Split -> Train -> Evaluate ->
    (retry loop back to Clean, or) -> Report -> END.

Profile Data / Quality Analysis are the deterministic Data Intelligence
layer (tools/profiling.py) - pandas-only, no LLM call. They run before the
Planner so it reasons over structured facts (DatasetProfile, TargetAnalysis,
DataQualityReport) rather than inferring everything itself from a thin
schema summary.

Entity-aware branches, chosen by the Planner via PipelinePlan.scope_strategy:
    single_entity  -> Plan -> filter_entity (subset to one entity, drop the
                       entity column) -> re-joins the standard flow at EDA.
    per_entity     -> Plan -> per_entity_pipeline (loops clean/engineer/
                       split/train once per entity value, in-process, never
                       touching an LLM per entity) -> Evaluate.
    hierarchical   -> Plan -> hierarchical_train (AutoGluon TimeSeriesPredictor
                       across all entities via item_id grouping) -> Evaluate.

The DataFrame lives only in this process's local state (never serialized to
an LLM); only `schema_summary` / `eda_summary` / `metrics` are ever passed to
agents.planner / agents.evaluator / agents.reporter.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Optional, TypedDict

from langgraph.graph import END, StateGraph

from agents import evaluator, planner, reporter
from agents.schemas import (
    DataQualityReport,
    DatasetProfile,
    EvaluatorDecision,
    PipelinePlan,
    ScopeStrategy,
    TargetAnalysis,
)
from data_ingestion.loader import extract_schema, load_dataset
from tools import automl_training, cleaning, eda, feature_engineering, profiling, splitting

MAX_RETRIES_DEFAULT = 2

# per_entity scope safeguards: cap total entities trained (each entity is a
# full AutoML fit) and require a minimum row count per entity to bother.
MAX_PER_ENTITY = 25
MIN_ROWS_PER_ENTITY = 10


class PipelineState(TypedDict, total=False):
    file_path: str
    sensitive_columns: list[str]
    business_description: str
    max_retries: int
    time_limit_s: int

    df: Any
    schema_summary: dict
    dataset_profile: DatasetProfile
    data_quality_report: DataQualityReport
    target_analysis: TargetAnalysis
    plan: PipelinePlan
    needs_clarification: bool

    eda_summary: dict
    cleaned_df: Any
    cleaning_log: dict
    engineered_df: Any
    feature_log: dict
    train_df: Any
    test_df: Any
    split_log: dict

    metrics: dict
    retry_count: int
    decision: EvaluatorDecision

    report_path: str
    error: Optional[str]


def node_ingest(state: PipelineState) -> PipelineState:
    df = load_dataset(state["file_path"])
    schema_summary = extract_schema(df, state.get("sensitive_columns"))
    return {"df": df, "schema_summary": schema_summary}


def node_profile_data(state: PipelineState) -> PipelineState:
    """Deterministic Data Intelligence, step 1: dataset-level profiling."""
    dataset_profile = profiling.profile_dataset(state["df"], state.get("sensitive_columns"))
    return {"dataset_profile": dataset_profile}


def node_quality_analysis(state: PipelineState) -> PipelineState:
    """Deterministic Data Intelligence, step 2: target signals + quality report.

    Both are pandas-only (no LLM call) and depend only on node_profile_data's
    output, so they run once per pipeline invocation regardless of scope
    strategy or retries.
    """
    dataset_profile = state["dataset_profile"]
    sensitive_columns = state.get("sensitive_columns")
    target_analysis = profiling.analyze_target(state["df"], dataset_profile, sensitive_columns)
    data_quality_report = profiling.analyze_data_quality(state["df"], dataset_profile, sensitive_columns)
    return {"target_analysis": target_analysis, "data_quality_report": data_quality_report}


def node_plan(state: PipelineState) -> PipelineState:
    plan = planner.build_plan(
        state["business_description"],
        state["schema_summary"],
        dataset_profile=state.get("dataset_profile"),
        quality_report=state.get("data_quality_report"),
        target_analysis=state.get("target_analysis"),
    )
    return {"plan": plan, "needs_clarification": plan.needs_clarification}


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


def node_eda(state: PipelineState) -> PipelineState:
    summary = eda.run_eda(state["df"], state.get("sensitive_columns"))
    return {"eda_summary": summary}


def node_cleaning(state: PipelineState) -> PipelineState:
    plan = state["plan"]
    aggressive = state.get("retry_count", 0) > 0
    cleaned_df, log = cleaning.clean_data(
        state["df"],
        target_column=plan.target_column,
        time_column=plan.time_column,
        aggressive_outlier_handling=aggressive,
    )
    return {"cleaned_df": cleaned_df, "cleaning_log": log}


def node_feature_engineering(state: PipelineState) -> PipelineState:
    plan = state["plan"]
    # entity_column is only threaded through for pooled forecasting, so lag/
    # rolling features stay scoped per entity instead of leaking across them.
    entity_column = plan.entity_column if plan.scope_strategy == ScopeStrategy.POOLED else None
    engineered_df, log = feature_engineering.engineer_features(
        state["cleaned_df"],
        problem_type=plan.problem_type.value if plan.problem_type else "regression",
        target_column=plan.target_column,
        time_column=plan.time_column,
        entity_column=entity_column,
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
    )
    return {"train_df": train_df, "test_df": test_df, "split_log": log}


def node_train(state: PipelineState) -> PipelineState:
    plan = state["plan"]
    time_limit = state.get("time_limit_s", 60)
    metrics = automl_training.train_models(
        state["train_df"],
        state["test_df"],
        target_column=plan.target_column,
        problem_type=plan.problem_type.value if plan.problem_type else "regression",
        time_column=plan.time_column,
        time_limit=time_limit,
    )
    return {"metrics": metrics}


def node_per_entity_pipeline(state: PipelineState) -> PipelineState:
    """per_entity scope: run clean -> engineer -> split -> train independently
    for each entity value, entirely in local process/tool code (no LLM call
    per entity). Produces a combined metrics dict with a per-entity breakdown
    plus an averaged 'models' view the Evaluator/Reporter can reason about.
    """
    plan = state["plan"]
    df = state["df"]
    problem_type = plan.problem_type.value if plan.problem_type else "regression"
    aggressive = state.get("retry_count", 0) > 0
    total_time_budget = state.get("time_limit_s", 60)

    entity_values = sorted(df[plan.entity_column].dropna().unique().tolist(), key=str)
    truncated = len(entity_values) > MAX_PER_ENTITY
    if truncated:
        entity_values = entity_values[:MAX_PER_ENTITY]

    per_entity_time_limit = max(10, total_time_budget // max(1, len(entity_values)))

    per_entity_metrics: dict[str, dict] = {}
    skipped: list[dict] = []

    for value in entity_values:
        subset = df[df[plan.entity_column] == value].drop(columns=[plan.entity_column])
        if len(subset) < MIN_ROWS_PER_ENTITY:
            skipped.append({"entity": str(value), "reason": f"only {len(subset)} rows (< {MIN_ROWS_PER_ENTITY})"})
            continue
        try:
            cleaned, _ = cleaning.clean_data(
                subset,
                target_column=plan.target_column,
                time_column=plan.time_column,
                aggressive_outlier_handling=aggressive,
            )
            engineered, _ = feature_engineering.engineer_features(
                cleaned,
                problem_type=problem_type,
                target_column=plan.target_column,
                time_column=plan.time_column,
            )
            train_df, test_df, _ = splitting.split_data(
                engineered, problem_type=problem_type, time_column=plan.time_column
            )
            if len(train_df) < 5 or len(test_df) < 1:
                skipped.append({"entity": str(value), "reason": "not enough rows after split"})
                continue
            per_entity_metrics[str(value)] = automl_training.train_models(
                train_df,
                test_df,
                target_column=plan.target_column,
                problem_type=problem_type,
                time_column=plan.time_column,
                time_limit=per_entity_time_limit,
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
    return {"metrics": metrics}


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
        time_limit=state.get("time_limit_s", 60),
    )
    return {"metrics": metrics}


def node_evaluate(state: PipelineState) -> PipelineState:
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", MAX_RETRIES_DEFAULT)
    decision = evaluator.evaluate_results(state["metrics"], state["plan"], retry_count, max_retries)
    next_retry_count = retry_count + 1 if decision.decision == "retry" else retry_count
    return {"decision": decision, "retry_count": next_retry_count}


def node_report(state: PipelineState) -> PipelineState:
    path = reporter.generate_report(
        state["business_description"], state["plan"], state["metrics"], state["decision"]
    )
    return {"report_path": path}


def route_after_plan(state: PipelineState) -> str:
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
        return "report"
    scope = state["plan"].scope_strategy
    if scope == ScopeStrategy.PER_ENTITY:
        return "retry_per_entity"
    if scope == ScopeStrategy.HIERARCHICAL:
        return "retry_hierarchical"
    return "retry_standard"


def build_graph():
    graph = StateGraph(PipelineState)

    graph.add_node("ingest", node_ingest)
    graph.add_node("profile_data", node_profile_data)
    graph.add_node("quality_analysis", node_quality_analysis)
    graph.add_node("planner", node_plan)
    graph.add_node("filter_entity", node_filter_entity)
    graph.add_node("eda", node_eda)
    graph.add_node("clean", node_cleaning)
    graph.add_node("feature_engineer", node_feature_engineering)
    graph.add_node("split", node_split)
    graph.add_node("train", node_train)
    graph.add_node("per_entity_pipeline", node_per_entity_pipeline)
    graph.add_node("hierarchical_train", node_hierarchical_train)
    graph.add_node("evaluate", node_evaluate)
    graph.add_node("report", node_report)

    graph.set_entry_point("ingest")
    graph.add_edge("ingest", "profile_data")
    graph.add_edge("profile_data", "quality_analysis")
    graph.add_edge("quality_analysis", "planner")
    graph.add_conditional_edges(
        "planner",
        route_after_plan,
        {
            "filter_entity": "filter_entity",
            "per_entity": "per_entity_pipeline",
            "hierarchical": "hierarchical_train",
            "eda": "eda",
            "end_clarification": END,
        },
    )
    graph.add_edge("filter_entity", "eda")
    graph.add_edge("eda", "clean")
    graph.add_edge("clean", "feature_engineer")
    graph.add_edge("feature_engineer", "split")
    graph.add_edge("split", "train")
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
            "report": "report",
        },
    )
    graph.add_edge("report", END)

    return graph.compile()


def run_pipeline(
    file_path: str,
    business_description: str,
    sensitive_columns: Optional[list[str]] = None,
    max_retries: int = MAX_RETRIES_DEFAULT,
    time_limit_s: int = 60,
) -> PipelineState:
    app = build_graph()
    initial_state: PipelineState = {
        "file_path": file_path,
        "business_description": business_description,
        "sensitive_columns": sensitive_columns or [],
        "max_retries": max_retries,
        "time_limit_s": time_limit_s,
        "retry_count": 0,
    }
    return app.invoke(initial_state, config={"recursion_limit": 50})
