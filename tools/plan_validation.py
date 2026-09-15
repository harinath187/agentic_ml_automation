"""Deterministic validation/repair layer for the Planner Agent's ExperimentPlan.

Runs after the LLM call and before the pipeline commits to a scope/strategy.
Repairs obviously-fixable issues (hallucinated column names, missing
defaults) deterministically using the Data Intelligence report; anything it
cannot safely repair is surfaced via needs_clarification instead of raising -
the same mechanism the Planner itself already uses, so the pipeline always
ends in a normal END/report rather than crashing on a bad LLM output.
"""
from __future__ import annotations

from agents.schemas import (
    DataQualityReport,
    DatasetProfile,
    ExperimentPlan,
    ProblemType,
    ScopeStrategy,
    TargetAnalysis,
    ValidationStrategy,
    ValidationStrategyType,
)

DEFAULT_CANDIDATE_MODEL_FAMILIES = {
    # Canonical tools/model_registry.py names, so these resolve directly
    # without relying on resolve_candidates()'s alias/fuzzy matching.
    ProblemType.CLASSIFICATION: ["baseline", "logistic_regression", "random_forest", "lightgbm", "autogluon_tabular"],
    ProblemType.REGRESSION: ["baseline", "linear_regression", "random_forest", "lightgbm", "autogluon_tabular"],
    ProblemType.FORECASTING: ["naive", "seasonal_naive", "autogluon_timeseries"],
    ProblemType.CLUSTERING: ["KMeans"],
}

DEFAULT_EVALUATION_METRICS = {
    ProblemType.CLASSIFICATION: ["accuracy", "roc_auc", "f1"],
    ProblemType.REGRESSION: ["rmse", "mae", "r2"],
    ProblemType.FORECASTING: ["mape", "rmse"],
    ProblemType.CLUSTERING: ["silhouette_score"],
}

DEFAULT_VALIDATION_STRATEGY = {
    ProblemType.CLASSIFICATION: ValidationStrategyType.STRATIFIED_K_FOLD,
    ProblemType.REGRESSION: ValidationStrategyType.K_FOLD,
    ProblemType.FORECASTING: ValidationStrategyType.TIME_SERIES_SPLIT,
}

DEFAULT_FORECAST_HORIZON = 7


def validate_plan(
    plan: ExperimentPlan,
    dataset_profile: DatasetProfile,
    target_analysis: TargetAnalysis,
    quality_report: DataQualityReport,
) -> ExperimentPlan:
    """Returns a repaired copy of `plan`. Never raises: an unrepairable issue
    sets needs_clarification=True with a clarification_question instead, so
    the graph's existing end_clarification route handles it.
    """
    if plan.needs_clarification:
        return plan  # The Planner already asked; nothing left to validate.

    plan = plan.model_copy(deep=True)
    notes: list[str] = []
    known_columns = set(dataset_profile.column_names)

    if plan.problem_type is None:
        return _reject(plan, "Which kind of problem is this - classification, regression, or forecasting?")

    if not plan.target_column or plan.target_column not in known_columns:
        if target_analysis.recommended_target:
            notes.append(
                f"target_column {plan.target_column!r} was missing/unknown; repaired to the "
                f"deterministic recommendation {target_analysis.recommended_target!r}."
            )
            plan.target_column = target_analysis.recommended_target
        else:
            return _reject(plan, "Which column should be predicted (the target)?")

    if plan.problem_type == ProblemType.FORECASTING and not plan.time_column:
        if dataset_profile.datetime_columns:
            plan.time_column = dataset_profile.datetime_columns[0]
            notes.append(f"time_column was missing; repaired to {plan.time_column!r}.")
        else:
            return _reject(plan, "Forecasting needs a datetime column - which column is it?")

    if plan.entity_column and plan.entity_column not in known_columns:
        notes.append(f"entity_column {plan.entity_column!r} does not exist in the dataset; cleared.")
        plan.entity_column = None
        plan.entity_filter_value = None
        plan.scope_strategy = None

    if plan.scope_strategy == ScopeStrategy.SINGLE_ENTITY and (
        not plan.entity_column or plan.entity_filter_value is None
    ):
        return _reject(plan, "Which single entity value should this be scoped to?")

    plan.feature_columns = [
        c for c in plan.feature_columns if c in known_columns and c != plan.target_column
    ]
    if not plan.feature_columns:
        excluded = {plan.target_column} | set(quality_report.suspicious_columns) | set(
            quality_report.possible_leakage_columns
        )
        plan.feature_columns = [c for c in dataset_profile.column_names if c not in excluded]
        notes.append("feature_columns was empty/invalid; repaired to all non-target, non-suspicious columns.")

    if not plan.candidate_model_families:
        plan.candidate_model_families = DEFAULT_CANDIDATE_MODEL_FAMILIES.get(plan.problem_type, [])
        notes.append("candidate_model_families was empty; repaired with problem-type defaults.")

    if not plan.evaluation_metrics:
        plan.evaluation_metrics = DEFAULT_EVALUATION_METRICS.get(plan.problem_type, [])
        notes.append("evaluation_metrics was empty; repaired with problem-type defaults.")

    plan.validation_strategy = _repair_validation_strategy(plan, notes)

    if plan.problem_type == ProblemType.FORECASTING and not (plan.forecast_horizon and plan.forecast_horizon > 0):
        plan.forecast_horizon = DEFAULT_FORECAST_HORIZON
        notes.append(f"forecast_horizon was missing/invalid; repaired to default of {DEFAULT_FORECAST_HORIZON}.")

    plan.validation_notes = notes
    return plan


def _repair_validation_strategy(plan: ExperimentPlan, notes: list[str]) -> ValidationStrategy | None:
    expected = DEFAULT_VALIDATION_STRATEGY.get(plan.problem_type)

    if plan.validation_strategy is None:
        if expected is None:
            return None
        notes.append(f"validation_strategy was missing; repaired to {expected.value!r}.")
        return ValidationStrategy(strategy_type=expected, notes="Repaired: no validation_strategy was provided.")

    is_forecasting = plan.problem_type == ProblemType.FORECASTING
    strategy_is_time_series = plan.validation_strategy.strategy_type == ValidationStrategyType.TIME_SERIES_SPLIT

    if is_forecasting and not strategy_is_time_series:
        notes.append(
            f"validation_strategy {plan.validation_strategy.strategy_type.value!r} is invalid for "
            "forecasting; repaired to 'time_series_split'."
        )
        return ValidationStrategy(
            strategy_type=ValidationStrategyType.TIME_SERIES_SPLIT, notes="Repaired: forecasting requires a time-based split."
        )

    if not is_forecasting and strategy_is_time_series:
        repaired = expected or ValidationStrategyType.K_FOLD
        notes.append(f"validation_strategy 'time_series_split' is invalid outside forecasting; repaired to {repaired.value!r}.")
        return ValidationStrategy(strategy_type=repaired, notes="Repaired: not a forecasting problem.")

    return plan.validation_strategy


def _reject(plan: ExperimentPlan, question: str) -> ExperimentPlan:
    """'Reject' an unrepairable plan by routing it through the existing
    clarification flow instead of raising - the pipeline still ends cleanly."""
    plan.needs_clarification = True
    plan.clarification_question = plan.clarification_question or question
    return plan
