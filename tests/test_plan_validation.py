"""Tests for deterministic ExperimentPlan validation/repair
(tools/plan_validation.py). No LLM involved - plans are constructed by hand
to simulate both well-formed and broken Planner output.
"""
from agents.schemas import (
    ExperimentPlan,
    ProblemType,
    ScopeStrategy,
    ValidationStrategy,
    ValidationStrategyType,
)
from tools.plan_validation import validate_plan
from tools.profiling import analyze_data_quality, analyze_target, profile_dataset


def _intelligence(df):
    profile = profile_dataset(df)
    target_analysis = analyze_target(df, profile)
    quality_report = analyze_data_quality(df, profile)
    return profile, target_analysis, quality_report


# --- valid planner output: passes through with defaults filled in ----------


def test_valid_plan_keeps_target_and_gets_defaults_filled(regression_df):
    profile, target_analysis, quality_report = _intelligence(regression_df)
    plan = ExperimentPlan(problem_type=ProblemType.REGRESSION, target_column="price")

    result = validate_plan(plan, profile, target_analysis, quality_report)

    assert result.needs_clarification is False
    assert result.target_column == "price"
    assert result.candidate_model_families  # repaired with defaults
    assert result.evaluation_metrics  # repaired with defaults
    assert result.validation_strategy is not None
    assert result.validation_strategy.strategy_type == ValidationStrategyType.K_FOLD


def test_valid_plan_with_everything_already_filled_is_left_alone(regression_df):
    profile, target_analysis, quality_report = _intelligence(regression_df)
    plan = ExperimentPlan(
        problem_type=ProblemType.REGRESSION,
        target_column="price",
        feature_columns=["sqft", "bedrooms"],
        candidate_model_families=["LightGBM"],
        evaluation_metrics=["rmse"],
        validation_strategy=ValidationStrategy(strategy_type=ValidationStrategyType.TRAIN_TEST_SPLIT),
    )

    result = validate_plan(plan, profile, target_analysis, quality_report)

    assert result.feature_columns == ["sqft", "bedrooms"]
    assert result.candidate_model_families == ["LightGBM"]
    assert result.evaluation_metrics == ["rmse"]
    assert result.validation_strategy.strategy_type == ValidationStrategyType.TRAIN_TEST_SPLIT
    assert result.needs_clarification is False


def test_unmatched_business_requirements_survive_validation_untouched(regression_df):
    """validate_plan() must never drop the Planner's advisory flag for a
    business-description term with no matching column - it's not a field
    validate_plan repairs/touches, so plan.model_copy(deep=True) must
    preserve it as-is."""
    profile, target_analysis, quality_report = _intelligence(regression_df)
    plan = ExperimentPlan(
        problem_type=ProblemType.REGRESSION,
        target_column="price",
        unmatched_business_requirements=["distance to nearest school"],
    )

    result = validate_plan(plan, profile, target_analysis, quality_report)

    assert result.unmatched_business_requirements == ["distance to nearest school"]


def test_forecasting_plan_gets_time_series_split_and_horizon_defaults(forecasting_df):
    profile, target_analysis, quality_report = _intelligence(forecasting_df)
    plan = ExperimentPlan(problem_type=ProblemType.FORECASTING, target_column="sales")

    result = validate_plan(plan, profile, target_analysis, quality_report)

    assert result.time_column == "date"
    assert result.forecast_horizon and result.forecast_horizon > 0
    assert result.validation_strategy.strategy_type == ValidationStrategyType.TIME_SERIES_SPLIT


# --- invalid planner output: repaired or rejected via clarification --------


def test_hallucinated_target_column_is_repaired_from_recommendation():
    import pandas as pd

    # id+target-only frame so analyze_target has exactly one candidate
    # (id-like integer columns are excluded), making recommended_target
    # unambiguous.
    df = pd.DataFrame({"listing_id": range(100), "price": [float(i) * 1.5 for i in range(100)]})
    profile, target_analysis, quality_report = _intelligence(df)
    plan = ExperimentPlan(problem_type=ProblemType.REGRESSION, target_column="not_a_real_column")

    result = validate_plan(plan, profile, target_analysis, quality_report)

    assert result.needs_clarification is False
    assert result.target_column == target_analysis.recommended_target == "price"
    assert result.validation_notes  # repair was logged


def test_missing_target_with_no_recommendation_needs_clarification():
    import pandas as pd

    df = pd.DataFrame({"a": [1, 2, 3], "b": [1, 2, 3]})  # a,b perfectly correlated -> no clean candidate either way
    profile, target_analysis, quality_report = _intelligence(df)
    plan = ExperimentPlan(problem_type=ProblemType.REGRESSION, target_column=None)

    result = validate_plan(plan, profile, target_analysis, quality_report)

    if target_analysis.recommended_target is None:
        assert result.needs_clarification is True
        assert result.clarification_question


def test_missing_problem_type_needs_clarification(regression_df):
    profile, target_analysis, quality_report = _intelligence(regression_df)
    plan = ExperimentPlan(target_column="price")

    result = validate_plan(plan, profile, target_analysis, quality_report)

    assert result.needs_clarification is True
    assert result.clarification_question


def test_forecasting_without_datetime_column_needs_clarification(regression_df):
    profile, target_analysis, quality_report = _intelligence(regression_df)
    plan = ExperimentPlan(problem_type=ProblemType.FORECASTING, target_column="price")

    result = validate_plan(plan, profile, target_analysis, quality_report)

    assert result.needs_clarification is True


def test_forecasting_missing_time_column_is_repaired_from_datetime_columns(forecasting_df):
    profile, target_analysis, quality_report = _intelligence(forecasting_df)
    plan = ExperimentPlan(problem_type=ProblemType.FORECASTING, target_column="sales", time_column=None)

    result = validate_plan(plan, profile, target_analysis, quality_report)

    assert result.needs_clarification is False
    assert result.time_column == "date"


def test_hallucinated_entity_column_is_cleared(regression_df):
    profile, target_analysis, quality_report = _intelligence(regression_df)
    plan = ExperimentPlan(
        problem_type=ProblemType.REGRESSION,
        target_column="price",
        entity_column="does_not_exist",
        scope_strategy=ScopeStrategy.POOLED,
    )

    result = validate_plan(plan, profile, target_analysis, quality_report)

    assert result.entity_column is None
    assert result.scope_strategy is None


def test_single_entity_without_filter_value_needs_clarification(multi_entity_forecasting_df):
    profile, target_analysis, quality_report = _intelligence(multi_entity_forecasting_df)
    plan = ExperimentPlan(
        problem_type=ProblemType.FORECASTING,
        target_column="sales",
        time_column="date",
        entity_column="store_id",
        scope_strategy=ScopeStrategy.SINGLE_ENTITY,
        entity_filter_value=None,
    )

    result = validate_plan(plan, profile, target_analysis, quality_report)

    assert result.needs_clarification is True
    assert result.clarification_question


def test_invalid_feature_columns_are_dropped_and_repaired(regression_df):
    profile, target_analysis, quality_report = _intelligence(regression_df)
    plan = ExperimentPlan(
        problem_type=ProblemType.REGRESSION,
        target_column="price",
        feature_columns=["sqft", "not_a_real_column", "price"],  # includes target + hallucination
    )

    result = validate_plan(plan, profile, target_analysis, quality_report)

    assert "not_a_real_column" not in result.feature_columns
    assert "price" not in result.feature_columns
    assert "sqft" in result.feature_columns


def test_wrong_validation_strategy_for_forecasting_is_repaired(forecasting_df):
    profile, target_analysis, quality_report = _intelligence(forecasting_df)
    plan = ExperimentPlan(
        problem_type=ProblemType.FORECASTING,
        target_column="sales",
        time_column="date",
        validation_strategy=ValidationStrategy(strategy_type=ValidationStrategyType.STRATIFIED_K_FOLD),
    )

    result = validate_plan(plan, profile, target_analysis, quality_report)

    assert result.validation_strategy.strategy_type == ValidationStrategyType.TIME_SERIES_SPLIT


def test_time_series_split_outside_forecasting_is_repaired(regression_df):
    profile, target_analysis, quality_report = _intelligence(regression_df)
    plan = ExperimentPlan(
        problem_type=ProblemType.REGRESSION,
        target_column="price",
        validation_strategy=ValidationStrategy(strategy_type=ValidationStrategyType.TIME_SERIES_SPLIT),
    )

    result = validate_plan(plan, profile, target_analysis, quality_report)

    assert result.validation_strategy.strategy_type != ValidationStrategyType.TIME_SERIES_SPLIT


def test_already_needs_clarification_is_left_untouched(regression_df):
    profile, target_analysis, quality_report = _intelligence(regression_df)
    plan = ExperimentPlan(needs_clarification=True, clarification_question="Which column is the target?")

    result = validate_plan(plan, profile, target_analysis, quality_report)

    assert result.needs_clarification is True
    assert result.clarification_question == "Which column is the target?"
    assert result.validation_notes == []  # no repair attempted
