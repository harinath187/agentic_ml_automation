"""Tests for tools/feature_selection.py - the deterministic step that makes
DataQualityReport's exclusion lists and ExperimentPlan.feature_columns
actually subset the dataframe before cleaning/feature engineering, instead of
being advisory-only context for the LLM.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from agents.schemas import ExperimentPlan, ProblemType, ScopeStrategy
from tools.feature_selection import apply_feature_selection
from tools.plan_validation import validate_plan
from tools.profiling import analyze_data_quality, analyze_target, profile_dataset


@pytest.fixture
def messy_df() -> pd.DataFrame:
    rng = np.random.default_rng(3)
    n = 200
    # Discrete with repeats (not ~100% unique) so it isn't itself mistaken for
    # an id-like column - a genuinely useful, "kept" predictor.
    good_feat = rng.integers(0, 50, n).astype(float)
    dup_base = rng.normal(0, 1, n)
    return pd.DataFrame(
        {
            "record_id": range(n),  # ~100% unique -> suspicious (id-like)
            "good_feat": good_feat,
            "dup_a": dup_base,  # near-duplicate pair -> both flagged possible_leakage
            "dup_b": dup_base + rng.normal(0, 1e-6, n),
            "const_col": [5] * n,  # single unique value -> constant
            "near_const_col": [1] * (n - 1) + [2],  # 199/200 one value -> near-constant
            "leak_col": rng.integers(0, 20, n),  # name hint ("leak") -> possible_leakage
            # noisy enough relative to good_feat that it never approaches the
            # 0.995 near-duplicate correlation threshold itself.
            "target": 2 * good_feat + rng.normal(0, 30, n),
        }
    )


def _intelligence(df, sensitive_columns=None):
    profile = profile_dataset(df, sensitive_columns=sensitive_columns)
    target_analysis = analyze_target(df, profile, sensitive_columns=sensitive_columns)
    quality_report = analyze_data_quality(df, profile, sensitive_columns=sensitive_columns)
    return profile, target_analysis, quality_report


def test_default_repair_drops_junk_columns_and_apply_enforces_it(messy_df):
    """End-to-end of the two deterministic steps: validate_plan computes the
    default feature_columns list, apply_feature_selection then actually
    subsets the dataframe to it."""
    profile, target_analysis, quality_report = _intelligence(messy_df)
    plan = ExperimentPlan(problem_type=ProblemType.REGRESSION, target_column="target")

    validated = validate_plan(plan, profile, target_analysis, quality_report)
    assert "record_id" not in validated.feature_columns
    assert "const_col" not in validated.feature_columns
    assert "near_const_col" not in validated.feature_columns
    assert "leak_col" not in validated.feature_columns
    assert "dup_a" not in validated.feature_columns
    assert "dup_b" not in validated.feature_columns
    assert "good_feat" in validated.feature_columns

    selected_df, log = apply_feature_selection(messy_df, validated, quality_report)

    assert set(selected_df.columns) == {"good_feat", "target"}
    dropped_columns = {item["column"] for item in log["dropped"]}
    assert {"record_id", "const_col", "near_const_col", "leak_col", "dup_a", "dup_b"} <= dropped_columns
    assert "target" in log["kept"]


def test_dropped_reasons_match_quality_report_lists(messy_df):
    profile, target_analysis, quality_report = _intelligence(messy_df)
    plan = ExperimentPlan(problem_type=ProblemType.REGRESSION, target_column="target")
    validated = validate_plan(plan, profile, target_analysis, quality_report)

    _, log = apply_feature_selection(messy_df, validated, quality_report)
    reasons = {item["column"]: item["reason"] for item in log["dropped"]}

    assert reasons["record_id"] == "id_like"
    assert reasons["const_col"] == "constant"
    assert reasons["near_const_col"] == "near_constant"
    assert reasons["leak_col"] == "possible_leakage"


def test_target_time_entity_always_survive_even_if_not_in_feature_columns():
    df = pd.DataFrame(
        {
            "store_id": ["a", "a", "b", "b"],
            "date": pd.date_range("2023-01-01", periods=4),
            "sales": [1.0, 2.0, 3.0, 4.0],
            "junk": [1, 1, 1, 1],
        }
    )
    profile, _target_analysis, quality_report = _intelligence(df)
    plan = ExperimentPlan(
        problem_type=ProblemType.FORECASTING,
        target_column="sales",
        time_column="date",
        entity_column="store_id",
        scope_strategy=ScopeStrategy.POOLED,
        feature_columns=[],  # explicitly nothing selected as a "feature"
    )

    selected_df, log = apply_feature_selection(df, plan, quality_report)

    assert {"sales", "date", "store_id"} <= set(selected_df.columns)
    assert "junk" not in selected_df.columns


def test_explicit_narrowed_feature_columns_is_respected_as_is(regression_df):
    """An LLM-narrowed explicit list is honored - only target/time/entity get
    force-kept on top of it, nothing else is added back in."""
    profile, target_analysis, quality_report = _intelligence(regression_df)
    plan = ExperimentPlan(
        problem_type=ProblemType.REGRESSION,
        target_column="price",
        feature_columns=["sqft"],
    )

    selected_df, log = apply_feature_selection(regression_df, plan, quality_report)

    assert set(selected_df.columns) == {"sqft", "price"}
    dropped_columns = {item["column"] for item in log["dropped"]}
    assert {"bedrooms", "age_years", "neighborhood"} <= dropped_columns
    for column in ("bedrooms", "age_years", "neighborhood"):
        assert next(item["reason"] for item in log["dropped"] if item["column"] == column) == "not selected by plan"


def test_hierarchical_plan_with_feature_columns_gets_a_validation_note(multi_entity_forecasting_df):
    profile, target_analysis, quality_report = _intelligence(multi_entity_forecasting_df)
    plan = ExperimentPlan(
        problem_type=ProblemType.FORECASTING,
        target_column="sales",
        time_column="date",
        entity_column="store_id",
        scope_strategy=ScopeStrategy.HIERARCHICAL,
        feature_columns=["store_id"],
    )

    result = validate_plan(plan, profile, target_analysis, quality_report)

    assert any("hierarchical" in note for note in result.validation_notes)
