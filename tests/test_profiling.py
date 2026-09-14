"""Tests for the deterministic Data Intelligence layer (tools/profiling.py).

No LLM involved anywhere here - these are pure pandas/Pydantic assertions.
"""
import numpy as np
import pandas as pd
import pytest

from agents.schemas import ColumnKind
from tools.profiling import analyze_data_quality, analyze_target, profile_dataset


# --- profile_dataset -------------------------------------------------------


def test_profile_dataset_basic_counts(classification_df):
    profile = profile_dataset(classification_df)
    assert profile.row_count == len(classification_df)
    assert profile.column_count == len(classification_df.columns)
    assert set(profile.column_names) == set(classification_df.columns)


def test_profile_dataset_excludes_sensitive_columns(classification_df):
    profile = profile_dataset(classification_df, sensitive_columns=["customer_name"])
    assert "customer_name" not in profile.column_names
    assert profile.column_count == len(classification_df.columns) - 1
    assert profile.excluded_sensitive_column_count == 1


def test_profile_dataset_classifies_column_kinds(classification_df):
    profile = profile_dataset(classification_df)
    assert "tenure_months" in profile.numerical_columns
    assert "monthly_charge" in profile.numerical_columns
    assert "contract_type" in profile.categorical_columns


def test_profile_dataset_datetime_kind_and_range(forecasting_df):
    profile = profile_dataset(forecasting_df)
    assert "date" in profile.datetime_columns
    date_col = next(c for c in profile.columns if c.name == "date")
    assert date_col.datetime_range is not None
    assert date_col.datetime_range["min"] is not None
    assert date_col.datetime_range["max"] is not None


def test_profile_dataset_missing_counts(classification_df):
    profile = profile_dataset(classification_df)
    monthly_charge = next(c for c in profile.columns if c.name == "monthly_charge")
    assert monthly_charge.missing_count == int(classification_df["monthly_charge"].isna().sum())
    assert monthly_charge.missing_count > 0
    assert monthly_charge.missing_pct > 0


def test_profile_dataset_duplicate_row_count():
    df = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "x", "y"]})
    profile = profile_dataset(df)
    assert profile.duplicate_row_count == 1
    assert profile.duplicate_row_pct == pytest.approx(33.33, abs=0.01)


def test_profile_dataset_constant_and_near_constant_columns():
    df = pd.DataFrame(
        {
            "constant": [5] * 100,
            "near_constant": [1] * 99 + [2],
            "varied": list(range(100)),
        }
    )
    profile = profile_dataset(df)
    assert "constant" in profile.constant_columns
    assert "near_constant" in profile.near_constant_columns
    assert "varied" not in profile.constant_columns
    assert "varied" not in profile.near_constant_columns


def test_profile_dataset_numeric_stats_present(regression_df):
    profile = profile_dataset(regression_df)
    sqft_col = next(c for c in profile.columns if c.name == "sqft")
    assert sqft_col.numeric_stats is not None
    assert sqft_col.numeric_stats["min"] <= sqft_col.numeric_stats["mean"] <= sqft_col.numeric_stats["max"]


def test_profile_dataset_handles_empty_dataframe():
    df = pd.DataFrame({"a": pd.Series(dtype=float), "b": pd.Series(dtype=object)})
    profile = profile_dataset(df)
    assert profile.row_count == 0
    assert profile.duplicate_row_count == 0
    assert profile.duplicate_row_pct == 0.0
    for col in profile.columns:
        assert col.missing_pct == 0.0


# --- analyze_target ----------------------------------------------------


def test_analyze_target_finds_binary_classification_candidate(classification_df):
    profile = profile_dataset(classification_df, sensitive_columns=["customer_name"])
    analysis = analyze_target(classification_df, profile, sensitive_columns=["customer_name"])
    names = [c.name for c in analysis.candidate_targets]
    assert "churn" in names
    churn_candidate = next(c for c in analysis.candidate_targets if c.name == "churn")
    assert "classification" in churn_candidate.likely_problem_types


def test_analyze_target_finds_regression_candidate(regression_df):
    profile = profile_dataset(regression_df)
    analysis = analyze_target(regression_df, profile)
    price_candidate = next(c for c in analysis.candidate_targets if c.name == "price")
    assert "regression" in price_candidate.likely_problem_types


def test_analyze_target_excludes_id_like_columns():
    df = pd.DataFrame({"row_id": list(range(50)), "value": list(range(50)), "label": [0, 1] * 25})
    profile = profile_dataset(df)
    analysis = analyze_target(df, profile)
    names = [c.name for c in analysis.candidate_targets]
    assert "row_id" not in names


def test_analyze_target_excludes_datetime_columns(forecasting_df):
    profile = profile_dataset(forecasting_df)
    analysis = analyze_target(forecasting_df, profile)
    names = [c.name for c in analysis.candidate_targets]
    assert "date" not in names
    assert "sales" in names


def test_analyze_target_hints_forecasting_when_datetime_present(forecasting_df):
    profile = profile_dataset(forecasting_df)
    analysis = analyze_target(forecasting_df, profile)
    sales_candidate = next(c for c in analysis.candidate_targets if c.name == "sales")
    assert "forecasting" in sales_candidate.likely_problem_types


def test_analyze_target_recommends_when_single_unambiguous_candidate(forecasting_df):
    profile = profile_dataset(forecasting_df)
    analysis = analyze_target(forecasting_df, profile)
    assert analysis.recommended_target == "sales"


def test_analyze_target_handles_empty_dataframe():
    df = pd.DataFrame({"a": pd.Series(dtype=float)})
    profile = profile_dataset(df)
    analysis = analyze_target(df, profile)
    assert analysis.candidate_targets == []
    assert analysis.recommended_target is None


# --- analyze_data_quality -----------------------------------------------


def test_analyze_data_quality_reports_missing_values(classification_df):
    profile = profile_dataset(classification_df)
    report = analyze_data_quality(classification_df, profile)
    assert "monthly_charge" in report.missing_value_columns
    assert report.missing_value_columns["monthly_charge"] > 0


def test_analyze_data_quality_reports_duplicates():
    df = pd.DataFrame({"a": [1, 1, 2, 2], "b": ["x", "x", "y", "z"]})
    profile = profile_dataset(df)
    report = analyze_data_quality(df, profile)
    assert report.duplicate_row_count == 1
    assert any(issue.issue_type == "duplicate_rows" for issue in report.issues)


def test_analyze_data_quality_flags_constant_columns():
    df = pd.DataFrame({"constant": [1] * 20, "varied": list(range(20))})
    profile = profile_dataset(df)
    report = analyze_data_quality(df, profile)
    assert "constant" in report.constant_columns


def test_analyze_data_quality_flags_outliers():
    rng = np.random.default_rng(3)
    values = list(10 + rng.uniform(-0.1, 0.1, 40)) + [10_000.0]  # single extreme outlier
    df = pd.DataFrame({"amount": values})
    profile = profile_dataset(df)
    report = analyze_data_quality(df, profile)
    assert "amount" in report.possible_outlier_columns
    assert report.possible_outlier_columns["amount"]["count"] == 1


def test_analyze_data_quality_flags_numeric_text_as_invalid_dtype():
    df = pd.DataFrame({"amount_as_text": ["1", "2", "3", "4", "5"] * 5})
    profile = profile_dataset(df)
    report = analyze_data_quality(df, profile)
    assert "amount_as_text" in report.invalid_dtype_columns


def test_analyze_data_quality_flags_id_like_columns_as_suspicious():
    df = pd.DataFrame({"uuid": [f"id-{i}" for i in range(50)], "value": list(range(50))})
    profile = profile_dataset(df)
    report = analyze_data_quality(df, profile)
    assert "uuid" in report.suspicious_columns


def test_analyze_data_quality_flags_leakage_by_column_name():
    df = pd.DataFrame({"feature": list(range(20)), "target_leak": list(range(20))})
    profile = profile_dataset(df)
    report = analyze_data_quality(df, profile)
    assert "target_leak" in report.possible_leakage_columns


def test_analyze_data_quality_flags_leakage_via_near_perfect_correlation():
    rng = np.random.default_rng(0)
    base = rng.normal(0, 1, 100)
    df = pd.DataFrame({"a": base, "b": base * 2.0 + 1e-9, "c": rng.normal(0, 1, 100)})
    profile = profile_dataset(df)
    report = analyze_data_quality(df, profile)
    assert "a" in report.possible_leakage_columns
    assert "b" in report.possible_leakage_columns
    assert "c" not in report.possible_leakage_columns


def test_analyze_data_quality_excludes_sensitive_columns(classification_df):
    report = analyze_data_quality(
        classification_df,
        profile_dataset(classification_df, sensitive_columns=["customer_name"]),
        sensitive_columns=["customer_name"],
    )
    assert "customer_name" not in report.missing_value_columns
    assert "customer_name" not in report.suspicious_columns


def test_analyze_data_quality_perfect_score_for_clean_data():
    # Every (category, flag, group) combination appears exactly once, so rows
    # are unique without any single column being near-100%-unique (ID-like).
    import itertools

    combos = list(itertools.product(["a", "b", "c"], ["x", "y"], range(10)))
    df = pd.DataFrame(combos, columns=["category", "flag", "group"])
    profile = profile_dataset(df)
    report = analyze_data_quality(df, profile)
    assert report.overall_quality_score == 100.0
    assert report.issues == []


def test_analyze_data_quality_handles_empty_dataframe():
    df = pd.DataFrame({"a": pd.Series(dtype=float)})
    profile = profile_dataset(df)
    report = analyze_data_quality(df, profile)
    assert report.overall_quality_score == 0.0
    assert any(issue.issue_type == "empty_dataset" for issue in report.issues)
