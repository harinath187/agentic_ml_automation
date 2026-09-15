"""Tests for deterministic problem-type detection (tools/problem_detection.py).

No LLM involved anywhere here - these are pure pandas/Pydantic assertions.
"""
import pandas as pd

from agents.schemas import ProblemType
from tools.problem_detection import detect_problem
from tools.profiling import analyze_data_quality, analyze_target, profile_dataset


def _detect(df, sensitive_columns=None):
    profile = profile_dataset(df, sensitive_columns=sensitive_columns)
    target_analysis = analyze_target(df, profile, sensitive_columns=sensitive_columns)
    quality_report = analyze_data_quality(df, profile, sensitive_columns=sensitive_columns)
    return detect_problem(df, profile, target_analysis, quality_report, sensitive_columns=sensitive_columns)


# --- classification --------------------------------------------------------
#
# tools.profiling.analyze_target flags every non-ID column as a POSSIBLE
# target (it has no way to know which columns are meant as features without
# business context) - so on a realistic multi-feature dataset there is
# rarely a single unambiguous candidate. These tests use a minimal
# id+target-only frame to exercise detection in the case it's designed for
# (unambiguous), and separately assert the honest "ambiguous -> low
# confidence, let the Planner decide" behavior on a realistic dataset.


def test_detects_classification_problem():
    df = pd.DataFrame({"customer_id": range(100), "churn": [0, 1] * 50})
    definition = _detect(df)
    assert definition.detected_problem_type == ProblemType.CLASSIFICATION
    assert definition.target_column == "churn"
    assert definition.confidence == "high"


def test_classification_has_no_time_series_signals():
    df = pd.DataFrame({"customer_id": range(100), "churn": [0, 1] * 50})
    definition = _detect(df)
    assert definition.time_series_signals is None


def test_classification_on_ambiguous_multi_feature_dataset_defers_to_planner(classification_df):
    # tenure_months/monthly_charge/support_calls are all plausible regression
    # or classification candidates in their own right, so the deterministic
    # layer correctly refuses to guess rather than assert "churn" outright.
    definition = _detect(classification_df, sensitive_columns=["customer_name"])
    assert definition.confidence in ("low", "medium")
    if definition.detected_problem_type is not None:
        assert definition.target_column is None or definition.confidence != "high"


# --- regression -------------------------------------------------------------


def test_detects_regression_problem():
    df = pd.DataFrame({"listing_id": range(100), "price": [float(i) * 1.5 for i in range(100)]})
    definition = _detect(df)
    assert definition.detected_problem_type == ProblemType.REGRESSION
    assert definition.target_column == "price"
    assert definition.confidence == "high"


def test_regression_has_no_time_series_signals():
    df = pd.DataFrame({"listing_id": range(100), "price": [float(i) * 1.5 for i in range(100)]})
    definition = _detect(df)
    assert definition.time_series_signals is None


def test_regression_on_ambiguous_multi_feature_dataset_defers_to_planner(regression_df):
    # sqft/bedrooms/age_years are all plausible target-like numeric columns
    # too, so a single high-confidence pick isn't deterministically safe.
    definition = _detect(regression_df)
    assert definition.confidence in ("low", "medium")


# --- forecasting -------------------------------------------------------------


def test_detects_forecasting_problem(forecasting_df):
    definition = _detect(forecasting_df)
    assert definition.detected_problem_type == ProblemType.FORECASTING
    assert definition.target_column == "sales"
    assert definition.confidence == "high"


def test_forecasting_time_series_signals_datetime_and_frequency(forecasting_df):
    definition = _detect(forecasting_df)
    signals = definition.time_series_signals
    assert signals is not None
    assert signals.datetime_column == "date"
    assert signals.frequency == "D"
    assert signals.time_range is not None
    assert signals.time_range["min"] is not None


def test_forecasting_time_series_signals_no_missing_periods_for_continuous_daily_data(forecasting_df):
    definition = _detect(forecasting_df)
    assert definition.time_series_signals.missing_periods_count == 0
    assert definition.time_series_signals.missing_periods_pct == 0.0


def test_forecasting_time_series_signals_detects_seasonality(forecasting_df):
    # forecasting_df bakes in a 15-amplitude weekly sine term against noise of
    # std 5, so day-of-week means should vary meaningfully.
    definition = _detect(forecasting_df)
    assert definition.time_series_signals.seasonality_detected is True


def test_forecasting_time_series_signals_detects_trend(forecasting_df):
    # forecasting_df bakes in a monotonic trend from 100 to 200.
    definition = _detect(forecasting_df)
    assert definition.time_series_signals.trend_detected is True
    assert "ncreasing" in definition.time_series_signals.trend_notes


def test_forecasting_time_series_signals_detects_item_column(multi_entity_forecasting_df):
    definition = _detect(multi_entity_forecasting_df)
    assert definition.time_series_signals is not None
    assert definition.time_series_signals.item_column == "store_id"


def test_forecasting_missing_periods_detected_for_sparse_dates():
    import pandas as pd

    dates = pd.to_datetime(
        ["2023-01-01", "2023-01-02", "2023-01-05", "2023-01-10"]
    )  # gaps on 01-03, 01-04, 01-06..01-09
    df = pd.DataFrame({"date": dates, "value": [1.0, 2.0, 3.0, 4.0]})
    definition = _detect(df)
    assert definition.time_series_signals.missing_periods_count > 0


# --- edge cases --------------------------------------------------------------


def test_detect_problem_handles_empty_dataframe():
    import pandas as pd

    df = pd.DataFrame({"a": pd.Series(dtype=float)})
    definition = _detect(df)
    assert definition.detected_problem_type is None
    assert definition.confidence == "low"


def test_detect_problem_low_confidence_when_no_candidates():
    import pandas as pd

    df = pd.DataFrame({"constant": [1] * 20})
    definition = _detect(df)
    assert definition.detected_problem_type is None
    assert definition.target_column is None
    assert definition.confidence == "low"
