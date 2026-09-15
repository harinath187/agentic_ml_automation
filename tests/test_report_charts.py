"""Tests for deterministic report chart generation (tools/report_charts.py).

Every chart function here takes plain numbers/labels - no LLM involved -
and returns a base64 PNG data URI or None. These tests mostly check the
"does it produce a sane image or gracefully return None" contract, plus
build_report_charts()'s per-problem-type selection logic.
"""
from tools.report_charts import (
    build_report_charts,
    confusion_matrix_chart,
    feature_importance_chart,
    historical_vs_predicted_chart,
    model_comparison_chart,
    prediction_vs_actual_chart,
    residual_chart,
)

DATA_URI_PREFIX = "data:image/png;base64,"


def _comparison(problem_type="regression", winner="good"):
    return {
        "problem_type": problem_type,
        "primary_metric": "rmse",
        "higher_is_better": False,
        "results": [
            {"model_name": "good", "status": "success", "metrics": {"rmse": 1.0}, "feature_importance": {"x": 0.7, "y": 0.3}},
            {"model_name": "bad", "status": "success", "metrics": {"rmse": 5.0}, "feature_importance": None},
            {"model_name": "broken", "status": "failed", "metrics": {}, "feature_importance": None},
        ],
        "ranked_model_names": ["good", "bad"],
        "winner": winner,
        "winner_reasoning": "good wins",
    }


# --- individual chart functions ----------------------------------------------


def test_model_comparison_chart_returns_data_uri():
    chart = model_comparison_chart(_comparison())
    assert chart is not None
    assert chart.startswith(DATA_URI_PREFIX)


def test_model_comparison_chart_none_without_primary_metric():
    comparison = _comparison()
    comparison["primary_metric"] = None
    assert model_comparison_chart(comparison) is None


def test_model_comparison_chart_none_when_nothing_scored():
    comparison = {"primary_metric": "rmse", "results": [{"model_name": "a", "status": "failed", "metrics": {}}]}
    assert model_comparison_chart(comparison) is None


def test_historical_vs_predicted_chart_returns_data_uri_when_data_present():
    chart = historical_vs_predicted_chart([1.0, 2.0, 3.0], [1.1, 1.9, 3.2])
    assert chart is not None
    assert chart.startswith(DATA_URI_PREFIX)


def test_historical_vs_predicted_chart_none_when_empty():
    assert historical_vs_predicted_chart([], []) is None
    assert historical_vs_predicted_chart(None, None) is None


def test_prediction_vs_actual_chart_returns_data_uri():
    chart = prediction_vs_actual_chart([1.0, 2.0, 3.0], [1.2, 1.8, 3.3])
    assert chart is not None
    assert chart.startswith(DATA_URI_PREFIX)


def test_residual_chart_returns_data_uri():
    chart = residual_chart([1.0, 2.0, 3.0], [1.2, 1.8, 3.3])
    assert chart is not None
    assert chart.startswith(DATA_URI_PREFIX)


def test_confusion_matrix_chart_returns_data_uri():
    chart = confusion_matrix_chart(["a", "b", "a", "b", "a"], ["a", "b", "b", "b", "a"])
    assert chart is not None
    assert chart.startswith(DATA_URI_PREFIX)


def test_confusion_matrix_chart_none_when_empty():
    assert confusion_matrix_chart([], []) is None


def test_feature_importance_chart_returns_data_uri():
    chart = feature_importance_chart({"a": 0.5, "b": 0.3, "c": 0.2})
    assert chart is not None
    assert chart.startswith(DATA_URI_PREFIX)


def test_feature_importance_chart_none_when_empty():
    assert feature_importance_chart({}) is None
    assert feature_importance_chart(None) is None


# --- build_report_charts orchestration ---------------------------------------


def test_build_report_charts_forecasting_includes_historical_vs_predicted():
    comparison = _comparison(problem_type="forecasting", winner="good")
    chart_data = {"good": {"actual": [1.0, 2.0, 3.0], "predicted": [1.1, 1.9, 3.2]}}

    charts = build_report_charts("forecasting", comparison, chart_data)

    assert "model_comparison" in charts
    assert "historical_vs_predicted" in charts
    assert "confusion_matrix" not in charts
    assert "prediction_vs_actual" not in charts


def test_build_report_charts_classification_includes_confusion_matrix():
    comparison = _comparison(problem_type="classification", winner="good")
    chart_data = {"good": {"actual": ["a", "b", "a"], "predicted": ["a", "a", "a"]}}

    charts = build_report_charts("classification", comparison, chart_data)

    assert "confusion_matrix" in charts
    assert "historical_vs_predicted" not in charts


def test_build_report_charts_regression_includes_prediction_vs_actual_and_residuals():
    comparison = _comparison(problem_type="regression", winner="good")
    chart_data = {"good": {"actual": [1.0, 2.0, 3.0], "predicted": [1.2, 1.8, 3.3]}}

    charts = build_report_charts("regression", comparison, chart_data)

    assert "prediction_vs_actual" in charts
    assert "residuals" in charts
    assert "confusion_matrix" not in charts


def test_build_report_charts_includes_feature_importance_for_winner():
    comparison = _comparison(problem_type="regression", winner="good")
    chart_data = {"good": {"actual": [1.0, 2.0], "predicted": [1.1, 1.9]}}

    charts = build_report_charts("regression", comparison, chart_data)

    assert "feature_importance" in charts  # "good"'s feature_importance is {"x": 0.7, "y": 0.3}


def test_build_report_charts_no_feature_importance_when_winner_lacks_it():
    comparison = _comparison(problem_type="regression", winner="bad")  # "bad" has feature_importance=None
    chart_data = {"bad": {"actual": [1.0, 2.0], "predicted": [1.1, 1.9]}}

    charts = build_report_charts("regression", comparison, chart_data)

    assert "feature_importance" not in charts


def test_build_report_charts_empty_when_no_model_comparison():
    assert build_report_charts("regression", None, {"good": {"actual": [1.0], "predicted": [1.0]}}) == {}


def test_build_report_charts_handles_missing_chart_data_gracefully():
    comparison = _comparison(problem_type="regression", winner="good")
    charts = build_report_charts("regression", comparison, None)

    assert "model_comparison" in charts  # doesn't need chart_data
    assert "prediction_vs_actual" not in charts  # no sample available
    assert "residuals" not in charts


def test_build_report_charts_handles_no_winner_gracefully():
    comparison = _comparison(problem_type="regression", winner=None)
    comparison["ranked_model_names"] = []
    charts = build_report_charts("regression", comparison, {})
    # model_comparison_chart still renders from results even with no winner highlighted
    assert isinstance(charts, dict)
