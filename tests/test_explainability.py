"""Tests for deterministic Model Explainability (tools/explainability.py):
feature/permutation/SHAP importance for tabular models, trend/seasonality/
decomposition for forecasting, and - critically - graceful fallback when a
technique doesn't reliably apply (never an error, never mandatory).
"""
import numpy as np
import pandas as pd
import pytest

from agents.schemas import ProblemType
from tools.explainability import (
    compute_permutation_importance,
    compute_shap_importance,
    explain_forecast,
    explain_tabular_model,
    shap_unavailable_reason,
    unsupported_automl_explainability,
)
from tools.model_registry import resolve_candidates


@pytest.fixture
def classification_train_test():
    rng = np.random.default_rng(0)
    n = 200
    f1 = rng.normal(0, 1, n)
    f2 = rng.normal(0, 1, n)
    target = (f1 * 2 + f2 * 0.1 > 0).astype(int)  # f1 clearly dominant
    df = pd.DataFrame({"f1": f1, "f2": f2, "target": target})
    return df.iloc[:150].reset_index(drop=True), df.iloc[150:].reset_index(drop=True)


@pytest.fixture
def regression_train_test():
    rng = np.random.default_rng(1)
    n = 200
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    y = 5 * x1 + 0.01 * x2 + rng.normal(0, 0.1, n)
    df = pd.DataFrame({"x1": x1, "x2": x2, "y": y})
    return df.iloc[:150].reset_index(drop=True), df.iloc[150:].reset_index(drop=True)


# --- permutation importance ---------------------------------------------------


def test_permutation_importance_identifies_dominant_feature(classification_train_test):
    train_df, test_df = classification_train_test
    (rf,), _ = resolve_candidates(ProblemType.CLASSIFICATION, ["random_forest"])
    fitted = rf.train_fn(train_df, "target", None)

    result = compute_permutation_importance(rf, fitted, test_df, "target", None, ProblemType.CLASSIFICATION)

    assert result is not None
    assert result["f1"] > result["f2"]


def test_permutation_importance_none_for_baseline(classification_train_test):
    train_df, test_df = classification_train_test
    (baseline,), _ = resolve_candidates(ProblemType.CLASSIFICATION, ["baseline"])
    fitted = baseline.train_fn(train_df, "target", None)

    result = compute_permutation_importance(baseline, fitted, test_df, "target", None, ProblemType.CLASSIFICATION)

    assert result is None


def test_permutation_importance_none_for_forecasting():
    (naive,), _ = resolve_candidates(ProblemType.FORECASTING, ["naive"])
    result = compute_permutation_importance(naive, {}, pd.DataFrame(), "sales", "date", ProblemType.FORECASTING)
    assert result is None


def test_permutation_importance_handles_prediction_failure_gracefully(classification_train_test):
    train_df, test_df = classification_train_test
    (rf,), _ = resolve_candidates(ProblemType.CLASSIFICATION, ["random_forest"])
    fitted = rf.train_fn(train_df, "target", None)
    broken_rf = rf.__class__(**{**rf.__dict__, "predict_fn": lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))})

    result = compute_permutation_importance(broken_rf, fitted, test_df, "target", None, ProblemType.CLASSIFICATION)

    assert result is None  # never raises


# --- SHAP -----------------------------------------------------------------------


def test_shap_importance_identifies_dominant_feature_for_tree_model(classification_train_test):
    train_df, test_df = classification_train_test
    (rf,), _ = resolve_candidates(ProblemType.CLASSIFICATION, ["random_forest"])
    fitted = rf.train_fn(train_df, "target", None)

    result = compute_shap_importance(rf, fitted, test_df, "target", None)

    assert result is not None
    assert result["f1"] > result["f2"]


def test_shap_importance_none_for_linear_model(classification_train_test):
    train_df, test_df = classification_train_test
    (logreg,), _ = resolve_candidates(ProblemType.CLASSIFICATION, ["logistic_regression"])
    fitted = logreg.train_fn(train_df, "target", None)

    result = compute_shap_importance(logreg, fitted, test_df, "target", None)

    assert result is None


def test_shap_importance_none_when_dependency_missing(monkeypatch, classification_train_test):
    train_df, test_df = classification_train_test
    (rf,), _ = resolve_candidates(ProblemType.CLASSIFICATION, ["random_forest"])
    fitted = rf.train_fn(train_df, "target", None)

    import tools.explainability as explainability_module

    monkeypatch.setattr(explainability_module, "is_dependency_available", lambda name: False)

    result = compute_shap_importance(rf, fitted, test_df, "target", None)

    assert result is None


def test_shap_unavailable_reason_distinguishes_missing_dependency_vs_family(monkeypatch, classification_train_test):
    (rf,), _ = resolve_candidates(ProblemType.CLASSIFICATION, ["random_forest"])
    (logreg,), _ = resolve_candidates(ProblemType.CLASSIFICATION, ["logistic_regression"])

    import tools.explainability as explainability_module

    monkeypatch.setattr(explainability_module, "is_dependency_available", lambda name: False)
    assert "not installed" in shap_unavailable_reason(rf)

    monkeypatch.setattr(explainability_module, "is_dependency_available", lambda name: True)
    assert "not applicable" in shap_unavailable_reason(logreg)


# --- explain_tabular_model (combined result) ------------------------------------


def test_explain_tabular_model_supported_for_random_forest(classification_train_test):
    train_df, test_df = classification_train_test
    (rf,), _ = resolve_candidates(ProblemType.CLASSIFICATION, ["random_forest"])
    fitted = rf.train_fn(train_df, "target", None)
    predictions = rf.predict_fn(fitted, test_df, "target", None)

    from tools.model_runner import _compute_feature_importance

    feature_importance = _compute_feature_importance(rf, fitted, ProblemType.CLASSIFICATION, train_df, "target", None)
    result = explain_tabular_model(rf, fitted, ProblemType.CLASSIFICATION, test_df, "target", None, feature_importance)

    assert result.supported is True
    assert result.feature_importance_method == "impurity"
    assert result.feature_importance[0].feature == "f1"
    assert result.permutation_importance_supported is True
    assert result.shap_supported is True


def test_explain_tabular_model_labels_xgboost_gain_not_impurity(regression_train_test):
    """Regression coverage: XGBoost's native feature_importances_ default is
    "gain", not sklearn tree impurity/Gini reduction - only random_forest
    actually uses impurity. See tools/explainability.py's
    _IMPORTANCE_METHOD_BY_MODEL_NAME."""
    train_df, test_df = regression_train_test
    (xgb,), _ = resolve_candidates(ProblemType.REGRESSION, ["xgboost"])
    fitted = xgb.train_fn(train_df, "y", None)

    from tools.model_runner import _compute_feature_importance

    feature_importance = _compute_feature_importance(xgb, fitted, ProblemType.REGRESSION, train_df, "y", None)
    result = explain_tabular_model(xgb, fitted, ProblemType.REGRESSION, test_df, "y", None, feature_importance)

    assert result.feature_importance_method == "gain"


def test_explain_tabular_model_labels_lightgbm_split_not_impurity(regression_train_test):
    train_df, test_df = regression_train_test
    (lgbm,), _ = resolve_candidates(ProblemType.REGRESSION, ["lightgbm"])
    fitted = lgbm.train_fn(train_df, "y", None)

    from tools.model_runner import _compute_feature_importance

    feature_importance = _compute_feature_importance(lgbm, fitted, ProblemType.REGRESSION, train_df, "y", None)
    result = explain_tabular_model(lgbm, fitted, ProblemType.REGRESSION, test_df, "y", None, feature_importance)

    assert result.feature_importance_method == "split"


def test_explain_tabular_model_unsupported_for_baseline(classification_train_test):
    train_df, test_df = classification_train_test
    (baseline,), _ = resolve_candidates(ProblemType.CLASSIFICATION, ["baseline"])
    fitted = baseline.train_fn(train_df, "target", None)

    result = explain_tabular_model(baseline, fitted, ProblemType.CLASSIFICATION, test_df, "target", None, None)

    assert result.supported is False
    assert result.reason_unsupported
    assert result.feature_importance == []
    assert result.permutation_importance == []
    assert result.shap_importance == []


def test_explain_tabular_model_linear_uses_coefficient_method(regression_train_test):
    train_df, test_df = regression_train_test
    (linreg,), _ = resolve_candidates(ProblemType.REGRESSION, ["linear_regression"])
    fitted = linreg.train_fn(train_df, "y", None)

    from tools.model_runner import _compute_feature_importance

    feature_importance = _compute_feature_importance(linreg, fitted, ProblemType.REGRESSION, train_df, "y", None)
    result = explain_tabular_model(linreg, fitted, ProblemType.REGRESSION, test_df, "y", None, feature_importance)

    assert result.supported is True
    assert result.feature_importance_method == "coefficient"
    assert result.feature_importance[0].feature == "x1"
    assert result.shap_supported is False  # SHAP is tree-only here
    assert result.shap_unavailable_reason


def test_unsupported_automl_explainability_is_always_unsupported_by_policy():
    result = unsupported_automl_explainability("WeightedEnsemble_L2", ProblemType.CLASSIFICATION)
    assert result.supported is False
    assert "AutoGluon" in result.reason_unsupported


# --- forecasting: trend/seasonality/decomposition -------------------------------


def test_explain_forecast_detects_trend_and_seasonality():
    n = 70
    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    values = np.linspace(100, 150, n) + 15 * np.sin(np.arange(n) * 2 * np.pi / 7)
    df = pd.DataFrame({"date": dates, "sales": values})

    result = explain_forecast(df, "sales", "date", seasonal_period=7)

    assert result.supported is True
    assert result.trend_detected is True
    assert result.seasonality_detected is True
    assert result.forecast_components_available is True
    assert result.trend_strength is not None
    assert result.seasonal_strength is not None


def test_explain_forecast_decomposition_unavailable_with_too_little_data():
    """Too few rows to decompose into trend/seasonal/residual components,
    but the lightweight trend/seasonality signals (tools/problem_detection.py)
    still return a real (if low-confidence) answer rather than nothing -
    so `supported` stays True and only forecast_components_available is False.
    """
    dates = pd.date_range("2023-01-01", periods=5, freq="D")
    df = pd.DataFrame({"date": dates, "sales": [1.0, 2.0, 1.5, 2.5, 1.0]})

    result = explain_forecast(df, "sales", "date", seasonal_period=7)

    assert result.forecast_components_available is False
    assert result.trend_strength is None
    assert result.seasonal_strength is None
    assert result.supported is True
    assert result.trend_detected is not None


def test_explain_forecast_never_raises_on_malformed_data():
    df = pd.DataFrame({"date": ["not", "a", "date"], "sales": [1, 2, 3]})
    result = explain_forecast(df, "sales", "date", seasonal_period=7)
    assert result is not None  # degrades gracefully, never raises
