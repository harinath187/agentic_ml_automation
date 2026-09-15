"""Deterministic Model Explainability (Phase 7).

Computes feature importance, permutation importance, and SHAP importance
(tree-based models only) for tabular candidates, and trend/seasonality/
forecast-decomposition signals for forecasting. No LLM involved anywhere -
every number here is plain sklearn/numpy/statsmodels/shap arithmetic, and
(like tools/evaluation.py's ModelComparison) this is the ground truth
agents/recommender.py's explanation narrative is checked against.

Explainability is best-effort and NEVER mandatory: a model/technique that
doesn't apply reliably (a baseline predictor, a statistical forecasting
model with no feature matrix, SHAP without the tree structure it needs, the
`shap` package missing) simply produces `supported=False` with a
`reason_unsupported` - never an error, and never a fabricated substitute.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from agents.schemas import ProblemType
from tools import model_registry
from tools.model_registry import ModelDefinition, is_dependency_available

MAX_IMPORTANCE_ENTRIES = 10
SHAP_MAX_SAMPLES = 100
PERMUTATION_REPEATS = 5
PERMUTATION_RANDOM_STATE = 42
_SHAP_SUPPORTED_FAMILIES = ("tree_ensemble", "gradient_boosting")
# tools/model_registry.py never overrides importance_type in xgboost's/
# lightgbm's default_params, so their sklearn-API default labels apply here:
# XGBoost's native default is "gain", LightGBM's is "split" - neither is
# Gini/impurity reduction, which is specific to sklearn's own tree models
# (random_forest) and is what _SHAP_SUPPORTED_FAMILIES' fallback below means.
_IMPORTANCE_METHOD_BY_MODEL_NAME = {"xgboost": "gain", "lightgbm": "split"}


class FeatureImportanceEntry(BaseModel):
    feature: str
    importance: float


class ExplainabilityResult(BaseModel):
    """Standardized explainability outcome - one per model (tabular), or one
    shared "_dataset" entry for forecasting's trend/seasonality/decomposition
    (a property of the time series, not of any one candidate model)."""

    model_name: str
    problem_type: str
    supported: bool = True
    reason_unsupported: Optional[str] = None

    # tabular (classification/regression)
    feature_importance_method: Optional[str] = Field(
        default=None, description="'impurity' (tree-based) or 'coefficient' (linear), or null if unavailable."
    )
    feature_importance: list[FeatureImportanceEntry] = Field(default_factory=list)
    permutation_importance_supported: bool = False
    permutation_importance: list[FeatureImportanceEntry] = Field(default_factory=list)
    shap_supported: bool = False
    shap_unavailable_reason: Optional[str] = None
    shap_importance: list[FeatureImportanceEntry] = Field(default_factory=list)

    # forecasting
    trend_detected: Optional[bool] = None
    trend_notes: str = ""
    seasonality_detected: Optional[bool] = None
    seasonality_notes: str = ""
    forecast_components_available: bool = False
    trend_strength: Optional[float] = None
    seasonal_strength: Optional[float] = None


def _top_entries(scores: dict, n: int = MAX_IMPORTANCE_ENTRIES) -> list[FeatureImportanceEntry]:
    items = sorted(scores.items(), key=lambda kv: abs(kv[1]), reverse=True)[:n]
    return [FeatureImportanceEntry(feature=name, importance=round(float(value), 6)) for name, value in items]


def _quick_score(problem_type: ProblemType, y_true, y_pred) -> float:
    """Tiny local copy of tools/model_runner.py's scoring convention
    (higher is always better) - kept local to avoid a circular import
    between explainability and model_runner."""
    if problem_type == ProblemType.CLASSIFICATION:
        from sklearn.metrics import accuracy_score

        return float(accuracy_score(np.asarray(y_true), np.asarray(y_pred)))

    from sklearn.metrics import mean_squared_error

    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float).ravel()
    if len(y_pred_arr) != len(y_true_arr):
        y_pred_arr = y_pred_arr[: len(y_true_arr)]
    return -float(np.sqrt(mean_squared_error(y_true_arr, y_pred_arr)))


def compute_permutation_importance(
    definition: ModelDefinition,
    fitted,
    test_df: pd.DataFrame,
    target_column: str,
    time_column: Optional[str],
    problem_type: ProblemType,
    n_repeats: int = PERMUTATION_REPEATS,
    random_state: int = PERMUTATION_RANDOM_STATE,
) -> Optional[dict]:
    """Model-agnostic: shuffles one feature at a time and measures the drop
    in the same score tools/model_runner.py already computed, so it works
    for every classical registry entry regardless of the underlying
    estimator - not just sklearn ones. None (not an error) when there's no
    feature matrix to permute, or the baseline is a constant predictor for
    which permutation is meaningless (always ~0 drop).
    """
    if problem_type not in (ProblemType.CLASSIFICATION, ProblemType.REGRESSION):
        return None
    if definition.model_family == "baseline":
        return None
    try:
        feature_names = model_registry.feature_columns_for(test_df, target_column, time_column)
        if not feature_names:
            return None

        baseline_preds = definition.predict_fn(fitted, test_df, target_column, time_column)
        baseline_score = _quick_score(problem_type, test_df[target_column], baseline_preds)

        rng = np.random.default_rng(random_state)
        importances: dict[str, float] = {}
        for feature in feature_names:
            drops = []
            for _ in range(n_repeats):
                shuffled = test_df.copy()
                shuffled[feature] = rng.permutation(shuffled[feature].to_numpy())
                preds = definition.predict_fn(fitted, shuffled, target_column, time_column)
                score = _quick_score(problem_type, test_df[target_column], preds)
                drops.append(baseline_score - score)
            importances[feature] = float(np.mean(drops))
        return importances
    except Exception:  # noqa: BLE001 - explainability is supplementary, never fails the candidate
        return None


def compute_shap_importance(
    definition: ModelDefinition,
    fitted,
    test_df: pd.DataFrame,
    target_column: str,
    time_column: Optional[str],
    max_samples: int = SHAP_MAX_SAMPLES,
) -> Optional[dict]:
    """SHAP "where practical": TreeExplainer only (fast, exact) for tree
    ensembles/gradient boosting - linear models and baselines don't attempt
    SHAP here (a KernelExplainer would be slow and approximate for little
    added value over their coefficients). None when shap isn't installed or
    the model family isn't tree-based; see shap_unavailable_reason().
    """
    if definition.model_family not in _SHAP_SUPPORTED_FAMILIES:
        return None
    if not is_dependency_available("shap"):
        return None
    try:
        import shap

        model = fitted.get("model") if isinstance(fitted, dict) else fitted
        X = model_registry.feature_frame_for(test_df, target_column, time_column).iloc[:max_samples]
        if X.empty:
            return None

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        if isinstance(shap_values, list):
            shap_values = shap_values[-1]  # binary/multiclass: positive/last class
        shap_values = np.asarray(shap_values)
        if shap_values.ndim == 3:
            shap_values = shap_values[..., -1]  # some wrappers put class axis last
        mean_abs = np.abs(shap_values).mean(axis=0)
        return {col: float(value) for col, value in zip(X.columns, mean_abs)}
    except Exception:  # noqa: BLE001 - explainability is supplementary, never fails the candidate
        return None


def shap_unavailable_reason(definition: ModelDefinition) -> str:
    if not is_dependency_available("shap"):
        return "shap package is not installed"
    if definition.model_family not in _SHAP_SUPPORTED_FAMILIES:
        return f"SHAP TreeExplainer is not applicable to model family {definition.model_family!r}"
    return "SHAP could not be computed for this model"


def explain_tabular_model(
    definition: ModelDefinition,
    fitted,
    problem_type: ProblemType,
    test_df: pd.DataFrame,
    target_column: str,
    time_column: Optional[str],
    feature_importance: Optional[dict],
) -> ExplainabilityResult:
    """Classification/regression only. Combines whatever native feature
    importance tools/model_runner.py already computed with permutation
    importance and (where practical) SHAP - never mandatory, never fabricated
    when a technique doesn't apply.
    """
    result = ExplainabilityResult(model_name=definition.name, problem_type=problem_type.value)

    if feature_importance:
        if definition.name in _IMPORTANCE_METHOD_BY_MODEL_NAME:
            result.feature_importance_method = _IMPORTANCE_METHOD_BY_MODEL_NAME[definition.name]
        elif definition.model_family in _SHAP_SUPPORTED_FAMILIES:
            result.feature_importance_method = "impurity"
        else:
            result.feature_importance_method = "coefficient"
        result.feature_importance = _top_entries(feature_importance)

    permutation = compute_permutation_importance(definition, fitted, test_df, target_column, time_column, problem_type)
    if permutation is not None:
        result.permutation_importance_supported = True
        result.permutation_importance = _top_entries(permutation)

    shap_scores = compute_shap_importance(definition, fitted, test_df, target_column, time_column)
    if shap_scores is not None:
        result.shap_supported = True
        result.shap_importance = _top_entries(shap_scores)
    else:
        result.shap_unavailable_reason = shap_unavailable_reason(definition)

    if not (result.feature_importance or result.permutation_importance or result.shap_importance):
        result.supported = False
        result.reason_unsupported = (
            f"{definition.name!r} does not reliably support feature importance, permutation importance, or SHAP."
        )

    return result


def unsupported_automl_explainability(model_name: str, problem_type: ProblemType) -> ExplainabilityResult:
    """AutoGluon ensembles: explainability is deliberately not attempted -
    reliably attributing an ensemble/stacked predictor would require
    reloading the fitted predictor and can be slow; "not mandatory when the
    selected model does not support it reliably" applies here by policy,
    not by a failed attempt."""
    return ExplainabilityResult(
        model_name=model_name,
        problem_type=problem_type.value,
        supported=False,
        reason_unsupported="Explainability is not computed for AutoGluon ensemble candidates in this run.",
    )


def explain_forecast(
    train_df: pd.DataFrame, target_column: str, time_column: str, seasonal_period: int = 7
) -> ExplainabilityResult:
    """One shared result for the forecasting problem as a whole - trend and
    seasonality are properties of the historical series, not of whichever
    candidate model happens to win. Reuses tools/problem_detection.py's
    trend/seasonality detection (same numbers Phase 2 already surfaced) and
    adds a statsmodels seasonal_decompose-based trend/seasonal strength
    score "where supported" (enough historical data for at least 2 full
    periods).
    """
    from tools import problem_detection

    result = ExplainabilityResult(model_name="_dataset", problem_type=ProblemType.FORECASTING.value)

    try:
        trend_detected, trend_notes = problem_detection._detect_trend(train_df, time_column, target_column)
        seasonality_detected, seasonality_notes = problem_detection._detect_seasonality(train_df, time_column, target_column)
        result.trend_detected = trend_detected
        result.trend_notes = trend_notes
        result.seasonality_detected = seasonality_detected
        result.seasonality_notes = seasonality_notes
    except Exception:  # noqa: BLE001
        pass

    try:
        from statsmodels.tsa.seasonal import seasonal_decompose

        series = train_df.sort_values(time_column)[target_column].astype(float).reset_index(drop=True)
        if len(series) >= 2 * seasonal_period:
            decomposition = seasonal_decompose(series, model="additive", period=seasonal_period, extrapolate_trend="period")
            resid = decomposition.resid.dropna()
            if len(resid) >= 2:
                trend_plus_resid = (decomposition.trend + decomposition.resid).dropna()
                season_plus_resid = (decomposition.seasonal + decomposition.resid).dropna()
                var_resid = float(np.var(resid))
                var_trend = float(np.var(trend_plus_resid)) if len(trend_plus_resid) else 0.0
                var_season = float(np.var(season_plus_resid)) if len(season_plus_resid) else 0.0
                result.trend_strength = round(max(0.0, 1 - var_resid / var_trend), 4) if var_trend else None
                result.seasonal_strength = round(max(0.0, 1 - var_resid / var_season), 4) if var_season else None
                result.forecast_components_available = True
    except Exception:  # noqa: BLE001
        pass

    if result.trend_detected is None and not result.forecast_components_available:
        result.supported = False
        result.reason_unsupported = "Not enough historical data to compute trend/seasonality signals."

    return result
