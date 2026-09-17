"""Deterministic Model Validation and Evaluation Engine (Phase 4).

Computes problem-specific metrics and ranks trained candidates. This is the
ONLY place a "winner" is decided - agents/evaluator.py is handed the result
of build_model_comparison() as ground truth and may only decide whether to
retry and narrate why; it is never allowed to name a different model as
best. That override is enforced in agents/evaluator.py, not here, but this
module is what makes the enforcement meaningful: every number it produces is
plain pandas/numpy/sklearn arithmetic, never an LLM call.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from pydantic import BaseModel, Field
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

from agents.schemas import ProblemType

# Metrics where a LOWER value is better (everything else is higher-is-better).
LOWER_IS_BETTER_METRICS = {"rmse", "mae", "mape", "smape", "mase"}

# AutoGluon's own eval_metric names, mapped to ours so a leaderboard row and
# a classical candidate can be ranked on the identical metric key.
_AUTOGLUON_METRIC_ALIASES = {
    "root_mean_squared_error": "rmse",
    "mean_absolute_error": "mae",
    "mean_absolute_percentage_error": "mape",
    "symmetric_mean_absolute_percentage_error": "smape",
}


class EvaluationResult(BaseModel):
    """Standardized per-model evaluation outcome. Every candidate produces
    exactly one of these, success or not - "preserve all model results"."""

    model_name: str
    problem_type: str
    status: str
    validation_strategy: str = "train_test_split"
    metrics: dict = Field(default_factory=dict)
    training_time: Optional[float] = None
    prediction_time: Optional[float] = None
    errors: Optional[str] = None
    feature_importance: Optional[dict] = Field(
        default=None,
        description="Top feature -> importance score (classification/regression only). "
        "Feature names and relative magnitudes only - never raw data rows.",
    )
    explainability: Optional[dict] = Field(
        default=None,
        description="tools/explainability.py's ExplainabilityResult, as a dict - feature/permutation/SHAP "
        "importance, or forecasting trend/seasonality signals. Same safety rationale as feature_importance.",
    )


class ModelComparison(BaseModel):
    """The deterministic ranking engine's output: which metric it judged by,
    every candidate's result (successful or not), the ranking, and the
    winner. agents/evaluator.py treats `winner` as final."""

    problem_type: str
    primary_metric: Optional[str] = None
    higher_is_better: bool = True
    results: list[EvaluationResult] = Field(default_factory=list)
    ranked_model_names: list[str] = Field(default_factory=list)
    winner: Optional[str] = None
    winner_reasoning: str = ""
    dataset_explainability: Optional[dict] = Field(
        default=None,
        description="A property of the dataset/run as a whole, not of any one candidate model, so it "
        "lives here rather than on a per-model EvaluationResult. Shape depends on problem_type: "
        "forecasting gets tools/explainability.py's explain_forecast() (trend/seasonality/decomposition, "
        "an ExplainabilityResult as a dict); classification/regression get explain_dataset_consensus() "
        "(a cross-model feature-importance consensus ranking, {num_models_aggregated, consensus_ranking}). "
        "Null when there weren't enough successfully-explained candidates to compute either (forecasting: "
        "insufficient history; classification/regression: fewer than 2 candidates with usable importance).",
    )


def normalize_autogluon_metric(eval_metric_name: str, score_test: float) -> tuple[str, float]:
    """AutoGluon's leaderboard score_test is always higher-is-better (it
    flips the sign of loss-like metrics internally); this recovers our own
    metric name/sign convention so AutoGluon and classical candidates land
    on the same primary_metric key and can be ranked together honestly.
    """
    key = _AUTOGLUON_METRIC_ALIASES.get(eval_metric_name, eval_metric_name)
    value = -score_test if key in LOWER_IS_BETTER_METRICS else score_test
    return key, float(value)


def _smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.abs(y_true) + np.abs(y_pred)
    denom = np.where(denom == 0, 1e-9, denom)
    return float(np.mean(2.0 * np.abs(y_pred - y_true) / denom) * 100)


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> Optional[float]:
    if len(y_true) == 0 or np.any(y_true == 0):
        return None  # undefined/unstable with zeros in y_true - "where appropriate"
    return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)


def _mase(y_true: np.ndarray, y_pred: np.ndarray, y_train: Optional[np.ndarray], seasonal_period: int = 1) -> Optional[float]:
    """MASE: MAE scaled by the in-sample seasonal-naive MAE. Requires the
    training series (to compute the naive baseline's error) - "where
    possible" per the spec, so this returns None rather than a misleading
    number when the training series is missing or too short.
    """
    if y_train is None or len(y_train) <= seasonal_period:
        return None
    naive_errors = np.abs(y_train[seasonal_period:] - y_train[:-seasonal_period])
    scale = float(naive_errors.mean())
    if scale == 0:
        return None
    mae = float(np.mean(np.abs(y_true - y_pred)))
    return mae / scale


def compute_classification_metrics(y_true, y_pred, y_proba=None) -> dict:
    """Accuracy/precision/recall/F1 always; ROC-AUC only when binary and
    probability scores are available ("where applicable")."""
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    metrics = {
        "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
        "precision": float(precision_score(y_true_arr, y_pred_arr, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_true_arr, y_pred_arr, average="weighted", zero_division=0)),
        "f1": float(f1_score(y_true_arr, y_pred_arr, average="weighted", zero_division=0)),
    }
    if y_proba is not None and len(np.unique(y_true_arr)) == 2:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true_arr, y_proba))
        except ValueError:
            pass  # e.g. degenerate probability vector - leave roc_auc unset rather than fail the candidate
    return metrics


def compute_classification_metrics_detailed(
    y_true,
    y_pred,
    y_proba=None,
    multiclass_roc_strategy: str = "ovr",
) -> dict:
    """Richer classification metric set for tools/classification_cycle.py -
    additive alongside compute_classification_metrics() above (which stays
    exactly as-is for tools/model_runner.py/tools/automl_training.py; this
    function is never called from either of those).

    Adds confusion_matrix, per-class precision/recall/f1, PR-AUC, and a
    configurable multiclass ROC-AUC strategy (One-vs-Rest by default, or
    One-vs-One) on top of the same accuracy/precision/recall/f1 base as
    compute_classification_metrics(). `y_proba` for multiclass must be the
    FULL per-class probability matrix (n_samples, n_classes) in the same
    class order as `y_true`'s sorted unique labels - binary keeps accepting
    either a 1-D positive-class-probability vector or a (n, 2) matrix.

    Every metric that can fail for a data-shape reason (e.g. a class with
    zero validation examples, a degenerate probability vector) is wrapped
    defensively and simply omitted rather than raising - a metrics glitch
    must never crash the candidate's whole result (same policy already used
    throughout tools/model_runner.py).
    """
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    labels = sorted(np.unique(y_true_arr).tolist())

    metrics: dict = {
        "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
        "precision": float(precision_score(y_true_arr, y_pred_arr, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_true_arr, y_pred_arr, average="weighted", zero_division=0)),
        "f1": float(f1_score(y_true_arr, y_pred_arr, average="weighted", zero_division=0)),
    }

    try:
        metrics["confusion_matrix"] = confusion_matrix(y_true_arr, y_pred_arr, labels=labels).tolist()
    except ValueError:
        pass

    try:
        p, r, f, _ = precision_recall_fscore_support(y_true_arr, y_pred_arr, labels=labels, average=None, zero_division=0)
        metrics["per_class"] = {
            str(label): {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i])}
            for i, label in enumerate(labels)
        }
    except ValueError:
        pass

    is_binary = len(labels) == 2
    if y_proba is not None:
        proba_arr = np.asarray(y_proba)
        if is_binary:
            # Accept either a 1-D positive-class vector or an (n, 2) matrix.
            positive_proba = proba_arr[:, 1] if proba_arr.ndim == 2 else proba_arr
            try:
                metrics["roc_auc"] = float(roc_auc_score(y_true_arr, positive_proba))
            except ValueError:
                pass
            try:
                metrics["pr_auc"] = float(average_precision_score(y_true_arr, positive_proba))
            except ValueError:
                pass
        elif proba_arr.ndim == 2 and proba_arr.shape[1] == len(labels):
            try:
                metrics["roc_auc"] = float(
                    roc_auc_score(y_true_arr, proba_arr, labels=labels, multi_class=multiclass_roc_strategy, average="macro")
                )
            except ValueError:
                pass  # e.g. a class missing entirely from this validation split
            try:
                y_true_binarized = np.stack([(y_true_arr == label).astype(int) for label in labels], axis=1)
                metrics["pr_auc"] = float(average_precision_score(y_true_binarized, proba_arr, average="macro"))
            except ValueError:
                pass

    return metrics


def compute_regression_metrics(y_true, y_pred) -> dict:
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    return {
        "mae": float(mean_absolute_error(y_true_arr, y_pred_arr)),
        "rmse": float(np.sqrt(mean_squared_error(y_true_arr, y_pred_arr))),
        "r2": float(r2_score(y_true_arr, y_pred_arr)) if len(y_true_arr) >= 2 else None,
        "mape": _mape(y_true_arr, y_pred_arr),
        "smape": _smape(y_true_arr, y_pred_arr),
    }


def compute_forecasting_metrics(y_true, y_pred, y_train=None, seasonal_period: int = 7) -> dict:
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    y_train_arr = np.asarray(y_train, dtype=float) if y_train is not None else None
    return {
        "mae": float(mean_absolute_error(y_true_arr, y_pred_arr)),
        "rmse": float(np.sqrt(mean_squared_error(y_true_arr, y_pred_arr))),
        "smape": _smape(y_true_arr, y_pred_arr),
        "mase": _mase(y_true_arr, y_pred_arr, y_train_arr, seasonal_period),
    }


def compute_metrics(
    problem_type: ProblemType, y_true, y_pred, y_proba=None, y_train=None, seasonal_period: int = 7
) -> dict:
    """Single entry point tools/model_runner.py calls per candidate - picks
    the right metric set for the problem type, deterministically."""
    if problem_type == ProblemType.CLASSIFICATION:
        return compute_classification_metrics(y_true, y_pred, y_proba)
    if problem_type == ProblemType.FORECASTING:
        return compute_forecasting_metrics(y_true, y_pred, y_train, seasonal_period)
    return compute_regression_metrics(y_true, y_pred)


def select_primary_metric(
    problem_type: ProblemType, results: list[EvaluationResult], preferred_metrics: Optional[list[str]] = None
) -> tuple[str, bool]:
    """Deterministic - never asks the LLM.

    preferred_metrics (plan.evaluation_metrics, in priority order) is tried
    first: the first named metric that EVERY successfully-scored candidate
    actually reports wins, so a candidate missing it (e.g. a multiclass
    classifier lacking roc_auc) can't get unfairly excluded from ranking on a
    metric it was never able to produce. This is the single source of truth
    for "which metric did we rank on" - tools/model_runner.py's top-level
    metrics["eval_metric"] is set FROM this function's result (via
    ModelComparison.primary_metric), never computed independently, so the two
    can no longer disagree.

    If preferred_metrics is empty/unset, or none of its entries are usable,
    falls back to the previous default policy: classification prefers
    ROC-AUC only when every successfully-scored candidate reports it,
    otherwise accuracy; regression and forecasting always rank on RMSE,
    since it's the one metric every candidate here can always produce
    (MAPE/MASE are situational - "where appropriate"/"where possible" - so
    they can't be relied on as the ranking metric).
    """
    successful_metrics = [r.metrics for r in results if r.status == "success" and r.metrics]
    if successful_metrics and preferred_metrics:
        for raw_name in preferred_metrics:
            key = raw_name.strip().lower()
            if key and all(m.get(key) is not None for m in successful_metrics):
                return key, key not in LOWER_IS_BETTER_METRICS
    if problem_type == ProblemType.CLASSIFICATION:
        if successful_metrics and all(m.get("roc_auc") is not None for m in successful_metrics):
            return "roc_auc", True
        return "accuracy", True
    if problem_type in (ProblemType.REGRESSION, ProblemType.FORECASTING):
        return "rmse", False
    return "score", True


def build_model_comparison(
    problem_type: ProblemType, results: list[EvaluationResult], preferred_metrics: Optional[list[str]] = None
) -> ModelComparison:
    """The ranking engine itself:
    1. select_primary_metric() picks the metric (honoring preferred_metrics -
       normally plan.evaluation_metrics - ahead of the problem-type default)
    2. successfully-scored candidates are sorted by it
    3. failed candidates are kept in `results` but never ranked
    4. candidates missing the primary metric (even if status == "success")
       are likewise excluded from ranking, not crashed on
    5. ranked_model_names[0] is the winner
    6. `results` always equals the full input list - nothing is dropped
    """
    primary_metric, higher_is_better = select_primary_metric(problem_type, results, preferred_metrics)

    scored = [r for r in results if r.status == "success" and r.metrics.get(primary_metric) is not None]
    ranked = sorted(scored, key=lambda r: r.metrics[primary_metric], reverse=higher_is_better)
    ranked_names = [r.model_name for r in ranked]

    winner = ranked_names[0] if ranked_names else None
    if winner:
        direction = "highest" if higher_is_better else "lowest"
        winner_reasoning = (
            f"{winner!r} has the {direction} {primary_metric} "
            f"({ranked[0].metrics[primary_metric]:.4f}) among {len(ranked)} successfully "
            f"evaluated candidate(s) out of {len(results)} attempted."
        )
    else:
        winner_reasoning = (
            f"No candidate produced a usable {primary_metric!r} value out of "
            f"{len(results)} attempted; no winner can be determined."
        )

    return ModelComparison(
        problem_type=problem_type.value,
        primary_metric=primary_metric,
        higher_is_better=higher_is_better,
        results=list(results),
        ranked_model_names=ranked_names,
        winner=winner,
        winner_reasoning=winner_reasoning,
    )


def to_llm_summary(comparison: ModelComparison, top_n_features: int = 5) -> dict:
    """Lossy, compact view of a ModelComparison for LLM prompts only (Phase 9
    413-fix) - agents/evaluator.py, agents/recommender.py, and
    agents/reporter.py should send this instead of the full object's JSON.

    Drops everything an agent's narrative doesn't need to read: per-model
    training/prediction time, error strings, validation_strategy labels, and
    the full feature/permutation/SHAP importance score lists (already capped
    at MAX_IMPORTANCE_ENTRIES by tools/explainability.py, but still 3 lists x
    10 entries x every candidate) collapse to a single flat `top_features`
    name list of at most `top_n_features` entries, taken from whichever of
    shap/permutation/native importance is populated first - the same
    preference order agents/recommender.py already explains to the LLM.

    Returns a brand-new dict; `comparison` itself is never mutated, and
    nothing here is a substitute for validating against the full object -
    see agents/recommender.py's validate_recommendation, which must keep
    checking claims against the untrimmed ModelComparison, not this summary.

    dataset_explainability (forecasting-only trend/seasonality/strength
    scalars, never a list) is passed through as-is when present: it is
    already small and agents/recommender.py's system prompt directs the LLM
    to use it directly for forecasting explanations.
    """
    results_summary = []
    for result in comparison.results:
        results_summary.append(
            {
                "model_name": result.model_name,
                "metrics": result.metrics,
                "top_features": _top_feature_names(result.explainability, top_n_features),
            }
        )

    summary = {
        "winner": comparison.winner,
        "winner_reasoning": comparison.winner_reasoning,
        "results": results_summary,
    }
    if comparison.dataset_explainability:
        summary["dataset_explainability"] = comparison.dataset_explainability
    return summary


def _top_feature_names(explainability: Optional[dict], top_n: int) -> list[str]:
    """First non-empty of shap/permutation/native importance, names only."""
    if not explainability:
        return []
    for key in ("shap_importance", "permutation_importance", "feature_importance"):
        entries = explainability.get(key)
        if entries:
            return [e["feature"] for e in entries[:top_n] if isinstance(e, dict) and e.get("feature")]
    return []


def build_comparison_from_score_test(problem_type: ProblemType, models: dict) -> ModelComparison:
    """Fallback ranking for pipeline paths that bypass tools/model_runner.py
    entirely (the per_entity/hierarchical scope strategies still call
    tools/automl_training.py directly - see orchestration/graph.py) and only
    have the old flat {"score_test": ...} shape, already higher-is-better.
    Still 100% deterministic, still never via the LLM.
    """
    results = [
        EvaluationResult(
            model_name=name,
            problem_type=problem_type.value,
            status="success" if scores.get("score_test") is not None else "failed",
            metrics={"score_test": scores["score_test"]} if scores.get("score_test") is not None else {},
            training_time=scores.get("fit_time_s"),
            errors=None if scores.get("score_test") is not None else "no score_test reported",
        )
        for name, scores in models.items()
    ]
    scored = [r for r in results if r.metrics.get("score_test") is not None]
    ranked = sorted(scored, key=lambda r: r.metrics["score_test"], reverse=True)
    ranked_names = [r.model_name for r in ranked]
    winner = ranked_names[0] if ranked_names else None
    winner_reasoning = (
        f"{winner!r} has the highest score_test ({ranked[0].metrics['score_test']:.4f}) among "
        f"{len(ranked)} candidate(s)."
        if winner
        else "No candidate reported a usable score_test; no winner can be determined."
    )
    return ModelComparison(
        problem_type=problem_type.value,
        primary_metric="score_test",
        higher_is_better=True,
        results=results,
        ranked_model_names=ranked_names,
        winner=winner,
        winner_reasoning=winner_reasoning,
    )
