"""Deterministic, production-oriented 3-cycle classification improvement
workflow. No LLM involved anywhere in this module - it is a standalone,
classification-only alternative to orchestration/graph.py's LLM-driven
Planner/Evaluator loop, and does not touch that pipeline at all. No AutoGluon.

Reuses existing building blocks rather than reimplementing them:
  - tools/model_registry.py: the classification ModelDefinitions (train_fn/
    predict_fn/predict_proba_fn/default_params) and feature_frame_for().
  - tools/evaluation.py: compute_classification_metrics_detailed() (added
    alongside the existing compute_classification_metrics(), which stays
    untouched and is still used by tools/model_runner.py/automl_training.py).
  - tools/splitting.py: split_train_val_test() (added alongside the existing
    split_data(), which stays untouched).
  - tools/technical_retry.py: run_with_technical_retry()/PermanentTrainingError
    for transient-vs-permanent failure handling, kept entirely separate from
    the cycle counter below.
  - tools/logging_config.py: get_logger() for structured JSON logging.

Two distinct concepts, never conflated:
  TECHNICAL RETRY   - a transient infrastructure failure (timeout, temporary
                       connection/I/O problem). Retried a few times with
                       backoff; never advances `cycle`.
  MODEL IMPROVEMENT CYCLE - training succeeded but validation metrics didn't
                       meet the configured acceptance criteria. Up to
                       MAX_CYCLES controlled, non-identical retraining
                       attempts; each one IS a cycle.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd

from agents.schemas import ProblemType
from tools import evaluation, splitting
from tools.logging_config import get_logger
from tools.model_registry import (
    feature_frame_for,
    get_registry_for_problem_type,
    is_dependency_available,
)
from tools.technical_retry import PermanentTrainingError, run_with_technical_retry

logger = get_logger(__name__)

MIN_ROWS_REQUIRED = 20
MIN_CLASSES_REQUIRED = 2

# Example defaults only - NOT assumed universally correct for every dataset.
# Callers should pass their own ClassificationCycleConfig when these don't fit.
DEFAULT_ACCEPTANCE_CRITERIA = {"accuracy": 0.70, "roc_auc": 0.70, "f1": 0.65}
DEFAULT_SELECTION_METRIC = "f1"

# Minority-class share below which the dataset is treated as imbalanced -
# drives the cycle-2/3 improvement policy (class_weight / oversampling).
IMBALANCE_MINORITY_SHARE_THRESHOLD = 0.15

# sklearn estimators that natively accept class_weight="balanced" - the
# allowlist the imbalance-driven cycle policy restricts itself to, rather
# than trying it on every model and seeing what happens.
CLASS_WEIGHT_SUPPORTED_MODELS = {"logistic_regression", "random_forest", "decision_tree", "svm"}

# Small, fixed, deterministic hyperparameter variants tried in cycles 2/3 for
# datasets that are NOT imbalanced - a controlled step, not an open-ended
# search. Models with no entry are left at their registry default_params.
HYPERPARAM_VARIANTS_CYCLE_2 = {
    "random_forest": {"n_estimators": 300, "max_depth": 12},
    "logistic_regression": {"C": 0.5},
    "xgboost": {"n_estimators": 300, "max_depth": 6},
    "lightgbm": {"n_estimators": 300, "max_depth": 6},
    "decision_tree": {"max_depth": 8},
    "svm": {"C": 2.0},
    "knn": {"n_neighbors": 7},
    "neural_network": {"hidden_layer_sizes": (100, 50)},
}
HYPERPARAM_VARIANTS_CYCLE_3 = {
    "random_forest": {"n_estimators": 500, "max_depth": 20},
    "logistic_regression": {"C": 2.0},
    "xgboost": {"n_estimators": 500, "max_depth": 8},
    "lightgbm": {"n_estimators": 500, "max_depth": 8},
    "decision_tree": {"max_depth": 4},
    "svm": {"C": 0.5},
    "knn": {"n_neighbors": 15},
    "neural_network": {"hidden_layer_sizes": (100, 50, 25)},
}


@dataclass
class ClassificationCycleConfig:
    # Defaults read from env vars (see .env.example) so a deployment can
    # override them without changing code; a caller-supplied value on
    # construction always wins over both.
    max_cycles: int = int(os.environ.get("CLASSIFICATION_CYCLE_MAX_CYCLES", "3"))
    # "metric must be >= threshold" - ALL of these must pass (AND semantics).
    # Distinct from selection_metric below, which is comparison-only.
    acceptance_criteria: dict = field(default_factory=lambda: dict(DEFAULT_ACCEPTANCE_CRITERIA))
    # "metric is used only for comparison" across candidates/cycles - never a pass/fail gate.
    selection_metric: str = DEFAULT_SELECTION_METRIC
    technical_max_attempts: int = int(os.environ.get("CLASSIFICATION_CYCLE_TECHNICAL_MAX_ATTEMPTS", "3"))
    technical_base_delay_s: float = 1.0
    multiclass_roc_strategy: str = "ovr"  # or "ovo"
    val_size: float = 0.2
    test_size: float = 0.2
    imbalance_minority_share_threshold: float = IMBALANCE_MINORITY_SHARE_THRESHOLD
    random_state: int = 42


def _validate_input(df: pd.DataFrame, target_column: str, feature_columns: list[str]) -> None:
    """Raises PermanentTrainingError for anything retrying can never fix.
    Checked once, eagerly, before any split/train - never technically
    retried, never counted as a model-improvement cycle."""
    if target_column not in df.columns:
        raise PermanentTrainingError(f"Missing target column: {target_column!r}")
    missing_features = [c for c in feature_columns if c not in df.columns]
    if missing_features:
        raise PermanentTrainingError(f"feature_columns not found in dataset: {missing_features}")
    if len(df) < MIN_ROWS_REQUIRED:
        raise PermanentTrainingError(f"Insufficient data: {len(df)} rows (need at least {MIN_ROWS_REQUIRED}).")
    y = df[target_column]
    if y.isna().all():
        raise PermanentTrainingError("Invalid target values: target column is entirely missing/NaN.")
    n_classes = y.dropna().nunique()
    if n_classes < MIN_CLASSES_REQUIRED:
        raise PermanentTrainingError(
            f"Invalid target values: found {n_classes} class(es); classification requires at least "
            f"{MIN_CLASSES_REQUIRED}."
        )


def _class_distribution(y: pd.Series) -> dict:
    counts = y.value_counts(normalize=True)
    return {str(label): round(float(pct), 4) for label, pct in counts.items()}


def _is_imbalanced(distribution: dict, threshold: float) -> bool:
    return bool(distribution) and min(distribution.values()) < threshold


def _oversample_minority(train_df: pd.DataFrame, target_column: str, random_state: int) -> pd.DataFrame:
    """Random oversampling of every non-majority class up to the majority
    class's row count, with replacement. Plain pandas - no imbalanced-learn
    dependency needed. Applied ONLY to a local copy of train_df for one
    candidate's fit call; val_df/test_df are never touched, so there is no
    leakage of validation/test information into training.
    """
    counts = train_df[target_column].value_counts()
    majority_n = int(counts.max())
    parts = []
    for label, group in train_df.groupby(target_column):
        if len(group) < majority_n:
            group = group.sample(n=majority_n, replace=True, random_state=random_state)
        parts.append(group)
    return pd.concat(parts).sample(frac=1.0, random_state=random_state).reset_index(drop=True)


def _available_classification_models() -> dict:
    registry = get_registry_for_problem_type(ProblemType.CLASSIFICATION)
    return {
        d.name: d
        for d in registry
        if not d.required_dependencies or all(is_dependency_available(dep) for dep in d.required_dependencies)
    }


def _best_names_so_far(records: list[dict], selection_metric: str, top_n: int) -> list[str]:
    """Unique model names from every completed record so far, ranked by
    their own best validation `selection_metric` value, highest first."""
    best_by_name: dict[str, float] = {}
    for r in records:
        if r.get("status") != "completed":
            continue
        value = r.get("validation_metrics", {}).get(selection_metric)
        if value is None:
            continue
        if r["model_name"] not in best_by_name or value > best_by_name[r["model_name"]]:
            best_by_name[r["model_name"]] = value
    ranked = sorted(best_by_name, key=lambda name: best_by_name[name], reverse=True)
    return ranked[:top_n]


def _plan_cycle(
    cycle: int,
    available_models: dict,
    previous_records: list[dict],
    imbalanced: bool,
    selection_metric: str,
) -> tuple[list[dict], str]:
    """Deterministic, rule-based improvement policy - never retrains the
    identical configuration, never applies every possible technique blindly.
    Returns (candidates, technique_label); each candidate is
    {"model_name": ..., "extra_params": ..., "oversample": bool}.
    """
    if cycle == 1:
        candidates = [{"model_name": name, "extra_params": {}, "oversample": False} for name in available_models]
        return candidates, "baseline"

    if imbalanced:
        supported = [name for name in available_models if name in CLASS_WEIGHT_SUPPORTED_MODELS]
        if not supported:
            supported = list(available_models)  # fall back below rather than run zero candidates
        else:
            oversample = cycle == 3  # cycle 2: class_weight only; cycle 3: class_weight + oversampling
            return (
                [
                    {"model_name": name, "extra_params": {"class_weight": "balanced"}, "oversample": oversample}
                    for name in supported
                ],
                "class_weight_balanced_plus_oversample" if oversample else "class_weight_balanced",
            )

    variants = HYPERPARAM_VARIANTS_CYCLE_2 if cycle == 2 else HYPERPARAM_VARIANTS_CYCLE_3
    top_names = _best_names_so_far(previous_records, selection_metric, top_n=3) or list(available_models)[:3]
    candidates = [
        {"model_name": name, "extra_params": dict(variants.get(name, {})), "oversample": False}
        for name in top_names
        if name in available_models
    ]
    return candidates, f"hyperparameter_variant_cycle_{cycle}"


def _full_predict_proba(fitted: Any, df: pd.DataFrame, target_column: str) -> Optional[np.ndarray]:
    model = fitted.get("model") if isinstance(fitted, dict) else fitted
    if model is None or not hasattr(model, "predict_proba"):
        return None
    X = feature_frame_for(df, target_column, None)
    if isinstance(fitted, dict) and "scaler" in fitted:
        X = fitted["scaler"].transform(X)
    return model.predict_proba(X)


def _meets_acceptance(metrics: dict, acceptance_criteria: dict) -> bool:
    """ALL configured thresholds must be satisfied (AND semantics) - a
    missing/uncomputable metric counts as not-met rather than being ignored,
    and a single passing metric never overrides a failing required one."""
    if not acceptance_criteria:
        return False
    for metric_name, threshold in acceptance_criteria.items():
        value = metrics.get(metric_name)
        if value is None or value < threshold:
            return False
    return True


def _train_one_candidate(
    definition,
    extra_params: dict,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    target_column: str,
    multiclass_roc_strategy: str,
) -> tuple[dict, Any]:
    """Fits + predicts + scores exactly one candidate on VALIDATION data.
    Returns (validation_metrics, fitted_model) - the whole thing runs inside
    one technical-retry wrapper at the call site, so a transient failure
    here retries this exact attempt without touching the cycle counter.
    """
    params = dict(definition.default_params)
    params.update(extra_params)
    fitted = definition.train_fn(train_df, target_column, None, **params)
    predictions = definition.predict_fn(fitted, val_df, target_column, None)
    y_proba = _full_predict_proba(fitted, val_df, target_column)
    metrics = evaluation.compute_classification_metrics_detailed(
        val_df[target_column], predictions, y_proba=y_proba, multiclass_roc_strategy=multiclass_roc_strategy
    )
    return metrics, fitted


def run_classification_cycles(
    df: pd.DataFrame,
    target_column: str,
    feature_columns: Optional[list[str]] = None,
    config: Optional[ClassificationCycleConfig] = None,
    job_id: Optional[str] = None,
) -> dict:
    """Main entry point. See module docstring. Returns a JSON-safe result
    dict shaped per the section-11 contract (status/completed_cycles/
    best_model/best_cycle/selection_metric/validation_metrics/
    acceptance_criteria_met/test_metrics or message), plus the full `cycles`
    list (section-10 shape) for audit/debugging.
    """
    config = config or ClassificationCycleConfig()
    feature_columns = feature_columns if feature_columns is not None else [c for c in df.columns if c != target_column]
    log_ctx = {"job_id": job_id}

    _validate_input(df, target_column, feature_columns)

    class_distribution = _class_distribution(df[target_column])
    imbalanced = _is_imbalanced(class_distribution, config.imbalance_minority_share_threshold)
    logger.info(
        "classification_cycle_started",
        extra={**log_ctx, "class_distribution": class_distribution, "imbalanced": imbalanced},
    )

    working_df = df[feature_columns + [target_column]]
    train_df, val_df, test_df, split_log = splitting.split_train_val_test(
        working_df, target_column, val_size=config.val_size, test_size=config.test_size, random_state=config.random_state
    )
    logger.info("classification_cycle_split", extra={**log_ctx, **split_log})

    available_models = _available_classification_models()
    all_records: list[dict] = []
    fitted_by_key: dict[tuple, Any] = {}
    acceptance_met = False
    completed_cycles = 0

    for cycle in range(1, config.max_cycles + 1):
        completed_cycles = cycle
        candidates, technique = _plan_cycle(
            cycle, available_models, all_records, imbalanced, config.selection_metric
        )
        logger.info(
            "classification_cycle_plan",
            extra={**log_ctx, "cycle": cycle, "technique": technique, "candidates": [c["model_name"] for c in candidates]},
        )

        for candidate in candidates:
            model_name = candidate["model_name"]
            definition = available_models[model_name]
            cycle_train_df = (
                _oversample_minority(train_df, target_column, config.random_state)
                if candidate["oversample"]
                else train_df
            )
            configuration = {**candidate["extra_params"], "technique": technique}
            start = time.perf_counter()
            try:
                metrics, fitted = run_with_technical_retry(
                    _train_one_candidate,
                    definition,
                    candidate["extra_params"],
                    cycle_train_df,
                    val_df,
                    target_column,
                    config.multiclass_roc_strategy,
                    max_attempts=config.technical_max_attempts,
                    base_delay_s=config.technical_base_delay_s,
                )
            except Exception as exc:  # noqa: BLE001 - one bad candidate must not sink the cycle/run
                from tools.technical_retry import is_retryable_error

                record = {
                    "cycle": cycle,
                    "model_name": model_name,
                    "configuration": configuration,
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "retryable": is_retryable_error(exc),
                }
                all_records.append(record)
                logger.warning("classification_cycle_candidate_failed", extra={**log_ctx, **record})
                continue

            duration = round(time.perf_counter() - start, 3)
            met = _meets_acceptance(metrics, config.acceptance_criteria)
            record = {
                "cycle": cycle,
                "model_name": model_name,
                "configuration": configuration,
                "training_duration_seconds": duration,
                "validation_metrics": metrics,
                "acceptance_criteria_met": met,
                "status": "completed",
            }
            all_records.append(record)
            fitted_by_key[(cycle, model_name)] = fitted
            logger.info(
                "classification_cycle_candidate_done",
                extra={**log_ctx, "cycle": cycle, "model_name": model_name, "acceptance_criteria_met": met},
            )
            if met:
                acceptance_met = True

        if acceptance_met:
            break

    completed = [r for r in all_records if r["status"] == "completed"]
    scored = [r for r in completed if r["validation_metrics"].get(config.selection_metric) is not None]
    ranking_pool = scored or completed

    if not ranking_pool:
        logger.warning("classification_cycle_no_successful_candidate", extra={**log_ctx})
        return {
            "status": "threshold_not_met",
            "completed_cycles": completed_cycles,
            "best_model": None,
            "best_cycle": None,
            "selection_metric": config.selection_metric,
            "validation_metrics": {},
            "acceptance_criteria_met": False,
            "class_distribution": class_distribution,
            "message": "No candidate model trained successfully across the configured improvement cycles.",
            "cycles": all_records,
        }

    def _sort_key(r: dict) -> float:
        return r["validation_metrics"].get(config.selection_metric, r["validation_metrics"].get("accuracy", 0.0)) or 0.0

    best_record = max(ranking_pool, key=_sort_key)
    best_model = best_record["model_name"]
    best_cycle = best_record["cycle"]

    result: dict = {
        "completed_cycles": completed_cycles,
        "best_model": best_model,
        "best_cycle": best_cycle,
        "selection_metric": config.selection_metric,
        "validation_metrics": best_record["validation_metrics"],
        "acceptance_criteria_met": best_record["acceptance_criteria_met"],
        "class_distribution": class_distribution,
        "cycles": all_records,
    }

    if not best_record["acceptance_criteria_met"]:
        result["status"] = "threshold_not_met"
        result["message"] = (
            "No candidate model met the configured acceptance criteria after "
            f"{completed_cycles} improvement cycle(s)."
        )
        logger.info("classification_cycle_threshold_not_met", extra={**log_ctx, "best_model": best_model, "best_cycle": best_cycle})
        return result

    # Final TEST evaluation - once, on the fixed test set, using the already-
    # fitted best model (no retraining). Never feeds back into cycle/acceptance logic.
    fitted_best = fitted_by_key[(best_cycle, best_model)]
    definition = available_models[best_model]
    test_predictions = definition.predict_fn(fitted_best, test_df, target_column, None)
    test_proba = _full_predict_proba(fitted_best, test_df, target_column)
    test_metrics = evaluation.compute_classification_metrics_detailed(
        test_df[target_column], test_predictions, y_proba=test_proba, multiclass_roc_strategy=config.multiclass_roc_strategy
    )

    result["status"] = "success"
    result["test_metrics"] = test_metrics
    logger.info(
        "classification_cycle_success",
        extra={**log_ctx, "best_model": best_model, "best_cycle": best_cycle, "completed_cycles": completed_cycles},
    )
    return result
