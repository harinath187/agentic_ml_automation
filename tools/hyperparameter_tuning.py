"""Bounded, train-only hyperparameter tuning for classical model definitions."""
from __future__ import annotations

import time
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, ParameterGrid, StratifiedKFold

from agents.schemas import ProblemType, ValidationStrategyType
from tools.model_registry import ModelDefinition


def _score(problem_type: ProblemType, y_true, predictions, metric: str) -> float:
    from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score

    if problem_type == ProblemType.CLASSIFICATION:
        if metric == "accuracy":
            return float(accuracy_score(y_true, predictions))
        return float(f1_score(y_true, predictions, average="weighted", zero_division=0))
    if problem_type == ProblemType.REGRESSION:
        if metric == "mae":
            return -float(mean_absolute_error(y_true, predictions))
        if metric == "r2":
            return float(r2_score(y_true, predictions))
        return -float(np.sqrt(mean_squared_error(y_true, predictions)))
    raise ValueError(f"Tuning is not supported for {problem_type.value}.")


def tune_model(
    definition: ModelDefinition,
    train_df: pd.DataFrame,
    target_column: str,
    time_column: Optional[str],
    problem_type: ProblemType,
    validation_strategy: ValidationStrategyType = ValidationStrategyType.STRATIFIED_K_FOLD,
    folds: int = 3,
    max_trials: int = 24,
    random_state: int = 42,
    metric: Optional[str] = None,
) -> dict:
    """Search a model's declared parameter grid using train_df only.

    The returned parameters are intended for one final fit on all of train_df;
    test data is deliberately not accepted by this API.
    """
    started = time.perf_counter()
    if not definition.tuning_space:
        return {"model_name": definition.name, "status": "skipped", "reason": "no tuning space"}
    if problem_type not in (ProblemType.CLASSIFICATION, ProblemType.REGRESSION):
        return {"model_name": definition.name, "status": "skipped", "reason": "unsupported problem type"}

    candidates = list(ParameterGrid(definition.tuning_space))[:max_trials]
    metric = metric or ("f1" if problem_type == ProblemType.CLASSIFICATION else "rmse")
    y = train_df[target_column]
    if problem_type == ProblemType.CLASSIFICATION and validation_strategy == ValidationStrategyType.STRATIFIED_K_FOLD:
        folds = min(folds, int(y.value_counts().min()))
        splitter = StratifiedKFold(n_splits=max(2, folds), shuffle=True, random_state=random_state)
        split_iter = splitter.split(train_df, y)
    else:
        splitter = KFold(n_splits=max(2, folds), shuffle=True, random_state=random_state)
        split_iter = splitter.split(train_df)

    best_score = float("-inf")
    best_params: dict = {}
    completed_trials = 0
    try:
        split_indices = list(split_iter)
        for params in candidates:
            fold_scores = []
            for train_indices, validation_indices in split_indices:
                fitted = definition.train_fn(
                    train_df.iloc[train_indices], target_column, time_column, **params
                )
                predictions = definition.predict_fn(
                    fitted, train_df.iloc[validation_indices], target_column, time_column
                )
                fold_scores.append(
                    _score(problem_type, y.iloc[validation_indices], predictions, metric)
                )
            score = float(np.mean(fold_scores))
            completed_trials += 1
            if score > best_score:
                best_score = score
                best_params = params
    except Exception as exc:  # noqa: BLE001 - tuning is supplementary per candidate
        return {
            "model_name": definition.name,
            "status": "failed",
            "error": str(exc),
            "trials_run": completed_trials,
            "duration_seconds": round(time.perf_counter() - started, 3),
        }

    return {
        "model_name": definition.name,
        "status": "success",
        "best_params": best_params,
        "best_score": round(best_score, 6),
        "metric": metric,
        "trials_run": completed_trials,
        "duration_seconds": round(time.perf_counter() - started, 3),
    }


def tune_models(
    definitions: list[ModelDefinition],
    train_df: pd.DataFrame,
    target_column: str,
    time_column: Optional[str],
    problem_type: ProblemType,
    validation_strategy: Optional[ValidationStrategyType] = None,
    folds: int = 3,
    max_trials: int = 24,
    metric: Optional[str] = None,
) -> tuple[dict[str, dict], dict[str, dict]]:
    results: dict[str, dict] = {}
    params: dict[str, dict] = {}
    for definition in definitions:
        result = tune_model(
            definition, train_df, target_column, time_column, problem_type,
            validation_strategy=validation_strategy or ValidationStrategyType.K_FOLD,
            folds=folds, max_trials=max_trials, metric=metric,
        )
        results[definition.name] = result
        if result.get("status") == "success":
            params[definition.name] = result["best_params"]
    return params, results