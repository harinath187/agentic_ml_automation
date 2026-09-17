"""Runs a list of resolved ModelDefinitions (tools/model_registry.py) against
an already-cleaned/engineered/split train/test frame.

Three things come out of every run (run_candidates):
  1. metrics (dict) - shaped exactly like the pre-Phase-3 output of
     tools/automl_training.train_models(), so agents.reporter and the
     frontend need no changes.
  2. metrics["model_comparison"] - the Phase 4 deterministic ranking engine's
     output (tools/evaluation.py): every candidate's full metric breakdown,
     ranked, with a winner. agents/evaluator.py treats this as ground truth
     and is not allowed to name a different winner.
  3. chart_data (dict, returned SEPARATELY - never merged into `metrics`) -
     Phase 6 report charts: {model_name: {"actual": [...], "predicted": [...]}}.
     Kept out of `metrics` deliberately, since `metrics` and
     metrics["model_comparison"] are both JSON-dumped into LLM prompts
     (agents/evaluator.py, agents/reporter.py, agents/recommender.py) and
     these are real observed target values - the same "never send raw data
     to an LLM" rule data_ingestion/loader.py already enforces for the
     Planner. tools/report_charts.py is the only consumer.

Every candidate produces a result, success or not - a missing dependency, an
unsupported validation strategy, or a training exception never aborts the
run; it just narrows metrics["models"]/the ranking to whatever actually
trained.

Note: `_run_automl`/the `model_family == "automl"` branch below has no
candidates to run against currently - tools/model_registry.py's REGISTRY
registers no `automl`-family ModelDefinition for classification/regression
(see that module's docstring), so `resolve_candidates()`/`default_candidates()`
never hand this function one. AutoGluon only trains via the separate
per_entity (tools/automl_training.train_models, called directly per entity
by orchestration/graph.py) and hierarchical (train_hierarchical_timeseries)
scope strategies, which bypass run_candidates entirely. The branch is kept
so a future `automl`-family registry entry (e.g. for the pooled path) would
work without further changes here.
"""
from __future__ import annotations

import time
from typing import Optional

import numpy as np
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold

from agents.schemas import ProblemType, ValidationStrategyType
from tools import evaluation, explainability
from tools.model_registry import ModelDefinition, ModelResult, feature_columns_for, is_dependency_available

CHART_SAMPLE_CAP = 200

DEFAULT_CV_FOLDS = 5
# Skip cross-validation (fall back to the single train/test split) if a fold
# would average out to fewer than this many training rows - same style of
# minimum-rows guard as orchestration/graph.py's MIN_ROWS_PER_ENTITY.
MIN_ROWS_PER_FOLD = 10
_CV_STRATEGIES = (ValidationStrategyType.K_FOLD, ValidationStrategyType.STRATIFIED_K_FOLD)


def run_candidates(
    candidates: list[ModelDefinition],
    train_df,
    test_df,
    target_column: str,
    time_column: Optional[str],
    problem_type: ProblemType,
    validation_strategy: Optional[ValidationStrategyType] = None,
    validation_folds: Optional[int] = None,
    evaluation_metrics: Optional[list[str]] = None,
) -> tuple[dict, dict]:
    """Returns (metrics, chart_data) - see module docstring for why chart_data
    is a separate return value rather than a key inside metrics.

    evaluation_metrics is the validated ExperimentPlan's evaluation_metrics
    (LLM-stated priority order, e.g. ['roc_auc', 'f1']) - passed straight
    through to tools/evaluation.py's select_primary_metric() so the metric
    actually used to rank/pick a winner is plan-driven rather than an
    independently hardcoded default.
    """
    models: dict[str, dict] = {}
    model_results: list[ModelResult] = []
    autogluon_best_model: Optional[str] = None
    autogluon_model_path: Optional[str] = None

    for definition in candidates:
        if definition.model_family == "automl":
            results, best_model, model_path = _run_automl(
                definition, train_df, test_df, target_column, time_column, problem_type
            )
            for result in results:
                model_results.append(result)
                if result.status == "success":
                    models[result.model_name] = _score_row(result)
            if best_model:
                autogluon_best_model = best_model
                autogluon_model_path = model_path
            continue

        result = run_model(
            definition, train_df, test_df, target_column, time_column, problem_type,
            validation_strategy, validation_folds,
        )
        model_results.append(result)
        if result.status == "success":
            models[result.model_name] = _score_row(result)

    comparison = evaluation.build_model_comparison(
        problem_type, [_to_evaluation_result(r, problem_type) for r in model_results], evaluation_metrics
    )

    # comparison.dataset_explainability (inside metrics["model_comparison"]
    # below) is the ONE place this lives - do not also copy it into a
    # top-level metrics["explainability"] key. That used to exist and was
    # dead weight: nothing (agents/recommender.py, reports/template.html,
    # or any test) ever read it, only model_comparison.dataset_explainability.
    if problem_type == ProblemType.FORECASTING and time_column:
        # Trend/seasonality/decomposition are properties of the historical
        # series, not of any one candidate model - one shared entry rather
        # than duplicating it per forecasting candidate. Lives on
        # ModelComparison (not a per-model EvaluationResult) so it reaches
        # agents/recommender.py's prompt the same way per-model
        # explainability does.
        try:
            comparison.dataset_explainability = explainability.explain_forecast(
                train_df, target_column, time_column
            ).model_dump()
        except Exception:  # noqa: BLE001 - explainability is supplementary, never fails the run
            pass
    elif problem_type in (ProblemType.CLASSIFICATION, ProblemType.REGRESSION):
        # Classification/regression's counterpart: a cross-model consensus
        # feature-importance ranking (never null just because only one
        # candidate trained - see explain_dataset_consensus's 2-model floor).
        try:
            comparison.dataset_explainability = explainability.explain_dataset_consensus(model_results)
        except Exception:  # noqa: BLE001 - explainability is supplementary, never fails the run
            pass

    metrics: dict = {
        # Single source of truth for "which metric did we rank on" - always
        # comparison.primary_metric (tools/evaluation.py's
        # select_primary_metric()), never computed independently here. Do
        # not reintroduce a second, separately-derived eval_metric: that is
        # exactly the two-disagreeing-fields bug this field once had.
        "eval_metric": comparison.primary_metric,
        "models": models,
        "candidate_results": [r.to_dict() for r in model_results],
        "model_comparison": comparison.model_dump(),
    }
    if autogluon_best_model:
        metrics["autogluon_best_model"] = autogluon_best_model
    if autogluon_model_path:
        metrics["model_path"] = autogluon_model_path

    chart_data = {
        r.model_name: sample
        for r in model_results
        if (sample := getattr(r, "chart_sample", None)) is not None
    }
    return metrics, chart_data


def _score_row(result: ModelResult) -> dict:
    return {"score_test": result.score_test, "score_val": result.score_val, "fit_time_s": result.training_time}


def _to_evaluation_result(result: ModelResult, problem_type: ProblemType) -> evaluation.EvaluationResult:
    is_automl = result.artifacts.get("source") is not None
    # "validation_strategy_executed" (set in run_model) reflects what actually
    # ran, not what the plan asked for - a candidate that skipped CV (too few
    # rows, or an unsupported/no strategy) must not be labeled k_fold/
    # stratified_k_fold when it only ever saw the single train/test split.
    executed = result.artifacts.get("validation_strategy_executed")
    strategy_label = "automl_internal_holdout" if is_automl else (executed or "train_test_split")
    return evaluation.EvaluationResult(
        model_name=result.model_name,
        problem_type=problem_type.value,
        status=result.status,
        validation_strategy=strategy_label,
        metrics=result.metrics,
        training_time=result.training_time,
        prediction_time=result.prediction_time,
        errors=result.errors,
        feature_importance=result.feature_importance,
        explainability=result.explainability,
    )


def _run_cross_validation(
    definition: ModelDefinition,
    train_df,
    target_column: str,
    time_column: Optional[str],
    problem_type: ProblemType,
    strategy: ValidationStrategyType,
    folds: int,
    params: dict,
) -> float:
    """Real k-fold/stratified-k-fold CV, trained/scored entirely on train_df
    (test_df is never touched here - it stays the untouched final holdout).
    Returns the mean per-fold score in the same higher-is-better sign
    convention as _score(), so it's directly comparable to score_test."""
    y = train_df[target_column]
    use_stratified = strategy == ValidationStrategyType.STRATIFIED_K_FOLD and problem_type == ProblemType.CLASSIFICATION
    if use_stratified and y.value_counts().min() >= folds:
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
        split_iter = splitter.split(train_df, y)
    else:
        splitter = KFold(n_splits=folds, shuffle=True, random_state=42)
        split_iter = splitter.split(train_df)

    scores = []
    for train_idx, val_idx in split_iter:
        fold_train = train_df.iloc[train_idx]
        fold_val = train_df.iloc[val_idx]
        fitted = definition.train_fn(fold_train, target_column, time_column, **params)
        predictions = definition.predict_fn(fitted, fold_val, target_column, time_column)
        score, _ = _score(problem_type, fold_val[target_column], predictions)
        scores.append(score)
    return float(np.mean(scores))


def run_model(
    definition: ModelDefinition,
    train_df,
    test_df,
    target_column: str,
    time_column: Optional[str],
    problem_type: ProblemType,
    validation_strategy: Optional[ValidationStrategyType] = None,
    validation_folds: Optional[int] = None,
) -> ModelResult:
    """Runs one non-AutoML candidate end to end: dependency check ->
    validation-strategy check -> (k_fold/stratified_k_fold: cross-validate on
    train_df for score_val) -> train on the full train_df -> predict on
    test_df -> score. Never raises - every failure mode becomes a ModelResult
    with status != "success".
    """
    params = dict(definition.default_params)

    missing = [dep for dep in definition.required_dependencies if not is_dependency_available(dep)]
    if missing:
        return ModelResult(
            definition.name,
            "skipped_missing_dependency",
            errors=f"Missing dependencies: {', '.join(missing)}",
            parameters=params,
        )

    if (
        validation_strategy is not None
        and definition.supported_validation_strategies
        and validation_strategy not in definition.supported_validation_strategies
    ):
        return ModelResult(
            definition.name,
            "skipped_unsupported_validation_strategy",
            errors=f"{validation_strategy.value!r} is not supported by {definition.name!r}",
            parameters=params,
        )

    score_val: Optional[float] = None
    validation_strategy_executed: Optional[str] = None
    if validation_strategy in _CV_STRATEGIES and problem_type in (ProblemType.CLASSIFICATION, ProblemType.REGRESSION):
        folds = validation_folds or DEFAULT_CV_FOLDS
        if len(train_df) >= folds * MIN_ROWS_PER_FOLD:
            try:
                score_val = _run_cross_validation(
                    definition, train_df, target_column, time_column, problem_type, validation_strategy, folds, params
                )
                validation_strategy_executed = validation_strategy.value
            except Exception:  # noqa: BLE001 - CV is supplementary; a failure here must not sink the candidate
                score_val = None

    start = time.perf_counter()
    try:
        fitted = definition.train_fn(train_df, target_column, time_column, **params)
    except Exception as exc:  # noqa: BLE001 - one bad candidate must not sink the run
        return ModelResult(
            definition.name, "failed", errors=str(exc), parameters=params, training_time=round(time.perf_counter() - start, 3)
        )
    training_time = round(time.perf_counter() - start, 3)

    predict_start = time.perf_counter()
    try:
        predictions = definition.predict_fn(fitted, test_df, target_column, time_column)
    except Exception as exc:  # noqa: BLE001
        return ModelResult(
            definition.name, "failed", errors=str(exc), parameters=params, training_time=training_time
        )
    prediction_time = round(time.perf_counter() - predict_start, 3)

    try:
        score_test, eval_metric = _score(problem_type, test_df[target_column], predictions)
    except Exception as exc:  # noqa: BLE001 - scoring failure is still just this candidate's failure
        return ModelResult(
            definition.name, "failed", errors=f"scoring failed: {exc}", parameters=params,
            training_time=training_time, prediction_time=prediction_time,
        )

    rich_metrics = _compute_rich_metrics(definition, fitted, problem_type, train_df, test_df, target_column, time_column, predictions)
    feature_importance = _compute_feature_importance(definition, fitted, problem_type, train_df, target_column, time_column)
    explainability_result = _compute_explainability(
        definition, fitted, problem_type, test_df, target_column, time_column, feature_importance
    )

    preview = [float(x) for x in np.asarray(predictions).ravel()[:5]]
    artifacts = {"validation_strategy_executed": validation_strategy_executed} if validation_strategy_executed else None
    model_result = ModelResult(
        model_name=definition.name,
        status="success",
        training_time=training_time,
        prediction_time=prediction_time,
        prediction=preview,
        parameters=params,
        artifacts=artifacts,
        score_test=score_test,
        score_val=score_val,
        eval_metric=eval_metric,
        metrics=rich_metrics,
        feature_importance=feature_importance,
        explainability=explainability_result,
    )
    # Deliberately NOT part of ModelResult's constructor/to_dict() - see this
    # module's docstring on why chart_data must never reach metrics/the LLM.
    model_result.chart_sample = _chart_sample(problem_type, test_df[target_column], predictions)
    return model_result


def _compute_rich_metrics(definition, fitted, problem_type, train_df, test_df, target_column, time_column, predictions) -> dict:
    """Best-effort Phase 4 metric breakdown, computed from the SAME
    predictions run_model() already produced (no extra training). Wrapped
    defensively so a metrics-computation issue degrades to an empty dict
    rather than failing a candidate that otherwise trained and scored fine.
    """
    try:
        y_true = test_df[target_column]
        y_proba = None
        if problem_type == ProblemType.CLASSIFICATION and definition.predict_proba_fn is not None:
            y_proba = definition.predict_proba_fn(fitted, test_df, target_column, time_column)
        y_train = train_df[target_column] if problem_type == ProblemType.FORECASTING else None
        return evaluation.compute_metrics(problem_type, y_true, predictions, y_proba=y_proba, y_train=y_train)
    except Exception:  # noqa: BLE001 - supplementary info only, never fails the candidate
        return {}


def _compute_feature_importance(
    definition: ModelDefinition, fitted, problem_type: ProblemType, train_df, target_column: str, time_column: Optional[str]
) -> Optional[dict]:
    """Best-effort, classification/regression only - forecasting baselines
    aren't feature-based models. Wrapped defensively: a model that doesn't
    expose feature_importances_/coef_ (baseline, statistical models) simply
    gets None, never an error.
    """
    if problem_type not in (ProblemType.CLASSIFICATION, ProblemType.REGRESSION):
        return None
    try:
        model = fitted.get("model") if isinstance(fitted, dict) else fitted
        if hasattr(model, "feature_importances_"):
            values = np.asarray(model.feature_importances_, dtype=float)
        elif hasattr(model, "coef_"):
            values = np.abs(np.asarray(model.coef_, dtype=float)).ravel()
        else:
            return None
        names = feature_columns_for(train_df, target_column, time_column)
        if len(names) != len(values):
            return None
        pairs = sorted(zip(names, values), key=lambda kv: abs(kv[1]), reverse=True)[:10]
        return {name: round(float(value), 6) for name, value in pairs}
    except Exception:  # noqa: BLE001 - supplementary info only
        return None


def _compute_explainability(
    definition: ModelDefinition,
    fitted,
    problem_type: ProblemType,
    test_df,
    target_column: str,
    time_column: Optional[str],
    feature_importance: Optional[dict],
) -> Optional[dict]:
    """Phase 7: permutation importance + SHAP (tree-based, where practical)
    on top of the native feature importance already computed. Classification/
    regression only - forecasting's explainability is dataset-level, not
    per-model (see run_candidates()'s explain_forecast() call). Wrapped
    defensively so a failure here never affects the candidate's core result.
    """
    if problem_type not in (ProblemType.CLASSIFICATION, ProblemType.REGRESSION):
        return None
    try:
        result = explainability.explain_tabular_model(
            definition, fitted, problem_type, test_df, target_column, time_column, feature_importance
        )
        return result.model_dump()
    except Exception:  # noqa: BLE001 - explainability is supplementary, never fails the candidate
        return None


def _chart_sample(problem_type: ProblemType, y_true, y_pred) -> Optional[dict]:
    """Bounded (never the full dataset) actual/predicted sample for report
    charts (tools/report_charts.py) - classification keeps original labels,
    regression/forecasting keeps floats. Deliberately NOT stored on
    ModelResult/EvaluationResult; see this module's docstring.
    """
    try:
        y_true_arr = np.asarray(y_true)
        y_pred_arr = np.asarray(y_pred).ravel()
        n = min(len(y_true_arr), len(y_pred_arr), CHART_SAMPLE_CAP)
        if n == 0:
            return None
        actual_tail, predicted_tail = y_true_arr[-n:], y_pred_arr[-n:]
        if problem_type == ProblemType.CLASSIFICATION:
            actual = [str(v) for v in actual_tail]
            predicted = [str(v) for v in predicted_tail]
        else:
            actual = [float(v) for v in actual_tail]
            predicted = [float(v) for v in predicted_tail]
        return {"actual": actual, "predicted": predicted}
    except Exception:  # noqa: BLE001 - charts are supplementary, never fail the candidate
        return None


def _score(problem_type: ProblemType, y_true, y_pred) -> tuple[float, str]:
    """Same sign convention AutoGluon uses so classical and AutoGluon scores
    are directly comparable in one leaderboard: higher is always better -
    plain accuracy for classification, negative RMSE for regression/forecasting.
    Kept unchanged from Phase 3 for backward compatibility (ModelResult.score_test/
    eval_metric); tools/evaluation.py's richer metrics dict is Phase 4's addition.
    """
    if problem_type == ProblemType.CLASSIFICATION:
        return float(accuracy_score(np.asarray(y_true), np.asarray(y_pred))), "accuracy"

    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float).ravel()
    if len(y_pred_arr) != len(y_true_arr):
        y_pred_arr = y_pred_arr[: len(y_true_arr)]
    rmse = float(np.sqrt(mean_squared_error(y_true_arr, y_pred_arr)))
    return -rmse, "root_mean_squared_error"


def _run_automl(
    definition: ModelDefinition,
    train_df,
    test_df,
    target_column: str,
    time_column: Optional[str],
    problem_type: ProblemType,
) -> tuple[list[ModelResult], Optional[str], Optional[str]]:
    """Expands AutoGluon's internal leaderboard into one ModelResult per
    underlying model, calling tools/automl_training.train_models exactly as
    before Phase 3 (see tools/model_registry.py's module docstring). Each
    row's own score_test is sign/name-normalized (tools/evaluation.py) onto
    our metric convention so it ranks fairly against classical candidates.
    """
    missing = [dep for dep in definition.required_dependencies if not is_dependency_available(dep)]
    if missing:
        return (
            [ModelResult(definition.name, "skipped_missing_dependency", errors=f"Missing dependencies: {', '.join(missing)}")],
            None,
            None,
        )

    start = time.perf_counter()
    try:
        raw = definition.train_fn(
            train_df,
            test_df,
            target_column=target_column,
            problem_type=problem_type.value,
            time_column=time_column,
        )
    except Exception as exc:  # noqa: BLE001 - AutoGluon failing must not sink other candidates
        return (
            [ModelResult(definition.name, "failed", errors=str(exc), training_time=round(time.perf_counter() - start, 3))],
            None,
            None,
        )

    eval_metric_name = raw.get("eval_metric") or ""
    best_model_name = raw.get("autogluon_best_model")
    best_predictions = raw.get("best_model_predictions")
    results = []
    for model_name, scores in raw.get("models", {}).items():
        # tools/automl_training.py now computes a full MAE/R2/MAPE/etc
        # breakdown per leaderboard row (not just the single best model) -
        # merge it in, then let the leaderboard's own score_test/score_val
        # (already sign/name-normalized below) win for the primary metric key.
        rich_metrics = dict(scores.get("metrics") or {})
        score_test = scores.get("score_test")
        if score_test is not None:
            key, value = evaluation.normalize_autogluon_metric(eval_metric_name, score_test)
            rich_metrics[key] = value
        result = ModelResult(
            model_name=model_name,
            status="success",
            training_time=scores.get("fit_time_s"),
            score_test=score_test,
            score_val=scores.get("score_val"),
            eval_metric=raw.get("eval_metric"),
            artifacts={"source": definition.name},
            metrics=rich_metrics,
            explainability=explainability.unsupported_automl_explainability(model_name, problem_type).model_dump(),
        )
        # Only the leaderboard row matching AutoGluon's own best model gets
        # chart data - that's the one row a single extra predict() call
        # bought us (tools/automl_training.py); the other rows would each
        # need their own predict() call, which isn't worth the extra cost
        # for entries that aren't the recommendation anyway.
        if model_name == best_model_name and best_predictions is not None:
            result.chart_sample = _chart_sample(problem_type, test_df[target_column], best_predictions)
        results.append(result)
    return results, best_model_name, raw.get("model_path")
