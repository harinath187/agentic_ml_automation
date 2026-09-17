"""Model Registry: a catalogue of trainable model definitions for
classification, regression, and (non-hierarchical) forecasting.

Each ModelDefinition pairs a `train_fn`/`predict_fn` with declarative
metadata (problem types, model family, supported validation strategies,
required dependencies, default params). The registry itself never decides
which models to run for a given dataset - resolve_candidates() filters it by
problem_type and by the validated ExperimentPlan's candidate_model_families
(see tools/model_runner.py, which does the actual training/scoring).

AutoGluon is not registered here - it is used only for the `hierarchical`
forecasting scope strategy, which trains via
tools/automl_training.train_hierarchical_timeseries directly from
orchestration/graph.py, bypassing this registry entirely (see CLAUDE.md's
scope_strategy routing table).
"""
from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from agents.schemas import ProblemType, ValidationStrategyType


class ModelResult:
    """Standardized result of running one candidate model. A plain class
    (not Pydantic) since it never crosses the LLM boundary and can hold
    arbitrary artifacts (e.g. a fitted estimator reference is deliberately
    NOT stored here - only JSON-safe summaries are)."""

    def __init__(
        self,
        model_name: str,
        status: str,
        training_time: Optional[float] = None,
        prediction: Optional[list] = None,
        parameters: Optional[dict] = None,
        errors: Optional[str] = None,
        artifacts: Optional[dict] = None,
        score_test: Optional[float] = None,
        score_val: Optional[float] = None,
        eval_metric: Optional[str] = None,
        metrics: Optional[dict] = None,
        prediction_time: Optional[float] = None,
        feature_importance: Optional[dict] = None,
        explainability: Optional[dict] = None,
    ):
        self.model_name = model_name
        self.status = status  # "success" | "failed" | "skipped_missing_dependency" | "skipped_unsupported_validation_strategy"
        self.training_time = training_time
        self.prediction = prediction
        self.parameters = parameters or {}
        self.errors = errors
        self.artifacts = artifacts or {}
        self.score_test = score_test
        self.score_val = score_val
        self.eval_metric = eval_metric
        # Phase 4: the full problem-specific metric breakdown (accuracy/
        # precision/recall/f1/roc_auc, or mae/rmse/r2/mape/smape, or
        # mae/rmse/smape/mase) - see tools/evaluation.py. score_test/
        # eval_metric above are kept unchanged for backward compatibility.
        self.metrics = metrics or {}
        self.prediction_time = prediction_time
        # Phase 6: top feature -> importance score (classification/regression
        # only). Safe to surface to an LLM/report - feature names and
        # relative importances, never raw data rows.
        self.feature_importance = feature_importance
        # Phase 7: tools/explainability.py's ExplainabilityResult, as a dict
        # (feature/permutation/SHAP importance, or forecasting trend/
        # seasonality signals). Same safety rationale as feature_importance.
        self.explainability = explainability

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "status": self.status,
            "training_time": self.training_time,
            "prediction": self.prediction,
            "parameters": self.parameters,
            "errors": self.errors,
            "artifacts": self.artifacts,
            "score_test": self.score_test,
            "score_val": self.score_val,
            "eval_metric": self.eval_metric,
            "metrics": self.metrics,
            "prediction_time": self.prediction_time,
            "feature_importance": self.feature_importance,
            "explainability": self.explainability,
        }


@dataclass
class ModelDefinition:
    name: str
    problem_types: tuple[ProblemType, ...]
    model_family: str
    train_fn: Callable[..., Any]
    predict_fn: Callable[..., Any]
    supported_validation_strategies: tuple[ValidationStrategyType, ...] = ()
    required_dependencies: tuple[str, ...] = ()
    default_params: dict = field(default_factory=dict)
    predict_proba_fn: Optional[Callable[..., Any]] = None
    """Classifiers only: returns the positive-class probability for binary
    targets, enabling ROC-AUC (tools/evaluation.py). None for regressors,
    forecasters, and any classifier that doesn't expose probabilities."""


def is_dependency_available(module_name: str) -> bool:
    """Cheap availability check (no import side effects) so a missing
    optional dependency (statsmodels, xgboost, lightgbm, ...) becomes a
    skipped ModelResult instead of crashing the whole training step."""
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ValueError):
        return False


def _feature_frame(df: pd.DataFrame, target_column: Optional[str], time_column: Optional[str]) -> pd.DataFrame:
    drop_cols = [c for c in (target_column, time_column) if c and c in df.columns]
    numeric = df.drop(columns=drop_cols).select_dtypes(include=[np.number])
    return numeric.fillna(0.0)


def feature_columns_for(df: pd.DataFrame, target_column: Optional[str], time_column: Optional[str]) -> list[str]:
    """Public wrapper around _feature_frame's column selection - lets
    tools/model_runner.py name feature-importance scores consistently with
    what the classical train_fn/predict_fn factories actually used."""
    return list(_feature_frame(df, target_column, time_column).columns)


def feature_frame_for(df: pd.DataFrame, target_column: Optional[str], time_column: Optional[str]) -> pd.DataFrame:
    """Public wrapper returning the actual feature matrix (not just column
    names) - used by tools/explainability.py for permutation importance/SHAP,
    so they see exactly what the model was trained/predicted on."""
    return _feature_frame(df, target_column, time_column)


# --- generic sklearn-style tabular models -----------------------------------


def _predict_proba_fn(fitted, test_df, target_column, time_column):
    """Shared by every classifier factory below: they all fit as
    {"model": ..., "encoder": ...}. Returns the positive-class (encoder's
    2nd class) probability for binary targets, or None otherwise - the
    contract tools/evaluation.py's ROC-AUC computation expects.
    """
    model = fitted["model"]
    if not hasattr(model, "predict_proba"):
        return None
    encoder = fitted["encoder"]
    if len(encoder.classes_) != 2:
        return None
    X = _feature_frame(test_df, target_column, time_column)
    proba = model.predict_proba(X)
    return proba[:, 1]


def _make_classifier(estimator_cls, **fixed_params):
    def train_fn(train_df, target_column, time_column, **params):
        X = _feature_frame(train_df, target_column, time_column)
        y = train_df[target_column]
        encoder = LabelEncoder().fit(y)
        model = estimator_cls(**{**fixed_params, **params})
        model.fit(X, encoder.transform(y))
        return {"model": model, "encoder": encoder}

    def predict_fn(fitted, test_df, target_column, time_column):
        X = _feature_frame(test_df, target_column, time_column)
        encoded_preds = fitted["model"].predict(X)
        return fitted["encoder"].inverse_transform(np.asarray(encoded_preds).astype(int))

    return train_fn, predict_fn, _predict_proba_fn


def _make_regressor(estimator_cls, **fixed_params):
    def train_fn(train_df, target_column, time_column, **params):
        X = _feature_frame(train_df, target_column, time_column)
        y = train_df[target_column].astype(float)
        model = estimator_cls(**{**fixed_params, **params})
        model.fit(X, y)
        return model

    def predict_fn(fitted, test_df, target_column, time_column):
        X = _feature_frame(test_df, target_column, time_column)
        return fitted.predict(X)

    return train_fn, predict_fn


def _make_scaled_classifier(estimator_cls, **fixed_params):
    """Like _make_classifier, but fits a StandardScaler on the training
    features first - for scale-sensitive estimators (SVM, KNN, MLP) that
    _make_classifier's raw _feature_frame() would otherwise disadvantage,
    unlike tree/boosting models which don't need scaling."""

    def train_fn(train_df, target_column, time_column, **params):
        X = _feature_frame(train_df, target_column, time_column)
        y = train_df[target_column]
        encoder = LabelEncoder().fit(y)
        scaler = StandardScaler().fit(X)
        model = estimator_cls(**{**fixed_params, **params})
        model.fit(scaler.transform(X), encoder.transform(y))
        return {"model": model, "encoder": encoder, "scaler": scaler}

    def predict_fn(fitted, test_df, target_column, time_column):
        X = _feature_frame(test_df, target_column, time_column)
        encoded_preds = fitted["model"].predict(fitted["scaler"].transform(X))
        return fitted["encoder"].inverse_transform(np.asarray(encoded_preds).astype(int))

    def predict_proba_fn(fitted, test_df, target_column, time_column):
        model = fitted["model"]
        if not hasattr(model, "predict_proba"):
            return None
        encoder = fitted["encoder"]
        if len(encoder.classes_) != 2:
            return None
        X = _feature_frame(test_df, target_column, time_column)
        proba = model.predict_proba(fitted["scaler"].transform(X))
        return proba[:, 1]

    return train_fn, predict_fn, predict_proba_fn


def _make_scaled_regressor(estimator_cls, **fixed_params):
    def train_fn(train_df, target_column, time_column, **params):
        X = _feature_frame(train_df, target_column, time_column)
        y = train_df[target_column].astype(float)
        scaler = StandardScaler().fit(X)
        model = estimator_cls(**{**fixed_params, **params})
        model.fit(scaler.transform(X), y)
        return {"model": model, "scaler": scaler}

    def predict_fn(fitted, test_df, target_column, time_column):
        X = _feature_frame(test_df, target_column, time_column)
        return fitted["model"].predict(fitted["scaler"].transform(X))

    return train_fn, predict_fn


def _xgboost_classifier_train_predict():
    def train_fn(train_df, target_column, time_column, **params):
        from xgboost import XGBClassifier

        X = _feature_frame(train_df, target_column, time_column)
        y = train_df[target_column]
        encoder = LabelEncoder().fit(y)
        model = XGBClassifier(**{"eval_metric": "logloss", "random_state": 42, **params})
        model.fit(X, encoder.transform(y))
        return {"model": model, "encoder": encoder}

    def predict_fn(fitted, test_df, target_column, time_column):
        X = _feature_frame(test_df, target_column, time_column)
        encoded_preds = fitted["model"].predict(X)
        return fitted["encoder"].inverse_transform(np.asarray(encoded_preds).astype(int))

    return train_fn, predict_fn, _predict_proba_fn


def _xgboost_regressor_train_predict():
    def train_fn(train_df, target_column, time_column, **params):
        from xgboost import XGBRegressor

        X = _feature_frame(train_df, target_column, time_column)
        y = train_df[target_column].astype(float)
        model = XGBRegressor(**{"random_state": 42, **params})
        model.fit(X, y)
        return model

    def predict_fn(fitted, test_df, target_column, time_column):
        X = _feature_frame(test_df, target_column, time_column)
        return fitted.predict(X)

    return train_fn, predict_fn


def _lightgbm_classifier_train_predict():
    def train_fn(train_df, target_column, time_column, **params):
        from lightgbm import LGBMClassifier

        X = _feature_frame(train_df, target_column, time_column)
        y = train_df[target_column]
        encoder = LabelEncoder().fit(y)
        model = LGBMClassifier(**{"verbosity": -1, "random_state": 42, **params})
        model.fit(X, encoder.transform(y))
        return {"model": model, "encoder": encoder}

    def predict_fn(fitted, test_df, target_column, time_column):
        X = _feature_frame(test_df, target_column, time_column)
        encoded_preds = fitted["model"].predict(X)
        return fitted["encoder"].inverse_transform(np.asarray(encoded_preds).astype(int))

    return train_fn, predict_fn, _predict_proba_fn


def _lightgbm_regressor_train_predict():
    def train_fn(train_df, target_column, time_column, **params):
        from lightgbm import LGBMRegressor

        X = _feature_frame(train_df, target_column, time_column)
        y = train_df[target_column].astype(float)
        model = LGBMRegressor(**{"verbosity": -1, "random_state": 42, **params})
        model.fit(X, y)
        return model

    def predict_fn(fitted, test_df, target_column, time_column):
        X = _feature_frame(test_df, target_column, time_column)
        return fitted.predict(X)

    return train_fn, predict_fn


# --- classical forecasting baselines ----------------------------------------
#
# All operate on the univariate target series (sorted by time_column) already
# present in train_df/test_df after the existing clean/engineer/split steps -
# no new state is threaded through the graph for these.


def _sorted_series(df: pd.DataFrame, target_column: str, time_column: Optional[str]) -> pd.Series:
    if time_column and time_column in df.columns:
        df = df.sort_values(time_column)
    return df[target_column].astype(float).reset_index(drop=True)


def _naive_train_fn(train_df, target_column, time_column, **params):
    return {"last_value": _sorted_series(train_df, target_column, time_column).iloc[-1]}


def _naive_predict_fn(fitted, test_df, target_column, time_column):
    return np.full(len(test_df), fitted["last_value"])


def _seasonal_naive_train_fn(train_df, target_column, time_column, season_length: int = 7, **params):
    series = _sorted_series(train_df, target_column, time_column)
    season_length = max(1, min(season_length, len(series)))
    return {"seasonal_tail": series.iloc[-season_length:].to_numpy(), "season_length": season_length}


def _seasonal_naive_predict_fn(fitted, test_df, target_column, time_column):
    horizon = len(test_df)
    tail = fitted["seasonal_tail"]
    reps = int(np.ceil(horizon / len(tail)))
    return np.tile(tail, reps)[:horizon]


def _ets_train_fn(train_df, target_column, time_column, **params):
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    series = _sorted_series(train_df, target_column, time_column)
    model = ExponentialSmoothing(series, trend=params.get("trend", "add"), seasonal=None).fit()
    return model


def _statsmodels_forecast_predict_fn(fitted, test_df, target_column, time_column):
    return np.asarray(fitted.forecast(len(test_df)))


def _arima_train_fn(train_df, target_column, time_column, order=(1, 1, 1), **params):
    from statsmodels.tsa.arima.model import ARIMA

    series = _sorted_series(train_df, target_column, time_column)
    return ARIMA(series, order=order).fit()


def _sarima_train_fn(train_df, target_column, time_column, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7), **params):
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    series = _sorted_series(train_df, target_column, time_column)
    return SARIMAX(
        series, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False
    ).fit(disp=False)


_TABULAR_VALIDATION_STRATEGIES = (
    ValidationStrategyType.TRAIN_TEST_SPLIT,
    ValidationStrategyType.K_FOLD,
    ValidationStrategyType.STRATIFIED_K_FOLD,
)
_FORECASTING_VALIDATION_STRATEGIES = (ValidationStrategyType.TIME_SERIES_SPLIT,)


def _classification_registry() -> list[ModelDefinition]:
    baseline_train, baseline_predict, baseline_proba = _make_classifier(DummyClassifier, strategy="most_frequent")
    logreg_train, logreg_predict, logreg_proba = _make_classifier(LogisticRegression, max_iter=1000)
    rf_train, rf_predict, rf_proba = _make_classifier(RandomForestClassifier, n_estimators=100, random_state=42)
    xgb_train, xgb_predict, xgb_proba = _xgboost_classifier_train_predict()
    lgbm_train, lgbm_predict, lgbm_proba = _lightgbm_classifier_train_predict()
    dtree_train, dtree_predict, dtree_proba = _make_classifier(DecisionTreeClassifier, random_state=42)
    svm_train, svm_predict, svm_proba = _make_scaled_classifier(SVC, probability=True, random_state=42)
    knn_train, knn_predict, knn_proba = _make_scaled_classifier(KNeighborsClassifier)
    nb_train, nb_predict, nb_proba = _make_classifier(GaussianNB)
    mlp_train, mlp_predict, mlp_proba = _make_scaled_classifier(MLPClassifier, random_state=42, max_iter=500)

    return [
        ModelDefinition("baseline", (ProblemType.CLASSIFICATION,), "baseline", baseline_train, baseline_predict, _TABULAR_VALIDATION_STRATEGIES, predict_proba_fn=baseline_proba),
        ModelDefinition("logistic_regression", (ProblemType.CLASSIFICATION,), "linear", logreg_train, logreg_predict, _TABULAR_VALIDATION_STRATEGIES, predict_proba_fn=logreg_proba),
        ModelDefinition("random_forest", (ProblemType.CLASSIFICATION,), "tree_ensemble", rf_train, rf_predict, _TABULAR_VALIDATION_STRATEGIES, predict_proba_fn=rf_proba),
        ModelDefinition("xgboost", (ProblemType.CLASSIFICATION,), "gradient_boosting", xgb_train, xgb_predict, _TABULAR_VALIDATION_STRATEGIES, required_dependencies=("xgboost",), predict_proba_fn=xgb_proba),
        ModelDefinition("lightgbm", (ProblemType.CLASSIFICATION,), "gradient_boosting", lgbm_train, lgbm_predict, _TABULAR_VALIDATION_STRATEGIES, required_dependencies=("lightgbm",), predict_proba_fn=lgbm_proba),
        ModelDefinition("decision_tree", (ProblemType.CLASSIFICATION,), "tree", dtree_train, dtree_predict, _TABULAR_VALIDATION_STRATEGIES, predict_proba_fn=dtree_proba),
        ModelDefinition("svm", (ProblemType.CLASSIFICATION,), "svm", svm_train, svm_predict, _TABULAR_VALIDATION_STRATEGIES, predict_proba_fn=svm_proba),
        ModelDefinition("knn", (ProblemType.CLASSIFICATION,), "instance_based", knn_train, knn_predict, _TABULAR_VALIDATION_STRATEGIES, predict_proba_fn=knn_proba),
        ModelDefinition("naive_bayes", (ProblemType.CLASSIFICATION,), "naive_bayes", nb_train, nb_predict, _TABULAR_VALIDATION_STRATEGIES, predict_proba_fn=nb_proba),
        ModelDefinition("neural_network", (ProblemType.CLASSIFICATION,), "neural_network", mlp_train, mlp_predict, _TABULAR_VALIDATION_STRATEGIES, predict_proba_fn=mlp_proba),
    ]


def _regression_registry() -> list[ModelDefinition]:
    baseline_train, baseline_predict = _make_regressor(DummyRegressor, strategy="mean")
    linreg_train, linreg_predict = _make_regressor(LinearRegression)
    rf_train, rf_predict = _make_regressor(RandomForestRegressor, n_estimators=100, random_state=42)
    xgb_train, xgb_predict = _xgboost_regressor_train_predict()
    lgbm_train, lgbm_predict = _lightgbm_regressor_train_predict()
    dtree_train, dtree_predict = _make_regressor(DecisionTreeRegressor, random_state=42)
    svm_train, svm_predict = _make_scaled_regressor(SVR)
    knn_train, knn_predict = _make_scaled_regressor(KNeighborsRegressor)
    mlp_train, mlp_predict = _make_scaled_regressor(MLPRegressor, random_state=42, max_iter=500)

    return [
        ModelDefinition("baseline", (ProblemType.REGRESSION,), "baseline", baseline_train, baseline_predict, _TABULAR_VALIDATION_STRATEGIES),
        ModelDefinition("linear_regression", (ProblemType.REGRESSION,), "linear", linreg_train, linreg_predict, _TABULAR_VALIDATION_STRATEGIES),
        ModelDefinition("random_forest", (ProblemType.REGRESSION,), "tree_ensemble", rf_train, rf_predict, _TABULAR_VALIDATION_STRATEGIES),
        ModelDefinition("xgboost", (ProblemType.REGRESSION,), "gradient_boosting", xgb_train, xgb_predict, _TABULAR_VALIDATION_STRATEGIES, required_dependencies=("xgboost",)),
        ModelDefinition("lightgbm", (ProblemType.REGRESSION,), "gradient_boosting", lgbm_train, lgbm_predict, _TABULAR_VALIDATION_STRATEGIES, required_dependencies=("lightgbm",)),
        ModelDefinition("decision_tree", (ProblemType.REGRESSION,), "tree", dtree_train, dtree_predict, _TABULAR_VALIDATION_STRATEGIES),
        ModelDefinition("svm", (ProblemType.REGRESSION,), "svm", svm_train, svm_predict, _TABULAR_VALIDATION_STRATEGIES),
        ModelDefinition("knn", (ProblemType.REGRESSION,), "instance_based", knn_train, knn_predict, _TABULAR_VALIDATION_STRATEGIES),
        ModelDefinition("neural_network", (ProblemType.REGRESSION,), "neural_network", mlp_train, mlp_predict, _TABULAR_VALIDATION_STRATEGIES),
    ]


def _forecasting_registry() -> list[ModelDefinition]:
    return [
        ModelDefinition("naive", (ProblemType.FORECASTING,), "baseline", _naive_train_fn, _naive_predict_fn, _FORECASTING_VALIDATION_STRATEGIES),
        ModelDefinition("seasonal_naive", (ProblemType.FORECASTING,), "baseline", _seasonal_naive_train_fn, _seasonal_naive_predict_fn, _FORECASTING_VALIDATION_STRATEGIES),
        ModelDefinition("ets", (ProblemType.FORECASTING,), "statistical", _ets_train_fn, _statsmodels_forecast_predict_fn, _FORECASTING_VALIDATION_STRATEGIES, required_dependencies=("statsmodels",)),
        ModelDefinition("arima", (ProblemType.FORECASTING,), "statistical", _arima_train_fn, _statsmodels_forecast_predict_fn, _FORECASTING_VALIDATION_STRATEGIES, required_dependencies=("statsmodels",)),
        ModelDefinition("sarima", (ProblemType.FORECASTING,), "statistical", _sarima_train_fn, _statsmodels_forecast_predict_fn, _FORECASTING_VALIDATION_STRATEGIES, required_dependencies=("statsmodels",)),
    ]


REGISTRY: list[ModelDefinition] = [
    *_classification_registry(),
    *_regression_registry(),
    *_forecasting_registry(),
]


def _normalize(name: str) -> str:
    return name.strip().lower().replace("-", "_").replace(" ", "_")


def _squash(name: str) -> str:
    return name.replace("_", "")


def get_registry_for_problem_type(problem_type: ProblemType) -> list[ModelDefinition]:
    return [d for d in REGISTRY if problem_type in d.problem_types]


def resolve_candidates(problem_type: ProblemType, requested_names: list[str]) -> tuple[list[ModelDefinition], list[str]]:
    """Filters the registry to (problem_type, requested_names), tolerating
    naming variations from the LLM (e.g. "LightGBM", "Random Forest").
    Returns (matched_definitions, unmatched_names) - never raises, and never
    silently runs the full registry when nothing matches (the caller decides
    what to do with an empty match list, e.g. fall back to a safe default).
    """
    available = get_registry_for_problem_type(problem_type)
    by_normalized = {_normalize(d.name): d for d in available}
    by_squashed = {_squash(_normalize(d.name)): d for d in available}

    matched: list[ModelDefinition] = []
    unmatched: list[str] = []
    seen_names = set()

    for requested in requested_names:
        key = _normalize(requested)
        definition = by_normalized.get(key) or by_squashed.get(_squash(key))
        if definition is None:
            unmatched.append(requested)
        elif definition.name not in seen_names:
            matched.append(definition)
            seen_names.add(definition.name)

    return matched, unmatched


def default_candidates(problem_type: ProblemType) -> list[ModelDefinition]:
    """A minimal, safe fallback (a single baseline model) used only when
    resolve_candidates() couldn't match anything the plan requested - keeps
    a run from training zero models without ever running "every model for
    every dataset" by default."""
    by_name = {d.name: d for d in get_registry_for_problem_type(problem_type)}
    fallback_names = ("naive",) if problem_type == ProblemType.FORECASTING else ("baseline",)
    return [by_name[name] for name in fallback_names if name in by_name]
