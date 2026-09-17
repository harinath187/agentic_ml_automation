import pandas as pd

from agents.schemas import ProblemType, ValidationStrategyType
from orchestration.graph import build_graph
from tools.hyperparameter_tuning import tune_model
from tools.model_registry import resolve_candidates


def test_tuning_selects_parameters_without_using_test_data():
    train_df = pd.DataFrame(
        {
            "feature": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            "target": ["ham", "spam"] * 6,
        }
    )
    (definition,), _ = resolve_candidates(ProblemType.CLASSIFICATION, ["logistic_regression"])

    result = tune_model(
        definition,
        train_df,
        target_column="target",
        time_column=None,
        problem_type=ProblemType.CLASSIFICATION,
        validation_strategy=ValidationStrategyType.STRATIFIED_K_FOLD,
        folds=2,
        max_trials=2,
        metric="accuracy",
    )

    assert result["status"] == "success"
    assert result["metric"] == "accuracy"
    assert result["trials_run"] == 2
    assert set(result["best_params"]) == {"C", "class_weight"}


def test_baseline_without_tuning_space_is_skipped():
    train_df = pd.DataFrame({"feature": [0, 1, 0, 1], "target": [0, 1, 0, 1]})
    (definition,), _ = resolve_candidates(ProblemType.CLASSIFICATION, ["baseline"])

    result = tune_model(
        definition, train_df, "target", None, ProblemType.CLASSIFICATION
    )

    assert result["status"] == "skipped"


def test_graph_places_tuning_after_vectorization():
    edges = {(edge.source, edge.target) for edge in build_graph().get_graph().edges}

    assert ("vectorize_text", "tune_models") in edges
    assert ("tune_models", "train") in edges