"""Proves the core Phase 4 requirement: the LLM must NOT determine which
model has the lowest error - the Python evaluation engine (tools/evaluation.py)
does, and agents/evaluator.py enforces that even when the LLM disagrees.
"""
import pandas as pd
import pytest

from agents.evaluator import evaluate_results
from agents.schemas import EvaluatorDecision, ExperimentPlan, ProblemType


def test_llm_naming_a_different_model_is_overridden(monkeypatch):
    """The deterministic ranking engine says "good" wins (lowest rmse). The
    LLM is mocked to insist "bad" is best_model. The final decision must
    still say "good" - the override in agents/evaluator.py must win.
    """
    metrics = {
        "eval_metric": "rmse",
        "models": {"good": {"score_test": -1.0, "fit_time_s": 0.1}, "bad": {"score_test": -5.0, "fit_time_s": 0.1}},
        "model_comparison": {
            "problem_type": "regression",
            "primary_metric": "rmse",
            "higher_is_better": False,
            "results": [
                {"model_name": "good", "problem_type": "regression", "status": "success", "validation_strategy": "train_test_split", "metrics": {"rmse": 1.0}, "training_time": 0.1, "prediction_time": None, "errors": None},
                {"model_name": "bad", "problem_type": "regression", "status": "success", "validation_strategy": "train_test_split", "metrics": {"rmse": 5.0}, "training_time": 0.1, "prediction_time": None, "errors": None},
            ],
            "ranked_model_names": ["good", "bad"],
            "winner": "good",
            "winner_reasoning": "'good' has the lowest rmse (1.0000) among 2 successfully evaluated candidate(s) out of 2 attempted.",
        },
    }
    plan = ExperimentPlan(problem_type=ProblemType.REGRESSION, target_column="y")

    def fake_call_llm_json(system_prompt, user_prompt, schema, **kwargs):
        # The LLM tries to name a different winner - this must be discarded.
        return EvaluatorDecision(decision="proceed", best_model="bad", reasoning="I like 'bad' better")

    import agents.evaluator as evaluator_module

    monkeypatch.setattr(evaluator_module, "call_llm_json", fake_call_llm_json)

    decision = evaluate_results(metrics, plan, retry_count=0, max_retries=2)

    assert decision.best_model == "good"
    assert decision.best_model != "bad"


def test_llm_omitting_best_model_still_gets_deterministic_winner(monkeypatch):
    metrics = {
        "eval_metric": "accuracy",
        "models": {"only_model": {"score_test": 0.8, "fit_time_s": 0.1}},
        "model_comparison": {
            "problem_type": "classification",
            "primary_metric": "accuracy",
            "higher_is_better": True,
            "results": [
                {"model_name": "only_model", "problem_type": "classification", "status": "success", "validation_strategy": "train_test_split", "metrics": {"accuracy": 0.8}, "training_time": 0.1, "prediction_time": None, "errors": None},
            ],
            "ranked_model_names": ["only_model"],
            "winner": "only_model",
            "winner_reasoning": "'only_model' has the highest accuracy (0.8000) among 1 successfully evaluated candidate(s) out of 1 attempted.",
        },
    }
    plan = ExperimentPlan(problem_type=ProblemType.CLASSIFICATION, target_column="y")

    def fake_call_llm_json(system_prompt, user_prompt, schema, **kwargs):
        return EvaluatorDecision(decision="proceed", best_model=None, reasoning="ok")

    import agents.evaluator as evaluator_module

    monkeypatch.setattr(evaluator_module, "call_llm_json", fake_call_llm_json)

    decision = evaluate_results(metrics, plan, retry_count=0, max_retries=2)

    assert decision.best_model == "only_model"


def test_max_retries_reached_skips_llm_and_uses_deterministic_winner(monkeypatch):
    metrics = {
        "eval_metric": "rmse",
        "models": {"a": {"score_test": -2.0}, "b": {"score_test": -1.0}},
        "model_comparison": {
            "problem_type": "regression",
            "primary_metric": "rmse",
            "higher_is_better": False,
            "results": [
                {"model_name": "a", "problem_type": "regression", "status": "success", "metrics": {"rmse": 2.0}},
                {"model_name": "b", "problem_type": "regression", "status": "success", "metrics": {"rmse": 1.0}},
            ],
            "ranked_model_names": ["b", "a"],
            "winner": "b",
            "winner_reasoning": "b wins",
        },
    }
    plan = ExperimentPlan(problem_type=ProblemType.REGRESSION, target_column="y")

    def fail_if_called(*args, **kwargs):
        raise AssertionError("LLM must not be called once max_retries is reached")

    import agents.evaluator as evaluator_module

    monkeypatch.setattr(evaluator_module, "call_llm_json", fail_if_called)

    decision = evaluate_results(metrics, plan, retry_count=2, max_retries=2)

    assert decision.decision == "proceed"
    assert decision.best_model == "b"


def test_evaluate_results_falls_back_when_no_model_comparison_present(monkeypatch):
    """per_entity/hierarchical scopes don't populate model_comparison - the
    deterministic fallback (build_comparison_from_score_test) must still
    produce a winner, and the LLM still can't override it.
    """
    metrics = {
        "eval_metric": "rmse",
        "models": {"x": {"score_test": 0.3, "fit_time_s": 1.0}, "y": {"score_test": 0.9, "fit_time_s": 1.0}},
    }
    plan = ExperimentPlan(problem_type=ProblemType.REGRESSION, target_column="target")

    def fake_call_llm_json(system_prompt, user_prompt, schema, **kwargs):
        return EvaluatorDecision(decision="proceed", best_model="x", reasoning="wrong on purpose")

    import agents.evaluator as evaluator_module

    monkeypatch.setattr(evaluator_module, "call_llm_json", fake_call_llm_json)

    decision = evaluate_results(metrics, plan, retry_count=0, max_retries=2)

    assert decision.best_model == "y"  # highest score_test, not the LLM's pick
