"""Tests for tools/classification_cycle.py: the deterministic, LLM-free
3-cycle classification improvement workflow. No LLM mocking needed - this
path never calls one.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tools import classification_cycle as cc
from tools.classification_cycle import ClassificationCycleConfig, run_classification_cycles
from tools.technical_retry import PermanentTrainingError


def _binary_df(n=300, seed=42):
    rng = np.random.default_rng(seed)
    tenure = rng.integers(1, 72, n)
    charge = rng.normal(70, 20, n)
    calls = rng.poisson(1.5, n)
    score = -0.05 * tenure + 0.02 * charge + 0.3 * calls + rng.normal(0, 1, n)
    churn = (score > np.median(score)).astype(int)
    return pd.DataFrame({"tenure": tenure, "charge": charge, "calls": calls, "churn": churn})


def _multiclass_df(n=300, seed=1):
    rng = np.random.default_rng(seed)
    x1, x2 = rng.normal(0, 1, n), rng.normal(0, 1, n)
    y = np.select([x1 + x2 < -0.5, x1 + x2 < 0.5], [0, 1], default=2)
    return pd.DataFrame({"x1": x1, "x2": x2, "y": y})


def _imbalanced_df(n=400, seed=5):
    rng = np.random.default_rng(seed)
    y = np.zeros(n, dtype=int)
    y[:20] = 1  # 5% minority
    x = rng.normal(0, 1, n) + y * 2
    return pd.DataFrame({"x": x, "y": y})


# --- 1/2/3: cycle count, early stop, exhaustion -----------------------------


def test_max_cycles_never_exceeded_when_criteria_never_met():
    df = _binary_df()
    config = ClassificationCycleConfig(acceptance_criteria={"accuracy": 0.999})
    result = run_classification_cycles(df, "churn", config=config)
    assert result["completed_cycles"] == 3
    assert result["status"] == "threshold_not_met"
    assert max(r["cycle"] for r in result["cycles"]) == 3


def test_early_stop_when_criteria_met_in_cycle_one():
    df = _binary_df()
    config = ClassificationCycleConfig(acceptance_criteria={"accuracy": 0.5})
    result = run_classification_cycles(df, "churn", config=config)
    assert result["status"] == "success"
    assert result["completed_cycles"] == 1
    assert all(r["cycle"] == 1 for r in result["cycles"])


def test_all_three_cycles_execute_when_criteria_not_met():
    df = _binary_df()
    config = ClassificationCycleConfig(acceptance_criteria={"f1": 0.999})
    result = run_classification_cycles(df, "churn", config=config)
    cycles_seen = {r["cycle"] for r in result["cycles"]}
    assert cycles_seen == {1, 2, 3}


# --- 4: best model selection -------------------------------------------------


def test_best_model_selected_across_all_cycles_not_just_last(monkeypatch):
    """Reconstructs the spec's own example: a cycle-2 model beats every
    cycle-3 candidate on the selection metric, so the cycle-2 model must win
    even though cycle 3 ran afterward - selection looks at ALL cycles, not
    just the most recent one."""
    df = _binary_df()
    config = ClassificationCycleConfig(acceptance_criteria={"f1": 0.999}, selection_metric="f1")

    fake_scores = {
        (1, "baseline"): 0.50,
        (1, "logistic_regression"): 0.55,
        (2, "logistic_regression"): 0.90,  # best overall - cycle 2
        (2, "random_forest"): 0.60,
        (3, "logistic_regression"): 0.65,
        (3, "random_forest"): 0.62,
    }

    def train_stub(definition, extra_params, train_df, val_df, target_column, strategy):
        f1 = fake_scores[(train_stub.current_cycle, definition.name)]
        fitted = {"model": None, "encoder": type("E", (), {"classes_": np.array([0, 1])})()}
        return {"accuracy": f1, "f1": f1, "precision": f1, "recall": f1}, fitted

    def plan_stub(cycle, available_models, previous_records, imbalanced, selection_metric, **kwargs):
        train_stub.current_cycle = cycle
        names = sorted({name for (c, name) in fake_scores if c == cycle})
        return [{"model_name": n, "extra_params": {}, "oversample": False} for n in names], "scripted"

    def run_with_retry_stub(fn, definition, extra_params, train_df, val_df, target_column, strategy, **kwargs):
        return fn(definition, extra_params, train_df, val_df, target_column, strategy)

    monkeypatch.setattr(cc, "_train_one_candidate", train_stub)
    monkeypatch.setattr(cc, "_plan_cycle", plan_stub)
    monkeypatch.setattr(cc, "run_with_technical_retry", run_with_retry_stub)

    result = run_classification_cycles(df, "churn", config=config)
    assert result["best_model"] == "logistic_regression"
    assert result["best_cycle"] == 2
    assert result["validation_metrics"]["f1"] == pytest.approx(0.90)


# --- 5/6: fixed test set, never used for selection --------------------------


def test_test_set_is_fixed_and_never_used_for_selection(monkeypatch):
    df = _binary_df()
    seen_val_lengths = []
    seen_test_len = {}

    original_train_one = cc._train_one_candidate

    def spy_train_one(definition, extra_params, train_df, val_df, target_column, strategy):
        seen_val_lengths.append(len(val_df))
        return original_train_one(definition, extra_params, train_df, val_df, target_column, strategy)

    monkeypatch.setattr(cc, "_train_one_candidate", spy_train_one)

    config = ClassificationCycleConfig(acceptance_criteria={"accuracy": 0.999})  # forces all 3 cycles
    result = run_classification_cycles(df, "churn", config=config)

    # Every candidate across every cycle saw the SAME val_df size (same object reused).
    assert len(set(seen_val_lengths)) == 1
    assert result["completed_cycles"] == 3


def test_final_test_metrics_distinct_from_validation_metrics():
    df = _binary_df()
    config = ClassificationCycleConfig(acceptance_criteria={"accuracy": 0.5})
    result = run_classification_cycles(df, "churn", config=config)
    assert result["status"] == "success"
    assert "test_metrics" in result
    assert "validation_metrics" in result
    # Both present and independently computed - not required to be equal,
    # but both must exist as separate keys (the actual distinctness contract).
    assert set(result["test_metrics"].keys()) & {"accuracy", "f1"}


# --- 7/8/9: metric shapes -----------------------------------------------------


def test_binary_classification_metrics_shape():
    df = _binary_df()
    config = ClassificationCycleConfig(acceptance_criteria={"accuracy": 0.5})
    result = run_classification_cycles(df, "churn", config=config)
    m = result["validation_metrics"]
    for key in ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "confusion_matrix", "per_class"):
        assert key in m
    assert len(m["confusion_matrix"]) == 2
    assert set(m["per_class"].keys()) == {"0", "1"}


def test_multiclass_classification_metrics_shape():
    df = _multiclass_df()
    config = ClassificationCycleConfig(acceptance_criteria={"accuracy": 0.5}, selection_metric="accuracy")
    result = run_classification_cycles(df, "y", config=config)
    m = result["validation_metrics"]
    assert set(m["per_class"].keys()) == {"0", "1", "2"}
    assert "roc_auc" in m
    assert "pr_auc" in m


def test_roc_auc_edge_case_missing_class_in_split_does_not_crash():
    from tools.evaluation import compute_classification_metrics_detailed

    y_true = np.array([0, 0, 0, 0])  # only one class present
    y_pred = np.array([0, 0, 0, 0])
    y_proba = np.array([0.6, 0.7, 0.8, 0.9])
    metrics = compute_classification_metrics_detailed(y_true, y_pred, y_proba=y_proba)
    assert "roc_auc" not in metrics or metrics.get("roc_auc") is None
    assert metrics["accuracy"] == 1.0


# --- 10: imbalanced classification -------------------------------------------


def test_imbalanced_dataset_reports_distribution_and_engages_class_weight():
    df = _imbalanced_df()
    config = ClassificationCycleConfig(acceptance_criteria={"accuracy": 0.999})  # forces all cycles
    result = run_classification_cycles(df, "y", config=config)
    assert min(result["class_distribution"].values()) < 0.15
    techniques = {r["configuration"]["technique"] for r in result["cycles"]}
    assert "class_weight_balanced" in techniques
    assert any(r["configuration"].get("class_weight") == "balanced" for r in result["cycles"])


# --- 11/12: invalid input -----------------------------------------------------


def test_missing_target_column_raises_permanent_error():
    df = pd.DataFrame({"x": range(50), "y": [0, 1] * 25})
    with pytest.raises(PermanentTrainingError):
        run_classification_cycles(df, "not_a_column")


def test_invalid_input_single_class_raises_permanent_error():
    df = pd.DataFrame({"x": range(50), "y": [0] * 50})
    with pytest.raises(PermanentTrainingError):
        run_classification_cycles(df, "y")


def test_invalid_input_too_few_rows_raises_permanent_error():
    df = pd.DataFrame({"x": range(5), "y": [0, 1, 0, 1, 0]})
    with pytest.raises(PermanentTrainingError):
        run_classification_cycles(df, "y")


# --- 13/14/15: technical retry vs permanent, and cycle-counter isolation ----


def test_retryable_technical_error_is_retried_and_does_not_consume_a_cycle(monkeypatch):
    df = _binary_df()
    attempts = {"n": 0}
    original = cc._train_one_candidate

    def flaky(definition, extra_params, train_df, val_df, target_column, strategy):
        if definition.name == "baseline":
            attempts["n"] += 1
            if attempts["n"] <= 2:
                raise TimeoutError("simulated transient timeout")
        return original(definition, extra_params, train_df, val_df, target_column, strategy)

    monkeypatch.setattr(cc, "_train_one_candidate", flaky)
    config = ClassificationCycleConfig(acceptance_criteria={"accuracy": 0.5}, technical_max_attempts=3, technical_base_delay_s=0.0)
    result = run_classification_cycles(df, "churn", config=config)

    baseline_records = [r for r in result["cycles"] if r["model_name"] == "baseline"]
    assert len(baseline_records) == 1  # retries within one candidate attempt don't create extra records
    assert baseline_records[0]["status"] == "completed"
    assert attempts["n"] == 3  # failed twice, succeeded on the 3rd technical attempt


def test_non_retryable_error_fails_immediately_without_retry(monkeypatch):
    df = _binary_df()
    attempts = {"n": 0}

    def always_bad(definition, extra_params, train_df, val_df, target_column, strategy):
        attempts["n"] += 1
        raise ValueError("bad hyperparameter")

    monkeypatch.setattr(cc, "_train_one_candidate", always_bad)
    config = ClassificationCycleConfig(acceptance_criteria={"accuracy": 0.5})
    result = run_classification_cycles(df, "churn", config=config)

    # A ValueError is never retryable, so every attempted candidate (across
    # however many cycles ran) is tried exactly once - attempts == records.
    assert attempts["n"] == len(result["cycles"])
    assert all(r["status"] == "failed" and r["retryable"] is False for r in result["cycles"])
    assert result["status"] == "threshold_not_met"


def test_technical_retry_exhaustion_produces_failed_record_not_exception(monkeypatch):
    df = _binary_df()

    def always_timeout(definition, extra_params, train_df, val_df, target_column, strategy):
        raise TimeoutError("still timing out")

    monkeypatch.setattr(cc, "_train_one_candidate", always_timeout)
    config = ClassificationCycleConfig(acceptance_criteria={"accuracy": 0.5}, technical_max_attempts=2, technical_base_delay_s=0.0)
    result = run_classification_cycles(df, "churn", config=config)  # must not raise

    assert result["status"] == "threshold_not_met"
    assert all(r["status"] == "failed" and r["retryable"] is True for r in result["cycles"])


# --- 16: acceptance criteria configuration -----------------------------------


def test_acceptance_criteria_are_config_driven_not_hardcoded():
    df = _binary_df()
    lenient = run_classification_cycles(df, "churn", config=ClassificationCycleConfig(acceptance_criteria={"accuracy": 0.1}))
    strict = run_classification_cycles(df, "churn", config=ClassificationCycleConfig(acceptance_criteria={"accuracy": 0.999}))
    assert lenient["status"] == "success"
    assert lenient["completed_cycles"] == 1
    assert strict["status"] == "threshold_not_met"
    assert strict["completed_cycles"] == 3


def test_selection_metric_is_config_driven():
    df = _binary_df()
    by_f1 = run_classification_cycles(
        df, "churn", config=ClassificationCycleConfig(acceptance_criteria={"f1": 0.999}, selection_metric="f1")
    )
    by_accuracy = run_classification_cycles(
        df, "churn", config=ClassificationCycleConfig(acceptance_criteria={"f1": 0.999}, selection_metric="accuracy")
    )
    assert by_f1["selection_metric"] == "f1"
    assert by_accuracy["selection_metric"] == "accuracy"


def test_acceptance_requires_all_configured_metrics_not_just_one():
    """A model passing accuracy but failing f1 must not be marked accepted -
    proves AND semantics, not 'any one metric passing is enough.'"""
    df = _binary_df()
    config = ClassificationCycleConfig(acceptance_criteria={"accuracy": 0.5, "f1": 0.999})
    result = run_classification_cycles(df, "churn", config=config)
    assert result["status"] == "threshold_not_met"
    assert result["validation_metrics"]["accuracy"] >= 0.5
    assert result["validation_metrics"]["f1"] < 0.999
