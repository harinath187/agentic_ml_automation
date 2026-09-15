"""Tests for the Recommendation Agent (agents/recommender.py).

Critical rule under test: the LLM must NOT change the model ranking, and
cannot invent metrics or reference unevaluated models. validate_recommendation
is the deterministic enforcement layer - every test here calls it directly
(or through generate_recommendation with a monkeypatched LLM) rather than
trusting prompt wording alone.
"""
import pytest

from agents.recommender import generate_recommendation, validate_recommendation
from agents.schemas import (
    EvaluatorDecision,
    ExperimentPlan,
    ProblemType,
    RecommendationOutput,
)
from tools.evaluation import EvaluationResult, ModelComparison


def _comparison(winner="sarima", extra_results=None) -> ModelComparison:
    results = [
        EvaluationResult(model_name="sarima", problem_type="forecasting", status="success", metrics={"rmse": 1.2}),
        EvaluationResult(model_name="ets", problem_type="forecasting", status="success", metrics={"rmse": 3.4}),
        EvaluationResult(model_name="naive", problem_type="forecasting", status="failed", errors="boom"),
    ]
    if extra_results:
        results.extend(extra_results)
    return ModelComparison(
        problem_type="forecasting",
        primary_metric="rmse",
        higher_is_better=False,
        results=results,
        ranked_model_names=["sarima", "ets"],
        winner=winner,
        winner_reasoning=f"{winner!r} has the lowest rmse.",
    )


# --- valid recommendation output ---------------------------------------------


def test_valid_recommendation_passes_through_unchanged():
    comparison = _comparison()
    output = RecommendationOutput(
        recommended_model="sarima",
        reason="Lowest RMSE among evaluated candidates.",
        performance_summary="SARIMA achieved an RMSE of 1.2 on the holdout period.",
        comparison_to_alternatives="ETS was considerably worse (RMSE 3.4); naive failed to run.",
        business_interpretation="Forecasts should be accurate to within about 1.2 units.",
        limitations="Only one holdout period was evaluated.",
        confidence_statement="Moderate confidence given the small test set.",
        cited_metrics={"rmse": 1.2},
        alternative_models_mentioned=["ets", "naive"],
    )

    result = validate_recommendation(output, comparison)

    assert result.recommended_model == "sarima"
    assert result.cited_metrics == {"rmse": 1.2}
    assert result.alternative_models_mentioned == ["ets", "naive"]
    assert result.validation_notes == []  # nothing needed repairing


def test_valid_recommendation_with_legitimate_flag_is_kept():
    comparison = _comparison()
    output = RecommendationOutput(
        recommended_model="sarima",
        reason="test",
        cited_metrics={"rmse": 1.2},
        flagged_for_review=True,
        flag_reason="The dataset profile shows very few rows for a seasonal model; treat with caution.",
    )

    result = validate_recommendation(output, comparison)

    assert result.recommended_model == "sarima"  # flag never changes the recommendation
    assert result.flagged_for_review is True
    assert result.flag_reason


# --- invalid recommendation output: the LLM tries to change the ranking -----


def test_llm_naming_a_different_winner_is_overridden():
    comparison = _comparison(winner="sarima")
    output = RecommendationOutput(recommended_model="ets", reason="I prefer ets")

    result = validate_recommendation(output, comparison)

    assert result.recommended_model == "sarima"
    assert result.recommended_model != "ets"
    assert any("overridden" in note for note in result.validation_notes)


def test_llm_naming_the_failed_model_is_overridden():
    comparison = _comparison(winner="sarima")
    output = RecommendationOutput(recommended_model="naive")  # naive actually failed

    result = validate_recommendation(output, comparison)

    assert result.recommended_model == "sarima"


def test_llm_omitting_recommended_model_still_gets_the_winner():
    comparison = _comparison(winner="sarima")
    output = RecommendationOutput(recommended_model=None)

    result = validate_recommendation(output, comparison)

    assert result.recommended_model == "sarima"


# --- invalid recommendation output: invented/mismatched metrics -------------


def test_invented_metric_not_in_evaluation_results_is_dropped():
    comparison = _comparison(winner="sarima")
    output = RecommendationOutput(
        recommended_model="sarima",
        cited_metrics={"rmse": 1.2, "mape": 42.0},  # mape was never computed for sarima
    )

    result = validate_recommendation(output, comparison)

    assert result.cited_metrics == {"rmse": 1.2}
    assert "mape" not in result.cited_metrics
    assert any("mape" in note for note in result.validation_notes)


def test_mismatched_metric_value_is_dropped():
    comparison = _comparison(winner="sarima")
    output = RecommendationOutput(
        recommended_model="sarima",
        cited_metrics={"rmse": 999.0},  # real value is 1.2
    )

    result = validate_recommendation(output, comparison)

    assert "rmse" not in result.cited_metrics
    assert any("rmse" in note for note in result.validation_notes)


def test_metric_value_close_enough_is_kept():
    comparison = _comparison(winner="sarima")
    output = RecommendationOutput(recommended_model="sarima", cited_metrics={"rmse": 1.2001})

    result = validate_recommendation(output, comparison)

    assert result.cited_metrics == {"rmse": 1.2}  # snapped to the ground-truth value


def test_all_metrics_dropped_when_winner_has_none():
    results = [EvaluationResult(model_name="sarima", problem_type="forecasting", status="success", metrics={})]
    comparison = ModelComparison(
        problem_type="forecasting", primary_metric="rmse", higher_is_better=False,
        results=results, ranked_model_names=["sarima"], winner="sarima", winner_reasoning="only one",
    )
    output = RecommendationOutput(recommended_model="sarima", cited_metrics={"rmse": 1.2})

    result = validate_recommendation(output, comparison)

    assert result.cited_metrics == {}


# --- invalid recommendation output: unsupported/unevaluated model names ----


def test_unevaluated_alternative_model_name_is_dropped():
    comparison = _comparison(winner="sarima")
    output = RecommendationOutput(
        recommended_model="sarima",
        alternative_models_mentioned=["ets", "arima"],  # arima was never evaluated in this comparison
    )

    result = validate_recommendation(output, comparison)

    assert result.alternative_models_mentioned == ["ets"]
    assert any("arima" in note for note in result.validation_notes)


def test_flag_without_reason_is_cleared():
    comparison = _comparison(winner="sarima")
    output = RecommendationOutput(recommended_model="sarima", flagged_for_review=True, flag_reason=None)

    result = validate_recommendation(output, comparison)

    assert result.flagged_for_review is False


def test_winner_not_in_results_is_noted_as_anomaly():
    comparison = _comparison(winner="ghost_model")  # winner name doesn't exist in results - shouldn't happen, but must not crash
    output = RecommendationOutput(recommended_model="sarima")

    result = validate_recommendation(output, comparison)

    assert result.recommended_model == "ghost_model"  # still trusts the evaluation engine's own winner field
    assert any("anomaly" in note for note in result.validation_notes)


# --- no-winner case: recommendation must not fabricate one -------------------


def test_no_winner_skips_llm_and_returns_no_recommendation(monkeypatch):
    comparison = ModelComparison(
        problem_type="regression", primary_metric="rmse", higher_is_better=False,
        results=[EvaluationResult(model_name="a", problem_type="regression", status="failed", errors="x")],
        ranked_model_names=[], winner=None, winner_reasoning="no candidate produced a usable value",
    )
    plan = ExperimentPlan(problem_type=ProblemType.REGRESSION, target_column="y")
    decision = EvaluatorDecision(decision="proceed", best_model=None, reasoning="nothing trained")

    def fail_if_called(*args, **kwargs):
        raise AssertionError("LLM must not be called when there is no winner to explain")

    import agents.recommender as recommender_module

    monkeypatch.setattr(recommender_module, "call_llm_json", fail_if_called)

    result = generate_recommendation(None, None, plan, comparison, decision)

    assert result.recommended_model is None
    assert result.cited_metrics == {}


# --- end-to-end through generate_recommendation ------------------------------


def test_generate_recommendation_calls_llm_and_still_enforces_winner(monkeypatch):
    comparison = _comparison(winner="sarima")
    plan = ExperimentPlan(problem_type=ProblemType.FORECASTING, target_column="sales", time_column="date")
    decision = EvaluatorDecision(decision="proceed", best_model="sarima", reasoning="test")

    captured = {}

    def fake_call_llm_json(system_prompt, user_prompt, schema, **kwargs):
        captured["user_prompt"] = user_prompt
        return RecommendationOutput(recommended_model="ets", reason="the LLM disagrees")  # tries to override

    import agents.recommender as recommender_module

    monkeypatch.setattr(recommender_module, "call_llm_json", fake_call_llm_json)

    result = generate_recommendation(None, None, plan, comparison, decision)

    assert result.recommended_model == "sarima"
    assert "sarima" in captured["user_prompt"]


# --- Phase 7: cited_top_features must not contradict calculated importance --


def _comparison_with_explainability(winner="rf") -> ModelComparison:
    results = [
        EvaluationResult(
            model_name="rf",
            problem_type="classification",
            status="success",
            metrics={"accuracy": 0.9},
            explainability={
                "model_name": "rf",
                "problem_type": "classification",
                "supported": True,
                "feature_importance": [{"feature": "f1", "importance": 0.8}, {"feature": "f2", "importance": 0.2}],
                "permutation_importance": [{"feature": "f1", "importance": 0.5}],
                "shap_importance": [],
            },
        ),
        EvaluationResult(
            model_name="baseline",
            problem_type="classification",
            status="success",
            metrics={"accuracy": 0.5},
            explainability={"model_name": "baseline", "problem_type": "classification", "supported": False},
        ),
    ]
    return ModelComparison(
        problem_type="classification",
        primary_metric="accuracy",
        higher_is_better=True,
        results=results,
        ranked_model_names=["rf", "baseline"],
        winner=winner,
        winner_reasoning="rf wins",
    )


def test_valid_cited_top_features_are_kept():
    comparison = _comparison_with_explainability()
    output = RecommendationOutput(recommended_model="rf", cited_top_features=["f1", "f2"])

    result = validate_recommendation(output, comparison)

    assert result.cited_top_features == ["f1", "f2"]
    assert result.validation_notes == []


def test_invented_feature_not_in_importance_results_is_dropped():
    comparison = _comparison_with_explainability()
    output = RecommendationOutput(recommended_model="rf", cited_top_features=["f1", "totally_made_up_feature"])

    result = validate_recommendation(output, comparison)

    assert result.cited_top_features == ["f1"]
    assert any("totally_made_up_feature" in note for note in result.validation_notes)


def test_cited_top_features_cleared_when_winner_explainability_unsupported():
    comparison = _comparison_with_explainability(winner="baseline")
    output = RecommendationOutput(recommended_model="baseline", cited_top_features=["f1"])

    result = validate_recommendation(output, comparison)

    assert result.cited_top_features == []
    assert any("could not be verified" in note for note in result.validation_notes)


def test_cited_top_features_empty_when_llm_cites_nothing():
    comparison = _comparison_with_explainability()
    output = RecommendationOutput(recommended_model="rf", cited_top_features=[])

    result = validate_recommendation(output, comparison)

    assert result.cited_top_features == []
    assert result.validation_notes == []  # nothing to repair


# --- Phase 9 413-fix: validation must be unaffected by the prompt's trim ----


def test_feature_beyond_prompt_top_n_still_validates_against_full_comparison():
    """generate_recommendation's prompt only shows the LLM
    to_llm_summary()'s top-N feature names (see agents/recommender.py), but
    validate_recommendation always checks against the full, untrimmed
    ModelComparison - so a feature ranked below that cutoff (here, the 7th of
    7) must still be accepted if the LLM somehow cites it. Trimming the
    prompt must never trim what gets validated.
    """
    full_importance = [{"feature": f"f{i}", "importance": float(7 - i)} for i in range(7)]
    results = [
        EvaluationResult(
            model_name="rf",
            problem_type="classification",
            status="success",
            metrics={"accuracy": 0.9},
            explainability={
                "model_name": "rf",
                "problem_type": "classification",
                "supported": True,
                "feature_importance": full_importance,
                "permutation_importance": [],
                "shap_importance": [],
            },
        ),
    ]
    comparison = ModelComparison(
        problem_type="classification",
        primary_metric="accuracy",
        higher_is_better=True,
        results=results,
        ranked_model_names=["rf"],
        winner="rf",
        winner_reasoning="only candidate",
    )

    # A to_llm_summary() prompt built from this comparison would only ever
    # show the LLM the top 5 feature names - "f6" (rank 7 of 7) is not
    # among them, yet it is a real entry in the full explainability data.
    from tools.evaluation import to_llm_summary

    prompt_features = next(r["top_features"] for r in to_llm_summary(comparison, top_n_features=5)["results"])
    assert "f6" not in prompt_features  # confirms the prompt really is trimmed

    output = RecommendationOutput(recommended_model="rf", cited_top_features=["f6"])
    result = validate_recommendation(output, comparison)

    assert result.cited_top_features == ["f6"]  # still validated against the FULL comparison
    assert result.validation_notes == []
