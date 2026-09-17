"""Recommendation Agent: turns the deterministic evaluation engine's winner
(tools/evaluation.py) into a business-facing recommendation - reason,
performance summary, comparison to alternatives, model explanation, business
interpretation, limitations, and a qualified confidence statement.

CRITICAL: this agent does not, and cannot, change the model ranking. The
deterministic ModelComparison.winner IS the recommendation; recommended_model
is unconditionally overwritten with it after the LLM call
(validate_recommendation), regardless of what the LLM returns. The LLM may
flag a suspected data/configuration inconsistency (flagged_for_review +
flag_reason) but that never substitutes a different model - it only
surfaces a concern for a human to review. Every number in cited_metrics is
checked against the actual evaluation results and dropped if it doesn't
match, and every feature name in cited_top_features is checked against the
winner's actual tools/explainability.py results (feature/permutation/SHAP
importance) and dropped if it doesn't match - so the LLM cannot invent
metrics OR contradict the calculated feature importance (Phase 7).

The report (reports/template.html) keeps this agent's three kinds of output
in clearly separate sections: model PERFORMANCE (tools/evaluation.py's
metrics, not written by this agent), model EXPLANATION (explanation_narrative/
cited_top_features - HOW the model works), and business INTERPRETATION
(business_interpretation - WHAT it means for the business).
"""
from __future__ import annotations

import json
import math
from typing import Optional

from agents.llm_client import call_llm_json
from agents.schemas import (
    DatasetProfile,
    EvaluatorDecision,
    ExperimentPlan,
    ProblemDefinition,
    RecommendationOutput,
    ValidationStrategy,
)
from tools.evaluation import ModelComparison, to_llm_summary

SYSTEM_PROMPT = """You are the Recommendation Agent in an automated ML pipeline.

A deterministic Python evaluation engine has ALREADY ranked every trained
model candidate and decided the winner - you are given that ModelComparison
as ground truth, along with the dataset profile, problem definition,
experiment plan, and evaluator decision. Your job is to explain and
contextualize that decision for a business audience, NOT to re-rank models.

CRITICAL RULES:
1. recommended_model MUST be the ModelComparison's `winner`. Naming a
   different model has no effect - it will be overridden by validation.
2. Every number in cited_metrics MUST be copied EXACTLY from the winning
   model's entry in ModelComparison.results (its `metrics` dict). Never
   invent, round differently, or estimate a metric that isn't present there.
3. Every model name you put in alternative_models_mentioned MUST be one of
   the OTHER model names present in ModelComparison.results. Never reference
   a model that wasn't actually evaluated.
4. If you notice something that looks like a genuine data or configuration
   inconsistency in the evaluation results (e.g. the winner's metric looks
   implausible given the dataset profile, or the validation strategy looks
   mismatched for the problem type), set flagged_for_review=true and explain
   why in flag_reason. This does NOT change recommended_model - it only
   raises a concern for a human to look at. Do not use this as a way to
   recommend a different model than the winner.
5. explanation_narrative is a MODEL EXPLANATION, not a business statement:
   describe HOW the winning model reaches its predictions, using the
   winner's explainability data - ModelComparison.results[*].explainability
   (feature_importance/permutation_importance/shap_importance) for
   classification/regression, or ModelComparison.dataset_explainability
   (trend_detected/seasonality_detected/trend_strength/seasonal_strength)
   for forecasting. If explainability is unsupported for this model
   (`explainability.supported == false`), say so plainly instead of
   guessing - do not fabricate an explanation.
6. cited_top_features MUST be feature names that actually appear in the
   winner's explainability importance lists, in an order consistent with
   their actual ranking there. Never name a feature as important that isn't
   in those results, and never invert the real ranking.

Write for a business reader: plain language, concrete numbers taken from
cited_metrics, and an honest confidence_statement (e.g. call out a small
test set, a low overall_quality_score, or skipped/failed candidates, when
relevant) rather than false certainty. Keep explanation_narrative (how the
model works) and business_interpretation (what it means for the business)
distinct - do not merge them into one field.
"""


def generate_recommendation(
    dataset_profile: Optional[DatasetProfile],
    problem_definition: Optional[ProblemDefinition],
    plan: ExperimentPlan,
    model_comparison: ModelComparison,
    decision: EvaluatorDecision,
    validation_strategy: Optional[ValidationStrategy] = None,
) -> RecommendationOutput:
    if model_comparison.winner is None:
        # Nothing to recommend - skip the LLM call entirely rather than ask
        # it to explain a winner that doesn't exist.
        return RecommendationOutput(
            recommended_model=None,
            reason="No candidate model produced a usable result, so no recommendation can be made.",
            performance_summary=model_comparison.winner_reasoning,
            confidence_statement="No confidence - re-run with different candidates, more data, or a longer training budget.",
            validation_notes=["Skipped the LLM call: model_comparison.winner is None."],
        )

    prompt_parts = [
        f"Dataset profile (JSON):\n{dataset_profile.model_dump_json()}"
        if dataset_profile is not None
        else "Dataset profile: not available.",
        f"Problem definition (JSON):\n{problem_definition.model_dump_json()}"
        if problem_definition is not None
        else "Problem definition: not available.",
        f"Experiment plan (JSON):\n{plan.model_dump_json()}",
        f"Deterministic model comparison summary (JSON) - the winner is ALREADY "
        f"DECIDED, you cannot change it:\n{json.dumps(to_llm_summary(model_comparison), separators=(',', ':'))}",
        f"Evaluator decision (JSON):\n{decision.model_dump_json()}",
    ]
    if validation_strategy is not None:
        prompt_parts.append(f"Validation strategy (JSON):\n{validation_strategy.model_dump_json()}")

    user_prompt = "\n\n".join(prompt_parts)
    output = call_llm_json(SYSTEM_PROMPT, user_prompt, RecommendationOutput, caller_name="agents.recommender")
    # validate_recommendation checks the LLM's claims against model_comparison
    # (the full, untrimmed object) - never against the summary above. Trimming
    # what's sent to the LLM must never trim what's validated against it.
    return validate_recommendation(output, model_comparison)


def validate_recommendation(output: RecommendationOutput, model_comparison: ModelComparison) -> RecommendationOutput:
    """Validates against the FULL comparison object, not the trimmed
    to_llm_summary() view used in the prompt above - the prompt is allowed to
    be lossy (fewer features, no per-model timing/errors) precisely because
    every claim the LLM makes is checked back against the real, untrimmed
    `model_comparison` here. Never validate against the summary instead.

    Deterministic enforcement layer - never trusts the LLM's word alone:
      - recommended_model exists and is forced to the deterministic winner
      - cited_metrics values match the winner's actual evaluation metrics
      - cited_top_features only names features in the winner's actual
        calculated feature/permutation/SHAP importance (Phase 7)
      - alternative_models_mentioned only names models that were actually evaluated
      - flagged_for_review requires a flag_reason (never a silent flag)
    Every repair is logged in validation_notes rather than applied silently.
    """
    output = output.model_copy(deep=True)
    notes: list[str] = list(output.validation_notes)

    winner = model_comparison.winner
    known_model_names = {r.model_name for r in model_comparison.results}

    if winner is not None and winner not in known_model_names:
        notes.append(
            f"model_comparison.winner {winner!r} is not among the evaluated results - this is an "
            "anomaly in the evaluation engine's own output, not the LLM's."
        )

    if output.recommended_model != winner:
        notes.append(
            f"LLM proposed recommended_model {output.recommended_model!r}; overridden to the "
            f"deterministic winner {winner!r} - the evaluation engine decides the ranking, not the LLM."
        )
    output.recommended_model = winner

    winner_result = next((r for r in model_comparison.results if r.model_name == winner), None)
    winner_metrics = winner_result.metrics if winner_result else {}

    verified_metrics: dict[str, float] = {}
    for name, value in output.cited_metrics.items():
        actual = winner_metrics.get(name)
        if actual is None or not _isclose(actual, value):
            notes.append(f"cited_metrics[{name!r}] = {value!r} did not match the evaluation results ({actual!r}); removed.")
            continue
        verified_metrics[name] = actual
    output.cited_metrics = verified_metrics

    known_top_features = _known_top_features(winner_result.explainability if winner_result else None)
    if known_top_features:
        valid_features = [f for f in output.cited_top_features if f in known_top_features]
        removed_features = [f for f in output.cited_top_features if f not in known_top_features]
        if removed_features:
            notes.append(
                f"cited_top_features referenced feature(s) not found in the calculated importance "
                f"results {removed_features!r}; removed."
            )
        output.cited_top_features = valid_features
    elif output.cited_top_features:
        notes.append(
            "cited_top_features could not be verified (no explainability importance data available "
            "for this model); cleared."
        )
        output.cited_top_features = []

    valid_alternatives = [name for name in output.alternative_models_mentioned if name in known_model_names]
    removed_alternatives = [name for name in output.alternative_models_mentioned if name not in known_model_names]
    if removed_alternatives:
        notes.append(f"alternative_models_mentioned referenced unevaluated model(s) {removed_alternatives!r}; removed.")
    output.alternative_models_mentioned = valid_alternatives

    if output.flagged_for_review and not output.flag_reason:
        notes.append("flagged_for_review was true without a flag_reason; cleared the flag.")
        output.flagged_for_review = False

    output.validation_notes = notes
    return output


def _known_top_features(explainability: Optional[dict]) -> set[str]:
    """Every feature name that appears in ANY of the winner's calculated
    importance results (tools/explainability.py) - SHAP, permutation, or
    native feature importance. Empty when explainability wasn't computed or
    isn't supported for this model, in which case validate_recommendation
    can't verify any claim and clears cited_top_features entirely rather
    than trusting the LLM.
    """
    if not explainability:
        return set()
    names: set[str] = set()
    for key in ("shap_importance", "permutation_importance", "feature_importance"):
        for entry in explainability.get(key) or []:
            name = entry.get("feature") if isinstance(entry, dict) else None
            if name:
                names.add(name)
    return names


def _isclose(a: float, b: float, rel_tol: float = 1e-3, abs_tol: float = 1e-6) -> bool:
    try:
        return math.isclose(float(a), float(b), rel_tol=rel_tol, abs_tol=abs_tol)
    except (TypeError, ValueError):
        return False
