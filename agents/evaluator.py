"""Evaluator Agent: decides whether to retry or proceed - but does NOT pick
the winning model. tools/evaluation.py's deterministic ranking engine
already decided that (see tools/model_runner.py, which populates
metrics["model_comparison"]); this module only reads that decision and
enforces it, overriding EvaluatorDecision.best_model with the ranking
engine's winner regardless of what the LLM returns. The LLM's only real
say is proceed-vs-retry and the narrative `reasoning`.

The prompt sends evaluation.to_llm_summary(comparison) - a trimmed view -
rather than the raw `metrics` dict and the full `comparison` object
separately; those two used to carry the same model_comparison data twice in
one prompt, which is what tipped a real run over Groq's free-tier TPM cap
(413) here first, ahead of agents/recommender.py and agents/reporter.py.
"""
from __future__ import annotations

import json

from agents.llm_client import call_llm_json
from agents.schemas import EvaluatorDecision, ExperimentPlan, ProblemType
from tools import evaluation

SYSTEM_PROMPT = """You are the Evaluator Agent in an automated ML pipeline.
A deterministic Python ranking engine has ALREADY decided which model
performed best on the validation metric that matters for this problem type -
you are given that winner and the full model comparison as ground truth.
You do NOT choose the winner, and naming a different model as best_model
has no effect (it will be overridden). If you think a different model
would be a better real-world choice (e.g. for interpretability or training
time), say so in `reasoning` - you cannot change the selection itself.

Decide whether to:
- "proceed": accept the deterministic winner and move on.
- "retry": ONLY if every candidate performed poorly for this kind of
  problem (not merely because you would have preferred a different model),
  suggest concrete, actionable adjustments (different imputation strategy,
  different features, or additional model families to try). Do not retry
  indefinitely.

Note: when the plan's scope_strategy is "per_entity" or "hierarchical", each
result's metrics are an aggregate (averaged across entities, or the joint
fit) rather than a single model's own numbers. Mention this in your
reasoning when relevant (e.g. if you'd want to know the entity-level
breakdown before fully trusting an averaged score).
"""


def evaluate_results(
    metrics: dict, plan: ExperimentPlan, retry_count: int, max_retries: int
) -> EvaluatorDecision:
    comparison = _get_or_build_comparison(metrics, plan)
    deterministic_winner = comparison.winner

    if retry_count >= max_retries:
        return EvaluatorDecision(
            decision="proceed",
            best_model=deterministic_winner,
            reasoning=(
                f"Max retries ({max_retries}) reached; proceeding with the deterministic "
                f"ranking engine's winner. {comparison.winner_reasoning}"
            ),
        )

    user_prompt = (
        f"Pipeline plan (JSON):\n{plan.model_dump_json()}\n\n"
        f"Deterministic model comparison summary (JSON) - the winner is ALREADY "
        f"DECIDED, you cannot change it:\n{json.dumps(evaluation.to_llm_summary(comparison), separators=(',', ':'))}\n\n"
        f"Retry count so far: {retry_count} / max {max_retries}"
    )
    decision = call_llm_json(SYSTEM_PROMPT, user_prompt, EvaluatorDecision, caller_name="agents.evaluator")

    # The ranking engine, not the LLM, determines the winner - enforced here
    # unconditionally, regardless of what best_model the LLM returned.
    decision.best_model = deterministic_winner
    return decision


def _get_or_build_comparison(metrics: dict, plan: ExperimentPlan) -> evaluation.ModelComparison:
    raw_comparison = metrics.get("model_comparison")
    if raw_comparison:
        return evaluation.ModelComparison.model_validate(raw_comparison)

    # per_entity/hierarchical scopes bypass tools/model_runner.py today (see
    # orchestration/graph.py) and only have the old flat score_test shape -
    # rank directly off that instead of leaving them without a comparison.
    problem_type = plan.problem_type or ProblemType.REGRESSION
    return evaluation.build_comparison_from_score_test(problem_type, metrics.get("models", {}))
