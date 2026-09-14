"""Evaluator Agent: compares model metrics and decides whether to retry or proceed.

Only receives the metrics dict (model name -> scores) and plan context -
never the underlying data.
"""
from __future__ import annotations

import json

from agents.llm_client import call_llm_json
from agents.schemas import EvaluatorDecision, PipelinePlan

SYSTEM_PROMPT = """You are the Evaluator Agent in an automated ML pipeline.
You are given the trained models' metrics for this run and the original
pipeline plan (business context, problem type, target column).

Decide whether to:
- "proceed": pick the best model considering the eval metric, but also
  interpretability and training time when relevant to the business goal.
- "retry": if all models perform poorly (e.g. the primary metric is far
  worse than a reasonable baseline for this problem type), suggest concrete
  adjustments (different imputation strategy, different features, or
  additional model families to try).

Only recommend "retry" when there's a clear, actionable adjustment - do not
retry indefinitely.

Note: when the plan's scope_strategy is "per_entity" or "hierarchical", the
metrics JSON's "models" field is an aggregate (averaged across entities, or
the joint fit) - a "per_entity" breakdown or entity count may also be
present. Pick best_model from the aggregate view and mention the entity
breakdown in your reasoning when relevant (e.g. if performance is uneven
across entities, or if many entities were skipped for having too little data).
"""


def evaluate_results(
    metrics: dict, plan: PipelinePlan, retry_count: int, max_retries: int
) -> EvaluatorDecision:
    if retry_count >= max_retries:
        best_model = _fallback_best_model(metrics)
        return EvaluatorDecision(
            decision="proceed",
            best_model=best_model,
            reasoning=f"Max retries ({max_retries}) reached; proceeding with best available model.",
        )

    user_prompt = (
        f"Pipeline plan (JSON):\n{plan.model_dump_json(indent=2)}\n\n"
        f"Model metrics (JSON):\n{json.dumps(metrics, indent=2)}\n\n"
        f"Retry count so far: {retry_count} / max {max_retries}"
    )
    return call_llm_json(SYSTEM_PROMPT, user_prompt, EvaluatorDecision)


def _fallback_best_model(metrics: dict) -> str | None:
    models = metrics.get("models", {})
    if not models:
        return None
    return max(models.items(), key=lambda kv: kv[1].get("score_test", float("-inf")))[0]
