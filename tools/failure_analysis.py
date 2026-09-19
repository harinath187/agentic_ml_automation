"""Deterministic failure classification and safe LLM explanations."""
from __future__ import annotations

import json
from typing import Optional

from agents.llm_client import LLMCallError, call_llm_json
from agents.schemas import DataQualityReport, DatasetProfile, FailureExplanation, ExperimentPlan


FAILURE_MESSAGES = {
    "missing_dependency": "This model was skipped because a required Python package is not installed.",
    "unsupported_validation_strategy": "This model was skipped because it does not support the selected validation method.",
    "invalid_input": "The model could not run because the training input was invalid.",
    "insufficient_data": "The dataset does not contain enough usable data for this model or validation setup.",
    "preprocessing_failure": "The model could not run because the feature preparation step failed.",
    "training_failure": "The model failed while learning from the training data.",
    "prediction_failure": "The model trained, but could not produce predictions for the evaluation data.",
    "scoring_failure": "The model produced output, but the evaluation metrics could not be calculated.",
    "unknown": "The model failed for an unclassified reason.",
}


def classify_model_failure(status: str, error: Optional[str]) -> tuple[str, str]:
    message = (error or "").lower()
    if status == "skipped_missing_dependency":
        category = "missing_dependency"
    elif status == "skipped_unsupported_validation_strategy":
        category = "unsupported_validation_strategy"
    elif "scoring failed" in message:
        category = "scoring_failure"
    elif "predict" in message:
        category = "prediction_failure"
    elif any(token in message for token in ("column", "class", "empty", "invalid", "nan")):
        category = "invalid_input"
    elif any(token in message for token in ("feature", "transform", "encode", "scale")):
        category = "preprocessing_failure"
    else:
        category = "training_failure" if status == "failed" else "unknown"
    return category, FAILURE_MESSAGES[category]


def annotate_model_failure(result) -> None:
    if result.status == "success":
        result.failure_category = None
        result.failure_message = None
        return
    category, message = classify_model_failure(result.status, result.errors)
    result.failure_category = category
    result.failure_message = message


def build_failure_summary(
    metrics: dict,
    dataset_profile: Optional[DatasetProfile],
    quality_report: Optional[DataQualityReport],
    plan: Optional[ExperimentPlan],
) -> Optional[dict]:
    failures = []
    for result in metrics.get("candidate_results", []):
        if result.get("status") == "success":
            continue
        category, message = classify_model_failure(result.get("status", "unknown"), result.get("errors"))
        failures.append({
            "model_name": result.get("model_name", "unknown"),
            "status": result.get("status", "unknown"),
            "category": category,
            "message": message,
            "technical_error": str(result.get("errors") or "")[:500],
        })

    dataset_issues = []
    if quality_report:
        if quality_report.overall_quality_score < 60:
            dataset_issues.append({"category": "low_quality_score", "detail": f"Quality score is {quality_report.overall_quality_score:.1f}/100."})
        if quality_report.missing_value_columns:
            dataset_issues.append({"category": "missing_values", "detail": f"{len(quality_report.missing_value_columns)} column(s) contain missing values."})
        if quality_report.constant_columns:
            dataset_issues.append({"category": "constant_columns", "detail": f"Constant columns: {', '.join(quality_report.constant_columns[:10])}."})
        if quality_report.possible_leakage_columns:
            dataset_issues.append({"category": "possible_leakage", "detail": f"Possible leakage columns: {', '.join(quality_report.possible_leakage_columns[:10])}."})
    if dataset_profile and dataset_profile.row_count == 0:
        dataset_issues.append({"category": "empty_dataset", "detail": "The dataset contains no rows."})
    if plan and plan.problem_type and not plan.feature_columns:
        dataset_issues.append({"category": "no_features", "detail": "No usable feature columns were selected."})

    if not failures and not dataset_issues:
        return None
    suitability = "suitable_with_warnings" if dataset_issues else "suitable"
    if any(issue["category"] in {"empty_dataset", "no_features"} for issue in dataset_issues):
        suitability = "not_suitable"
    return {
        "status": "issues_detected",
        "dataset_suitability": suitability,
        "model_failures": failures,
        "dataset_issues": dataset_issues,
        "successful_model_count": sum(1 for result in metrics.get("candidate_results", []) if result.get("status") == "success"),
        "candidate_count": len(metrics.get("candidate_results", [])),
    }


SYSTEM_PROMPT = """You explain deterministic ML pipeline issues for a business user.
Use only the supplied structured issue summary. Do not invent causes, metrics,
model status, or dataset facts. Do not propose a different winning model.
Return a short explanation and practical next steps. Never mention raw rows.
"""


def explain_failure_summary(summary: Optional[dict]) -> Optional[dict]:
    if not summary:
        return None
    try:
        explanation = call_llm_json(
            SYSTEM_PROMPT,
            "Issue summary (JSON):\n" + json.dumps(summary, separators=(",", ":")),
            FailureExplanation,
            caller_name="tools.failure_analysis",
        )
        return explanation.model_dump(mode="json")
    except Exception as exc:  # noqa: BLE001 - explanation must never fail the run
        return {
            "summary": "Some models or dataset checks reported issues. See the detailed deterministic messages below.",
            "severity": "warning",
            "recommendations": [],
            "llm_unavailable": True,
            "llm_error": str(exc)[:300],
        }