"""Pydantic schemas shared by the Planner and Evaluator agents, plus the
deterministic Data Intelligence schemas (DatasetProfile, TargetAnalysis,
DataQualityReport) produced by tools/profiling.py before the Planner runs.

These are the ONLY structures that carry LLM input/output. They intentionally
never contain raw data rows - only column names, stats, and text.
"""
from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class ProblemType(str, Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    FORECASTING = "forecasting"
    CLUSTERING = "clustering"


class ScopeStrategy(str, Enum):
    SINGLE_ENTITY = "single_entity"
    POOLED = "pooled"
    PER_ENTITY = "per_entity"
    HIERARCHICAL = "hierarchical"


class PipelinePlan(BaseModel):
    """Structured output of the Planner Agent."""

    needs_clarification: bool = Field(
        default=False,
        description="True if the problem type/target/entity scope cannot be determined confidently.",
    )
    clarification_question: Optional[str] = Field(
        default=None,
        description="Question to ask the user when needs_clarification is True.",
    )
    problem_type: Optional[ProblemType] = Field(
        default=None, description="The identified ML problem type."
    )
    target_column: Optional[str] = Field(
        default=None, description="Column name to predict. Null for clustering."
    )
    time_column: Optional[str] = Field(
        default=None, description="Datetime column, required for forecasting."
    )
    entity_column: Optional[str] = Field(
        default=None,
        description="Column identifying the entity/group (store, customer, machine, etc.), if any.",
    )
    entity_filter_value: Optional[str] = Field(
        default=None,
        description="The specific entity value to scope to, when scope_strategy is single_entity.",
    )
    scope_strategy: Optional[ScopeStrategy] = Field(
        default=None,
        description="How to handle entity/grouping structure: single_entity, pooled, per_entity, hierarchical, or null if no grouping structure applies.",
    )
    entity_selection_reasoning: str = Field(
        default="",
        description="Explanation of the scope_strategy choice, including 'no grouping structure detected' when applicable.",
    )
    reasoning: str = Field(
        default="", description="Short explanation of why this plan was chosen."
    )
    pipeline_steps: List[str] = Field(
        default_factory=list,
        description="Ordered list of steps to run, e.g. ['clean', 'engineer_features', 'split', 'train'].",
    )
    candidate_models: List[str] = Field(
        default_factory=list,
        description="Model families to try, e.g. ['LightGBM', 'RandomForest', 'CatBoost'].",
    )


class RetryAdjustment(BaseModel):
    change_imputation_strategy: bool = False
    change_feature_engineering: bool = False
    additional_candidate_models: List[str] = Field(default_factory=list)
    notes: str = ""


class ReportContent(BaseModel):
    """Structured output of the Reporter Agent."""

    problem_summary: str = Field(default="", description="Plain-language restatement of the business problem.")
    approach_taken: str = Field(default="", description="Summary of the pipeline steps executed.")
    models_compared: str = Field(default="", description="Narrative comparison of candidate models and their scores.")
    final_recommendation: str = Field(default="", description="The recommended model and why.")
    business_impact: str = Field(default="", description="Expected business impact of using this model.")
    caveats_and_limitations: str = Field(default="", description="Known limitations, risks, or caveats.")


class EvaluatorDecision(BaseModel):
    """Structured output of the Evaluator Agent."""

    decision: str = Field(
        description="One of: 'proceed' or 'retry'."
    )
    best_model: Optional[str] = Field(
        default=None, description="Name of the best-performing model, when decision is 'proceed'."
    )
    reasoning: str = Field(default="", description="Why this model/decision was chosen.")
    retry_adjustment: Optional[RetryAdjustment] = Field(
        default=None, description="What to change, when decision is 'retry'."
    )


class ColumnKind(str, Enum):
    NUMERICAL = "numerical"
    CATEGORICAL = "categorical"
    DATETIME = "datetime"
    BOOLEAN = "boolean"
    OTHER = "other"


class ColumnProfile(BaseModel):
    """Deterministic per-column profile. No raw values - aggregated stats only."""

    name: str
    dtype: str
    inferred_kind: ColumnKind
    missing_count: int = 0
    missing_pct: float = 0.0
    unique_count: int = 0
    unique_pct: float = 0.0
    is_constant: bool = False
    is_near_constant: bool = False
    numeric_stats: Optional[dict] = Field(
        default=None, description="min/max/mean/median/std, only for numerical columns."
    )
    datetime_range: Optional[dict] = Field(
        default=None, description="min/max/distinct_days, only for datetime columns."
    )


class DatasetProfile(BaseModel):
    """Deterministic dataset-level profile, computed with pandas only (no LLM)."""

    row_count: int
    column_count: int
    column_names: List[str] = Field(default_factory=list)
    duplicate_row_count: int = 0
    duplicate_row_pct: float = 0.0
    numerical_columns: List[str] = Field(default_factory=list)
    categorical_columns: List[str] = Field(default_factory=list)
    datetime_columns: List[str] = Field(default_factory=list)
    boolean_columns: List[str] = Field(default_factory=list)
    constant_columns: List[str] = Field(default_factory=list)
    near_constant_columns: List[str] = Field(default_factory=list)
    excluded_sensitive_column_count: int = 0
    columns: List[ColumnProfile] = Field(default_factory=list)


class TargetCandidate(BaseModel):
    """One deterministic candidate for the prediction target, with signals
    (not a decision) - the Planner still makes the final call."""

    name: str
    inferred_kind: ColumnKind
    cardinality: int
    cardinality_ratio: float = Field(
        description="unique_count / row_count. Near 1.0 suggests an ID column, not a target."
    )
    missing_pct: float = 0.0
    likely_problem_types: List[str] = Field(default_factory=list)
    signal_notes: str = ""


class TargetAnalysis(BaseModel):
    """Deterministic target-selection signals. Never guesses in place of the
    Planner - only narrows the search space and flags what it can measure."""

    candidate_targets: List[TargetCandidate] = Field(default_factory=list)
    recommended_target: Optional[str] = Field(
        default=None,
        description="Set only when exactly one non-ID-like candidate exists unambiguously.",
    )
    recommendation_reasoning: str = ""


class DataQualityIssue(BaseModel):
    column: Optional[str] = None
    issue_type: str
    severity: str = Field(description="One of: 'low', 'medium', 'high'.")
    detail: str = ""


class DataQualityReport(BaseModel):
    """Deterministic data-quality findings, computed with pandas only."""

    missing_value_columns: Dict[str, float] = Field(
        default_factory=dict, description="column -> missing_pct, only columns with missing values."
    )
    duplicate_row_count: int = 0
    duplicate_row_pct: float = 0.0
    constant_columns: List[str] = Field(default_factory=list)
    near_constant_columns: List[str] = Field(default_factory=list)
    possible_outlier_columns: Dict[str, dict] = Field(
        default_factory=dict, description="column -> {count, pct}, via IQR rule."
    )
    invalid_dtype_columns: List[str] = Field(
        default_factory=list,
        description="Columns stored as text/object that actually hold numeric or datetime values.",
    )
    suspicious_columns: List[str] = Field(
        default_factory=list,
        description="ID-like or all-unique columns unlikely to be useful model features.",
    )
    possible_leakage_columns: List[str] = Field(
        default_factory=list,
        description="Columns whose name or near-duplicate relationship with another column "
        "suggests target leakage - heuristic, not confirmed.",
    )
    issues: List[DataQualityIssue] = Field(default_factory=list)
    overall_quality_score: float = Field(
        default=100.0, description="0-100 heuristic score; 100 = no issues detected."
    )
