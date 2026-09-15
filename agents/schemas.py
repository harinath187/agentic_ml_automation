"""Pydantic schemas shared by the Planner and Evaluator agents, plus the
deterministic Data Intelligence schemas (DatasetProfile, TargetAnalysis,
DataQualityReport, ProblemDefinition) produced by tools/profiling.py and
tools/problem_detection.py before the Planner runs.

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


class ValidationStrategyType(str, Enum):
    TRAIN_TEST_SPLIT = "train_test_split"
    K_FOLD = "k_fold"
    STRATIFIED_K_FOLD = "stratified_k_fold"
    TIME_SERIES_SPLIT = "time_series_split"


class ValidationStrategy(BaseModel):
    """How candidate approaches should be validated. Descriptive only in this
    phase - tools/splitting.py still decides the actual split deterministically
    by problem_type; this is surfaced for transparency/review, not yet wired
    into training behavior."""

    strategy_type: ValidationStrategyType
    folds: Optional[int] = Field(default=None, description="Number of folds, for k_fold/stratified_k_fold.")
    test_size: Optional[float] = Field(default=None, description="Held-out fraction, for train_test_split (0-1).")
    notes: str = ""


class ExperimentPlan(BaseModel):
    """Structured output of the Planner Agent: an EXPERIMENT PLAN listing which
    candidate approaches to test - never the final model choice. The Evaluator
    Agent still picks the winner after every candidate has been trained and
    scored (see agents/evaluator.py)."""

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
    feature_columns: List[str] = Field(
        default_factory=list,
        description="Columns to use as model inputs. Empty is repaired by validate_plan to "
        "'all remaining non-target, non-suspicious columns'.",
    )
    unmatched_business_requirements: List[str] = Field(
        default_factory=list,
        description="Phrases from business_description that reference a metric/feature/attribute "
        "with no matching column in the dataset schema, flagged for human review only - "
        "never a reason by itself to set needs_clarification.",
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
    candidate_model_families: List[str] = Field(
        default_factory=list,
        description="Model families to TEST, e.g. ['LightGBM', 'RandomForest', 'CatBoost']. "
        "A shortlist of approaches, NOT the final model decision.",
    )
    validation_strategy: Optional[ValidationStrategy] = Field(
        default=None, description="How candidates should be validated."
    )
    evaluation_metrics: List[str] = Field(
        default_factory=list,
        description="Metrics to score candidates on, e.g. ['roc_auc', 'f1'] or ['rmse', 'mae'].",
    )
    forecast_horizon: Optional[int] = Field(
        default=None,
        description="Number of future periods to forecast. Required when problem_type is forecasting.",
    )
    preprocessing_requirements: List[str] = Field(
        default_factory=list,
        description="Preprocessing steps needed before training, e.g. ['impute_missing', 'encode_categoricals'].",
    )
    reasoning: str = Field(
        default="", description="Short explanation of why this plan was chosen."
    )
    pipeline_steps: List[str] = Field(
        default_factory=list,
        description="Ordered list of steps to run, e.g. ['clean', 'engineer_features', 'split', 'train'].",
    )
    validation_notes: List[str] = Field(
        default_factory=list,
        description="Notes added by the deterministic validate_plan repair step - for transparency/debugging, "
        "not set by the LLM.",
    )


class RetryAdjustment(BaseModel):
    change_imputation_strategy: bool = False
    change_feature_engineering: bool = False
    additional_candidate_models: List[str] = Field(default_factory=list)
    notes: str = ""


class ReportContent(BaseModel):
    """Structured output of the Reporter Agent - narrative-only sections.

    Model-selection reasoning, business interpretation, and limitations come
    from the already-validated RecommendationOutput (agents/recommender.py)
    instead of being re-derived here, so there is one validated source of
    truth for "why this model" rather than two independent LLM outputs that
    could disagree.
    """

    executive_summary: str = Field(
        default="", description="A short, plain-language overview of the business problem, approach, and outcome."
    )
    approach_narrative: str = Field(
        default="", description="Plain-language description of the preprocessing/experiment approach taken."
    )


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


class TimeSeriesSignals(BaseModel):
    """Deterministic time-series signals, computed with pandas only and only
    when a datetime column is present. Narrows the search space for the
    Planner - never a decision."""

    datetime_column: Optional[str] = None
    item_column: Optional[str] = Field(
        default=None, description="Entity/group column (store, customer, ...), if one was detected."
    )
    frequency: Optional[str] = Field(
        default=None, description="Inferred pandas offset alias, e.g. 'D', 'W', 'M', or null if irregular/unknown."
    )
    time_range: Optional[dict] = Field(default=None, description="{'min':..., 'max':..., 'distinct_days':...}")
    missing_periods_count: int = 0
    missing_periods_pct: float = 0.0
    seasonality_detected: bool = False
    seasonality_notes: str = ""
    trend_detected: bool = False
    trend_notes: str = ""


class RecommendationOutput(BaseModel):
    """Structured output of the Recommendation Agent (agents/recommender.py).

    recommended_model is ALWAYS forced to match tools/evaluation.py's
    deterministic ModelComparison.winner - the LLM's proposed value is
    discarded if it disagrees (see agents/recommender.py's
    validate_recommendation). The LLM may raise a concern via
    flagged_for_review/flag_reason without ever substituting a different
    model as the recommendation. Every value in cited_metrics is checked
    against the real evaluation results and dropped if it doesn't match -
    the LLM cannot invent metrics. cited_top_features is checked the same
    way against tools/explainability.py's ExplainabilityResult - the LLM
    cannot contradict the actual calculated feature/permutation/SHAP
    importance either (Phase 7).
    """

    recommended_model: Optional[str] = Field(
        default=None, description="Must equal the deterministic evaluation engine's winning model."
    )
    reason: str = Field(default="", description="Concise reason the winning model is recommended.")
    performance_summary: str = Field(
        default="", description="Plain-language summary of the winner's validation performance."
    )
    comparison_to_alternatives: str = Field(
        default="", description="Narrative comparison of the winner against the other evaluated candidates."
    )
    explanation_narrative: str = Field(
        default="",
        description="Plain-language MODEL EXPLANATION - what drives the model's predictions (e.g. which "
        "features matter most, or trend/seasonality for forecasting). Distinct from business_interpretation: "
        "this is about HOW the model works, not what it means for the business.",
    )
    cited_top_features: List[str] = Field(
        default_factory=list,
        description="Feature names referenced in explanation_narrative as most important. Every entry MUST "
        "be among the top-ranked features in the evaluation engine's explainability results for the "
        "recommended model - never invented or reordered.",
    )
    business_interpretation: str = Field(
        default="", description="What the metrics mean for the business problem."
    )
    limitations: str = Field(default="", description="Known limitations, risks, or caveats.")
    confidence_statement: str = Field(
        default="", description="A qualified statement of confidence in the recommendation."
    )
    cited_metrics: Dict[str, float] = Field(
        default_factory=dict,
        description="metric_name -> value for the recommended model. Every value MUST come directly "
        "from the evaluation results provided - never invented or estimated.",
    )
    alternative_models_mentioned: List[str] = Field(
        default_factory=list,
        description="Model names referenced in comparison_to_alternatives. Must be models that were "
        "actually evaluated.",
    )
    flagged_for_review: bool = Field(
        default=False,
        description="Set True ONLY when a data/configuration inconsistency in the evaluation results "
        "is suspected. Never used to substitute a different model as the recommendation.",
    )
    flag_reason: Optional[str] = Field(
        default=None, description="Required when flagged_for_review is True; explains the suspected inconsistency."
    )
    validation_notes: List[str] = Field(
        default_factory=list,
        description="Notes added by the deterministic validate_recommendation repair step - for "
        "transparency/debugging, not set by the LLM.",
    )


class ProblemDefinition(BaseModel):
    """Deterministic problem-type detection, computed before the Planner runs
    (tools/problem_detection.py) - no LLM involved. Narrows problem_type/
    target_column candidates; the Planner still makes the final call using
    business context, especially when confidence is 'low'."""

    detected_problem_type: Optional[ProblemType] = None
    target_column: Optional[str] = None
    confidence: str = Field(default="low", description="One of: 'high', 'medium', 'low'.")
    reasoning: str = ""
    time_series_signals: Optional[TimeSeriesSignals] = None
