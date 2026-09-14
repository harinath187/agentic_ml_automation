"""Tests for scope-strategy routing and the entity-filtering node in
orchestration/graph.py - pure logic, no LLM or AutoML calls involved.
"""
import pandas as pd
import pytest

from agents.schemas import (
    DataQualityReport,
    DatasetProfile,
    EvaluatorDecision,
    PipelinePlan,
    ProblemType,
    ScopeStrategy,
    TargetAnalysis,
)
from orchestration.graph import (
    build_graph,
    node_filter_entity,
    node_profile_data,
    node_quality_analysis,
    route_after_evaluate,
    route_after_plan,
)


def _plan(**overrides) -> PipelinePlan:
    defaults = dict(problem_type=ProblemType.FORECASTING, target_column="sales", time_column="date")
    defaults.update(overrides)
    return PipelinePlan(**defaults)


@pytest.mark.parametrize(
    "scope_strategy,expected_route",
    [
        (None, "eda"),
        (ScopeStrategy.POOLED, "eda"),
        (ScopeStrategy.SINGLE_ENTITY, "filter_entity"),
        (ScopeStrategy.PER_ENTITY, "per_entity"),
        (ScopeStrategy.HIERARCHICAL, "hierarchical"),
    ],
)
def test_route_after_plan(scope_strategy, expected_route):
    state = {"needs_clarification": False, "plan": _plan(scope_strategy=scope_strategy)}
    assert route_after_plan(state) == expected_route


def test_route_after_plan_clarification_overrides_scope():
    state = {
        "needs_clarification": True,
        "plan": _plan(scope_strategy=ScopeStrategy.PER_ENTITY, needs_clarification=True),
    }
    assert route_after_plan(state) == "end_clarification"


@pytest.mark.parametrize(
    "scope_strategy,expected_route",
    [
        (None, "retry_standard"),
        (ScopeStrategy.POOLED, "retry_standard"),
        (ScopeStrategy.SINGLE_ENTITY, "retry_standard"),
        (ScopeStrategy.PER_ENTITY, "retry_per_entity"),
        (ScopeStrategy.HIERARCHICAL, "retry_hierarchical"),
    ],
)
def test_route_after_evaluate_retry(scope_strategy, expected_route):
    state = {
        "plan": _plan(scope_strategy=scope_strategy),
        "decision": EvaluatorDecision(decision="retry", reasoning="test"),
    }
    assert route_after_evaluate(state) == expected_route


def test_route_after_evaluate_proceed_always_reports():
    state = {
        "plan": _plan(scope_strategy=ScopeStrategy.PER_ENTITY),
        "decision": EvaluatorDecision(decision="proceed", best_model="LightGBM", reasoning="test"),
    }
    assert route_after_evaluate(state) == "report"


def test_node_filter_entity_subsets_and_drops_entity_column():
    df = pd.DataFrame(
        {
            "store_id": ["a", "a", "b", "b"],
            "date": pd.date_range("2023-01-01", periods=4),
            "sales": [1, 2, 3, 4],
        }
    )
    plan = _plan(entity_column="store_id", entity_filter_value="a", scope_strategy=ScopeStrategy.SINGLE_ENTITY)
    state = {"df": df, "plan": plan}

    result = node_filter_entity(state)

    assert "store_id" not in result["df"].columns
    assert len(result["df"]) == 2
    assert result["df"]["sales"].tolist() == [1, 2]


def test_node_filter_entity_raises_without_filter_value():
    df = pd.DataFrame({"store_id": ["a"], "sales": [1]})
    plan = _plan(entity_column="store_id", entity_filter_value=None, scope_strategy=ScopeStrategy.SINGLE_ENTITY)
    with pytest.raises(ValueError, match="entity_filter_value"):
        node_filter_entity({"df": df, "plan": plan})


def test_node_filter_entity_raises_when_value_not_found():
    df = pd.DataFrame({"store_id": ["a"], "sales": [1]})
    plan = _plan(entity_column="store_id", entity_filter_value="zzz", scope_strategy=ScopeStrategy.SINGLE_ENTITY)
    with pytest.raises(ValueError, match="No rows found"):
        node_filter_entity({"df": df, "plan": plan})


# --- Data Intelligence nodes (profile_data -> quality_analysis) -----------


def test_node_profile_data_populates_dataset_profile():
    df = pd.DataFrame({"store_id": ["a", "a", "b"], "sales": [1, 2, 3]})
    result = node_profile_data({"df": df, "sensitive_columns": []})
    assert "dataset_profile" in result
    assert isinstance(result["dataset_profile"], DatasetProfile)
    assert result["dataset_profile"].row_count == 3


def test_node_quality_analysis_populates_target_and_quality_reports():
    df = pd.DataFrame({"store_id": ["a", "a", "b"], "sales": [1, 2, 3]})
    profile_result = node_profile_data({"df": df, "sensitive_columns": []})
    state = {"df": df, "sensitive_columns": [], **profile_result}

    result = node_quality_analysis(state)

    assert isinstance(result["target_analysis"], TargetAnalysis)
    assert isinstance(result["data_quality_report"], DataQualityReport)


def test_graph_includes_data_intelligence_nodes_before_planner():
    app = build_graph()
    nodes = app.get_graph().nodes
    assert "profile_data" in nodes
    assert "quality_analysis" in nodes

    edges = {(e.source, e.target) for e in app.get_graph().edges}
    assert ("ingest", "profile_data") in edges
    assert ("profile_data", "quality_analysis") in edges
    assert ("quality_analysis", "planner") in edges
