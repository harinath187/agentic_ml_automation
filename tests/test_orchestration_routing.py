"""Tests for scope-strategy routing and the entity-filtering node in
orchestration/graph.py - pure logic, no LLM or AutoML calls involved.
"""
import pandas as pd
import pytest

from agents.schemas import (
    DataQualityReport,
    DatasetProfile,
    EvaluatorDecision,
    ExperimentPlan,
    ProblemDefinition,
    ProblemType,
    RecommendationOutput,
    ScopeStrategy,
    TargetAnalysis,
)
from orchestration.graph import (
    build_graph,
    node_cleaning,
    node_detect_problem,
    node_filter_entity,
    node_profile_data,
    node_quality_analysis,
    node_recommend,
    node_train,
    node_validate_plan,
    route_after_evaluate,
    route_after_validate_plan,
)
from tools.profiling import analyze_data_quality, analyze_target, profile_dataset


def _plan(**overrides) -> ExperimentPlan:
    defaults = dict(problem_type=ProblemType.FORECASTING, target_column="sales", time_column="date")
    defaults.update(overrides)
    return ExperimentPlan(**defaults)


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
def test_route_after_validate_plan(scope_strategy, expected_route):
    state = {"needs_clarification": False, "plan": _plan(scope_strategy=scope_strategy)}
    assert route_after_validate_plan(state) == expected_route


def test_route_after_validate_plan_clarification_overrides_scope():
    state = {
        "needs_clarification": True,
        "plan": _plan(scope_strategy=ScopeStrategy.PER_ENTITY, needs_clarification=True),
    }
    assert route_after_validate_plan(state) == "end_clarification"


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


def test_route_after_evaluate_proceed_always_recommends():
    state = {
        "plan": _plan(scope_strategy=ScopeStrategy.PER_ENTITY),
        "decision": EvaluatorDecision(decision="proceed", best_model="LightGBM", reasoning="test"),
    }
    assert route_after_evaluate(state) == "recommend"


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


# --- Data Intelligence nodes (profile_data -> quality_analysis -> detect_problem) --


def test_node_profile_data_populates_dataset_profile():
    df = pd.DataFrame({"store_id": ["a", "a", "b"], "sales": [1, 2, 3]})
    result = node_profile_data({"df": df})
    assert "dataset_profile" in result
    assert isinstance(result["dataset_profile"], DatasetProfile)
    assert result["dataset_profile"].row_count == 3


def test_node_quality_analysis_populates_target_and_quality_reports():
    df = pd.DataFrame({"store_id": ["a", "a", "b"], "sales": [1, 2, 3]})
    profile_result = node_profile_data({"df": df})
    state = {"df": df, **profile_result}

    result = node_quality_analysis(state)

    assert isinstance(result["target_analysis"], TargetAnalysis)
    assert isinstance(result["data_quality_report"], DataQualityReport)


def test_node_quality_analysis_excludes_unambiguous_target_from_name_based_leakage():
    # "Outcome" is the only binary column here, so analyze_target() resolves
    # it as the sole unambiguous recommended_target - node_quality_analysis
    # must feed that into analyze_data_quality() so the naming-hint leakage
    # check doesn't flag the target against itself (see tools/profiling.py's
    # likely_target_column param).
    df = pd.DataFrame({"Glucose": list(range(20)), "Outcome": [0, 1] * 10})
    profile_result = node_profile_data({"df": df})
    state = {"df": df, **profile_result}

    result = node_quality_analysis(state)

    assert result["target_analysis"].recommended_target == "Outcome"
    assert "Outcome" not in result["data_quality_report"].possible_leakage_columns


def _intelligence_state(df: pd.DataFrame) -> dict:
    profile = profile_dataset(df)
    target_analysis = analyze_target(df, profile)
    quality_report = analyze_data_quality(df, profile)
    return {
        "df": df,
        "dataset_profile": profile,
        "target_analysis": target_analysis,
        "data_quality_report": quality_report,
    }


def test_node_detect_problem_populates_problem_definition():
    # id+target-only frame so analyze_target has exactly one candidate,
    # keeping this a test of node wiring rather than ambiguity heuristics.
    df = pd.DataFrame({"listing_id": range(100), "price": [float(i) * 1.5 for i in range(100)]})
    state = _intelligence_state(df)
    result = node_detect_problem(state)
    assert isinstance(result["problem_definition"], ProblemDefinition)
    assert result["problem_definition"].target_column == "price"


def test_node_validate_plan_repairs_invalid_plan():
    df = pd.DataFrame({"listing_id": range(100), "price": [float(i) * 1.5 for i in range(100)]})
    state = _intelligence_state(df)
    invalid_plan = ExperimentPlan(problem_type=ProblemType.REGRESSION, target_column="does_not_exist")

    result = node_validate_plan({**state, "plan": invalid_plan})

    assert result["plan"].target_column == "price"
    assert result["needs_clarification"] is False


def test_graph_includes_phase2_nodes_in_order():
    app = build_graph()
    nodes = app.get_graph().nodes
    for expected_node in ("profile_data", "quality_analysis", "detect_problem", "planner_agent", "validate_plan"):
        assert expected_node in nodes

    edges = {(e.source, e.target) for e in app.get_graph().edges}
    assert ("ingest", "profile_data") in edges
    assert ("profile_data", "quality_analysis") in edges
    assert ("quality_analysis", "detect_problem") in edges
    assert ("detect_problem", "planner_agent") in edges


def test_graph_vectorizes_text_after_split_before_train():
    app = build_graph()
    edges = {(e.source, e.target) for e in app.get_graph().edges}
    assert "vectorize_text" in app.get_graph().nodes
    assert ("split", "vectorize_text") in edges
    assert ("vectorize_text", "train") in edges
    assert ("planner_agent", "validate_plan") in edges


# --- node_cleaning respects plan.preprocessing_requirements -----------------


def test_node_cleaning_caps_outliers_when_plan_requests_it_even_without_retry():
    """Regression coverage: previously aggressive_outlier_handling (which
    drives outlier capping in tools/cleaning.py) was only ever True on a
    retry, ignoring plan.preprocessing_requirements entirely on a first pass."""
    df = pd.DataFrame(
        {
            "value": [10, 11, 9, 10, 12, 11, 9, 10, 500],  # 500 is a clear IQR outlier
            "target": list(range(9)),
        }
    )
    plan = _plan(
        problem_type=ProblemType.REGRESSION,
        target_column="target",
        time_column=None,
        preprocessing_requirements=["cap_outliers"],
    )
    state = {"df": df, "plan": plan, "retry_count": 0}

    result = node_cleaning(state)

    assert "value" in result["cleaning_log"]["outliers_capped"]
    assert result["cleaned_df"]["value"].max() < 500


def test_node_cleaning_does_not_cap_outliers_without_request_or_retry():
    df = pd.DataFrame(
        {
            "value": [10, 11, 9, 10, 12, 11, 9, 10, 500],
            "target": list(range(9)),
        }
    )
    plan = _plan(
        problem_type=ProblemType.REGRESSION, target_column="target", time_column=None, preprocessing_requirements=[]
    )
    state = {"df": df, "plan": plan, "retry_count": 0}

    result = node_cleaning(state)

    assert result["cleaning_log"]["outliers_capped"] == {}


# --- node_train (Phase 3 model registry wiring) -----------------------------


def _classification_train_test_dfs():
    df = pd.DataFrame(
        {
            "f1": [0.1, 0.4, 0.9, 0.2, 0.8, 0.3, 0.95, 0.05, 0.7, 0.15] * 4,
            "target": [0, 0, 1, 0, 1, 0, 1, 0, 1, 0] * 4,
        }
    )
    return df.iloc[:30].reset_index(drop=True), df.iloc[30:].reset_index(drop=True)


def test_node_train_uses_only_plan_requested_candidates():
    train_df, test_df = _classification_train_test_dfs()
    plan = _plan(
        problem_type=ProblemType.CLASSIFICATION,
        target_column="target",
        time_column=None,
        candidate_model_families=["baseline", "logistic_regression"],
    )
    state = {"plan": plan, "train_df": train_df, "test_df": test_df}

    result = node_train(state)

    assert set(result["metrics"]["models"].keys()) == {"baseline", "logistic_regression"}


def test_node_train_falls_back_to_default_when_nothing_matches(monkeypatch):
    # Classification's default fallback is just "baseline" (no AutoGluon), so
    # this test only needs to prove the fallback kicks in - baseline trains
    # fast for real, no faking needed.
    train_df, test_df = _classification_train_test_dfs()
    plan = _plan(
        problem_type=ProblemType.CLASSIFICATION,
        target_column="target",
        time_column=None,
        candidate_model_families=["TotallyMadeUpModelName"],
    )
    state = {"plan": plan, "train_df": train_df, "test_df": test_df}

    result = node_train(state)

    # Falls back to model_registry.default_candidates() rather than training nothing.
    assert result["metrics"]["models"]


# --- node_recommend (Phase 5 recommendation agent wiring) -------------------


def test_node_recommend_uses_model_comparison_and_overrides_llm(monkeypatch):
    import agents.recommender as recommender_module

    def fake_call_llm_json(system_prompt, user_prompt, schema, **kwargs):
        return RecommendationOutput(recommended_model="a_model_the_llm_made_up", reason="test")

    monkeypatch.setattr(recommender_module, "call_llm_json", fake_call_llm_json)

    plan = _plan(problem_type=ProblemType.REGRESSION, target_column="y", time_column=None)
    metrics = {
        "eval_metric": "rmse",
        "models": {"good": {"score_test": -1.0}, "bad": {"score_test": -5.0}},
        "model_comparison": {
            "problem_type": "regression",
            "primary_metric": "rmse",
            "higher_is_better": False,
            "results": [
                {"model_name": "good", "problem_type": "regression", "status": "success", "metrics": {"rmse": 1.0}},
                {"model_name": "bad", "problem_type": "regression", "status": "success", "metrics": {"rmse": 5.0}},
            ],
            "ranked_model_names": ["good", "bad"],
            "winner": "good",
            "winner_reasoning": "good wins",
        },
    }
    state = {
        "plan": plan,
        "metrics": metrics,
        "decision": EvaluatorDecision(decision="proceed", best_model="good", reasoning="test"),
    }

    result = node_recommend(state)

    assert result["recommendation"].recommended_model == "good"


def test_graph_includes_recommend_node_between_evaluate_and_report():
    app = build_graph()
    nodes = app.get_graph().nodes
    assert "recommend" in nodes

    edges = {(e.source, e.target) for e in app.get_graph().edges}
    assert ("recommend", "report") in edges
