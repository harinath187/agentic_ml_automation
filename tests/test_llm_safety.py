"""Verifies the core privacy boundary: no raw data rows ever reach an LLM prompt.

Monkeypatches the LLM call sites in each agent to capture exactly what text
would have been sent, then asserts specific raw cell values never appear in it.
"""
from data_ingestion.loader import extract_schema
from agents.schemas import EvaluatorDecision, ExperimentPlan, ProblemType
from tools.problem_detection import detect_problem
from tools.profiling import analyze_data_quality, analyze_target, profile_dataset


def test_planner_prompt_never_contains_raw_rows(classification_df, monkeypatch):
    captured = {}

    def fake_call_llm_json(system_prompt, user_prompt, schema, **kwargs):
        captured["user_prompt"] = user_prompt
        return ExperimentPlan(
            problem_type=ProblemType.CLASSIFICATION,
            target_column="churn",
            reasoning="test",
            pipeline_steps=["clean", "train"],
            candidate_model_families=["LightGBM"],
        )

    import agents.planner as planner_module

    monkeypatch.setattr(planner_module, "call_llm_json", fake_call_llm_json)

    schema = extract_schema(classification_df)
    planner_module.build_plan("Predict churn", schema)

    prompt = captured["user_prompt"]
    # Column names/stats are expected to appear (schema_summary is built from
    # them); the actual privacy boundary is that no raw cell value ever does.
    assert "Customer 0" not in prompt
    for raw_value in classification_df["monthly_charge"].dropna().astype(str).head(20):
        assert raw_value not in prompt


def test_planner_prompt_with_data_intelligence_never_contains_raw_rows(classification_df, monkeypatch):
    captured = {}

    def fake_call_llm_json(system_prompt, user_prompt, schema, **kwargs):
        captured["user_prompt"] = user_prompt
        return ExperimentPlan(
            problem_type=ProblemType.CLASSIFICATION,
            target_column="churn",
            reasoning="test",
            pipeline_steps=["clean", "train"],
            candidate_model_families=["LightGBM"],
        )

    import agents.planner as planner_module

    monkeypatch.setattr(planner_module, "call_llm_json", fake_call_llm_json)

    schema = extract_schema(classification_df)
    dataset_profile = profile_dataset(classification_df)
    target_analysis = analyze_target(classification_df, dataset_profile)
    quality_report = analyze_data_quality(classification_df, dataset_profile)
    problem_definition = detect_problem(classification_df, dataset_profile, target_analysis, quality_report)

    planner_module.build_plan(
        "Predict churn",
        schema,
        dataset_profile=dataset_profile,
        quality_report=quality_report,
        target_analysis=target_analysis,
        problem_definition=problem_definition,
    )

    prompt = captured["user_prompt"]
    # Column names/stats are expected to appear; only raw cell values must not.
    assert "Customer 0" not in prompt
    for raw_value in classification_df["monthly_charge"].dropna().astype(str).head(20):
        assert raw_value not in prompt
    # The deterministic Data Intelligence report should actually be present.
    assert "Dataset profile" in prompt
    assert "Target analysis" in prompt
    assert "Data quality report" in prompt
    assert "Problem definition" in prompt


def test_evaluator_prompt_only_contains_metrics_and_plan(monkeypatch):
    captured = {}

    def fake_call_llm_json(system_prompt, user_prompt, schema, **kwargs):
        captured["user_prompt"] = user_prompt
        return EvaluatorDecision(decision="proceed", best_model="LightGBM", reasoning="test")

    import agents.evaluator as evaluator_module

    monkeypatch.setattr(evaluator_module, "call_llm_json", fake_call_llm_json)

    plan = ExperimentPlan(problem_type=ProblemType.CLASSIFICATION, target_column="churn")
    metrics = {"eval_metric": "roc_auc", "models": {"LightGBM": {"score_test": 0.9, "score_val": 0.88, "fit_time_s": 1.2}}}
    evaluator_module.evaluate_results(metrics, plan, retry_count=0, max_retries=2)

    prompt = captured["user_prompt"]
    assert "LightGBM" in prompt
    assert "0.9" in prompt
