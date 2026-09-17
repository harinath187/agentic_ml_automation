"""Tests for agents/planner.py's model-family catalogue: the prompt must be
built dynamically from tools/model_registry.py's REGISTRY, so a model
registered there is immediately reachable as a Planner candidate with no
second, hand-maintained list to keep in sync.
"""
from agents.planner import _build_system_prompt
from agents.schemas import ExperimentPlan, ProblemType
from data_ingestion.loader import extract_schema
from tools.model_registry import get_registry_for_problem_type, resolve_candidates


def test_system_prompt_lists_every_registered_classification_model():
    prompt = _build_system_prompt()
    registered_names = {d.name for d in get_registry_for_problem_type(ProblemType.CLASSIFICATION)}

    # Every model actually registered for classification must appear in the
    # prompt text the Planner LLM sees - including models added after the
    # prompt was first written (decision_tree, svm, knn, naive_bayes,
    # neural_network), with no hardcoded list to fall out of sync.
    for name in registered_names:
        assert name in prompt


def test_system_prompt_never_lists_an_unregistered_model():
    prompt = _build_system_prompt()
    # catboost was deliberately never registered - the dynamic catalogue must
    # not invent options the registry can't actually resolve.
    assert "catboost" not in prompt.lower()


def test_planner_can_select_a_newly_registered_model_and_it_resolves(classification_df, monkeypatch):
    """Simulates an LLM choosing one of the newly-registered models: confirms
    it is in the allowed set (resolve_candidates matches it, not unmatched),
    not silently excluded by some other hardcoded allowlist downstream.
    """
    captured = {}

    def fake_call_llm_json(system_prompt, user_prompt, schema, **kwargs):
        captured["system_prompt"] = system_prompt
        return ExperimentPlan(
            problem_type=ProblemType.CLASSIFICATION,
            target_column="churn",
            candidate_model_families=["baseline", "svm", "naive_bayes"],
        )

    import agents.planner as planner_module

    monkeypatch.setattr(planner_module, "call_llm_json", fake_call_llm_json)

    schema = extract_schema(classification_df)
    plan = planner_module.build_plan("Predict which customers will churn", schema)

    assert "svm" in captured["system_prompt"]
    matched, unmatched = resolve_candidates(ProblemType.CLASSIFICATION, plan.candidate_model_families)
    assert unmatched == []
    assert {d.name for d in matched} == {"baseline", "svm", "naive_bayes"}
