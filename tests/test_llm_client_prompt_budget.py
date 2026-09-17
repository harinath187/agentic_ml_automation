"""Tests for call_llm_json's last-resort prompt-size safety net (Phase 9
413-fix): a deliberately oversized prompt must be logged loudly and
truncated to fit under max_prompt_chars, never raise, and never silently
reach Groq at full size (which is what produced the real 413 this guards
against - see agents/evaluator.py's docstring).
"""
import logging

from pydantic import BaseModel

import agents.llm_client as llm_client


class _Choice:
    def __init__(self, content):
        self.message = type("Msg", (), {"content": content})


class _Response:
    def __init__(self, content):
        self.choices = [_Choice(content)]


class _Echo(BaseModel):
    ok: bool


def _fake_client_returning(content: str):
    captured = {}

    class FakeCompletions:
        def create(self, **kwargs):
            captured["messages"] = kwargs["messages"]
            return _Response(content)

    return type("FakeClient", (), {"chat": type("Chat", (), {"completions": FakeCompletions()})()})(), captured


def test_oversized_prompt_is_truncated_and_logged(monkeypatch, caplog):
    fake_client, captured = _fake_client_returning('{"ok": true}')
    monkeypatch.setattr(llm_client, "_get_client", lambda: fake_client)

    oversized_user_prompt = "x" * 50_000  # far beyond any reasonable budget

    with caplog.at_level(logging.WARNING, logger="agents.llm_client"):
        result = llm_client.call_llm_json(
            "system prompt",
            oversized_user_prompt,
            _Echo,
            max_prompt_chars=1000,
            caller_name="tests.fake_caller",
        )

    assert result.ok is True  # never raises - degrades gracefully

    assert any("llm_prompt_exceeds_budget" in record.message for record in caplog.records)
    warning_record = next(r for r in caplog.records if "llm_prompt_exceeds_budget" in r.message)
    schema_hint = llm_client._compact_schema_hint(_Echo)
    assert warning_record.caller == "tests.fake_caller"
    assert warning_record.total_chars == len("system prompt") + len(oversized_user_prompt) + len(schema_hint)
    assert warning_record.budget_chars == 1000

    sent_user_content = captured["messages"][1]["content"]
    assert len(sent_user_content) < len(oversized_user_prompt)
    assert "[...truncated - payload exceeded token budget...]" in sent_user_content
    assert len(sent_user_content) + len("system prompt") + len(schema_hint) <= 1000 + len(
        "\n\n[...truncated - payload exceeded token budget...]"
    )


def test_within_budget_prompt_is_untouched_and_no_warning_logged(monkeypatch, caplog):
    fake_client, captured = _fake_client_returning('{"ok": true}')
    monkeypatch.setattr(llm_client, "_get_client", lambda: fake_client)

    small_user_prompt = "short user prompt"

    with caplog.at_level(logging.WARNING, logger="agents.llm_client"):
        result = llm_client.call_llm_json("system prompt", small_user_prompt, _Echo, max_prompt_chars=1000)

    assert result.ok is True
    assert not any("llm_prompt_exceeds_budget" in record.message for record in caplog.records)
    assert captured["messages"][1]["content"] == small_user_prompt


def test_caller_name_is_inferred_when_not_provided(monkeypatch, caplog):
    fake_client, _captured = _fake_client_returning('{"ok": true}')
    monkeypatch.setattr(llm_client, "_get_client", lambda: fake_client)

    with caplog.at_level(logging.WARNING, logger="agents.llm_client"):
        llm_client.call_llm_json("system prompt", "x" * 5000, _Echo, max_prompt_chars=100)

    warning_record = next(r for r in caplog.records if "llm_prompt_exceeds_budget" in r.message)
    assert warning_record.caller == __name__  # this test module, inferred via inspect
