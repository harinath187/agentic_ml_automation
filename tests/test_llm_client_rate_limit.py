"""Verifies call_llm_json backs off and retries on Groq 429 rate-limit errors
instead of failing the whole pipeline run immediately (relevant for free-tier
accounts, which have low requests/tokens-per-minute limits).
"""
import httpx
import pytest
from groq import RateLimitError
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


def _make_rate_limit_error(retry_after: str | None = None) -> RateLimitError:
    headers = {"retry-after": retry_after} if retry_after else {}
    response = httpx.Response(status_code=429, headers=headers, request=httpx.Request("POST", "https://api.groq.com"))
    return RateLimitError("rate limited", response=response, body=None)


def test_retries_after_rate_limit_then_succeeds(monkeypatch):
    sleep_calls = []
    monkeypatch.setattr(llm_client.time, "sleep", lambda s: sleep_calls.append(s))

    attempts = {"count": 0}

    class FakeCompletions:
        def create(self, **kwargs):
            attempts["count"] += 1
            if attempts["count"] < 3:
                raise _make_rate_limit_error(retry_after="0.5")
            return _Response('{"ok": true}')

    fake_client = type("FakeClient", (), {"chat": type("Chat", (), {"completions": FakeCompletions()})()})()

    monkeypatch.setattr(llm_client, "_get_client", lambda: fake_client)

    result = llm_client.call_llm_json("system", "user", _Echo)

    assert result.ok is True
    assert attempts["count"] == 3
    assert sleep_calls == [0.5, 0.5]


def test_raises_llm_call_error_after_exhausting_rate_limit_retries(monkeypatch):
    monkeypatch.setattr(llm_client.time, "sleep", lambda s: None)

    class FakeCompletions:
        def create(self, **kwargs):
            raise _make_rate_limit_error()

    fake_client = type("FakeClient", (), {"chat": type("Chat", (), {"completions": FakeCompletions()})()})()
    monkeypatch.setattr(llm_client, "_get_client", lambda: fake_client)

    with pytest.raises(llm_client.LLMCallError, match="rate limit"):
        llm_client.call_llm_json("system", "user", _Echo)
