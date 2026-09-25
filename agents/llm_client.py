"""Thin wrapper around the Groq API with strict Pydantic-validated JSON output.

Safety rule: callers must only ever pass schema/metric/text-level content in
`user_prompt`. This module does not enforce that (it cannot inspect intent),
but every call site in this project is expected to pass only sanitized
summaries - never raw DataFrame rows. See data_ingestion/loader.py.
"""
from __future__ import annotations

import inspect
import json
import os
import time
from typing import Optional, Type, TypeVar

from pydantic import BaseModel, ValidationError

from tools.logging_config import get_logger

T = TypeVar("T", bound=BaseModel)

logger = get_logger(__name__)

DEFAULT_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")

# Free-tier Groq accounts have low requests/tokens-per-minute limits, so a
# 429 here is expected occasionally, not exceptional - back off and retry
# rather than failing the whole pipeline run.
RATE_LIMIT_MAX_ATTEMPTS = 5
RATE_LIMIT_BASE_DELAY_S = 2.0
RATE_LIMIT_MAX_DELAY_S = 30.0

# Last-resort prompt-size guard (Phase 9 413-fix). Originally 24000 chars on
# a ~4 chars/token assumption, but a live 413 measured 8185 actual tokens
# (prompt + completion) against Groq free-tier's 8000 TPM cap for
# openai/gpt-oss-120b - the compact JSON schema hint (punctuation-heavy)
# tokenizes denser than ~4 chars/token, so that budget wasn't leaving real
# margin. Tightened to ~3 chars/token-equivalent with headroom. This is NOT
# the primary fix - call sites (agents/evaluator.py, agents/recommender.py,
# agents/reporter.py) should already be sending a trimmed view
# (tools.evaluation.to_llm_summary() or equivalent) long before a prompt
# could reach this size. This guard exists only so an oversized prompt
# becomes a loud, logged, debuggable event during development instead of a
# silent Groq 413 at request time.
DEFAULT_MAX_PROMPT_CHARS = 16000
_TRUNCATION_MARKER = "\n\n[...truncated - payload exceeded token budget...]"

# Upper bound on completion tokens for every structured-output call. Groq's
# TPM rate limit counts prompt + completion tokens together, so leaving this
# unset lets the model's default completion allowance silently eat into the
# same budget DEFAULT_MAX_PROMPT_CHARS is trying to protect. Every schema in
# agents/schemas.py used with call_llm_json is a handful of short fields, so
# 1024 is still generous headroom, not a tight fit - lowered from 2048 as
# part of the same TPM-cap fix above.
DEFAULT_MAX_COMPLETION_TOKENS = 1024


class LLMCallError(RuntimeError):
    pass


def _compact_schema_hint(schema: Type[BaseModel]) -> str:
    """Builds the 'respond with this schema' hint from schema.model_json_schema(),
    stripping `description`/`title` keys first.

    Pydantic's Field(description=...) text is there for humans reading
    agents/schemas.py, not for the model - repeating every field's prose
    description a second time inside the JSON schema roughly doubles its
    size for no benefit, and was part of what pushed planner.py's prompt
    over Groq's TPM cap (see the 413 this guards against).
    """

    def _strip(node):
        if isinstance(node, dict):
            return {k: _strip(v) for k, v in node.items() if k not in ("description", "title")}
        if isinstance(node, list):
            return [_strip(v) for v in node]
        return node

    compact = _strip(schema.model_json_schema())
    return (
        "Respond with ONLY a single valid JSON object matching this schema "
        f"(no prose, no markdown fences):\n{json.dumps(compact, separators=(',', ':'))}"
    )


def _get_client():
    from groq import Groq

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise LLMCallError(
            "GROQ_API_KEY is not set. Copy .env.example to .env and fill it in."
        )
    return Groq(api_key=api_key)


def _create_completion_with_backoff(client, **kwargs):
    """Call the Groq chat completion API, retrying on 429 with backoff.

    Honors the `Retry-After` header when Groq sends one; otherwise falls
    back to exponential backoff capped at RATE_LIMIT_MAX_DELAY_S.
    """
    from groq import RateLimitError

    last_error: Exception | None = None
    for attempt in range(1, RATE_LIMIT_MAX_ATTEMPTS + 1):
        try:
            return client.chat.completions.create(**kwargs)
        except RateLimitError as exc:
            last_error = exc
            if attempt == RATE_LIMIT_MAX_ATTEMPTS:
                break
            retry_after = _parse_retry_after(exc)
            delay = retry_after if retry_after is not None else min(
                RATE_LIMIT_BASE_DELAY_S * (2 ** (attempt - 1)), RATE_LIMIT_MAX_DELAY_S
            )
            time.sleep(delay)

    raise LLMCallError(
        f"Groq rate limit exceeded after {RATE_LIMIT_MAX_ATTEMPTS} attempts: {last_error}"
    )


def _parse_retry_after(exc) -> float | None:
    response = getattr(exc, "response", None)
    header = response.headers.get("retry-after") if response is not None else None
    if header is None:
        return None
    try:
        return float(header)
    except ValueError:
        return None


def _infer_caller_name() -> str:
    """Best-effort module name of call_llm_json's caller, for the prompt-size
    warning log below - never required (pass `caller_name` explicitly to
    skip this), and never allowed to fail the call if introspection can't
    find a frame for some reason (e.g. an unusual execution environment).
    """
    try:
        frame = inspect.stack()[2].frame
        return frame.f_globals.get("__name__", "unknown")
    except Exception:  # noqa: BLE001 - purely diagnostic, never fatal
        return "unknown"


def call_llm_json(
    system_prompt: str,
    user_prompt: str,
    schema: Type[T],
    max_retries: int = 3,
    model: str = DEFAULT_MODEL,
    max_prompt_chars: int = DEFAULT_MAX_PROMPT_CHARS,
    max_tokens: int = DEFAULT_MAX_COMPLETION_TOKENS,
    caller_name: Optional[str] = None,
) -> T:
    """Call the LLM and parse+validate its response against `schema`.

    On validation failure, re-prompts with the validation error appended,
    up to `max_retries` attempts total, then raises LLMCallError. Rate-limit
    (429) responses are retried with backoff separately and don't consume
    this validation-retry budget.

    `max_prompt_chars` is a last-resort safety net (see DEFAULT_MAX_PROMPT_CHARS
    above): if `system_prompt` + `user_prompt` + the schema hint exceeds it,
    this logs a warning naming `caller_name` (inferred from the call stack
    when omitted) and truncates `user_prompt` to fit before ever calling
    Groq. Callers should trim their own payload (tools.evaluation.to_llm_summary()
    or equivalent) well before this triggers - it's a safety net, not the fix.

    `max_tokens` bounds the completion Groq is asked to generate - Groq's TPM
    rate limit counts prompt + completion tokens together, so an unbounded
    completion allowance can push an otherwise-fine prompt over the limit.
    """
    caller_name = caller_name or _infer_caller_name()
    schema_hint = _compact_schema_hint(schema)
    total_chars = len(system_prompt) + len(user_prompt) + len(schema_hint)
    if total_chars > max_prompt_chars:
        logger.warning(
            "llm_prompt_exceeds_budget",
            extra={
                "caller": caller_name,
                "total_chars": total_chars,
                "budget_chars": max_prompt_chars,
            },
        )
        keep_chars = max(
            0, max_prompt_chars - len(system_prompt) - len(schema_hint) - len(_TRUNCATION_MARKER)
        )
        user_prompt = user_prompt[:keep_chars] + _TRUNCATION_MARKER

    client = _get_client()
    messages = [
        {"role": "system", "content": f"{system_prompt}\n\n{schema_hint}"},
        {"role": "user", "content": user_prompt},
    ]

    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        response = _create_completion_with_backoff(
            client,
            model=model,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.1,
            max_tokens=max_tokens,
        )
        raw = response.choices[0].message.content
        try:
            data = json.loads(raw)
            return schema.model_validate(data)
        except (json.JSONDecodeError, ValidationError) as exc:
            last_error = exc
            messages.append({"role": "assistant", "content": raw})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Your previous response was invalid JSON or failed schema "
                        f"validation with this error:\n{exc}\n"
                        "Return a corrected JSON object only."
                    ),
                }
            )

    raise LLMCallError(
        f"LLM failed to produce valid '{schema.__name__}' JSON after {max_retries} attempts: {last_error}"
    )
