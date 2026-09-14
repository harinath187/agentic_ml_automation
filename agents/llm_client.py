"""Thin wrapper around the Groq API with strict Pydantic-validated JSON output.

Safety rule: callers must only ever pass schema/metric/text-level content in
`user_prompt`. This module does not enforce that (it cannot inspect intent),
but every call site in this project is expected to pass only sanitized
summaries - never raw DataFrame rows. See data_ingestion/loader.py.
"""
from __future__ import annotations

import json
import os
import time
from typing import Type, TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)

DEFAULT_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")

# Free-tier Groq accounts have low requests/tokens-per-minute limits, so a
# 429 here is expected occasionally, not exceptional - back off and retry
# rather than failing the whole pipeline run.
RATE_LIMIT_MAX_ATTEMPTS = 5
RATE_LIMIT_BASE_DELAY_S = 2.0
RATE_LIMIT_MAX_DELAY_S = 30.0


class LLMCallError(RuntimeError):
    pass


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


def call_llm_json(
    system_prompt: str,
    user_prompt: str,
    schema: Type[T],
    max_retries: int = 3,
    model: str = DEFAULT_MODEL,
) -> T:
    """Call the LLM and parse+validate its response against `schema`.

    On validation failure, re-prompts with the validation error appended,
    up to `max_retries` attempts total, then raises LLMCallError. Rate-limit
    (429) responses are retried with backoff separately and don't consume
    this validation-retry budget.
    """
    client = _get_client()
    schema_hint = (
        "Respond with ONLY a single valid JSON object matching this schema "
        f"(no prose, no markdown fences):\n{json.dumps(schema.model_json_schema())}"
    )
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
