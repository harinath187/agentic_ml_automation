"""Technical-retry helper, deliberately generic and separate from
tools/classification_cycle.py's model-improvement cycles.

A TECHNICAL retry (this module) is for transient infrastructure failures -
timeouts, temporary connection/I/O problems - that have nothing to do with
model quality and should just be retried a few times with backoff. A MODEL
IMPROVEMENT CYCLE (tools/classification_cycle.py) is for a model that trained
successfully but didn't meet the configured acceptance criteria. The two
counters are completely independent: a technical retry here never advances a
classification-cycle's cycle number.
"""
from __future__ import annotations

import time
from typing import Callable, TypeVar

T = TypeVar("T")


class PermanentTrainingError(ValueError):
    """Raised for failures that retrying can never fix - invalid dataset,
    missing target column, invalid target values, insufficient data, invalid
    configuration, unsupported model, invalid hyperparameters, schema errors.
    Never retried, technical or cycle-wise."""


def is_retryable_error(exc: BaseException) -> bool:
    """Default classifier: transient infrastructure failures (timeout,
    temporary connection/I/O problems) are retryable; everything else -
    including PermanentTrainingError and plain ValueError/KeyError/etc, which
    signal a real, non-transient problem - is not. Overridable per call site
    via run_with_technical_retry's `is_retryable` argument.
    """
    if isinstance(exc, PermanentTrainingError):
        return False
    return isinstance(exc, (TimeoutError, ConnectionError, OSError))


def run_with_technical_retry(
    fn: Callable[..., T],
    *args,
    max_attempts: int = 3,
    base_delay_s: float = 1.0,
    is_retryable: Callable[[BaseException], bool] = is_retryable_error,
    sleep_fn: Callable[[float], None] = time.sleep,
    **kwargs,
) -> T:
    """Calls fn(*args, **kwargs), retrying only retryable exceptions with
    exponential backoff (base_delay_s * 2**(attempt-1)) up to max_attempts
    total attempts. A non-retryable exception is re-raised immediately, on
    the first attempt, with no delay. Never loops forever - max_attempts is
    a hard cap. `sleep_fn` is injectable so tests never actually sleep.
    """
    last_exc: BaseException | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - re-classified below, not swallowed
            if not is_retryable(exc):
                raise
            last_exc = exc
            if attempt < max_attempts:
                sleep_fn(base_delay_s * (2 ** (attempt - 1)))
    assert last_exc is not None
    raise last_exc
