"""Tests for tools/logging_config.py (Phase 9): structured (JSON-line)
logging used across the API layer and orchestration/graph.py.
"""
from __future__ import annotations

import json
import logging

from tools.logging_config import JSONFormatter, configure_logging


def _make_record(**extra) -> logging.LogRecord:
    record = logging.LogRecord(
        name="test.logger", level=logging.INFO, pathname=__file__, lineno=1,
        msg="something happened", args=(), exc_info=None,
    )
    for key, value in extra.items():
        setattr(record, key, value)
    return record


def test_formatter_produces_valid_json_with_core_fields():
    formatter = JSONFormatter()
    line = formatter.format(_make_record())
    payload = json.loads(line)  # must not raise
    assert payload["level"] == "INFO"
    assert payload["logger"] == "test.logger"
    assert payload["message"] == "something happened"
    assert "timestamp" in payload


def test_formatter_includes_extra_context_fields():
    formatter = JSONFormatter()
    line = formatter.format(_make_record(run_id="abc123", dataset_id="d1"))
    payload = json.loads(line)
    assert payload["run_id"] == "abc123"
    assert payload["dataset_id"] == "d1"


def test_formatter_never_raises_on_unserializable_extra():
    formatter = JSONFormatter()
    record = _make_record(weird=object())
    line = formatter.format(record)
    payload = json.loads(line)  # falls back to str(...) rather than raising
    assert "weird" in payload


def test_formatter_includes_exception_traceback():
    formatter = JSONFormatter()
    try:
        raise ValueError("boom")
    except ValueError:
        import sys

        record = _make_record()
        record.exc_info = sys.exc_info()
    line = formatter.format(record)
    payload = json.loads(line)
    assert "ValueError: boom" in payload["exception"]


def test_configure_logging_is_idempotent(monkeypatch):
    """Isolated from whatever state other modules (api/main.py imports it at
    module load time) left the root logger in - reset to "never configured"
    first, then assert 3 calls add exactly one handler, not three."""
    import tools.logging_config as logging_config_module

    root = logging.getLogger()
    added_by_other_tests = [h for h in root.handlers if isinstance(h.formatter, JSONFormatter)]
    for handler in added_by_other_tests:
        root.removeHandler(handler)
    monkeypatch.setattr(logging_config_module, "_CONFIGURED", False)

    before = len(root.handlers)
    configure_logging()
    configure_logging()
    configure_logging()
    after = len(root.handlers)
    assert after == before + 1  # only ever adds one handler, regardless of call count
