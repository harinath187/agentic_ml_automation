"""Phase 9: Structured logging.

One JSON object per log line (timestamp, level, logger name, message, plus
whatever `extra={...}` a call site attaches - typically `run_id`/`dataset_id`
for correlation across a run's queued -> running -> completed/failed/
cancelled lifecycle). Deliberately just the stdlib `logging` module with a
custom formatter - no external log-shipping dependency, consistent with
"simple, suitable for the current application size" rather than a full
observability stack.

configure_logging() is idempotent and safe to call from both the API process
and the CLI (main.py) - a second call is a no-op rather than duplicating
handlers/log lines.
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone

_CONFIGURED = False

# Attributes every standard LogRecord already has - anything else passed via
# `extra={...}` is assumed to be call-site context worth surfacing (run_id,
# dataset_id, job status, ...).
_STANDARD_RECORD_ATTRS = frozenset(logging.LogRecord("", 0, "", 0, "", (), None).__dict__.keys())


class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _STANDARD_RECORD_ATTRS and key not in payload:
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        try:
            return json.dumps(payload, default=str)
        except (TypeError, ValueError):
            return json.dumps({**payload, "message": str(payload.get("message"))}, default=str)


def configure_logging(level: int = logging.INFO) -> None:
    """Attaches a single JSON stdout handler to the root logger. Safe to call
    more than once (e.g. once from api/main.py and once from main.py in the
    same process during tests) - only the first call has any effect."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(JSONFormatter())

    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(handler)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
