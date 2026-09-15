"""Phase 9: File cleanup.

Three directories grow without bound as the app is used: uploaded datasets
(`uploads/`), generated reports (`reports/generated/`), and AutoGluon's own
per-run model artifacts (`AutogluonModels/run_*`). None of these are cleaned
up anywhere else in the codebase. This module deletes entries older than a
configurable retention window - age-based only (no cross-check against "is a
run still using this file"), so a generous default retention plus a safety
margin (anything modified in the last hour is never touched, regardless of
retention setting) protects an in-flight run's files from being swept up
mid-run.

Deliberately simple: no external scheduler/cron dependency. api/main.py runs
this on a plain daemon thread loop at startup; it's also safe to call
directly (e.g. from a manual maintenance script) since it never raises -
a cleanup bug must never crash the API process or a pipeline run.
"""
from __future__ import annotations

import os
import shutil
import time
from pathlib import Path
from typing import Iterable

from tools.logging_config import get_logger

logger = get_logger(__name__)

DEFAULT_RETENTION_HOURS = float(os.environ.get("FILE_RETENTION_HOURS", "168"))  # 7 days
MIN_SAFETY_MARGIN_S = 60 * 60  # never delete anything modified in the last hour, regardless of retention

UPLOAD_DIR = Path("uploads")
REPORTS_DIR = Path("reports") / "generated"
AUTOGLUON_DIR = Path("AutogluonModels")


def _is_stale(path: Path, cutoff: float) -> bool:
    try:
        return path.stat().st_mtime < cutoff
    except OSError:
        return False


def _cleanup_files(directory: Path, cutoff: float, patterns: Iterable[str]) -> list[str]:
    removed: list[str] = []
    if not directory.exists():
        return removed
    for pattern in patterns:
        for path in directory.glob(pattern):
            if not path.is_file() or not _is_stale(path, cutoff):
                continue
            try:
                path.unlink()
                removed.append(str(path))
            except OSError as exc:
                logger.warning("cleanup_failed", extra={"path": str(path), "error": str(exc)})
    return removed


def _cleanup_dirs(directory: Path, cutoff: float, prefix: str) -> list[str]:
    removed: list[str] = []
    if not directory.exists():
        return removed
    for path in directory.iterdir():
        if not path.is_dir() or not path.name.startswith(prefix) or not _is_stale(path, cutoff):
            continue
        try:
            shutil.rmtree(path)
            removed.append(str(path))
        except OSError as exc:
            logger.warning("cleanup_failed", extra={"path": str(path), "error": str(exc)})
    return removed


def cleanup_old_files(retention_hours: float = DEFAULT_RETENTION_HOURS) -> dict:
    """Deletes uploads/reports/AutoGluon artifacts older than
    `retention_hours` (bounded below by a 1-hour safety margin regardless of
    the configured retention). Never raises - returns what it removed (or
    would have, best-effort) so callers can log it; any per-file/per-dir
    failure is logged and skipped rather than propagated.
    """
    retention_s = max(retention_hours, 0) * 3600
    cutoff = time.time() - max(retention_s, MIN_SAFETY_MARGIN_S)

    try:
        removed_uploads = _cleanup_files(UPLOAD_DIR, cutoff, ("*.csv", "*.xls", "*.xlsx"))
        removed_reports = _cleanup_files(REPORTS_DIR, cutoff, ("*.html",))
        removed_models = _cleanup_dirs(AUTOGLUON_DIR, cutoff, "run_")
    except Exception as exc:  # noqa: BLE001 - cleanup must never break the caller
        logger.error("cleanup_run_failed", extra={"error": str(exc)})
        return {"uploads": [], "reports": [], "autogluon_models": [], "error": str(exc)}

    summary = {"uploads": removed_uploads, "reports": removed_reports, "autogluon_models": removed_models}
    total = len(removed_uploads) + len(removed_reports) + len(removed_models)
    if total:
        logger.info(
            "cleanup_completed",
            extra={
                "removed_uploads": len(removed_uploads),
                "removed_reports": len(removed_reports),
                "removed_autogluon_models": len(removed_models),
            },
        )
    return summary
