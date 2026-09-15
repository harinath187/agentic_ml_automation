"""Tests for tools/file_cleanup.py (Phase 9): age-based retention for
uploads/reports/AutoGluon artifacts. Points the module's directory constants
at tmp_path so tests never touch the real uploads/reports/AutogluonModels.
"""
from __future__ import annotations

import os
import time

from tools import file_cleanup


def _age_file(path, hours_old: float) -> None:
    old_time = time.time() - hours_old * 3600
    os.utime(path, (old_time, old_time))


def test_cleanup_removes_only_stale_uploads(tmp_path, monkeypatch):
    monkeypatch.setattr(file_cleanup, "UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setattr(file_cleanup, "REPORTS_DIR", tmp_path / "reports_generated")
    monkeypatch.setattr(file_cleanup, "AUTOGLUON_DIR", tmp_path / "AutogluonModels")
    file_cleanup.UPLOAD_DIR.mkdir()

    stale = file_cleanup.UPLOAD_DIR / "old.csv"
    fresh = file_cleanup.UPLOAD_DIR / "new.csv"
    stale.write_text("a,b\n1,2\n")
    fresh.write_text("a,b\n3,4\n")
    _age_file(stale, hours_old=200)  # older than the default 168h retention
    _age_file(fresh, hours_old=1)

    summary = file_cleanup.cleanup_old_files(retention_hours=168)

    assert str(stale) in summary["uploads"]
    assert str(fresh) not in summary["uploads"]
    assert not stale.exists()
    assert fresh.exists()


def test_cleanup_never_removes_anything_within_the_safety_margin(tmp_path, monkeypatch):
    """Even a retention_hours of 0 must not delete a file modified in the
    last hour - MIN_SAFETY_MARGIN_S protects an in-flight run's files."""
    monkeypatch.setattr(file_cleanup, "UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setattr(file_cleanup, "REPORTS_DIR", tmp_path / "reports_generated")
    monkeypatch.setattr(file_cleanup, "AUTOGLUON_DIR", tmp_path / "AutogluonModels")
    file_cleanup.UPLOAD_DIR.mkdir()

    recent = file_cleanup.UPLOAD_DIR / "just_uploaded.csv"
    recent.write_text("a,b\n1,2\n")

    summary = file_cleanup.cleanup_old_files(retention_hours=0)

    assert summary["uploads"] == []
    assert recent.exists()


def test_cleanup_removes_stale_reports(tmp_path, monkeypatch):
    monkeypatch.setattr(file_cleanup, "UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setattr(file_cleanup, "REPORTS_DIR", tmp_path / "reports_generated")
    monkeypatch.setattr(file_cleanup, "AUTOGLUON_DIR", tmp_path / "AutogluonModels")
    file_cleanup.REPORTS_DIR.mkdir()

    stale_report = file_cleanup.REPORTS_DIR / "report_old.html"
    stale_report.write_text("<html></html>")
    _age_file(stale_report, hours_old=200)

    summary = file_cleanup.cleanup_old_files(retention_hours=168)

    assert not stale_report.exists()
    assert str(stale_report) in summary["reports"]


def test_cleanup_removes_stale_autogluon_run_directories(tmp_path, monkeypatch):
    monkeypatch.setattr(file_cleanup, "UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setattr(file_cleanup, "REPORTS_DIR", tmp_path / "reports_generated")
    monkeypatch.setattr(file_cleanup, "AUTOGLUON_DIR", tmp_path / "AutogluonModels")
    file_cleanup.AUTOGLUON_DIR.mkdir()

    stale_dir = file_cleanup.AUTOGLUON_DIR / "run_deadbeef"
    stale_dir.mkdir()
    (stale_dir / "predictor.pkl").write_text("fake")
    _age_file(stale_dir, hours_old=200)

    not_a_run_dir = file_cleanup.AUTOGLUON_DIR / "not_a_run"
    not_a_run_dir.mkdir()
    _age_file(not_a_run_dir, hours_old=200)

    summary = file_cleanup.cleanup_old_files(retention_hours=168)

    assert not stale_dir.exists()
    assert not_a_run_dir.exists()  # doesn't match the "run_" prefix - left alone
    assert str(stale_dir) in summary["autogluon_models"]


def test_cleanup_on_missing_directories_is_a_noop(tmp_path, monkeypatch):
    monkeypatch.setattr(file_cleanup, "UPLOAD_DIR", tmp_path / "does_not_exist_uploads")
    monkeypatch.setattr(file_cleanup, "REPORTS_DIR", tmp_path / "does_not_exist_reports")
    monkeypatch.setattr(file_cleanup, "AUTOGLUON_DIR", tmp_path / "does_not_exist_models")

    summary = file_cleanup.cleanup_old_files(retention_hours=1)

    assert summary == {"uploads": [], "reports": [], "autogluon_models": []}
