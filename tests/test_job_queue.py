"""Tests for api/job_queue.py (Phase 9): the bounded worker pool that
replaced one-unbounded-thread-per-run. Uses small, fast fake jobs - actual
pipeline execution is exercised in tests/test_pipeline_integration.py.
"""
from __future__ import annotations

import threading
import time

import pytest

from api.job_queue import JobQueue, QueueFullError


def _blocking_job(name, results, started, cancel_event=None):
    started.set()
    while not cancel_event.is_set():
        time.sleep(0.01)
    results.append(name)


def _instant_job(name, results, cancel_event=None):
    results.append(name)


def test_submit_runs_a_job_and_forgets_it_on_completion():
    queue = JobQueue(max_workers=2, max_queued=5)
    results: list[str] = []
    queue.submit("j1", _instant_job, "j1", results)

    deadline = time.time() + 2
    while queue.depth() > 0 and time.time() < deadline:
        time.sleep(0.01)

    assert results == ["j1"]
    assert queue.depth() == 0
    queue.shutdown(wait=True)


def test_submit_beyond_max_queued_raises_queue_full_error():
    queue = JobQueue(max_workers=1, max_queued=1)
    started = threading.Event()
    results: list[str] = []
    queue.submit("j1", _blocking_job, "j1", results, started)
    started.wait(timeout=2)

    with pytest.raises(QueueFullError):
        queue.submit("j2", _instant_job, "j2", results)

    # release j1 so the pool shuts down cleanly
    queue.cancel("j1")
    deadline = time.time() + 2
    while queue.depth() > 0 and time.time() < deadline:
        time.sleep(0.01)
    queue.shutdown(wait=True)


def test_cancel_before_start_prevents_the_job_from_ever_running():
    queue = JobQueue(max_workers=1, max_queued=5)
    started = threading.Event()
    results: list[str] = []

    # occupy the single worker so j2 sits in the internal queue, unstarted
    queue.submit("j1", _blocking_job, "j1", results, started)
    started.wait(timeout=2)
    queue.submit("j2", _instant_job, "j2", results)

    outcome = queue.cancel("j2")
    assert outcome == "cancelled_before_start"

    queue.cancel("j1")  # release j1 so nothing lingers
    deadline = time.time() + 2
    while queue.depth() > 0 and time.time() < deadline:
        time.sleep(0.01)

    assert "j2" not in results  # never ran
    queue.shutdown(wait=True)


def test_cancel_running_job_sets_cancel_event_cooperatively():
    queue = JobQueue(max_workers=1, max_queued=5)
    started = threading.Event()
    results: list[str] = []
    queue.submit("j1", _blocking_job, "j1", results, started)
    started.wait(timeout=2)

    outcome = queue.cancel("j1")
    assert outcome == "cancel_requested"

    deadline = time.time() + 2
    while queue.depth() > 0 and time.time() < deadline:
        time.sleep(0.01)
    assert results == ["j1"]  # the job noticed cancel_event and returned
    queue.shutdown(wait=True)


def test_cancel_unknown_job_id_returns_none():
    queue = JobQueue(max_workers=1, max_queued=5)
    assert queue.cancel("does-not-exist") is None
    queue.shutdown(wait=True)


def test_can_accept_reflects_queue_depth():
    queue = JobQueue(max_workers=1, max_queued=1)
    assert queue.can_accept() is True
    started = threading.Event()
    queue.submit("j1", _blocking_job, "j1", [], started)
    started.wait(timeout=2)
    assert queue.can_accept() is False
    queue.cancel("j1")
    queue.shutdown(wait=True)
