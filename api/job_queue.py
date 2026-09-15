"""Phase 9: bounded background job execution.

Before Phase 9, api/run_store.py spawned one raw `threading.Thread` per run
with no cap - every incoming run request got its own thread immediately, so
N simultaneous uploads meant N simultaneous AutoGluon fits competing for the
same CPU/RAM, with no queueing and no way to reason about how many jobs were
in flight. That's the single-process assumption this module removes.

JobQueue wraps a fixed-size `concurrent.futures.ThreadPoolExecutor`: jobs
beyond `max_workers` sit in the executor's internal queue (job status
"queued") until a worker is free (status flips to "running" - see
api/run_store.py, which does that transition since only it knows about the
DB row). A thread pool, not a process pool, because the existing pipeline
(DataFrames, model objects) isn't designed to cross a process boundary, and
Phase 9 explicitly avoids introducing a distributed/multiprocess
architecture unless the app's size actually needs it.

Cancellation is two-tier, matching what's actually possible without killing
a thread outright (Python can't do that safely):
  - A job still sitting in the pool's internal queue: `Future.cancel()`
    succeeds outright (it never starts).
  - A job already running: cancel() sets a `threading.Event` that
    orchestration/graph.py checks between pipeline steps (see
    PipelineCancelled) - cooperative, takes effect at the next node
    boundary, not instantly.
"""
from __future__ import annotations

import os
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable, Optional

DEFAULT_MAX_WORKERS = int(os.environ.get("PIPELINE_MAX_WORKERS", "2"))
DEFAULT_MAX_QUEUED = int(os.environ.get("PIPELINE_MAX_QUEUED_RUNS", "20"))


@dataclass
class _JobHandle:
    future: Future
    cancel_event: threading.Event


class QueueFullError(RuntimeError):
    """Raised by submit() when accepting another job would exceed
    max_queued - a resource limit (Phase 9) so an unbounded number of heavy
    AutoML jobs can never pile up in memory waiting for a free worker."""


class JobQueue:
    def __init__(self, max_workers: int = DEFAULT_MAX_WORKERS, max_queued: int = DEFAULT_MAX_QUEUED) -> None:
        self.max_workers = max_workers
        self.max_queued = max_queued
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="pipeline-worker")
        self._lock = threading.Lock()
        self._jobs: dict[str, _JobHandle] = {}

    def depth(self) -> int:
        """Jobs currently tracked (queued + running) - not yet cleaned up
        after completion (see forget())."""
        with self._lock:
            return len(self._jobs)

    def can_accept(self) -> bool:
        return self.depth() < self.max_queued

    def submit(self, job_id: str, fn: Callable, *args, **kwargs) -> threading.Event:
        """Submits `fn(*args, cancel_event=event, **kwargs)` to the pool.
        Raises QueueFullError if max_queued would be exceeded. Returns the
        cancel_event so the caller can pass it through to run_pipeline()."""
        with self._lock:
            if len(self._jobs) >= self.max_queued:
                raise QueueFullError(
                    f"Too many runs already queued/in progress (limit {self.max_queued}). Try again shortly."
                )
            cancel_event = threading.Event()
            future = self._executor.submit(self._run_and_forget, job_id, fn, cancel_event, args, kwargs)
            self._jobs[job_id] = _JobHandle(future=future, cancel_event=cancel_event)
            return cancel_event

    def _run_and_forget(self, job_id: str, fn: Callable, cancel_event: threading.Event, args, kwargs) -> None:
        try:
            fn(*args, cancel_event=cancel_event, **kwargs)
        finally:
            self.forget(job_id)

    def cancel(self, job_id: str) -> Optional[str]:
        """Returns "cancelled_before_start", "cancel_requested" (already
        running - cooperative), or None (unknown job_id, e.g. it already
        finished)."""
        with self._lock:
            handle = self._jobs.get(job_id)
        if handle is None:
            return None
        if handle.future.cancel():
            # The executor will never invoke _run_and_forget for a future
            # cancelled before it started, so its usual `finally: forget()`
            # never runs - do it here instead, or _jobs would leak this id.
            self.forget(job_id)
            return "cancelled_before_start"
        handle.cancel_event.set()
        return "cancel_requested"

    def forget(self, job_id: str) -> None:
        with self._lock:
            self._jobs.pop(job_id, None)

    def shutdown(self, wait: bool = False) -> None:
        self._executor.shutdown(wait=wait, cancel_futures=True)


_default_queue: Optional[JobQueue] = None
_default_queue_lock = threading.Lock()


def get_default_queue() -> JobQueue:
    global _default_queue
    if _default_queue is None:
        with _default_queue_lock:
            if _default_queue is None:
                _default_queue = JobQueue()
    return _default_queue


def set_default_queue(queue: JobQueue) -> None:
    """Test hook - lets tests swap in a JobQueue with different limits."""
    global _default_queue
    _default_queue = queue
