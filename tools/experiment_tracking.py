"""Phase 8: Experiment Tracking and Reproducibility.

Records everything about one pipeline run into a single JSON-serializable
ExperimentRecord: run identity/timing, dataset metadata/hash, configuration,
the problem definition and experiment plan, every candidate model and its
parameters/metrics, the selected model, runtime, errors, and the generated
report's location. This is observability only - it never influences pipeline
behavior, and building/saving a record must never be able to fail a run (see
build_record()/every ExperimentStore.save()).

No external experiment-tracking platform (MLflow, W&B, ...) yet, per the
Phase 8 scope - runs are appended to a local JSON-lines file
(reports/experiments.jsonl, next to the existing HTML reports) via
ExperimentStore, a tiny interface with an in-memory and a JSON-lines
implementation. ExperimentRecord is already a plain JSON-safe pydantic model,
so adding real persistent storage later (SQLite, Postgres, ...) means writing
one new ExperimentStore implementation - the data model and every caller
(orchestration/graph.py) stay unchanged.

Reproducibility support:
  - set_global_seeds() fixes Python's and numpy's global RNGs at the start of
    every run. tools/splitting.py's train_test_split and
    tools/model_registry.py's RandomForest/XGBoost/LightGBM factories already
    pass their own fixed random_state; this covers everything else that reads
    the global RNG instead of taking a seed argument.
  - dataset_hash is a SHA-256 of the raw file bytes, so two runs can be
    compared for "same data, same config" instead of just "same file path".
  - library_versions/python snapshot what was actually installed at run time,
    so a re-run months later that behaves differently can be explained.
"""
from __future__ import annotations

import hashlib
import platform
import random
import sqlite3
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Protocol

import numpy as np
from pydantic import BaseModel, Field

GLOBAL_RANDOM_SEED = 42
_seed_lock = threading.Lock()

# Only libraries whose version can actually change training/evaluation
# results are tracked - not every transitive dependency.
TRACKED_LIBRARIES = (
    "pandas",
    "numpy",
    "scikit-learn",
    "langgraph",
    "pydantic",
    "xgboost",
    "lightgbm",
    "statsmodels",
    "shap",
    "autogluon.tabular",
    "autogluon.timeseries",
    "jinja2",
)

DEFAULT_STORE_PATH = Path("reports") / "experiments.jsonl"


def set_global_seeds(seed: int = GLOBAL_RANDOM_SEED) -> None:
    """Fixes Python's and numpy's global RNGs. Called once at the start of
    every pipeline run (orchestration/graph.py's run_pipeline).

    Phase 9 note: this mutates process-wide state, and Phase 9's bounded
    worker pool can run more than one pipeline concurrently in the same
    process. The lock only makes the mutation itself atomic - it does not
    make concurrent runs individually reproducible against each other's
    interleaving, since they still share one global RNG afterwards. That's
    an accepted, narrow gap: every model factory that actually matters for
    result determinism (RandomForest/XGBoost/LightGBM in
    tools/model_registry.py, the train/test split in tools/splitting.py)
    already takes its own explicit random_state instead of reading global
    state, so this only affects the few remaining calls (e.g.
    DummyClassifier) that don't.
    """
    with _seed_lock:
        random.seed(seed)
        np.random.seed(seed)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def hash_file(file_path: str | Path) -> Optional[str]:
    """SHA-256 of the raw dataset bytes - a content fingerprint independent of
    file path/mtime. Returns None (never raises) if the file can't be read."""
    try:
        digest = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def capture_library_versions() -> dict[str, str]:
    """Best-effort installed-version snapshot for the libraries that affect
    training/evaluation results. Missing/uninstalled optional dependencies
    (xgboost, lightgbm, statsmodels, shap, autogluon) are simply omitted
    rather than raising - tools/model_registry.py already tolerates them
    being absent."""
    from importlib.metadata import PackageNotFoundError, version

    versions: dict[str, str] = {"python": platform.python_version()}
    for name in TRACKED_LIBRARIES:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            continue
    return versions


class CandidateModelRecord(BaseModel):
    """One trained candidate, as recorded for reproducibility/audit - a
    narrowed view of tools/model_registry.py's ModelResult.to_dict(),
    keeping only what's needed to reproduce or explain a run's model choice."""

    model_name: str
    status: str
    parameters: dict = Field(default_factory=dict)
    metrics: dict = Field(default_factory=dict)
    score_test: Optional[float] = None
    training_time_s: Optional[float] = None
    errors: Optional[str] = None


class ExperimentRecord(BaseModel):
    """Everything about one pipeline run, in one JSON-serializable object.

    This is the data model persisted by ExperimentStore - designed to be
    stored as-is (model_dump(mode="json")) so a future persistent store
    (a real database) can adopt it without a schema rewrite; every field is a
    plain JSON-safe type, and run_id is a natural primary key.
    """

    run_id: str
    created_at: str
    completed_at: Optional[str] = None
    runtime_seconds: Optional[float] = None
    status: str = Field(
        default="running", description="One of: 'running', 'complete', 'needs_clarification', 'error'."
    )

    # configuration (reproducibility inputs)
    file_path: str
    business_description: str
    max_retries: int
    random_seed: int = GLOBAL_RANDOM_SEED

    # dataset metadata/hash
    dataset_hash: Optional[str] = None
    dataset_row_count: Optional[int] = None
    dataset_column_count: Optional[int] = None

    # problem definition / experiment plan / validation strategy
    problem_definition: Optional[dict] = None
    experiment_plan: Optional[dict] = None
    validation_strategy: Optional[dict] = None
    quality_report: Optional[dict] = Field(
        default=None,
        description="tools/profiling.py's analyze_data_quality() DataQualityReport (missing values, "
        "duplicates, possible outliers, invalid dtypes, suspicious/ID-like columns, possible leakage, "
        "possible sentinel-missing values) - the raw checks, independent of feature_selection below "
        "(what was actually dropped as a result). Persisted even when every list came back empty, so "
        "an empty feature_selection.dropped can be told apart from 'quality_analysis never ran'.",
    )
    feature_selection: Optional[dict] = Field(
        default=None,
        description="tools/feature_selection.py's kept/dropped column log. None for hierarchical "
        "scope (no selection applies there) - a valid, expected state, not a missing value.",
    )

    # candidate models / model parameters
    candidate_models: list[CandidateModelRecord] = Field(default_factory=list)

    # metrics / selected model
    metrics_summary: Optional[dict] = None
    selected_model: Optional[str] = None
    evaluator_decision: Optional[dict] = None
    retry_count: int = 0

    # reproducibility: recorded library/model versions
    library_versions: dict[str, str] = Field(default_factory=dict)
    model_versions: dict[str, str] = Field(default_factory=dict)

    # errors / report location
    errors: list[str] = Field(default_factory=list)
    report_path: Optional[str] = None
    clarification_question: Optional[str] = None


def build_record(
    run_id: str,
    started_at_iso: str,
    started_perf: float,
    file_path: str,
    business_description: str,
    max_retries: int,
    state: dict[str, Any],
    exception: Optional[BaseException] = None,
    status_override: Optional[str] = None,
) -> ExperimentRecord:
    """Assembles the final ExperimentRecord from run_pipeline()'s inputs and
    the graph's resulting (possibly partial, on failure) state. Never raises
    - a tracking bug must never mask or replace the pipeline's real result.

    status_override (Phase 9) lets a caller distinguish a deliberately
    cancelled run from a genuine error/exception (e.g. orchestration/graph.py
    catches its own PipelineCancelled and passes status_override="cancelled"
    instead of letting it fall through to the generic exception-derived
    "error" status).
    """
    runtime_seconds = round(time.perf_counter() - started_perf, 3)

    plan = state.get("plan")
    metrics = state.get("metrics") or {}
    decision = state.get("decision")
    dataset_profile = state.get("dataset_profile")
    problem_definition = state.get("problem_definition")
    data_quality_report = state.get("data_quality_report")

    candidate_models: list[CandidateModelRecord] = []
    errors: list[str] = []
    for result in metrics.get("candidate_results", []) or []:
        candidate_models.append(
            CandidateModelRecord(
                model_name=result.get("model_name", "unknown"),
                status=result.get("status", "unknown"),
                parameters=result.get("parameters") or {},
                metrics=result.get("metrics") or {},
                score_test=result.get("score_test"),
                training_time_s=result.get("training_time"),
                errors=result.get("errors"),
            )
        )
        if result.get("errors"):
            errors.append(f"{result.get('model_name', 'unknown')}: {result['errors']}")

    if status_override is not None:
        status = status_override
        if exception is not None:
            errors.append(str(exception))
    elif exception is not None:
        status = "error"
        errors.append(str(exception))
    elif state.get("needs_clarification"):
        status = "needs_clarification"
    else:
        status = "complete"

    model_versions: dict[str, str] = {}
    if metrics.get("model_path"):
        model_versions["autogluon_model_path"] = metrics["model_path"]

    return ExperimentRecord(
        run_id=run_id,
        created_at=started_at_iso,
        completed_at=now_iso(),
        runtime_seconds=runtime_seconds,
        status=status,
        file_path=str(file_path),
        business_description=business_description,
        max_retries=max_retries,
        random_seed=GLOBAL_RANDOM_SEED,
        dataset_hash=hash_file(file_path),
        dataset_row_count=dataset_profile.row_count if dataset_profile else None,
        dataset_column_count=dataset_profile.column_count if dataset_profile else None,
        problem_definition=problem_definition.model_dump(mode="json") if problem_definition else None,
        experiment_plan=plan.model_dump(mode="json") if plan else None,
        validation_strategy=(
            plan.validation_strategy.model_dump(mode="json") if plan and plan.validation_strategy else None
        ),
        quality_report=data_quality_report.model_dump(mode="json") if data_quality_report else None,
        feature_selection=state.get("feature_selection_log"),
        candidate_models=candidate_models,
        metrics_summary={k: v for k, v in metrics.items() if k != "candidate_results"},
        selected_model=decision.best_model if decision else None,
        evaluator_decision=decision.model_dump(mode="json") if decision else None,
        retry_count=state.get("retry_count", 0),
        library_versions=capture_library_versions(),
        model_versions=model_versions,
        errors=errors,
        report_path=state.get("report_path"),
        clarification_question=plan.clarification_question if plan and plan.needs_clarification else None,
    )


class ExperimentStore(Protocol):
    """Storage contract for ExperimentRecord. A future persistent store (a
    real database) only needs to implement this - orchestration/graph.py and
    api/run_store.py never depend on how records are actually stored."""

    def save(self, record: ExperimentRecord) -> None: ...

    def get(self, run_id: str) -> Optional[ExperimentRecord]: ...

    def list(self) -> list[ExperimentRecord]: ...


class InMemoryExperimentStore:
    """Single-process, non-persistent store - mirrors api/run_store.py's
    existing in-memory-only pattern. Useful for tests."""

    def __init__(self) -> None:
        self._records: dict[str, ExperimentRecord] = {}

    def save(self, record: ExperimentRecord) -> None:
        self._records[record.run_id] = record

    def get(self, run_id: str) -> Optional[ExperimentRecord]:
        return self._records.get(run_id)

    def list(self) -> list[ExperimentRecord]:
        return list(self._records.values())


class JSONLinesExperimentStore:
    """Simple local persistence: appends one JSON line per saved record to
    `path`. get()/list() read the file back and keep only the last line seen
    per run_id, so callers still see one current record per run even though
    the file itself is append-only.

    Deliberately the extent of persistence for Phase 8 - it is not a
    database, just enough to survive process restarts locally. Swapping this
    for a real database later is a new ExperimentStore implementation, not a
    change to ExperimentRecord or its callers.
    """

    def __init__(self, path: str | Path = DEFAULT_STORE_PATH) -> None:
        self.path = Path(path)

    def save(self, record: ExperimentRecord) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(record.model_dump_json() + "\n")
        except OSError:
            pass  # tracking must never fail the pipeline run

    def get(self, run_id: str) -> Optional[ExperimentRecord]:
        latest: Optional[ExperimentRecord] = None
        for record in self._read_all():
            if record.run_id == run_id:
                latest = record
        return latest

    def list(self) -> list[ExperimentRecord]:
        latest_by_id: dict[str, ExperimentRecord] = {}
        for record in self._read_all():
            latest_by_id[record.run_id] = record
        return list(latest_by_id.values())

    def _read_all(self):
        if not self.path.exists():
            return
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield ExperimentRecord.model_validate_json(line)
                except Exception:  # noqa: BLE001 - skip unreadable/corrupt lines
                    continue


class SQLiteExperimentStore:
    """Phase 9: persistent model/results metadata backed by SQLite instead of
    a JSON-lines file - a real query surface (WHERE status = ..., ORDER BY
    created_at) without adding a database server. One row per run_id
    (INSERT OR REPLACE on save, so re-saving the same run_id updates it in
    place rather than growing the table); the full ExperimentRecord is kept
    as a single JSON blob column plus a few plain columns for filtering, so
    adding a field to ExperimentRecord never requires a migration here.

    Uses a short-lived connection per call (SQLite handles many short
    connections from different threads fine, especially in WAL mode - see
    _connect()) rather than one long-lived connection shared across threads,
    so this is safe to use from Phase 9's bounded worker pool without an
    external lock.
    """

    def __init__(self, path: str | Path = "reports/experiments.db") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @contextmanager
    def _connect(self):
        """Opens a connection, commits/rolls back as a transaction, and
        always closes it - a bare `with sqlite3.connect(...) as conn:` only
        manages the transaction, not the connection itself, which leaks open
        file handles (and on Windows can leave the WAL/SHM files locked)."""
        conn = sqlite3.connect(str(self.path), timeout=30)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=30000")
            with conn:
                yield conn
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS experiment_records (
                    run_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    selected_model TEXT,
                    data TEXT NOT NULL
                )
                """
            )

    def save(self, record: ExperimentRecord) -> None:
        try:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO experiment_records (run_id, status, created_at, selected_model, data)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(run_id) DO UPDATE SET
                        status=excluded.status,
                        selected_model=excluded.selected_model,
                        data=excluded.data
                    """,
                    (record.run_id, record.status, record.created_at, record.selected_model, record.model_dump_json()),
                )
        except sqlite3.Error:
            pass  # tracking must never fail the pipeline run

    def get(self, run_id: str) -> Optional[ExperimentRecord]:
        try:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT data FROM experiment_records WHERE run_id = ?", (run_id,)
                ).fetchone()
        except sqlite3.Error:
            return None
        return ExperimentRecord.model_validate_json(row[0]) if row else None

    def list(self) -> list[ExperimentRecord]:
        try:
            with self._connect() as conn:
                rows = conn.execute(
                    "SELECT data FROM experiment_records ORDER BY created_at DESC"
                ).fetchall()
        except sqlite3.Error:
            return []
        return [ExperimentRecord.model_validate_json(row[0]) for row in rows]


_default_store: ExperimentStore = JSONLinesExperimentStore()


def get_default_store() -> ExperimentStore:
    return _default_store


def set_default_store(store: ExperimentStore) -> None:
    """Test/host hook - lets tests or an API layer swap in an
    InMemoryExperimentStore instead of writing to disk."""
    global _default_store
    _default_store = store
