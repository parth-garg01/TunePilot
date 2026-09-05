"""SQLite-backed experiment registry.

Implements the tables listed in PRD section 34.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from ..logging_utils import get_logger
from .records import (
    ArtifactRecord,
    BackendRecord,
    CheckpointRecord,
    DatasetRecord,
    EvaluationRecord,
    EventRecord,
    ExperimentRecord,
    JobRecord,
    JobStatus,
    ModelRecord,
    ProjectRecord,
)

log = get_logger(__name__)


SCHEMA = """
CREATE TABLE IF NOT EXISTS projects (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,
    description TEXT DEFAULT '',
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS datasets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id INTEGER NOT NULL REFERENCES projects(id),
    path TEXT NOT NULL,
    fingerprint TEXT NOT NULL,
    examples INTEGER NOT NULL,
    tokens_est INTEGER NOT NULL,
    stats TEXT NOT NULL,
    UNIQUE(project_id, fingerprint)
);

CREATE TABLE IF NOT EXISTS models (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT NOT NULL,
    identifier TEXT NOT NULL,
    revision TEXT DEFAULT '',
    architecture TEXT DEFAULT '',
    parameters INTEGER DEFAULT 0,
    context_length INTEGER DEFAULT 0,
    license TEXT DEFAULT '',
    metadata TEXT NOT NULL,
    UNIQUE(source, identifier, revision)
);

CREATE TABLE IF NOT EXISTS experiments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id INTEGER NOT NULL REFERENCES projects(id),
    name TEXT NOT NULL,
    model_identifier TEXT NOT NULL,
    dataset_fingerprint TEXT NOT NULL,
    soup_config TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS backends (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,
    capabilities TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id INTEGER NOT NULL REFERENCES experiments(id),
    backend TEXT NOT NULL,
    backend_job_id TEXT,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    started_at TEXT,
    completed_at TEXT,
    gpu_type TEXT,
    gpu_count INTEGER DEFAULT 0,
    estimated_hours REAL DEFAULT 0,
    actual_hours REAL DEFAULT 0,
    error TEXT,
    retries INTEGER DEFAULT 0,
    priority INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS checkpoints (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id INTEGER NOT NULL REFERENCES jobs(id),
    step INTEGER NOT NULL,
    path TEXT NOT NULL,
    created_at TEXT NOT NULL,
    metrics TEXT NOT NULL,
    valid INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS evaluations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id INTEGER NOT NULL REFERENCES experiments(id),
    layer TEXT NOT NULL,
    metric TEXT NOT NULL,
    value REAL NOT NULL,
    normalized REAL NOT NULL,
    metadata TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS artifacts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id INTEGER NOT NULL REFERENCES experiments(id),
    kind TEXT NOT NULL,
    path TEXT NOT NULL,
    digest TEXT NOT NULL,
    size_bytes INTEGER NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id INTEGER,
    job_id INTEGER,
    kind TEXT NOT NULL,
    message TEXT NOT NULL,
    payload TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _dumps(x: Any) -> str:
    return json.dumps(x, default=str, sort_keys=True)


def _loads(x: str | None) -> Any:
    return json.loads(x) if x else {}


class ExperimentRegistry:
    """Small typed wrapper around SQLite for orchestrator metadata."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    @contextmanager
    def transaction(self):
        try:
            yield self._conn
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    # ------------------------- projects -------------------------

    def ensure_project(self, name: str, description: str = "") -> ProjectRecord:
        row = self._conn.execute("SELECT * FROM projects WHERE name = ?", (name,)).fetchone()
        if row:
            return ProjectRecord(**dict(row))
        with self.transaction() as c:
            cur = c.execute(
                "INSERT INTO projects (name, description, created_at) VALUES (?, ?, ?)",
                (name, description, _now()),
            )
        return ProjectRecord(id=cur.lastrowid, name=name, description=description, created_at=_now())

    # ------------------------- datasets -------------------------

    def upsert_dataset(self, r: DatasetRecord) -> DatasetRecord:
        with self.transaction() as c:
            cur = c.execute(
                """INSERT INTO datasets (project_id, path, fingerprint, examples, tokens_est, stats)
                   VALUES (?, ?, ?, ?, ?, ?)
                   ON CONFLICT(project_id, fingerprint) DO UPDATE SET
                     path=excluded.path, examples=excluded.examples,
                     tokens_est=excluded.tokens_est, stats=excluded.stats""",
                (r.project_id, r.path, r.fingerprint, r.examples, r.tokens_est, _dumps(r.stats)),
            )
        r.id = cur.lastrowid
        return r

    # ------------------------- models -------------------------

    def upsert_model(self, r: ModelRecord) -> ModelRecord:
        with self.transaction() as c:
            cur = c.execute(
                """INSERT INTO models (source, identifier, revision, architecture, parameters,
                                       context_length, license, metadata)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(source, identifier, revision) DO UPDATE SET
                     architecture=excluded.architecture, parameters=excluded.parameters,
                     context_length=excluded.context_length, license=excluded.license,
                     metadata=excluded.metadata""",
                (
                    r.source, r.identifier, r.revision, r.architecture,
                    r.parameters, r.context_length, r.license, _dumps(r.metadata),
                ),
            )
        r.id = cur.lastrowid
        return r

    # ------------------------- experiments -------------------------

    def create_experiment(self, r: ExperimentRecord) -> ExperimentRecord:
        r.created_at = r.created_at or _now()
        r.updated_at = _now()
        with self.transaction() as c:
            cur = c.execute(
                """INSERT INTO experiments (project_id, name, model_identifier, dataset_fingerprint,
                                            soup_config, status, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    r.project_id, r.name, r.model_identifier, r.dataset_fingerprint,
                    _dumps(r.soup_config), r.status, r.created_at, r.updated_at,
                ),
            )
        r.id = cur.lastrowid
        return r

    def set_experiment_status(self, experiment_id: int, status: str) -> None:
        with self.transaction() as c:
            c.execute(
                "UPDATE experiments SET status = ?, updated_at = ? WHERE id = ?",
                (status, _now(), experiment_id),
            )

    def list_experiments(self, project_id: int | None = None) -> list[ExperimentRecord]:
        if project_id is None:
            rows = self._conn.execute("SELECT * FROM experiments ORDER BY id").fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM experiments WHERE project_id = ? ORDER BY id", (project_id,)
            ).fetchall()
        return [
            ExperimentRecord(
                id=r["id"],
                project_id=r["project_id"],
                name=r["name"],
                model_identifier=r["model_identifier"],
                dataset_fingerprint=r["dataset_fingerprint"],
                soup_config=_loads(r["soup_config"]),
                status=r["status"],
                created_at=r["created_at"],
                updated_at=r["updated_at"],
            )
            for r in rows
        ]

    # ------------------------- jobs -------------------------

    def create_job(self, r: JobRecord) -> JobRecord:
        r.created_at = r.created_at or _now()
        with self.transaction() as c:
            cur = c.execute(
                """INSERT INTO jobs (experiment_id, backend, backend_job_id, status, created_at,
                                     gpu_type, gpu_count, estimated_hours, priority)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    r.experiment_id, r.backend, r.backend_job_id, r.status, r.created_at,
                    r.gpu_type, r.gpu_count, r.estimated_hours, r.priority,
                ),
            )
        r.id = cur.lastrowid
        return r

    def update_job(self, job_id: int, **fields: Any) -> None:
        if not fields:
            return
        keys = ", ".join(f"{k} = ?" for k in fields)
        with self.transaction() as c:
            c.execute(f"UPDATE jobs SET {keys} WHERE id = ?", (*fields.values(), job_id))

    def set_job_status(self, job_id: int, status: JobStatus | str, *, error: str | None = None) -> None:
        s = status.value if isinstance(status, JobStatus) else str(status)
        fields: dict[str, Any] = {"status": s}
        if s == JobStatus.RUNNING.value:
            fields["started_at"] = _now()
        if s in {"COMPLETED", "FAILED", "CANCELLED"}:
            fields["completed_at"] = _now()
        if error:
            fields["error"] = error
        self.update_job(job_id, **fields)

    def list_jobs(self, experiment_id: int | None = None) -> list[JobRecord]:
        query = "SELECT * FROM jobs"
        params: tuple = ()
        if experiment_id is not None:
            query += " WHERE experiment_id = ?"
            params = (experiment_id,)
        query += " ORDER BY id"
        rows = self._conn.execute(query, params).fetchall()
        return [JobRecord(**dict(r)) for r in rows]

    def get_job(self, job_id: int) -> JobRecord | None:
        row = self._conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return JobRecord(**dict(row)) if row else None

    # ------------------------- checkpoints -------------------------

    def record_checkpoint(self, r: CheckpointRecord) -> CheckpointRecord:
        r.created_at = r.created_at or _now()
        with self.transaction() as c:
            cur = c.execute(
                """INSERT INTO checkpoints (job_id, step, path, created_at, metrics, valid)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (r.job_id, r.step, r.path, r.created_at, _dumps(r.metrics), int(r.valid)),
            )
        r.id = cur.lastrowid
        return r

    def latest_valid_checkpoint(self, job_id: int) -> CheckpointRecord | None:
        row = self._conn.execute(
            "SELECT * FROM checkpoints WHERE job_id = ? AND valid = 1 ORDER BY step DESC LIMIT 1",
            (job_id,),
        ).fetchone()
        if not row:
            return None
        d = dict(row)
        d["metrics"] = _loads(d["metrics"])
        d["valid"] = bool(d["valid"])
        return CheckpointRecord(**d)

    # ------------------------- evaluations -------------------------

    def record_evaluation(self, r: EvaluationRecord) -> EvaluationRecord:
        r.created_at = r.created_at or _now()
        with self.transaction() as c:
            cur = c.execute(
                """INSERT INTO evaluations (experiment_id, layer, metric, value, normalized,
                                            metadata, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (r.experiment_id, r.layer, r.metric, r.value, r.normalized,
                 _dumps(r.metadata), r.created_at),
            )
        r.id = cur.lastrowid
        return r

    def evaluations_for(self, experiment_id: int) -> list[EvaluationRecord]:
        rows = self._conn.execute(
            "SELECT * FROM evaluations WHERE experiment_id = ? ORDER BY id", (experiment_id,),
        ).fetchall()
        out: list[EvaluationRecord] = []
        for r in rows:
            d = dict(r)
            d["metadata"] = _loads(d["metadata"])
            out.append(EvaluationRecord(**d))
        return out

    # ------------------------- artifacts -------------------------

    def record_artifact(self, r: ArtifactRecord) -> ArtifactRecord:
        r.created_at = r.created_at or _now()
        with self.transaction() as c:
            cur = c.execute(
                """INSERT INTO artifacts (experiment_id, kind, path, digest, size_bytes, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (r.experiment_id, r.kind, r.path, r.digest, r.size_bytes, r.created_at),
            )
        r.id = cur.lastrowid
        return r

    # ------------------------- events -------------------------

    def log_event(self, r: EventRecord) -> EventRecord:
        r.created_at = r.created_at or _now()
        with self.transaction() as c:
            cur = c.execute(
                """INSERT INTO events (experiment_id, job_id, kind, message, payload, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (r.experiment_id, r.job_id, r.kind, r.message, _dumps(r.payload), r.created_at),
            )
        r.id = cur.lastrowid
        return r

    def recent_events(self, limit: int = 50) -> list[EventRecord]:
        rows = self._conn.execute(
            "SELECT * FROM events ORDER BY id DESC LIMIT ?", (limit,),
        ).fetchall()
        out: list[EventRecord] = []
        for r in rows:
            d = dict(r)
            d["payload"] = _loads(d["payload"])
            out.append(EventRecord(**d))
        return out
