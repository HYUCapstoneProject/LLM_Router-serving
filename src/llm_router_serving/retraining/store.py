from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock

ACTIVE_STATUSES = ("submitting", "submitted", "running", "pending", "canceling", "paused")

# A lease left in `submitting` never reached Vertex, or reached it and the
# process died before recording the run ID. `abandoned` releases such a lease
# once the service has reconciled it against Vertex.
ABANDONED_STATUS = "abandoned"

_COLUMNS = "batch_id, request_fingerprint, pipeline_run_id, status, updated_at"


@dataclass(frozen=True)
class RetrainingJob:
    batch_id: str
    request_fingerprint: str
    pipeline_run_id: str | None
    status: str
    updated_at: datetime | None


class BatchIdConflictError(RuntimeError):
    pass


class RetrainingInProgressError(RuntimeError):
    pass


class RetrainingJobStore:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._path, timeout=30)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS retraining_jobs (
                    batch_id TEXT PRIMARY KEY,
                    request_fingerprint TEXT NOT NULL,
                    pipeline_run_id TEXT,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

    def get(self, batch_id: str) -> RetrainingJob | None:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                f"SELECT {_COLUMNS} FROM retraining_jobs WHERE batch_id = ?",
                (batch_id,),
            ).fetchone()
        return self._to_job(row) if row else None

    def list_active(self) -> list[RetrainingJob]:
        placeholders = ",".join("?" for _ in ACTIVE_STATUSES)
        query = f"SELECT {_COLUMNS} FROM retraining_jobs WHERE status IN ({placeholders})"
        with self._lock, self._connect() as connection:
            rows = connection.execute(query, list(ACTIVE_STATUSES)).fetchall()
        return [self._to_job(row) for row in rows]

    def claim(self, *, batch_id: str, request_fingerprint: str) -> RetrainingJob | None:
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                f"SELECT {_COLUMNS} FROM retraining_jobs WHERE batch_id = ?",
                (batch_id,),
            ).fetchone()
            if existing:
                job = self._to_job(existing)
                if job.request_fingerprint != request_fingerprint:
                    raise BatchIdConflictError(batch_id)
                if job.pipeline_run_id:
                    return job
                if job.status in ACTIVE_STATUSES:
                    raise RetrainingInProgressError(batch_id)
                connection.execute(
                    "UPDATE retraining_jobs SET status = 'submitting', "
                    "updated_at = CURRENT_TIMESTAMP WHERE batch_id = ?",
                    (batch_id,),
                )
                return None

            placeholders = ",".join("?" for _ in ACTIVE_STATUSES)
            active = connection.execute(
                f"SELECT batch_id FROM retraining_jobs WHERE status IN ({placeholders}) LIMIT 1",
                ACTIVE_STATUSES,
            ).fetchone()
            if active:
                raise RetrainingInProgressError(str(active["batch_id"]))
            connection.execute(
                "INSERT INTO retraining_jobs "
                "(batch_id, request_fingerprint, status) VALUES (?, ?, 'submitting')",
                (batch_id, request_fingerprint),
            )
            return None

    def mark_submitted(self, *, batch_id: str, pipeline_run_id: str) -> None:
        self._update(batch_id, status="submitted", pipeline_run_id=pipeline_run_id)

    def mark_status(self, *, batch_id: str, status: str) -> None:
        self._update(batch_id, status=status)

    def _update(
        self,
        batch_id: str,
        *,
        status: str,
        pipeline_run_id: str | None = None,
    ) -> None:
        with self._lock, self._connect() as connection:
            if pipeline_run_id is None:
                connection.execute(
                    "UPDATE retraining_jobs SET status = ?, updated_at = CURRENT_TIMESTAMP "
                    "WHERE batch_id = ?",
                    (status, batch_id),
                )
            else:
                connection.execute(
                    "UPDATE retraining_jobs SET status = ?, pipeline_run_id = ?, "
                    "updated_at = CURRENT_TIMESTAMP WHERE batch_id = ?",
                    (status, pipeline_run_id, batch_id),
                )

    @staticmethod
    def _to_job(row: sqlite3.Row) -> RetrainingJob:
        return RetrainingJob(
            batch_id=str(row["batch_id"]),
            request_fingerprint=str(row["request_fingerprint"]),
            pipeline_run_id=(str(row["pipeline_run_id"]) if row["pipeline_run_id"] else None),
            status=str(row["status"]),
            updated_at=_parse_timestamp(row["updated_at"]),
        )


def _parse_timestamp(value: object) -> datetime | None:
    """Read a SQLite CURRENT_TIMESTAMP value, which is naive UTC."""
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed
