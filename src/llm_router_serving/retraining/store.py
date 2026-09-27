from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock

logger = logging.getLogger(__name__)

ACTIVE_STATUSES = ("submitting", "submitted", "running", "pending", "canceling", "paused")

# A lease left in `submitting` never reached Vertex, or reached it and the
# process died before recording the run ID. `abandoned` releases such a lease
# once the service has reconciled it against Vertex.
ABANDONED_STATUS = "abandoned"

# A succeeded direct-promotion run whose CURRENT pointer was manually rolled
# back outside the ledger (see RetrainingJobStore.reconcile_manual_rollback).
# Distinct from `cancelled`, which means Vertex itself cancelled the run.
ROLLED_BACK_LEDGER_STATUS = "rolled_back"

_COLUMNS = (
    "batch_id, request_fingerprint, pipeline_run_id, status, created_at, updated_at, "
    "submitted_at, started_at, ended_at, "
    "logical_model_version, candidate_model_version, training_mode, "
    "initialization_mode, version_policy_mode, promotion_mode, canonical_batch_digest, "
    "base_model_version, base_model_uri, "
    "feedback_manifest_uri, feedback_events_uri, replay_manifest_uri, "
    "replay_events_uri"
)


@dataclass(frozen=True)
class RetrainingJob:
    batch_id: str
    request_fingerprint: str
    pipeline_run_id: str | None
    status: str
    created_at: datetime | None = None
    updated_at: datetime | None = None
    submitted_at: datetime | None = None
    started_at: datetime | None = None
    ended_at: datetime | None = None
    logical_model_version: str | None = None
    candidate_model_version: str | None = None
    training_mode: str | None = None
    initialization_mode: str | None = None
    version_policy_mode: str | None = None
    promotion_mode: str = "quality_gate"
    canonical_batch_digest: str | None = None
    base_model_version: str | None = None
    base_model_uri: str | None = None
    feedback_manifest_uri: str | None = None
    feedback_events_uri: str | None = None
    replay_manifest_uri: str | None = None
    replay_events_uri: str | None = None


class BatchIdConflictError(RuntimeError):
    pass


class RetrainingInProgressError(RuntimeError):
    pass


class VersionPolicyRevisionConflictError(RuntimeError):
    pass


class LedgerBatchNotFoundError(RuntimeError):
    pass


class LedgerReconciliationError(RuntimeError):
    """Raised when a manual-rollback reconciliation cannot be applied as requested."""


@dataclass(frozen=True)
class VersionPolicyState:
    revision: int
    mode: str
    pending_mode: str | None
    pending_requested_at: datetime | None
    pending_requested_by: str | None
    pending_reason: str | None
    updated_at: datetime
    updated_by: str


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
            existing = {
                str(row["name"]) for row in connection.execute("PRAGMA table_info(retraining_jobs)")
            }
            migrations = {
                "logical_model_version": "TEXT",
                "candidate_model_version": "TEXT",
                "training_mode": "TEXT",
                "initialization_mode": "TEXT",
                "version_policy_mode": "TEXT",
                "promotion_mode": "TEXT NOT NULL DEFAULT 'quality_gate'",
                "canonical_batch_digest": "TEXT",
                "base_model_version": "TEXT",
                "base_model_uri": "TEXT",
                "feedback_manifest_uri": "TEXT",
                "feedback_events_uri": "TEXT",
                "replay_manifest_uri": "TEXT",
                "replay_events_uri": "TEXT",
                "submitted_at": "TEXT",
                "started_at": "TEXT",
                "ended_at": "TEXT",
            }
            for column, sql_type in migrations.items():
                if column not in existing:
                    connection.execute(
                        f"ALTER TABLE retraining_jobs ADD COLUMN {column} {sql_type}"
                    )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS retraining_version_policy (
                    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                    revision INTEGER NOT NULL,
                    mode TEXT NOT NULL,
                    pending_mode TEXT,
                    pending_requested_at TEXT,
                    pending_requested_by TEXT,
                    pending_reason TEXT,
                    updated_at TEXT NOT NULL,
                    updated_by TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS retraining_version_policy_audit (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    revision INTEGER NOT NULL,
                    action TEXT NOT NULL,
                    active_mode TEXT NOT NULL,
                    pending_mode TEXT,
                    actor TEXT NOT NULL,
                    reason TEXT,
                    created_at TEXT NOT NULL
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

    def list_succeeded(self) -> list[RetrainingJob]:
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                f"SELECT {_COLUMNS} FROM retraining_jobs WHERE status = 'succeeded' ORDER BY rowid"
            ).fetchall()
        return [self._to_job(row) for row in rows]

    def claim(
        self,
        *,
        batch_id: str,
        request_fingerprint: str,
        logical_model_version: str | None = None,
        candidate_model_version: str | None = None,
        training_mode: str | None = None,
        initialization_mode: str | None = None,
        version_policy_mode: str | None = None,
        promotion_mode: str = "quality_gate",
        canonical_batch_digest: str | None = None,
        base_model_version: str | None = None,
        base_model_uri: str | None = None,
        feedback_manifest_uri: str | None = None,
        feedback_events_uri: str | None = None,
        replay_manifest_uri: str | None = None,
        replay_events_uri: str | None = None,
    ) -> RetrainingJob | None:
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
                    "logical_model_version = ?, candidate_model_version = ?, "
                    "training_mode = ?, initialization_mode = ?, base_model_version = ?, "
                    "version_policy_mode = ?, promotion_mode = ?, canonical_batch_digest = ?, "
                    "base_model_uri = ?, feedback_manifest_uri = NULL, "
                    "feedback_events_uri = NULL, replay_manifest_uri = NULL, "
                    "replay_events_uri = NULL, updated_at = CURRENT_TIMESTAMP "
                    "WHERE batch_id = ?",
                    (
                        logical_model_version,
                        candidate_model_version,
                        training_mode,
                        initialization_mode,
                        base_model_version,
                        version_policy_mode,
                        promotion_mode,
                        canonical_batch_digest,
                        base_model_uri,
                        batch_id,
                    ),
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
                "(batch_id, request_fingerprint, status, logical_model_version, "
                "candidate_model_version, training_mode, initialization_mode, "
                "version_policy_mode, promotion_mode, canonical_batch_digest, "
                "base_model_version, base_model_uri, feedback_manifest_uri, "
                "feedback_events_uri, replay_manifest_uri, replay_events_uri) "
                "VALUES (?, ?, 'submitting', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    batch_id,
                    request_fingerprint,
                    logical_model_version,
                    candidate_model_version,
                    training_mode,
                    initialization_mode,
                    version_policy_mode,
                    promotion_mode,
                    canonical_batch_digest,
                    base_model_version,
                    base_model_uri,
                    feedback_manifest_uri,
                    feedback_events_uri,
                    replay_manifest_uri,
                    replay_events_uri,
                ),
            )
            return None

    def mark_submitted(self, *, batch_id: str, pipeline_run_id: str) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                "UPDATE retraining_jobs SET status = 'submitted', pipeline_run_id = ?, "
                "submitted_at = COALESCE(submitted_at, CURRENT_TIMESTAMP), "
                "updated_at = CURRENT_TIMESTAMP WHERE batch_id = ?",
                (pipeline_run_id, batch_id),
            )

    def set_materialized_feedback(
        self,
        *,
        batch_id: str,
        feedback_manifest_uri: str,
        feedback_events_uri: str,
        replay_manifest_uri: str,
        replay_events_uri: str,
    ) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                "UPDATE retraining_jobs SET feedback_manifest_uri = ?, "
                "feedback_events_uri = ?, replay_manifest_uri = ?, replay_events_uri = ?, "
                "updated_at = CURRENT_TIMESTAMP WHERE batch_id = ?",
                (
                    feedback_manifest_uri,
                    feedback_events_uri,
                    replay_manifest_uri,
                    replay_events_uri,
                    batch_id,
                ),
            )

    def mark_status(self, *, batch_id: str, status: str) -> None:
        terminal = status in {"succeeded", "rejected", "failed", "cancelled"}
        with self._lock, self._connect() as connection:
            if terminal:
                connection.execute(
                    "UPDATE retraining_jobs SET status = ?, "
                    "started_at = COALESCE(started_at, submitted_at), "
                    "ended_at = COALESCE(ended_at, CURRENT_TIMESTAMP), "
                    "updated_at = CURRENT_TIMESTAMP WHERE batch_id = ?",
                    (status, batch_id),
                )
            else:
                connection.execute(
                    "UPDATE retraining_jobs SET status = ?, "
                    "started_at = CASE WHEN ? = 'running' "
                    "THEN COALESCE(started_at, CURRENT_TIMESTAMP) ELSE started_at END, "
                    "updated_at = CURRENT_TIMESTAMP WHERE batch_id = ?",
                    (status, status, batch_id),
                )

    def reconcile_manual_rollback(self, *, batch_id: str, reason: str) -> RetrainingJob:
        """Exclude a succeeded run from ledger planning after a manual CURRENT rollback.

        The training-plan trigger (`build_continual_training_plan`) counts rows with
        `status = 'succeeded'` and reads the most recent one for the next baseline and
        anchor. A manual `model_publish_cli rollback` of CURRENT does not touch this
        table, so an untouched ledger keeps planning against a baseline that no longer
        matches the published CURRENT pointer — the next direct-promotion run then
        fails only after training completes (`CURRENT differs from direct baseline`).

        This only transitions a row that is currently `succeeded`, so it cannot hide
        an active or already-terminal run, and it is intentionally distinct from
        `cancelled` (which means Vertex itself cancelled the pipeline).
        """
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                f"SELECT {_COLUMNS} FROM retraining_jobs WHERE batch_id = ?",
                (batch_id,),
            ).fetchone()
            if existing is None:
                raise LedgerBatchNotFoundError(batch_id)
            job = self._to_job(existing)
            if job.status != "succeeded":
                raise LedgerReconciliationError(
                    f"batch {batch_id!r} is not in 'succeeded' status "
                    f"(found {job.status!r}); nothing changed"
                )
            connection.execute(
                "UPDATE retraining_jobs SET status = ?, updated_at = CURRENT_TIMESTAMP "
                "WHERE batch_id = ?",
                (ROLLED_BACK_LEDGER_STATUS, batch_id),
            )
            updated = connection.execute(
                f"SELECT {_COLUMNS} FROM retraining_jobs WHERE batch_id = ?",
                (batch_id,),
            ).fetchone()
        logger.warning(
            "Reconciled ledger for batch %s after manual CURRENT rollback: %s",
            batch_id,
            reason,
        )
        return self._to_job(updated)

    def get_or_create_version_policy(
        self, *, default_mode: str, actor: str = "configuration"
    ) -> VersionPolicyState:
        now = datetime.now(UTC).isoformat()
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM retraining_version_policy WHERE singleton = 1"
            ).fetchone()
            if row is None:
                connection.execute(
                    "INSERT INTO retraining_version_policy "
                    "(singleton, revision, mode, updated_at, updated_by) VALUES (1, 0, ?, ?, ?)",
                    (default_mode, now, actor),
                )
                row = connection.execute(
                    "SELECT * FROM retraining_version_policy WHERE singleton = 1"
                ).fetchone()
        assert row is not None
        return self._to_policy(row)

    def schedule_version_policy_change(
        self,
        *,
        expected_revision: int,
        mode: str,
        actor: str,
        reason: str,
    ) -> VersionPolicyState:
        now = datetime.now(UTC).isoformat()
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM retraining_version_policy WHERE singleton = 1"
            ).fetchone()
            if row is None or int(row["revision"]) != expected_revision:
                raise VersionPolicyRevisionConflictError(str(expected_revision))
            revision = expected_revision + 1
            active_mode = str(row["mode"])
            pending_mode = None if mode == active_mode else mode
            connection.execute(
                "UPDATE retraining_version_policy SET revision = ?, pending_mode = ?, "
                "pending_requested_at = ?, pending_requested_by = ?, pending_reason = ?, "
                "updated_at = ?, updated_by = ? WHERE singleton = 1",
                (
                    revision,
                    pending_mode,
                    now if pending_mode else None,
                    actor if pending_mode else None,
                    reason if pending_mode else None,
                    now,
                    actor,
                ),
            )
            connection.execute(
                "INSERT INTO retraining_version_policy_audit "
                "(revision, action, active_mode, pending_mode, actor, reason, created_at) "
                "VALUES (?, 'scheduled', ?, ?, ?, ?, ?)",
                (revision, active_mode, pending_mode, actor, reason, now),
            )
            updated = connection.execute(
                "SELECT * FROM retraining_version_policy WHERE singleton = 1"
            ).fetchone()
        assert updated is not None
        return self._to_policy(updated)

    def activate_pending_version_policy(self, *, actor: str = "major-boundary") -> bool:
        now = datetime.now(UTC).isoformat()
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM retraining_version_policy WHERE singleton = 1"
            ).fetchone()
            if row is None or not row["pending_mode"]:
                return False
            revision = int(row["revision"]) + 1
            mode = str(row["pending_mode"])
            reason = _optional_text(row["pending_reason"])
            connection.execute(
                "UPDATE retraining_version_policy SET revision = ?, mode = ?, "
                "pending_mode = NULL, pending_requested_at = NULL, "
                "pending_requested_by = NULL, pending_reason = NULL, "
                "updated_at = ?, updated_by = ? WHERE singleton = 1",
                (revision, mode, now, actor),
            )
            connection.execute(
                "INSERT INTO retraining_version_policy_audit "
                "(revision, action, active_mode, pending_mode, actor, reason, created_at) "
                "VALUES (?, 'activated', ?, NULL, ?, ?, ?)",
                (revision, mode, actor, reason, now),
            )
        return True

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
            created_at=_parse_timestamp(row["created_at"]),
            updated_at=_parse_timestamp(row["updated_at"]),
            submitted_at=_parse_timestamp(row["submitted_at"]),
            started_at=_parse_timestamp(row["started_at"]),
            ended_at=_parse_timestamp(row["ended_at"]),
            logical_model_version=_optional_text(row["logical_model_version"]),
            candidate_model_version=_optional_text(row["candidate_model_version"]),
            training_mode=_optional_text(row["training_mode"]),
            initialization_mode=_optional_text(row["initialization_mode"]),
            version_policy_mode=_optional_text(row["version_policy_mode"]),
            promotion_mode=str(row["promotion_mode"] or "quality_gate"),
            canonical_batch_digest=_optional_text(row["canonical_batch_digest"]),
            base_model_version=_optional_text(row["base_model_version"]),
            base_model_uri=_optional_text(row["base_model_uri"]),
            feedback_manifest_uri=_optional_text(row["feedback_manifest_uri"]),
            feedback_events_uri=_optional_text(row["feedback_events_uri"]),
            replay_manifest_uri=_optional_text(row["replay_manifest_uri"]),
            replay_events_uri=_optional_text(row["replay_events_uri"]),
        )

    @staticmethod
    def _to_policy(row: sqlite3.Row) -> VersionPolicyState:
        return VersionPolicyState(
            revision=int(row["revision"]),
            mode=str(row["mode"]),
            pending_mode=_optional_text(row["pending_mode"]),
            pending_requested_at=_parse_timestamp(row["pending_requested_at"]),
            pending_requested_by=_optional_text(row["pending_requested_by"]),
            pending_reason=_optional_text(row["pending_reason"]),
            updated_at=_parse_timestamp(row["updated_at"]) or datetime.now(UTC),
            updated_by=str(row["updated_by"]),
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


def _optional_text(value: object) -> str | None:
    text = str(value or "").strip()
    return text or None
