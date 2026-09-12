from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC
from io import BytesIO
from typing import Protocol
from urllib.parse import urlparse

from llm_router_serving.schemas import EscalateRequest

# Canonical Parquet columns consumed by feedback_retraining_preprocessing.py in
# the sibling LLM_Router repository.
REQUIRED_EVENT_COLUMNS = frozenset(
    {
        "event_id",
        "request_id",
        "question",
        "initial_label",
        "initial_model",
        "final_label",
        "final_model",
        "user_satisfied",
        "escalation_count",
        "created_at",
        "router_model_version",
        "pii_redacted",
    }
)

PIPELINE_FEEDBACK_SCHEMA_VERSION = "feedback-export-v0.1"


class BatchValidationError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class ObjectAccessError(RuntimeError):
    def __init__(self, code: str, message: str, *, status_code: int) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


class ObjectStore(Protocol):
    def read_bytes(self, uri: str) -> bytes: ...

    def write_bytes(self, uri: str, data: bytes, *, content_type: str) -> None: ...

    def delete_uri(self, uri: str) -> None: ...


def split_gcs_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "gs" or not parsed.netloc or not parsed.path.lstrip("/"):
        raise BatchValidationError("MANIFEST_MISMATCH", f"Invalid GCS URI: {uri}")
    return parsed.netloc, parsed.path.lstrip("/")


class GcsObjectStore:
    def __init__(self, *, project: str | None = None) -> None:
        self._project = project
        self._client = None

    def _get_client(self):
        if self._client is None:
            from google.cloud import storage

            self._client = storage.Client(project=self._project)
        return self._client

    def read_bytes(self, uri: str) -> bytes:
        from google.api_core.exceptions import (
            Forbidden,
            GoogleAPICallError,
            NotFound,
            Unauthorized,
        )

        bucket_name, object_name = split_gcs_uri(uri)
        blob = self._get_client().bucket(bucket_name).blob(object_name)
        try:
            return blob.download_as_bytes(checksum="auto")
        except NotFound as exc:
            raise ObjectAccessError(
                "GCS_OBJECT_NOT_FOUND",
                f"GCS object does not exist: {uri}",
                status_code=422,
            ) from exc
        except Unauthorized as exc:
            raise ObjectAccessError(
                "UNAUTHORIZED",
                f"GCS authentication failed while reading: {uri}",
                status_code=401,
            ) from exc
        except Forbidden as exc:
            raise ObjectAccessError(
                "FORBIDDEN",
                f"GCS object access was denied: {uri}",
                status_code=403,
            ) from exc
        except GoogleAPICallError as exc:
            raise ObjectAccessError(
                "GCS_BACKEND_UNAVAILABLE",
                f"GCS request failed while reading: {uri}",
                status_code=503,
            ) from exc

    def write_bytes(self, uri: str, data: bytes, *, content_type: str) -> None:
        from google.api_core.exceptions import (
            Forbidden,
            GoogleAPICallError,
            PreconditionFailed,
            Unauthorized,
        )

        bucket_name, object_name = split_gcs_uri(uri)
        blob = self._get_client().bucket(bucket_name).blob(object_name)
        try:
            blob.upload_from_string(
                data,
                content_type=content_type,
                checksum="auto",
                if_generation_match=0,
            )
        except PreconditionFailed:
            # Derived artifacts are content-addressed. A retry may encounter the
            # object written by its first attempt; accept it only if identical.
            if self.read_bytes(uri) != data:
                raise ObjectAccessError(
                    "GCS_OBJECT_CONFLICT",
                    f"A different object already exists at immutable URI: {uri}",
                    status_code=409,
                ) from None
        except Unauthorized as exc:
            raise ObjectAccessError(
                "UNAUTHORIZED",
                f"GCS authentication failed while writing: {uri}",
                status_code=401,
            ) from exc
        except Forbidden as exc:
            raise ObjectAccessError(
                "FORBIDDEN",
                f"GCS object write was denied: {uri}",
                status_code=403,
            ) from exc
        except GoogleAPICallError as exc:
            raise ObjectAccessError(
                "GCS_BACKEND_UNAVAILABLE",
                f"GCS request failed while writing: {uri}",
                status_code=503,
            ) from exc

    def delete_uri(self, uri: str) -> None:
        from google.api_core.exceptions import Forbidden, GoogleAPICallError, NotFound

        bucket_name, object_name = split_gcs_uri(uri)
        try:
            self._get_client().bucket(bucket_name).blob(object_name).delete()
        except NotFound:
            return
        except Forbidden as exc:
            raise ObjectAccessError(
                "FORBIDDEN",
                f"GCS object delete was denied: {uri}",
                status_code=403,
            ) from exc
        except GoogleAPICallError as exc:
            raise ObjectAccessError(
                "GCS_BACKEND_UNAVAILABLE",
                f"GCS request failed while deleting: {uri}",
                status_code=503,
            ) from exc


@dataclass(frozen=True)
class ValidatedFeedbackBatch:
    events_bytes: bytes
    events_sha256: str
    row_count: int


@dataclass(frozen=True)
class MaterializedFeedback:
    manifest_uri: str
    events_uri: str
    event_count: int
    replay_manifest_uri: str
    replay_events_uri: str
    replay_event_count: int


class FeedbackReplayMaterializer:
    """Build one bounded, immutable feedback input for each training call.

    The current three-call cycle remains complete. Older cycles are represented
    by a deterministic, label-stratified replay snapshot, so history never grows
    without bound and Spring does not need to resend or curate earlier batches.
    """

    def __init__(
        self,
        *,
        object_store: ObjectStore,
        root_uri: str,
        rows_per_label: int,
    ) -> None:
        if rows_per_label < 1:
            raise ValueError("rows_per_label must be positive")
        split_gcs_uri(root_uri)
        self._object_store = object_store
        self._root_uri = root_uri.rstrip("/")
        self._rows_per_label = rows_per_label

    def materialize(
        self,
        *,
        request: EscalateRequest,
        validated: ValidatedFeedbackBatch,
        artifact_key: str,
        logical_model_version: str,
        previous_cycle_events_uri: str | None,
        previous_replay_events_uri: str | None,
    ) -> MaterializedFeedback:
        import pyarrow as pa
        import pyarrow.parquet as parquet

        incoming = parquet.read_table(BytesIO(validated.events_bytes))
        cycle_inputs = []
        if previous_cycle_events_uri:
            cycle_inputs.append(
                parquet.read_table(
                    BytesIO(self._object_store.read_bytes(previous_cycle_events_uri))
                )
            )
        cycle_inputs.append(incoming)
        cycle = _deduplicate_events(pa.concat_tables(cycle_inputs, promote_options="default"))

        replay_previous = None
        if previous_replay_events_uri:
            replay_previous = parquet.read_table(
                BytesIO(self._object_store.read_bytes(previous_replay_events_uri))
            )
        training_inputs = [cycle]
        if replay_previous is not None:
            training_inputs.append(replay_previous)
        training_feedback = _deduplicate_events(
            pa.concat_tables(training_inputs, promote_options="default")
        )

        replay_inputs = [incoming]
        if replay_previous is not None:
            replay_inputs.insert(0, replay_previous)
        replay = _compact_replay(
            _deduplicate_events(pa.concat_tables(replay_inputs, promote_options="default")),
            rows_per_label=self._rows_per_label,
        )

        prefix = f"{self._root_uri}/{artifact_key}"
        feedback_events_uri = f"{prefix}/feedback/events.parquet"
        feedback_manifest_uri = f"{prefix}/feedback/manifest.json"
        replay_events_uri = f"{prefix}/replay/events.parquet"
        replay_manifest_uri = f"{prefix}/replay/manifest.json"
        feedback_bytes = _parquet_bytes(training_feedback)
        replay_bytes = _parquet_bytes(replay)
        created_at = (
            max(event.createdAt for event in request.events)
            .astimezone(UTC)
            .isoformat()
            .replace("+00:00", "Z")
        )
        self._write_artifact(
            events_uri=feedback_events_uri,
            manifest_uri=feedback_manifest_uri,
            events_table=training_feedback,
            events_bytes=feedback_bytes,
            event_count=training_feedback.num_rows,
            batch_id=f"continual-{logical_model_version}-{artifact_key[-12:]}",
            created_at=created_at,
            watermark=request.batchId,
        )
        self._write_artifact(
            events_uri=replay_events_uri,
            manifest_uri=replay_manifest_uri,
            events_table=replay,
            events_bytes=replay_bytes,
            event_count=replay.num_rows,
            batch_id=f"replay-{logical_model_version}-{artifact_key[-12:]}",
            created_at=created_at,
            watermark=request.batchId,
        )
        return MaterializedFeedback(
            manifest_uri=feedback_manifest_uri,
            events_uri=feedback_events_uri,
            event_count=training_feedback.num_rows,
            replay_manifest_uri=replay_manifest_uri,
            replay_events_uri=replay_events_uri,
            replay_event_count=replay.num_rows,
        )

    def _write_artifact(
        self,
        *,
        events_uri: str,
        manifest_uri: str,
        events_table,
        events_bytes: bytes,
        event_count: int,
        batch_id: str,
        created_at: str,
        watermark: str,
    ) -> None:
        window_start, window_end = _event_window(events_table, fallback=created_at)
        manifest = {
            "schema_version": PIPELINE_FEEDBACK_SCHEMA_VERSION,
            "batch_id": batch_id,
            "created_at": created_at,
            "window_start": window_start,
            "window_end": window_end,
            "event_count": event_count,
            "data_format": "parquet",
            "data_uri": events_uri,
            "sha256": hashlib.sha256(events_bytes).hexdigest(),
            "producer_version": "llm-router-serving-inline-v1",
            "watermark": watermark,
        }
        manifest_bytes = (
            json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
        ).encode()
        self._object_store.write_bytes(
            events_uri, events_bytes, content_type="application/vnd.apache.parquet"
        )
        self._object_store.write_bytes(
            manifest_uri, manifest_bytes, content_type="application/json"
        )

    def retire_superseded(
        self,
        *,
        superseded_uris: list[str],
        protected_uris: set[str],
    ) -> list[str]:
        warnings: list[str] = []
        for uri in sorted(set(superseded_uris) - protected_uris):
            if not uri.startswith(f"{self._root_uri}/"):
                continue
            try:
                self._object_store.delete_uri(uri)
            except ObjectAccessError as exc:
                warnings.append(f"{uri}: {exc.message}")
        return warnings


def _deduplicate_events(table):
    import pyarrow as pa

    seen: set[str] = set()
    keep: list[int] = []
    for index, value in enumerate(table.column("event_id").to_pylist()):
        event_id = str(value or "")
        if event_id not in seen:
            seen.add(event_id)
            keep.append(index)
    return table.take(pa.array(keep, type=pa.int64()))


def _compact_replay(table, *, rows_per_label: int):
    import pyarrow as pa

    selected: list[int] = []
    buckets: dict[str, list[tuple[str, int]]] = {}
    event_ids = table.column("event_id").to_pylist()
    initial_labels = table.column("initial_label").to_pylist()
    labels = table.column("final_label").to_pylist()
    satisfied = table.column("user_satisfied").to_pylist()
    pii_redacted = table.column("pii_redacted").to_pylist()
    label_rank = {"easy": 0, "medium": 1, "hard": 2}
    values = zip(event_ids, initial_labels, labels, satisfied, pii_redacted, strict=True)
    for index, (event_id, initial_label, label, is_satisfied, is_redacted) in enumerate(values):
        initial = str(initial_label).casefold()
        final = str(label).casefold()
        if (
            initial not in label_rank
            or final not in label_rank
            or label_rank[final] <= label_rank[initial]
            or is_satisfied is not True
            or is_redacted is not True
        ):
            continue
        rank = hashlib.sha256(str(event_id).encode()).hexdigest()
        buckets.setdefault(final, []).append((rank, index))
    for label in sorted(buckets):
        selected.extend(index for _, index in sorted(buckets[label])[:rows_per_label])
    return table.take(pa.array(sorted(selected), type=pa.int64()))


def _parquet_bytes(table) -> bytes:
    import pyarrow.parquet as parquet

    output = BytesIO()
    parquet.write_table(table, output)
    return output.getvalue()


def _inline_events_table(request: EscalateRequest):
    import pyarrow as pa

    # Event order is not part of batch identity. Canonicalizing here keeps the
    # immutable Parquet bytes stable when a retry sends the same events in a
    # different array order.
    events = sorted(request.events, key=lambda event: event.eventId)
    return pa.table(
        {
            "event_id": pa.array([event.eventId for event in events], type=pa.string()),
            "request_id": pa.array([event.requestId for event in events], type=pa.string()),
            "question": pa.array([event.question for event in events], type=pa.string()),
            "initial_label": pa.array(
                [event.initialLabel.value for event in events], type=pa.string()
            ),
            "initial_model": pa.array([event.initialModel for event in events], type=pa.string()),
            "final_label": pa.array([event.finalLabel.value for event in events], type=pa.string()),
            "final_model": pa.array([event.finalModel for event in events], type=pa.string()),
            "user_satisfied": pa.array([event.userSatisfied for event in events], type=pa.bool_()),
            "escalation_count": pa.array(
                [event.escalationCount for event in events], type=pa.int32()
            ),
            "created_at": pa.array(
                [event.createdAt.astimezone(UTC) for event in events],
                type=pa.timestamp("ms", tz="UTC"),
            ),
            "router_model_version": pa.array(
                [event.routerModelVersion for event in events], type=pa.string()
            ),
            "pii_redacted": pa.array([event.piiRedacted for event in events], type=pa.bool_()),
        }
    )


def _event_window(table, *, fallback: str) -> tuple[str, str]:
    values = table.column("created_at").to_pylist()
    if not values:
        return fallback, fallback

    def isoformat(value) -> str:
        return value.astimezone(UTC).isoformat().replace("+00:00", "Z")

    return isoformat(min(values)), isoformat(max(values))


class FeedbackBatchValidator:
    def __init__(
        self,
        *,
        supported_schemas: tuple[str, ...],
    ) -> None:
        self._supported_schemas = frozenset(supported_schemas)

    def validate(self, request: EscalateRequest) -> ValidatedFeedbackBatch:
        if request.schemaVersion not in self._supported_schemas:
            raise BatchValidationError(
                "UNSUPPORTED_SCHEMA_VERSION",
                f"Unsupported feedback schema: {request.schemaVersion}",
            )
        table = _inline_events_table(request)
        if frozenset(table.schema.names) != REQUIRED_EVENT_COLUMNS:
            raise RuntimeError("Inline feedback converter does not match the ML event contract")
        events_bytes = _parquet_bytes(table)
        actual_sha256 = hashlib.sha256(events_bytes).hexdigest()
        return ValidatedFeedbackBatch(
            events_bytes=events_bytes,
            events_sha256=actual_sha256,
            row_count=len(request.events),
        )
