from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from io import BytesIO
from typing import Protocol
from urllib.parse import urlparse

from pydantic import ValidationError

from llm_router_serving.schemas import EscalateRequest, FeedbackManifest

# Mirrors REQUIRED_EVENT_COLUMNS in the LLM_Router repository
# (ml/src/llm_router_labeling/training/feedback_retraining_preprocessing.py).
# The pipeline's prepare step fails on a missing column, so check it up front.
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

# The pipeline accepts parquet or jsonl; this service only reads the Parquet form
# the API contract specifies, so an unreadable format is rejected explicitly
# rather than surfacing as a confusing parse failure.
SERVING_DATA_FORMATS = frozenset({"parquet"})


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


@dataclass(frozen=True)
class ValidatedFeedbackBatch:
    manifest: FeedbackManifest
    events_sha256: str
    row_count: int


class FeedbackBatchValidator:
    def __init__(
        self,
        *,
        object_store: ObjectStore,
        allowed_buckets: tuple[str, ...],
        supported_schemas: tuple[str, ...],
    ) -> None:
        self._object_store = object_store
        self._allowed_buckets = frozenset(allowed_buckets)
        self._supported_schemas = frozenset(supported_schemas)

    def validate(self, request: EscalateRequest) -> ValidatedFeedbackBatch:
        if request.schemaVersion not in self._supported_schemas:
            raise BatchValidationError(
                "UNSUPPORTED_SCHEMA_VERSION",
                f"Unsupported feedback schema: {request.schemaVersion}",
            )
        self._validate_bucket(request.manifestUri)
        self._validate_bucket(request.eventsUri)

        manifest_bytes = self._object_store.read_bytes(request.manifestUri)
        try:
            manifest_payload = json.loads(manifest_bytes)
            manifest = FeedbackManifest.model_validate(manifest_payload)
        except (json.JSONDecodeError, UnicodeDecodeError, ValidationError, TypeError) as exc:
            raise BatchValidationError(
                "MANIFEST_MISMATCH", "Feedback manifest is not valid JSON or schema"
            ) from exc

        mismatches: list[str] = []
        if manifest.schema_version != request.schemaVersion:
            mismatches.append("schema_version")
        if manifest.batch_id != request.batchId:
            mismatches.append("batch_id")
        if manifest.event_count != request.eventCount:
            mismatches.append("event_count")
        if manifest.data_uri != request.eventsUri:
            mismatches.append("data_uri")
        if mismatches:
            raise BatchValidationError(
                "MANIFEST_MISMATCH",
                "Request and manifest differ: " + ", ".join(mismatches),
            )
        if manifest.data_format not in SERVING_DATA_FORMATS:
            raise BatchValidationError(
                "UNSUPPORTED_DATA_FORMAT",
                f"Feedback data_format must be parquet, got {manifest.data_format!r}",
            )

        events_bytes = self._object_store.read_bytes(request.eventsUri)
        actual_sha256 = hashlib.sha256(events_bytes).hexdigest()
        if actual_sha256 != manifest.sha256_digest:
            raise BatchValidationError(
                "CHECKSUM_MISMATCH", "Feedback events SHA-256 does not match the manifest"
            )

        try:
            import pyarrow.parquet as parquet

            metadata = parquet.read_metadata(BytesIO(events_bytes))
        except Exception as exc:
            raise BatchValidationError(
                "MANIFEST_MISMATCH", "Feedback events object is not a readable Parquet file"
            ) from exc

        missing_columns = sorted(REQUIRED_EVENT_COLUMNS - set(metadata.schema.names))
        if missing_columns:
            raise BatchValidationError(
                "EVENT_SCHEMA_MISMATCH",
                "Feedback events are missing columns: " + ", ".join(missing_columns),
            )

        row_count = metadata.num_rows
        if row_count != request.eventCount:
            raise BatchValidationError(
                "EVENT_COUNT_MISMATCH",
                f"Feedback Parquet contains {row_count} rows; expected {request.eventCount}",
            )
        return ValidatedFeedbackBatch(
            manifest=manifest,
            events_sha256=actual_sha256,
            row_count=row_count,
        )

    def _validate_bucket(self, uri: str) -> None:
        bucket, _ = split_gcs_uri(uri)
        if self._allowed_buckets and bucket not in self._allowed_buckets:
            raise ObjectAccessError(
                "FORBIDDEN",
                f"GCS bucket is not allowed: {bucket}",
                status_code=403,
            )
