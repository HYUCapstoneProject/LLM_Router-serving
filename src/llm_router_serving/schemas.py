from __future__ import annotations

from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class MessageRequest(StrictModel):
    message: str = Field(min_length=1)

    @field_validator("message")
    @classmethod
    def reject_blank_message(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("message must not be blank")
        return value


class EmbedResponse(StrictModel):
    embedding: list[float]
    dim: int
    modelVersion: str


class Tier(StrEnum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class RoutedModel(StrEnum):
    HAIKU = "haiku"
    SONNET = "sonnet"
    OPUS = "opus"


class ClassifyResponse(StrictModel):
    tier: Tier
    model: RoutedModel
    confidence: float = Field(ge=0.0, le=1.0)
    modelVersion: str


class ModelLoadStatus(StrEnum):
    LOADED = "loaded"
    LOADING = "loading"
    ERROR = "error"


class HealthStatus(StrEnum):
    OK = "ok"
    NOT_READY = "not_ready"


class HealthResponse(StrictModel):
    status: HealthStatus
    embedModel: ModelLoadStatus
    routerModel: ModelLoadStatus


class EscalateRequest(StrictModel):
    batchId: str = Field(min_length=1, max_length=200, pattern=r"^[A-Za-z0-9._-]+$")
    schemaVersion: str = Field(min_length=1, max_length=100)
    manifestUri: str = Field(pattern=r"^gs://[^/]+/.+$")
    eventsUri: str = Field(pattern=r"^gs://[^/]+/.+$")
    eventCount: Literal[1000]
    requestedAt: datetime

    @field_validator("requestedAt")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("requestedAt must include a UTC offset")
        if value.utcoffset() != timedelta(0):
            raise ValueError("requestedAt must be UTC")
        return value


class EscalateResponse(StrictModel):
    status: str = "accepted"
    batchId: str
    pipelineRunId: str


class ErrorResponse(StrictModel):
    error: str
    code: str
    requestId: str


class FeedbackManifest(BaseModel):
    """Mirror of the manifest contract the retraining pipeline enforces.

    The field set and the checks below track ``REQUIRED_MANIFEST_FIELDS`` and
    ``validate_feedback_manifest`` in the ``LLM_Router`` repository
    (``ml/src/llm_router_labeling/training/feedback_retraining_preprocessing.py``).
    Anything the pipeline would reject must be rejected here instead, while the
    caller is still holding the request, rather than hours later inside a run.
    """

    model_config = ConfigDict(extra="allow")

    schema_version: str = Field(min_length=1)
    batch_id: str = Field(min_length=1)
    created_at: datetime
    window_start: datetime
    window_end: datetime
    event_count: int = Field(gt=0)
    data_format: str = Field(min_length=1)
    data_uri: str = Field(pattern=r"^gs://[^/]+/.+$")
    # The ML validator strips an optional "sha256:" prefix, so accept both forms
    # and expose the bare digest through `sha256_digest`.
    sha256: str = Field(pattern=r"^(?:sha256:)?[a-fA-F0-9]{64}$")
    producer_version: str = Field(min_length=1)
    watermark: str | int

    @field_validator("watermark")
    @classmethod
    def reject_blank_watermark(cls, value: str | int) -> str | int:
        if isinstance(value, str) and not value.strip():
            raise ValueError("watermark must not be blank")
        return value

    @field_validator("created_at", "window_start", "window_end")
    @classmethod
    def assume_utc(cls, value: datetime) -> datetime:
        # The pipeline reads these with pd.to_datetime(utc=True), which treats a
        # naive timestamp as UTC. Match that instead of rejecting the batch.
        return value.replace(tzinfo=UTC) if value.tzinfo is None else value

    @model_validator(mode="after")
    def require_ordered_window(self) -> FeedbackManifest:
        if self.window_start > self.window_end:
            raise ValueError("window_start must not exceed window_end")
        return self

    @property
    def sha256_digest(self) -> str:
        return self.sha256.removeprefix("sha256:").lower()
