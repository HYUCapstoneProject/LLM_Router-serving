from __future__ import annotations

from datetime import datetime, timedelta
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, StrictBool, field_validator, model_validator


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


class RerankRequest(StrictModel):
    """Gateway sends one query with the pgvector candidates it already retrieved.

    The gateway holds the cache and decides hit or miss. This endpoint only
    scores pairs; it never sees the threshold and never returns a decision.
    """

    query: str = Field(min_length=1)
    candidates: list[str] = Field(min_length=1, max_length=32)

    @field_validator("query")
    @classmethod
    def reject_blank_query(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("query must not be blank")
        return value

    @field_validator("candidates")
    @classmethod
    def reject_blank_candidates(cls, value: list[str]) -> list[str]:
        if any(not candidate.strip() for candidate in value):
            raise ValueError("candidates must not contain blank entries")
        return value


class RerankResponse(StrictModel):
    """`scores` is positionally aligned with the request `candidates`.

    Score ranges are model specific: a sigmoid model lands in [0, 1] while a raw
    logit model does not. The caller must tie its threshold to `modelVersion`
    and re-measure whenever that value changes. Never compare scores across models.
    """

    scores: list[float]
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


class ModelLoadStatusOrDisabled(StrEnum):
    LOADED = "loaded"
    LOADING = "loading"
    ERROR = "error"
    DISABLED = "disabled"


class HealthResponse(StrictModel):
    status: HealthStatus
    embedModel: ModelLoadStatus
    routerModel: ModelLoadStatus
    # Readiness deliberately ignores this field. The reranker is an optional
    # precision layer; if it is missing the gateway falls back to the cosine
    # threshold it already uses, and taking the whole service down for that
    # would trade a small quality loss for a total outage.
    rerankModel: ModelLoadStatusOrDisabled = ModelLoadStatusOrDisabled.DISABLED


class EscalationEvent(StrictModel):
    eventId: str = Field(min_length=1, max_length=200)
    requestId: str = Field(min_length=1, max_length=200)
    question: str = Field(min_length=1, max_length=100_000)
    initialLabel: Tier
    initialModel: str = Field(min_length=1, max_length=300)
    finalLabel: Tier
    finalModel: str = Field(min_length=1, max_length=300)
    userSatisfied: StrictBool
    escalationCount: int = Field(ge=1)
    createdAt: datetime
    routerModelVersion: str = Field(min_length=1, max_length=300)
    piiRedacted: StrictBool

    @field_validator(
        "eventId",
        "requestId",
        "question",
        "initialModel",
        "finalModel",
        "routerModelVersion",
    )
    @classmethod
    def reject_blank_strings(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("value must not be blank")
        return value

    @field_validator("createdAt")
    @classmethod
    def require_created_at_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("createdAt must include a UTC offset")
        if value.utcoffset() != timedelta(0):
            raise ValueError("createdAt must be UTC")
        return value

    @model_validator(mode="after")
    def require_usable_escalation(self) -> EscalationEvent:
        rank = {Tier.EASY: 0, Tier.MEDIUM: 1, Tier.HARD: 2}
        if rank[self.finalLabel] <= rank[self.initialLabel]:
            raise ValueError("finalLabel must be a stricter tier than initialLabel")
        if self.userSatisfied is not True:
            raise ValueError("userSatisfied must be true")
        if self.piiRedacted is not True:
            raise ValueError("piiRedacted must be true")
        return self


class EscalateRequest(StrictModel):
    batchId: str = Field(min_length=1, max_length=200, pattern=r"^[A-Za-z0-9._-]+$")
    schemaVersion: str = Field(min_length=1, max_length=100)
    requestedAt: datetime
    events: list[EscalationEvent] = Field(min_length=1)

    @field_validator("requestedAt")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("requestedAt must include a UTC offset")
        if value.utcoffset() != timedelta(0):
            raise ValueError("requestedAt must be UTC")
        return value

    @model_validator(mode="after")
    def require_unique_event_ids(self) -> EscalateRequest:
        event_ids = [event.eventId for event in self.events]
        if len(set(event_ids)) != len(event_ids):
            raise ValueError("events must have unique eventId values")
        return self


class TrainingMode(StrEnum):
    MINOR_FINE_TUNE = "minor_fine_tune"
    MAJOR_CONSOLIDATION = "major_consolidation"
    PRISTINE_REBASE = "pristine_rebase"


class InitializationMode(StrEnum):
    APPROVED_ANCHOR = "approved_anchor"
    PRISTINE_MODERNBERT = "pristine_modernbert"


class VersionPolicyMode(StrEnum):
    SUCCESSFUL_HTTP_REQUEST = "successfulHttpRequest"
    VALIDATED_UNIQUE_BATCH = "validatedUniqueBatch"


class EscalateResponse(StrictModel):
    schemaVersion: str = "escalation-acceptance-v1"
    status: str = "accepted"
    batchId: str
    pipelineRunId: str
    logicalModelVersion: str
    candidateModelVersion: str
    trainingMode: TrainingMode
    versionPolicyMode: VersionPolicyMode
    statusUrl: str


class RetrainingOverallStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    CANCELLING = "cancelling"
    PAUSED = "paused"
    SUCCEEDED = "succeeded"
    REJECTED = "rejected"
    FAILED = "failed"
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"


class RetrainingStageName(StrEnum):
    INITIALIZING = "initializing"
    FEEDBACK_PREPARATION = "feedbackPreparation"
    FEEDBACK_MERGE = "feedbackMerge"
    TRAINING = "training"
    EVALUATION = "evaluation"
    PACKAGING = "packaging"
    GATE = "gate"
    PUBLISHING = "publishing"
    COMPLETE = "complete"


class StageExecutionStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    SKIPPED = "skipped"
    UNKNOWN = "unknown"


class TrainingVersion(StrictModel):
    logicalModelVersion: str
    candidateModelVersion: str
    trainingMode: TrainingMode
    initializationMode: InitializationMode
    versionPolicyMode: VersionPolicyMode


class StageProgress(StrictModel):
    current: int = Field(ge=0)
    total: int = Field(ge=1)
    unit: str = "optimizerStep"
    percent: float = Field(ge=0, le=100)
    epoch: int = Field(ge=1)
    totalEpochs: int = Field(ge=1)
    heartbeatAt: datetime


class StageRecord(StrictModel):
    name: RetrainingStageName
    status: StageExecutionStatus
    startedAt: datetime | None
    endedAt: datetime | None


class FilteringMetrics(StrictModel):
    inputEventCount: int = Field(ge=1)
    acceptedEventCount: int = Field(ge=0)
    excludedEventCount: int = Field(ge=0)
    collapsedDuplicateEventCount: int = Field(ge=0)
    canonicalTrainingRowCount: int = Field(ge=0)
    exclusionDistribution: dict[str, int]
    labelDistribution: dict[Tier, int]


class MergeMetrics(StrictModel):
    feedbackInputRowCount: int = Field(ge=0)
    acceptedFeedbackRowCount: int = Field(ge=0)
    excludedFeedbackRowCount: int = Field(ge=0)
    replacedBaseRowCount: int = Field(ge=0)
    mergedTrainingRowCount: int = Field(ge=0)
    decisionDistribution: dict[str, int]


class RetrainingOutcome(StrEnum):
    PROMOTED = "promoted"
    REJECTED = "rejected"
    FAILED = "failed"
    CANCELLED = "cancelled"


class FailureCode(StrEnum):
    FEEDBACK_VALIDATION_FAILED = "FEEDBACK_VALIDATION_FAILED"
    NO_UNIQUE_FEEDBACK = "NO_UNIQUE_FEEDBACK"
    TRAINING_FAILED = "TRAINING_FAILED"
    EVALUATION_FAILED = "EVALUATION_FAILED"
    DEPLOY_GATE_REJECTED = "DEPLOY_GATE_REJECTED"
    PACKAGE_FAILED = "PACKAGE_FAILED"
    PUBLISH_FAILED = "PUBLISH_FAILED"
    PIPELINE_CANCELLED = "PIPELINE_CANCELLED"
    UNKNOWN_PIPELINE_FAILURE = "UNKNOWN_PIPELINE_FAILURE"


class RetrainingResult(StrictModel):
    outcome: RetrainingOutcome
    candidateModelVersion: str
    logicalModelVersion: str
    gatePassed: bool | None
    failedGateChecks: list[str]
    failureCode: FailureCode | None
    failureStage: RetrainingStageName | None
    retryable: bool
    message: str | None
    approvedAt: datetime | None


class ActivationStatusValue(StrEnum):
    PENDING = "pending"
    ACTIVE = "active"
    LOAD_FAILED = "loadFailed"
    SUPERSEDED = "superseded"
    NOT_APPLICABLE = "notApplicable"


class ActivationStatus(StrictModel):
    status: ActivationStatusValue
    desiredModelVersion: str | None
    servingModelVersion: str | None
    checkedAt: datetime


class RetrainingStatusResponse(StrictModel):
    schemaVersion: str = "retraining-status-v1"
    batchId: str
    pipelineRunId: str
    status: RetrainingOverallStatus
    terminal: bool
    stage: RetrainingStageName
    version: TrainingVersion
    stageProgress: StageProgress | None
    stages: list[StageRecord]
    filtering: FilteringMetrics | None
    merge: MergeMetrics | None
    result: RetrainingResult | None
    activation: ActivationStatus
    submittedAt: datetime
    startedAt: datetime | None
    endedAt: datetime | None
    lastUpdatedAt: datetime
    stale: bool
    warnings: list[str]


class ModelSource(StrEnum):
    CURRENT_POINTER = "currentPointer"
    FIXED_PACKAGE = "fixedPackage"
    LOCAL_PATH = "localPath"


class DesiredModel(StrictModel):
    modelVersion: str
    logicalModelVersion: str | None
    trainingMode: TrainingMode | None
    manifestSha256: str
    approvedAt: datetime | None


class ServingModel(StrictModel):
    modelVersion: str
    logicalModelVersion: str | None
    loadedAt: datetime


class PreviousKnownGood(StrictModel):
    modelVersion: str


class CurrentModelStatus(StrEnum):
    IN_SYNC = "inSync"
    LOADING = "loading"
    LOAD_FAILED = "loadFailed"
    STATIC = "static"
    UNAVAILABLE = "unavailable"


class CurrentModelResponse(StrictModel):
    schemaVersion: str = "router-model-status-v1"
    status: CurrentModelStatus
    source: ModelSource
    desired: DesiredModel | None
    serving: ServingModel | None
    previousKnownGood: PreviousKnownGood | None
    lastPointerCheckAt: datetime | None
    lastSuccessfulSwapAt: datetime | None
    lastLoadError: str | None


class PendingPolicyChange(StrictModel):
    mode: VersionPolicyMode
    activation: str = "nextMajorBoundary"
    requestedAt: datetime
    requestedBy: str
    reason: str


class VersionPolicyResponse(StrictModel):
    schemaVersion: str = "retraining-version-policy-v1"
    mode: VersionPolicyMode
    callsPerMajor: int
    rebaseEveryMajorCycles: int
    committedSuccessfulRuns: int
    cyclePosition: int
    currentLogicalModelVersion: str
    nextLogicalModelVersion: str
    revision: int
    pendingChange: PendingPolicyChange | None
    updatedAt: datetime
    updatedBy: str


class VersionPolicyUpdateRequest(StrictModel):
    mode: VersionPolicyMode
    activation: str
    expectedRevision: int = Field(ge=0)
    reason: str = Field(min_length=1, max_length=500)

    @field_validator("activation")
    @classmethod
    def require_next_major_boundary(cls, value: str) -> str:
        if value != "nextMajorBoundary":
            raise ValueError("activation must be nextMajorBoundary")
        return value


class ErrorResponse(StrictModel):
    error: str
    code: str
    requestId: str
