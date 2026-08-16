from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from datetime import UTC, datetime, timedelta
from io import BytesIO

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from llm_router_serving.config import Settings
from llm_router_serving.retraining.gcs import (
    BatchValidationError,
    FeedbackBatchValidator,
)
from llm_router_serving.retraining.pipeline import (
    PipelineAuthorizationError,
    PipelineState,
    PipelineSubmission,
    VertexPipelineClient,
    vertex_job_id,
)
from llm_router_serving.retraining.service import EscalationService
from llm_router_serving.retraining.store import (
    BatchIdConflictError,
    RetrainingInProgressError,
    RetrainingJobStore,
)
from llm_router_serving.schemas import EscalateRequest


class MemoryObjectStore:
    def __init__(self, objects: dict[str, bytes]) -> None:
        self.objects = objects

    def read_bytes(self, uri: str) -> bytes:
        return self.objects[uri]


class FakePipelineClient:
    def __init__(self) -> None:
        self.submissions: list[tuple[str, dict[str, str]]] = []
        self.states: dict[str, PipelineState] = {}
        self.runs_by_name: dict[str, str] = {}

    def submit(self, *, run_name: str, arguments: dict[str, str]) -> PipelineSubmission:
        self.submissions.append((run_name, arguments))
        run_id = f"pipeline-{len(self.submissions)}"
        self.states[run_id] = PipelineState.ACTIVE
        self.runs_by_name[run_name] = run_id
        return PipelineSubmission(run_id=run_id)

    def state(self, run_id: str) -> PipelineState:
        return self.states[run_id]

    def find_run(self, run_name: str) -> str | None:
        return self.runs_by_name.get(run_name)


def parquet_bytes(row_count: int = 1000, *, drop_columns: tuple[str, ...] = ()) -> bytes:
    columns = {
        "event_id": [f"event-{index}" for index in range(row_count)],
        "request_id": [f"req-{index}" for index in range(row_count)],
        "question": ["question" for _ in range(row_count)],
        "initial_label": ["easy" for _ in range(row_count)],
        "initial_model": ["haiku" for _ in range(row_count)],
        "final_label": ["hard" for _ in range(row_count)],
        "final_model": ["opus" for _ in range(row_count)],
        "user_satisfied": [True for _ in range(row_count)],
        "escalation_count": [1 for _ in range(row_count)],
        "created_at": ["2026-08-12T15:00:00Z" for _ in range(row_count)],
        "router_model_version": ["router-x-20260731-21c2f1b" for _ in range(row_count)],
        "pii_redacted": [True for _ in range(row_count)],
    }
    for name in drop_columns:
        del columns[name]
    output = BytesIO()
    pq.write_table(pa.table(columns), output)
    return output.getvalue()


def build_request(batch_id: str = "feedback-20260812-0001") -> EscalateRequest:
    return EscalateRequest(
        batchId=batch_id,
        schemaVersion="feedback-export-v0.1",
        manifestUri=f"gs://llm-router/{batch_id}/manifest.json",
        eventsUri=f"gs://llm-router/{batch_id}/events.parquet",
        eventCount=1000,
        requestedAt=datetime(2026, 8, 12, 15, 30, tzinfo=UTC),
    )


def build_objects(
    request: EscalateRequest,
    *,
    event_count: int = 1000,
    drop_columns: tuple[str, ...] = (),
    **manifest_overrides: object,
) -> dict[str, bytes]:
    events = parquet_bytes(event_count, drop_columns=drop_columns)
    manifest = {
        "schema_version": request.schemaVersion,
        "batch_id": request.batchId,
        "created_at": "2026-08-12T15:29:00Z",
        "window_start": "2026-08-01T00:00:00Z",
        "window_end": "2026-08-12T15:00:00Z",
        "event_count": request.eventCount,
        "data_format": "parquet",
        "data_uri": request.eventsUri,
        "sha256": hashlib.sha256(events).hexdigest(),
        "producer_version": "spring-exporter-v1",
        "watermark": "event-1000",
    }
    for key, value in manifest_overrides.items():
        if value is None:
            manifest.pop(key, None)
        else:
            manifest[key] = value
    return {
        request.manifestUri: json.dumps(manifest).encode(),
        request.eventsUri: events,
    }


def build_validator(request: EscalateRequest, **object_kwargs) -> FeedbackBatchValidator:
    return FeedbackBatchValidator(
        object_store=MemoryObjectStore(build_objects(request, **object_kwargs)),
        allowed_buckets=("llm-router",),
        supported_schemas=("feedback-export-v0.1",),
    )


def _age_job(tmp_path, batch_id: str, *, seconds: int) -> None:
    """Backdate a lease's updated_at so it reads as stale."""
    stale = datetime.now(UTC) - timedelta(seconds=seconds)
    connection = sqlite3.connect(tmp_path / "retraining.sqlite3")
    with connection:
        connection.execute(
            "UPDATE retraining_jobs SET updated_at = ? WHERE batch_id = ?",
            (stale.strftime("%Y-%m-%d %H:%M:%S"), batch_id),
        )
    connection.close()


def build_service(tmp_path, request: EscalateRequest):
    pipeline = FakePipelineClient()
    settings = Settings(
        load_models_on_startup=False,
        retraining_db_path=tmp_path / "retraining.sqlite3",
        allowed_gcs_buckets=("llm-router",),
        base_training_dataset_uri="gs://llm-router/base/training.parquet",
        feedback_preprocessing_config_uri="gs://llm-router/config/feedback.json",
        training_config_uri="gs://llm-router/config/training.json",
        routerarena_dataset_uri="gs://llm-router/eval/routerarena.parquet",
        evaluation_config_uri="gs://llm-router/config/evaluation.json",
        package_metadata_uri="gs://llm-router/config/package.json",
        baseline_metrics_uri="gs://llm-router/eval/baseline.json",
        gate_rules_uri="gs://llm-router/config/gate.json",
        router_model_version="router-modernbert-base-routerbench-20260731-21c2f1b",
        model_code_commit="21c2f1b",
        vertex_project="test-project",
        vertex_location="us-west1",
        vertex_pipeline_template_uri="gs://llm-router/pipelines/v1/feedback-retraining.yaml",
        vertex_pipeline_root="gs://llm-router/pipeline-root",
        vertex_training_service_account="train@test-project.iam.gserviceaccount.com",
        vertex_training_output_directory="gs://llm-router/vertex-training-output",
        model_publish_root_uri="gs://llm-router/models/router-modernbert",
    )
    validator = FeedbackBatchValidator(
        object_store=MemoryObjectStore(build_objects(request)),
        allowed_buckets=settings.allowed_gcs_buckets,
        supported_schemas=settings.supported_feedback_schemas,
    )
    service = EscalationService(
        settings=settings,
        validator=validator,
        job_store=RetrainingJobStore(settings.retraining_db_path),
        pipeline_client=pipeline,
    )
    return service, pipeline


async def test_submit_is_idempotent_for_same_batch_and_payload(tmp_path):
    request = build_request()
    service, pipeline = build_service(tmp_path, request)

    first = await service.submit(request)
    second = await service.submit(request)

    assert first.pipeline_run_id == "pipeline-1"
    assert second.pipeline_run_id == first.pipeline_run_id
    assert len(pipeline.submissions) == 1
    arguments = pipeline.submissions[0][1]
    assert arguments["feedback_manifest_uri"] == request.manifestUri
    assert arguments["feedback_events_uri"] == request.eventsUri
    assert arguments["baseline_model_version"].startswith("router-modernbert")
    assert arguments["candidate_model_version"].startswith(
        "router-modernbert-base-feedback-20260812-"
    )
    assert arguments["candidate_model_version"].endswith("21c2f1b")


async def test_idempotency_survives_service_recreation(tmp_path):
    request = build_request()
    service, first_pipeline = build_service(tmp_path, request)
    first = await service.submit(request)

    recreated_service, second_pipeline = build_service(tmp_path, request)
    second = await recreated_service.submit(request)

    assert first.pipeline_run_id == second.pipeline_run_id
    assert len(first_pipeline.submissions) == 1
    assert second_pipeline.submissions == []


async def test_same_batch_with_different_payload_conflicts(tmp_path):
    request = build_request()
    service, _ = build_service(tmp_path, request)
    await service.submit(request)

    # Same batch ID, different data. The batch is valid on its own, so the
    # conflict has to come from the idempotency record rather than validation.
    changed = request.model_copy(
        update={
            "manifestUri": "gs://llm-router/reissued/manifest.json",
            "eventsUri": "gs://llm-router/reissued/events.parquet",
        }
    )
    service._validator = FeedbackBatchValidator(
        object_store=MemoryObjectStore({**build_objects(request), **build_objects(changed)}),
        allowed_buckets=("llm-router",),
        supported_schemas=("feedback-export-v0.1",),
    )

    with pytest.raises(BatchIdConflictError):
        await service.submit(changed)


async def test_retry_with_a_fresh_timestamp_is_still_idempotent(tmp_path):
    # A client retrying after a timeout usually re-stamps requestedAt. That must
    # return the original run, not a BATCH_ID_CONFLICT.
    request = build_request()
    service, pipeline = build_service(tmp_path, request)
    first = await service.submit(request)

    retried = request.model_copy(update={"requestedAt": datetime(2026, 8, 12, 16, tzinfo=UTC)})
    second = await service.submit(retried)

    assert second.pipeline_run_id == first.pipeline_run_id
    assert len(pipeline.submissions) == 1


async def test_only_one_non_terminal_pipeline_can_run(tmp_path):
    first_request = build_request("feedback-20260812-0001")
    service, pipeline = build_service(tmp_path, first_request)
    first = await service.submit(first_request)

    second_request = build_request("feedback-20260812-0002")
    service._validator = build_validator(second_request)
    with pytest.raises(RetrainingInProgressError):
        await service.submit(second_request)

    pipeline.states[first.pipeline_run_id] = PipelineState.TERMINAL
    accepted = await service.submit(second_request)
    assert accepted.pipeline_run_id == "pipeline-2"


def test_validator_rejects_parquet_row_count_mismatch():
    request = build_request()

    with pytest.raises(BatchValidationError) as error:
        build_validator(request, event_count=999).validate(request)

    assert error.value.code == "EVENT_COUNT_MISMATCH"


@pytest.mark.parametrize(
    "missing_field",
    ["created_at", "window_start", "window_end", "data_format"],
)
def test_validator_rejects_manifest_missing_a_pipeline_required_field(missing_field):
    # The pipeline's own validator requires these; catching them here turns a
    # mid-run failure into a 422 the caller can act on.
    request = build_request()

    with pytest.raises(BatchValidationError) as error:
        build_validator(request, **{missing_field: None}).validate(request)

    assert error.value.code == "MANIFEST_MISMATCH"


def test_validator_rejects_inverted_collection_window():
    request = build_request()

    with pytest.raises(BatchValidationError) as error:
        build_validator(request, window_start="2026-08-13T00:00:00Z").validate(request)

    assert error.value.code == "MANIFEST_MISMATCH"


def test_validator_rejects_events_missing_a_required_column():
    request = build_request()

    with pytest.raises(BatchValidationError) as error:
        build_validator(request, drop_columns=("user_satisfied", "question")).validate(request)

    assert error.value.code == "EVENT_SCHEMA_MISMATCH"
    assert "question" in error.value.message
    assert "user_satisfied" in error.value.message


def test_validator_rejects_a_data_format_it_cannot_read():
    request = build_request()

    with pytest.raises(BatchValidationError) as error:
        build_validator(request, data_format="jsonl").validate(request)

    assert error.value.code == "UNSUPPORTED_DATA_FORMAT"


def test_validator_accepts_a_prefixed_sha256_like_the_pipeline_does():
    request = build_request()
    objects = build_objects(request)
    digest = json.loads(objects[request.manifestUri])["sha256"]
    validator = FeedbackBatchValidator(
        object_store=MemoryObjectStore(build_objects(request, sha256=f"sha256:{digest}")),
        allowed_buckets=("llm-router",),
        supported_schemas=("feedback-export-v0.1",),
    )

    assert validator.validate(request).row_count == 1000


async def test_lease_left_by_a_crashed_submission_is_released(tmp_path):
    request = build_request()
    service, pipeline = build_service(tmp_path, request)
    # A claim that never reached Vertex, aged past the stale threshold.
    service._job_store.claim(
        batch_id=request.batchId, request_fingerprint=service._fingerprint(request)
    )
    _age_job(tmp_path, request.batchId, seconds=3600)

    accepted = await service.submit(request)

    assert accepted.pipeline_run_id == "pipeline-1"
    assert len(pipeline.submissions) == 1


async def test_a_stale_lease_does_not_block_other_batches(tmp_path):
    stuck = build_request("feedback-20260812-0001")
    service, _ = build_service(tmp_path, stuck)
    service._job_store.claim(batch_id=stuck.batchId, request_fingerprint="stale-fingerprint")
    _age_job(tmp_path, stuck.batchId, seconds=3600)

    other = build_request("feedback-20260812-0002")
    service._validator = build_validator(other)
    accepted = await service.submit(other)

    assert accepted.pipeline_run_id == "pipeline-1"


async def test_an_orphaned_vertex_run_is_adopted_rather_than_resubmitted(tmp_path):
    # The process died after Vertex accepted the job but before the run ID was
    # recorded. Resubmitting would be a second GPU run for the same batch.
    request = build_request()
    service, pipeline = build_service(tmp_path, request)
    fingerprint = service._fingerprint(request)
    service._job_store.claim(batch_id=request.batchId, request_fingerprint=fingerprint)
    _age_job(tmp_path, request.batchId, seconds=3600)
    pipeline.runs_by_name[f"retrain-{request.batchId}"] = "orphaned-run"
    pipeline.states["orphaned-run"] = PipelineState.ACTIVE

    accepted = await service.submit(request)

    assert accepted.pipeline_run_id == "orphaned-run"
    assert pipeline.submissions == []


async def test_a_fresh_lease_is_left_alone(tmp_path):
    # Guards against stealing the lease from a submission still in flight.
    request = build_request("feedback-20260812-0001")
    service, _ = build_service(tmp_path, request)
    service._job_store.claim(batch_id=request.batchId, request_fingerprint="in-flight")

    other = build_request("feedback-20260812-0002")
    service._validator = build_validator(other)
    with pytest.raises(RetrainingInProgressError):
        await service.submit(other)


async def test_submit_passes_service_account_and_publish_root(tmp_path):
    request = build_request()
    service, pipeline = build_service(tmp_path, request)

    await service.submit(request)

    arguments = pipeline.submissions[0][1]
    assert arguments["training_service_account"] == "train@test-project.iam.gserviceaccount.com"
    assert arguments["training_base_output_directory"] == "gs://llm-router/vertex-training-output"
    assert arguments["model_publish_root_uri"] == "gs://llm-router/models/router-modernbert"
    # An unset optional must not be sent as an empty string; the IR default wins.
    assert "training_network" not in arguments


def test_vertex_job_id_is_deterministic_and_valid():
    first = vertex_job_id("retrain-Feedback_20260812/0001")
    second = vertex_job_id("retrain-Feedback_20260812/0001")

    assert first == second
    assert re.fullmatch(r"[a-z0-9][-a-z0-9]{0,127}", first)
    assert vertex_job_id("retrain-batch-a") != vertex_job_id("retrain-batch-b")


def test_vertex_job_id_separates_batches_that_normalize_identically():
    assert vertex_job_id("retrain-Batch_1") != vertex_job_id("retrain-batch/1")


class FakeVertexJob:
    def __init__(self, resource_name: str, state: str) -> None:
        self.resource_name = resource_name
        self.state = state
        self.submit_kwargs: dict | None = None

    def submit(self, **kwargs):
        self.submit_kwargs = kwargs


def _vertex_client(**overrides) -> VertexPipelineClient:
    kwargs = {
        "project": "test-project",
        "location": "us-west1",
        "template_uri": "gs://llm-router/pipelines/v1/feedback-retraining.yaml",
        "pipeline_root": "gs://llm-router/pipeline-root",
        "service_account": "pipeline@test-project.iam.gserviceaccount.com",
        "labels": {"component": "feedback-retraining"},
    }
    kwargs.update(overrides)
    return VertexPipelineClient(**kwargs)


def test_vertex_client_submits_gcs_template_and_returns_resource_name(monkeypatch):
    created: dict = {}
    job = FakeVertexJob(
        "projects/1/locations/us-west1/pipelineJobs/retrain-x",
        "PipelineState.PIPELINE_STATE_RUNNING",
    )

    def fake_pipeline_job(**kwargs):
        created.update(kwargs)
        return job

    monkeypatch.setattr("google.cloud.aiplatform.PipelineJob", fake_pipeline_job)

    submission = _vertex_client().submit(
        run_name="retrain-feedback-1",
        arguments={"feedback_manifest_uri": "gs://bucket/manifest.json"},
    )

    assert submission.run_id == "projects/1/locations/us-west1/pipelineJobs/retrain-x"
    assert created["template_path"] == "gs://llm-router/pipelines/v1/feedback-retraining.yaml"
    assert created["pipeline_root"] == "gs://llm-router/pipeline-root"
    assert created["enable_caching"] is False
    assert created["job_id"] == vertex_job_id("retrain-feedback-1")
    assert created["project"] == "test-project"
    assert job.submit_kwargs == {
        "service_account": "pipeline@test-project.iam.gserviceaccount.com",
        "network": None,
    }


def test_vertex_client_treats_already_exists_as_the_same_run(monkeypatch):
    from google.api_core.exceptions import AlreadyExists

    class ConflictingJob(FakeVertexJob):
        def submit(self, **kwargs):
            raise AlreadyExists("pipeline job already exists")

    monkeypatch.setattr(
        "google.cloud.aiplatform.PipelineJob",
        lambda **kwargs: ConflictingJob("", "PipelineState.PIPELINE_STATE_RUNNING"),
    )

    submission = _vertex_client().submit(run_name="retrain-feedback-1", arguments={})

    job_id = vertex_job_id("retrain-feedback-1")
    assert submission.run_id == (f"projects/test-project/locations/us-west1/pipelineJobs/{job_id}")


@pytest.mark.parametrize(
    ("raw_state", "expected"),
    [
        ("PipelineState.PIPELINE_STATE_SUCCEEDED", PipelineState.TERMINAL),
        ("PipelineState.PIPELINE_STATE_FAILED", PipelineState.TERMINAL),
        ("PipelineState.PIPELINE_STATE_CANCELLED", PipelineState.TERMINAL),
        ("PipelineState.PIPELINE_STATE_RUNNING", PipelineState.ACTIVE),
        ("PipelineState.PIPELINE_STATE_QUEUED", PipelineState.ACTIVE),
        ("PipelineState.PIPELINE_STATE_PAUSED", PipelineState.ACTIVE),
        ("PipelineState.PIPELINE_STATE_UNSPECIFIED", PipelineState.UNKNOWN),
    ],
)
def test_vertex_client_maps_pipeline_states(monkeypatch, raw_state, expected):
    monkeypatch.setattr(
        "google.cloud.aiplatform.PipelineJob.get",
        staticmethod(lambda **kwargs: FakeVertexJob("resource", raw_state)),
    )

    assert _vertex_client().state("resource") is expected


def test_vertex_client_maps_permission_denied_to_authorization_error(monkeypatch):
    from google.api_core.exceptions import PermissionDenied

    def raise_denied(**kwargs):
        raise PermissionDenied("caller lacks aiplatform.pipelineJobs.create")

    monkeypatch.setattr("google.cloud.aiplatform.PipelineJob", raise_denied)

    with pytest.raises(PipelineAuthorizationError):
        _vertex_client().submit(run_name="retrain-feedback-1", arguments={})
