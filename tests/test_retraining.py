from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from datetime import UTC, datetime, timedelta
from io import BytesIO

import pyarrow.parquet as pq
import pytest
from pydantic import ValidationError

from llm_router_serving.config import Settings
from llm_router_serving.retraining.gcs import (
    BatchValidationError,
    FeedbackBatchValidator,
    FeedbackReplayMaterializer,
)
from llm_router_serving.retraining.observability import RetrainingObservabilityReader
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
    VersionPolicyRevisionConflictError,
)
from llm_router_serving.schemas import EscalateRequest


class MemoryObjectStore:
    def __init__(self, objects: dict[str, bytes]) -> None:
        self.objects = objects

    def read_bytes(self, uri: str) -> bytes:
        return self.objects[uri]

    def write_bytes(self, uri: str, data: bytes, *, content_type: str) -> None:
        del content_type
        existing = self.objects.get(uri)
        if existing is not None and existing != data:
            raise AssertionError(f"immutable object conflict: {uri}")
        self.objects[uri] = data

    def delete_uri(self, uri: str) -> None:
        self.objects.pop(uri, None)


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


def build_event(event_id: str) -> dict[str, object]:
    return {
        "eventId": event_id,
        "requestId": f"request-{event_id}",
        "question": f"question for {event_id}",
        "initialLabel": "easy",
        "initialModel": "haiku",
        "finalLabel": "hard",
        "finalModel": "opus",
        "userSatisfied": True,
        "escalationCount": 1,
        "createdAt": "2026-08-12T15:00:00Z",
        "routerModelVersion": "router-x-20260731-21c2f1b",
        "piiRedacted": True,
    }


def build_request(
    batch_id: str = "feedback-20260812-0001",
    *,
    event_count: int = 3,
    id_prefix: str | None = None,
) -> EscalateRequest:
    prefix = id_prefix or batch_id
    return EscalateRequest(
        batchId=batch_id,
        schemaVersion="feedback-inline-v1",
        requestedAt=datetime(2026, 8, 12, 15, 30, tzinfo=UTC),
        events=[build_event(f"{prefix}-{index}") for index in range(event_count)],
    )


def build_validator() -> FeedbackBatchValidator:
    return FeedbackBatchValidator(
        supported_schemas=("feedback-inline-v1",),
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
    del request
    pipeline = FakePipelineClient()
    settings = Settings(
        load_models_on_startup=False,
        retraining_db_path=tmp_path / "retraining.sqlite3",
        allowed_gcs_buckets=("llm-router",),
        supported_feedback_schemas=("feedback-inline-v1",),
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
    store = MemoryObjectStore({})
    validator = FeedbackBatchValidator(supported_schemas=settings.supported_feedback_schemas)
    service = EscalationService(
        settings=settings,
        validator=validator,
        job_store=RetrainingJobStore(settings.retraining_db_path),
        pipeline_client=pipeline,
        materializer=FeedbackReplayMaterializer(
            object_store=store,
            root_uri="gs://llm-router/derived",
            rows_per_label=2,
        ),
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
    assert arguments["feedback_manifest_uri"].startswith("gs://llm-router/derived/")
    assert arguments["feedback_events_uri"].startswith("gs://llm-router/derived/")
    assert arguments["baseline_model_version"].startswith("router-modernbert")
    assert arguments["candidate_model_version"].startswith(
        "router-modernbert-feedback-v1.1-20260812-"
    )
    assert arguments["candidate_model_version"].endswith("21c2f1b")
    assert arguments["logical_model_version"] == "1.1"
    assert arguments["training_mode"] == "minor_fine_tune"
    assert arguments["initialization_mode"] == "approved_anchor"
    assert first.logical_model_version == "1.1"
    assert first.training_mode == "minor_fine_tune"
    assert first.version_policy_mode == "successfulHttpRequest"


async def test_status_distinguishes_pipeline_success_from_model_activation(tmp_path):
    request = build_request(event_count=1)
    service, pipeline = build_service(tmp_path, request)
    accepted = await service.submit(request)

    running = await service.status(
        request.batchId,
        serving_model_version="router-old",
        desired_model_version=None,
        desired_load_error=None,
    )
    assert running.status == "running"
    assert running.terminal is False
    assert running.activation.status == "pending"

    pipeline.states[accepted.pipeline_run_id] = PipelineState.SUCCEEDED
    succeeded = await service.status(
        request.batchId,
        serving_model_version="router-old",
        desired_model_version=accepted.candidate_model_version,
        desired_load_error=None,
    )
    assert succeeded.status == "succeeded"
    assert succeeded.terminal is True
    assert succeeded.result is not None
    assert succeeded.result.outcome == "promoted"
    assert succeeded.activation.status == "pending"

    active = await service.status(
        request.batchId,
        serving_model_version=accepted.candidate_model_version,
        desired_model_version=accepted.candidate_model_version,
        desired_load_error=None,
    )
    assert active.activation.status == "active"


async def test_status_projects_observability_metrics_and_gate_rejection(tmp_path):
    request = build_request(event_count=3)
    service, pipeline = build_service(tmp_path, request)
    observed_store = MemoryObjectStore({})
    root = "gs://llm-router/retraining-runs"
    service._observability_reader = RetrainingObservabilityReader(
        object_store=observed_store,
        root_uri=root,
    )
    accepted = await service.submit(request)
    run_root = f"{root}/{request.batchId}/observability-v1"

    def observe(relative: str, payload: dict) -> None:
        observed_store.objects[f"{run_root}/{relative}"] = json.dumps(
            {
                "batch_id": request.batchId,
                "candidate_model_version": accepted.candidate_model_version,
                "updated_at": "2026-09-12T07:12:30Z",
                **payload,
            }
        ).encode()

    observe(
        "metrics/feedback-prepare.json",
        {
            "input_event_count": 3,
            "accepted_event_count": 3,
            "excluded_event_count": 0,
            "training_row_count": 2,
            "exclusion_distribution": {},
            "label_distribution": {"medium": 1, "hard": 1},
        },
    )
    observe(
        "metrics/feedback-merge.json",
        {
            "feedback_input_row_count": 2,
            "accepted_feedback_row_count": 2,
            "excluded_feedback_row_count": 0,
            "replaced_base_row_count": 1,
            "merged_training_row_count": 146,
            "merge_decision_distribution": {"append_new_question": 1},
        },
    )
    observe(
        "progress/training.json",
        {
            "optimizer_step": 25,
            "total_steps": 100,
            "epoch": 1,
            "total_epochs": 3,
        },
    )
    running = await service.status(
        request.batchId,
        serving_model_version="router-old",
        desired_model_version=None,
        desired_load_error=None,
    )
    assert running.stage == "training"
    assert running.stageProgress is not None
    assert running.stageProgress.percent == 25
    assert running.filtering is not None
    assert running.filtering.collapsedDuplicateEventCount == 1
    assert running.merge is not None
    assert running.merge.mergedTrainingRowCount == 146

    observe(
        "gate/gate-result.json",
        {
            "passed": False,
            "failed_checks": [{"name": "macroF1Regression"}],
        },
    )
    pipeline.states[accepted.pipeline_run_id] = PipelineState.FAILED
    rejected = await service.status(
        request.batchId,
        serving_model_version="router-old",
        desired_model_version=None,
        desired_load_error=None,
    )
    assert rejected.status == "rejected"
    assert rejected.result is not None
    assert rejected.result.failureCode == "DEPLOY_GATE_REJECTED"
    assert rejected.result.failedGateChecks == ["macroF1Regression"]


def test_version_policy_change_is_durable_and_revision_guarded(tmp_path):
    request = build_request()
    service, _ = build_service(tmp_path, request)

    original = service.version_policy()
    assert original.mode == "successfulHttpRequest"
    changed = service.schedule_version_policy(
        expected_revision=original.revision,
        mode="validatedUniqueBatch",
        actor="test-developer",
        reason="contract test",
    )
    assert changed.mode == "successfulHttpRequest"
    assert changed.pendingChange is not None
    assert changed.pendingChange.mode == "validatedUniqueBatch"
    assert service.version_policy().revision == changed.revision

    with pytest.raises(VersionPolicyRevisionConflictError, match=str(original.revision)):
        service.schedule_version_policy(
            expected_revision=original.revision,
            mode="successfulHttpRequest",
            actor="stale-client",
            reason="stale write",
        )


def test_candidate_versions_do_not_collide_for_same_day_batches(tmp_path):
    first_request = build_request("feedback-20260812-0001")
    service, _ = build_service(tmp_path, first_request)
    second_request = build_request("feedback-20260812-0002")

    first_version = service.candidate_model_version(first_request)
    second_version = service.candidate_model_version(second_request)

    assert first_version != second_version
    assert first_version.endswith("21c2f1b")
    assert second_version.endswith("21c2f1b")


def test_candidate_version_ignores_retry_timestamp(tmp_path):
    request = build_request()
    service, _ = build_service(tmp_path, request)
    retried_next_day = request.model_copy(
        update={"requestedAt": datetime(2026, 8, 13, 1, tzinfo=UTC)}
    )

    assert service.candidate_model_version(request) == service.candidate_model_version(
        retried_next_day
    )


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

    # Same batch ID, different inline data. The conflict comes from the durable
    # idempotency record before another pipeline can be submitted.
    changed_event = request.events[0].model_copy(update={"question": "changed question"})
    changed = request.model_copy(update={"events": [changed_event, *request.events[1:]]})

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


async def test_retry_with_same_events_in_another_order_is_idempotent(tmp_path):
    request = build_request(event_count=3)
    service, pipeline = build_service(tmp_path, request)
    first = await service.submit(request)

    reordered = request.model_copy(update={"events": list(reversed(request.events))})
    second = await service.submit(reordered)

    assert second.pipeline_run_id == first.pipeline_run_id
    assert len(pipeline.submissions) == 1


async def test_only_one_non_terminal_pipeline_can_run(tmp_path):
    first_request = build_request("feedback-20260812-0001")
    service, pipeline = build_service(tmp_path, first_request)
    first = await service.submit(first_request)

    second_request = build_request("feedback-20260812-0002")
    service._validator = build_validator()
    with pytest.raises(RetrainingInProgressError):
        await service.submit(second_request)

    pipeline.states[first.pipeline_run_id] = PipelineState.SUCCEEDED
    accepted = await service.submit(second_request)
    assert accepted.pipeline_run_id == "pipeline-2"


async def test_three_successful_calls_make_a_major_and_fifth_major_rebases(tmp_path):
    first_request = build_request("feedback-20260812-0001")
    service, pipeline = build_service(tmp_path, first_request)

    versions: list[str] = []
    modes: list[str] = []
    initializations: list[str] = []
    for index in range(1, 16):
        request = build_request(f"feedback-20260812-{index:04d}")
        service._validator = build_validator()
        accepted = await service.submit(request)
        arguments = pipeline.submissions[-1][1]
        versions.append(arguments["logical_model_version"])
        modes.append(arguments["training_mode"])
        initializations.append(arguments["initialization_mode"])
        pipeline.states[accepted.pipeline_run_id] = PipelineState.SUCCEEDED

    assert versions[:4] == ["1.1", "1.2", "2.0", "2.1"]
    assert modes[2] == "major_consolidation"
    assert versions[-1] == "6.0"
    assert modes[-1] == "pristine_rebase"
    assert initializations[-1] == "pristine_modernbert"


def test_materializer_keeps_full_cycle_input_and_caps_replay_per_label():
    request = build_request(event_count=5)
    validated = build_validator().validate(request)
    store = MemoryObjectStore({})
    materializer = FeedbackReplayMaterializer(
        object_store=store,
        root_uri="gs://llm-router/derived",
        rows_per_label=2,
    )

    result = materializer.materialize(
        request=request,
        validated=validated,
        artifact_key="router-modernbert-feedback-v1.1-test",
        logical_model_version="1.1",
        previous_cycle_events_uri=None,
        previous_replay_events_uri=None,
    )

    assert result.event_count == 5
    assert result.replay_event_count == 2
    assert pq.read_table(BytesIO(store.objects[result.events_uri])).num_rows == 5
    assert pq.read_table(BytesIO(store.objects[result.replay_events_uri])).num_rows == 2
    replay_manifest = json.loads(store.objects[result.replay_manifest_uri])
    assert replay_manifest["schema_version"] == "feedback-export-v0.1"
    assert replay_manifest["event_count"] == 2
    assert replay_manifest["data_uri"] == result.replay_events_uri
    assert (
        replay_manifest["sha256"]
        == hashlib.sha256(store.objects[result.replay_events_uri]).hexdigest()
    )
    assert replay_manifest["producer_version"] == "llm-router-serving-inline-v1"
    assert replay_manifest["window_start"] == "2026-08-12T15:00:00Z"
    assert replay_manifest["window_end"] == "2026-08-12T15:00:00Z"


async def test_service_accumulates_the_cycle_and_retires_superseded_derivatives(tmp_path):
    first_request = build_request("feedback-20260812-0001")
    second_request = build_request("feedback-20260812-0002")
    service, pipeline = build_service(tmp_path, first_request)
    shared_store = MemoryObjectStore({})
    service._validator = build_validator()
    service._materializer = FeedbackReplayMaterializer(
        object_store=shared_store,
        root_uri="gs://llm-router/derived",
        rows_per_label=2,
    )

    first = await service.submit(first_request)
    first_events_uri = pipeline.submissions[-1][1]["feedback_events_uri"]
    pipeline.states[first.pipeline_run_id] = PipelineState.SUCCEEDED
    second = await service.submit(second_request)
    second_events_uri = pipeline.submissions[-1][1]["feedback_events_uri"]

    assert pq.read_table(BytesIO(shared_store.objects[second_events_uri])).num_rows == 6
    pipeline.states[second.pipeline_run_id] = PipelineState.SUCCEEDED
    await service._reconcile_active_jobs()
    assert first_events_uri not in shared_store.objects
    assert second_events_uri in shared_store.objects


def test_validator_converts_one_inline_event_to_pipeline_parquet():
    request = build_request(event_count=1)

    result = build_validator().validate(request)
    table = pq.read_table(BytesIO(result.events_bytes))

    assert result.row_count == 1
    assert table.schema.names == [
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
    ]
    assert str(table.column("created_at").type) == "timestamp[ms, tz=UTC]"


def test_inline_parquet_is_stable_when_event_order_changes():
    request = build_request(event_count=3)
    reordered = request.model_copy(update={"events": list(reversed(request.events))})

    first = build_validator().validate(request)
    second = build_validator().validate(reordered)

    assert first.events_sha256 == second.events_sha256
    assert first.events_bytes == second.events_bytes


def test_validator_rejects_an_unsupported_inline_schema():
    request = build_request().model_copy(update={"schemaVersion": "feedback-inline-v2"})

    with pytest.raises(BatchValidationError) as error:
        build_validator().validate(request)

    assert error.value.code == "UNSUPPORTED_SCHEMA_VERSION"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("finalLabel", "easy"),
        ("userSatisfied", False),
        ("piiRedacted", False),
        ("createdAt", "2026-08-13T00:00:00+09:00"),
    ],
)
def test_inline_event_rejects_unusable_training_rows(field, value):
    event = build_event("event-1")
    event[field] = value

    with pytest.raises(ValidationError):
        EscalateRequest.model_validate(
            {
                "batchId": "feedback-20260812-0001",
                "schemaVersion": "feedback-inline-v1",
                "requestedAt": "2026-08-12T15:30:00Z",
                "events": [event],
            }
        )


def test_inline_request_rejects_duplicate_event_ids():
    event = build_event("duplicate")

    with pytest.raises(ValidationError):
        EscalateRequest.model_validate(
            {
                "batchId": "feedback-20260812-0001",
                "schemaVersion": "feedback-inline-v1",
                "requestedAt": "2026-08-12T15:30:00Z",
                "events": [event, event],
            }
        )


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
    service._validator = build_validator()
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
    service._validator = build_validator()
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


async def test_successor_uses_approved_gate_metrics_as_its_baseline(tmp_path):
    first_request = build_request("feedback-20260812-first")
    service, pipeline = build_service(tmp_path, first_request)
    service._settings.model_evaluation_metrics_filename = "gate_metrics.json"
    first = await service.submit(first_request)
    assert pipeline.submissions[0][1]["baseline_metrics_uri"] == (
        "gs://llm-router/eval/baseline.json"
    )
    pipeline.states[first.pipeline_run_id] = PipelineState.SUCCEEDED
    second_request = build_request("feedback-20260812-second")
    await service.submit(second_request)
    assert pipeline.submissions[1][1]["baseline_metrics_uri"] == (
        "gs://llm-router/models/router-modernbert/"
        f"{first.candidate_model_version}/approved/gate_metrics.json"
    )


def test_structured_digest_preserves_code_and_excludes_conflicting_feedback(tmp_path):
    questions = [
        "계산: x+y",
        "계산: x-y",
        "  if X:\n    return Y",
        "if X:\n    return Y",
        "계산: ｘ＋ｙ",  # noqa: RUF001 - NFKC-equivalent feedback must collapse.
        "동일 질문",
        "동일 질문",
    ]
    events = [
        {
            **build_event(f"event-{index}"),
            "question": question,
            "finalLabel": "medium" if index == 6 else "hard",
        }
        for index, question in enumerate(questions)
    ]
    request = build_request().model_copy(
        update={
            "events": [type(build_request().events[0]).model_validate(event) for event in events]
        }
    )
    service, _ = build_service(tmp_path, request)
    service._settings.retraining_input_contract = "structure-preserving-v2"
    expected_rows = [
        {"normalized_question": question, "final_label": "hard", "sample_weight": weight}
        for question, weight in sorted(
            [
                ("계산: x+y", 1.25),
                ("계산: x-y", 1.0),
                ("  if X:\n    return Y", 1.0),
                ("if X:\n    return Y", 1.0),
            ]
        )
    ]
    expected = hashlib.sha256(
        json.dumps(
            {
                "input_contract": "structure-preserving-v2",
                "rows": expected_rows,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    assert service._canonical_batch_digest(request) == expected
    assert (
        service._canonical_batch_digest(
            request.model_copy(update={"events": list(reversed(request.events))})
        )
        == expected
    )
    service._settings.retraining_input_contract = "legacy-word-only-v1"
    assert service._canonical_batch_digest(request) != expected


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
        ("PipelineState.PIPELINE_STATE_SUCCEEDED", PipelineState.SUCCEEDED),
        ("PipelineState.PIPELINE_STATE_FAILED", PipelineState.FAILED),
        ("PipelineState.PIPELINE_STATE_CANCELLED", PipelineState.CANCELLED),
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


@pytest.mark.parametrize("representation", ["enum", "integer", "numeric_string"])
def test_vertex_client_maps_real_sdk_states(monkeypatch, representation):
    from google.cloud.aiplatform_v1.types import PipelineState as VertexPipelineState

    for sdk_state in VertexPipelineState:
        raw_state = {
            "enum": sdk_state,
            "integer": int(sdk_state),
            "numeric_string": str(int(sdk_state)),
        }[representation]
        monkeypatch.setattr(
            "google.cloud.aiplatform.PipelineJob.get",
            staticmethod(lambda state=raw_state, **kwargs: FakeVertexJob("resource", state)),
        )
        expected = {
            "PIPELINE_STATE_SUCCEEDED": PipelineState.SUCCEEDED,
            "PIPELINE_STATE_FAILED": PipelineState.FAILED,
            "PIPELINE_STATE_CANCELLED": PipelineState.CANCELLED,
            "PIPELINE_STATE_UNSPECIFIED": PipelineState.UNKNOWN,
        }.get(sdk_state.name, PipelineState.ACTIVE)
        assert _vertex_client().state("resource") is expected


@pytest.mark.asyncio
async def test_unknown_pipeline_state_is_stale_retains_lease_and_recovers(tmp_path):
    request = build_request()
    service, pipeline = build_service(tmp_path, request)
    accepted = await service.submit(request)
    pipeline.states[accepted.pipeline_run_id] = PipelineState.UNKNOWN
    stale = await service.status(
        request.batchId,
        serving_model_version="router-old",
        desired_model_version=None,
        desired_load_error=None,
    )
    assert stale.stale is True
    assert stale.terminal is False
    assert any("unrecognized state" in warning for warning in stale.warnings)
    assert service._job_store.get(request.batchId).status == "submitted"
    with pytest.raises(RetrainingInProgressError):
        await service.submit(build_request("feedback-next"))
    pipeline.states[accepted.pipeline_run_id] = PipelineState.FAILED
    failed = await service.status(
        request.batchId,
        serving_model_version="router-old",
        desired_model_version=None,
        desired_load_error=None,
    )
    assert failed.status == "failed"
    assert failed.terminal is True
    assert failed.stale is False
    assert service._job_store.list_active() == []


def test_vertex_client_maps_permission_denied_to_authorization_error(monkeypatch):
    from google.api_core.exceptions import PermissionDenied

    def raise_denied(**kwargs):
        raise PermissionDenied("caller lacks aiplatform.pipelineJobs.create")

    monkeypatch.setattr("google.cloud.aiplatform.PipelineJob", raise_denied)

    with pytest.raises(PipelineAuthorizationError):
        _vertex_client().submit(run_name="retrain-feedback-1", arguments={})
