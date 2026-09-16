from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime

from llm_router_serving.config import Settings
from llm_router_serving.retraining.gcs import (
    FeedbackBatchValidator,
    FeedbackReplayMaterializer,
    MaterializedFeedback,
    ValidatedFeedbackBatch,
)
from llm_router_serving.retraining.observability import (
    ObservabilitySnapshot,
    RetrainingObservabilityReader,
)
from llm_router_serving.retraining.pipeline import PipelineClient, PipelineState
from llm_router_serving.retraining.store import (
    ABANDONED_STATUS,
    RetrainingJob,
    RetrainingJobStore,
    VersionPolicyState,
)
from llm_router_serving.retraining.strategy import (
    ContinualTrainingPlan,
    build_continual_training_plan,
)
from llm_router_serving.schemas import (
    ActivationStatus,
    ActivationStatusValue,
    EscalateRequest,
    FailureCode,
    FilteringMetrics,
    InitializationMode,
    MergeMetrics,
    PendingPolicyChange,
    RetrainingOutcome,
    RetrainingOverallStatus,
    RetrainingResult,
    RetrainingStageName,
    RetrainingStatusResponse,
    StageExecutionStatus,
    StageProgress,
    StageRecord,
    TrainingMode,
    TrainingVersion,
    VersionPolicyMode,
    VersionPolicyResponse,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AcceptedRetraining:
    batch_id: str
    pipeline_run_id: str
    logical_model_version: str
    candidate_model_version: str
    training_mode: str
    version_policy_mode: str


class RetrainingRunNotFoundError(RuntimeError):
    pass


class EscalationService:
    def __init__(
        self,
        *,
        settings: Settings,
        validator: FeedbackBatchValidator,
        job_store: RetrainingJobStore,
        pipeline_client: PipelineClient,
        materializer: FeedbackReplayMaterializer,
        observability_reader: RetrainingObservabilityReader | None = None,
    ) -> None:
        self._settings = settings
        self._validator = validator
        self._job_store = job_store
        self._pipeline_client = pipeline_client
        self._materializer = materializer
        self._observability_reader = observability_reader
        self._job_store.get_or_create_version_policy(
            default_mode=settings.retraining_version_policy_mode
        )

    async def submit(self, request: EscalateRequest) -> AcceptedRetraining:
        fingerprint = self._fingerprint(request)
        existing = await asyncio.to_thread(self._job_store.get, request.batchId)
        if existing and existing.request_fingerprint == fingerprint and existing.pipeline_run_id:
            return self._accepted(existing)

        validated = await asyncio.to_thread(self._validator.validate, request)
        await self._reconcile_active_jobs()
        policy = await asyncio.to_thread(
            self._job_store.get_or_create_version_policy,
            default_mode=self._settings.retraining_version_policy_mode,
        )
        successful_runs = await asyncio.to_thread(self._job_store.list_succeeded)
        plan = self._training_plan(successful_runs)
        candidate_version = self.candidate_model_version(
            request, logical_model_version=plan.logical_model_version
        )
        claimed = await asyncio.to_thread(
            self._job_store.claim,
            batch_id=request.batchId,
            request_fingerprint=fingerprint,
            logical_model_version=plan.logical_model_version,
            candidate_model_version=candidate_version,
            training_mode=plan.training_mode.value,
            initialization_mode=plan.initialization_mode.value,
            version_policy_mode=policy.mode,
            canonical_batch_digest=self._canonical_batch_digest(request),
            base_model_version=plan.anchor_model_version,
            base_model_uri=plan.anchor_model_uri,
        )
        if claimed and claimed.pipeline_run_id:
            return self._accepted(claimed)

        run_name = self._run_name(request.batchId)
        try:
            materialized = await asyncio.to_thread(
                self._materialize_feedback,
                request,
                validated,
                plan,
                candidate_version,
            )
            await asyncio.to_thread(
                self._job_store.set_materialized_feedback,
                batch_id=request.batchId,
                feedback_manifest_uri=materialized.manifest_uri,
                feedback_events_uri=materialized.events_uri,
                replay_manifest_uri=materialized.replay_manifest_uri,
                replay_events_uri=materialized.replay_events_uri,
            )
            arguments = self._pipeline_arguments(
                batch_id=request.batchId,
                plan=plan,
                materialized=materialized,
                candidate_version=candidate_version,
                version_policy_mode=policy.mode,
            )
            submission = await asyncio.to_thread(
                self._pipeline_client.submit,
                run_name=run_name,
                arguments=arguments,
            )
        except Exception:
            await asyncio.to_thread(
                self._job_store.mark_status, batch_id=request.batchId, status="failed"
            )
            raise
        await asyncio.to_thread(
            self._job_store.mark_submitted,
            batch_id=request.batchId,
            pipeline_run_id=submission.run_id,
        )
        return AcceptedRetraining(
            batch_id=request.batchId,
            pipeline_run_id=submission.run_id,
            logical_model_version=plan.logical_model_version,
            candidate_model_version=candidate_version,
            training_mode=plan.training_mode.value,
            version_policy_mode=policy.mode,
        )

    async def _reconcile_active_jobs(self) -> None:
        """Release leases whose pipeline is no longer running.

        This covers the current batch too: a lease it left behind after a crash
        would otherwise block its own retry as well as every other batch.
        """
        for job in await asyncio.to_thread(self._job_store.list_active):
            if job.pipeline_run_id:
                await self._reconcile_submitted_job(job)
            else:
                await self._reconcile_unsubmitted_job(job)

    async def _reconcile_submitted_job(self, job: RetrainingJob) -> None:
        assert job.pipeline_run_id is not None
        state = await asyncio.to_thread(self._pipeline_client.state, job.pipeline_run_id)
        if state is PipelineState.SUCCEEDED:
            await asyncio.to_thread(
                self._job_store.mark_status,
                batch_id=job.batch_id,
                status="succeeded",
            )
            await self._retire_superseded_feedback(job)
            if (job.logical_model_version or "").endswith(".0"):
                await asyncio.to_thread(self._job_store.activate_pending_version_policy)
        elif state in {PipelineState.FAILED, PipelineState.CANCELLED}:
            await asyncio.to_thread(
                self._job_store.mark_status,
                batch_id=job.batch_id,
                status=state.value,
            )

    async def _reconcile_unsubmitted_job(self, job: RetrainingJob) -> None:
        """Resolve a lease claimed but never recorded as submitted.

        Submission takes seconds, so a lease older than the stale threshold
        means the process died mid-submit rather than that a submit is still in
        flight. The threshold is what keeps this from stealing the lease out
        from under a concurrent request in this same process.
        """
        if not self._is_stale(job):
            return
        run_id = await asyncio.to_thread(
            self._pipeline_client.find_run, self._run_name(job.batch_id)
        )
        if run_id is None:
            logger.warning(
                "Releasing stale retraining lease for batch %s; it never reached Vertex",
                job.batch_id,
            )
            await asyncio.to_thread(
                self._job_store.mark_status,
                batch_id=job.batch_id,
                status=ABANDONED_STATUS,
            )
            return
        logger.warning("Adopting orphaned Vertex run %s for batch %s", run_id, job.batch_id)
        await asyncio.to_thread(
            self._job_store.mark_submitted,
            batch_id=job.batch_id,
            pipeline_run_id=run_id,
        )
        await self._reconcile_submitted_job(
            RetrainingJob(
                batch_id=job.batch_id,
                request_fingerprint=job.request_fingerprint,
                pipeline_run_id=run_id,
                status="submitted",
                updated_at=job.updated_at,
            )
        )

    async def _retire_superseded_feedback(self, current: RetrainingJob) -> None:
        succeeded = await asyncio.to_thread(self._job_store.list_succeeded)
        protected = {
            uri
            for uri in (
                current.feedback_manifest_uri,
                current.feedback_events_uri,
                current.replay_manifest_uri,
                current.replay_events_uri,
            )
            if uri
        }
        superseded = [
            uri
            for job in succeeded
            if job.batch_id != current.batch_id
            for uri in (
                job.feedback_manifest_uri,
                job.feedback_events_uri,
                job.replay_manifest_uri,
                job.replay_events_uri,
            )
            if uri
        ]
        warnings = await asyncio.to_thread(
            self._materializer.retire_superseded,
            superseded_uris=superseded,
            protected_uris=protected,
        )
        for warning in warnings:
            logger.warning("Derived feedback retention cleanup failed: %s", warning)

    def _is_stale(self, job: RetrainingJob) -> bool:
        if job.updated_at is None:
            return True
        age = (datetime.now(UTC) - job.updated_at).total_seconds()
        return age >= self._settings.retraining_stale_submission_seconds

    @staticmethod
    def _run_name(batch_id: str) -> str:
        return f"retrain-{batch_id}"

    def _pipeline_arguments(
        self,
        *,
        batch_id: str,
        plan: ContinualTrainingPlan,
        materialized: MaterializedFeedback,
        candidate_version: str,
        version_policy_mode: str,
    ) -> dict[str, str]:
        settings = self._settings
        baseline_metrics_uri = str(settings.baseline_metrics_uri)
        if plan.baseline_model_version != settings.router_model_version:
            baseline_metrics_uri = f"{plan.baseline_model_uri.rstrip('/')}/metrics.json"
        arguments = {
            "base_training_dataset_uri": str(settings.base_training_dataset_uri),
            "feedback_manifest_uri": materialized.manifest_uri,
            "feedback_events_uri": materialized.events_uri,
            "feedback_preprocessing_config_uri": str(settings.feedback_preprocessing_config_uri),
            "training_config_uri": str(settings.training_config_uri),
            "routerarena_dataset_uri": str(settings.routerarena_dataset_uri),
            "evaluation_config_uri": str(settings.evaluation_config_uri),
            "package_metadata_uri": str(settings.package_metadata_uri),
            "baseline_metrics_uri": baseline_metrics_uri,
            "gate_rules_uri": str(settings.gate_rules_uri),
            "candidate_model_version": candidate_version,
            "baseline_model_version": plan.baseline_model_version,
            "initial_model_uri": plan.anchor_model_uri,
            "initialization_mode": plan.initialization_mode.value,
            "logical_model_version": plan.logical_model_version,
            "training_mode": plan.training_mode.value,
            "batch_id": batch_id,
            "retention_anchor_model_version": (
                candidate_version
                if plan.logical_model_version.endswith(".0")
                else plan.anchor_model_version
            ),
        }
        # Without these the compiled IR falls back to its own defaults, which
        # would silently run the CustomJob as the default compute service account.
        optional = {
            "model_publish_root_uri": settings.model_publish_root_uri,
            "training_service_account": settings.vertex_training_service_account,
            "training_network": settings.vertex_pipeline_network,
            "training_base_output_directory": settings.vertex_training_output_directory,
            "observability_run_uri": (
                f"{settings.retraining_observability_root_uri.rstrip('/')}/{batch_id}/"
                "observability-v1"
                if settings.retraining_observability_root_uri
                else None
            ),
            "version_policy_mode": version_policy_mode,
            "retraining_control_root_uri": settings.retraining_control_root_uri,
        }
        arguments.update({key: value for key, value in optional.items() if value})
        return arguments

    def candidate_model_version(
        self,
        request: EscalateRequest,
        *,
        logical_model_version: str | None = None,
    ) -> str:
        if logical_model_version is None:
            logical_model_version = self._training_plan(
                self._job_store.list_succeeded()
            ).logical_model_version
        requested_date = (
            max(event.createdAt for event in request.events).astimezone(UTC).strftime("%Y%m%d")
        )
        batch_digest = hashlib.sha256(request.batchId.encode("utf-8")).hexdigest()[:8]
        return (
            f"router-modernbert-feedback-v{logical_model_version}-{requested_date}-"
            f"{batch_digest}-"
            f"{str(self._settings.model_code_commit)[:7]}"
        )

    def _training_plan(self, successful_runs: list[RetrainingJob]) -> ContinualTrainingPlan:
        initial_uri = self._settings.initial_approved_model_uri()
        if not initial_uri or not self._settings.router_model_version:
            raise RuntimeError("An initial approved model package is required for retraining")
        return build_continual_training_plan(
            successful_runs,
            initial_logical_major=self._settings.retraining_initial_logical_major,
            calls_per_major=self._settings.retraining_calls_per_major,
            rebase_every_major_cycles=self._settings.retraining_rebase_every_major_cycles,
            initial_model_version=self._settings.router_model_version,
            initial_model_uri=initial_uri,
        )

    async def status(
        self,
        batch_id: str,
        *,
        serving_model_version: str | None,
        desired_model_version: str | None,
        desired_load_error: str | None,
    ) -> RetrainingStatusResponse:
        job = await asyncio.to_thread(self._job_store.get, batch_id)
        if job is None or not job.pipeline_run_id:
            raise RetrainingRunNotFoundError(batch_id)

        stale = False
        warnings: list[str] = []
        if job.status in {"submitted", "running", "pending", "canceling", "paused"}:
            try:
                await self._reconcile_submitted_job(job)
            except Exception as exc:
                stale = True
                warnings.append(f"pipeline status refresh unavailable: {type(exc).__name__}")
            job = await asyncio.to_thread(self._job_store.get, batch_id) or job

        now = datetime.now(UTC)
        overall = self._overall_status(job.status)
        observed = self._read_observability(job)
        unique_rejection = (
            observed.feedback_prepare
            and observed.feedback_prepare.get("failure_code") == "NO_UNIQUE_FEEDBACK"
        )
        if unique_rejection or (observed.gate and observed.gate.get("passed") is False):
            overall = RetrainingOverallStatus.REJECTED
            if job.status != "rejected":
                await asyncio.to_thread(
                    self._job_store.mark_status, batch_id=batch_id, status="rejected"
                )
                job = await asyncio.to_thread(self._job_store.get, batch_id) or job
        terminal = overall in {
            RetrainingOverallStatus.SUCCEEDED,
            RetrainingOverallStatus.REJECTED,
            RetrainingOverallStatus.FAILED,
            RetrainingOverallStatus.CANCELLED,
        }
        if not terminal and not stale and overall is RetrainingOverallStatus.RUNNING:
            await asyncio.to_thread(
                self._job_store.mark_status, batch_id=batch_id, status="running"
            )
            job = await asyncio.to_thread(self._job_store.get, batch_id) or job

        result = self._result(job, overall, observed=observed)
        activation = self._activation(
            job,
            overall=overall,
            serving_model_version=serving_model_version,
            desired_model_version=desired_model_version,
            desired_load_error=desired_load_error,
            checked_at=now,
        )
        return RetrainingStatusResponse(
            batchId=job.batch_id,
            pipelineRunId=job.pipeline_run_id,
            status=overall,
            terminal=terminal,
            stage=self._current_stage(overall, observed),
            version=TrainingVersion(
                logicalModelVersion=job.logical_model_version or "0.0",
                candidateModelVersion=job.candidate_model_version or "unknown",
                trainingMode=TrainingMode(job.training_mode or TrainingMode.MINOR_FINE_TUNE),
                initializationMode=InitializationMode(
                    job.initialization_mode or InitializationMode.APPROVED_ANCHOR
                ),
                versionPolicyMode=VersionPolicyMode(
                    job.version_policy_mode or VersionPolicyMode.SUCCESSFUL_HTTP_REQUEST
                ),
            ),
            stageProgress=self._stage_progress(observed.training_progress),
            stages=self._stages(overall, observed=observed, timestamp=job.ended_at),
            filtering=self._filtering_metrics(observed.feedback_prepare),
            merge=self._merge_metrics(observed.feedback_merge),
            result=result,
            activation=activation,
            submittedAt=job.submitted_at or job.created_at or now,
            startedAt=job.started_at,
            endedAt=job.ended_at,
            lastUpdatedAt=job.updated_at or now,
            stale=stale,
            warnings=[*warnings, *observed.warnings],
        )

    def version_policy(self) -> VersionPolicyResponse:
        state = self._job_store.get_or_create_version_policy(
            default_mode=self._settings.retraining_version_policy_mode
        )
        successful = self._job_store.list_succeeded()
        next_plan = self._training_plan(successful)
        current_version = (
            successful[-1].logical_model_version
            if successful and successful[-1].logical_model_version
            else f"{self._settings.retraining_initial_logical_major}.0"
        )
        return self._version_policy_response(
            state,
            committed_successful_runs=len(successful),
            current_version=current_version,
            next_version=next_plan.logical_model_version,
        )

    def schedule_version_policy(
        self, *, expected_revision: int, mode: str, actor: str, reason: str
    ) -> VersionPolicyResponse:
        self._job_store.get_or_create_version_policy(
            default_mode=self._settings.retraining_version_policy_mode
        )
        self._job_store.schedule_version_policy_change(
            expected_revision=expected_revision,
            mode=mode,
            actor=actor,
            reason=reason,
        )
        return self.version_policy()

    def _version_policy_response(
        self,
        state: VersionPolicyState,
        *,
        committed_successful_runs: int,
        current_version: str,
        next_version: str,
    ) -> VersionPolicyResponse:
        pending = None
        if state.pending_mode:
            pending = PendingPolicyChange(
                mode=VersionPolicyMode(state.pending_mode),
                requestedAt=state.pending_requested_at or state.updated_at,
                requestedBy=state.pending_requested_by or "developer-api",
                reason=state.pending_reason or "policy change",
            )
        return VersionPolicyResponse(
            mode=VersionPolicyMode(state.mode),
            callsPerMajor=self._settings.retraining_calls_per_major,
            rebaseEveryMajorCycles=self._settings.retraining_rebase_every_major_cycles,
            committedSuccessfulRuns=committed_successful_runs,
            cyclePosition=committed_successful_runs % self._settings.retraining_calls_per_major,
            currentLogicalModelVersion=current_version,
            nextLogicalModelVersion=next_version,
            revision=state.revision,
            pendingChange=pending,
            updatedAt=state.updated_at,
            updatedBy=state.updated_by,
        )

    @staticmethod
    def _overall_status(value: str) -> RetrainingOverallStatus:
        mapping = {
            "submitting": RetrainingOverallStatus.QUEUED,
            "submitted": RetrainingOverallStatus.RUNNING,
            "running": RetrainingOverallStatus.RUNNING,
            "pending": RetrainingOverallStatus.QUEUED,
            "canceling": RetrainingOverallStatus.CANCELLING,
            "paused": RetrainingOverallStatus.PAUSED,
            "succeeded": RetrainingOverallStatus.SUCCEEDED,
            "rejected": RetrainingOverallStatus.REJECTED,
            "failed": RetrainingOverallStatus.FAILED,
            "cancelled": RetrainingOverallStatus.CANCELLED,
        }
        return mapping.get(value, RetrainingOverallStatus.UNKNOWN)

    @staticmethod
    def _stages(
        overall: RetrainingOverallStatus,
        *,
        observed: ObservabilitySnapshot,
        timestamp: datetime | None,
    ) -> list[StageRecord]:
        stages = (
            (RetrainingStageName.FEEDBACK_PREPARATION, observed.feedback_prepare),
            (RetrainingStageName.FEEDBACK_MERGE, observed.feedback_merge),
            (RetrainingStageName.TRAINING, observed.training_final),
            (RetrainingStageName.EVALUATION, observed.evaluation),
            (RetrainingStageName.PACKAGING, observed.packaging),
            (RetrainingStageName.GATE, observed.gate),
            (RetrainingStageName.PUBLISHING, observed.publishing),
        )
        if overall is RetrainingOverallStatus.SUCCEEDED:
            return [
                StageRecord(
                    name=name,
                    status=StageExecutionStatus.SUCCEEDED,
                    startedAt=None,
                    endedAt=timestamp,
                )
                for name, _ in stages
            ]
        records: list[StageRecord] = []
        terminal_failure_seen = False
        for name, artifact in stages:
            state = StageExecutionStatus.PENDING
            if artifact is not None:
                state = StageExecutionStatus.SUCCEEDED
                if (
                    name is RetrainingStageName.FEEDBACK_PREPARATION
                    and artifact.get("failure_code") == "NO_UNIQUE_FEEDBACK"
                ) or (name is RetrainingStageName.GATE and artifact.get("passed") is False):
                    state = StageExecutionStatus.FAILED
                    terminal_failure_seen = True
            elif name is RetrainingStageName.TRAINING and observed.training_progress:
                state = StageExecutionStatus.RUNNING
            elif terminal_failure_seen or overall in {
                RetrainingOverallStatus.REJECTED,
                RetrainingOverallStatus.FAILED,
                RetrainingOverallStatus.CANCELLED,
            }:
                state = StageExecutionStatus.SKIPPED
            records.append(
                StageRecord(
                    name=name,
                    status=state,
                    startedAt=(artifact or {}).get("started_at"),
                    endedAt=(artifact or {}).get("ended_at") or (artifact or {}).get("updated_at"),
                )
            )
        return records

    @staticmethod
    def _result(
        job: RetrainingJob,
        overall: RetrainingOverallStatus,
        *,
        observed: ObservabilitySnapshot,
    ) -> RetrainingResult | None:
        if overall is RetrainingOverallStatus.SUCCEEDED:
            return RetrainingResult(
                outcome=RetrainingOutcome.PROMOTED,
                candidateModelVersion=job.candidate_model_version or "unknown",
                logicalModelVersion=job.logical_model_version or "0.0",
                gatePassed=True,
                failedGateChecks=[],
                failureCode=None,
                failureStage=None,
                retryable=False,
                message=None,
                approvedAt=(observed.publishing or {}).get("approved_at") or job.ended_at,
            )
        if overall is RetrainingOverallStatus.REJECTED:
            outcome = RetrainingOutcome.REJECTED
            unique_rejection = (
                observed.feedback_prepare
                and observed.feedback_prepare.get("failure_code") == "NO_UNIQUE_FEEDBACK"
            )
            code = (
                FailureCode.NO_UNIQUE_FEEDBACK
                if unique_rejection
                else FailureCode.DEPLOY_GATE_REJECTED
            )
            stage = (
                RetrainingStageName.FEEDBACK_PREPARATION
                if unique_rejection
                else RetrainingStageName.GATE
            )
            retryable = False
        elif overall is RetrainingOverallStatus.CANCELLED:
            outcome = RetrainingOutcome.CANCELLED
            code = FailureCode.PIPELINE_CANCELLED
            stage = None
            retryable = False
        elif overall is RetrainingOverallStatus.FAILED:
            outcome = RetrainingOutcome.FAILED
            code = FailureCode.UNKNOWN_PIPELINE_FAILURE
            stage = None
            retryable = True
        else:
            return None
        return RetrainingResult(
            outcome=outcome,
            candidateModelVersion=job.candidate_model_version or "unknown",
            logicalModelVersion=job.logical_model_version or "0.0",
            gatePassed=False if outcome is RetrainingOutcome.REJECTED else None,
            failedGateChecks=[
                str(check.get("name", "unknown")) if isinstance(check, dict) else str(check)
                for check in (observed.gate or {}).get("failed_checks", [])
            ],
            failureCode=code,
            failureStage=stage,
            retryable=retryable,
            message=(observed.feedback_prepare or {}).get("message")
            or (observed.gate or {}).get("message"),
            approvedAt=None,
        )

    @staticmethod
    def _activation(
        job: RetrainingJob,
        *,
        overall: RetrainingOverallStatus,
        serving_model_version: str | None,
        desired_model_version: str | None,
        desired_load_error: str | None,
        checked_at: datetime,
    ) -> ActivationStatus:
        candidate = job.candidate_model_version
        if overall in {
            RetrainingOverallStatus.REJECTED,
            RetrainingOverallStatus.FAILED,
            RetrainingOverallStatus.CANCELLED,
        }:
            value = ActivationStatusValue.NOT_APPLICABLE
        elif candidate and serving_model_version == candidate:
            value = ActivationStatusValue.ACTIVE
        elif candidate and desired_model_version == candidate and desired_load_error:
            value = ActivationStatusValue.LOAD_FAILED
        elif (
            overall is RetrainingOverallStatus.SUCCEEDED
            and desired_model_version
            and desired_model_version != candidate
        ):
            value = ActivationStatusValue.SUPERSEDED
        else:
            value = ActivationStatusValue.PENDING
        return ActivationStatus(
            status=value,
            desiredModelVersion=desired_model_version,
            servingModelVersion=serving_model_version,
            checkedAt=checked_at,
        )

    @staticmethod
    def _accepted(job: RetrainingJob) -> AcceptedRetraining:
        assert job.pipeline_run_id
        return AcceptedRetraining(
            batch_id=job.batch_id,
            pipeline_run_id=job.pipeline_run_id,
            logical_model_version=job.logical_model_version or "0.0",
            candidate_model_version=job.candidate_model_version or "unknown",
            training_mode=job.training_mode or TrainingMode.MINOR_FINE_TUNE,
            version_policy_mode=(
                job.version_policy_mode or VersionPolicyMode.SUCCESSFUL_HTTP_REQUEST
            ),
        )

    def _read_observability(self, job: RetrainingJob) -> ObservabilitySnapshot:
        if self._observability_reader is None:
            return ObservabilitySnapshot(
                feedback_prepare=None,
                feedback_merge=None,
                training_progress=None,
                training_final=None,
                evaluation=None,
                packaging=None,
                gate=None,
                publishing=None,
                warnings=(),
            )
        return self._observability_reader.read(
            batch_id=job.batch_id,
            candidate_model_version=job.candidate_model_version or "unknown",
        )

    @staticmethod
    def _current_stage(
        overall: RetrainingOverallStatus, observed: ObservabilitySnapshot
    ) -> RetrainingStageName:
        if overall in {
            RetrainingOverallStatus.SUCCEEDED,
            RetrainingOverallStatus.REJECTED,
            RetrainingOverallStatus.FAILED,
            RetrainingOverallStatus.CANCELLED,
        }:
            return RetrainingStageName.COMPLETE
        if observed.training_progress and not observed.training_final:
            return RetrainingStageName.TRAINING
        ordered = (
            (RetrainingStageName.FEEDBACK_PREPARATION, observed.feedback_prepare),
            (RetrainingStageName.FEEDBACK_MERGE, observed.feedback_merge),
            (RetrainingStageName.TRAINING, observed.training_final),
            (RetrainingStageName.EVALUATION, observed.evaluation),
            (RetrainingStageName.PACKAGING, observed.packaging),
            (RetrainingStageName.GATE, observed.gate),
            (RetrainingStageName.PUBLISHING, observed.publishing),
        )
        for name, value in ordered:
            if value is None:
                return name
        return RetrainingStageName.INITIALIZING

    @staticmethod
    def _stage_progress(payload: dict | None) -> StageProgress | None:
        if not payload:
            return None
        current = int(payload.get("optimizer_step", 0))
        total = int(payload.get("total_steps", 0))
        if total < 1:
            return None
        return StageProgress(
            current=current,
            total=total,
            percent=round(current * 100 / total, 2),
            epoch=max(1, int(payload.get("epoch", 1))),
            totalEpochs=max(1, int(payload.get("total_epochs", 1))),
            heartbeatAt=payload.get("updated_at") or datetime.now(UTC),
        )

    @staticmethod
    def _filtering_metrics(payload: dict | None) -> FilteringMetrics | None:
        if not payload:
            return None
        accepted = int(payload.get("accepted_event_count", 0))
        canonical = int(payload.get("training_row_count", 0))
        labels = payload.get("label_distribution", {})
        return FilteringMetrics(
            inputEventCount=int(payload.get("input_event_count", 0)),
            acceptedEventCount=accepted,
            excludedEventCount=int(payload.get("excluded_event_count", 0)),
            collapsedDuplicateEventCount=max(0, accepted - canonical),
            canonicalTrainingRowCount=canonical,
            exclusionDistribution=dict(payload.get("exclusion_distribution", {})),
            labelDistribution={
                "easy": int(labels.get("easy", 0)),
                "medium": int(labels.get("medium", 0)),
                "hard": int(labels.get("hard", 0)),
            },
        )

    @staticmethod
    def _merge_metrics(payload: dict | None) -> MergeMetrics | None:
        if not payload:
            return None
        return MergeMetrics(
            feedbackInputRowCount=int(payload.get("feedback_input_row_count", 0)),
            acceptedFeedbackRowCount=int(payload.get("accepted_feedback_row_count", 0)),
            excludedFeedbackRowCount=int(payload.get("excluded_feedback_row_count", 0)),
            replacedBaseRowCount=int(payload.get("replaced_base_row_count", 0)),
            mergedTrainingRowCount=int(payload.get("merged_training_row_count", 0)),
            decisionDistribution=dict(payload.get("merge_decision_distribution", {})),
        )

    @staticmethod
    def _canonical_batch_digest(request: EscalateRequest) -> str:
        groups: dict[str, list[str]] = defaultdict(list)
        for event in request.events:
            normalized = " ".join(
                re.findall(r"\w+", unicodedata.normalize("NFKC", event.question).casefold())
            )
            groups[normalized].append(event.finalLabel.value)
        rows = []
        for question, labels in sorted(groups.items()):
            distinct = set(labels)
            if not question or len(distinct) != 1:
                continue
            repeat_count = len(labels)
            rows.append(
                {
                    "normalizedQuestion": question,
                    "finalLabel": labels[0],
                    "sampleWeight": min(2.0, 1.0 + (repeat_count - 1) * 0.25),
                }
            )
        canonical = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()

    def _materialize_feedback(
        self,
        request: EscalateRequest,
        validated: ValidatedFeedbackBatch,
        plan: ContinualTrainingPlan,
        candidate_version: str,
    ) -> MaterializedFeedback:
        return self._materializer.materialize(
            request=request,
            validated=validated,
            artifact_key=candidate_version,
            logical_model_version=plan.logical_model_version,
            previous_cycle_events_uri=plan.previous_cycle_events_uri,
            previous_replay_events_uri=plan.previous_replay_events_uri,
        )

    @staticmethod
    def _fingerprint(request: EscalateRequest) -> str:
        """Hash the fields that define *which* batch this is.

        ``requestedAt`` is deliberately excluded. It carries no batch identity,
        and a client retrying after a timeout is very likely to stamp a fresh
        one; including it would turn the exact case idempotency exists for into
        a BATCH_ID_CONFLICT.
        """
        payload = request.model_dump(mode="json", exclude={"requestedAt"})
        payload["events"] = sorted(payload["events"], key=lambda event: event["eventId"])
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()
