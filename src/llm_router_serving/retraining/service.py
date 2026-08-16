from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime

from llm_router_serving.config import Settings
from llm_router_serving.retraining.gcs import FeedbackBatchValidator
from llm_router_serving.retraining.pipeline import PipelineClient, PipelineState
from llm_router_serving.retraining.store import (
    ABANDONED_STATUS,
    RetrainingJob,
    RetrainingJobStore,
)
from llm_router_serving.schemas import EscalateRequest

logger = logging.getLogger(__name__)

# Identity of a batch, for the idempotency key. See `_fingerprint`.
_FINGERPRINT_FIELDS = (
    "batchId",
    "schemaVersion",
    "manifestUri",
    "eventsUri",
    "eventCount",
)


@dataclass(frozen=True)
class AcceptedRetraining:
    batch_id: str
    pipeline_run_id: str


class EscalationService:
    def __init__(
        self,
        *,
        settings: Settings,
        validator: FeedbackBatchValidator,
        job_store: RetrainingJobStore,
        pipeline_client: PipelineClient,
    ) -> None:
        self._settings = settings
        self._validator = validator
        self._job_store = job_store
        self._pipeline_client = pipeline_client

    async def submit(self, request: EscalateRequest) -> AcceptedRetraining:
        fingerprint = self._fingerprint(request)
        existing = await asyncio.to_thread(self._job_store.get, request.batchId)
        if existing and existing.request_fingerprint == fingerprint and existing.pipeline_run_id:
            return AcceptedRetraining(request.batchId, existing.pipeline_run_id)

        await asyncio.to_thread(self._validator.validate, request)
        await self._reconcile_active_jobs()
        claimed = await asyncio.to_thread(
            self._job_store.claim,
            batch_id=request.batchId,
            request_fingerprint=fingerprint,
        )
        if claimed and claimed.pipeline_run_id:
            return AcceptedRetraining(request.batchId, claimed.pipeline_run_id)

        arguments = self._pipeline_arguments(request)
        run_name = self._run_name(request.batchId)
        try:
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
        return AcceptedRetraining(request.batchId, submission.run_id)

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
        if state is PipelineState.TERMINAL:
            await asyncio.to_thread(
                self._job_store.mark_status,
                batch_id=job.batch_id,
                status="terminal",
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

    def _is_stale(self, job: RetrainingJob) -> bool:
        if job.updated_at is None:
            return True
        age = (datetime.now(UTC) - job.updated_at).total_seconds()
        return age >= self._settings.retraining_stale_submission_seconds

    @staticmethod
    def _run_name(batch_id: str) -> str:
        return f"retrain-{batch_id}"

    def _pipeline_arguments(self, request: EscalateRequest) -> dict[str, str]:
        settings = self._settings
        candidate_version = self.candidate_model_version(request)
        arguments = {
            "base_training_dataset_uri": str(settings.base_training_dataset_uri),
            "feedback_manifest_uri": request.manifestUri,
            "feedback_events_uri": request.eventsUri,
            "feedback_preprocessing_config_uri": str(settings.feedback_preprocessing_config_uri),
            "training_config_uri": str(settings.training_config_uri),
            "routerarena_dataset_uri": str(settings.routerarena_dataset_uri),
            "evaluation_config_uri": str(settings.evaluation_config_uri),
            "package_metadata_uri": str(settings.package_metadata_uri),
            "baseline_metrics_uri": str(settings.baseline_metrics_uri),
            "gate_rules_uri": str(settings.gate_rules_uri),
            "candidate_model_version": candidate_version,
            "baseline_model_version": str(settings.router_model_version),
        }
        # Without these the compiled IR falls back to its own defaults, which
        # would silently run the CustomJob as the default compute service account.
        optional = {
            "model_publish_root_uri": settings.model_publish_root_uri,
            "training_service_account": settings.vertex_training_service_account,
            "training_network": settings.vertex_pipeline_network,
            "training_base_output_directory": settings.vertex_training_output_directory,
        }
        arguments.update({key: value for key, value in optional.items() if value})
        return arguments

    def candidate_model_version(self, request: EscalateRequest) -> str:
        requested_date = request.requestedAt.astimezone(UTC).strftime("%Y%m%d")
        return (
            f"router-modernbert-base-feedback-{requested_date}-"
            f"{str(self._settings.model_code_commit)[:7]}"
        )

    @staticmethod
    def _fingerprint(request: EscalateRequest) -> str:
        """Hash the fields that define *which* batch this is.

        ``requestedAt`` is deliberately excluded. It carries no batch identity,
        and a client retrying after a timeout is very likely to stamp a fresh
        one; including it would turn the exact case idempotency exists for into
        a BATCH_ID_CONFLICT.
        """
        canonical = json.dumps(
            request.model_dump(mode="json", include=set(_FINGERPRINT_FIELDS)),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()
