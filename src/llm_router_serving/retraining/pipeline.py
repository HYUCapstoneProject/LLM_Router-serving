from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar, NoReturn, Protocol


class PipelineState(StrEnum):
    ACTIVE = "active"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"


class PipelineAuthorizationError(RuntimeError):
    pass


class PipelineBackendUnavailable(RuntimeError):
    pass


class PipelineSubmissionError(RuntimeError):
    pass


@dataclass(frozen=True)
class PipelineSubmission:
    run_id: str


class PipelineClient(Protocol):
    def submit(self, *, run_name: str, arguments: dict[str, object]) -> PipelineSubmission: ...

    def state(self, run_id: str) -> PipelineState: ...

    def find_run(self, run_name: str) -> str | None: ...


_JOB_ID_UNSAFE = re.compile(r"[^a-z0-9-]")
_JOB_ID_MAX_LENGTH = 128


def vertex_job_id(run_name: str) -> str:
    """Derive a deterministic Vertex PipelineJob ID from a run name.

    Vertex requires ``[a-z0-9][-a-z0-9]{0,127}`` and enforces uniqueness per
    project and location, so the same batch always maps to the same job ID and a
    duplicate submission fails server-side with ``AlreadyExists``. Lower-casing
    and character replacement are lossy, so a digest of the original run name is
    appended to keep distinct batches distinct.
    """
    digest = hashlib.sha256(run_name.encode("utf-8")).hexdigest()[:8]
    normalized = _JOB_ID_UNSAFE.sub("-", run_name.lower()).strip("-")
    suffix = f"-{digest}"
    head = normalized or "pipeline"
    return (head[: _JOB_ID_MAX_LENGTH - len(suffix)].rstrip("-") + suffix).lstrip("-")


class VertexPipelineClient:
    """Submit and observe KFP v2 pipelines on managed Vertex AI Pipelines.

    The compiled IR lives in GCS, not in the serving image, so the template URI
    is a ``gs://`` path. Submission is asynchronous: ``submit()`` returns as soon
    as Vertex accepts the job, and the retraining run itself can take hours.
    """

    _TERMINAL_STATES: ClassVar[dict[str, PipelineState]] = {
        "PIPELINE_STATE_SUCCEEDED": PipelineState.SUCCEEDED,
        "PIPELINE_STATE_FAILED": PipelineState.FAILED,
        "PIPELINE_STATE_CANCELLED": PipelineState.CANCELLED,
    }
    _ACTIVE_STATES: ClassVar[frozenset[str]] = frozenset(
        {
            "PIPELINE_STATE_QUEUED",
            "PIPELINE_STATE_PENDING",
            "PIPELINE_STATE_RUNNING",
            "PIPELINE_STATE_CANCELLING",
            "PIPELINE_STATE_PAUSED",
        }
    )

    def __init__(
        self,
        *,
        project: str,
        location: str,
        template_uri: str,
        pipeline_root: str,
        service_account: str | None = None,
        network: str | None = None,
        labels: dict[str, str] | None = None,
    ) -> None:
        self._project = project
        self._location = location
        self._template_uri = template_uri
        self._pipeline_root = pipeline_root
        self._service_account = service_account
        self._network = network
        self._labels = dict(labels or {})

    def submit(self, *, run_name: str, arguments: dict[str, object]) -> PipelineSubmission:
        from google.cloud import aiplatform

        job_id = vertex_job_id(run_name)
        try:
            job = aiplatform.PipelineJob(
                display_name=run_name[:128],
                template_path=self._template_uri,
                pipeline_root=self._pipeline_root,
                job_id=job_id,
                parameter_values=dict(arguments),
                enable_caching=False,
                labels=self._labels or None,
                project=self._project,
                location=self._location,
            )
        except Exception as exc:
            self._raise_mapped(exc, submission=True)
        try:
            job.submit(
                service_account=self._service_account or None,
                network=self._network or None,
            )
        except Exception as exc:
            # The job ID is derived from the batch ID, so AlreadyExists means this
            # exact batch was already submitted and must not start a second run.
            if self._is_already_exists(exc):
                return PipelineSubmission(run_id=self._resource_name(job_id))
            self._raise_mapped(exc, submission=True)
        run_id = str(getattr(job, "resource_name", "") or "") or self._resource_name(job_id)
        return PipelineSubmission(run_id=run_id)

    def find_run(self, run_name: str) -> str | None:
        """Return the resource name of an already-submitted run, or None.

        The job ID is derived deterministically from the run name, so this
        answers "did this batch ever reach Vertex?" without any local record.
        It is how a lease left behind by a crashed submission gets resolved.
        """
        from google.api_core.exceptions import NotFound
        from google.cloud import aiplatform

        resource_name = self._resource_name(vertex_job_id(run_name))
        try:
            aiplatform.PipelineJob.get(
                resource_name=resource_name,
                project=self._project,
                location=self._location,
            )
        except NotFound:
            return None
        except Exception as exc:
            self._raise_mapped(exc, submission=False)
        return resource_name

    def state(self, run_id: str) -> PipelineState:
        from google.cloud import aiplatform

        try:
            job = aiplatform.PipelineJob.get(
                resource_name=run_id,
                project=self._project,
                location=self._location,
            )
        except Exception as exc:
            self._raise_mapped(exc, submission=False)
        raw_state = str(getattr(job, "state", "") or "").split(".")[-1].upper()
        if raw_state in self._TERMINAL_STATES:
            return self._TERMINAL_STATES[raw_state]
        if raw_state in self._ACTIVE_STATES:
            return PipelineState.ACTIVE
        return PipelineState.UNKNOWN

    def _resource_name(self, job_id: str) -> str:
        return f"projects/{self._project}/locations/{self._location}/pipelineJobs/{job_id}"

    @staticmethod
    def _is_already_exists(exc: Exception) -> bool:
        try:
            from google.api_core.exceptions import AlreadyExists
        except ImportError:  # pragma: no cover - google libraries are a runtime dependency
            return False
        return isinstance(exc, AlreadyExists) or getattr(exc, "code", None) == 409

    @staticmethod
    def _raise_mapped(exc: Exception, *, submission: bool) -> NoReturn:
        try:
            from google.api_core.exceptions import (
                Forbidden,
                InvalidArgument,
                NotFound,
                PermissionDenied,
                Unauthenticated,
            )
        except ImportError:  # pragma: no cover - google libraries are a runtime dependency
            Forbidden = PermissionDenied = Unauthenticated = InvalidArgument = NotFound = ()

        if isinstance(exc, (Forbidden, PermissionDenied, Unauthenticated)):
            raise PipelineAuthorizationError(str(exc)) from exc
        if getattr(exc, "code", None) in {401, 403}:
            raise PipelineAuthorizationError(str(exc)) from exc
        if isinstance(exc, (InvalidArgument, NotFound)):
            raise PipelineSubmissionError(str(exc)) from exc
        if isinstance(exc, (ConnectionError, TimeoutError, OSError)):
            raise PipelineBackendUnavailable(str(exc)) from exc
        if submission:
            raise PipelineSubmissionError(str(exc)) from exc
        raise PipelineBackendUnavailable(str(exc)) from exc
