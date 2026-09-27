import asyncio
import hmac
import logging
import math
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from typing import Annotated
from uuid import uuid4

from fastapi import Depends, FastAPI, Request, Response, Security, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from llm_router_serving.config import Settings
from llm_router_serving.errors import ApiError, ModelContractError
from llm_router_serving.inference import (
    ClassificationEngine,
    EmbeddingEngine,
    InvalidRoutingInputError,
    RerankEngine,
    TransformersBgeEmbedder,
    TransformersCrossEncoderReranker,
    TransformersModernBertClassifier,
)
from llm_router_serving.model_manager import ModelManager
from llm_router_serving.model_source import (
    CurrentModelPointer,
    GcsPackageStore,
    ModelSourceError,
    read_current_model_pointer,
    resolve_router_model,
)
from llm_router_serving.model_updates import ModelPointerWatcher, ModelPointerWatcherSnapshot
from llm_router_serving.profiles import validate_router_profile
from llm_router_serving.retraining.gcs import (
    BatchValidationError,
    FeedbackBatchValidator,
    FeedbackReplayMaterializer,
    GcsObjectStore,
    ObjectAccessError,
)
from llm_router_serving.retraining.observability import RetrainingObservabilityReader
from llm_router_serving.retraining.pipeline import (
    PipelineAuthorizationError,
    PipelineBackendUnavailable,
    PipelineSubmissionError,
    VertexPipelineClient,
)
from llm_router_serving.retraining.service import EscalationService, RetrainingRunNotFoundError
from llm_router_serving.retraining.store import (
    BatchIdConflictError,
    RetrainingInProgressError,
    RetrainingJobStore,
    VersionPolicyRevisionConflictError,
)
from llm_router_serving.schemas import (
    ClassifyResponse,
    ClassifyV2Response,
    CurrentModelResponse,
    CurrentModelStatus,
    DesiredModel,
    EmbedResponse,
    ErrorResponse,
    EscalateRequest,
    EscalateResponse,
    HealthResponse,
    HealthStatus,
    MessageRequest,
    ModelLoadStatus,
    ModelSource,
    PreviousKnownGood,
    RerankRequest,
    RerankResponse,
    RetrainingStatusResponse,
    ServingModel,
    TrainingMode,
    VersionPolicyResponse,
    VersionPolicyUpdateRequest,
)

logger = logging.getLogger(__name__)

internal_bearer = HTTPBearer(auto_error=False, scheme_name="internalBearer")
developer_bearer = HTTPBearer(auto_error=False, scheme_name="developerBearer")


async def require_internal_auth(
    request: Request,
    credential: Annotated[HTTPAuthorizationCredentials | None, Security(internal_bearer)],
) -> None:
    settings: Settings = request.app.state.settings
    expected_secret = settings.internal_api_token
    if expected_secret is None:
        return
    if credential is None or credential.scheme.lower() != "bearer":
        raise ApiError(status.HTTP_401_UNAUTHORIZED, "UNAUTHORIZED", "authentication required")
    if not hmac.compare_digest(credential.credentials, expected_secret.get_secret_value()):
        raise ApiError(status.HTTP_401_UNAUTHORIZED, "UNAUTHORIZED", "invalid credential")


InternalAuth = Annotated[None, Depends(require_internal_auth)]


async def require_developer_auth(
    request: Request,
    credential: Annotated[HTTPAuthorizationCredentials | None, Security(developer_bearer)],
) -> None:
    settings: Settings = request.app.state.settings
    if not settings.enable_developer_api or settings.developer_api_token is None:
        raise ApiError(status.HTTP_403_FORBIDDEN, "FORBIDDEN", "developer API is disabled")
    if credential is None or credential.scheme.lower() != "bearer":
        raise ApiError(status.HTTP_401_UNAUTHORIZED, "UNAUTHORIZED", "authentication required")
    if not hmac.compare_digest(
        credential.credentials, settings.developer_api_token.get_secret_value()
    ):
        raise ApiError(status.HTTP_401_UNAUTHORIZED, "UNAUTHORIZED", "invalid credential")


DeveloperAuth = Annotated[None, Depends(require_developer_auth)]


def _error_response(request: Request, *, status_code: int, code: str, message: str) -> JSONResponse:
    request_id = getattr(request.state, "request_id", str(uuid4()))
    body = ErrorResponse(error=message, code=code, requestId=request_id)
    return JSONResponse(status_code=status_code, content=body.model_dump())


def _build_model_manager(settings: Settings) -> ModelManager:
    package_store = GcsPackageStore(project=settings.vertex_project)

    def load_embed() -> EmbeddingEngine:
        return TransformersBgeEmbedder.load(
            model_id=settings.embed_model_id,
            revision=settings.embed_model_revision,
            model_version=settings.embed_model_version,
            expected_dimension=settings.embed_expected_dim,
            max_length=settings.embed_max_length,
            device_name=settings.model_device,
            local_files_only=settings.embed_local_files_only,
        )

    def load_router_version(
        package_uri: str,
        model_version: str,
        manifest_sha256: str = "",
    ) -> ClassificationEngine:
        try:
            resolved = resolve_router_model(
                package_uri=package_uri,
                local_path=None,
                model_version=model_version,
                cache_dir=settings.router_model_cache_dir,
                object_store=package_store,
                require_approved=settings.router_model_require_approved,
                expected_manifest_sha256=manifest_sha256 or None,
            )
        except ModelSourceError as exc:
            raise ModelContractError(str(exc)) from exc
        validate_router_profile(settings, resolved)
        return TransformersModernBertClassifier.load(
            model_path=resolved.path,
            model_version=resolved.model_version,
            max_length=resolved.routing_policy.max_length
            if resolved.routing_policy
            else resolved.max_length or settings.router_max_length,
            device_name=settings.model_device,
            routing_policy=resolved.routing_policy,
            package_manifest_sha256=resolved.manifest_sha256,
            input_contract=resolved.input_contract,
        )

    def load_router() -> ClassificationEngine:
        source_errors = settings.router_model_source_errors()
        if source_errors:
            raise ModelContractError(
                "Router model source is not configured: " + "; ".join(source_errors)
            )
        pointer_uri = settings.current_model_pointer_uri()
        if pointer_uri:
            pointer = read_current_model_pointer(
                pointer_uri,
                object_store=package_store,
            )
            return load_router_version(
                pointer.package_uri,
                pointer.model_version,
                pointer.manifest_sha256,
            )
        if settings.router_model_path is None:
            return load_router_version(
                str(settings.router_model_package_uri),
                str(settings.router_model_version),
            )
        resolved = resolve_router_model(
            package_uri=None,
            local_path=settings.router_model_path,
            model_version=str(settings.router_model_version),
            cache_dir=settings.router_model_cache_dir,
            require_approved=settings.router_model_require_approved,
        )
        validate_router_profile(settings, resolved)
        return TransformersModernBertClassifier.load(
            model_path=resolved.path,
            model_version=resolved.model_version,
            max_length=resolved.routing_policy.max_length
            if resolved.routing_policy
            else resolved.max_length or settings.router_max_length,
            device_name=settings.model_device,
            routing_policy=resolved.routing_policy,
            package_manifest_sha256=resolved.manifest_sha256,
            input_contract=resolved.input_contract,
        )

    def load_rerank() -> RerankEngine:
        return TransformersCrossEncoderReranker.load(
            model_id=str(settings.rerank_model_id),
            model_version=settings.rerank_model_version,
            max_length=settings.rerank_max_length,
            device_name=settings.model_device,
            local_files_only=settings.rerank_local_files_only,
        )

    return ModelManager(
        embed_loader=load_embed,
        router_loader=load_router,
        router_version_loader=load_router_version,
        rerank_loader=load_rerank if settings.rerank_model_id else None,
    )


def _build_model_pointer_watcher(
    settings: Settings, model_manager: ModelManager
) -> ModelPointerWatcher | None:
    pointer_uri = settings.current_model_pointer_uri()
    if not settings.router_model_auto_reload or not pointer_uri:
        return None
    pointer_store = GcsPackageStore(project=settings.vertex_project)

    def load_pointer() -> CurrentModelPointer:
        pointer = read_current_model_pointer(pointer_uri, object_store=pointer_store)
        engine = model_manager.classification_engine()
        loaded_digest = getattr(engine, "package_manifest_sha256", None)
        if (
            engine is not None
            and engine.model_version == pointer.model_version
            and loaded_digest is not None
            and loaded_digest != pointer.manifest_sha256
        ):
            raise ModelSourceError(
                "An immutable model version cannot change its package or policy digest"
            )
        return pointer

    return ModelPointerWatcher(
        model_manager=model_manager,
        pointer_loader=load_pointer,
        poll_interval_seconds=settings.router_model_poll_interval_seconds,
    )


def _build_escalation_service(settings: Settings) -> EscalationService | None:
    if settings.retraining_configuration_errors():
        return None
    pipeline_client = VertexPipelineClient(
        project=str(settings.vertex_project),
        location=settings.vertex_location,
        template_uri=str(settings.vertex_pipeline_template_uri),
        pipeline_root=str(settings.vertex_pipeline_root),
        service_account=settings.vertex_pipeline_service_account,
        network=settings.vertex_pipeline_network,
        labels={"component": "feedback-retraining"},
    )
    object_store = GcsObjectStore(project=settings.vertex_project)
    validator = FeedbackBatchValidator(
        supported_schemas=settings.supported_feedback_schemas,
    )
    derived_root = settings.derived_retraining_data_root_uri()
    assert derived_root is not None
    return EscalationService(
        settings=settings,
        validator=validator,
        job_store=RetrainingJobStore(settings.retraining_db_path),
        pipeline_client=pipeline_client,
        materializer=FeedbackReplayMaterializer(
            object_store=object_store,
            root_uri=derived_root,
            rows_per_label=settings.retraining_replay_rows_per_label,
        ),
        observability_reader=(
            RetrainingObservabilityReader(
                object_store=object_store,
                root_uri=str(settings.retraining_observability_root_uri),
            )
            if settings.retraining_observability_root_uri
            else None
        ),
    )


def _current_model_response(
    settings: Settings,
    model_manager: ModelManager,
    watcher: ModelPointerWatcher | None,
) -> CurrentModelResponse:
    manager = model_manager.snapshot()
    watcher_state = (
        watcher.snapshot()
        if watcher is not None
        else ModelPointerWatcherSnapshot(None, False, None, None, None)
    )
    pointer: CurrentModelPointer | None = watcher_state.desired
    serving_version = manager.router_model_version
    if settings.router_model_path is not None:
        source = ModelSource.LOCAL_PATH
    elif settings.current_model_pointer_uri():
        source = ModelSource.CURRENT_POINTER
    else:
        source = ModelSource.FIXED_PACKAGE

    if serving_version is None:
        current_status = CurrentModelStatus.UNAVAILABLE
    elif watcher is None:
        current_status = CurrentModelStatus.STATIC
    elif watcher_state.last_load_error and (
        pointer is None or pointer.model_version != serving_version
    ):
        current_status = CurrentModelStatus.LOAD_FAILED
    elif watcher_state.loading or (pointer and pointer.model_version != serving_version):
        current_status = CurrentModelStatus.LOADING
    else:
        current_status = CurrentModelStatus.IN_SYNC

    desired = None
    if pointer is not None:
        desired = DesiredModel(
            modelVersion=pointer.model_version,
            logicalModelVersion=pointer.logical_model_version,
            trainingMode=(TrainingMode(pointer.training_mode) if pointer.training_mode else None),
            manifestSha256=pointer.manifest_sha256,
            approvedAt=pointer.approved_at,
        )
    serving = None
    if serving_version is not None:
        logical_version = (
            pointer.logical_model_version
            if pointer is not None and pointer.model_version == serving_version
            else None
        )
        serving = ServingModel(
            modelVersion=serving_version,
            logicalModelVersion=logical_version,
            loadedAt=manager.router_loaded_at or watcher_state.last_pointer_check_at,
        )
    previous_version = manager.previous_router_model_version or (
        pointer.previous_model_version if pointer is not None else None
    )
    return CurrentModelResponse(
        status=current_status,
        source=source,
        desired=desired,
        serving=serving,
        previousKnownGood=(
            PreviousKnownGood(modelVersion=previous_version) if previous_version else None
        ),
        lastPointerCheckAt=watcher_state.last_pointer_check_at,
        lastSuccessfulSwapAt=watcher_state.last_successful_swap_at,
        lastLoadError=(
            "router model load failed"
            if watcher_state.last_load_error or manager.router_error
            else None
        ),
    )


def create_app(
    *,
    settings: Settings | None = None,
    model_manager: ModelManager | None = None,
    escalation_service: EscalationService | None = None,
    model_pointer_watcher: ModelPointerWatcher | None = None,
) -> FastAPI:
    settings = settings or Settings()
    model_manager = model_manager or _build_model_manager(settings)
    if escalation_service is None:
        escalation_service = _build_escalation_service(settings)
    if model_pointer_watcher is None:
        model_pointer_watcher = _build_model_pointer_watcher(settings, model_manager)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        if settings.model_num_threads is not None:
            import torch

            # Set before loaders or inference work begin; environment defaults can
            # differ from the effective PyTorch thread count on the serving host.
            torch.set_num_threads(settings.model_num_threads)
        load_task: asyncio.Task[None] | None = None
        watcher_task: asyncio.Task[None] | None = None
        if settings.load_models_on_startup:
            load_task = asyncio.create_task(model_manager.load_all())
            if model_pointer_watcher is not None:

                async def watch_after_initial_load() -> None:
                    await load_task
                    await model_pointer_watcher.run()

                watcher_task = asyncio.create_task(watch_after_initial_load())
        yield
        for task in (watcher_task, load_task):
            if task and not task.done():
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task

    app = FastAPI(
        title=settings.app_name,
        version="1.0.0",
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.model_manager = model_manager
    app.state.escalation_service = escalation_service
    app.state.model_pointer_watcher = model_pointer_watcher

    @app.middleware("http")
    async def request_id_middleware(request: Request, call_next):
        request.state.request_id = request.headers.get("X-Request-ID") or str(uuid4())
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        return response

    @app.exception_handler(ApiError)
    async def api_error_handler(request: Request, exc: ApiError) -> JSONResponse:
        return _error_response(
            request,
            status_code=exc.status_code,
            code=exc.code,
            message=exc.message,
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        logger.info("Request validation failed: %s", exc.errors())
        if request.url.path in {"/router/embed", "/router/classify", "/router/classify/v2"}:
            return _error_response(
                request,
                status_code=status.HTTP_400_BAD_REQUEST,
                code="INVALID_MESSAGE",
                message="message is missing or invalid",
            )
        return _error_response(
            request,
            status_code=422,
            code="REQUEST_VALIDATION_FAILED",
            message="request body does not match the API contract",
        )

    @app.exception_handler(Exception)
    async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled request error", exc_info=exc)
        return _error_response(
            request,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            code="INTERNAL_SERVER_ERROR",
            message="request processing failed",
        )

    def validate_message_length(message: str) -> None:
        if len(message) > settings.max_message_characters:
            raise ApiError(
                status.HTTP_400_BAD_REQUEST,
                "INVALID_MESSAGE",
                f"message exceeds {settings.max_message_characters} characters",
            )

    @app.post(
        "/router/embed",
        response_model=EmbedResponse,
        responses={400: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
    )
    async def embed(payload: MessageRequest, _: InternalAuth) -> EmbedResponse:
        validate_message_length(payload.message)
        engine = model_manager.embedding_engine()
        if engine is None:
            raise ApiError(
                status.HTTP_503_SERVICE_UNAVAILABLE,
                "EMBED_MODEL_NOT_READY",
                "embedding model is not loaded",
            )
        try:
            result = await asyncio.to_thread(engine.embed, payload.message)
        except Exception as exc:
            logger.exception("Embedding inference failed")
            raise ApiError(
                status.HTTP_500_INTERNAL_SERVER_ERROR,
                "EMBEDDING_INFERENCE_FAILED",
                "embedding inference failed",
            ) from exc
        if len(result.vector) != settings.embed_expected_dim or not all(
            math.isfinite(value) for value in result.vector
        ):
            raise ApiError(
                status.HTTP_500_INTERNAL_SERVER_ERROR,
                "EMBEDDING_INFERENCE_FAILED",
                "embedding output violates the serving contract",
            )
        return EmbedResponse(
            embedding=result.vector,
            dim=len(result.vector),
            modelVersion=result.model_version,
        )

    @app.post(
        "/router/classify",
        response_model=ClassifyResponse,
        responses={400: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
    )
    async def classify(payload: MessageRequest, _: InternalAuth) -> ClassifyResponse:
        validate_message_length(payload.message)
        engine = model_manager.classification_engine()
        if engine is None:
            raise ApiError(
                status.HTTP_503_SERVICE_UNAVAILABLE,
                "ROUTER_MODEL_NOT_READY",
                "router model is not loaded",
            )
        if getattr(engine, "requires_v2", False):
            raise ApiError(
                status.HTTP_409_CONFLICT,
                "ROUTER_API_VERSION_REQUIRED",
                "this routing policy requires /router/classify/v2",
            )
        try:
            result = await asyncio.to_thread(engine.classify, payload.message)
        except InvalidRoutingInputError as exc:
            raise ApiError(status.HTTP_400_BAD_REQUEST, "INVALID_MESSAGE", str(exc)) from exc
        except Exception as exc:
            logger.exception("Router inference failed")
            raise ApiError(
                status.HTTP_500_INTERNAL_SERVER_ERROR,
                "ROUTER_INFERENCE_FAILED",
                "router inference failed",
            ) from exc
        return ClassifyResponse(
            tier=result.tier,
            model=result.model,
            confidence=result.confidence,
            modelVersion=result.model_version,
        )

    @app.post(
        "/router/classify/v2",
        response_model=ClassifyV2Response,
        responses={400: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
    )
    async def classify_v2(payload: MessageRequest, _: InternalAuth) -> ClassifyV2Response:
        validate_message_length(payload.message)
        engine = model_manager.classification_engine()
        if engine is None:
            raise ApiError(
                status.HTTP_503_SERVICE_UNAVAILABLE,
                "ROUTER_MODEL_NOT_READY",
                "router model is not loaded",
            )
        if not getattr(engine, "supports_routing_v2", False):
            raise ApiError(
                status.HTTP_503_SERVICE_UNAVAILABLE,
                "ROUTER_POLICY_NOT_CONFIGURED",
                "an explicit service routing policy is not configured",
            )
        try:
            decision = await asyncio.to_thread(engine.classify_v2, payload.message)
        except InvalidRoutingInputError as exc:
            raise ApiError(status.HTTP_400_BAD_REQUEST, "INVALID_MESSAGE", str(exc)) from exc
        except Exception as exc:
            logger.exception("Router v2 inference failed")
            raise ApiError(
                status.HTTP_500_INTERNAL_SERVER_ERROR,
                "ROUTER_INFERENCE_FAILED",
                "router inference failed",
            ) from exc
        return ClassifyV2Response(
            tier=decision.tier_projection,
            model=decision.model,
            modelVersion=engine.model_version,
            objective=decision.objective,
            policyVersion=decision.policy_version,
            policySha256=decision.policy_sha256,
            candidatePoolVersion=decision.candidate_pool_version,
            candidateModelIds=engine.candidate_model_ids,
            action=decision.action,
            scoreSemantics=decision.score_semantics,
            actionScores=decision.action_scores,
            confidence=decision.confidence,
        )

    @app.post(
        "/router/rerank",
        response_model=RerankResponse,
        responses={400: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
    )
    async def rerank(payload: RerankRequest, _: InternalAuth) -> RerankResponse:
        """Score query-candidate pairs. The caller keeps the hit/miss decision.

        The gateway owns the cache and the threshold, so this returns raw scores
        and never a verdict. Scores are positionally aligned with `candidates`.
        """
        validate_message_length(payload.query)
        for candidate in payload.candidates:
            validate_message_length(candidate)
        if len(payload.candidates) > settings.rerank_max_candidates:
            raise ApiError(
                status.HTTP_400_BAD_REQUEST,
                "TOO_MANY_CANDIDATES",
                f"candidates must not exceed {settings.rerank_max_candidates}",
            )

        engine = model_manager.rerank_engine()
        if engine is None:
            raise ApiError(
                status.HTTP_503_SERVICE_UNAVAILABLE,
                "RERANK_MODEL_NOT_READY",
                "reranker model is not loaded",
            )
        try:
            result = await asyncio.to_thread(engine.rerank, payload.query, payload.candidates)
        except ModelContractError as exc:
            # A non-finite score compares false against every threshold and would
            # become a silent cache miss. Surface it so the caller can log why.
            logger.error("Rerank contract violation: %s", exc)
            raise ApiError(
                status.HTTP_500_INTERNAL_SERVER_ERROR,
                "RERANK_INFERENCE_FAILED",
                "reranker output violates the serving contract",
            ) from exc
        except Exception as exc:
            logger.exception("Rerank inference failed")
            raise ApiError(
                status.HTTP_500_INTERNAL_SERVER_ERROR,
                "RERANK_INFERENCE_FAILED",
                "rerank inference failed",
            ) from exc
        return RerankResponse(scores=result.scores, modelVersion=result.model_version)

    @app.get(
        "/health",
        response_model=HealthResponse,
        responses={503: {"model": HealthResponse}},
    )
    async def health(response: Response) -> HealthResponse:
        snapshot = model_manager.snapshot()
        ready = (
            snapshot.embed_status is ModelLoadStatus.LOADED
            and snapshot.router_status is ModelLoadStatus.LOADED
        )
        if not ready:
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        # `ready` deliberately excludes the reranker: it is an optional precision
        # layer and the gateway already falls back to the cosine threshold without it.
        return HealthResponse(
            status=HealthStatus.OK if ready else HealthStatus.NOT_READY,
            embedModel=snapshot.embed_status,
            routerModel=snapshot.router_status,
            rerankModel=snapshot.rerank_status,
        )

    @app.post(
        "/escalate",
        response_model=EscalateResponse,
        status_code=status.HTTP_202_ACCEPTED,
        responses={
            401: {"model": ErrorResponse},
            403: {"model": ErrorResponse},
            409: {"model": ErrorResponse},
            422: {"model": ErrorResponse},
            502: {"model": ErrorResponse},
            503: {"model": ErrorResponse},
        },
    )
    async def escalate(payload: EscalateRequest, _: InternalAuth) -> EscalateResponse:
        engine = model_manager.classification_engine()
        if getattr(engine, "routing_objective", "ordinal") != "ordinal":
            raise ApiError(
                status.HTTP_409_CONFLICT,
                "R0_OBJECTIVE_REQUIRED",
                "finalLabel retraining requires an ordinal model; "
                "non-ordinal models need an explicit conversion release",
            )
        service = app.state.escalation_service
        if service is None:
            missing = ", ".join(settings.retraining_configuration_errors())
            logger.error("Retraining is not configured: %s", missing)
            raise ApiError(
                status.HTTP_503_SERVICE_UNAVAILABLE,
                "PIPELINE_BACKEND_UNAVAILABLE",
                "retraining pipeline is not configured",
            )
        try:
            accepted = await service.submit(payload)
        except BatchValidationError as exc:
            raise ApiError(422, exc.code, exc.message) from exc
        except ObjectAccessError as exc:
            raise ApiError(exc.status_code, exc.code, exc.message) from exc
        except BatchIdConflictError as exc:
            raise ApiError(
                status.HTTP_409_CONFLICT,
                "BATCH_ID_CONFLICT",
                "batchId already exists with different request content",
            ) from exc
        except RetrainingInProgressError as exc:
            raise ApiError(
                status.HTTP_409_CONFLICT,
                "RETRAINING_IN_PROGRESS",
                "another retraining pipeline is in progress",
            ) from exc
        except PipelineAuthorizationError as exc:
            raise ApiError(
                status.HTTP_403_FORBIDDEN,
                "FORBIDDEN",
                "pipeline submission is not authorized",
            ) from exc
        except PipelineBackendUnavailable as exc:
            raise ApiError(
                status.HTTP_503_SERVICE_UNAVAILABLE,
                "PIPELINE_BACKEND_UNAVAILABLE",
                "Vertex AI Pipelines backend is unavailable",
            ) from exc
        except PipelineSubmissionError as exc:
            raise ApiError(
                status.HTTP_502_BAD_GATEWAY,
                "PIPELINE_SUBMISSION_FAILED",
                "Vertex AI rejected the pipeline submission",
            ) from exc
        return EscalateResponse(
            batchId=accepted.batch_id,
            pipelineRunId=accepted.pipeline_run_id,
            logicalModelVersion=accepted.logical_model_version,
            candidateModelVersion=accepted.candidate_model_version,
            trainingMode=accepted.training_mode,
            versionPolicyMode=accepted.version_policy_mode,
            statusUrl=f"/retraining-runs/{accepted.batch_id}",
        )

    @app.get(
        "/retraining-runs/{batchId}",
        response_model=RetrainingStatusResponse,
        responses={
            401: {"model": ErrorResponse},
            404: {"model": ErrorResponse},
            503: {"model": ErrorResponse},
        },
    )
    async def retraining_status(batchId: str, _: InternalAuth) -> RetrainingStatusResponse:
        service = app.state.escalation_service
        if service is None:
            raise ApiError(
                status.HTTP_503_SERVICE_UNAVAILABLE,
                "PIPELINE_STATUS_UNAVAILABLE",
                "retraining status backend is not configured",
            )
        watcher = app.state.model_pointer_watcher
        watcher_state = watcher.snapshot() if watcher is not None else None
        try:
            return await service.status(
                batchId,
                serving_model_version=model_manager.router_model_version(),
                desired_model_version=(
                    watcher_state.desired.model_version
                    if watcher_state is not None and watcher_state.desired is not None
                    else None
                ),
                desired_load_error=(
                    watcher_state.last_load_error if watcher_state is not None else None
                ),
            )
        except RetrainingRunNotFoundError as exc:
            raise ApiError(
                status.HTTP_404_NOT_FOUND,
                "RETRAINING_RUN_NOT_FOUND",
                "no retraining run exists for batchId",
            ) from exc

    @app.get(
        "/router/model/current",
        response_model=CurrentModelResponse,
        responses={401: {"model": ErrorResponse}, 503: {"model": CurrentModelResponse}},
    )
    async def current_router_model(response: Response, _: InternalAuth) -> CurrentModelResponse:
        result = _current_model_response(
            settings,
            model_manager,
            app.state.model_pointer_watcher,
        )
        if result.status is CurrentModelStatus.UNAVAILABLE:
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return result

    @app.get(
        "/developer/retraining/version-policy",
        response_model=VersionPolicyResponse,
        responses={401: {"model": ErrorResponse}, 403: {"model": ErrorResponse}},
    )
    async def get_version_policy(_: DeveloperAuth) -> VersionPolicyResponse:
        service = app.state.escalation_service
        if service is None:
            raise ApiError(
                status.HTTP_503_SERVICE_UNAVAILABLE,
                "PIPELINE_STATUS_UNAVAILABLE",
                "retraining backend is not configured",
            )
        return await asyncio.to_thread(service.version_policy)

    @app.put(
        "/developer/retraining/version-policy",
        response_model=VersionPolicyResponse,
        responses={
            401: {"model": ErrorResponse},
            403: {"model": ErrorResponse},
            409: {"model": ErrorResponse},
            422: {"model": ErrorResponse},
        },
    )
    async def put_version_policy(
        payload: VersionPolicyUpdateRequest,
        _: DeveloperAuth,
    ) -> VersionPolicyResponse:
        service = app.state.escalation_service
        if service is None:
            raise ApiError(
                status.HTTP_503_SERVICE_UNAVAILABLE,
                "PIPELINE_STATUS_UNAVAILABLE",
                "retraining backend is not configured",
            )
        try:
            return await asyncio.to_thread(
                service.schedule_version_policy,
                expected_revision=payload.expectedRevision,
                mode=payload.mode.value,
                actor="developer-api",
                reason=payload.reason,
            )
        except VersionPolicyRevisionConflictError as exc:
            raise ApiError(
                status.HTTP_409_CONFLICT,
                "REVISION_CONFLICT",
                "expectedRevision does not match current policy revision",
            ) from exc

    return app


app = create_app()
