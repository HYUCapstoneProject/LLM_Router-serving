import asyncio
import hmac
import logging
import math
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from typing import Annotated
from uuid import uuid4

from fastapi import Depends, FastAPI, Request, Response, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from llm_router_serving.config import Settings
from llm_router_serving.errors import ApiError, ModelContractError
from llm_router_serving.inference import (
    ClassificationEngine,
    EmbeddingEngine,
    RerankEngine,
    TransformersCrossEncoderReranker,
    TransformersBgeEmbedder,
    TransformersModernBertClassifier,
)
from llm_router_serving.model_manager import ModelManager
from llm_router_serving.model_source import ModelSourceError, resolve_router_model
from llm_router_serving.retraining.gcs import (
    BatchValidationError,
    FeedbackBatchValidator,
    GcsObjectStore,
    ObjectAccessError,
)
from llm_router_serving.retraining.pipeline import (
    PipelineAuthorizationError,
    PipelineBackendUnavailable,
    PipelineSubmissionError,
    VertexPipelineClient,
)
from llm_router_serving.retraining.service import EscalationService
from llm_router_serving.retraining.store import (
    BatchIdConflictError,
    RetrainingInProgressError,
    RetrainingJobStore,
)
from llm_router_serving.schemas import (
    ClassifyResponse,
    EmbedResponse,
    ErrorResponse,
    EscalateRequest,
    EscalateResponse,
    HealthResponse,
    HealthStatus,
    MessageRequest,
    ModelLoadStatus,
    RerankRequest,
    RerankResponse,
)

logger = logging.getLogger(__name__)


async def require_internal_auth(request: Request) -> None:
    settings: Settings = request.app.state.settings
    expected_secret = settings.internal_api_token
    if expected_secret is None:
        return
    authorization = request.headers.get("Authorization", "")
    scheme, _, supplied = authorization.partition(" ")
    if scheme.lower() != "bearer" or not supplied:
        raise ApiError(status.HTTP_401_UNAUTHORIZED, "UNAUTHORIZED", "authentication required")
    if not hmac.compare_digest(supplied, expected_secret.get_secret_value()):
        raise ApiError(status.HTTP_401_UNAUTHORIZED, "UNAUTHORIZED", "invalid credential")


InternalAuth = Annotated[None, Depends(require_internal_auth)]


def _error_response(request: Request, *, status_code: int, code: str, message: str) -> JSONResponse:
    request_id = getattr(request.state, "request_id", str(uuid4()))
    body = ErrorResponse(error=message, code=code, requestId=request_id)
    return JSONResponse(status_code=status_code, content=body.model_dump())


def _build_model_manager(settings: Settings) -> ModelManager:
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

    def load_router() -> ClassificationEngine:
        source_errors = settings.router_model_source_errors()
        if source_errors:
            raise ModelContractError(
                "Router model source is not configured: " + "; ".join(source_errors)
            )
        try:
            resolved = resolve_router_model(
                package_uri=settings.router_model_package_uri,
                local_path=settings.router_model_path,
                model_version=str(settings.router_model_version),
                cache_dir=settings.router_model_cache_dir,
                require_approved=settings.router_model_require_approved,
            )
        except ModelSourceError as exc:
            raise ModelContractError(str(exc)) from exc
        return TransformersModernBertClassifier.load(
            model_path=resolved.path,
            model_version=resolved.model_version,
            max_length=settings.router_max_length,
            device_name=settings.model_device,
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
        rerank_loader=load_rerank if settings.rerank_model_id else None,
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
    validator = FeedbackBatchValidator(
        object_store=GcsObjectStore(),
        allowed_buckets=settings.allowed_gcs_buckets,
        supported_schemas=settings.supported_feedback_schemas,
    )
    return EscalationService(
        settings=settings,
        validator=validator,
        job_store=RetrainingJobStore(settings.retraining_db_path),
        pipeline_client=pipeline_client,
    )


def create_app(
    *,
    settings: Settings | None = None,
    model_manager: ModelManager | None = None,
    escalation_service: EscalationService | None = None,
) -> FastAPI:
    settings = settings or Settings()
    model_manager = model_manager or _build_model_manager(settings)
    if escalation_service is None:
        escalation_service = _build_escalation_service(settings)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        load_task: asyncio.Task[None] | None = None
        if settings.load_models_on_startup:
            load_task = asyncio.create_task(model_manager.load_all())
        yield
        if load_task and not load_task.done():
            load_task.cancel()
            with suppress(asyncio.CancelledError):
                await load_task

    app = FastAPI(
        title=settings.app_name,
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.model_manager = model_manager
    app.state.escalation_service = escalation_service

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
        if request.url.path in {"/router/embed", "/router/classify"}:
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
        try:
            result = await asyncio.to_thread(engine.classify, payload.message)
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
        )

    return app


app = create_app()
