from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from threading import Lock

from llm_router_serving.inference import ClassificationEngine, EmbeddingEngine, RerankEngine
from llm_router_serving.schemas import ModelLoadStatus, ModelLoadStatusOrDisabled

logger = logging.getLogger(__name__)

RouterVersionLoader = Callable[[str, str, str], ClassificationEngine]


@dataclass(frozen=True)
class ModelSnapshot:
    embed_status: ModelLoadStatus
    router_status: ModelLoadStatus
    router_model_version: str | None
    router_loaded_at: datetime | None
    previous_router_model_version: str | None
    router_error: str | None
    rerank_status: ModelLoadStatusOrDisabled = ModelLoadStatusOrDisabled.DISABLED


class _ModelSlot:
    def __init__(self) -> None:
        self.status = ModelLoadStatus.LOADING
        self.value: object | None = None
        self.error: str | None = None


class ModelManager:
    def __init__(
        self,
        *,
        embed_loader: Callable[[], EmbeddingEngine],
        router_loader: Callable[[], ClassificationEngine],
        router_version_loader: RouterVersionLoader | None = None,
        rerank_loader: Callable[[], RerankEngine] | None = None,
    ) -> None:
        self._embed_loader = embed_loader
        self._router_loader = router_loader
        self._router_version_loader = router_version_loader
        # None means the reranker is not configured for this deployment.
        self._rerank_loader = rerank_loader
        self._embed = _ModelSlot()
        self._router = _ModelSlot()
        self._rerank = _ModelSlot()
        if rerank_loader is None:
            self._rerank.status = ModelLoadStatus.ERROR
        self._lock = Lock()
        self._router_reload_lock = asyncio.Lock()
        self._router_loaded_at: datetime | None = None
        self._previous_router_model_version: str | None = None

    @classmethod
    def loaded(
        cls,
        *,
        embed_engine: EmbeddingEngine,
        router_engine: ClassificationEngine,
        rerank_engine: RerankEngine | None = None,
    ) -> ModelManager:
        manager = cls(
            embed_loader=lambda: embed_engine,
            router_loader=lambda: router_engine,
            rerank_loader=(lambda: rerank_engine) if rerank_engine is not None else None,
        )
        with manager._lock:
            manager._embed.status = ModelLoadStatus.LOADED
            manager._embed.value = embed_engine
            manager._router.status = ModelLoadStatus.LOADED
            manager._router.value = router_engine
            manager._router_loaded_at = datetime.now(UTC)
            if rerank_engine is not None:
                manager._rerank.status = ModelLoadStatus.LOADED
                manager._rerank.value = rerank_engine
        return manager

    async def load_all(self) -> None:
        loads = [
            self._load_slot("embed", self._embed_loader),
            self._load_slot("router", self._router_loader),
        ]
        if self._rerank_loader is not None:
            loads.append(self._load_slot("rerank", self._rerank_loader))
        await asyncio.gather(*loads)

    async def _load_slot(self, name: str, loader: Callable[[], object]) -> None:
        slot = {"embed": self._embed, "router": self._router, "rerank": self._rerank}[name]
        with self._lock:
            slot.status = ModelLoadStatus.LOADING
            slot.value = None
            slot.error = None
        try:
            value = await asyncio.to_thread(loader)
        except Exception as exc:  # model libraries expose many loader-specific exceptions
            logger.exception("Failed to load %s model", name)
            with self._lock:
                slot.status = ModelLoadStatus.ERROR
                slot.error = str(exc)
            return
        with self._lock:
            slot.value = value
            slot.status = ModelLoadStatus.LOADED
            if name == "router":
                self._router_loaded_at = datetime.now(UTC)

    def snapshot(self) -> ModelSnapshot:
        with self._lock:
            return ModelSnapshot(
                embed_status=self._embed.status,
                router_status=self._router.status,
                router_model_version=getattr(self._router.value, "model_version", None),
                router_loaded_at=self._router_loaded_at,
                previous_router_model_version=self._previous_router_model_version,
                router_error=self._router.error,
                rerank_status=(
                    ModelLoadStatusOrDisabled.DISABLED
                    if self._rerank_loader is None
                    else ModelLoadStatusOrDisabled(self._rerank.status.value)
                ),
            )

    def embedding_engine(self) -> EmbeddingEngine | None:
        with self._lock:
            if self._embed.status is not ModelLoadStatus.LOADED:
                return None
            return self._embed.value  # type: ignore[return-value]

    def classification_engine(self) -> ClassificationEngine | None:
        with self._lock:
            if self._router.status is not ModelLoadStatus.LOADED:
                return None
            return self._router.value  # type: ignore[return-value]

    def rerank_engine(self) -> RerankEngine | None:
        """None when the reranker is not configured or not loaded yet.

        The caller turns this into a 503 and the gateway fails closed to a cache
        miss, which costs one LLM call rather than serving a wrong cached answer.
        """
        with self._lock:
            if self._rerank_loader is None or self._rerank.status is not ModelLoadStatus.LOADED:
                return None
            return self._rerank.value  # type: ignore[return-value]

    def router_model_version(self) -> str | None:
        engine = self.classification_engine()
        return engine.model_version if engine is not None else None

    async def reload_router(
        self,
        *,
        package_uri: str,
        model_version: str,
        manifest_sha256: str,
    ) -> bool:
        """Load a verified router beside the active one, then atomically swap it.

        In-flight requests retain their reference to the previous engine. A bad
        download or checkpoint therefore leaves the last known-good model active.
        """
        if self._router_version_loader is None:
            raise RuntimeError("router model auto-reload is not configured")
        if self.router_model_version() == model_version:
            return False

        async with self._router_reload_lock:
            if self.router_model_version() == model_version:
                return False
            try:
                engine = await asyncio.to_thread(
                    self._router_version_loader,
                    package_uri,
                    model_version,
                    manifest_sha256,
                )
            except Exception as exc:
                logger.exception("Failed to load router model version %s", model_version)
                with self._lock:
                    self._router.error = str(exc)
                raise
            with self._lock:
                previous_version = getattr(self._router.value, "model_version", None)
                self._router.value = engine
                self._router.status = ModelLoadStatus.LOADED
                self._router.error = None
                self._previous_router_model_version = previous_version
                self._router_loaded_at = datetime.now(UTC)
            logger.info("Router model swapped from %s to %s", previous_version, model_version)
            return True
