from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass
from threading import Lock

from llm_router_serving.inference import ClassificationEngine, EmbeddingEngine, RerankEngine
from llm_router_serving.schemas import ModelLoadStatus, ModelLoadStatusOrDisabled

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelSnapshot:
    embed_status: ModelLoadStatus
    router_status: ModelLoadStatus
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
        rerank_loader: Callable[[], RerankEngine] | None = None,
    ) -> None:
        self._embed_loader = embed_loader
        self._router_loader = router_loader
        # None means the reranker is not configured for this deployment.
        self._rerank_loader = rerank_loader
        self._embed = _ModelSlot()
        self._router = _ModelSlot()
        self._rerank = _ModelSlot()
        if rerank_loader is None:
            self._rerank.status = ModelLoadStatus.ERROR
        self._lock = Lock()

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

    def snapshot(self) -> ModelSnapshot:
        with self._lock:
            return ModelSnapshot(
                embed_status=self._embed.status,
                router_status=self._router.status,
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
