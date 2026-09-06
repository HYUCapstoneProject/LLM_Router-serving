from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from threading import Lock

from llm_router_serving.model_manager import ModelManager
from llm_router_serving.model_source import CurrentModelPointer

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelPointerWatcherSnapshot:
    desired: CurrentModelPointer | None
    loading: bool
    last_pointer_check_at: datetime | None
    last_successful_swap_at: datetime | None
    last_load_error: str | None


class ModelPointerWatcher:
    """Poll an atomic model pointer and hot-swap only verified checkpoints."""

    def __init__(
        self,
        *,
        model_manager: ModelManager,
        pointer_loader: Callable[[], CurrentModelPointer],
        poll_interval_seconds: float,
    ) -> None:
        if poll_interval_seconds <= 0:
            raise ValueError("poll_interval_seconds must be positive")
        self._model_manager = model_manager
        self._pointer_loader = pointer_loader
        self._poll_interval_seconds = poll_interval_seconds
        self._lock = Lock()
        self._desired: CurrentModelPointer | None = None
        self._loading = False
        self._last_pointer_check_at: datetime | None = None
        self._last_successful_swap_at: datetime | None = None
        self._last_load_error: str | None = None

    async def check_once(self) -> bool:
        try:
            pointer = await asyncio.to_thread(self._pointer_loader)
            with self._lock:
                self._desired = pointer
                self._last_pointer_check_at = datetime.now(UTC)
                self._loading = self._model_manager.router_model_version() != pointer.model_version
                self._last_load_error = None
            changed = await self._model_manager.reload_router(
                package_uri=pointer.package_uri,
                model_version=pointer.model_version,
                manifest_sha256=pointer.manifest_sha256,
            )
        except Exception as exc:
            with self._lock:
                self._loading = False
                self._last_load_error = str(exc)
                self._last_pointer_check_at = datetime.now(UTC)
            raise
        with self._lock:
            self._loading = False
            if changed:
                self._last_successful_swap_at = datetime.now(UTC)
        return changed

    def snapshot(self) -> ModelPointerWatcherSnapshot:
        with self._lock:
            return ModelPointerWatcherSnapshot(
                desired=self._desired,
                loading=self._loading,
                last_pointer_check_at=self._last_pointer_check_at,
                last_successful_swap_at=self._last_successful_swap_at,
                last_load_error=self._last_load_error,
            )

    async def run(self) -> None:
        while True:
            try:
                await self.check_once()
            except asyncio.CancelledError:
                raise
            except Exception:
                # Keep serving the last known-good model and retry. A pointer may
                # become visible slightly before every approved object reaches a
                # reader, and a malformed package must never take down inference.
                logger.exception("Automatic router model refresh failed")
            await asyncio.sleep(self._poll_interval_seconds)
