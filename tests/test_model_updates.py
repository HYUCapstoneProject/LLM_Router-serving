from __future__ import annotations

import pytest

from llm_router_serving.inference import ClassificationResult, EmbeddingResult
from llm_router_serving.model_manager import ModelManager
from llm_router_serving.model_source import CurrentModelPointer
from llm_router_serving.model_updates import ModelPointerWatcher
from llm_router_serving.schemas import RoutedModel, Tier


class FakeEmbedder:
    dimension = 1
    model_version = "embed-v1"

    def embed(self, message: str) -> EmbeddingResult:
        return EmbeddingResult(vector=[1.0], model_version=self.model_version)


class FakeClassifier:
    def __init__(self, model_version: str) -> None:
        self.model_version = model_version

    def classify(self, message: str) -> ClassificationResult:
        return ClassificationResult(
            tier=Tier.EASY,
            model=RoutedModel.HAIKU,
            confidence=1.0,
            model_version=self.model_version,
        )


@pytest.mark.asyncio
async def test_pointer_watcher_atomically_swaps_to_a_new_model():
    loaded: list[tuple[str, str, str]] = []

    def load_version(package_uri: str, model_version: str, manifest_sha256: str):
        loaded.append((package_uri, model_version, manifest_sha256))
        return FakeClassifier(model_version)

    manager = ModelManager(
        embed_loader=FakeEmbedder,
        router_loader=lambda: FakeClassifier("router-old"),
        router_version_loader=load_version,
    )
    await manager.load_all()
    pointer = CurrentModelPointer(
        model_version="router-new",
        package_uri="gs://bucket/models/router-new/approved",
        manifest_sha256="a" * 64,
    )
    watcher = ModelPointerWatcher(
        model_manager=manager,
        pointer_loader=lambda: pointer,
        poll_interval_seconds=1,
    )

    assert await watcher.check_once() is True
    assert manager.router_model_version() == "router-new"
    assert loaded == [(pointer.package_uri, pointer.model_version, pointer.manifest_sha256)]
    assert await watcher.check_once() is False
    assert len(loaded) == 1
    snapshot = watcher.snapshot()
    assert snapshot.desired == pointer
    assert snapshot.last_pointer_check_at is not None
    assert snapshot.last_successful_swap_at is not None
    assert snapshot.last_load_error is None


@pytest.mark.asyncio
async def test_failed_reload_keeps_the_last_known_good_model():
    def fail_load(package_uri: str, model_version: str, manifest_sha256: str):
        raise RuntimeError("broken candidate")

    manager = ModelManager(
        embed_loader=FakeEmbedder,
        router_loader=lambda: FakeClassifier("router-old"),
        router_version_loader=fail_load,
    )
    await manager.load_all()
    watcher = ModelPointerWatcher(
        model_manager=manager,
        pointer_loader=lambda: CurrentModelPointer(
            model_version="router-broken",
            package_uri="gs://bucket/models/router-broken/approved",
            manifest_sha256="b" * 64,
        ),
        poll_interval_seconds=1,
    )

    with pytest.raises(RuntimeError, match="broken candidate"):
        await watcher.check_once()
    assert manager.router_model_version() == "router-old"
    assert manager.classification_engine() is not None
    assert watcher.snapshot().last_load_error == "broken candidate"
