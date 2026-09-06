from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from llm_router_serving.model_source import (
    ModelSourceError,
    read_current_model_pointer,
    resolve_router_model,
    verify_package,
)

MODEL_VERSION = "router-modernbert-base-feedback-20260901-a1b2c3d"
PACKAGE_URI = f"gs://llm-router/models/router-modernbert/{MODEL_VERSION}/approved"

FILE_CONTENT = {
    "config.json": '{"id2label": {"0": "easy", "1": "medium", "2": "hard"}}',
    "tokenizer.json": '{"version": "1"}',
    "tokenizer_config.json": '{"model_max_length": 8192}',
    "metrics.json": '{"accuracy": 0.81}',
    "training_config.json": '{"epochs": 3}',
    "model.safetensors": "weights",
}


def build_package(directory: Path, *, gate_status: str = "approved", **manifest_overrides) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, content in FILE_CONTENT.items():
        (directory / name).write_text(content, encoding="utf-8")
        files[name] = hashlib.sha256(content.encode("utf-8")).hexdigest()
    manifest = {
        "schema_version": "model-package-v1",
        "model_version": MODEL_VERSION,
        "gate_status": gate_status,
        "id2label": {"0": "easy", "1": "medium", "2": "hard"},
        "files": files,
        **manifest_overrides,
    }
    (directory / "model_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return directory


class MemoryPackageStore:
    """Serves a local directory as if it were a GCS package prefix."""

    def __init__(self, source: Path, prefix_uri: str = PACKAGE_URI) -> None:
        self._source = source
        self._prefix = prefix_uri.rstrip("/")
        self.downloads = 0

    def list_uris(self, prefix_uri: str) -> list[str]:
        assert prefix_uri.rstrip("/") == self._prefix
        return sorted(f"{self._prefix}/{path.name}" for path in self._source.iterdir())

    def read_bytes(self, uri: str) -> bytes:
        name = uri.rsplit("/", 1)[-1]
        return (self._source / name).read_bytes()

    def download_file(self, source_uri: str, destination: Path) -> None:
        self.downloads += 1
        name = source_uri.rsplit("/", 1)[-1]
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((self._source / name).read_bytes())


def test_verify_package_accepts_an_approved_package(tmp_path):
    package = build_package(tmp_path / "pkg")

    manifest = verify_package(package, expected_model_version=MODEL_VERSION)

    assert manifest["gate_status"] == "approved"


def test_verify_package_rejects_a_candidate_package(tmp_path):
    package = build_package(tmp_path / "pkg", gate_status="candidate")

    with pytest.raises(ModelSourceError, match="gate_status"):
        verify_package(package, expected_model_version=MODEL_VERSION)


def test_verify_package_allows_candidates_when_approval_is_not_required(tmp_path):
    package = build_package(tmp_path / "pkg", gate_status="candidate")

    manifest = verify_package(package, expected_model_version=MODEL_VERSION, require_approved=False)

    assert manifest["gate_status"] == "candidate"


def test_verify_package_rejects_a_version_mismatch(tmp_path):
    package = build_package(tmp_path / "pkg")

    with pytest.raises(ModelSourceError, match="model_version"):
        verify_package(package, expected_model_version="router-other-20260101-abcdef1")


def test_verify_package_rejects_a_tampered_file(tmp_path):
    package = build_package(tmp_path / "pkg")
    (package / "model.safetensors").write_text("tampered", encoding="utf-8")

    with pytest.raises(ModelSourceError, match="SHA-256 mismatch"):
        verify_package(package, expected_model_version=MODEL_VERSION)


def test_verify_package_rejects_wrong_label_order(tmp_path):
    package = build_package(tmp_path / "pkg", id2label={"0": "hard", "1": "medium", "2": "easy"})

    with pytest.raises(ModelSourceError, match="labels"):
        verify_package(package, expected_model_version=MODEL_VERSION)


def test_resolve_downloads_verifies_and_caches_the_package(tmp_path):
    source = build_package(tmp_path / "remote")
    store = MemoryPackageStore(source)
    cache = tmp_path / "cache"

    resolved = resolve_router_model(
        package_uri=PACKAGE_URI,
        local_path=None,
        model_version=MODEL_VERSION,
        cache_dir=cache,
        object_store=store,
    )

    assert resolved.path == cache / MODEL_VERSION
    assert (resolved.path / "model.safetensors").is_file()
    downloads_after_first = store.downloads

    again = resolve_router_model(
        package_uri=PACKAGE_URI,
        local_path=None,
        model_version=MODEL_VERSION,
        cache_dir=cache,
        object_store=store,
    )

    assert again.path == resolved.path
    assert store.downloads == downloads_after_first


def test_resolve_leaves_no_usable_directory_when_verification_fails(tmp_path):
    source = build_package(tmp_path / "remote", gate_status="candidate")
    store = MemoryPackageStore(source)
    cache = tmp_path / "cache"

    with pytest.raises(ModelSourceError):
        resolve_router_model(
            package_uri=PACKAGE_URI,
            local_path=None,
            model_version=MODEL_VERSION,
            cache_dir=cache,
            object_store=store,
        )

    assert not (cache / MODEL_VERSION).exists()


def test_resolve_rejects_both_sources(tmp_path):
    with pytest.raises(ModelSourceError, match="not both"):
        resolve_router_model(
            package_uri=PACKAGE_URI,
            local_path=tmp_path,
            model_version=MODEL_VERSION,
            cache_dir=tmp_path / "cache",
        )


def test_resolve_uses_a_local_directory_without_a_store(tmp_path):
    local = tmp_path / "local-model"
    local.mkdir()

    resolved = resolve_router_model(
        package_uri=None,
        local_path=local,
        model_version=MODEL_VERSION,
        cache_dir=tmp_path / "cache",
    )

    assert resolved.path == local
    assert resolved.package_uri is None


class PointerStore:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def read_bytes(self, uri: str) -> bytes:
        return json.dumps(self.payload).encode("utf-8")

    def list_uris(self, prefix_uri: str) -> list[str]:
        raise AssertionError("not used")

    def download_file(self, source_uri: str, destination: Path) -> None:
        raise AssertionError("not used")


def test_read_current_model_pointer_validates_the_approved_target():
    pointer = read_current_model_pointer(
        "gs://llm-router/models/router-modernbert/CURRENT.json",
        object_store=PointerStore(
            {
                "schema_version": "router-current-model-v1",
                "model_version": MODEL_VERSION,
                "package_uri": PACKAGE_URI,
                "manifest_sha256": "a" * 64,
                "id2label": {"0": "easy", "1": "medium", "2": "hard"},
                "logical_model_version": "2.1",
                "training_mode": "minor_fine_tune",
                "previous_model_version": "router-modernbert-base-20260814-45dcc6d",
                "approved_at": "2026-09-12T09:07:00Z",
            }
        ),
    )

    assert pointer.model_version == MODEL_VERSION
    assert pointer.package_uri == PACKAGE_URI
    assert pointer.manifest_sha256 == "a" * 64
    assert pointer.logical_model_version == "2.1"
    assert pointer.training_mode == "minor_fine_tune"
    assert pointer.previous_model_version == "router-modernbert-base-20260814-45dcc6d"
    assert pointer.approved_at is not None


def test_read_current_model_pointer_rejects_a_package_outside_its_root():
    with pytest.raises(ModelSourceError, match="pointer-root"):
        read_current_model_pointer(
            "gs://llm-router/models/router-modernbert/CURRENT.json",
            object_store=PointerStore(
                {
                    "schema_version": "router-current-model-v1",
                    "model_version": MODEL_VERSION,
                    "package_uri": "gs://other-bucket/models/router/approved",
                    "manifest_sha256": "a" * 64,
                    "id2label": {"0": "easy", "1": "medium", "2": "hard"},
                }
            ),
        )


def test_read_current_model_pointer_rejects_a_package_path_for_another_version():
    with pytest.raises(ModelSourceError, match="pointer-root"):
        read_current_model_pointer(
            "gs://llm-router/models/router-modernbert/CURRENT.json",
            object_store=PointerStore(
                {
                    "schema_version": "router-current-model-v1",
                    "model_version": MODEL_VERSION,
                    "package_uri": (
                        "gs://llm-router/models/router-modernbert/router-other/approved"
                    ),
                    "manifest_sha256": "a" * 64,
                    "id2label": {"0": "easy", "1": "medium", "2": "hard"},
                }
            ),
        )


def test_read_current_model_pointer_rejects_an_unsafe_version():
    with pytest.raises(ModelSourceError, match="not a safe router version"):
        read_current_model_pointer(
            "gs://llm-router/models/router-modernbert/CURRENT.json",
            object_store=PointerStore(
                {
                    "schema_version": "router-current-model-v1",
                    "model_version": "../router-evil",
                    "package_uri": PACKAGE_URI,
                    "manifest_sha256": "a" * 64,
                    "id2label": {"0": "easy", "1": "medium", "2": "hard"},
                }
            ),
        )
