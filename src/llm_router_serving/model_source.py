"""Resolve the router model directory that this process should serve.

Model delivery is pull-based. The Vertex AI pipeline publishes a package to GCS
and a human approves it; deployment then passes the approved package URI and
version to this process. Startup downloads and verifies the package, so a rolling
restart is what swaps the serving model. There is no runtime endpoint that
replaces a loaded model.

Verification mirrors the ML-side ``model-package-v1`` contract: the manifest must
be approved, its schema and labels must match, and every listed file must match
its recorded SHA-256 before the checkpoint is handed to Transformers.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Protocol
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

MANIFEST_FILENAME = "model_manifest.json"
PACKAGE_SCHEMA_VERSION = "model-package-v1"
EXPECTED_ID2LABEL = {"0": "easy", "1": "medium", "2": "hard"}
REQUIRED_PACKAGE_FILES = frozenset(
    {
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "metrics.json",
        "training_config.json",
    }
)
_READY_MARKER = ".package_verified"


class ModelSourceError(RuntimeError):
    """Raised when the configured model package violates the delivery contract."""


@dataclass(frozen=True)
class ResolvedModel:
    path: Path
    model_version: str
    package_uri: str | None


class PackageObjectStore(Protocol):
    def list_uris(self, prefix_uri: str) -> list[str]: ...

    def download_file(self, source_uri: str, destination: Path) -> None: ...


def split_gcs_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(str(uri))
    if parsed.scheme != "gs" or not parsed.netloc:
        raise ModelSourceError(f"Not a GCS URI: {uri}")
    return parsed.netloc, parsed.path.strip("/")


class GcsPackageStore:
    def __init__(self, *, project: str | None = None, client: object | None = None) -> None:
        self._project = project
        self._client = client

    def _get_client(self):
        if self._client is None:
            from google.cloud import storage

            self._client = storage.Client(project=self._project)
        return self._client

    def list_uris(self, prefix_uri: str) -> list[str]:
        bucket_name, prefix = split_gcs_uri(prefix_uri)
        normalized = prefix.rstrip("/") + "/" if prefix else ""
        blobs = self._get_client().list_blobs(bucket_name, prefix=normalized)
        return sorted(f"gs://{bucket_name}/{blob.name}" for blob in blobs)

    def download_file(self, source_uri: str, destination: Path) -> None:
        bucket_name, object_name = split_gcs_uri(source_uri)
        destination.parent.mkdir(parents=True, exist_ok=True)
        blob = self._get_client().bucket(bucket_name).blob(object_name)
        blob.download_to_filename(str(destination))


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def verify_package(
    package_dir: Path,
    *,
    expected_model_version: str,
    require_approved: bool = True,
) -> dict:
    """Verify a downloaded package against its own manifest.

    Returns the manifest. Raises ``ModelSourceError`` on any mismatch so that a
    corrupted or unapproved package never reaches the inference engine.
    """
    manifest_path = package_dir / MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise ModelSourceError(f"Model package has no {MANIFEST_FILENAME}: {package_dir}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ModelSourceError(f"{MANIFEST_FILENAME} is not valid JSON") from exc
    if not isinstance(manifest, dict):
        raise ModelSourceError(f"{MANIFEST_FILENAME} must contain a JSON object")

    if manifest.get("schema_version") != PACKAGE_SCHEMA_VERSION:
        raise ModelSourceError(f"Unsupported package schema: {manifest.get('schema_version')!r}")
    if manifest.get("id2label") != EXPECTED_ID2LABEL:
        raise ModelSourceError("Package labels must be easy=0, medium=1, hard=2")
    if manifest.get("model_version") != expected_model_version:
        raise ModelSourceError(
            f"Package model_version is {manifest.get('model_version')!r}, "
            f"expected {expected_model_version!r}"
        )
    gate_status = manifest.get("gate_status")
    if require_approved and gate_status != "approved":
        raise ModelSourceError(
            f"Refusing to serve a package with gate_status={gate_status!r}; "
            "only approved packages may be served"
        )

    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ModelSourceError(f"{MANIFEST_FILENAME} files must be a non-empty object")
    missing_required = sorted(REQUIRED_PACKAGE_FILES - set(files))
    if missing_required:
        raise ModelSourceError(f"Package manifest is missing required files: {missing_required}")
    if not any(name in files for name in ("model.safetensors", "pytorch_model.bin")):
        raise ModelSourceError("Package manifest lists no model weight file")

    for relative_name, expected_sha256 in sorted(files.items()):
        name = PurePosixPath(str(relative_name))
        if name.is_absolute() or ".." in name.parts or len(name.parts) != 1:
            raise ModelSourceError(f"Package file must be a top-level name: {relative_name!r}")
        file_path = package_dir / str(relative_name)
        if not file_path.is_file():
            raise ModelSourceError(f"Package file not found: {relative_name}")
        actual = _sha256_file(file_path)
        expected = str(expected_sha256).removeprefix("sha256:").lower()
        if actual != expected:
            raise ModelSourceError(f"SHA-256 mismatch for {relative_name}")
    return manifest


def download_package(
    package_uri: str,
    destination: Path,
    *,
    object_store: PackageObjectStore,
) -> list[str]:
    """Download every top-level object under a package prefix."""
    _, prefix = split_gcs_uri(package_uri)
    normalized = prefix.rstrip("/") + "/" if prefix else ""
    destination.mkdir(parents=True, exist_ok=True)

    names: list[str] = []
    for uri in object_store.list_uris(package_uri):
        _, object_name = split_gcs_uri(uri)
        relative = object_name[len(normalized) :] if object_name.startswith(normalized) else ""
        if not relative or "/" in relative:
            # Model packages are flat by contract.
            continue
        object_store.download_file(uri, destination / relative)
        names.append(relative)
    if not names:
        raise ModelSourceError(f"No package objects found under {package_uri}")
    return sorted(names)


def resolve_router_model(
    *,
    package_uri: str | None,
    local_path: Path | None,
    model_version: str,
    cache_dir: Path,
    object_store: PackageObjectStore | None = None,
    require_approved: bool = True,
) -> ResolvedModel:
    """Return the local directory holding the verified router model.

    A local path is used as-is for development. A ``gs://`` package URI is
    downloaded into ``cache_dir/<model_version>`` and verified. A previously
    downloaded and verified version is reused, which keeps container restarts
    cheap without weakening the checksum guarantee.
    """
    if local_path is not None and package_uri is not None:
        raise ModelSourceError(
            "Configure either a local router model path or a GCS package URI, not both"
        )
    if local_path is not None:
        if not local_path.is_dir():
            raise ModelSourceError(f"Router model directory does not exist: {local_path}")
        return ResolvedModel(path=local_path, model_version=model_version, package_uri=None)
    if package_uri is None:
        raise ModelSourceError("No router model path or package URI is configured")

    target = Path(cache_dir) / model_version
    marker = target / _READY_MARKER
    if marker.is_file() and marker.read_text(encoding="utf-8").strip() == package_uri:
        logger.info("Reusing cached router model package %s", model_version)
        verify_package(
            target,
            expected_model_version=model_version,
            require_approved=require_approved,
        )
        return ResolvedModel(path=target, model_version=model_version, package_uri=package_uri)

    store = object_store or GcsPackageStore()
    staging = Path(cache_dir) / f".staging-{model_version}"
    if staging.exists():
        shutil.rmtree(staging)
    logger.info("Downloading router model package %s from %s", model_version, package_uri)
    download_package(package_uri, staging, object_store=store)
    verify_package(
        staging,
        expected_model_version=model_version,
        require_approved=require_approved,
    )

    if target.exists():
        shutil.rmtree(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    # Verify first, publish second: the marker is only written once the staged
    # copy is proven good and moved into place.
    staging.replace(target)
    marker.write_text(package_uri, encoding="utf-8")
    logger.info("Router model package %s is ready at %s", model_version, target)
    return ResolvedModel(path=target, model_version=model_version, package_uri=package_uri)
