"""Resolve the router model directory that this process should serve.

Model delivery is pull-based. The Vertex AI pipeline publishes and automatically
approves a gate-passed package, then atomically updates ``CURRENT.json``. The
serving process reads that pointer at startup and polls it while running. A new
package is downloaded and verified before the in-memory classifier is swapped.

Verification mirrors the ML-side ``model-package-v1`` contract: the manifest must
be approved, its schema and labels must match, and every listed file must match
its recorded SHA-256 before the checkpoint is handed to Transformers.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Protocol
from urllib.parse import urlparse

from llm_router_serving.routing_policy import RoutingPolicy, RoutingPolicyError, load_routing_policy

logger = logging.getLogger(__name__)

MANIFEST_FILENAME = "model_manifest.json"
PACKAGE_SCHEMA_VERSION = "model-package-v1"
POLICY_PACKAGE_SCHEMA_VERSION = "model-package-v2"
CURRENT_POINTER_SCHEMA_VERSION = "router-current-model-v1"
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
    routing_policy: RoutingPolicy | None = None
    manifest_sha256: str | None = None


@dataclass(frozen=True)
class CurrentModelPointer:
    model_version: str
    package_uri: str
    manifest_sha256: str
    logical_model_version: str | None = None
    training_mode: str | None = None
    previous_model_version: str | None = None
    approved_at: datetime | None = None


class PackageObjectStore(Protocol):
    def read_bytes(self, uri: str) -> bytes: ...

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

    def read_bytes(self, uri: str) -> bytes:
        bucket_name, object_name = split_gcs_uri(uri)
        return (
            self._get_client()
            .bucket(bucket_name)
            .blob(object_name)
            .download_as_bytes(checksum="auto")
        )

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


def read_current_model_pointer(
    pointer_uri: str,
    *,
    object_store: PackageObjectStore | None = None,
) -> CurrentModelPointer:
    """Read and validate the atomic pointer written by the promotion pipeline."""
    bucket, object_name = split_gcs_uri(pointer_uri)
    if not object_name.endswith("/CURRENT.json") and object_name != "CURRENT.json":
        raise ModelSourceError("Current model pointer URI must end with CURRENT.json")
    store = object_store or GcsPackageStore()
    try:
        payload = json.loads(store.read_bytes(pointer_uri))
    except (json.JSONDecodeError, UnicodeDecodeError, TypeError) as exc:
        raise ModelSourceError("CURRENT.json is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise ModelSourceError("CURRENT.json must contain a JSON object")
    if payload.get("schema_version") != CURRENT_POINTER_SCHEMA_VERSION:
        raise ModelSourceError(
            f"Unsupported CURRENT.json schema: {payload.get('schema_version')!r}"
        )

    model_version = str(payload.get("model_version", "")).strip()
    package_uri = str(payload.get("package_uri", "")).rstrip("/")
    manifest_sha256 = str(payload.get("manifest_sha256", "")).removeprefix("sha256:").lower()
    if not re.fullmatch(r"router-[a-z0-9._-]+", model_version):
        raise ModelSourceError("CURRENT.json model_version is not a safe router version")
    package_bucket, package_prefix = split_gcs_uri(package_uri)
    pointer_root = object_name.rsplit("/", 1)[0] if "/" in object_name else ""
    expected_package_prefix = "/".join(
        part for part in (pointer_root.rstrip("/"), model_version, "approved") if part
    )
    if package_bucket != bucket or package_prefix != expected_package_prefix:
        raise ModelSourceError(
            "CURRENT.json package_uri must be <pointer-root>/<model_version>/approved"
        )
    if payload.get("id2label") != EXPECTED_ID2LABEL:
        raise ModelSourceError("CURRENT.json labels must be easy=0, medium=1, hard=2")
    if len(manifest_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in manifest_sha256
    ):
        raise ModelSourceError("CURRENT.json manifest_sha256 must be a SHA-256 digest")
    logical_model_version = _optional_string(payload.get("logical_model_version"))
    if logical_model_version and not re.fullmatch(r"[0-9]+[.][0-9]+", logical_model_version):
        raise ModelSourceError("CURRENT.json logical_model_version must be <major>.<minor>")
    training_mode = _optional_string(payload.get("training_mode"))
    if training_mode not in {None, "minor_fine_tune", "major_consolidation", "pristine_rebase"}:
        # The original qualification pointer used `initial_training`; expose it
        # as unknown rather than making an otherwise valid baseline unloadable.
        training_mode = None
    previous_model_version = _optional_string(payload.get("previous_model_version"))
    if previous_model_version and not re.fullmatch(r"router-[a-z0-9._-]+", previous_model_version):
        raise ModelSourceError("CURRENT.json previous_model_version is not safe")
    approved_at = _optional_datetime(payload.get("approved_at"))
    return CurrentModelPointer(
        model_version=model_version,
        package_uri=package_uri,
        manifest_sha256=manifest_sha256,
        logical_model_version=logical_model_version,
        training_mode=training_mode,
        previous_model_version=previous_model_version,
        approved_at=approved_at,
    )


def _optional_string(value: object) -> str | None:
    text = str(value or "").strip()
    return text or None


def _optional_datetime(value: object) -> datetime | None:
    text = _optional_string(value)
    if text is None:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ModelSourceError("CURRENT.json approved_at is not a valid timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ModelSourceError("CURRENT.json approved_at must include a timezone")
    return parsed.astimezone(UTC)


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

    package_schema = manifest.get("schema_version")
    if package_schema not in {PACKAGE_SCHEMA_VERSION, POLICY_PACKAGE_SCHEMA_VERSION}:
        raise ModelSourceError(f"Unsupported package schema: {manifest.get('schema_version')!r}")
    if package_schema == PACKAGE_SCHEMA_VERSION and manifest.get("id2label") != EXPECTED_ID2LABEL:
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
    required_files = REQUIRED_PACKAGE_FILES | (
        {"routing_policy.json"} if package_schema == POLICY_PACKAGE_SCHEMA_VERSION else set()
    )
    missing_required = sorted(required_files - set(files))
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
    try:
        model_config = json.loads((package_dir / "config.json").read_text())
        if package_schema == POLICY_PACKAGE_SCHEMA_VERSION:
            policy = _read_package_policy(package_dir, manifest)
            expected_labels = {
                str(index): label for index, label in enumerate(policy.output_label_order)
            }
            expected_label_ids = {label: int(index) for index, label in expected_labels.items()}
            training_config = json.loads((package_dir / "training_config.json").read_text())
            if (
                manifest.get("objective") != policy.objective
                or model_config.get("router_objective") != policy.objective
                or training_config.get("objective_method", "ordinal") != policy.objective
            ):
                raise ModelSourceError(
                    "Package, checkpoint, training and policy objectives must match"
                )
            if (
                manifest.get("id2label") != expected_labels
                or model_config.get("id2label") != expected_labels
                or model_config.get("label2id") != expected_label_ids
            ):
                raise ModelSourceError("Policy package output labels differ from its objective")
            if (
                manifest.get("input_contract") != policy.input_contract
                or manifest.get("tokenizer_max_length") != policy.max_length
                or training_config.get("max_length") != policy.max_length
            ):
                raise ModelSourceError(
                    "Policy package requires matching input contract and 512-token limit"
                )
        elif (
            model_config.get("router_objective", "ordinal") != "ordinal"
            or manifest.get("objective", "ordinal") != "ordinal"
        ):
            raise ModelSourceError(
                "Non-ordinal checkpoints cannot use the legacy v1 package contract"
            )
    except (ValueError, TypeError, AttributeError) as exc:
        raise ModelSourceError(f"Invalid router package contract: {exc}") from exc
    return manifest


def _read_package_policy(package_dir: Path, manifest: dict) -> RoutingPolicy:
    if manifest.get("routing_policy_file") != "routing_policy.json":
        raise ModelSourceError("Policy package must name routing_policy.json explicitly")
    expected = manifest.get("routing_policy_sha256")
    if not expected or manifest.get("files", {}).get("routing_policy.json") != expected:
        raise ModelSourceError("Policy file digest must match the package file manifest")
    try:
        return load_routing_policy(
            package_dir / "routing_policy.json", expected_file_sha256=expected
        )
    except (RoutingPolicyError, OSError) as exc:
        raise ModelSourceError(str(exc)) from exc


def _resolved_package(
    path: Path, model_version: str, package_uri: str | None, manifest: dict
) -> ResolvedModel:
    policy = (
        _read_package_policy(path, manifest)
        if manifest.get("schema_version") == POLICY_PACKAGE_SCHEMA_VERSION
        else None
    )
    return ResolvedModel(
        path=path,
        model_version=model_version,
        package_uri=package_uri,
        routing_policy=policy,
        manifest_sha256=_sha256_file(path / MANIFEST_FILENAME),
    )


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
    expected_manifest_sha256: str | None = None,
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
        if (local_path / MANIFEST_FILENAME).is_file():
            manifest = verify_package(
                local_path, expected_model_version=model_version, require_approved=require_approved
            )
            _verify_expected_manifest_sha256(local_path, expected_manifest_sha256)
            return _resolved_package(local_path, model_version, None, manifest)
        return ResolvedModel(path=local_path, model_version=model_version, package_uri=None)
    if package_uri is None:
        raise ModelSourceError("No router model path or package URI is configured")

    target = Path(cache_dir) / model_version
    marker = target / _READY_MARKER
    if marker.is_file() and marker.read_text(encoding="utf-8").strip() == package_uri:
        logger.info("Reusing cached router model package %s", model_version)
        manifest = verify_package(
            target,
            expected_model_version=model_version,
            require_approved=require_approved,
        )
        _verify_expected_manifest_sha256(target, expected_manifest_sha256)
        return _resolved_package(target, model_version, package_uri, manifest)

    store = object_store or GcsPackageStore()
    staging = Path(cache_dir) / f".staging-{model_version}"
    if staging.exists():
        shutil.rmtree(staging)
    logger.info("Downloading router model package %s from %s", model_version, package_uri)
    download_package(package_uri, staging, object_store=store)
    manifest = verify_package(
        staging,
        expected_model_version=model_version,
        require_approved=require_approved,
    )
    _verify_expected_manifest_sha256(staging, expected_manifest_sha256)

    if target.exists():
        shutil.rmtree(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    # Verify first, publish second: the marker is only written once the staged
    # copy is proven good and moved into place.
    staging.replace(target)
    marker.write_text(package_uri, encoding="utf-8")
    logger.info("Router model package %s is ready at %s", model_version, target)
    return _resolved_package(target, model_version, package_uri, manifest)


def _verify_expected_manifest_sha256(
    package_dir: Path, expected_manifest_sha256: str | None
) -> None:
    if expected_manifest_sha256 is None:
        return
    expected = expected_manifest_sha256.removeprefix("sha256:").lower()
    actual = _sha256_file(package_dir / MANIFEST_FILENAME)
    if actual != expected:
        raise ModelSourceError(
            f"Model manifest SHA-256 mismatch: expected {expected}, got {actual}"
        )
