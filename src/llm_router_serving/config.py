from __future__ import annotations

import re
from pathlib import Path
from typing import Annotated

from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="LLM_ROUTER_",
        extra="ignore",
    )

    app_name: str = "LLM Router Serving"
    internal_api_token: SecretStr | None = None
    max_message_characters: int = 100_000

    embed_model_id: str = "BAAI/bge-large-en-v1.5"
    embed_model_revision: str | None = "d4aa6901d3a41ba39fb536a557fa166f842b0e09"
    embed_model_version: str = "bge-large-en-v1.5"
    embed_expected_dim: int = 1024
    embed_max_length: int = 512
    embed_local_files_only: bool = False

    # Optional cross-encoder reranker. Leave `rerank_model_id` unset to disable it;
    # the service then reports rerankModel=disabled and /router/rerank returns 503.
    # Screening picked cross-encoder/quora-distilroberta-base as the leading
    # candidate but no model met the false-hit gate, so nothing is enabled by default.
    rerank_model_id: str | None = None
    # Pin this to the exact checkpoint the threshold was measured on. A threshold
    # is only meaningful for one set of weights.
    rerank_model_version: str | None = None
    rerank_max_length: int = 512
    rerank_max_candidates: int = 16
    rerank_model_cache_dir: Path | None = None

    router_model_path: Path | None = None
    router_model_version: str | None = None
    model_code_commit: str | None = None
    router_max_length: int = 8192
    model_device: str = "auto"
    load_models_on_startup: bool = True

    # Approved model delivery. The pipeline publishes to GCS; deployment passes the
    # approved URI and version here and the process downloads them during startup.
    router_model_package_uri: str | None = None
    router_model_cache_dir: Path = Path("var/models")
    router_model_require_approved: bool = True

    retraining_db_path: Path = Path("var/retraining.sqlite3")
    # How old an unsubmitted lease must be before it is reconciled against
    # Vertex. Submission takes seconds; this only has to outlast a slow call.
    retraining_stale_submission_seconds: int = 900
    allowed_gcs_buckets: Annotated[tuple[str, ...], NoDecode] = ()
    supported_feedback_schemas: Annotated[tuple[str, ...], NoDecode] = ("feedback-export-v0.1",)

    vertex_project: str | None = None
    vertex_location: str = "us-west1"
    vertex_pipeline_template_uri: str | None = None
    vertex_pipeline_root: str | None = None
    vertex_pipeline_service_account: str | None = None
    vertex_pipeline_network: str | None = None
    vertex_training_service_account: str | None = None
    vertex_training_output_directory: str | None = None
    model_publish_root_uri: str | None = None

    base_training_dataset_uri: str | None = None
    feedback_preprocessing_config_uri: str | None = None
    training_config_uri: str | None = None
    routerarena_dataset_uri: str | None = None
    evaluation_config_uri: str | None = None
    package_metadata_uri: str | None = None
    baseline_metrics_uri: str | None = None
    gate_rules_uri: str | None = None

    @field_validator("allowed_gcs_buckets", "supported_feedback_schemas", mode="before")
    @classmethod
    def split_csv(cls, value: object) -> object:
        if isinstance(value, str):
            return tuple(item.strip() for item in value.split(",") if item.strip())
        return value

    @field_validator(
        "embed_model_revision",
        # Without this an unset `LLM_ROUTER_ROUTER_MODEL_PATH=` parses as
        # Path("") == Path("."), which reads as a configured local source and
        # collides with the package URI.
        "router_model_path",
        "router_model_package_uri",
        "vertex_project",
        "vertex_pipeline_template_uri",
        "vertex_pipeline_root",
        "vertex_pipeline_service_account",
        "vertex_pipeline_network",
        "vertex_training_service_account",
        "vertex_training_output_directory",
        "model_publish_root_uri",
        mode="before",
    )
    @classmethod
    def empty_string_to_none(cls, value: object) -> object:
        if value == "":
            return None
        return value

    def retraining_configuration_errors(self) -> list[str]:
        required = {
            "vertex_project": self.vertex_project,
            "vertex_location": self.vertex_location,
            "vertex_pipeline_template_uri": self.vertex_pipeline_template_uri,
            "vertex_pipeline_root": self.vertex_pipeline_root,
            "base_training_dataset_uri": self.base_training_dataset_uri,
            "feedback_preprocessing_config_uri": self.feedback_preprocessing_config_uri,
            "training_config_uri": self.training_config_uri,
            "routerarena_dataset_uri": self.routerarena_dataset_uri,
            "evaluation_config_uri": self.evaluation_config_uri,
            "package_metadata_uri": self.package_metadata_uri,
            "baseline_metrics_uri": self.baseline_metrics_uri,
            "gate_rules_uri": self.gate_rules_uri,
            "router_model_version": self.router_model_version,
            "model_code_commit": self.model_code_commit,
        }
        missing = [name for name, value in required.items() if value in (None, "")]
        if not self.allowed_gcs_buckets:
            missing.append("allowed_gcs_buckets")
        for name in ("vertex_pipeline_template_uri", "vertex_pipeline_root"):
            value = getattr(self, name)
            if value and not str(value).startswith("gs://"):
                missing.append(f"{name}(must be a gs:// URI)")
        return missing

    def router_model_source_errors(self) -> list[str]:
        """Validate the approved-model source that startup downloads from GCS.

        Exactly one source must be configured: a local directory for development,
        or a GCS package URI for deployment. Both, or neither, is ambiguous.
        """
        has_local = self.router_model_path is not None
        has_remote = self.router_model_package_uri is not None
        errors: list[str] = []
        if has_local and has_remote:
            errors.append("router_model_path and router_model_package_uri are mutually exclusive")
        if not has_local and not has_remote:
            errors.append("router_model_path or router_model_package_uri must be set")
        if has_remote and not str(self.router_model_package_uri).startswith("gs://"):
            errors.append("router_model_package_uri must be a gs:// URI")
        if not self.router_model_version:
            errors.append("router_model_version must be set")
        return errors

    @field_validator("model_code_commit")
    @classmethod
    def validate_model_code_commit(cls, value: str | None) -> str | None:
        if value is not None and not re.fullmatch(r"[0-9a-fA-F]{7,40}", value):
            raise ValueError("model_code_commit must be a 7-40 character Git SHA")
        return value

    @field_validator("router_model_version")
    @classmethod
    def validate_router_model_version(cls, value: str | None) -> str | None:
        if value is not None and not re.fullmatch(
            r"router-[a-z0-9._-]+-\d{8}-[0-9a-fA-F]{7,40}", value
        ):
            raise ValueError(
                "router_model_version must follow router-{model}-{dataset}-{date}-{shortCommit}"
            )
        return value
