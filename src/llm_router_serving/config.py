from __future__ import annotations

import re
from pathlib import Path
from typing import Annotated, Literal
from urllib.parse import urlparse

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="LLM_ROUTER_",
        extra="ignore",
    )

    app_name: str = "LLM Router Serving"
    internal_api_token: SecretStr | None = None
    enable_developer_api: bool = False
    developer_api_token: SecretStr | None = None
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
    # How the checkpoint is scored. `cross_encoder` is a one-logit duplicate model
    # (sigmoid); `nli` is an NLI head scored by bidirectional entailment on the
    # sentences the two texts do not share. Each needs its own measured threshold.
    rerank_kind: Literal["cross_encoder", "nli"] = "cross_encoder"
    rerank_max_length: int = 512
    rerank_max_candidates: int = 16
    # The image bakes the weights in and the runtime is HF_HUB_OFFLINE=1,
    # so loading must never reach the network.
    rerank_local_files_only: bool = True

    router_model_path: Path | None = None
    router_model_version: str | None = None
    model_code_commit: str | None = None
    router_max_length: int = 8192
    router_profile: Literal["legacy", "r0"] = "legacy"
    model_device: str = "auto"
    model_num_threads: int | None = Field(default=None, ge=1, le=32)
    load_models_on_startup: bool = True

    # Approved model delivery. Deployment follows CURRENT.json by default; a fixed
    # approved URI/version remains available for pinned or rollback deployments.
    router_model_package_uri: str | None = None
    router_model_current_pointer_uri: str | None = None
    router_model_cache_dir: Path = Path("var/models")
    router_model_require_approved: bool = True
    router_model_auto_reload: bool = True
    router_model_poll_interval_seconds: float = 60.0

    retraining_db_path: Path = Path("var/retraining.sqlite3")
    # How old an unsubmitted lease must be before it is reconciled against
    # Vertex. Submission takes seconds; this only has to outlast a slow call.
    retraining_stale_submission_seconds: int = 900
    retraining_initial_logical_major: int = 1
    # The call-count trigger is intentional for the current research phase.
    retraining_calls_per_major: int = 3
    retraining_rebase_every_major_cycles: int = 5
    retraining_replay_rows_per_label: int = 32
    retraining_version_policy_mode: str = "successfulHttpRequest"
    retraining_input_contract: Literal["legacy-word-only-v1", "structure-preserving-v2"] = (
        "legacy-word-only-v1"
    )
    retraining_derived_data_root_uri: str | None = None
    retraining_observability_root_uri: str | None = None
    retraining_control_root_uri: str | None = None
    allowed_gcs_buckets: Annotated[tuple[str, ...], NoDecode] = ()
    supported_feedback_schemas: Annotated[tuple[str, ...], NoDecode] = ("feedback-inline-v1",)

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
    retraining_promotion_mode: Literal["quality_gate", "direct"] = "quality_gate"
    baseline_metrics_uri: str | None = None
    model_evaluation_metrics_filename: Literal["routerarena_metrics.json", "gate_metrics.json"] = (
        "routerarena_metrics.json"
    )
    gate_rules_uri: str | None = None

    @field_validator("allowed_gcs_buckets", "supported_feedback_schemas", mode="before")
    @classmethod
    def split_csv(cls, value: object) -> object:
        if isinstance(value, str):
            return tuple(item.strip() for item in value.split(",") if item.strip())
        return value

    @field_validator(
        "embed_model_revision",
        "developer_api_token",
        # Without this an unset `LLM_ROUTER_ROUTER_MODEL_PATH=` parses as
        # Path("") == Path("."), which reads as a configured local source and
        # collides with the package URI.
        "router_model_path",
        "router_model_package_uri",
        "router_model_current_pointer_uri",
        "router_model_version",
        "vertex_project",
        "vertex_pipeline_template_uri",
        "vertex_pipeline_root",
        "vertex_pipeline_service_account",
        "vertex_pipeline_network",
        "vertex_training_service_account",
        "vertex_training_output_directory",
        "model_publish_root_uri",
        "retraining_derived_data_root_uri",
        "retraining_observability_root_uri",
        "retraining_control_root_uri",
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
            "model_publish_root_uri": self.model_publish_root_uri,
            "retraining_derived_data_root_uri": self.retraining_derived_data_root_uri,
            "retraining_observability_root_uri": self.retraining_observability_root_uri,
            "router_model_version": self.router_model_version,
            "model_code_commit": self.model_code_commit,
        }
        missing = [name for name, value in required.items() if value in (None, "")]
        if not self.allowed_gcs_buckets:
            missing.append("allowed_gcs_buckets")
        for name in (
            "vertex_pipeline_template_uri",
            "vertex_pipeline_root",
            "model_publish_root_uri",
        ):
            value = getattr(self, name)
            if value and not str(value).startswith("gs://"):
                missing.append(f"{name}(must be a gs:// URI)")
        derived_uri = self.derived_retraining_data_root_uri()
        if derived_uri and not derived_uri.startswith("gs://"):
            missing.append("retraining_derived_data_root_uri(must be a gs:// URI)")
        elif derived_uri and urlparse(derived_uri).netloc not in self.allowed_gcs_buckets:
            missing.append("retraining_derived_data_root_uri(bucket must be allowed)")
        for name in (
            "retraining_observability_root_uri",
            "retraining_control_root_uri",
        ):
            value = getattr(self, name)
            if value and not str(value).startswith("gs://"):
                missing.append(f"{name}(must be a gs:// URI)")
            elif value and urlparse(str(value)).netloc not in self.allowed_gcs_buckets:
                missing.append(f"{name}(bucket must be allowed)")
        if (
            self.retraining_version_policy_mode == "validatedUniqueBatch"
            and not self.retraining_control_root_uri
        ):
            missing.append("retraining_control_root_uri")
        if (
            self.retraining_promotion_mode == "direct"
            and self.retraining_version_policy_mode != "successfulHttpRequest"
        ):
            missing.append("direct promotion requires successfulHttpRequest version policy")
        if self.retraining_calls_per_major != 3:
            missing.append("retraining_calls_per_major(must currently equal 3)")
        if self.retraining_initial_logical_major < 0:
            missing.append("retraining_initial_logical_major(must be non-negative)")
        if self.retraining_rebase_every_major_cycles < 1:
            missing.append("retraining_rebase_every_major_cycles(must be positive)")
        if self.retraining_replay_rows_per_label < 1:
            missing.append("retraining_replay_rows_per_label(must be positive)")
        return missing

    def developer_configuration_errors(self) -> list[str]:
        if self.enable_developer_api and self.developer_api_token is None:
            return ["developer_api_token"]
        return []

    def derived_retraining_data_root_uri(self) -> str | None:
        if self.retraining_derived_data_root_uri:
            return self.retraining_derived_data_root_uri.rstrip("/")
        return None

    def initial_approved_model_uri(self) -> str | None:
        if self.router_model_package_uri:
            return self.router_model_package_uri.rstrip("/")
        if self.model_publish_root_uri and self.router_model_version:
            return f"{self.model_publish_root_uri.rstrip('/')}/{self.router_model_version}/approved"
        return None

    def router_model_source_errors(self) -> list[str]:
        """Validate the approved-model source that startup downloads from GCS.

        A local directory is used for development. Deployment may use either a
        fixed approved GCS package or CURRENT.json; when both remote values are
        configured, CURRENT.json is authoritative.
        """
        has_local = self.router_model_path is not None
        has_remote = self.router_model_package_uri is not None
        pointer_uri = self.current_model_pointer_uri()
        has_pointer = pointer_uri is not None
        errors: list[str] = []
        if has_local and (has_remote or has_pointer):
            errors.append(
                "router_model_path cannot be combined with a GCS package or CURRENT pointer"
            )
        if not has_local and not has_remote and not has_pointer:
            errors.append(
                "router_model_path, router_model_package_uri, or "
                "router_model_current_pointer_uri must be set"
            )
        if has_remote and not str(self.router_model_package_uri).startswith("gs://"):
            errors.append("router_model_package_uri must be a gs:// URI")
        if has_pointer and not str(pointer_uri).startswith("gs://"):
            errors.append("router_model_current_pointer_uri must be a gs:// URI")
        if not has_pointer and not self.router_model_version:
            errors.append("router_model_version must be set")
        if self.router_model_poll_interval_seconds <= 0:
            errors.append("router_model_poll_interval_seconds must be positive")
        return errors

    def current_model_pointer_uri(self) -> str | None:
        if self.router_model_current_pointer_uri:
            return self.router_model_current_pointer_uri
        if self.model_publish_root_uri and self.router_model_path is None:
            return f"{self.model_publish_root_uri.rstrip('/')}/CURRENT.json"
        return None

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
            r"router-[a-z0-9._-]+-\d{8}(?:(?:-[0-9a-fA-F]{8})?-[0-9a-fA-F]{7,40}|-v[1-9][0-9]*)",
            value,
        ):
            raise ValueError(
                "router_model_version must follow router-{model}-{dataset}-{date}-"
                "[{batchDigest}-]{shortCommit} or "
                "router-{model}-{dataset}-{date}-v{positiveInteger}"
            )
        return value

    @field_validator("retraining_version_policy_mode")
    @classmethod
    def validate_version_policy_mode(cls, value: str) -> str:
        allowed = {"successfulHttpRequest", "validatedUniqueBatch"}
        if value not in allowed:
            raise ValueError(f"retraining_version_policy_mode must be one of {sorted(allowed)}")
        return value
