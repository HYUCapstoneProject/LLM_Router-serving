"""Validate the R0 swap profile locally, without reading or changing cloud state."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from pydantic import ValidationError

from .config import Settings

PROJECT = "project-763753da-fa8f-4afd-b31"
BUCKET = "gs://llm-router-mlserver"
ROOT = f"{BUCKET}/releases/r0"
REGISTRY = f"{ROOT}/models/router-modernbert"


def validate_r0_profile(settings: Settings) -> list[str]:
    """Return field-level errors only; never include tokens or setting values."""
    errors = [
        *settings.retraining_configuration_errors(),
        *settings.developer_configuration_errors(),
        *settings.router_model_source_errors(),
    ]
    expected = {
        "router_profile": "r0",
        "router_max_length": 8192,
        "router_model_require_approved": True,
        "router_model_auto_reload": True,
        "router_model_path": None,
        "retraining_input_contract": "structure-preserving-v2",
        "supported_feedback_schemas": ("feedback-inline-v1",),
        "retraining_version_policy_mode": "successfulHttpRequest",
        "model_evaluation_metrics_filename": "gate_metrics.json",
        "allowed_gcs_buckets": ("llm-router-mlserver",),
        "vertex_project": PROJECT,
        "vertex_location": "us-west1",
        "vertex_pipeline_root": f"{BUCKET}/pipeline-root/r0",
        "vertex_pipeline_service_account": f"vertex-pipeline-sa@{PROJECT}.iam.gserviceaccount.com",
        "vertex_training_service_account": f"vertex-training-sa@{PROJECT}.iam.gserviceaccount.com",
        "vertex_training_output_directory": f"{ROOT}/vertex-training-output",
        "model_publish_root_uri": REGISTRY,
        "retraining_derived_data_root_uri": f"{ROOT}/datasets/feedback",
        "retraining_observability_root_uri": f"{ROOT}/retraining-runs",
        "retraining_control_root_uri": f"{ROOT}/retraining-control",
        "router_model_cache_dir": Path("/app/var/r0/models"),
        "retraining_db_path": Path("/app/var/r0/retraining.sqlite3"),
    }
    for field, value in expected.items():
        if getattr(settings, field) != value:
            errors.append(f"{field}(does not match the R0 profile)")
    if settings.current_model_pointer_uri() != f"{REGISTRY}/CURRENT.json":
        errors.append("router_model_current_pointer_uri(must match the R0 registry)")
    if settings.router_model_package_uri and not (
        settings.router_model_package_uri.startswith(f"{REGISTRY}/")
        and settings.router_model_package_uri.endswith("/approved")
    ):
        errors.append("router_model_package_uri(must be approved in the R0 registry)")
    for field in (
        "vertex_pipeline_template_uri",
        "feedback_preprocessing_config_uri",
        "training_config_uri",
        "evaluation_config_uri",
        "package_metadata_uri",
        "baseline_metrics_uri",
        "gate_rules_uri",
        "base_training_dataset_uri",
        "routerarena_dataset_uri",
    ):
        value = getattr(settings, field)
        if not value:
            continue
        if field == "vertex_pipeline_template_uri":
            prefixes = (f"{ROOT}/pipelines/",)
        elif field in {"base_training_dataset_uri", "routerarena_dataset_uri"}:
            # Frozen shared input datasets are allowed; output/model groups remain separate.
            prefixes = (f"{BUCKET}/datasets/", f"{ROOT}/datasets/")
        elif field == "baseline_metrics_uri":
            prefixes = (f"{ROOT}/config/", f"{REGISTRY}/")
        else:
            prefixes = (f"{ROOT}/config/",)
        if not value.startswith(prefixes) or any(
            word in value.lower() for word in ("smoke", "replace", "placeholder", "empty")
        ):
            errors.append(f"{field}(must name an immutable non-smoke R0 release artifact)")
        if any(part in {".", ".."} for part in value.split("/")):
            errors.append(f"{field}(path segments must be canonical)")
    if not re.fullmatch(r"[0-9a-f]{40}", settings.model_code_commit or ""):
        errors.append("model_code_commit(full lowercase Git SHA required)")
    token = settings.internal_api_token
    if (
        token is None
        or not token.get_secret_value().strip()
        or any(marker in token.get_secret_value().lower() for marker in ("replace", "placeholder"))
    ):
        errors.append("internal_api_token(real secret required)")
    return sorted(set(errors))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, required=True)
    args = parser.parse_args()
    if not args.env_file.is_file():
        errors = ["env_file(file not found)"]
    else:
        try:
            errors = validate_r0_profile(Settings(_env_file=args.env_file))
        except ValidationError as exc:
            # Pydantic's normal exception includes input values: never print those.
            errors = sorted({".".join(map(str, item["loc"])) for item in exc.errors()})
    print(
        json.dumps(
            {
                "schema_version": "r0-profile-validation-v1",
                "profile_valid": not errors,
                "cloud_verified": False,
                "model_approved": False,
                "errors": errors,
            },
            sort_keys=True,
        )
    )
    raise SystemExit(1 if errors else 0)


if __name__ == "__main__":
    main()
