from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

from llm_router_serving.config import Settings
from llm_router_serving.r0_profile import ROOT, main, validate_r0_profile

EXAMPLE = Path(__file__).resolve().parents[1] / "deploy/r0.env.example"


@pytest.fixture(autouse=True)
def clean_profile_environment(monkeypatch):
    for name in os.environ:
        if name.startswith("LLM_ROUTER_"):
            monkeypatch.delenv(name)


@pytest.fixture
def profile_settings():
    return Settings(
        _env_file=EXAMPLE,
        internal_api_token="synthetic-test-secret",
        router_model_version="router-modernbert-base-feedback-20260926-abcdefab-abcdefa",
        model_code_commit="a" * 40,
        vertex_pipeline_template_uri=f"{ROOT}/pipelines/release-sha/retraining.yaml",
        base_training_dataset_uri="gs://llm-router-mlserver/datasets/frozen/base.parquet",
        routerarena_dataset_uri="gs://llm-router-mlserver/datasets/frozen/gate.parquet",
        **{
            field: f"{ROOT}/config/release-sha/{filename}"
            for field, filename in {
                "feedback_preprocessing_config_uri": "feedback.json",
                "training_config_uri": "training.json",
                "evaluation_config_uri": "evaluation.json",
                "package_metadata_uri": "package-metadata.json",
                "baseline_metrics_uri": "baseline-metrics.json",
                "gate_rules_uri": "gate-rules.json",
            }.items()
        },
    )


def test_completed_r0_profile_is_locally_valid(profile_settings):
    assert validate_r0_profile(profile_settings) == []


def test_example_cannot_pass_without_real_release_artifacts_and_token():
    errors = validate_r0_profile(Settings(_env_file=EXAMPLE))
    assert "gate_rules_uri" in errors
    assert "baseline_metrics_uri" in errors
    assert "internal_api_token(real secret required)" in errors


@pytest.mark.parametrize(
    "field,value",
    [
        ("router_max_length", 512),
        ("router_model_require_approved", False),
        ("retraining_input_contract", "legacy-word-only-v1"),
        ("retraining_version_policy_mode", "validatedUniqueBatch"),
        ("supported_feedback_schemas", ("feedback-inline-v2",)),
        ("gate_rules_uri", f"{ROOT}/config/release-sha/smoke.json"),
        ("baseline_metrics_uri", f"{ROOT}/config/release-sha/empty.json"),
        ("training_config_uri", f"{ROOT}/config/../r1/training.json"),
        ("retraining_control_root_uri", "gs://llm-router-mlserver/retraining-control"),
        ("vertex_pipeline_root", "gs://llm-router-mlserver/pipeline-root/r1"),
        ("router_model_current_pointer_uri", f"{ROOT}-other/models/CURRENT.json"),
        ("router_model_package_uri", "gs://llm-router-mlserver/releases/r1/v1/approved"),
        ("model_code_commit", "abcdefa"),
        ("model_evaluation_metrics_filename", "routerarena_metrics.json"),
    ],
)
def test_profile_rejects_mixed_contracts(profile_settings, field, value):
    settings = profile_settings.model_copy(update={field: value})
    assert any(field in error for error in validate_r0_profile(settings))


def test_cli_hides_invalid_configuration_values(tmp_path, monkeypatch, capsys):
    path = tmp_path / "bad.env"
    path.write_text("LLM_ROUTER_MODEL_CODE_COMMIT=do-not-print-this-secret\n")
    monkeypatch.setattr(sys, "argv", ["validate-r0-profile", "--env-file", str(path)])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 1
    output = capsys.readouterr().out
    assert "do-not-print-this-secret" not in output
    report = json.loads(output)
    assert report["errors"] == ["model_code_commit"]
    assert report["profile_valid"] is False
    assert report["cloud_verified"] is False
    assert report["model_approved"] is False
