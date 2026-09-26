"""Exact bytes produced by ML's real gate/package/publish/approve CLI chain."""

import json
from pathlib import Path

import pytest

from llm_router_serving.retraining.observability import (
    SUPPORTED_PROMOTION_CONTRACTS,
    PromotionEvidenceUnavailable,
    RetrainingObservabilityReader,
)


@pytest.fixture
def fixture():
    return json.loads(
        (Path(__file__).parent / "fixtures/shared-promotion-producer.json").read_text()
    )


def verify(fixture):
    class Store:
        def read_bytes(self, uri):
            return fixture["objects"][uri].encode()

    expected = fixture["expected"]
    suffix = f"/{expected['batch_id']}/observability-v1"
    reader = RetrainingObservabilityReader(
        object_store=Store(), root_uri=expected["observability_run_uri"][: -len(suffix)]
    )
    reader.verify_promotion(
        batch_id=expected["batch_id"],
        candidate_model_version=expected["candidate_model_version"],
        pipeline_run_id=expected["pipeline_job_resource_name"],
        publish_root_uri=expected["model_publish_root_uri"],
    )


def test_actual_shared_ml_fixture_is_accepted_without_translation(fixture):
    verify(fixture)
    assert {
        ("ordinal", "model-package-v1", "deploy-gate-result-v1")
    } == SUPPORTED_PROMOTION_CONTRACTS


@pytest.mark.parametrize(
    "field,value",
    [
        ("objective", "pointwise"),
        ("package_schema_version", "model-package-v2"),
        ("gate_schema_version", "future-pointwise-gate-v1"),
        ("candidate_manifest_sha256", "0" * 64),
        ("approved_manifest_sha256", "0" * 64),
        ("candidate_metrics_sha256", "0" * 64),
        ("current_pointer_generation", None),
        ("current_pointer_generation", True),
        ("previous_pointer_generation", 123),
        ("inference_max_length", 512),
        ("length_contract_version", None),
    ],
)
def test_shared_evidence_tampering_is_retryable_and_fail_closed(fixture, field, value):
    run = fixture["expected"]["observability_run_uri"]
    uri = run + "/result/publish-result.json"
    record = json.loads(fixture["objects"][uri])
    record[field] = value
    fixture["objects"][uri] = json.dumps(record)
    with pytest.raises(PromotionEvidenceUnavailable):
        verify(fixture)


def test_generic_envelope_cannot_bypass_unimplemented_pointwise_gate(fixture):
    uri = fixture["expected"]["observability_run_uri"] + "/result/publish-result.json"
    record = json.loads(fixture["objects"][uri])
    record.update(
        objective="pointwise",
        package_schema_version="model-package-v2",
        gate_schema_version="deploy-gate-result-v1",
    )
    fixture["objects"][uri] = json.dumps(record)
    with pytest.raises(PromotionEvidenceUnavailable, match="unsupported"):
        verify(fixture)
