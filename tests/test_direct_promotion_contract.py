"""Cross-repository, real ML producer bytes; no fabricated gate or evaluation."""

import hashlib
import json
from pathlib import Path

import pytest

from llm_router_serving.retraining.observability import (
    PromotionEvidenceUnavailable,
    RetrainingObservabilityReader,
)


@pytest.fixture
def direct_fixture():
    return json.loads(
        (Path(__file__).parent / "fixtures/direct-promotion-producer-v1.json").read_text()
    )


def verify(fixture, mode="direct"):
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
        promotion_mode=mode,
    )


def test_real_direct_ml_producer_is_accepted(direct_fixture):
    verify(direct_fixture)


def test_direct_evidence_cannot_authorize_quality_gate_run(direct_fixture):
    with pytest.raises(PromotionEvidenceUnavailable):
        verify(direct_fixture, "quality_gate")


@pytest.mark.parametrize(
    "field,value",
    [
        ("batch_id", "other"),
        ("candidate_model_version", "other"),
        ("pipeline_job_resource_name", "other"),
        ("promotion_mode", "quality_gate"),
        ("quality_gate_skipped", False),
        ("passed", True),
        ("candidate_manifest_sha256", "0" * 64),
        ("baseline_manifest_sha256", "0" * 64),
        ("baseline_model_version", "other"),
        ("expected_current_generation", 999),
        ("expected_current_generation", True),
        ("model_publish_root_uri", "gs://other"),
        ("inference_max_length", 512),
        ("length_contract_version", None),
        ("candidate_metrics_sha256", "0" * 64),
    ],
)
def test_tampered_direct_authorization_fails_even_with_refreshed_hash(direct_fixture, field, value):
    run = direct_fixture["expected"]["observability_run_uri"]
    key = run + "/promotion/direct-result.json"
    data = json.loads(direct_fixture["objects"][key])
    data[field] = value
    raw = json.dumps(data)
    direct_fixture["objects"][key] = raw
    pubkey = run + "/result/publish-result.json"
    pub = json.loads(direct_fixture["objects"][pubkey])
    pub["gate_result_sha256"] = hashlib.sha256(raw.encode()).hexdigest()
    direct_fixture["objects"][pubkey] = json.dumps(pub)
    with pytest.raises(PromotionEvidenceUnavailable):
        verify(direct_fixture)


@pytest.mark.parametrize(
    "field,value",
    [
        ("gate_result_sha256", "0" * 64),
        ("current_pointer_sha256", "0" * 64),
        ("manifest_sha256", "0" * 64),
        ("current_pointer_generation", None),
        ("candidate_metrics_sha256", "0" * 64),
        ("quality_gate_skipped", False),
        ("baseline_manifest_sha256", "0" * 64),
        ("expected_current_generation", 999),
    ],
)
def test_tampered_direct_publishing_fails(direct_fixture, field, value):
    key = direct_fixture["expected"]["observability_run_uri"] + "/result/publish-result.json"
    data = json.loads(direct_fixture["objects"][key])
    data[field] = value
    direct_fixture["objects"][key] = json.dumps(data)
    with pytest.raises(PromotionEvidenceUnavailable):
        verify(direct_fixture)


@pytest.mark.asyncio
async def test_direct_success_counter_requires_evidence_and_survives_mode_change(
    tmp_path, direct_fixture
):
    from test_retraining import build_request, build_service

    from llm_router_serving.retraining.pipeline import PipelineState

    expected = direct_fixture["expected"]
    service, pipeline = build_service(tmp_path, build_request())
    service._settings.model_publish_root_uri = expected["model_publish_root_uri"]
    service._observability_reader._root_uri = expected["observability_run_uri"].rsplit("/", 2)[0]
    service._job_store.claim(
        batch_id=expected["batch_id"],
        request_fingerprint="fp",
        candidate_model_version=expected["candidate_model_version"],
        logical_model_version="1.1",
        promotion_mode="direct",
    )
    service._job_store.mark_submitted(
        batch_id=expected["batch_id"], pipeline_run_id=expected["pipeline_job_resource_name"]
    )
    pipeline.states[expected["pipeline_job_resource_name"]] = PipelineState.SUCCEEDED

    async def status():
        return await service.status(
            expected["batch_id"],
            serving_model_version="baseline-r0",
            desired_model_version=None,
            desired_load_error=None,
        )

    waiting = await status()
    assert waiting.stale and not waiting.terminal
    assert service._job_store.list_succeeded() == []
    service._observability_reader._object_store.objects.update(
        {key: value.encode() for key, value in direct_fixture["objects"].items()}
    )
    promoted = await status()
    assert promoted.status == "succeeded" and promoted.result.gatePassed is None
    assert promoted.result.outcome == "promoted"
    assert len(service._job_store.list_succeeded()) == 1
    assert {str(x.name): str(x.status) for x in promoted.stages}["evaluation"] == "skipped"
    assert {str(x.name): str(x.status) for x in promoted.stages}["gate"] == "skipped"
    await status()
    assert len(service._job_store.list_succeeded()) == 1
