"""R0/R1 service choices retain the existing request and four response fields."""

import math

import pytest
import torch
from conftest import FakeEmbedder
from fastapi.testclient import TestClient
from test_inference import FakeRouterModel, FakeTokenizer
from test_routing_policy_v2 import classifier_for

from llm_router_serving import main
from llm_router_serving.inference import TransformersModernBertClassifier
from llm_router_serving.model_manager import ModelManager
from llm_router_serving.routing_policy import LENGTH_CONTRACT_VERSION, compute_policy_sha256


@pytest.mark.parametrize("objective", ["ordinal", "pointwise"])
@pytest.mark.parametrize(("penalty", "selected", "tier", "model"), [
    (0.0, 1, "medium", "sonnet"), (10.0, 0, "easy", "haiku"),
])
def test_existing_endpoint_projects_the_policy_decision_at_8192_without_schema_changes(
    settings, escalation_service, auth_headers, policy_payload_factory,
    objective, penalty, selected, tier, model,
):
    policy = policy_payload_factory(objective, lambda_cost=penalty)
    policy.update(length_contract_version=LENGTH_CONTRACT_VERSION, max_length=8192)
    policy["policy_sha256"] = compute_policy_sha256(policy)
    engine = classifier_for(policy)
    manager = ModelManager.loaded(embed_engine=FakeEmbedder(), router_engine=engine)
    app = main.create_app(
        settings=settings, model_manager=manager, escalation_service=escalation_service
    )
    with TestClient(app) as client:
        response = client.post(
            "/router/classify", json={"message": "question"}, headers=auth_headers
        )
        detailed = client.post(
            "/router/classify/v2", json={"message": "question"}, headers=auth_headers
        )
        assert response.status_code == detailed.status_code == 200
        payload = response.json()
        assert set(payload) == {"tier", "model", "confidence", "modelVersion"}
        assert payload == {
            key: detailed.json()[key] for key in ("tier", "model", "confidence", "modelVersion")
        }
        assert payload["tier"] == tier and payload["model"] == model
        logits = [0.1, 2.0, 0.3]
        score = (
            1.0 / (1.0 + math.exp(-logits[selected])) if objective == "pointwise"
            else math.exp(logits[selected]) / sum(math.exp(value) for value in logits)
        )
        assert payload["confidence"] == pytest.approx(score)
        assert client.post("/router/classify", json={"message": "question"}).status_code == 401
        assert client.post(
            "/router/classify", json={"message": "question", "objective": objective},
            headers=auth_headers,
        ).status_code == 400
        empty = client.post("/router/classify", json={"message": "\u200b"}, headers=auth_headers)
        assert empty.status_code == 400 and empty.json()["code"] == "INVALID_MESSAGE"
        schemas = app.openapi()["components"]["schemas"]
        assert set(schemas["ClassifyResponse"]["properties"]) == set(payload)
        assert set(schemas["MessageRequest"]["properties"]) == {"message"}


async def test_same_endpoint_supports_r0_to_r1_and_back_without_client_changes(
    settings, escalation_service, auth_headers, policy_payload_factory
):
    legacy = TransformersModernBertClassifier(
        tokenizer=FakeTokenizer(), model=FakeRouterModel(), torch=torch,
        device=torch.device("cpu"), model_version="r0-original", max_length=8192,
    )
    pointwise = classifier_for(policy_payload_factory("pointwise", lambda_cost=10.0))
    pointwise.model_version = "r1-replacement"
    engines = {legacy.model_version: legacy, pointwise.model_version: pointwise}
    manager = ModelManager(
        embed_loader=FakeEmbedder,
        router_loader=lambda: legacy,
        router_version_loader=lambda uri, version, digest: engines[version],
    )
    await manager.load_all()
    app = main.create_app(
        settings=settings, model_manager=manager, escalation_service=escalation_service
    )
    with TestClient(app) as client:
        for version, expected_tier in [
            ("r0-original", "medium"), ("r1-replacement", "easy"), ("r0-original", "medium"),
        ]:
            await manager.reload_router(
                package_uri=f"gs://fixture/{version}", model_version=version,
                manifest_sha256="a" * 64,
            )
            response = client.post(
                "/router/classify", json={"message": "same request"}, headers=auth_headers
            )
            assert response.status_code == 200
            assert set(response.json()) == {"tier", "model", "confidence", "modelVersion"}
            assert response.json()["tier"] == expected_tier
            assert response.json()["modelVersion"] == version
