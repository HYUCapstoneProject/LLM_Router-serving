from __future__ import annotations

from fastapi.testclient import TestClient

from llm_router_serving.main import create_app
from llm_router_serving.model_manager import ModelManager


def test_embed_returns_contract_vector(client, auth_headers):
    response = client.post(
        "/router/embed",
        headers=auth_headers,
        json={"message": "How does semantic caching work?"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["dim"] == 1024
    assert len(payload["embedding"]) == 1024
    assert payload["embedding"][0] == 1.0
    assert payload["modelVersion"] == "bge-large-en-v1.5"
    assert response.headers["X-Request-ID"] == "req-test-123"


def test_classify_returns_tier_even_with_confidence(client, auth_headers):
    response = client.post(
        "/router/classify",
        headers=auth_headers,
        json={"message": "What is 2 + 2?"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "tier": "easy",
        "model": "haiku",
        "confidence": 0.83,
        "modelVersion": "router-modernbert-base-routerbench-20260731-21c2f1b",
    }


def test_router_endpoints_require_bearer_token(client):
    response = client.post("/router/embed", json={"message": "hello"})

    assert response.status_code == 401
    assert response.json()["code"] == "UNAUTHORIZED"
    assert response.headers["X-Request-ID"]


def test_invalid_message_uses_contract_error(client, auth_headers):
    response = client.post(
        "/router/classify",
        headers=auth_headers,
        json={"message": "   "},
    )

    assert response.status_code == 400
    assert response.json() == {
        "error": "message is missing or invalid",
        "code": "INVALID_MESSAGE",
        "requestId": "req-test-123",
    }


def test_health_returns_ok_when_both_models_are_loaded(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "embedModel": "loaded",
        "routerModel": "loaded",
    }


def test_health_returns_503_while_models_are_loading(settings, escalation_service):
    manager = ModelManager(
        embed_loader=lambda: None,
        router_loader=lambda: None,
    )
    app = create_app(
        settings=settings,
        model_manager=manager,
        escalation_service=escalation_service,
    )

    with TestClient(app) as test_client:
        response = test_client.get("/health")

    assert response.status_code == 503
    assert response.json() == {
        "status": "not_ready",
        "embedModel": "loading",
        "routerModel": "loading",
    }


def test_escalate_returns_202_and_pipeline_run_id(client, auth_headers, escalation_service):
    response = client.post(
        "/escalate",
        headers=auth_headers,
        json={
            "batchId": "feedback-20260812-0001",
            "schemaVersion": "feedback-export-v0.1",
            "manifestUri": (
                "gs://llm-router/datasets/feedback/feedback-20260812-0001/manifest.json"
            ),
            "eventsUri": (
                "gs://llm-router/datasets/feedback/feedback-20260812-0001/events.parquet"
            ),
            "eventCount": 1000,
            "requestedAt": "2026-08-12T15:30:00Z",
        },
    )

    assert response.status_code == 202
    assert response.json() == {
        "status": "accepted",
        "batchId": "feedback-20260812-0001",
        "pipelineRunId": "pipeline-run-123",
    }
    assert escalation_service.calls == 1


def test_escalate_rejects_non_contract_event_count(client, auth_headers):
    response = client.post(
        "/escalate",
        headers=auth_headers,
        json={
            "batchId": "feedback-20260812-0001",
            "schemaVersion": "feedback-export-v0.1",
            "manifestUri": "gs://llm-router/manifest.json",
            "eventsUri": "gs://llm-router/events.parquet",
            "eventCount": 999,
            "requestedAt": "2026-08-12T15:30:00Z",
        },
    )

    assert response.status_code == 422
    assert response.json()["code"] == "REQUEST_VALIDATION_FAILED"
