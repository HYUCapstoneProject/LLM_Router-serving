from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from llm_router_serving.config import Settings
from llm_router_serving.errors import ModelContractError
from llm_router_serving.main import create_app
from llm_router_serving.model_manager import ModelManager
from llm_router_serving.model_source import ResolvedModel
from llm_router_serving.profiles import validate_router_profile


def package():
    return ResolvedModel(
        path=Path("/verified"),
        model_version="r0-approved",
        package_uri=None,
        manifest_sha256="a" * 64,
        input_contract="structure-preserving-v2",
        max_length=8192,
        training_max_length=512,
    )


def test_r0_accepts_verified_training512_inference8192():
    validate_router_profile(Settings(router_profile="r0"), package())


@pytest.mark.parametrize(
    "changes",
    [
        {"max_length": 512},
        {"training_max_length": None},
        {"manifest_sha256": None},
        {"input_contract": None},
        {"routing_policy": SimpleNamespace(objective="pointwise")},
        {"routing_policy": SimpleNamespace(objective="pairwise")},
    ],
)
def test_r0_rejects_wrong_release(changes):
    with pytest.raises(ModelContractError):
        validate_router_profile(Settings(router_profile="r0"), replace(package(), **changes))


@pytest.mark.parametrize(
    "changes", [{"router_max_length": 512}, {"router_model_require_approved": False}]
)
def test_r0_rejects_unsafe_settings(changes):
    with pytest.raises(ModelContractError):
        validate_router_profile(Settings(router_profile="r0", **changes), package())


@pytest.mark.parametrize("objective", ["pointwise", "pairwise"])
def test_escalate_rejects_nonordinal_before_pipeline(
    settings, escalation_service, auth_headers, objective
):
    manager = ModelManager.loaded(
        embed_engine=SimpleNamespace(model_version="bge"),
        router_engine=SimpleNamespace(model_version="nonordinal", routing_objective=objective),
    )
    app = create_app(
        settings=settings, model_manager=manager, escalation_service=escalation_service
    )
    with TestClient(app) as client:
        response = client.post(
            "/escalate",
            headers=auth_headers,
            json={
                "batchId": "feedback-20260927-0001",
                "schemaVersion": "feedback-inline-v1",
                "requestedAt": "2026-09-27T01:00:00Z",
                "events": [
                    {
                        "eventId": "esc-1",
                        "requestId": "req-1",
                        "question": "Explain gravity.",
                        "initialLabel": "easy",
                        "initialModel": "haiku",
                        "finalLabel": "medium",
                        "finalModel": "sonnet",
                        "userSatisfied": True,
                        "escalationCount": 1,
                        "createdAt": "2026-09-27T00:00:00Z",
                        "routerModelVersion": "nonordinal",
                        "piiRedacted": True,
                    }
                ],
            },
        )
    assert response.status_code == 409
    assert response.json()["code"] == "R0_OBJECTIVE_REQUIRED"
    assert escalation_service.calls == 0
