from __future__ import annotations

import asyncio
import hashlib
import json
from types import SimpleNamespace

import pytest
import torch
from conftest import FakeClassifier, FakeEmbedder
from fastapi.testclient import TestClient
from test_inference import FakeRouterModel, FakeTokenizer
from test_model_source import MODEL_VERSION, PACKAGE_URI, MemoryPackageStore, build_package

from llm_router_serving import main
from llm_router_serving.errors import ModelContractError
from llm_router_serving.inference import InvalidRoutingInputError, TransformersModernBertClassifier
from llm_router_serving.model_manager import ModelManager
from llm_router_serving.model_source import (
    CurrentModelPointer,
    ModelSourceError,
    ResolvedModel,
    resolve_router_model,
    verify_package,
)
from llm_router_serving.routing_policy import (
    RoutingPolicy,
    RoutingPolicyError,
    route_scores,
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_v2_package(directory, payload):
    directory = build_package(directory)
    objective = payload["objective"]
    labels = ("easy", "medium", "hard") if objective == "ordinal" else ("low", "medium", "high")
    id2label = {str(index): label for index, label in enumerate(labels)}
    (directory / "config.json").write_text(
        json.dumps(
            {
                "router_objective": objective,
                "num_labels": 3,
                "id2label": id2label,
                "label2id": {label: index for index, label in enumerate(labels)},
            }
        )
    )
    (directory / "training_config.json").write_text(
        json.dumps({"objective_method": objective, "max_length": 512, "epochs": 3})
    )
    (directory / "routing_policy.json").write_text(json.dumps(payload, indent=2))
    manifest = json.loads((directory / "model_manifest.json").read_text())
    manifest.update(
        schema_version="model-package-v2",
        objective=objective,
        id2label=id2label,
        input_contract="structure-preserving-v2",
        tokenizer_max_length=512,
        routing_policy_file="routing_policy.json",
        routing_policy_sha256=sha(directory / "routing_policy.json"),
    )
    manifest["files"] = {
        name: sha(directory / name) for name in (*manifest["files"], "routing_policy.json")
    }
    (directory / "model_manifest.json").write_text(json.dumps(manifest))
    return directory


def rewrite_json_artifact(directory, filename, value):
    path = directory / filename
    path.write_text(json.dumps(value))
    manifest_path = directory / "model_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"][filename] = sha(path)
    if filename == "routing_policy.json":
        manifest["routing_policy_sha256"] = sha(path)
    manifest_path.write_text(json.dumps(manifest))


def classifier_for(payload, tokenizer=None):
    policy = RoutingPolicy.from_mapping(payload)
    model = FakeRouterModel()
    model.config = SimpleNamespace(
        router_objective=policy.objective,
        num_labels=3,
        id2label=dict(enumerate(policy.output_label_order)),
    )
    return TransformersModernBertClassifier(
        tokenizer=tokenizer or FakeTokenizer(),
        model=model,
        torch=torch,
        device=torch.device("cpu"),
        model_version=MODEL_VERSION,
        max_length=policy.max_length,
        routing_policy=policy,
        package_manifest_sha256="a" * 64,
    )


@pytest.mark.parametrize("objective", ["ordinal", "pointwise", "pairwise"])
def test_v2_package_resolves_policy_without_changing_current_tier_aliases(
    tmp_path, policy_payload_factory, objective
):
    payload = policy_payload_factory(objective)
    package = build_v2_package(tmp_path / "package", payload)
    resolved = resolve_router_model(
        package_uri=PACKAGE_URI,
        local_path=None,
        model_version=MODEL_VERSION,
        cache_dir=tmp_path / "cache",
        object_store=MemoryPackageStore(package),
    )
    assert resolved.routing_policy.objective == objective
    assert resolved.routing_policy.policy_sha256 == payload["policy_sha256"]
    assert resolved.manifest_sha256 == sha(package / "model_manifest.json")
    assert sha(package / "routing_policy.json") != payload["policy_sha256"]
    assert (
        verify_package(resolved.path, expected_model_version=MODEL_VERSION)["schema_version"]
        == "model-package-v2"
    )


@pytest.mark.parametrize(
    "change", ["source_scope", "internal_hash", "input_length", "objective", "missing_pool"]
)
def test_v2_rejects_unsupported_or_inconsistent_policy_even_with_valid_file_hash(
    tmp_path, policy_payload_factory, change
):
    payload = policy_payload_factory()
    package = build_v2_package(tmp_path / "package", payload)
    if change == "source_scope":
        payload["scope"] = "source-local"
    elif change == "internal_hash":
        payload["lambda_cost"] = 0.5
    elif change == "input_length":
        payload["max_length"] = 8192
    elif change == "objective":
        payload = policy_payload_factory("pairwise")
    else:
        del payload["candidate_pool"]
    rewrite_json_artifact(package, "routing_policy.json", payload)
    with pytest.raises(ModelSourceError):
        verify_package(package, expected_model_version=MODEL_VERSION)


@pytest.mark.parametrize("field", ["router_objective", "label2id"])
def test_v2_requires_explicit_checkpoint_objective_and_output_order(
    tmp_path, policy_payload_factory, field
):
    package = build_v2_package(tmp_path / "package", policy_payload_factory("pairwise"))
    config = json.loads((package / "config.json").read_text())
    del config[field]
    rewrite_json_artifact(package, "config.json", config)
    with pytest.raises(ModelSourceError):
        verify_package(package, expected_model_version=MODEL_VERSION)


def test_v1_rejects_nonordinal_checkpoint_even_when_labels_look_legacy(tmp_path):
    package = build_package(tmp_path / "package")
    config = json.loads((package / "config.json").read_text())
    config["router_objective"] = "pointwise"
    rewrite_json_artifact(package, "config.json", config)
    with pytest.raises(ModelSourceError, match="legacy v1"):
        verify_package(package, expected_model_version=MODEL_VERSION)


def test_same_physical_version_cannot_replace_its_cached_policy(tmp_path, policy_payload_factory):
    package = build_v2_package(tmp_path / "package", policy_payload_factory())
    store = MemoryPackageStore(package)
    resolve_router_model(
        package_uri=PACKAGE_URI,
        local_path=None,
        model_version=MODEL_VERSION,
        cache_dir=tmp_path / "cache",
        object_store=store,
    )
    rewrite_json_artifact(package, "routing_policy.json", policy_payload_factory(lambda_cost=0.5))
    with pytest.raises(ModelSourceError, match="manifest SHA-256"):
        resolve_router_model(
            package_uri=PACKAGE_URI,
            local_path=None,
            model_version=MODEL_VERSION,
            cache_dir=tmp_path / "cache",
            object_store=store,
            expected_manifest_sha256=sha(package / "model_manifest.json"),
        )


@pytest.mark.parametrize("objective", ["ordinal", "pointwise", "pairwise"])
def test_classifier_uses_objective_scores_and_policy_cost_before_tier_projection(
    policy_payload_factory, objective
):
    classifier = classifier_for(policy_payload_factory(objective, lambda_cost=10.0))
    result = classifier.classify_v2("question")
    assert result.action == "low"
    assert result.tier_projection == "easy"
    assert result.model == "haiku"
    assert result.confidence == (None if objective == "pairwise" else result.action_scores["low"])
    if objective == "pointwise":
        assert sum(result.action_scores.values()) > 1.0
    if objective == "pairwise":
        with pytest.raises(ModelContractError, match="classify/v2"):
            classifier.classify("question")
    else:
        compatible = classifier.classify("question")
        assert compatible.tier.value == result.tier_projection
        assert compatible.model.value == result.model
        assert compatible.confidence == result.confidence


def test_nonordinal_checkpoint_cannot_load_without_policy():
    config = SimpleNamespace(
        num_labels=3, id2label={0: "easy", 1: "medium", 2: "hard"}, router_objective="pairwise"
    )
    with pytest.raises(ModelContractError, match="explicit v2"):
        TransformersModernBertClassifier._validate_label_contract(config)


@pytest.mark.parametrize("objective", ["ordinal", "pointwise", "pairwise"])
def test_objective_label_contract_matches_policy(policy_payload_factory, objective):
    policy = RoutingPolicy.from_mapping(policy_payload_factory(objective))
    config = SimpleNamespace(
        num_labels=3,
        id2label=dict(enumerate(policy.output_label_order)),
        router_objective=objective,
    )
    TransformersModernBertClassifier._validate_label_contract(config, routing_policy=policy)
    config.router_objective = "regret"
    with pytest.raises(ModelContractError, match="objective"):
        TransformersModernBertClassifier._validate_label_contract(config, routing_policy=policy)


def test_v2_normalization_preserves_indent_and_operators_and_v1_remains_raw(policy_payload_factory):
    tokenizer = FakeTokenizer()
    classifier = classifier_for(policy_payload_factory(), tokenizer)
    raw = (
        "    \uff58\uff0b\uff59\r\n\tif \uff58\u2212\uff59:\r\n"
        "        return \uff58*\uff59\u200b  \r\n"
    )
    classifier.classify_v2(raw)
    assert tokenizer.last_message == "    x+y\n\tif x\u2212y:\n        return x*y"
    legacy = TransformersModernBertClassifier(
        tokenizer=tokenizer,
        model=FakeRouterModel(),
        torch=torch,
        device=torch.device("cpu"),
        model_version=MODEL_VERSION,
        max_length=8192,
    )
    legacy.classify(raw)
    assert tokenizer.last_message == raw
    with pytest.raises(InvalidRoutingInputError, match="empty"):
        classifier.classify_v2("\u200b")


def test_helper_has_shared_fixture_hash_and_rejects_nonfinite_scores(policy_payload_factory):
    payload = policy_payload_factory()
    assert (
        payload["policy_sha256"]
        == "be1db3bffe07d6acd2ad2da5896edfaa878144308104d1c7995dd7461d432e7d"
    )
    policy = RoutingPolicy.from_mapping(payload)
    for bad in ([float("nan"), 0, 0], [float("inf"), 0, 0]):
        with pytest.raises(RoutingPolicyError):
            route_scores(policy, bad)


@pytest.mark.parametrize("objective", ["ordinal", "pointwise", "pairwise"])
def test_v2_endpoint_retains_three_tiers_and_marks_score_semantics(
    settings, escalation_service, auth_headers, policy_payload_factory, objective
):
    engine = classifier_for(policy_payload_factory(objective))
    manager = ModelManager.loaded(embed_engine=FakeEmbedder(), router_engine=engine)
    app = main.create_app(
        settings=settings, model_manager=manager, escalation_service=escalation_service
    )
    with TestClient(app) as client:
        response = client.post(
            "/router/classify/v2", json={"message": "question"}, headers=auth_headers
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["schemaVersion"] == "router-decision-v2"
        assert (payload["action"], payload["tier"], payload["model"]) == (
            "medium",
            "medium",
            "sonnet",
        )
        assert payload["candidateModelIds"] == engine.candidate_model_ids
        assert payload["policySha256"] == policy_payload_factory(objective)["policy_sha256"]
        assert payload["objective"] == objective
        assert payload["confidence"] == (
            None if objective == "pairwise" else payload["actionScores"][payload["action"]]
        )
        assert client.post("/router/classify/v2", json={"message": "question"}).status_code == 401
        empty = client.post("/router/classify/v2", json={"message": "\u200b"}, headers=auth_headers)
        assert empty.status_code == 400 and empty.json()["code"] == "INVALID_MESSAGE"
        legacy = client.post("/router/classify", json={"message": "question"}, headers=auth_headers)
        if objective == "pairwise":
            assert legacy.status_code == 409
            assert legacy.json()["code"] == "ROUTER_API_VERSION_REQUIRED"
        else:
            assert legacy.status_code == 200
            assert legacy.json() == {
                key: payload[key] for key in ("tier", "model", "confidence", "modelVersion")
            }


def test_existing_legacy_model_is_not_silently_given_a_service_policy(client, auth_headers):
    response = client.post(
        "/router/classify/v2", json={"message": "question"}, headers=auth_headers
    )
    assert response.status_code == 503
    assert response.json()["code"] == "ROUTER_POLICY_NOT_CONFIGURED"
    assert (
        client.post(
            "/router/classify", json={"message": "question"}, headers=auth_headers
        ).status_code
        == 200
    )


def test_loader_passes_verified_policy_and_uses_per_package_input_limit(
    settings, monkeypatch, tmp_path, policy_payload_factory
):
    policy = RoutingPolicy.from_mapping(policy_payload_factory())
    resolved = ResolvedModel(
        path=tmp_path,
        model_version=MODEL_VERSION,
        package_uri=PACKAGE_URI,
        routing_policy=policy,
        manifest_sha256="a" * 64,
    )
    monkeypatch.setattr(main, "resolve_router_model", lambda **kwargs: resolved)
    captured = {}

    def load(**kwargs):
        captured.update(kwargs)
        return FakeClassifier()

    monkeypatch.setattr(main.TransformersModernBertClassifier, "load", load)
    settings.router_max_length = 8192
    manager = main._build_model_manager(settings)
    manager._router_version_loader(PACKAGE_URI, MODEL_VERSION, "a" * 64)
    assert captured["routing_policy"] is policy
    assert captured["max_length"] == 512
    assert captured["package_manifest_sha256"] == "a" * 64


def test_pointer_watcher_rejects_same_version_new_manifest_and_keeps_engine(settings, monkeypatch):
    engine = FakeClassifier()
    engine.package_manifest_sha256 = "a" * 64
    manager = ModelManager.loaded(embed_engine=FakeEmbedder(), router_engine=engine)
    settings.router_model_current_pointer_uri = "gs://fixture/models/CURRENT.json"
    pointer = CurrentModelPointer(
        model_version=engine.model_version, package_uri=PACKAGE_URI, manifest_sha256="b" * 64
    )
    monkeypatch.setattr(main, "read_current_model_pointer", lambda *args, **kwargs: pointer)
    watcher = main._build_model_pointer_watcher(settings, manager)
    with pytest.raises(ModelSourceError, match="immutable"):
        asyncio.run(watcher.check_once())
    assert manager.classification_engine() is engine
    assert "immutable" in watcher.snapshot().last_load_error
