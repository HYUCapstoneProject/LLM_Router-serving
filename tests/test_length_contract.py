"""Package, policy and runtime contracts for independently qualified inference lengths."""

import hashlib
import json
from types import SimpleNamespace

import pytest
import torch
from conftest import FakeEmbedder
from fastapi.testclient import TestClient
from test_inference import FakeRouterModel, FakeTokenizer
from test_model_source import MODEL_VERSION, PACKAGE_URI, MemoryPackageStore, build_package
from test_routing_policy_v2 import build_v2_package, rewrite_json_artifact

from llm_router_serving import main
from llm_router_serving.errors import ModelContractError
from llm_router_serving.inference import TransformersModernBertClassifier
from llm_router_serving.model_manager import ModelManager
from llm_router_serving.model_source import ModelSourceError, resolve_router_model, verify_package
from llm_router_serving.routing_policy import (
    LENGTH_CONTRACT_VERSION,
    RoutingPolicy,
    RoutingPolicyError,
    compute_policy_sha256,
)


def length_policy(factory, *, inference_length=8192):
    payload = factory()
    payload.update(length_contract_version=LENGTH_CONTRACT_VERSION, max_length=inference_length)
    payload["policy_sha256"] = compute_policy_sha256(payload)
    return payload


def length_package(path, factory, *, schema="v2", inference_length=8192, training_length=512):
    if schema == "v2":
        build_v2_package(path, length_policy(factory, inference_length=inference_length))
    else:
        build_package(path)
    config = json.loads((path / "config.json").read_text())
    config["max_position_embeddings"] = 8192
    rewrite_json_artifact(path, "config.json", config)
    training = {
        "max_length": training_length,
        "objective_method": "pointwise" if schema == "v2" else "ordinal",
        "input_contract": "structure-preserving-v2",
    }
    rewrite_json_artifact(path, "training_config.json", training)
    manifest_path = path / "model_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.update(
        input_contract="structure-preserving-v2",
        length_contract_version=LENGTH_CONTRACT_VERSION,
        training_max_length=training_length,
        inference_max_length=inference_length,
        tokenizer_max_length=inference_length,
    )
    manifest_path.write_text(json.dumps(manifest))
    return path


@pytest.mark.parametrize("schema", ["v1", "v2"])
@pytest.mark.parametrize("inference_length", [512, 8192])
@pytest.mark.parametrize("training_length", [512, 8192])
def test_training_history_and_inference_length_are_independent(
    tmp_path, policy_payload_factory, schema, inference_length, training_length
):
    path = length_package(
        tmp_path / "package", policy_payload_factory, schema=schema,
        inference_length=inference_length, training_length=training_length,
    )
    resolved = resolve_router_model(
        package_uri=None, local_path=path, model_version=MODEL_VERSION, cache_dir=tmp_path / "cache"
    )
    assert resolved.max_length == inference_length
    assert resolved.training_max_length == training_length
    assert resolved.input_contract == "structure-preserving-v2"
    if schema == "v2":
        assert resolved.routing_policy.max_length == inference_length
        assert resolved.routing_policy.length_contract_version == LENGTH_CONTRACT_VERSION


@pytest.mark.parametrize("schema", ["v1", "v2"])
@pytest.mark.parametrize(
    ("target", "field", "value"),
    [
        ("manifest", "length_contract_version", None),
        ("manifest", "length_contract_version", "unknown"),
        ("manifest", "training_max_length", True),
        ("manifest", "training_max_length", "512"),
        ("manifest", "training_max_length", 8192),
        ("manifest", "inference_max_length", 4096),
        ("manifest", "inference_max_length", "8192"),
        ("manifest", "inference_max_length", True),
        ("manifest", "tokenizer_max_length", 512),
        ("manifest", "tokenizer_max_length", 8192.0),
        ("manifest", "input_contract", "legacy-word-only-v1"),
        ("training_config.json", "max_length", 512.0),
        ("training_config.json", "max_length", True),
        ("training_config.json", "input_contract", "legacy-word-only-v1"),
        ("config.json", "max_position_embeddings", 4096),
        ("config.json", "max_position_embeddings", "8192"),
        ("config.json", "max_position_embeddings", True),
        ("config.json", "max_position_embeddings", None),
    ],
)
def test_mismatched_lengths_are_rejected_even_with_valid_artifact_checksums(
    tmp_path, policy_payload_factory, schema, target, field, value
):
    path = length_package(tmp_path / "package", policy_payload_factory, schema=schema)
    filename = "model_manifest.json" if target == "manifest" else target
    payload = json.loads((path / filename).read_text())
    payload[field] = value
    if target == "manifest":
        (path / filename).write_text(json.dumps(payload))
    else:
        rewrite_json_artifact(path, filename, payload)
    with pytest.raises(ModelSourceError):
        verify_package(path, expected_model_version=MODEL_VERSION)


@pytest.mark.parametrize(
    "field", ["length_contract_version", "training_max_length", "inference_max_length"]
)
def test_new_length_contract_cannot_be_partially_removed(tmp_path, policy_payload_factory, field):
    path = length_package(tmp_path / "package", policy_payload_factory)
    filename = path / "model_manifest.json"
    manifest = json.loads(filename.read_text())
    del manifest[field]
    filename.write_text(json.dumps(manifest))
    with pytest.raises(ModelSourceError):
        verify_package(path, expected_model_version=MODEL_VERSION)


def test_policy_and_manifest_markers_must_match(tmp_path, policy_payload_factory):
    path = length_package(tmp_path / "package", policy_payload_factory, inference_length=512)
    rewrite_json_artifact(path, "routing_policy.json", policy_payload_factory())
    with pytest.raises(ModelSourceError, match="inference length contracts"):
        verify_package(path, expected_model_version=MODEL_VERSION)


@pytest.mark.parametrize("value", [True, "8192", 8192.0, 4096])
def test_policy_rejects_unqualified_or_nonstrict_lengths(policy_payload_factory, value):
    payload = length_policy(policy_payload_factory)
    payload["max_length"] = value
    with pytest.raises(RoutingPolicyError):
        compute_policy_sha256(payload)


def test_8192_policy_requires_marker_and_changes_canonical_hash(policy_payload_factory):
    original = policy_payload_factory()
    upgraded = length_policy(policy_payload_factory)
    assert original["policy_sha256"] != upgraded["policy_sha256"]
    assert RoutingPolicy.from_mapping(upgraded).max_length == 8192
    del upgraded["length_contract_version"]
    with pytest.raises(RoutingPolicyError):
        compute_policy_sha256(upgraded)


@pytest.mark.parametrize("schema", ["v1", "v2"])
async def test_startup_reload_and_failed_reload_use_the_verified_package_length(
    settings, tmp_path, monkeypatch, policy_payload_factory, schema
):
    path = length_package(tmp_path / "package", policy_payload_factory, schema=schema)
    store = MemoryPackageStore(path)
    monkeypatch.setattr(main, "GcsPackageStore", lambda **kwargs: store)
    monkeypatch.setattr(main.TransformersBgeEmbedder, "load", lambda **kwargs: FakeEmbedder())
    settings.router_model_path = path
    settings.router_model_version = MODEL_VERSION
    settings.router_model_cache_dir = tmp_path / "cache"
    settings.router_max_length = 512  # Environment defaults cannot downgrade a verified package.
    tokenizers = []

    class RecordingTokenizer(FakeTokenizer):
        def __call__(self, message, **kwargs):
            self.limit = kwargs["max_length"]
            return super().__call__(message, **kwargs)

    def load(**kwargs):
        tokenizer = RecordingTokenizer()
        tokenizers.append(tokenizer)
        model = FakeRouterModel()
        model.config = SimpleNamespace(
            router_objective="pointwise" if schema == "v2" else "ordinal",
            max_position_embeddings=8192,
        )
        return TransformersModernBertClassifier(
            tokenizer=tokenizer, model=model, torch=torch, device=torch.device("cpu"),
            **{
                key: value for key, value in kwargs.items()
                if key not in {"model_path", "device_name"}
            },
        )

    monkeypatch.setattr(main.TransformersModernBertClassifier, "load", load)
    manager = main._build_model_manager(settings)
    await manager.load_all()
    original = manager.classification_engine()
    assert original is not None

    async def reload_as(version, *, invalid=False):
        filename = path / "model_manifest.json"
        manifest = json.loads(filename.read_text())
        manifest["model_version"] = version
        if invalid:
            manifest["inference_max_length"] = 4096
        filename.write_text(json.dumps(manifest))
        store._prefix = PACKAGE_URI.replace(MODEL_VERSION, version)
        return await manager.reload_router(
            package_uri=store._prefix, model_version=version,
            manifest_sha256=hashlib.sha256(filename.read_bytes()).hexdigest(),
        )

    assert await reload_as("router-reloaded-8192") is True
    replacement = manager.classification_engine()
    assert replacement is not original
    for engine in (original, replacement):
        classify = engine.classify_v2 if schema == "v2" else engine.classify
        classify("    x+y\r\n\treturn x*y\x00")
    assert [tokenizer.limit for tokenizer in tokenizers] == [8192, 8192]
    assert all(tokenizer.last_message == "    x+y\n\treturn x*y" for tokenizer in tokenizers)
    with pytest.raises(ModelContractError):
        await reload_as("router-invalid-length", invalid=True)
    assert manager.classification_engine() is replacement
    assert manager.snapshot().previous_router_model_version == MODEL_VERSION


def test_runtime_rejects_length_exceeding_model_native_limit():
    model = FakeRouterModel()
    model.config = SimpleNamespace(max_position_embeddings=4096)
    with pytest.raises(ModelContractError, match="native limit"):
        TransformersModernBertClassifier(
            tokenizer=FakeTokenizer(), model=model, torch=torch,
            device=torch.device("cpu"), model_version=MODEL_VERSION, max_length=8192,
        )


def test_structured_v1_empty_input_is_client_error(settings, escalation_service, auth_headers):
    engine = TransformersModernBertClassifier(
        tokenizer=FakeTokenizer(), model=FakeRouterModel(), torch=torch,
        device=torch.device("cpu"), model_version=MODEL_VERSION, max_length=8192,
        input_contract="structure-preserving-v2",
    )
    app = main.create_app(
        settings=settings, escalation_service=escalation_service,
        model_manager=ModelManager.loaded(embed_engine=FakeEmbedder(), router_engine=engine),
    )
    with TestClient(app) as client:
        response = client.post(
            "/router/classify", json={"message": "\u200b"}, headers=auth_headers
        )
    assert response.status_code == 400
    assert response.json()["code"] == "INVALID_MESSAGE"


def test_prior_structured_v1_512_package_remains_readable(tmp_path):
    path = build_package(
        tmp_path / "package", input_contract="structure-preserving-v2", tokenizer_max_length=512
    )
    rewrite_json_artifact(path, "training_config.json", {"max_length": 512})
    resolved = resolve_router_model(
        package_uri=None, local_path=path, model_version=MODEL_VERSION, cache_dir=tmp_path / "cache"
    )
    assert resolved.max_length == 512
    assert resolved.input_contract == "structure-preserving-v2"
    assert resolved.training_max_length is None
