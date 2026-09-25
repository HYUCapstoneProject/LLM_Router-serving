"""Real ML assembly output with synthetic weights; no service approval is implied."""

import hashlib
import json
from pathlib import Path

import pytest

from llm_router_serving.model_source import ModelSourceError, resolve_router_model, verify_package


@pytest.mark.parametrize("schema", ["v1", "v2"])
def test_ml_assembler_length_package_loads_without_contract_translation(tmp_path, schema):
    fixture = json.loads(
        (Path(__file__).parent / "fixtures/length-contract-producer-v1.json").read_text()
    )
    assert fixture["synthetic_only"] is True
    package = fixture["packages"][schema]
    directory = tmp_path / "package"
    directory.mkdir()
    for filename, content in package["files"].items():
        (directory / filename).write_text(content)
    manifest = verify_package(
        directory, expected_model_version=package["model_version"], require_approved=False
    )
    assert manifest["training_max_length"] == 512
    assert manifest["inference_max_length"] == 8192
    resolved = resolve_router_model(
        package_uri=None, local_path=directory, model_version=package["model_version"],
        cache_dir=tmp_path / "cache", require_approved=False,
    )
    assert resolved.training_max_length == 512
    assert resolved.max_length == 8192
    assert resolved.manifest_sha256 == hashlib.sha256(
        (directory / "model_manifest.json").read_bytes()
    ).hexdigest()
    if schema == "v2":
        assert resolved.routing_policy.max_length == 8192
        assert resolved.routing_policy.policy_sha256 == (
            "3a4294af6a02b5c856fddc318735ac321b9bdee51db9022f9a9c381b70e42f2e"
        )
    # Assembly only creates candidates; the runtime's default still requires approval.
    with pytest.raises(ModelSourceError, match="gate_status"):
        verify_package(directory, expected_model_version=package["model_version"])
