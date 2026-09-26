import importlib.util
import io
import json
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "desired,serving,status,ready",
    [
        ("expected", "expected", "inSync", True),
        ("expected", "old", "loading", False),
        ("wrong", "wrong", "inSync", False),
        ("expected", "expected", "loadFailed", False),
    ],
)
def test_readiness_requires_expected_current_and_serving(
    monkeypatch, capsys, desired, serving, status, ready
):
    path = Path(__file__).resolve().parents[1] / "scripts/check-r0-readiness.py"
    spec = importlib.util.spec_from_file_location("readiness", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setenv("LLM_ROUTER_INTERNAL_API_TOKEN", "secret-not-output")
    monkeypatch.setattr("sys.argv", ["check", "--expected-model-version", "expected"])
    responses = iter(
        [
            {"status": "ok"},
            {
                "status": status,
                "serving": {"modelVersion": serving},
                "desired": {"modelVersion": desired},
                "lastLoadError": None,
            },
        ]
    )

    def open_request(request, timeout):
        assert request.get_method() == "GET"
        assert request.get_header("Authorization") == "Bearer secret-not-output"
        return io.BytesIO(json.dumps(next(responses)).encode())

    monkeypatch.setattr(module.urllib.request, "urlopen", open_request)
    assert module.main() == (0 if ready else 1)
    output = capsys.readouterr().out
    assert "secret-not-output" not in output
    assert json.loads(output)["http_ready"] is ready
    assert json.loads(output)["pipeline_verified"] is False
