from llm_router_serving.config import Settings


def test_csv_environment_settings_are_parsed(monkeypatch):
    monkeypatch.setenv("LLM_ROUTER_ALLOWED_GCS_BUCKETS", "first-bucket,second-bucket")
    monkeypatch.setenv(
        "LLM_ROUTER_SUPPORTED_FEEDBACK_SCHEMAS",
        "feedback-inline-v1,feedback-inline-v2",
    )

    settings = Settings(_env_file=None)

    assert settings.allowed_gcs_buckets == ("first-bucket", "second-bucket")
    assert settings.supported_feedback_schemas == (
        "feedback-inline-v1",
        "feedback-inline-v2",
    )


def test_bge_defaults_are_pinned_to_the_approved_artifact():
    settings = Settings(_env_file=None)

    assert settings.embed_model_id == "BAAI/bge-large-en-v1.5"
    assert settings.embed_model_revision == "d4aa6901d3a41ba39fb536a557fa166f842b0e09"
    assert settings.embed_model_version == "bge-large-en-v1.5"
    assert settings.embed_expected_dim == 1024


def test_blank_local_model_path_does_not_shadow_the_package_uri(monkeypatch):
    # An unset LLM_ROUTER_ROUTER_MODEL_PATH= would otherwise parse as Path("."),
    # which reads as a configured local source and blocks the GCS package route.
    monkeypatch.setenv("LLM_ROUTER_ROUTER_MODEL_PATH", "")
    monkeypatch.setenv(
        "LLM_ROUTER_ROUTER_MODEL_PACKAGE_URI", "gs://bucket/models/router/v1/approved"
    )
    monkeypatch.setenv(
        "LLM_ROUTER_ROUTER_MODEL_VERSION", "router-modernbert-base-routerbench-20260814-45dcc6d"
    )

    settings = Settings(_env_file=None)

    assert settings.router_model_path is None
    assert settings.router_model_source_errors() == []


def test_current_pointer_is_a_complete_router_source(monkeypatch):
    monkeypatch.setenv("LLM_ROUTER_ROUTER_MODEL_PATH", "")
    monkeypatch.setenv("LLM_ROUTER_ROUTER_MODEL_PACKAGE_URI", "")
    monkeypatch.setenv("LLM_ROUTER_ROUTER_MODEL_VERSION", "")
    monkeypatch.setenv(
        "LLM_ROUTER_ROUTER_MODEL_CURRENT_POINTER_URI",
        "gs://bucket/models/router-modernbert/CURRENT.json",
    )

    settings = Settings(_env_file=None)

    assert settings.router_model_source_errors() == []


def test_local_model_does_not_implicitly_follow_publish_root(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_ROUTER_ROUTER_MODEL_PATH", str(tmp_path))
    monkeypatch.setenv(
        "LLM_ROUTER_ROUTER_MODEL_VERSION",
        "router-modernbert-base-routerbench-20260731-21c2f1b",
    )
    monkeypatch.setenv(
        "LLM_ROUTER_MODEL_PUBLISH_ROOT_URI",
        "gs://bucket/models/router-modernbert",
    )
    monkeypatch.delenv("LLM_ROUTER_ROUTER_MODEL_PACKAGE_URI", raising=False)
    monkeypatch.delenv("LLM_ROUTER_ROUTER_MODEL_CURRENT_POINTER_URI", raising=False)

    settings = Settings(_env_file=None)

    assert settings.current_model_pointer_uri() is None
    assert settings.router_model_source_errors() == []


def test_router_model_version_accepts_feedback_batch_digest():
    settings = Settings(
        _env_file=None,
        router_model_version=("router-modernbert-base-feedback-20260906-acde1234-21c2f1b"),
    )

    assert settings.router_model_version.endswith("acde1234-21c2f1b")
