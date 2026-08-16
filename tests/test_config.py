from llm_router_serving.config import Settings


def test_csv_environment_settings_are_parsed(monkeypatch):
    monkeypatch.setenv("LLM_ROUTER_ALLOWED_GCS_BUCKETS", "first-bucket,second-bucket")
    monkeypatch.setenv(
        "LLM_ROUTER_SUPPORTED_FEEDBACK_SCHEMAS",
        "feedback-export-v0.1,feedback-export-v0.2",
    )

    settings = Settings(_env_file=None)

    assert settings.allowed_gcs_buckets == ("first-bucket", "second-bucket")
    assert settings.supported_feedback_schemas == (
        "feedback-export-v0.1",
        "feedback-export-v0.2",
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
