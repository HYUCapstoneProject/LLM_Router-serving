"""Offline contract against artifacts emitted by the sibling ML CLI chain.

Fixture weights are synthetic. Gate, package, publish, approve, CURRENT and
observability JSON were produced by real local CLIs, not hand-written schemas.
"""

import json
from pathlib import Path

from llm_router_serving.retraining.observability import RetrainingObservabilityReader


class ProducerObjects:
    def __init__(self, objects):
        self.objects = objects

    def read_bytes(self, uri):
        return self.objects[uri].encode("utf-8")


def test_ml_producer_evidence_is_accepted_without_field_translation():
    fixture = json.loads(
        (Path(__file__).parent / "fixtures/r0-promotion-producer.json").read_text()
    )
    expected = fixture["expected"]
    run_suffix = f"/{expected['batch_id']}/observability-v1"
    assert expected["observability_run_uri"].endswith(run_suffix)
    root = expected["observability_run_uri"][: -len(run_suffix)]
    reader = RetrainingObservabilityReader(
        object_store=ProducerObjects(fixture["objects"]),
        root_uri=root,
    )
    reader.verify_promotion(
        batch_id=expected["batch_id"],
        candidate_model_version=expected["candidate_model_version"],
        pipeline_run_id=expected["pipeline_job_resource_name"],
        publish_root_uri=expected["model_publish_root_uri"],
    )
