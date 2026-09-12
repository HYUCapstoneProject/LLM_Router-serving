from __future__ import annotations

from dataclasses import dataclass

import pytest
from fastapi.testclient import TestClient

from llm_router_serving.config import Settings
from llm_router_serving.inference import (
    ClassificationResult,
    EmbeddingResult,
    RerankResult,
)
from llm_router_serving.main import create_app
from llm_router_serving.model_manager import ModelManager
from llm_router_serving.retraining.service import AcceptedRetraining
from llm_router_serving.schemas import RoutedModel, Tier


class FakeEmbedder:
    dimension = 1024
    model_version = "bge-large-en-v1.5"

    def embed(self, message: str) -> EmbeddingResult:
        vector = [0.0] * self.dimension
        vector[0] = 1.0
        return EmbeddingResult(vector=vector, model_version=self.model_version)


class FakeClassifier:
    model_version = "router-modernbert-base-routerbench-20260731-21c2f1b"

    def classify(self, message: str) -> ClassificationResult:
        return ClassificationResult(
            tier=Tier.EASY,
            model=RoutedModel.HAIKU,
            confidence=0.83,
            model_version=self.model_version,
        )


class FakeReranker:
    """Scores by shared-word overlap. Deterministic and dependency free.

    The point of these tests is the contract (alignment, ordering, failure modes),
    not the model. Real score fidelity is the offline evaluation's job.
    """

    model_version = "cross-encoder/quora-distilroberta-base@f62e7a4"

    def rerank(self, query: str, candidates: list[str]) -> RerankResult:
        query_words = set(query.lower().split())
        scores = [
            len(query_words & set(candidate.lower().split())) / max(len(query_words), 1)
            for candidate in candidates
        ]
        return RerankResult(scores=scores, model_version=self.model_version)


@dataclass
class FakeEscalationService:
    calls: int = 0

    async def submit(self, request):
        self.calls += 1
        return AcceptedRetraining(
            batch_id=request.batchId,
            pipeline_run_id="pipeline-run-123",
            logical_model_version="1.1",
            candidate_model_version="router-modernbert-feedback-v1.1-20260812-a1b2c3d4-21c2f1b",
            training_mode="minor_fine_tune",
            version_policy_mode="successfulHttpRequest",
        )


@pytest.fixture
def settings(tmp_path):
    return Settings(
        internal_api_token="test-secret",
        load_models_on_startup=False,
        retraining_db_path=tmp_path / "retraining.sqlite3",
    )


@pytest.fixture
def escalation_service():
    return FakeEscalationService()


def _client(settings, escalation_service, *, rerank_engine=None):
    manager = ModelManager.loaded(
        embed_engine=FakeEmbedder(),
        router_engine=FakeClassifier(),
        rerank_engine=rerank_engine,
    )
    app = create_app(
        settings=settings,
        model_manager=manager,
        escalation_service=escalation_service,
    )
    return TestClient(app)


@pytest.fixture
def client(settings, escalation_service):
    """Reranker disabled — the default deployment shape today."""
    with _client(settings, escalation_service) as test_client:
        yield test_client


@pytest.fixture
def rerank_client(settings, escalation_service):
    with _client(settings, escalation_service, rerank_engine=FakeReranker()) as test_client:
        yield test_client


@pytest.fixture
def auth_headers():
    return {
        "Authorization": "Bearer test-secret",
        "X-Request-ID": "req-test-123",
    }
