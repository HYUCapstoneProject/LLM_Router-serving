from __future__ import annotations

import argparse
import json
import math
import time

from fastapi.testclient import TestClient

from llm_router_serving.config import Settings
from llm_router_serving.inference import (
    ClassificationResult,
    TransformersBgeEmbedder,
)
from llm_router_serving.main import create_app
from llm_router_serving.model_manager import ModelManager
from llm_router_serving.schemas import RoutedModel, Tier


class SmokeRouter:
    model_version = "router-smoke-only"

    def classify(self, message: str) -> ClassificationResult:
        return ClassificationResult(
            tier=Tier.EASY,
            model=RoutedModel.HAIKU,
            confidence=1.0,
            model_version=self.model_version,
        )


def cosine(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


def norm(vector: list[float]) -> float:
    return math.sqrt(sum(value * value for value in vector))


def build_parser() -> argparse.ArgumentParser:
    defaults = Settings(_env_file=None)
    parser = argparse.ArgumentParser(description="Smoke-test the production BGE embedding contract")
    parser.add_argument("--model-id", default=defaults.embed_model_id)
    parser.add_argument("--revision", default=defaults.embed_model_revision)
    parser.add_argument("--model-version", default=defaults.embed_model_version)
    parser.add_argument("--device", default=defaults.model_device)
    parser.add_argument("--local-files-only", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    load_started = time.perf_counter()
    embedder = TransformersBgeEmbedder.load(
        model_id=args.model_id,
        revision=args.revision or None,
        model_version=args.model_version,
        expected_dimension=1024,
        max_length=512,
        device_name=args.device,
        local_files_only=args.local_files_only,
    )
    load_seconds = time.perf_counter() - load_started

    original = "How can I reset my account password?"
    paraphrase = "What steps should I follow to change my password?"
    unrelated = "What is the capital city of France?"

    inference_started = time.perf_counter()
    original_vector = embedder.embed(original).vector
    repeated_vector = embedder.embed(original).vector
    paraphrase_vector = embedder.embed(paraphrase).vector
    unrelated_vector = embedder.embed(unrelated).vector
    inference_seconds = time.perf_counter() - inference_started

    original_norm = norm(original_vector)
    repeated_similarity = cosine(original_vector, repeated_vector)
    paraphrase_similarity = cosine(original_vector, paraphrase_vector)
    unrelated_similarity = cosine(original_vector, unrelated_vector)

    assert len(original_vector) == 1024
    assert math.isclose(original_norm, 1.0, rel_tol=1e-5, abs_tol=1e-5)
    assert repeated_similarity > 0.99999
    assert paraphrase_similarity > unrelated_similarity

    settings = Settings(
        _env_file=None,
        load_models_on_startup=False,
        embed_expected_dim=1024,
    )
    manager = ModelManager.loaded(embed_engine=embedder, router_engine=SmokeRouter())
    app = create_app(settings=settings, model_manager=manager)
    with TestClient(app) as client:
        response = client.post("/router/embed", json={"message": original})
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["dim"] == 1024
    assert payload["modelVersion"] == args.model_version
    assert len(payload["embedding"]) == 1024

    print(
        json.dumps(
            {
                "status": "ok",
                "modelId": args.model_id,
                "revision": args.revision,
                "modelVersion": args.model_version,
                "device": embedder.device,
                "dimension": len(original_vector),
                "norm": original_norm,
                "sameQuestionSimilarity": repeated_similarity,
                "paraphraseSimilarity": paraphrase_similarity,
                "unrelatedSimilarity": unrelated_similarity,
                "loadSeconds": load_seconds,
                "fourInferencesSeconds": inference_seconds,
                "apiStatus": response.status_code,
                "apiDimension": payload["dim"],
                "apiModelVersion": payload["modelVersion"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
