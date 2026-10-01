from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from llm_router_serving.errors import ModelContractError
from llm_router_serving.inference import (
    TransformersBgeEmbedder,
    TransformersModernBertClassifier,
)


class FakeTokenizer:
    def __init__(self):
        self.last_message = None

    def __call__(self, message, **kwargs):
        assert message
        assert kwargs["truncation"] is True
        self.last_message = message
        return {"input_ids": torch.tensor([[1, 2]])}


class FakeBgeModel:
    def __call__(self, **inputs):
        assert "input_ids" in inputs
        hidden = torch.zeros((1, 2, 1024), dtype=torch.float32)
        hidden[0, 0, 0] = 3.0
        hidden[0, 0, 1] = 4.0
        hidden[0, 1, 0] = 100.0
        return SimpleNamespace(last_hidden_state=hidden)


class FakeRouterModel:
    def __call__(self, **inputs):
        assert "input_ids" in inputs
        return SimpleNamespace(logits=torch.tensor([[0.1, 2.0, 0.3]]))


def test_bge_inference_uses_cls_pooling_and_l2_normalization():
    tokenizer = FakeTokenizer()
    embedder = TransformersBgeEmbedder(
        tokenizer=tokenizer,
        model=FakeBgeModel(),
        torch=torch,
        device=torch.device("cpu"),
        dimension=1024,
        model_version="bge-large-en-v1.5",
        max_length=512,
    )

    result = embedder.embed("question")

    assert tokenizer.last_message == "question"
    assert result.vector[:2] == pytest.approx([0.6, 0.8])
    assert result.vector[2:] == [0.0] * 1022


def test_router_inference_returns_argmax_tier_and_confidence():
    classifier = TransformersModernBertClassifier(
        tokenizer=FakeTokenizer(),
        model=FakeRouterModel(),
        torch=torch,
        device=torch.device("cpu"),
        model_version="router-modernbert-base-routerbench-20260731-21c2f1b",
        max_length=8192,
    )

    result = classifier.classify("question")

    assert result.tier.value == "medium"
    assert result.model.value == "sonnet"
    assert result.confidence == pytest.approx(torch.softmax(torch.tensor([0.1, 2.0, 0.3]), 0)[1])


def test_modernbert_label_contract_accepts_expected_order():
    config = SimpleNamespace(
        num_labels=3,
        id2label={0: "easy", 1: "medium", 2: "hard"},
    )

    TransformersModernBertClassifier._validate_label_contract(config)


def test_modernbert_label_contract_rejects_wrong_order():
    config = SimpleNamespace(
        num_labels=3,
        id2label={0: "hard", 1: "medium", 2: "easy"},
    )

    with pytest.raises(ModelContractError, match="label order"):
        TransformersModernBertClassifier._validate_label_contract(config)


def test_bge_token_count_includes_special_tokens_and_does_not_truncate():
    class CountingTokenizer:
        def __call__(self, message, **kwargs):
            assert "truncation" not in kwargs
            assert kwargs["add_special_tokens"] is True
            return {"input_ids": list(range(len(message.split()) + 2))}

    embedder = TransformersBgeEmbedder(
        tokenizer=CountingTokenizer(),
        model=FakeBgeModel(),
        torch=torch,
        device=torch.device("cpu"),
        dimension=1024,
        model_version="bge-large-en-v1.5",
        max_length=512,
    )

    assert embedder.count_tokens(" ".join(["word"] * 600)) == 602
