from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, ClassVar, Protocol

from llm_router_serving.errors import ModelContractError
from llm_router_serving.retraining.structured_text import clean_text
from llm_router_serving.routing_policy import (
    RoutingDecision,
    RoutingPolicy,
    RoutingPolicyError,
    route_scores,
)
from llm_router_serving.schemas import RoutedModel, Tier

# The embedder and the classifier are loaded concurrently on startup, each in its
# own worker thread. `transformers` resolves its public names lazily, so two
# threads pulling different names at once can observe a submodule that the other
# thread is still importing and fail with a spurious ImportError. Import the
# symbols under one lock so no thread ever sees a half-initialized module.
_TRANSFORMERS_IMPORT_LOCK = Lock()


class InvalidRoutingInputError(ValueError):
    """The v2 input is empty after the frozen text normalization contract."""


@dataclass(frozen=True)
class EmbeddingResult:
    vector: list[float]
    model_version: str


@dataclass(frozen=True)
class ClassificationResult:
    tier: Tier
    model: RoutedModel
    confidence: float
    model_version: str


@dataclass(frozen=True)
class RerankResult:
    scores: list[float]
    model_version: str


class EmbeddingEngine(Protocol):
    dimension: int
    model_version: str

    def embed(self, message: str) -> EmbeddingResult: ...


class ClassificationEngine(Protocol):
    model_version: str

    def classify(self, message: str) -> ClassificationResult: ...


class RerankEngine(Protocol):
    model_version: str

    def rerank(self, query: str, candidates: list[str]) -> RerankResult: ...


def resolve_torch_device(torch: Any, configured: str) -> Any:
    if configured != "auto":
        return torch.device(configured)
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class TransformersBgeEmbedder:
    def __init__(
        self,
        *,
        tokenizer: Any,
        model: Any,
        torch: Any,
        device: Any,
        dimension: int,
        model_version: str,
        max_length: int,
    ) -> None:
        self._tokenizer = tokenizer
        self._model = model
        self._torch = torch
        self._device = device
        self.dimension = dimension
        self.model_version = model_version
        self._max_length = max_length

    @property
    def device(self) -> str:
        return str(self._device)

    @classmethod
    def load(
        cls,
        *,
        model_id: str,
        revision: str | None,
        model_version: str,
        expected_dimension: int,
        max_length: int,
        device_name: str,
        local_files_only: bool = False,
    ) -> TransformersBgeEmbedder:
        with _TRANSFORMERS_IMPORT_LOCK:
            import torch
            from transformers import AutoModel, AutoTokenizer

        common_kwargs: dict[str, object] = {"local_files_only": local_files_only}
        if revision:
            common_kwargs["revision"] = revision
        tokenizer = AutoTokenizer.from_pretrained(model_id, **common_kwargs)
        model = AutoModel.from_pretrained(
            model_id,
            use_safetensors=True,
            **common_kwargs,
        )
        device = resolve_torch_device(torch, device_name)
        model.to(device)
        model.eval()

        configured_dimension = int(getattr(model.config, "hidden_size", 0))
        if configured_dimension != expected_dimension:
            raise ModelContractError(
                "BGE hidden size does not match the API contract: "
                f"expected {expected_dimension}, got {configured_dimension}"
            )
        return cls(
            tokenizer=tokenizer,
            model=model,
            torch=torch,
            device=device,
            dimension=expected_dimension,
            model_version=model_version,
            max_length=max_length,
        )

    def embed(self, message: str) -> EmbeddingResult:
        inputs = self._tokenizer(
            message,
            return_tensors="pt",
            truncation=True,
            max_length=self._max_length,
        )
        inputs = {name: value.to(self._device) for name, value in inputs.items()}
        with self._torch.inference_mode():
            outputs = self._model(**inputs)
            cls_embedding = outputs.last_hidden_state[:, 0]
            normalized = self._torch.nn.functional.normalize(cls_embedding, p=2, dim=1)

        values = normalized[0].detach().to("cpu", dtype=self._torch.float32).tolist()
        if len(values) != self.dimension:
            raise ModelContractError(
                f"BGE returned {len(values)} dimensions; expected {self.dimension}"
            )
        if not all(math.isfinite(value) for value in values):
            raise ModelContractError("BGE returned a non-finite embedding value")
        norm = math.sqrt(sum(value * value for value in values))
        if not math.isclose(norm, 1.0, rel_tol=1e-5, abs_tol=1e-5):
            raise ModelContractError(f"BGE returned a non-normalized vector (norm={norm})")
        return EmbeddingResult(vector=values, model_version=self.model_version)


class TransformersModernBertClassifier:
    _EXPECTED_LABELS: ClassVar[tuple[Tier, Tier, Tier]] = (
        Tier.EASY,
        Tier.MEDIUM,
        Tier.HARD,
    )
    _MODEL_HINTS: ClassVar[dict[Tier, RoutedModel]] = {
        Tier.EASY: RoutedModel.HAIKU,
        Tier.MEDIUM: RoutedModel.SONNET,
        Tier.HARD: RoutedModel.OPUS,
    }

    def __init__(
        self,
        *,
        tokenizer: Any,
        model: Any,
        torch: Any,
        device: Any,
        model_version: str,
        max_length: int,
        routing_policy: RoutingPolicy | None = None,
        package_manifest_sha256: str | None = None,
    ) -> None:
        self._tokenizer = tokenizer
        self._model = model
        self._torch = torch
        self._device = device
        self.model_version = model_version
        self._max_length = max_length
        if routing_policy is not None and max_length != routing_policy.max_length:
            raise ModelContractError("Router max_length differs from the package routing policy")
        self._routing_policy = routing_policy
        self._objective = getattr(getattr(model, "config", None), "router_objective", "ordinal")
        self.package_manifest_sha256 = package_manifest_sha256

    @property
    def requires_v2(self) -> bool:
        return self._routing_policy is not None or self._objective != "ordinal"

    @property
    def supports_routing_v2(self) -> bool:
        return self._routing_policy is not None

    @property
    def candidate_model_ids(self) -> dict[str, str]:
        if self._routing_policy is None:
            raise ModelContractError("An explicit service routing policy is not configured")
        return {
            candidate.action: candidate.model_id for candidate in self._routing_policy.candidates
        }

    @classmethod
    def load(
        cls,
        *,
        model_path: Path,
        model_version: str,
        max_length: int,
        device_name: str,
        routing_policy: RoutingPolicy | None = None,
        package_manifest_sha256: str | None = None,
    ) -> TransformersModernBertClassifier:
        with _TRANSFORMERS_IMPORT_LOCK:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        cls._validate_label_contract(model.config, routing_policy=routing_policy)
        device = resolve_torch_device(torch, device_name)
        model.to(device)
        model.eval()
        return cls(
            tokenizer=tokenizer,
            model=model,
            torch=torch,
            device=device,
            model_version=model_version,
            max_length=max_length,
            routing_policy=routing_policy,
            package_manifest_sha256=package_manifest_sha256,
        )

    @classmethod
    def _validate_label_contract(
        cls, config: Any, *, routing_policy: RoutingPolicy | None = None
    ) -> None:
        if int(getattr(config, "num_labels", 0)) != 3:
            raise ModelContractError("ModernBERT checkpoint must have exactly three labels")
        raw_id2label = getattr(config, "id2label", {}) or {}
        actual = tuple(str(raw_id2label.get(index, "")).lower() for index in range(3))
        objective = getattr(config, "router_objective", "ordinal")
        if routing_policy is None and objective != "ordinal":
            raise ModelContractError(
                "Non-ordinal checkpoints require an explicit v2 routing policy"
            )
        if routing_policy is not None and routing_policy.objective != objective:
            raise ModelContractError("Checkpoint objective differs from its routing policy")
        expected = (
            routing_policy.output_label_order
            if routing_policy is not None
            else tuple(label.value for label in cls._EXPECTED_LABELS)
        )
        if actual != expected:
            raise ModelContractError(
                f"ModernBERT label order must be {expected}; checkpoint contains {actual}"
            )

    def classify(self, message: str) -> ClassificationResult:
        if self.requires_v2:
            raise ModelContractError("This router requires /router/classify/v2")
        logits = self._predict_logits(message)
        probabilities = self._torch.softmax(logits, dim=-1)[0]
        label_id = int(self._torch.argmax(probabilities).item())
        confidence = float(probabilities[label_id].item())

        tier = self._EXPECTED_LABELS[label_id]
        return ClassificationResult(
            tier=tier,
            model=self._MODEL_HINTS[tier],
            confidence=confidence,
            model_version=self.model_version,
        )

    def _predict_logits(self, message: str) -> Any:
        if self._routing_policy is not None:
            message = clean_text(message)
            if not message.strip():
                raise InvalidRoutingInputError("message is empty after routing normalization")
        inputs = self._tokenizer(
            message,
            return_tensors="pt",
            truncation=True,
            max_length=self._max_length,
        )
        inputs = {name: value.to(self._device) for name, value in inputs.items()}
        with self._torch.inference_mode():
            logits = self._model(**inputs).logits
        if tuple(logits.shape) != (1, 3) or not bool(self._torch.isfinite(logits).all()):
            raise ModelContractError("Router must return three finite logits for one question")
        return logits

    def classify_v2(self, message: str) -> RoutingDecision:
        if self._routing_policy is None:
            raise ModelContractError("An explicit service routing policy is not configured")
        logits = (
            self._predict_logits(message)[0].detach().to("cpu", dtype=self._torch.float32).tolist()
        )
        try:
            return route_scores(self._routing_policy, logits)
        except RoutingPolicyError as exc:
            raise ModelContractError(str(exc)) from exc


class TransformersCrossEncoderReranker:
    """Cross-encoder that rescores pgvector candidates against the query.

    The bi-encoder embeds each text on its own, so swapping one entity
    (Python to Java, insertion to deletion) barely moves the vector. Measured on
    92 questions, paraphrases bottomed out at 0.8621 while hard negatives topped
    out at 0.9371: the two classes overlap and no cosine threshold separates them.
    A cross-encoder reads both texts together, which is exactly the case it fixes.

    **Scoring must reproduce the offline screening exactly**, because the
    candidate threshold was measured there and means nothing under a different
    activation. The screening ran ``sentence_transformers.CrossEncoder.predict``,
    which for a one-logit checkpoint applies a sigmoid — every recorded score sits
    in [0, 1]. That is reproduced here directly rather than by depending on
    sentence-transformers, whose current release requires transformers>=5 and
    would drag the BGE embedder and the ModernBERT classifier along with it.

    The equality is not assumed: ``scripts/verify_reranker_parity.py`` replays the
    58 screening pairs against the stored scores.
    """

    def __init__(
        self,
        *,
        tokenizer: Any,
        model: Any,
        torch: Any,
        device: Any,
        model_version: str,
        max_length: int,
    ) -> None:
        self._tokenizer = tokenizer
        self._model = model
        self._torch = torch
        self._device = device
        self.model_version = model_version
        self._max_length = max_length

    @classmethod
    def load(
        cls,
        *,
        model_id: str,
        model_version: str | None,
        max_length: int,
        device_name: str,
        local_files_only: bool = True,
    ) -> TransformersCrossEncoderReranker:
        with _TRANSFORMERS_IMPORT_LOCK:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=local_files_only)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_id, local_files_only=local_files_only
        )
        cls._validate_single_logit(model.config)
        device = resolve_torch_device(torch, device_name)
        model.to(device)
        model.eval()
        return cls(
            tokenizer=tokenizer,
            model=model,
            torch=torch,
            device=device,
            model_version=model_version or model_id,
            max_length=max_length,
        )

    @staticmethod
    def _validate_single_logit(config: Any) -> None:
        """A sigmoid over logit 0 is only meaningful for a one-logit head.

        Swap in a two-label checkpoint and the scores would still look plausible
        while meaning something else entirely, so refuse to load it.
        """
        num_labels = int(getattr(config, "num_labels", 0))
        if num_labels != 1:
            raise ModelContractError(
                f"reranker checkpoint must expose exactly one logit; found {num_labels}"
            )

    def rerank(self, query: str, candidates: list[str]) -> RerankResult:
        inputs = self._tokenizer(
            [query] * len(candidates),
            candidates,
            padding=True,
            truncation=True,
            max_length=self._max_length,
            return_tensors="pt",
        )
        inputs = {name: value.to(self._device) for name, value in inputs.items()}
        with self._torch.inference_mode():
            logits = self._model(**inputs).logits
            probabilities = self._torch.sigmoid(logits[:, 0])

        scores: list[float] = []
        for value in probabilities.tolist():
            score = float(value)
            if not math.isfinite(score):
                # A NaN would compare false against any threshold and turn into a
                # silent cache miss. Fail loudly so the caller records the reason.
                raise ModelContractError("reranker returned a non-finite score")
            scores.append(score)

        if len(scores) != len(candidates):
            raise ModelContractError(
                f"reranker returned {len(scores)} scores for {len(candidates)} candidates"
            )
        return RerankResult(scores=scores, model_version=self.model_version)
