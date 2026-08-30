from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, ClassVar, Protocol

from llm_router_serving.errors import ModelContractError
from llm_router_serving.schemas import RoutedModel, Tier

# The embedder and the classifier are loaded concurrently on startup, each in its
# own worker thread. `transformers` resolves its public names lazily, so two
# threads pulling different names at once can observe a submodule that the other
# thread is still importing and fail with a spurious ImportError. Import the
# symbols under one lock so no thread ever sees a half-initialized module.
_TRANSFORMERS_IMPORT_LOCK = Lock()


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
        model_path: Path,
        model_version: str,
        max_length: int,
        device_name: str,
    ) -> TransformersModernBertClassifier:
        with _TRANSFORMERS_IMPORT_LOCK:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        cls._validate_label_contract(model.config)
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
        )

    @classmethod
    def _validate_label_contract(cls, config: Any) -> None:
        if int(getattr(config, "num_labels", 0)) != 3:
            raise ModelContractError("ModernBERT checkpoint must have exactly three labels")
        raw_id2label = getattr(config, "id2label", {}) or {}
        actual = tuple(str(raw_id2label.get(index, "")).lower() for index in range(3))
        expected = tuple(label.value for label in cls._EXPECTED_LABELS)
        if actual != expected:
            raise ModelContractError(
                f"ModernBERT label order must be {expected}; checkpoint contains {actual}"
            )

    def classify(self, message: str) -> ClassificationResult:
        inputs = self._tokenizer(
            message,
            return_tensors="pt",
            truncation=True,
            max_length=self._max_length,
        )
        inputs = {name: value.to(self._device) for name, value in inputs.items()}
        with self._torch.inference_mode():
            logits = self._model(**inputs).logits
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


class SentenceTransformersCrossEncoderReranker:
    """Cross-encoder that rescores pgvector candidates against the query.

    The bi-encoder embeds each text on its own, so swapping one entity
    (Python to Java, insertion to deletion) barely moves the vector. Measured on
    92 questions, paraphrases bottomed out at 0.8621 while hard negatives topped
    out at 0.9371: the two classes overlap and no cosine threshold separates them.
    A cross-encoder reads both texts together, which is exactly the case it fixes.

    **This deliberately uses ``sentence_transformers.CrossEncoder`` rather than
    driving ``transformers`` directly.** The offline screening that produced the
    candidate threshold called ``CrossEncoder.predict``, and that call decides the
    activation applied to the logits. Reimplementing the head here would shift the
    scores and silently invalidate the measured threshold, which is the one number
    the whole evaluation exists to produce.
    """

    def __init__(
        self,
        *,
        model: Any,
        model_version: str,
        max_length: int,
    ) -> None:
        self._model = model
        self.model_version = model_version
        self._max_length = max_length

    @classmethod
    def load(
        cls,
        *,
        model_id: str,
        model_version: str,
        max_length: int,
        device_name: str,
        cache_dir: Path | None,
    ) -> SentenceTransformersCrossEncoderReranker:
        with _TRANSFORMERS_IMPORT_LOCK:
            import torch
            from sentence_transformers import CrossEncoder

        device = resolve_torch_device(torch, device_name)
        model = CrossEncoder(
            model_id,
            device=str(device),
            max_length=max_length,
            cache_folder=str(cache_dir) if cache_dir is not None else None,
        )
        return cls(
            model=model,
            model_version=cls._resolve_version(model, model_id, model_version),
            max_length=max_length,
        )

    @staticmethod
    def _resolve_version(model: Any, model_id: str, configured: str | None) -> str:
        """Prefer the checkpoint commit hash over a human-typed version string.

        A threshold is only valid for the exact weights it was measured on, so the
        value the gateway logs has to change whenever the weights do.
        """
        if configured:
            return configured
        config = getattr(getattr(model, "model", None), "config", None)
        revision = getattr(config, "_commit_hash", None)
        return f"{model_id}@{revision}" if revision else model_id

    def rerank(self, query: str, candidates: list[str]) -> RerankResult:
        pairs = [(query, candidate) for candidate in candidates]
        output = self._model.predict(pairs, batch_size=len(pairs), show_progress_bar=False)

        scores: list[float] = []
        for value in output:
            if hasattr(value, "item"):
                value = value.item()
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
