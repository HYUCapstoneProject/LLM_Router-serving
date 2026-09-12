from __future__ import annotations

import json
from dataclasses import dataclass
from typing import ClassVar, Protocol

from llm_router_serving.retraining.gcs import ObjectAccessError


class ObservabilityObjectStore(Protocol):
    def read_bytes(self, uri: str) -> bytes: ...


@dataclass(frozen=True)
class ObservabilitySnapshot:
    feedback_prepare: dict | None
    feedback_merge: dict | None
    training_progress: dict | None
    training_final: dict | None
    evaluation: dict | None
    packaging: dict | None
    gate: dict | None
    publishing: dict | None
    warnings: tuple[str, ...]

    def values(self) -> tuple[dict | None, ...]:
        return (
            self.feedback_prepare,
            self.feedback_merge,
            self.training_progress,
            self.training_final,
            self.evaluation,
            self.packaging,
            self.gate,
            self.publishing,
        )


class RetrainingObservabilityReader:
    _OBJECTS: ClassVar[dict[str, str]] = {
        "feedback_prepare": "metrics/feedback-prepare.json",
        "feedback_merge": "metrics/feedback-merge.json",
        "training_progress": "progress/training.json",
        "training_final": "metrics/training-final.json",
        "evaluation": "metrics/evaluation.json",
        "packaging": "metrics/packaging.json",
        "gate": "gate/gate-result.json",
        "publishing": "result/publish-result.json",
    }

    def __init__(self, *, object_store: ObservabilityObjectStore, root_uri: str) -> None:
        self._object_store = object_store
        self._root_uri = root_uri.rstrip("/")

    def run_root(self, batch_id: str) -> str:
        return f"{self._root_uri}/{batch_id}/observability-v1"

    def read(self, *, batch_id: str, candidate_model_version: str) -> ObservabilitySnapshot:
        values: dict[str, dict | None] = {}
        warnings: list[str] = []
        root = self.run_root(batch_id)
        for name, relative in self._OBJECTS.items():
            uri = f"{root}/{relative}"
            try:
                raw = self._object_store.read_bytes(uri)
            except ObjectAccessError as exc:
                if exc.code == "GCS_OBJECT_NOT_FOUND":
                    values[name] = None
                    continue
                warnings.append(f"{name} unavailable: {exc.code}")
                values[name] = None
                continue
            except KeyError:
                # In-memory stores used by contract tests signal missing keys this way.
                values[name] = None
                continue
            try:
                payload = json.loads(raw)
            except (json.JSONDecodeError, UnicodeDecodeError, TypeError):
                warnings.append(f"{name} ignored: invalid JSON")
                values[name] = None
                continue
            if not isinstance(payload, dict):
                warnings.append(f"{name} ignored: expected object")
                values[name] = None
                continue
            if payload.get("batch_id") != batch_id:
                warnings.append(f"{name} ignored: batch identity mismatch")
                values[name] = None
                continue
            observed_candidate = payload.get("candidate_model_version")
            if observed_candidate not in {None, candidate_model_version}:
                warnings.append(f"{name} ignored: candidate identity mismatch")
                values[name] = None
                continue
            values[name] = payload
        return ObservabilitySnapshot(warnings=tuple(warnings), **values)
