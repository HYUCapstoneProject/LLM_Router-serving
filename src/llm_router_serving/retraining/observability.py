from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import ClassVar, Protocol

from llm_router_serving.retraining.gcs import ObjectAccessError

# Must match the ML producer. A v2 head needs an explicit objective-aware gate;
# accepting the shared envelope alone never authorizes a pointwise promotion.
SUPPORTED_PROMOTION_CONTRACTS = frozenset(
    {
        ("ordinal", "model-package-v1", "deploy-gate-result-v1"),
    }
)


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


class PromotionEvidenceUnavailable(RuntimeError):
    """Promotion cannot yet be verified; retain feedback and retry observation."""


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

    def verify_promotion(
        self,
        *,
        batch_id: str,
        candidate_model_version: str,
        pipeline_run_id: str,
        publish_root_uri: str,
        promotion_mode: str = "quality_gate",
    ) -> None:
        """Verify an explicitly supported promotion independently of Vertex state.

        Read only fixed paths under configured roots; never follow evidence URIs.
        CURRENT may race with another writer, so check its bytes again last.
        Missing, malformed or mismatched evidence remains retryable and fail-closed.
        """

        def read_object(uri: str, stage: str) -> tuple[bytes, dict]:
            try:
                raw = self._object_store.read_bytes(uri)
                value = json.loads(raw)
            except Exception as exc:
                raise PromotionEvidenceUnavailable(f"{stage} unavailable") from exc
            if not isinstance(value, dict):
                raise PromotionEvidenceUnavailable(f"{stage} is not an object")
            return raw, value

        def require(condition: bool, reason: str) -> None:
            if not condition:
                raise PromotionEvidenceUnavailable(reason)

        require(
            bool(publish_root_uri and pipeline_run_id and candidate_model_version),
            "promotion identity is incomplete",
        )
        require(
            re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", candidate_model_version) is not None,
            "candidate version is not a safe path segment",
        )
        require(promotion_mode in {"quality_gate", "direct"}, "unsupported promotion mode")
        direct = promotion_mode == "direct"
        run_root = self.run_root(batch_id)
        evidence_path = "promotion/direct-result.json" if direct else "gate/gate-result.json"
        gate_raw, gate = read_object(f"{run_root}/{evidence_path}", "promotion evidence")
        _, publish = read_object(f"{run_root}/result/publish-result.json", "publish evidence")
        for name, value in (("gate", gate), ("publish", publish)):
            require(value.get("batch_id") == batch_id, f"{name} batch mismatch")
            require(
                value.get("candidate_model_version") == candidate_model_version,
                f"{name} candidate mismatch",
            )
            require(
                value.get("pipeline_job_resource_name") == pipeline_run_id,
                f"{name} pipeline mismatch",
            )
        generic = publish.get("source_schema_version") == "router-promotion-evidence-v1"
        contract = (
            (
                publish.get("objective"),
                publish.get("package_schema_version"),
                publish.get("gate_schema_version"),
            )
            if generic
            else ("ordinal", "model-package-v1", "deploy-gate-result-v1")
        )
        require(
            contract
            in (
                {("ordinal", "model-package-v1", "direct-promotion-result-v1")}
                if direct
                else SUPPORTED_PROMOTION_CONTRACTS
            ),
            "objective/package/gate contract is unsupported",
        )
        require(
            gate.get("schema_version") == "retraining-observability-v1"
            and gate.get("source_schema_version") == contract[2]
            and (
                (
                    "passed" not in gate
                    and gate.get("promotion_mode") == "direct"
                    and gate.get("quality_gate_skipped") is True
                )
                if direct
                else gate.get("passed") is True
            ),
            "gate approval is unverified",
        )
        require(
            publish.get("schema_version") == "retraining-observability-v1"
            and publish.get("source_schema_version")
            in {"router-promotion-evidence-v1", "r0-promotion-evidence-v1"}
            and publish.get("outcome") == "promoted",
            "publish approval is unverified",
        )
        require(
            hashlib.sha256(gate_raw).hexdigest() == publish.get("gate_result_sha256"),
            "gate evidence checksum mismatch",
        )
        root = publish_root_uri.rstrip("/")
        pointer_uri = f"{root}/CURRENT.json"
        package_uri = f"{root}/{candidate_model_version}/approved"
        require(publish.get("current_pointer_uri") == pointer_uri, "pointer URI mismatch")
        require(publish.get("package_uri") == package_uri, "package URI mismatch")
        pointer_raw, pointer = read_object(pointer_uri, "CURRENT")
        require(
            hashlib.sha256(pointer_raw).hexdigest() == publish.get("current_pointer_sha256"),
            "CURRENT checksum mismatch",
        )
        require(
            pointer.get("schema_version") == "router-current-model-v1", "CURRENT schema mismatch"
        )
        for name, value in (("CURRENT", pointer), ("publish", publish)):
            require(value.get("model_version") == candidate_model_version, f"{name} model mismatch")
            require(value.get("package_uri") == package_uri, f"{name} package mismatch")
        require(pointer.get("pipeline_run_id") == pipeline_run_id, "CURRENT pipeline mismatch")
        candidate_digest = gate.get("candidate_manifest_sha256")
        require(
            isinstance(candidate_digest, str)
            and re.fullmatch(r"[0-9a-f]{64}", candidate_digest) is not None
            and pointer.get("candidate_manifest_sha256") == candidate_digest,
            "gated candidate manifest mismatch",
        )
        digest = publish.get("manifest_sha256")
        require(
            isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest) is not None,
            "manifest checksum missing",
        )
        require(pointer.get("manifest_sha256") == digest, "manifest reference mismatch")
        raw, manifest = read_object(f"{package_uri}/model_manifest.json", "approved manifest")
        require(hashlib.sha256(raw).hexdigest() == digest, "approved manifest checksum mismatch")
        require(
            manifest.get("schema_version") == contract[1]
            and manifest.get("objective", "ordinal" if contract[1] == "model-package-v1" else None)
            == contract[0],
            "approved package differs from its objective contract",
        )
        if generic:
            require(
                gate.get("objective") == contract[0]
                and gate.get("package_schema_version") == contract[1]
                and gate.get("gate_schema_version") == contract[2],
                "gate does not declare the promoted objective/package contract",
            )
            require(
                publish.get("candidate_manifest_sha256") == candidate_digest,
                "published candidate manifest mismatch",
            )
            require(
                publish.get("approved_manifest_sha256") == digest,
                "approved manifest reference mismatch",
            )
            if manifest.get("length_contract_version") is not None:
                require(
                    all(
                        value.get("length_contract_version") == manifest["length_contract_version"]
                        and type(value.get("inference_max_length")) is int
                        and value["inference_max_length"] == manifest.get("inference_max_length")
                        for value in (publish, gate)
                    ),
                    "promotion inference length contract differs from package",
                )
            if direct:
                require(
                    all(
                        value.get("promotion_mode") == "direct"
                        and value.get("quality_gate_skipped") is True
                        for value in (gate, publish, manifest)
                    )
                    and manifest.get("length_contract_version") == "train-inference-lengths-v1"
                    and type(manifest.get("inference_max_length")) is int
                    and manifest["inference_max_length"] == 8192
                    and "passed" not in publish
                    and gate.get("candidate_metrics_sha256") is None
                    and publish.get("candidate_metrics_sha256") is None
                    and manifest.get("evaluation_role") == "not_evaluated"
                    and manifest.get("evaluation_version") == "not-evaluated-direct-promotion-v1"
                    and isinstance(manifest.get("files"), dict)
                    and not {
                        "gate_metrics.json",
                        "routerarena_metrics.json",
                        "development_metrics.json",
                    }.intersection(manifest["files"]),
                    "direct promotion must not claim a quality evaluation",
                )
                baseline_sha = gate.get("baseline_manifest_sha256")
                require(
                    isinstance(baseline_sha, str)
                    and re.fullmatch(r"[0-9a-f]{64}", baseline_sha) is not None
                    and publish.get("baseline_manifest_sha256") == baseline_sha
                    and publish.get("baseline_model_version") == gate.get("baseline_model_version")
                    and publish.get("expected_current_generation")
                    == gate.get("expected_current_generation")
                    and gate.get("model_publish_root_uri") == root
                    and isinstance(gate.get("baseline_model_version"), str)
                    and bool(gate["baseline_model_version"])
                    and pointer.get("previous_model_version") == gate["baseline_model_version"]
                    and type(gate.get("expected_current_generation")) is int
                    and gate["expected_current_generation"] > 0
                    and pointer.get("previous_pointer_generation")
                    == gate["expected_current_generation"],
                    "direct promotion baseline binding mismatch",
                )
            else:
                metrics_sha = publish.get("candidate_metrics_sha256")
                metric_name = {
                    "continual_gate": "gate_metrics.json",
                    "routerarena": "routerarena_metrics.json",
                    "development_calibration": "development_metrics.json",
                }.get(manifest.get("evaluation_role", "routerarena"))
                require(
                    isinstance(metrics_sha, str)
                    and re.fullmatch(r"[0-9a-f]{64}", metrics_sha) is not None
                    and gate.get("candidate_metrics_sha256") == metrics_sha
                    and isinstance(manifest.get("files"), dict)
                    and manifest["files"].get(metric_name) == metrics_sha,
                    "approved package does not bind the gated evaluation",
                )
            generation = publish.get("current_pointer_generation")
            require(
                type(generation) is int and generation > 0, "CURRENT generation evidence missing"
            )
            require(
                "previous_pointer_generation" in pointer
                and pointer["previous_pointer_generation"]
                == publish.get("previous_pointer_generation")
                and (
                    pointer["previous_pointer_generation"] is None
                    or (
                        type(pointer["previous_pointer_generation"]) is int
                        and pointer["previous_pointer_generation"] > 0
                    )
                ),
                "CURRENT CAS baseline evidence mismatch",
            )
        require(
            manifest.get("model_version") == candidate_model_version
            and manifest.get("gate_status") == "approved"
            and manifest.get("pipeline_run_id") == pipeline_run_id,
            "approved manifest identity mismatch",
        )
        approved_at = publish.get("approved_at")
        require(
            isinstance(approved_at, str)
            and bool(approved_at)
            and pointer.get("approved_at") == approved_at
            and manifest.get("approved_at") == approved_at,
            "approval timestamp mismatch",
        )
        reread, _ = read_object(pointer_uri, "CURRENT recheck")
        require(reread == pointer_raw, "CURRENT changed during verification")
