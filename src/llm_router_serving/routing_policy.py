"""Explicit service routing policies, independent of model/backend libraries.

No benchmark policy is converted into a service policy here. A package must
provide the complete service pool, costs, lambda, versions, and policy digest.
The internal policy digest covers canonical JSON without ``policy_sha256``;
a package's ``routing_policy_sha256`` separately covers the actual file bytes.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Any

POLICY_SCHEMA_VERSION = "routing-policy-v2"
LENGTH_CONTRACT_VERSION = "train-inference-lengths-v1"
ACTION_ORDER = ("low", "medium", "high")
TIER_ORDER = ("easy", "medium", "hard")
MODEL_HINT_ORDER = ("haiku", "sonnet", "opus")
OBJECTIVE_SCORE_SEMANTICS = {
    "ordinal": "difficulty_class_probabilities",
    "pointwise": "independent_success_probabilities",
    "pairwise": "relative_quality_scores",
}
OUTPUT_LABEL_ORDERS = {
    "ordinal": TIER_ORDER,
    "pointwise": ACTION_ORDER,
    "pairwise": ACTION_ORDER,
}


class RoutingPolicyError(ValueError):
    """The package policy or score vector does not satisfy the service contract."""


def _keys(value: Any, expected: set[str], name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RoutingPolicyError(f"{name} must be an object")
    missing = expected - set(value)
    extra = set(value) - expected
    if missing or extra:
        raise RoutingPolicyError(
            f"{name} fields differ: missing={sorted(missing)}, unexpected={sorted(extra)}"
        )
    return value


def _identifier(value: Any, name: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._:/@+\-]{0,255}", value
    ):
        raise RoutingPolicyError(f"{name} must be an explicit non-empty identifier")
    return value


def _finite_number(value: Any, name: str, *, non_negative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise RoutingPolicyError(f"{name} must be a finite number")
    try:
        number = float(value)
    except OverflowError as error:
        raise RoutingPolicyError(f"{name} exceeds the finite numeric range") from error
    if not math.isfinite(number):
        raise RoutingPolicyError(f"{name} must be finite")
    if non_negative and number < 0:
        raise RoutingPolicyError(f"{name} must be non-negative")
    # Canonicalize -0.0 and JSON integers/floats identically for policy hashes.
    return 0.0 if number == 0 else number


def _digest_value(value: Any, name: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise RoutingPolicyError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _validated_payload(values: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(values, Mapping):
        raise RoutingPolicyError("Policy must be an object")
    payload = {key: value for key, value in values.items() if key != "policy_sha256"}
    _keys(
        payload,
        {
            "schema_version",
            "scope",
            "policy_version",
            "objective",
            "input_contract",
            "max_length",
            "action_order",
            "score_semantics",
            "lambda_cost",
            "candidate_pool",
        } | ({"length_contract_version"} if "length_contract_version" in payload else set()),
        "policy",
    )
    if payload["schema_version"] != POLICY_SCHEMA_VERSION:
        raise RoutingPolicyError("Unsupported routing policy schema")
    if payload["scope"] != "service-explicit":
        raise RoutingPolicyError(
            "Only service-explicit policies may route service traffic"
        )
    objective = payload["objective"]
    if not isinstance(objective, str) or objective not in OBJECTIVE_SCORE_SEMANTICS:
        raise RoutingPolicyError(
            "Policy objective must be ordinal, pointwise or pairwise"
        )
    if payload["score_semantics"] != OBJECTIVE_SCORE_SEMANTICS[objective]:
        raise RoutingPolicyError("Score semantics do not match the objective")
    if payload["input_contract"] != "structure-preserving-v2":
        raise RoutingPolicyError(
            "Policy requires the structure-preserving-v2 input contract"
        )
    has_length_contract = "length_contract_version" in payload
    if has_length_contract and payload["length_contract_version"] != LENGTH_CONTRACT_VERSION:
        raise RoutingPolicyError("Unsupported policy length_contract_version")
    allowed_lengths = {512, 8192} if has_length_contract else {512}
    if type(payload["max_length"]) is not int or payload["max_length"] not in allowed_lengths:
        raise RoutingPolicyError(f"Service policy max_length must be in {sorted(allowed_lengths)}")
    if payload["action_order"] != list(ACTION_ORDER):
        raise RoutingPolicyError("Action order must be low, medium, high")
    pool = _keys(
        payload["candidate_pool"],
        {
            "version",
            "price_version",
            "decision_cost_version",
            "decision_cost_basis",
            "cost_unit",
            "candidates",
        },
        "candidate_pool",
    )
    if pool["decision_cost_basis"] != "pre-request-fixed":
        raise RoutingPolicyError("Decision costs must be fixed before the request")
    if not isinstance(pool["cost_unit"], str) or pool["cost_unit"] not in {
        "normalized-0-1",
        "usd-per-request",
    }:
        raise RoutingPolicyError("Unknown decision cost unit")
    candidates = pool["candidates"]
    if not isinstance(candidates, list) or len(candidates) != 3:
        raise RoutingPolicyError(
            "Exactly three explicit service candidates are required"
        )
    canonical_candidates = []
    for index, values_for_candidate in enumerate(candidates):
        candidate = _keys(
            values_for_candidate,
            {"action", "tier_projection", "model", "model_id", "decision_cost"},
            f"candidate[{index}]",
        )
        if (
            candidate["action"] != ACTION_ORDER[index]
            or candidate["tier_projection"] != TIER_ORDER[index]
            or candidate["model"] != MODEL_HINT_ORDER[index]
        ):
            raise RoutingPolicyError(
                "Candidate order must preserve low/easy/haiku, medium/medium/sonnet, high/hard/opus"
            )
        model_id = _identifier(candidate["model_id"], f"candidate[{index}].model_id")
        if model_id.casefold() in {*ACTION_ORDER, *TIER_ORDER, *MODEL_HINT_ORDER}:
            raise RoutingPolicyError(
                "model_id must identify an actual provider model, not a tier hint"
            )
        canonical_candidates.append(
            {
                "action": candidate["action"],
                "tier_projection": candidate["tier_projection"],
                "model": candidate["model"],
                "model_id": model_id,
                "decision_cost": _finite_number(
                    candidate["decision_cost"], "decision_cost", non_negative=True
                ),
            }
        )
    if len({candidate["model_id"] for candidate in canonical_candidates}) != 3:
        raise RoutingPolicyError("Three distinct provider model IDs are required")
    costs = [candidate["decision_cost"] for candidate in canonical_candidates]
    if costs != sorted(costs):
        raise RoutingPolicyError(
            "Service decision costs must follow low/medium/high order"
        )
    if pool["cost_unit"] == "normalized-0-1" and (costs[0] != 0.0 or costs[-1] != 1.0):
        raise RoutingPolicyError("Normalized costs require low=0 and high=1")
    return {
        "schema_version": POLICY_SCHEMA_VERSION,
        "scope": "service-explicit",
        "policy_version": _identifier(payload["policy_version"], "policy_version"),
        "objective": objective,
        "input_contract": "structure-preserving-v2",
        "max_length": payload["max_length"],
        **({"length_contract_version": LENGTH_CONTRACT_VERSION} if has_length_contract else {}),
        "action_order": list(ACTION_ORDER),
        "score_semantics": OBJECTIVE_SCORE_SEMANTICS[objective],
        "lambda_cost": _finite_number(
            payload["lambda_cost"], "lambda_cost", non_negative=True
        ),
        "candidate_pool": {
            "version": _identifier(pool["version"], "candidate_pool.version"),
            "price_version": _identifier(
                pool["price_version"], "candidate_pool.price_version"
            ),
            "decision_cost_version": _identifier(
                pool["decision_cost_version"], "candidate_pool.decision_cost_version"
            ),
            "decision_cost_basis": "pre-request-fixed",
            "cost_unit": pool["cost_unit"],
            "candidates": canonical_candidates,
        },
    }


def compute_policy_sha256(values: Mapping[str, Any]) -> str:
    """Hash validated canonical policy JSON, excluding its internal digest field.

    The producer supplies all values; this function does not choose costs,
    lambda or model IDs. Package file checksums cover the actual serialized file.
    """
    encoded = json.dumps(
        _validated_payload(values),
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class ServiceCandidate:
    action: str
    tier_projection: str
    model: str
    model_id: str
    decision_cost: float


@dataclass(frozen=True)
class RoutingPolicy:
    policy_version: str
    objective: str
    score_semantics: str
    lambda_cost: float
    candidate_pool_version: str
    price_version: str
    decision_cost_version: str
    cost_unit: str
    candidates: tuple[ServiceCandidate, ...]
    policy_sha256: str
    input_contract: str = "structure-preserving-v2"
    max_length: int = 512
    length_contract_version: str | None = None

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, Any],
        *,
        expected_policy_sha256: str | None = None,
    ) -> RoutingPolicy:
        """Validate internal policy hash; expected_policy_sha256 is NOT file SHA."""
        payload = _validated_payload(values)
        supplied = _digest_value(values.get("policy_sha256"), "policy_sha256")
        computed = compute_policy_sha256(payload)
        if supplied != computed:
            raise RoutingPolicyError("Policy content does not match policy_sha256")
        if expected_policy_sha256 is not None and supplied != _digest_value(
            expected_policy_sha256, "expected_policy_sha256"
        ):
            raise RoutingPolicyError(
                "Policy differs from the expected canonical policy digest"
            )
        pool = payload["candidate_pool"]
        return cls(
            policy_version=payload["policy_version"],
            objective=payload["objective"],
            score_semantics=payload["score_semantics"],
            lambda_cost=payload["lambda_cost"],
            candidate_pool_version=pool["version"],
            price_version=pool["price_version"],
            decision_cost_version=pool["decision_cost_version"],
            cost_unit=pool["cost_unit"],
            candidates=tuple(
                ServiceCandidate(**candidate) for candidate in pool["candidates"]
            ),
            policy_sha256=supplied,
            max_length=payload["max_length"],
            length_contract_version=payload.get("length_contract_version"),
        )

    @property
    def output_label_order(self) -> tuple[str, ...]:
        return OUTPUT_LABEL_ORDERS[self.objective]


def load_routing_policy(
    path: Path,
    *,
    expected_file_sha256: str | None = None,
) -> RoutingPolicy:
    """Check package file bytes separately, then validate the internal policy."""
    raw = path.read_bytes()
    if expected_file_sha256 is not None:
        expected = _digest_value(expected_file_sha256, "expected_file_sha256")
        if hashlib.sha256(raw).hexdigest() != expected:
            raise RoutingPolicyError("Routing policy file SHA-256 mismatch")
    try:
        values = json.loads(raw)
    except (ValueError, UnicodeError) as error:
        raise RoutingPolicyError(
            "Routing policy file must contain valid JSON"
        ) from error
    return RoutingPolicy.from_mapping(values)


@dataclass(frozen=True)
class RoutingDecision:
    action: str
    tier_projection: str
    model: str
    model_id: str
    action_scores: dict[str, float]
    utilities: dict[str, float]
    confidence: float | None
    score_semantics: str
    objective: str
    policy_version: str
    policy_sha256: str
    candidate_pool_version: str


def route_scores(policy: RoutingPolicy, logits: Sequence[float]) -> RoutingDecision:
    """Apply a validated policy to one three-logit query without source metadata.

    Confidence is the selected action's pre-cost softmax/sigmoid score. In
    particular, it is not the maximum probability if costs change the action.
    Pairwise scores are relative utilities, so their confidence is always None.
    """
    try:
        valid_length = not isinstance(logits, (str, bytes)) and len(logits) == 3
    except TypeError:
        valid_length = False
    if not valid_length:
        raise RoutingPolicyError("Exactly three ordered logits are required")
    values = tuple(_finite_number(value, "logit") for value in logits)
    if policy.objective == "ordinal":
        largest = max(values)
        exponential = tuple(math.exp(value - largest) for value in values)
        denominator = sum(exponential)
        scores = tuple(value / denominator for value in exponential)
    elif policy.objective == "pointwise":
        # Match the frozen offline evaluator, including its numerical clipping.
        scores = tuple(
            1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, value)))) for value in values
        )
    elif policy.objective == "pairwise":
        scores = values
    else:
        raise RoutingPolicyError("Unsupported policy objective")
    utilities = tuple(
        score - policy.lambda_cost * candidate.decision_cost
        for score, candidate in zip(scores, policy.candidates, strict=True)
    )
    if not all(math.isfinite(value) for value in utilities):
        raise RoutingPolicyError("Decision utilities overflowed their finite range")
    # Python max keeps the first maximum: exact ties use low, then medium, high.
    selected_index = max(range(3), key=utilities.__getitem__)
    selected = policy.candidates[selected_index]
    return RoutingDecision(
        action=selected.action,
        tier_projection=selected.tier_projection,
        model=selected.model,
        model_id=selected.model_id,
        action_scores=dict(zip(ACTION_ORDER, scores, strict=True)),
        utilities=dict(zip(ACTION_ORDER, utilities, strict=True)),
        confidence=None if policy.objective == "pairwise" else scores[selected_index],
        score_semantics=policy.score_semantics,
        objective=policy.objective,
        policy_version=policy.policy_version,
        policy_sha256=policy.policy_sha256,
        candidate_pool_version=policy.candidate_pool_version,
    )
