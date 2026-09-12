from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol


class TrainingMode(StrEnum):
    MINOR = "minor_fine_tune"
    MAJOR = "major_consolidation"
    REBASE = "pristine_rebase"


class InitializationMode(StrEnum):
    APPROVED_ANCHOR = "approved_anchor"
    PRISTINE = "pristine_modernbert"


class SuccessfulRun(Protocol):
    logical_model_version: str | None
    candidate_model_version: str | None
    base_model_uri: str | None
    feedback_manifest_uri: str | None
    feedback_events_uri: str | None
    replay_manifest_uri: str | None
    replay_events_uri: str | None


@dataclass(frozen=True)
class ContinualTrainingPlan:
    logical_model_version: str
    training_mode: TrainingMode
    initialization_mode: InitializationMode
    anchor_model_version: str
    anchor_model_uri: str
    baseline_model_version: str
    baseline_model_uri: str
    previous_cycle_manifest_uri: str | None
    previous_cycle_events_uri: str | None
    previous_replay_manifest_uri: str | None
    previous_replay_events_uri: str | None


def build_continual_training_plan(
    successful_runs: Sequence[SuccessfulRun],
    *,
    initial_logical_major: int,
    calls_per_major: int,
    rebase_every_major_cycles: int,
    initial_model_version: str,
    initial_model_uri: str,
) -> ContinualTrainingPlan:
    """Plan the next accepted HTTP call from durable successful-call history.

    The trigger is deliberately the count of successful ``POST /escalate`` jobs.
    Event-level metadata is not used as a trigger yet. With the current fixed
    three-call policy, versions progress ``1.1 -> 1.2 -> 2.0``.
    """
    if initial_logical_major < 0:
        raise ValueError("initial_logical_major must be non-negative")
    if calls_per_major < 1:
        raise ValueError("calls_per_major must be positive")
    if rebase_every_major_cycles < 1:
        raise ValueError("rebase_every_major_cycles must be positive")

    completed_calls = len(successful_runs)
    next_call = completed_calls + 1
    completed_major_cycles, position = divmod(next_call, calls_per_major)
    if position == 0:
        target_major = initial_logical_major + completed_major_cycles
        target_minor = 0
    else:
        target_major = initial_logical_major + completed_major_cycles
        target_minor = position
    logical_version = f"{target_major}.{target_minor}"

    major_cycle = target_minor == 0
    rebase = major_cycle and completed_major_cycles % rebase_every_major_cycles == 0
    if rebase:
        training_mode = TrainingMode.REBASE
        initialization_mode = InitializationMode.PRISTINE
    elif major_cycle:
        training_mode = TrainingMode.MAJOR
        initialization_mode = InitializationMode.APPROVED_ANCHOR
    else:
        training_mode = TrainingMode.MINOR
        initialization_mode = InitializationMode.APPROVED_ANCHOR

    baseline_version = initial_model_version
    baseline_uri = initial_model_uri
    if successful_runs:
        latest = successful_runs[-1]
        baseline_version = latest.candidate_model_version or initial_model_version
        baseline_uri = _approved_uri(latest.candidate_model_version, initial_model_uri)

    anchor_version = initial_model_version
    anchor_uri = initial_model_uri
    for run in reversed(successful_runs):
        if _minor_part(run.logical_model_version) == 0 and run.candidate_model_version:
            anchor_version = run.candidate_model_version
            anchor_uri = _approved_uri(run.candidate_model_version, initial_model_uri)
            break

    previous_cycle_manifest_uri: str | None = None
    previous_cycle_events_uri: str | None = None
    if successful_runs and _minor_part(successful_runs[-1].logical_model_version) != 0:
        previous_cycle_manifest_uri = successful_runs[-1].feedback_manifest_uri
        previous_cycle_events_uri = successful_runs[-1].feedback_events_uri

    previous_replay_manifest_uri = (
        successful_runs[-1].replay_manifest_uri if successful_runs else None
    )
    previous_replay_events_uri = successful_runs[-1].replay_events_uri if successful_runs else None
    return ContinualTrainingPlan(
        logical_model_version=logical_version,
        training_mode=training_mode,
        initialization_mode=initialization_mode,
        anchor_model_version=anchor_version,
        anchor_model_uri=anchor_uri,
        baseline_model_version=baseline_version,
        baseline_model_uri=baseline_uri,
        previous_cycle_manifest_uri=previous_cycle_manifest_uri,
        previous_cycle_events_uri=previous_cycle_events_uri,
        previous_replay_manifest_uri=previous_replay_manifest_uri,
        previous_replay_events_uri=previous_replay_events_uri,
    )


def _minor_part(logical_version: str | None) -> int | None:
    if not logical_version or "." not in logical_version:
        return None
    try:
        return int(logical_version.rsplit(".", 1)[1])
    except ValueError:
        return None


def _approved_uri(model_version: str | None, initial_model_uri: str) -> str:
    if not model_version:
        return initial_model_uri
    root = initial_model_uri.rstrip("/")
    if root.endswith("/approved"):
        root = root.rsplit("/", 2)[0]
    return f"{root}/{model_version}/approved"
