"""Inventory legacy v1 payloads without inventing missing v2 semantics."""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from typing import Any

from pydantic import Field

from cellstate.domain.common import SchemaModel, canonical_fingerprint


class LegacyMigrationError(ValueError):
    """Raised when a payload cannot enter v2 without an explicit scientific decision."""


class LegacyArtifactKind(StrEnum):
    STATE_QUERY = "state_query"
    CELL_HISTORY = "cell_history"
    ESTIMATE_REQUEST = "estimate_cell_state_request"
    CELL_STATE_BELIEF = "cell_state_belief"
    STATE_FORECAST = "state_forecast"
    INTERVENTION_PLAN = "intervention_plan"


class MigrationDisposition(StrEnum):
    EXPLICIT_ANNOTATIONS_REQUIRED = "explicit_annotations_required"
    REESTIMATION_REQUIRED = "reestimation_required"


class MigrationRequirement(SchemaModel):
    """Auditable result of inspecting one legacy artifact."""

    artifact_kind: LegacyArtifactKind
    legacy_schema_version: str = Field(pattern=r"^1\.0$")
    legacy_fingerprint: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    disposition: MigrationDisposition
    required_decisions: tuple[str, ...] = Field(min_length=1)
    explanation: str = Field(min_length=1)


_INPUT_DECISIONS: dict[LegacyArtifactKind, tuple[str, ...]] = {
    LegacyArtifactKind.STATE_QUERY: (
        "typed belief-subject and target aggregation",
        "bounded intervention and environment domains",
        "assay collection effects, cost, and delay",
        "query-scoped scientific-readiness thresholds",
    ),
    LegacyArtifactKind.CELL_HISTORY: (
        "typed target subject",
        "source-to-target evidence links",
        "observation collection effects and durations",
        "interval and completeness semantics",
    ),
    LegacyArtifactKind.ESTIMATE_REQUEST: (
        "a migrated v2 query",
        "a migrated v2 history",
        "typed subject/context compatibility",
    ),
}

_OUTPUT_DECISIONS: dict[LegacyArtifactKind, tuple[str, ...]] = {
    LegacyArtifactKind.CELL_STATE_BELIEF: (
        "re-estimate from migrated v2 inputs",
        "compile active factors and realization/nuisance blocks",
        "evaluate query-scoped support and readiness",
    ),
    LegacyArtifactKind.STATE_FORECAST: (
        "regenerate from a v2 belief and bounded scenario",
        "evaluate causal and transport status",
        "evaluate forecast support and abstention",
    ),
    LegacyArtifactKind.INTERVENTION_PLAN: (
        "regenerate under v2 control-readiness gates",
        "re-evaluate every candidate inside bounded support",
    ),
}


def inspect_v1_payload(
    payload: Mapping[str, Any],
    *,
    artifact_kind: LegacyArtifactKind,
) -> MigrationRequirement:
    """Return the required v2 disposition; never mutate or upgrade the payload."""

    if payload.get("schema_version") != "1.0":
        raise LegacyMigrationError("legacy migration inspection requires schema_version '1.0'")
    if artifact_kind in _INPUT_DECISIONS:
        return MigrationRequirement(
            artifact_kind=artifact_kind,
            legacy_schema_version="1.0",
            legacy_fingerprint=canonical_fingerprint(dict(payload)),
            disposition=MigrationDisposition.EXPLICIT_ANNOTATIONS_REQUIRED,
            required_decisions=_INPUT_DECISIONS[artifact_kind],
            explanation=(
                "v1 did not encode the scientific semantics required by v2; construct a new v2 "
                "input using explicit reviewed annotations"
            ),
        )
    if artifact_kind in _OUTPUT_DECISIONS:
        return MigrationRequirement(
            artifact_kind=artifact_kind,
            legacy_schema_version="1.0",
            legacy_fingerprint=canonical_fingerprint(dict(payload)),
            disposition=MigrationDisposition.REESTIMATION_REQUIRED,
            required_decisions=_OUTPUT_DECISIONS[artifact_kind],
            explanation=(
                "a v1 produced artifact cannot be relabeled because its subject, active-state, "
                "realization, causal-support, and readiness semantics are absent"
            ),
        )
    raise LegacyMigrationError(f"unsupported legacy artifact kind: {artifact_kind}")
