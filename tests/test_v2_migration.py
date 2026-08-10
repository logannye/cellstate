from __future__ import annotations

from copy import deepcopy
from typing import cast

import pytest

from cellstate.schema import (
    LegacyArtifactKind,
    LegacyMigrationError,
    MigrationDisposition,
    inspect_v1_payload,
)


@pytest.mark.parametrize(
    "artifact_kind",
    [
        LegacyArtifactKind.STATE_QUERY,
        LegacyArtifactKind.CELL_HISTORY,
        LegacyArtifactKind.ESTIMATE_REQUEST,
    ],
)
def test_v1_inputs_require_explicit_reviewed_annotations(
    artifact_kind: LegacyArtifactKind,
) -> None:
    payload = {"schema_version": "1.0", "legacy_id": artifact_kind.value}
    original = deepcopy(payload)

    requirement = inspect_v1_payload(payload, artifact_kind=artifact_kind)

    assert requirement.artifact_kind is artifact_kind
    assert requirement.disposition is MigrationDisposition.EXPLICIT_ANNOTATIONS_REQUIRED
    assert requirement.required_decisions
    assert len(requirement.legacy_fingerprint) == 64
    assert payload == original


@pytest.mark.parametrize(
    "artifact_kind",
    [
        LegacyArtifactKind.CELL_STATE_BELIEF,
        LegacyArtifactKind.STATE_FORECAST,
        LegacyArtifactKind.INTERVENTION_PLAN,
    ],
)
def test_v1_produced_artifacts_require_reestimation(
    artifact_kind: LegacyArtifactKind,
) -> None:
    requirement = inspect_v1_payload(
        {"schema_version": "1.0", "legacy_id": artifact_kind.value},
        artifact_kind=artifact_kind,
    )

    assert requirement.disposition is MigrationDisposition.REESTIMATION_REQUIRED
    assert any(
        decision.startswith(("re-estimate", "regenerate"))
        for decision in requirement.required_decisions
    )


@pytest.mark.parametrize("schema_version", ["2.0", "0.2", None])
def test_migration_inspection_rejects_non_v1_payloads(
    schema_version: str | None,
) -> None:
    with pytest.raises(LegacyMigrationError, match=r"requires schema_version '1\.0'"):
        inspect_v1_payload(
            {"schema_version": schema_version},
            artifact_kind=LegacyArtifactKind.STATE_QUERY,
        )


def test_migration_inspection_fails_closed_for_unknown_artifact_kinds() -> None:
    unknown_kind = cast(LegacyArtifactKind, "unknown")

    with pytest.raises(LegacyMigrationError, match="unsupported legacy artifact kind"):
        inspect_v1_payload(
            {"schema_version": "1.0"},
            artifact_kind=unknown_kind,
        )
