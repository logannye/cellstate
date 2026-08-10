"""Fail-closed migration helpers for legacy serialized contracts."""

from .v1_to_v2 import (
    LegacyArtifactKind,
    LegacyMigrationError,
    MigrationDisposition,
    MigrationRequirement,
    inspect_v1_payload,
)

__all__ = [
    "LegacyArtifactKind",
    "LegacyMigrationError",
    "MigrationDisposition",
    "MigrationRequirement",
    "inspect_v1_payload",
]
