"""Shared, serialization-safe primitives for the public domain model."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from datetime import UTC, datetime
from enum import Enum, StrEnum
from hashlib import sha256
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SchemaVersion = Literal["1.0"]
SCHEMA_VERSION: SchemaVersion = "1.0"


class SchemaModel(BaseModel):
    """Base for frozen top-level boundary objects with strict input validation."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        validate_default=True,
        allow_inf_nan=False,
    )


def _canonical_json_value(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return _canonical_json_value(value.model_dump(mode="python", exclude_none=False))
    if isinstance(value, Enum):
        return _canonical_json_value(value.value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, Mapping):
        return {
            str(key.value if isinstance(key, Enum) else key): _canonical_json_value(item)
            for key, item in value.items()
        }
    if isinstance(value, (set, frozenset)):
        normalized = [_canonical_json_value(item) for item in value]
        return sorted(
            normalized,
            key=lambda item: json.dumps(
                item,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ),
        )
    if isinstance(value, (tuple, list)):
        return [_canonical_json_value(item) for item in value]
    return value


def canonical_fingerprint(value: BaseModel | dict[str, Any]) -> str:
    """Return a stable SHA-256 over canonical JSON-compatible data."""

    payload = _canonical_json_value(value)
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return sha256(encoded.encode("utf-8")).hexdigest()


def require_finite(value: float, *, name: str) -> float:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


class OntologyTerm(SchemaModel):
    """An extensible biological term, optionally grounded in an ontology."""

    label: str = Field(min_length=1)
    identifier: str | None = None
    namespace: str | None = None

    @property
    def key(self) -> str:
        return self.identifier or self.label.casefold().replace(" ", "_")


class Quantity(SchemaModel):
    value: float
    units: str = Field(min_length=1)

    @field_validator("value")
    @classmethod
    def finite_value(cls, value: float) -> float:
        return require_finite(value, name="quantity value")


class ArtifactRef(SchemaModel):
    """Content-addressed reference for arrays, images, tables, or posterior samples."""

    uri: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    media_type: str = Field(min_length=1)
    schema_version: SchemaVersion = SCHEMA_VERSION
    dtype: str | None = None
    shape: tuple[int, ...] = ()
    dimensions: tuple[str, ...] = ()

    @field_validator("shape")
    @classmethod
    def nonnegative_shape(cls, shape: tuple[int, ...]) -> tuple[int, ...]:
        if any(size < 0 for size in shape):
            raise ValueError("artifact shape entries must be nonnegative")
        return shape

    @model_validator(mode="after")
    def axis_labels_match_shape(self) -> ArtifactRef:
        if self.dimensions and len(self.dimensions) != len(self.shape):
            raise ValueError("artifact axis labels must match the number of shape axes")
        if len(self.dimensions) != len(set(self.dimensions)):
            raise ValueError("artifact axis labels must be unique")
        return self


class EvidenceStatus(StrEnum):
    OBSERVED = "observed"
    INFERRED = "inferred_with_support"
    UNIDENTIFIABLE = "unidentifiable"


class SupportStatus(StrEnum):
    SUPPORTED = "supported"
    NOT_EVALUATED = "not_evaluated"
    UNSUPPORTED = "unsupported"


class ProvenanceRecord(SchemaModel):
    """Minimum audit trail attached to every produced belief or forecast."""

    model_id: str = Field(min_length=1)
    model_version: str = Field(min_length=1)
    model_fingerprint: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    posterior_schema_id: str = Field(min_length=1)
    schema_version: SchemaVersion = SCHEMA_VERSION
    code_revision: str | None = None
    query_fingerprint: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    history_fingerprint: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    history_structure_fingerprint: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    context_fingerprint: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    source_event_ids: tuple[str, ...] = ()
    source_event_fingerprints: dict[str, str] = Field(default_factory=dict)
    training_support_id: str | None = None
    calibration_data_id: str | None = None
    seed: int = Field(ge=0)
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    warnings: tuple[str, ...] = ()

    @model_validator(mode="after")
    def event_evidence_is_content_addressed(self) -> ProvenanceRecord:
        if len(self.source_event_ids) != len(set(self.source_event_ids)):
            raise ValueError("provenance source event IDs must be unique")
        if set(self.source_event_fingerprints) != set(self.source_event_ids):
            raise ValueError("provenance requires exactly one fingerprint per source event")
        if any(
            len(fingerprint) != 64
            or any(character not in "0123456789abcdefABCDEF" for character in fingerprint)
            for fingerprint in self.source_event_fingerprints.values()
        ):
            raise ValueError("source event fingerprints must be SHA-256 hex digests")
        if self.generated_at.tzinfo is None or self.generated_at.utcoffset() is None:
            raise ValueError("provenance generation time must be timezone-aware")
        return self
