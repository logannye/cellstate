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

SchemaVersion = Literal["2.0"]
SCHEMA_VERSION: SchemaVersion = "2.0"


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
    # IEEE-754 signed zero is not a distinct scientific value.  Normalizing it here keeps
    # content identities stable when equivalent values arrive through different serializers.
    if isinstance(value, float) and value == 0.0:
        return 0.0
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


def canonical_json_bytes(value: Any) -> bytes:
    """Encode one canonical JSON payload for both storage and content identity."""

    payload = _canonical_json_value(value)
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return encoded.encode("utf-8")


def canonical_fingerprint(value: Any) -> str:
    """Return a stable SHA-256 over canonical JSON-compatible data."""

    return sha256(canonical_json_bytes(value)).hexdigest()


def require_finite(value: float, *, name: str) -> float:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


class OntologyTerm(SchemaModel):
    """An extensible biological term, optionally grounded in an ontology."""

    label: str = Field(min_length=1)
    identifier: str | None = None
    namespace: str | None = None

    @model_validator(mode="after")
    def identity_is_unambiguous(self) -> OntologyTerm:
        for value, name in (
            (self.label, "ontology label"),
            (self.identifier, "ontology identifier"),
            (self.namespace, "ontology namespace"),
        ):
            if value is not None and (not value.strip() or value != value.strip()):
                raise ValueError(f"{name} must be nonblank and trimmed")
        if self.namespace is not None and self.identifier is None:
            raise ValueError("an ontology namespace requires an identifier")
        if self.identifier is not None and self.namespace is None and ":" not in self.identifier:
            raise ValueError("an unqualified ontology identifier requires an explicit namespace")
        if self.identifier is not None and self.namespace is not None and ":" in self.identifier:
            prefix, _ = self.identifier.split(":", 1)
            if prefix.casefold() != self.namespace.casefold():
                raise ValueError("ontology identifier prefix must match its explicit namespace")
        return self

    @property
    def key(self) -> str:
        if self.identifier is None:
            return "_".join(self.label.casefold().split())
        if self.namespace is not None:
            local_identifier = (
                self.identifier.split(":", 1)[1] if ":" in self.identifier else self.identifier
            )
            return f"{self.namespace.casefold()}:{local_identifier}"
        prefix, local_identifier = self.identifier.split(":", 1)
        return f"{prefix.casefold()}:{local_identifier}"


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


class CriterionOutcome(StrEnum):
    """Scientific pass/fail state, kept separate from calculation availability."""

    PASSED = "passed"
    FAILED = "failed"
    NOT_EVALUATED = "not_evaluated"
    UNSUPPORTED = "unsupported"


class CausalStatus(StrEnum):
    """Identification basis attached to a prediction or decision."""

    PREDICTIVE_ASSOCIATION = "predictive_association"
    IDENTIFIED_POPULATION_EFFECT = "identified_population_effect"
    TRANSPORTED_UNDER_ASSUMPTIONS = "transported_under_assumptions"
    MECHANISTIC_EXTRAPOLATION = "mechanistic_extrapolation"
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
    support_envelope_id: str | None = None
    support_envelope_fingerprint: str | None = Field(default=None, pattern=r"^[a-fA-F0-9]{64}$")
    training_support_id: str | None = None
    training_support_fingerprint: str | None = Field(default=None, pattern=r"^[a-fA-F0-9]{64}$")
    validation_evidence_ids: tuple[str, ...] = ()
    validation_evidence_fingerprints: dict[str, str] = Field(default_factory=dict)
    calibration_data_id: str | None = None
    seed: int = Field(ge=0)
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    warnings: tuple[str, ...] = ()

    @model_validator(mode="after")
    def event_evidence_is_content_addressed(self) -> ProvenanceRecord:
        required_text = (
            self.model_id,
            self.model_version,
            self.posterior_schema_id,
        )
        optional_text = (
            self.code_revision,
            self.support_envelope_id,
            self.training_support_id,
            self.calibration_data_id,
        )
        if any(not value.strip() for value in required_text) or any(
            value is not None and not value.strip() for value in optional_text
        ):
            raise ValueError("provenance identifiers must be nonblank")
        if any(not event_id.strip() for event_id in self.source_event_ids):
            raise ValueError("provenance source event IDs must be nonblank")
        if any(not evidence_id.strip() for evidence_id in self.validation_evidence_ids):
            raise ValueError("provenance validation evidence IDs must be nonblank")
        if len(self.source_event_ids) != len(set(self.source_event_ids)):
            raise ValueError("provenance source event IDs must be unique")
        if len(self.validation_evidence_ids) != len(set(self.validation_evidence_ids)):
            raise ValueError("provenance validation evidence IDs must be unique")
        if set(self.source_event_fingerprints) != set(self.source_event_ids):
            raise ValueError("provenance requires exactly one fingerprint per source event")
        if any(
            len(fingerprint) != 64
            or any(character not in "0123456789abcdefABCDEF" for character in fingerprint)
            for fingerprint in self.source_event_fingerprints.values()
        ):
            raise ValueError("source event fingerprints must be SHA-256 hex digests")
        if set(self.validation_evidence_fingerprints) != set(self.validation_evidence_ids):
            raise ValueError(
                "provenance requires exactly one fingerprint per validation evidence artifact"
            )
        if any(
            len(fingerprint) != 64
            or any(character not in "0123456789abcdefABCDEF" for character in fingerprint)
            for fingerprint in self.validation_evidence_fingerprints.values()
        ):
            raise ValueError("validation evidence fingerprints must be SHA-256 hex digests")
        envelope_fields = (self.support_envelope_id, self.support_envelope_fingerprint)
        if (envelope_fields[0] is None) is not (envelope_fields[1] is None):
            raise ValueError("support envelope ID and fingerprint must be declared together")
        training_fields = (self.training_support_id, self.training_support_fingerprint)
        if (training_fields[0] is None) is not (training_fields[1] is None):
            raise ValueError("training support ID and fingerprint must be declared together")
        if self.generated_at.tzinfo is None or self.generated_at.utcoffset() is None:
            raise ValueError("provenance generation time must be timezone-aware")
        return self

    @property
    def scientific_evidence_ids(self) -> frozenset[str]:
        """Evidence IDs available to scientific support, causal, and transport claims."""

        return frozenset((*self.source_event_ids, *self.validation_evidence_ids))
