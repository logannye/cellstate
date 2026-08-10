"""Experimental contracts for public-real biological dataset evidence.

These models describe source evidence and the claims it may support. They do not download,
normalize, impute, pair, or biologically interpret measurements.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import date, datetime
from enum import StrEnum
from typing import Any, Literal
from urllib.parse import urlparse

from pydantic import ConfigDict, Field, field_validator, model_validator

from cellstate.domain.common import (
    OntologyTerm,
    SchemaModel,
    canonical_fingerprint,
    require_finite,
)
from cellstate.domain.common import (
    canonical_json_bytes as encode_canonical_json_bytes,
)
from cellstate.domain.query import SystemBoundary
from cellstate.training.objectives import LossKind

DatasetManifestSchemaVersion = Literal["0.3-experimental"]
DATASET_MANIFEST_SCHEMA_VERSION: DatasetManifestSchemaVersion = "0.3-experimental"


class ManifestModel(SchemaModel):
    """Strict Python-input validation for the experimental manifest boundary."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        validate_default=True,
        revalidate_instances="always",
        allow_inf_nan=False,
        strict=True,
    )


def _term_keys(terms: tuple[OntologyTerm, ...]) -> list[str]:
    return [term.key for term in terms]


def _require_unique_terms(terms: tuple[OntologyTerm, ...], *, name: str) -> None:
    keys = _term_keys(terms)
    if len(keys) != len(set(keys)):
        raise ValueError(f"{name} must be unique")


def _require_unique_strings(values: tuple[str, ...], *, name: str) -> None:
    if any(not value.strip() for value in values):
        raise ValueError(f"{name} entries must be nonempty after trimming")
    if len(values) != len(set(values)):
        raise ValueError(f"{name} must be unique")


def _require_canonical_id(value: str, *, name: str) -> str:
    if not value.strip():
        raise ValueError(f"{name} must be nonempty after trimming")
    if value != value.strip():
        raise ValueError(f"{name} must not contain leading or trailing whitespace")
    return value


class AccessMode(StrEnum):
    """Public access may be direct or require registration, but not approval."""

    OPEN_DOWNLOAD = "open_download"
    REGISTRATION_REQUIRED = "registration_required"


class SourceKind(StrEnum):
    RAW = "raw"
    PROCESSED = "processed"
    METADATA = "metadata"
    DOCUMENTATION = "documentation"


class SourceArtifact(ManifestModel):
    source_id: str = Field(min_length=1)
    kind: SourceKind
    uri: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    media_type: str = Field(min_length=1)
    access_mode: AccessMode = AccessMode.OPEN_DOWNLOAD
    accession: str = Field(min_length=1)
    release: str = Field(min_length=1)
    parent_study_accession: str | None = Field(default=None, min_length=1)
    parent_study_release: str | None = Field(default=None, min_length=1)
    byte_count: int = Field(gt=0)
    retrieved_at: datetime

    @field_validator("uri")
    @classmethod
    def uri_is_public_and_remote(cls, value: str) -> str:
        parsed = urlparse(value)
        if parsed.scheme.casefold() not in {"https", "http", "ftp", "s3", "gs", "drs"}:
            raise ValueError("source artifact URI must use a public remote-data scheme")
        if not parsed.netloc:
            raise ValueError("source artifact URI must be absolute")
        if parsed.username is not None or parsed.password is not None:
            raise ValueError("public source artifact URI must not embed credentials")
        if (parsed.hostname or "").casefold() in {"localhost", "127.0.0.1", "::1"}:
            raise ValueError("public source artifact URI must not resolve to localhost")
        return value

    @field_validator("sha256")
    @classmethod
    def canonicalize_sha256(cls, value: str) -> str:
        return value.casefold()

    @field_validator("retrieved_at")
    @classmethod
    def retrieved_at_is_timezone_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("source retrieval time must be timezone-aware")
        return value

    @model_validator(mode="after")
    def supplementary_study_link_is_complete(self) -> SourceArtifact:
        if (self.parent_study_accession is None) != (self.parent_study_release is None):
            raise ValueError(
                "supplementary artifacts must declare parent study accession and release together"
            )
        return self


class DatasetSliceKind(StrEnum):
    """Whether the reviewed dataset is an entire record axis or an exact selected cohort."""

    WHOLE_ARTIFACT = "whole_artifact"
    CONTENT_ADDRESSED_SELECTION = "content_addressed_selection"


class CohortSelectionStage(ManifestModel):
    """One auditable attrition step in a content-addressed cohort selection."""

    stage_id: str = Field(min_length=1)
    input_record_count: int = Field(gt=0)
    output_record_count: int = Field(gt=0)
    criterion: str = Field(min_length=1)
    source_ids: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def stage_is_canonical_and_nonexpansive(self) -> CohortSelectionStage:
        _require_canonical_id(self.stage_id, name="cohort-selection stage ID")
        _require_unique_strings(self.source_ids, name="cohort-selection stage source IDs")
        if tuple(sorted(self.source_ids)) != self.source_ids:
            raise ValueError("cohort-selection stage source IDs must be sorted")
        if self.output_record_count > self.input_record_count:
            raise ValueError("a cohort-selection stage cannot create source records")
        if self.criterion != self.criterion.strip():
            raise ValueError("cohort-selection criterion must be trimmed")
        return self


class DatasetSliceSpec(ManifestModel):
    """Exact record membership to which one manifest's design and assessments apply."""

    kind: DatasetSliceKind
    slice_id: str = Field(min_length=1)
    selection_source_ids: tuple[str, ...] = Field(min_length=1)
    record_id_field: str = Field(min_length=1)
    selected_record_ids_uri: str | None = Field(default=None, min_length=1)
    selected_record_ids_encoding: Literal["canonical_json_utf8_string_array_v1"] = (
        "canonical_json_utf8_string_array_v1"
    )
    selected_record_ids_sha256: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    selected_record_count: int = Field(gt=0)
    selected_subject_count: int = Field(gt=0)
    selector_id: str | None = Field(default=None, min_length=1)
    selector_version: str | None = Field(default=None, min_length=1)
    selector_sha256: str | None = Field(default=None, pattern=r"^[a-fA-F0-9]{64}$")
    selection_stages: tuple[CohortSelectionStage, ...] = ()

    @property
    def fingerprint(self) -> str:
        payload = self.model_dump(mode="python")
        payload["selection_source_ids"] = sorted(self.selection_source_ids)
        return canonical_fingerprint(payload)

    @field_validator("selected_record_ids_sha256", "selector_sha256")
    @classmethod
    def canonicalize_sha256(cls, value: str | None) -> str | None:
        return value.casefold() if value is not None else None

    @field_validator("selected_record_ids_uri")
    @classmethod
    def membership_uri_is_public_and_remote(cls, value: str | None) -> str | None:
        if value is None:
            return None
        parsed = urlparse(value)
        if parsed.scheme.casefold() not in {"https", "http", "ftp", "s3", "gs", "drs"}:
            raise ValueError("selected-record membership URI must use a public remote scheme")
        if not parsed.netloc or parsed.username is not None or parsed.password is not None:
            raise ValueError("selected-record membership URI must be public and absolute")
        if (parsed.hostname or "").casefold() in {"localhost", "127.0.0.1", "::1"}:
            raise ValueError("selected-record membership URI must not resolve to localhost")
        return value

    @model_validator(mode="after")
    def selection_is_exact_and_reproducible(self) -> DatasetSliceSpec:
        _require_canonical_id(self.slice_id, name="dataset slice ID")
        _require_canonical_id(self.record_id_field, name="dataset slice record-ID field")
        if self.selected_record_ids_uri is not None:
            _require_canonical_id(
                self.selected_record_ids_uri,
                name="selected-record membership URI",
            )
        _require_unique_strings(self.selection_source_ids, name="slice selection source IDs")
        if tuple(sorted(self.selection_source_ids)) != self.selection_source_ids:
            raise ValueError("slice selection source IDs must be sorted")
        if self.selected_subject_count > self.selected_record_count:
            raise ValueError("selected subjects cannot exceed selected records")
        selector_values = (self.selector_id, self.selector_version, self.selector_sha256)
        if self.kind is DatasetSliceKind.WHOLE_ARTIFACT:
            if any(value is not None for value in selector_values) or self.selection_stages:
                raise ValueError("whole-artifact slices must not declare a selector or stages")
        elif not all(value is not None for value in selector_values):
            raise ValueError("content-addressed selections require an exact selector identity")
        else:
            assert self.selector_id is not None
            assert self.selector_version is not None
            _require_canonical_id(self.selector_id, name="dataset slice selector ID")
            _require_canonical_id(self.selector_version, name="dataset slice selector version")
            if not self.selection_stages:
                raise ValueError("content-addressed selections require cohort-selection stages")
            stage_ids = tuple(stage.stage_id for stage in self.selection_stages)
            _require_unique_strings(stage_ids, name="cohort-selection stage IDs")
            for previous, current in zip(
                self.selection_stages, self.selection_stages[1:], strict=False
            ):
                if previous.output_record_count != current.input_record_count:
                    raise ValueError("cohort-selection stage counts must form a contiguous flow")
            if self.selection_stages[-1].output_record_count != self.selected_record_count:
                raise ValueError("the final cohort-selection stage must yield the selected records")
            stage_sources = {
                source_id for stage in self.selection_stages for source_id in stage.source_ids
            }
            if not stage_sources <= set(self.selection_source_ids):
                raise ValueError("cohort-selection stage sources must be slice selection sources")
        return self


class PermissionStatus(StrEnum):
    PERMITTED = "permitted"
    PROHIBITED = "prohibited"
    CONDITIONAL = "conditional"
    UNKNOWN = "unknown"


class DataUseCase(StrEnum):
    RESEARCH_MODEL_TRAINING = "research_model_training"
    COMMERCIAL_MODEL_TRAINING = "commercial_model_training"
    BENCHMARK_EVALUATION = "benchmark_evaluation"
    SOURCE_DATA_REDISTRIBUTION = "source_data_redistribution"
    DERIVED_MODEL_DISTRIBUTION = "derived_model_distribution"
    DERIVED_MODEL_PUBLICATION = "derived_model_publication"


class UsePermission(ManifestModel):
    use_case: DataUseCase
    status: PermissionStatus
    conditions: tuple[str, ...] = ()

    @model_validator(mode="after")
    def unresolved_or_restricted_use_is_explained(self) -> UsePermission:
        _require_unique_strings(self.conditions, name="use-permission conditions")
        if self.status is not PermissionStatus.PERMITTED and not self.conditions:
            raise ValueError("non-permitted use requires an explicit condition or reason")
        if self.status is PermissionStatus.PERMITTED and self.conditions:
            raise ValueError(
                "permitted use cannot retain unresolved conditions; use conditional status"
            )
        return self


class DataUsePolicy(ManifestModel):
    policy_id: str = Field(min_length=1)
    source_ids: tuple[str, ...] = Field(min_length=1)
    license_name: str = Field(min_length=1)
    terms_url: str = Field(min_length=1)
    reviewed_on: date
    spdx_identifier: str | None = None
    permissions: tuple[UsePermission, ...] = Field(min_length=1)
    attribution_requirements: tuple[str, ...] = ()

    @field_validator("terms_url")
    @classmethod
    def terms_url_is_absolute(cls, value: str) -> str:
        parsed = urlparse(value)
        if parsed.scheme.casefold() not in {"https", "http"} or not parsed.netloc:
            raise ValueError("terms URL must be an absolute HTTP(S) URL")
        if parsed.username is not None or parsed.password is not None:
            raise ValueError("terms URL must not embed credentials")
        if (parsed.hostname or "").casefold() in {"localhost", "127.0.0.1", "::1"}:
            raise ValueError("terms URL must not resolve to localhost")
        return value

    @model_validator(mode="after")
    def scope_and_permission_matrix_are_complete(self) -> DataUsePolicy:
        _require_unique_strings(self.source_ids, name="data-use source IDs")
        _require_unique_strings(
            self.attribution_requirements,
            name="attribution requirements",
        )
        use_cases = [permission.use_case for permission in self.permissions]
        if len(use_cases) != len(set(use_cases)) or set(use_cases) != set(DataUseCase):
            raise ValueError("data-use policy requires exactly one permission per use case")
        return self


class EffectiveDataUsePermission(ManifestModel):
    """Most-restrictive legal result across every policy layer touching exact sources."""

    use_case: DataUseCase
    source_ids: tuple[str, ...] = Field(min_length=1)
    status: PermissionStatus
    applicable_policy_ids: tuple[str, ...] = Field(min_length=1)
    conditions: tuple[str, ...] = ()
    attribution_requirements: tuple[str, ...] = ()

    @model_validator(mode="after")
    def report_is_canonical(self) -> EffectiveDataUsePermission:
        for values, name in (
            (self.source_ids, "effective-permission source IDs"),
            (self.applicable_policy_ids, "effective-permission policy IDs"),
            (self.conditions, "effective-permission conditions"),
            (self.attribution_requirements, "effective-permission attribution requirements"),
        ):
            _require_unique_strings(values, name=name)
            if tuple(sorted(values)) != values:
                raise ValueError(f"{name} must be sorted")
        if self.status is not PermissionStatus.PERMITTED and not self.conditions:
            raise ValueError("non-permitted effective use requires a condition or reason")
        if self.status is PermissionStatus.PERMITTED and self.conditions:
            raise ValueError("permitted effective use cannot retain unresolved conditions")
        return self


class PublicRealDataOrigin(ManifestModel):
    kind: Literal["public_real"] = "public_real"
    publicly_downloadable: Literal[True] = True
    repository: str = Field(min_length=1)
    study_accession: str = Field(min_length=1)
    publication_doi: str | None = None
    release: str = Field(min_length=1)
    species: tuple[OntologyTerm, ...] = Field(min_length=1)
    biological_systems: tuple[OntologyTerm, ...] = Field(min_length=1)

    @field_validator("publicly_downloadable", mode="before")
    @classmethod
    def public_download_flag_is_a_boolean_literal(cls, value: object) -> object:
        if value is not True:
            raise ValueError("publicly downloadable must be the boolean literal true")
        return value

    @field_validator("publication_doi")
    @classmethod
    def doi_is_canonical(cls, value: str | None) -> str | None:
        if value is not None and (
            not value.startswith("10.") or any(character.isspace() for character in value)
        ):
            raise ValueError("publication DOI must be canonical and omit a URL prefix")
        return value

    @model_validator(mode="after")
    def ontology_scopes_are_unique(self) -> PublicRealDataOrigin:
        _require_unique_terms(self.species, name="origin species")
        _require_unique_terms(self.biological_systems, name="origin biological systems")
        return self


class ExperimentalUnitLevel(StrEnum):
    ORGANISM = "organism"
    DONOR = "donor"
    TISSUE = "tissue"
    CLONE = "clone"
    CULTURE = "culture"
    ORGANOID = "organoid"
    PLATE = "plate"
    WELL = "well"
    SAMPLE = "sample"
    CELL = "cell"
    SPATIAL_REGION = "spatial_region"


class UnitIdentityExpressionKind(StrEnum):
    """How one canonical experimental-unit identity is obtained."""

    SOURCE_FIELD = "source_field"
    COMPOSITE_SOURCE_FIELDS = "composite_source_fields"
    MANIFEST_CONSTANT = "manifest_constant"


class CompositeIdentityEncoding(StrEnum):
    """Collision-safe canonical encodings supported for composite unit identities."""

    CANONICAL_JSON_UTF8_STRING_ARRAY_V1 = "canonical_json_utf8_string_array_v1"


class UnitIdentityExpression(ManifestModel):
    """Typed, content-stable expression for an experimental-unit identity.

    Source-field order is significant for a composite expression. The declared JSON-array
    encoding prevents delimiter collisions and requires each source value to be represented as
    its exact UTF-8 string value before canonical JSON serialization.
    """

    kind: UnitIdentityExpressionKind
    source_fields: tuple[str, ...] = ()
    composite_encoding: CompositeIdentityEncoding | None = None
    constant_value: str | None = Field(default=None, min_length=1)

    @property
    def fingerprint(self) -> str:
        return canonical_fingerprint(self.model_dump(mode="python"))

    def evaluate(self, record: Mapping[str, object]) -> str:
        """Evaluate the exact identity, rejecting absent or non-string source values."""

        if self.kind is UnitIdentityExpressionKind.MANIFEST_CONSTANT:
            assert self.constant_value is not None
            return self.constant_value
        values: list[str] = []
        for field_name in self.source_fields:
            if field_name not in record:
                raise ValueError(f"unit-identity source field is absent: {field_name}")
            value = record[field_name]
            if type(value) is not str:
                raise ValueError(
                    f"unit-identity source field must contain exact strings: {field_name}"
                )
            values.append(value)
        if self.kind is UnitIdentityExpressionKind.SOURCE_FIELD:
            return values[0]
        return json.dumps(values, ensure_ascii=False, separators=(",", ":"))

    @model_validator(mode="after")
    def expression_is_unambiguous(self) -> UnitIdentityExpression:
        _require_unique_strings(self.source_fields, name="unit-identity source fields")
        for field_name in self.source_fields:
            _require_canonical_id(field_name, name="unit-identity source field")
        if self.constant_value is not None:
            _require_canonical_id(self.constant_value, name="unit-identity constant")

        if self.kind is UnitIdentityExpressionKind.SOURCE_FIELD:
            if (
                len(self.source_fields) != 1
                or self.composite_encoding is not None
                or self.constant_value is not None
            ):
                raise ValueError(
                    "source-field identity requires exactly one field and no encoding or constant"
                )
        elif self.kind is UnitIdentityExpressionKind.COMPOSITE_SOURCE_FIELDS:
            if (
                len(self.source_fields) < 2
                or self.composite_encoding is None
                or self.constant_value is not None
            ):
                raise ValueError(
                    "composite identity requires at least two fields and a canonical encoding"
                )
        elif (
            self.source_fields or self.composite_encoding is not None or self.constant_value is None
        ):
            raise ValueError("manifest-constant identity requires only one nonempty constant value")
        return self


def _source_field_identity(field_name: str) -> UnitIdentityExpression:
    """Resolve a legacy field declaration without weakening its exact identity semantics."""

    return UnitIdentityExpression(
        kind=UnitIdentityExpressionKind.SOURCE_FIELD,
        source_fields=(field_name,),
    )


class ExperimentalUnitSpec(ManifestModel):
    level: ExperimentalUnitLevel
    id_field: str | None = Field(default=None, min_length=1)
    identity: UnitIdentityExpression | None = None
    source_ids: tuple[str, ...] = Field(min_length=1)
    parent_level: ExperimentalUnitLevel | None = None

    @property
    def resolved_identity(self) -> UnitIdentityExpression:
        if self.identity is not None:
            return self.identity
        assert self.id_field is not None
        return _source_field_identity(self.id_field)

    @model_validator(mode="after")
    def unit_spec_is_coherent(self) -> ExperimentalUnitSpec:
        if self.parent_level is self.level:
            raise ValueError("an experimental-unit level cannot parent itself")
        if (self.id_field is None) is (self.identity is None):
            raise ValueError(
                "experimental unit requires exactly one legacy ID field or typed identity"
            )
        if self.id_field is not None:
            _require_canonical_id(self.id_field, name="experimental-unit ID field")
        _require_unique_strings(self.source_ids, name="experimental-unit source IDs")
        return self


class SamplingSubjectKind(StrEnum):
    INDIVIDUAL_CELL = "individual_cell"
    CLONE = "clone"
    SAMPLE = "sample"
    POPULATION = "population"
    SPATIAL_REGION = "spatial_region"


class SamplingMode(StrEnum):
    ENDPOINT_DESTRUCTIVE = "endpoint_destructive"
    REPEATED_POPULATION_DESTRUCTIVE = "repeated_population_destructive"
    LINEAGE_LINKED_ENDPOINT = "lineage_linked_endpoint"
    PARTIAL_NONDESTRUCTIVE = "partial_nondestructive"
    LONGITUDINAL_NONDESTRUCTIVE = "longitudinal_nondestructive"


class SubjectLinkage(StrEnum):
    NONE = "none"
    SAME_POPULATION = "same_population"
    SAME_CLONE = "same_clone"
    LINEAGE = "lineage"
    SAME_CELL = "same_cell"
    SAME_SPATIAL_REGION = "same_spatial_region"


class SamplingDesign(ManifestModel):
    subject_kind: SamplingSubjectKind
    subject_unit: ExperimentalUnitLevel
    subject_id_field: str | None = Field(default=None, min_length=1)
    subject_identity: UnitIdentityExpression | None = None
    source_ids: tuple[str, ...] = Field(min_length=1)
    mode: SamplingMode
    linkage: SubjectLinkage
    time_field: str | None = None
    source_time_units: str | None = None
    time_window_id: str | None = None
    canonical_time_units: Literal["s"] = "s"
    attrition_field: str | None = None

    @property
    def resolved_subject_identity(self) -> UnitIdentityExpression:
        if self.subject_identity is not None:
            return self.subject_identity
        assert self.subject_id_field is not None
        return _source_field_identity(self.subject_id_field)

    @model_validator(mode="after")
    def sampling_and_linkage_are_coherent(self) -> SamplingDesign:
        _require_unique_strings(self.source_ids, name="sampling source IDs")
        if (self.subject_id_field is None) is (self.subject_identity is None):
            raise ValueError(
                "sampling requires exactly one legacy subject ID field or typed identity"
            )
        if self.subject_id_field is not None:
            _require_canonical_id(self.subject_id_field, name="sampling subject ID field")
        if (self.time_field is None) != (self.source_time_units is None):
            raise ValueError("sampling time field and source units must be declared together")
        if self.time_window_id is not None:
            _require_canonical_id(self.time_window_id, name="sampling time-window ID")
        if self.time_field is not None and self.time_window_id is not None:
            raise ValueError("sampling must use either a time field or a time window, not both")
        repeated_modes = {
            SamplingMode.REPEATED_POPULATION_DESTRUCTIVE,
            SamplingMode.LINEAGE_LINKED_ENDPOINT,
            SamplingMode.PARTIAL_NONDESTRUCTIVE,
            SamplingMode.LONGITUDINAL_NONDESTRUCTIVE,
        }
        if self.mode in repeated_modes and self.time_field is None and self.time_window_id is None:
            raise ValueError(
                "repeated, lineage, or longitudinal sampling requires a time field or window"
            )
        if (
            self.mode is SamplingMode.ENDPOINT_DESTRUCTIVE
            and self.linkage is not SubjectLinkage.NONE
        ):
            raise ValueError("a single destructive endpoint cannot claim longitudinal linkage")
        if self.mode is SamplingMode.REPEATED_POPULATION_DESTRUCTIVE and (
            self.subject_kind is not SamplingSubjectKind.POPULATION
            or self.linkage is not SubjectLinkage.SAME_POPULATION
        ):
            raise ValueError("repeated destructive sampling requires population linkage")
        if self.mode in {
            SamplingMode.PARTIAL_NONDESTRUCTIVE,
            SamplingMode.LONGITUDINAL_NONDESTRUCTIVE,
        } and (
            self.subject_kind is not SamplingSubjectKind.INDIVIDUAL_CELL
            or self.linkage is not SubjectLinkage.SAME_CELL
        ):
            raise ValueError("nondestructive longitudinal sampling requires same-cell linkage")
        if self.mode is SamplingMode.LINEAGE_LINKED_ENDPOINT and self.linkage not in {
            SubjectLinkage.SAME_CLONE,
            SubjectLinkage.LINEAGE,
        }:
            raise ValueError("lineage-linked endpoints require clone or lineage linkage")
        if self.linkage is SubjectLinkage.SAME_SPATIAL_REGION and (
            self.subject_kind is not SamplingSubjectKind.SPATIAL_REGION
        ):
            raise ValueError("spatial-region linkage requires a spatial-region subject")
        return self


class ControlPredicateValueType(StrEnum):
    STRING = "string"
    INTEGER = "integer"
    NUMBER = "number"
    BOOLEAN = "boolean"


class MatchedControlPredicate(ManifestModel):
    """One typed scalar equality in a conjunctive control definition."""

    source_field: str = Field(min_length=1)
    value_type: ControlPredicateValueType
    equals: str | int | float | bool

    @model_validator(mode="after")
    def equality_is_strictly_typed(self) -> MatchedControlPredicate:
        _require_canonical_id(self.source_field, name="matched-control predicate field")
        expected_type = {
            ControlPredicateValueType.STRING: str,
            ControlPredicateValueType.INTEGER: int,
            ControlPredicateValueType.NUMBER: float,
            ControlPredicateValueType.BOOLEAN: bool,
        }[self.value_type]
        if type(self.equals) is not expected_type:
            raise ValueError("matched-control predicate value does not match its scalar type")
        if isinstance(self.equals, str):
            _require_canonical_id(self.equals, name="matched-control predicate value")
        elif isinstance(self.equals, float):
            require_finite(self.equals, name="matched-control predicate value")
        return self

    def matches(self, record: Mapping[str, object]) -> bool:
        """Apply exact scalar equality without Python's bool/int or int/float coercion."""

        observed = record.get(self.source_field)
        return type(observed) is type(self.equals) and observed == self.equals


class MatchedControlDefinition(ManifestModel):
    """Exact conjunctive control predicate and matching stratum for an endpoint contrast."""

    predicates: tuple[MatchedControlPredicate, ...] = Field(min_length=2)
    stratum_identity: UnitIdentityExpression
    source_ids: tuple[str, ...] = Field(min_length=1)

    def matches(self, record: Mapping[str, object]) -> bool:
        """Return true only when every typed equality in the conjunction holds."""

        return all(predicate.matches(record) for predicate in self.predicates)

    @model_validator(mode="after")
    def definition_is_exact_and_source_backed(self) -> MatchedControlDefinition:
        predicate_fields = tuple(predicate.source_field for predicate in self.predicates)
        _require_unique_strings(predicate_fields, name="matched-control predicate fields")
        if tuple(sorted(predicate_fields)) != predicate_fields:
            raise ValueError("matched-control predicates must be sorted by source field")
        _require_unique_strings(self.source_ids, name="matched-control source IDs")
        if tuple(sorted(self.source_ids)) != self.source_ids:
            raise ValueError("matched-control source IDs must be sorted")
        return self


class RandomizedEndpointContrast(ManifestModel):
    """A randomized post-assignment endpoint with explicitly absent baseline measurement."""

    assignment_time_seconds: float
    endpoint_time_seconds: float
    baseline_observation_present: Literal[False] = False
    matched_control: MatchedControlDefinition
    source_ids: tuple[str, ...] = Field(min_length=1)

    @field_validator("assignment_time_seconds", "endpoint_time_seconds")
    @classmethod
    def times_are_finite(cls, value: float) -> float:
        return require_finite(value, name="randomized endpoint-contrast time")

    @field_validator("baseline_observation_present", mode="before")
    @classmethod
    def baseline_absence_is_literal_false(cls, value: object) -> object:
        if value is not False:
            raise ValueError(
                "endpoint contrast must explicitly record that no baseline observation exists"
            )
        return value

    @model_validator(mode="after")
    def contrast_is_ordered_and_source_backed(self) -> RandomizedEndpointContrast:
        if self.endpoint_time_seconds <= self.assignment_time_seconds:
            raise ValueError("randomized endpoint must occur after assignment")
        _require_unique_strings(self.source_ids, name="endpoint-contrast source IDs")
        if tuple(sorted(self.source_ids)) != self.source_ids:
            raise ValueError("endpoint-contrast source IDs must be sorted")
        if not set(self.matched_control.source_ids) <= set(self.source_ids):
            raise ValueError("matched-control sources must be endpoint-contrast sources")
        return self


class ExperimentalDesign(ManifestModel):
    units: tuple[ExperimentalUnitSpec, ...] = Field(min_length=1)
    sampling: SamplingDesign
    default_split_unit: ExperimentalUnitLevel
    biological_replicate_unit: ExperimentalUnitLevel | None = None
    randomization_unit: ExperimentalUnitLevel | None = None
    matched_control_field: str | None = None
    randomized_endpoint_contrast: RandomizedEndpointContrast | None = None
    batch_fields: tuple[str, ...] = ()

    @model_validator(mode="after")
    def unit_graph_and_split_are_valid(self) -> ExperimentalDesign:
        by_level = {unit.level: unit for unit in self.units}
        if len(by_level) != len(self.units):
            raise ValueError("experimental-unit levels must be unique")
        identity_fingerprints = tuple(unit.resolved_identity.fingerprint for unit in self.units)
        _require_unique_strings(
            identity_fingerprints,
            name="experimental-unit identity expressions",
        )
        referenced = {
            self.default_split_unit,
            self.sampling.subject_unit,
            *(
                level
                for level in (self.biological_replicate_unit, self.randomization_unit)
                if level is not None
            ),
        }
        parents = {unit.parent_level for unit in self.units if unit.parent_level is not None}
        if unknown := (referenced | parents) - set(by_level):
            raise ValueError(f"experimental design references undeclared unit levels: {unknown}")
        for start in by_level:
            seen: set[ExperimentalUnitLevel] = set()
            current: ExperimentalUnitLevel | None = start
            while current is not None:
                if current in seen:
                    raise ValueError("experimental-unit hierarchy must be acyclic")
                seen.add(current)
                current = by_level[current].parent_level
        subject_unit = by_level[self.sampling.subject_unit]
        if subject_unit.resolved_identity != self.sampling.resolved_subject_identity:
            raise ValueError("sampling subject identity must match its declared unit")
        required_subject_units = {
            SamplingSubjectKind.INDIVIDUAL_CELL: {ExperimentalUnitLevel.CELL},
            SamplingSubjectKind.CLONE: {ExperimentalUnitLevel.CLONE},
            SamplingSubjectKind.SAMPLE: {ExperimentalUnitLevel.SAMPLE},
            SamplingSubjectKind.SPATIAL_REGION: {ExperimentalUnitLevel.SPATIAL_REGION},
            SamplingSubjectKind.POPULATION: set(ExperimentalUnitLevel)
            - {
                ExperimentalUnitLevel.CELL,
                ExperimentalUnitLevel.CLONE,
                ExperimentalUnitLevel.SPATIAL_REGION,
            },
        }
        if self.sampling.subject_unit not in required_subject_units[self.sampling.subject_kind]:
            raise ValueError("sampling subject kind is incompatible with its experimental unit")

        def is_same_or_ancestor(
            candidate: ExperimentalUnitLevel,
            level: ExperimentalUnitLevel,
        ) -> bool:
            current: ExperimentalUnitLevel | None = level
            while current is not None:
                if current is candidate:
                    return True
                current = by_level[current].parent_level
            return False

        shared_levels = (
            self.sampling.subject_unit,
            *(
                level
                for level in (self.randomization_unit, self.biological_replicate_unit)
                if level is not None
            ),
        )
        if any(not is_same_or_ancestor(self.default_split_unit, level) for level in shared_levels):
            raise ValueError("default split unit cannot be finer than a shared experimental unit")
        if self.biological_replicate_unit is not None and not is_same_or_ancestor(
            self.biological_replicate_unit,
            self.sampling.subject_unit,
        ):
            raise ValueError(
                "biological replicate unit cannot be finer than the sampling subject unit"
            )
        if (
            self.randomization_unit is not None
            and self.biological_replicate_unit is not None
            and not is_same_or_ancestor(self.biological_replicate_unit, self.randomization_unit)
        ):
            raise ValueError("biological replicate unit cannot be finer than randomization unit")
        endpoint_contrast = self.randomized_endpoint_contrast
        if endpoint_contrast is not None:
            if self.matched_control_field is not None:
                raise ValueError(
                    "structured endpoint controls cannot coexist with a legacy control field"
                )
            if self.randomization_unit is None:
                raise ValueError("randomized endpoint contrast requires a randomization unit")
            matching_levels = tuple(
                unit.level
                for unit in self.units
                if unit.resolved_identity == endpoint_contrast.matched_control.stratum_identity
            )
            if len(matching_levels) != 1:
                raise ValueError(
                    "matched-control stratum identity must exactly match one declared unit"
                )
            if not is_same_or_ancestor(matching_levels[0], self.randomization_unit):
                raise ValueError(
                    "matched-control stratum must be the randomization unit or its ancestor"
                )
        _require_unique_strings(self.batch_fields, name="batch fields")
        return self


class TemporalWindow(ManifestModel):
    """Closed experiment-relative interval; never coerced to a representative midpoint."""

    window_id: str = Field(min_length=1)
    earliest_seconds: float
    latest_seconds: float
    reference_event: str = Field(min_length=1)
    source_ids: tuple[str, ...] = Field(min_length=1)

    @property
    def fingerprint(self) -> str:
        payload = self.model_dump(mode="python")
        payload["source_ids"] = sorted(self.source_ids)
        return canonical_fingerprint(payload)

    @field_validator("earliest_seconds", "latest_seconds")
    @classmethod
    def bounds_are_finite(cls, value: float) -> float:
        return require_finite(value, name="temporal-window bound")

    @model_validator(mode="after")
    def interval_is_canonical(self) -> TemporalWindow:
        _require_canonical_id(self.window_id, name="temporal-window ID")
        _require_canonical_id(self.reference_event, name="temporal-window reference event")
        _require_unique_strings(self.source_ids, name="temporal-window source IDs")
        if tuple(sorted(self.source_ids)) != self.source_ids:
            raise ValueError("temporal-window source IDs must be sorted")
        if self.latest_seconds < self.earliest_seconds:
            raise ValueError("temporal-window bounds must be ordered")
        return self


class SubjectAlignment(StrEnum):
    SAME_CELL = "same_cell"
    SAME_SAMPLE = "same_sample"
    SAME_CLONE = "same_clone"
    SAME_POPULATION = "same_population"
    SAME_SPATIAL_REGION = "same_spatial_region"
    UNPAIRED = "unpaired"


class ModalitySpec(ManifestModel):
    modality: OntologyTerm
    source_ids: tuple[str, ...] = Field(min_length=1)
    subject_alignment: SubjectAlignment
    alignment_group: str | None = Field(default=None, min_length=1)
    alignment_key_field: str | None = Field(default=None, min_length=1)
    alignment_unit: ExperimentalUnitLevel | None = None
    alignment_identity: UnitIdentityExpression | None = None
    raw_available: bool = False
    processed_available: bool = False
    destructive: bool
    feature_identifier_namespace: str | None = None
    collection_time_window: TemporalWindow | None = None

    @model_validator(mode="after")
    def representation_and_alignment_are_explicit(self) -> ModalitySpec:
        if not self.raw_available and not self.processed_available:
            raise ValueError("a modality must expose raw or processed measurements")
        _require_unique_strings(self.source_ids, name="modality source IDs")
        alignment_selectors = (
            self.alignment_key_field,
            self.alignment_unit,
            self.alignment_identity,
        )
        if self.subject_alignment is SubjectAlignment.UNPAIRED:
            if self.alignment_group is not None or any(
                value is not None for value in alignment_selectors
            ):
                raise ValueError("unpaired modality must not declare an alignment key")
        elif (
            self.alignment_group is None
            or sum(value is not None for value in alignment_selectors) != 1
        ):
            raise ValueError(
                "paired modality requires an alignment group and exactly one unit reference, "
                "exact identity, or legacy key field"
            )
        if self.alignment_key_field is not None:
            _require_canonical_id(self.alignment_key_field, name="modality alignment-key field")
        return self


class AssignmentMechanism(StrEnum):
    NONE = "none"
    OBSERVATIONAL = "observational"
    ASSIGNED_NONRANDOM = "assigned_nonrandom"
    QUASI_EXPERIMENTAL = "quasi_experimental"
    RANDOMIZED = "randomized"


class RealizationEvidence(StrEnum):
    NONE = "none"
    ASSIGNMENT_ONLY = "assignment_only"
    INFERRED = "inferred"
    MEASURED = "measured"


class InterventionCapability(ManifestModel):
    source_ids: tuple[str, ...] = ()
    assignment: AssignmentMechanism = AssignmentMechanism.NONE
    kinds: tuple[OntologyTerm, ...] = ()
    targets_recorded: bool = False
    doses_recorded: bool = False
    durations_recorded: bool = False
    start_stop_recorded: bool = False
    washout_recorded: bool = False
    combinations_present: bool = False
    assignment_probabilities_recorded: bool = False
    matched_controls_present: bool = False
    realization_evidence: RealizationEvidence = RealizationEvidence.NONE

    @model_validator(mode="after")
    def intervention_metadata_matches_assignment(self) -> InterventionCapability:
        _require_unique_terms(self.kinds, name="intervention kinds")
        _require_unique_strings(self.source_ids, name="intervention source IDs")
        metadata_present = bool(self.kinds) or any(
            (
                self.targets_recorded,
                self.doses_recorded,
                self.durations_recorded,
                self.start_stop_recorded,
                self.washout_recorded,
                self.combinations_present,
                self.assignment_probabilities_recorded,
                self.matched_controls_present,
                self.realization_evidence is not RealizationEvidence.NONE,
            )
        )
        if self.assignment is AssignmentMechanism.NONE:
            if metadata_present or self.source_ids:
                raise ValueError("a dataset without interventions cannot declare intervention data")
        elif not self.kinds or not self.source_ids:
            raise ValueError("intervention support requires kinds and source artifacts")
        if self.assignment_probabilities_recorded and self.assignment not in {
            AssignmentMechanism.QUASI_EXPERIMENTAL,
            AssignmentMechanism.RANDOMIZED,
        }:
            raise ValueError("assignment probabilities require randomized or quasi evidence")
        return self


class TimingCapability(ManifestModel):
    source_ids: tuple[str, ...] = ()
    timepoints_seconds: tuple[float, ...] = ()
    observation_windows: tuple[TemporalWindow, ...] = ()
    observation_times_recorded: bool = False
    intervention_times_recorded: bool = False
    environment_times_recorded: bool = False
    event_ordering_recorded: bool = False

    @field_validator("timepoints_seconds")
    @classmethod
    def timepoints_are_finite_unique_and_sorted(
        cls, values: tuple[float, ...]
    ) -> tuple[float, ...]:
        for value in values:
            require_finite(value, name="dataset timepoint")
        if len(values) != len(set(values)):
            raise ValueError("dataset timepoints must be unique")
        if tuple(sorted(values)) != values:
            raise ValueError("dataset timepoints must be sorted")
        return values

    @model_validator(mode="after")
    def timing_assertions_have_sources(self) -> TimingCapability:
        _require_unique_strings(self.source_ids, name="timing source IDs")
        if tuple(sorted(self.source_ids)) != self.source_ids:
            raise ValueError("timing source IDs must be sorted")
        window_ids = tuple(window.window_id for window in self.observation_windows)
        _require_unique_strings(window_ids, name="observation-window IDs")
        window_order = tuple(
            (window.earliest_seconds, window.latest_seconds, window.window_id)
            for window in self.observation_windows
        )
        if tuple(sorted(window_order)) != window_order:
            raise ValueError("observation windows must be ordered by bounds and ID")
        window_sources = {
            source_id for window in self.observation_windows for source_id in window.source_ids
        }
        if not window_sources <= set(self.source_ids):
            raise ValueError("observation-window sources must be timing sources")
        assertions_present = bool(self.timepoints_seconds or self.observation_windows) or any(
            (
                self.observation_times_recorded,
                self.intervention_times_recorded,
                self.environment_times_recorded,
                self.event_ordering_recorded,
            )
        )
        if assertions_present and not self.source_ids:
            raise ValueError("timing assertions require a source artifact")
        if self.source_ids and not assertions_present:
            raise ValueError("timing source artifacts require a timing assertion")
        if (self.timepoints_seconds or self.observation_windows) and (
            not self.observation_times_recorded
        ):
            raise ValueError("declared time support requires recorded observation times")
        return self


class LineageResolution(StrEnum):
    NONE = "none"
    CLONE = "clone"
    SIBLING = "sibling"
    PARENT_CHILD = "parent_child"
    FULL_TREE = "full_tree"


class LineageCapability(ManifestModel):
    resolution: LineageResolution = LineageResolution.NONE
    source_ids: tuple[str, ...] = ()
    lineage_ids_recorded: bool = False
    division_times_recorded: bool = False
    confidence_recorded: bool = False

    @model_validator(mode="after")
    def lineage_metadata_requires_lineage(self) -> LineageCapability:
        _require_unique_strings(self.source_ids, name="lineage source IDs")
        metadata_present = (
            bool(self.source_ids)
            or self.lineage_ids_recorded
            or self.division_times_recorded
            or self.confidence_recorded
        )
        if self.resolution is LineageResolution.NONE and metadata_present:
            raise ValueError("lineage metadata requires a nonempty lineage resolution")
        if self.resolution is not LineageResolution.NONE and (
            not self.source_ids or not self.lineage_ids_recorded
        ):
            raise ValueError("lineage support requires source artifacts and lineage IDs")
        return self


class SpatialResolution(StrEnum):
    NONE = "none"
    SAMPLE_REGION = "sample_region"
    CELL_COORDINATES = "cell_coordinates"
    NEIGHBOR_GRAPH = "neighbor_graph"
    IMAGE_VOLUME = "image_volume"


class SpatialCapability(ManifestModel):
    resolution: SpatialResolution = SpatialResolution.NONE
    source_ids: tuple[str, ...] = ()
    coordinate_dimensions: Literal[2, 3] | None = None
    distances_recorded: bool = False
    contacts_recorded: bool = False
    regions_recorded: bool = False
    subject_alignment: SubjectAlignment | None = None
    alignment_group: str | None = None
    alignment_key_field: str | None = None

    @model_validator(mode="after")
    def spatial_metadata_requires_spatial_support(self) -> SpatialCapability:
        _require_unique_strings(self.source_ids, name="spatial source IDs")
        metadata_present = (
            bool(self.source_ids)
            or self.coordinate_dimensions is not None
            or self.distances_recorded
            or self.contacts_recorded
            or self.regions_recorded
            or self.subject_alignment is not None
            or self.alignment_group is not None
            or self.alignment_key_field is not None
        )
        if self.resolution is SpatialResolution.NONE and metadata_present:
            raise ValueError("spatial metadata requires a nonempty spatial resolution")
        if self.resolution is not SpatialResolution.NONE and not self.source_ids:
            raise ValueError("spatial support requires a source artifact")
        if (
            self.resolution
            in {
                SpatialResolution.CELL_COORDINATES,
                SpatialResolution.NEIGHBOR_GRAPH,
                SpatialResolution.IMAGE_VOLUME,
            }
            and self.coordinate_dimensions is None
        ):
            raise ValueError("coordinate-level spatial support requires dimensionality")
        alignment_values = (
            self.subject_alignment,
            self.alignment_group,
            self.alignment_key_field,
        )
        if any(value is not None for value in alignment_values) and not all(
            value is not None for value in alignment_values
        ):
            raise ValueError(
                "spatial alignment requires a subject, alignment group, and key field together"
            )
        if self.subject_alignment is SubjectAlignment.UNPAIRED:
            raise ValueError("spatial evidence cannot use unpaired subject alignment")
        return self


class ReadoutStatus(StrEnum):
    DIRECT = "direct"
    DERIVED = "derived"


class FunctionalReadoutDerivation(ManifestModel):
    """Content-addressed derivation for a functional value retained in a source artifact."""

    method_id: str = Field(min_length=1)
    method_version: str = Field(min_length=1)
    method_sha256: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    source_value_fields: tuple[str, ...] = Field(min_length=1)
    formula: str = Field(min_length=1)

    @field_validator("method_sha256")
    @classmethod
    def canonicalize_sha256(cls, value: str) -> str:
        return value.casefold()

    @model_validator(mode="after")
    def derivation_is_exact(self) -> FunctionalReadoutDerivation:
        _require_canonical_id(self.method_id, name="functional derivation method ID")
        _require_canonical_id(self.method_version, name="functional derivation method version")
        _require_unique_strings(
            self.source_value_fields,
            name="functional derivation source-value fields",
        )
        if tuple(sorted(self.source_value_fields)) != self.source_value_fields:
            raise ValueError("functional derivation source-value fields must be sorted")
        if self.formula != self.formula.strip():
            raise ValueError("functional derivation formula must be trimmed")
        return self


class FunctionalReadout(ManifestModel):
    readout_id: str = Field(min_length=1)
    output: OntologyTerm
    source_ids: tuple[str, ...] = Field(min_length=1)
    value_field: str = Field(min_length=1)
    units: str = Field(min_length=1)
    aggregation_level: ExperimentalUnitLevel
    subject_alignment: SubjectAlignment
    alignment_group: str = Field(min_length=1)
    alignment_key_field: str = Field(min_length=1)
    status: ReadoutStatus
    derivation: FunctionalReadoutDerivation | None = None
    measurement_time_seconds: float | None = None
    measurement_time_field: str | None = None
    measurement_time_window: TemporalWindow | None = None

    @field_validator("measurement_time_seconds")
    @classmethod
    def fixed_measurement_time_is_finite(cls, value: float | None) -> float | None:
        if value is not None:
            return require_finite(value, name="functional measurement time")
        return value

    @field_validator("measurement_time_field")
    @classmethod
    def measurement_time_field_is_canonical(cls, value: str | None) -> str | None:
        if value is not None:
            return _require_canonical_id(value, name="functional measurement-time field")
        return value

    @model_validator(mode="after")
    def measurement_time_is_explicit(self) -> FunctionalReadout:
        _require_canonical_id(self.readout_id, name="functional readout ID")
        _require_canonical_id(self.value_field, name="functional value field")
        _require_unique_strings(self.source_ids, name="functional source IDs")
        time_forms = (
            self.measurement_time_seconds,
            self.measurement_time_field,
            self.measurement_time_window,
        )
        if sum(value is not None for value in time_forms) != 1:
            raise ValueError(
                "functional readout requires exactly one fixed, field, or window measurement time"
            )
        if self.measurement_time_window is not None and not set(
            self.measurement_time_window.source_ids
        ) <= set(self.source_ids):
            raise ValueError("functional measurement-window sources must be readout sources")
        if self.subject_alignment is SubjectAlignment.UNPAIRED:
            raise ValueError("functional readout requires an explicit subject alignment")
        if self.status is ReadoutStatus.DERIVED and self.derivation is None:
            raise ValueError("derived functional readouts require an exact derivation")
        if self.status is ReadoutStatus.DIRECT and self.derivation is not None:
            raise ValueError("direct functional readouts must not declare a derivation")
        return self


class FunctionalCapability(ManifestModel):
    outputs: tuple[FunctionalReadout, ...] = ()

    @field_validator("outputs")
    @classmethod
    def outputs_are_unique(
        cls, outputs: tuple[FunctionalReadout, ...]
    ) -> tuple[FunctionalReadout, ...]:
        readout_ids = tuple(output.readout_id for output in outputs)
        _require_unique_strings(readout_ids, name="functional readout IDs")
        return outputs


class EnvironmentVariable(ManifestModel):
    variable: OntologyTerm
    source_ids: tuple[str, ...] = Field(min_length=1)
    units: str | None = None
    measured: bool
    assigned: bool
    time_resolved: bool = False

    @model_validator(mode="after")
    def variable_has_evidence_mode(self) -> EnvironmentVariable:
        _require_unique_strings(self.source_ids, name="environment source IDs")
        if not self.measured and not self.assigned:
            raise ValueError("an environment variable must be measured or assigned")
        return self


class EnvironmentCapability(ManifestModel):
    variables: tuple[EnvironmentVariable, ...] = ()

    @field_validator("variables")
    @classmethod
    def variables_are_unique(
        cls, variables: tuple[EnvironmentVariable, ...]
    ) -> tuple[EnvironmentVariable, ...]:
        _require_unique_terms(
            tuple(variable.variable for variable in variables),
            name="environment variables",
        )
        return variables


class DatasetCapabilities(ManifestModel):
    modalities: tuple[ModalitySpec, ...] = Field(min_length=1)
    interventions: InterventionCapability = Field(default_factory=InterventionCapability)
    timing: TimingCapability = Field(default_factory=TimingCapability)
    lineage: LineageCapability = Field(default_factory=LineageCapability)
    spatial: SpatialCapability = Field(default_factory=SpatialCapability)
    functional: FunctionalCapability = Field(default_factory=FunctionalCapability)
    environment: EnvironmentCapability = Field(default_factory=EnvironmentCapability)

    @field_validator("modalities")
    @classmethod
    def modalities_are_unique(
        cls, modalities: tuple[ModalitySpec, ...]
    ) -> tuple[ModalitySpec, ...]:
        _require_unique_terms(
            tuple(item.modality for item in modalities), name="dataset modalities"
        )
        return modalities


class ScientificClaim(StrEnum):
    ASSAY_MEASUREMENT_MODEL = "assay_measurement_model"
    SNAPSHOT_STATE_PRIOR = "snapshot_state_prior"
    SAME_CELL_MULTIMODAL_FUSION = "same_cell_multimodal_fusion"
    SAMPLE_LEVEL_MULTIMODAL_FUSION = "sample_level_multimodal_fusion"
    POPULATION_DYNAMICS = "population_dynamics"
    INDIVIDUAL_LONGITUDINAL_DYNAMICS = "individual_longitudinal_dynamics"
    INTERVENTION_EFFECT = "intervention_effect"
    COUNTERFACTUAL_GENERALIZATION = "counterfactual_generalization"
    LINEAGE_FATE = "lineage_fate"
    SPATIAL_CONTEXT = "spatial_context"
    FUNCTIONAL_OUTCOME = "functional_outcome"
    RETROSPECTIVE_INTERVENTION_SELECTION = "retrospective_intervention_selection"


class EligibilityStatus(StrEnum):
    ELIGIBLE = "eligible"
    CONDITIONALLY_ELIGIBLE = "conditionally_eligible"
    INELIGIBLE = "ineligible"
    NOT_ASSESSED = "not_assessed"


class IdentificationBasis(StrEnum):
    NONE = "none"
    DESCRIPTIVE = "descriptive"
    ASSOCIATIONAL = "associational"
    RANDOMIZED_WITHIN_STUDY = "randomized_within_study"
    QUASI_EXPERIMENTAL = "quasi_experimental"
    TRANSPORTED_UNDER_ASSUMPTIONS = "transported_under_assumptions"


class MetricFamily(StrEnum):
    """Broad metric semantics; concrete benchmark metrics remain separately versioned."""

    PREDICTIVE_PROPER_SCORE = "predictive_proper_score"
    CALIBRATION = "calibration"
    POPULATION_DISTRIBUTION = "population_distribution"
    INTERVENTION_EFFECT = "intervention_effect"
    INTERVENTION_RANKING = "intervention_ranking"
    EVENT_OR_SURVIVAL = "event_or_survival"
    LINEAGE = "lineage"
    OOD_OR_SELECTIVE_RISK = "ood_or_selective_risk"
    PREDICTIVE_SUFFICIENCY = "predictive_sufficiency"
    DECISION_UTILITY = "decision_utility"


class MetricPartitionPurpose(StrEnum):
    """Prospective partition role; membership belongs to a later split manifest."""

    CALIBRATION = "calibration"
    MODEL_SELECTION_VALIDATION = "model_selection_validation"
    UNTOUCHED_TEST = "untouched_test"
    EXTERNAL_VALIDATION = "external_validation"


class AssessmentKind(StrEnum):
    CLAIM = "claim"
    LOSS = "loss"
    METRIC = "metric"


class WorkflowEligibilityReason(StrEnum):
    SCIENTIFICALLY_ELIGIBLE = "scientifically_eligible"
    SCIENTIFIC_ASSUMPTIONS_PENDING = "scientific_assumptions_pending"
    SCIENTIFIC_BLOCKERS = "scientific_blockers"
    SCIENTIFIC_NOT_ASSESSED = "scientific_not_assessed"
    USE_PERMITTED = "use_permitted"
    USE_CONDITIONAL = "use_conditional"
    USE_PROHIBITED = "use_prohibited"
    USE_UNKNOWN = "use_unknown"
    NO_EXACT_DATA_SOURCES = "no_exact_data_sources"


class AssessmentScope(ManifestModel):
    subject_kind: SamplingSubjectKind
    system_boundary: SystemBoundary
    biological_systems: tuple[OntologyTerm, ...] = Field(min_length=1)
    modalities: tuple[OntologyTerm, ...] = ()
    intervention_kinds: tuple[OntologyTerm, ...] = ()
    functional_readout_ids: tuple[str, ...] = ()
    environment_variables: tuple[OntologyTerm, ...] = ()
    horizons_seconds: tuple[float, ...] = ()
    horizon_windows: tuple[TemporalWindow, ...] = ()
    inference_cutoff_seconds: float | None = None
    inference_cutoff_field: str | None = None
    inference_cutoff_window: TemporalWindow | None = None

    @property
    def fingerprint(self) -> str:
        """Order-independent identity for the scientific scope rather than display labels."""

        return canonical_fingerprint(
            {
                "subject_kind": self.subject_kind,
                "system_boundary": self.system_boundary,
                "biological_systems": sorted(_term_keys(self.biological_systems)),
                "modalities": sorted(_term_keys(self.modalities)),
                "intervention_kinds": sorted(_term_keys(self.intervention_kinds)),
                "functional_readout_ids": sorted(self.functional_readout_ids),
                "environment_variables": sorted(_term_keys(self.environment_variables)),
                "horizons_seconds": self.horizons_seconds,
                "horizon_windows": [
                    window.model_dump(mode="python") for window in self.horizon_windows
                ],
                "inference_cutoff_seconds": self.inference_cutoff_seconds,
                "inference_cutoff_field": self.inference_cutoff_field,
                "inference_cutoff_window": (
                    self.inference_cutoff_window.model_dump(mode="python")
                    if self.inference_cutoff_window is not None
                    else None
                ),
            }
        )

    @field_validator("horizons_seconds")
    @classmethod
    def horizons_are_positive_finite_unique_sorted(
        cls, values: tuple[float, ...]
    ) -> tuple[float, ...]:
        for value in values:
            require_finite(value, name="claim horizon")
            if value <= 0:
                raise ValueError("claim horizons must be positive")
        if len(values) != len(set(values)) or tuple(sorted(values)) != values:
            raise ValueError("claim horizons must be unique and sorted")
        return values

    @field_validator("inference_cutoff_seconds")
    @classmethod
    def cutoff_is_finite(cls, value: float | None) -> float | None:
        if value is not None:
            return require_finite(value, name="claim inference cutoff")
        return value

    @field_validator("inference_cutoff_field")
    @classmethod
    def cutoff_field_is_canonical(cls, value: str | None) -> str | None:
        if value is not None:
            return _require_canonical_id(value, name="assessment inference-cutoff field")
        return value

    @model_validator(mode="after")
    def scope_members_and_cutoff_are_coherent(self) -> AssessmentScope:
        compatible_boundaries = {
            SamplingSubjectKind.INDIVIDUAL_CELL: {
                SystemBoundary.ISOLATED_CELL,
                SystemBoundary.CELL_AND_SOLUBLE_ENVIRONMENT,
                SystemBoundary.CELL_AND_NEIGHBORS,
                SystemBoundary.SPATIAL_TISSUE_NICHE,
            },
            SamplingSubjectKind.CLONE: {SystemBoundary.CLONE},
            SamplingSubjectKind.SAMPLE: {
                SystemBoundary.POPULATION,
                SystemBoundary.SPATIAL_TISSUE_NICHE,
            },
            SamplingSubjectKind.POPULATION: {
                SystemBoundary.POPULATION,
                SystemBoundary.SPATIAL_TISSUE_NICHE,
            },
            SamplingSubjectKind.SPATIAL_REGION: {SystemBoundary.SPATIAL_TISSUE_NICHE},
        }
        if self.system_boundary not in compatible_boundaries[self.subject_kind]:
            raise ValueError("claim system boundary is incompatible with its sampling subject")
        for name, terms in (
            ("claim biological systems", self.biological_systems),
            ("claim modalities", self.modalities),
            ("claim intervention kinds", self.intervention_kinds),
            ("claim environment variables", self.environment_variables),
        ):
            _require_unique_terms(terms, name=name)
        _require_unique_strings(
            self.functional_readout_ids,
            name="claim functional readout IDs",
        )
        for readout_id in self.functional_readout_ids:
            _require_canonical_id(readout_id, name="claim functional readout ID")
        cutoff_values = (
            self.inference_cutoff_seconds,
            self.inference_cutoff_field,
            self.inference_cutoff_window,
        )
        if self.horizons_seconds and self.horizon_windows:
            raise ValueError("a claim cannot mix point horizons and interval target windows")
        if self.horizons_seconds and sum(value is not None for value in cutoff_values) != 1:
            raise ValueError("claims with horizons require exactly one inference cutoff")
        if self.horizons_seconds and self.inference_cutoff_window is not None:
            raise ValueError("point horizons cannot use an interval inference cutoff")
        if self.horizon_windows and (
            self.inference_cutoff_window is None
            or self.inference_cutoff_seconds is not None
            or self.inference_cutoff_field is not None
        ):
            raise ValueError("interval target windows require one interval inference cutoff")
        if (
            not self.horizons_seconds
            and not self.horizon_windows
            and any(value is not None for value in cutoff_values)
        ):
            raise ValueError("a claim without horizons must not declare an inference cutoff")
        window_ids = tuple(window.window_id for window in self.horizon_windows)
        _require_unique_strings(window_ids, name="claim horizon-window IDs")
        window_order = tuple(
            (window.earliest_seconds, window.latest_seconds, window.window_id)
            for window in self.horizon_windows
        )
        if tuple(sorted(window_order)) != window_order:
            raise ValueError("claim horizon windows must be ordered by bounds and ID")
        if self.inference_cutoff_window is not None:
            cutoff = self.inference_cutoff_window
            for window in self.horizon_windows:
                if window.reference_event != cutoff.reference_event:
                    raise ValueError(
                        "interval cutoff and target windows must use one reference event"
                    )
                if window.earliest_seconds <= cutoff.latest_seconds:
                    raise ValueError("interval target windows must occur after the cutoff window")
        return self


class ClaimAssessment(ManifestModel):
    assessment_id: str = Field(min_length=1)
    claim: ScientificClaim
    status: EligibilityStatus
    identification_basis: IdentificationBasis
    scope: AssessmentScope
    evidence_source_ids: tuple[str, ...] = ()
    execution_source_ids: tuple[str, ...] | None = Field(default=None, min_length=1)
    evidence_notes: tuple[str, ...] = ()
    assumptions: tuple[str, ...] = ()
    blockers: tuple[str, ...] = ()

    @field_validator("assessment_id")
    @classmethod
    def assessment_id_is_canonical(cls, value: str) -> str:
        return _require_canonical_id(value, name="claim assessment ID")

    @property
    def fingerprint(self) -> str:
        payload: dict[str, object] = {
            "assessment_id": self.assessment_id,
            "claim": self.claim,
            "status": self.status,
            "identification_basis": self.identification_basis,
            "scope_fingerprint": self.scope.fingerprint,
            "evidence_source_ids": sorted(self.evidence_source_ids),
            "evidence_notes": sorted(self.evidence_notes),
            "assumptions": sorted(self.assumptions),
            "blockers": sorted(self.blockers),
        }
        # Omission preserves the 0.3 legacy meaning: the full evidence set is also the permission
        # scope. New manifests should bind execution bytes explicitly.
        if self.execution_source_ids is not None:
            payload["execution_source_ids"] = sorted(self.execution_source_ids)
        return canonical_fingerprint(payload)

    @model_validator(mode="after")
    def status_has_honest_evidence(self) -> ClaimAssessment:
        _require_unique_strings(self.evidence_source_ids, name="claim evidence source IDs")
        if self.execution_source_ids is not None:
            _require_unique_strings(
                self.execution_source_ids,
                name="claim execution source IDs",
            )
        _require_unique_strings(self.evidence_notes, name="claim evidence notes")
        _require_unique_strings(self.assumptions, name="claim assumptions")
        _require_unique_strings(self.blockers, name="claim blockers")
        supported = self.status in {
            EligibilityStatus.ELIGIBLE,
            EligibilityStatus.CONDITIONALLY_ELIGIBLE,
        }
        if self.status is EligibilityStatus.ELIGIBLE:
            if not self.evidence_source_ids or not self.evidence_notes:
                raise ValueError("eligible claims require source-backed evidence")
            if self.blockers:
                raise ValueError("eligible claims cannot retain blockers")
        elif self.status is EligibilityStatus.CONDITIONALLY_ELIGIBLE:
            if not self.evidence_source_ids or not self.evidence_notes or not self.assumptions:
                raise ValueError("conditional claims require evidence and assumptions")
        elif not self.blockers:
            raise ValueError("ineligible or unassessed claims require an explicit blocker")
        if supported and self.identification_basis is IdentificationBasis.NONE:
            raise ValueError("supported claim eligibility requires an identification basis")
        if (
            self.identification_basis
            in {
                IdentificationBasis.QUASI_EXPERIMENTAL,
                IdentificationBasis.TRANSPORTED_UNDER_ASSUMPTIONS,
            }
            and self.status is not EligibilityStatus.CONDITIONALLY_ELIGIBLE
        ):
            raise ValueError("quasi-experimental or transported claims must remain conditional")
        if (
            self.claim is ScientificClaim.COUNTERFACTUAL_GENERALIZATION
            and supported
            and self.status is not EligibilityStatus.CONDITIONALLY_ELIGIBLE
        ):
            raise ValueError("counterfactual generalization must remain conditional")
        causal_claims = {
            ScientificClaim.INTERVENTION_EFFECT,
            ScientificClaim.COUNTERFACTUAL_GENERALIZATION,
            ScientificClaim.RETROSPECTIVE_INTERVENTION_SELECTION,
        }
        if (
            self.claim in causal_claims
            and supported
            and self.identification_basis
            not in {
                IdentificationBasis.RANDOMIZED_WITHIN_STUDY,
                IdentificationBasis.QUASI_EXPERIMENTAL,
                IdentificationBasis.TRANSPORTED_UNDER_ASSUMPTIONS,
            }
        ):
            raise ValueError("supported intervention claims require causal evidence")
        return self


class ClaimAssessmentReference(ManifestModel):
    assessment_id: str = Field(min_length=1)
    assessment_fingerprint: str = Field(pattern=r"^[a-fA-F0-9]{64}$")

    @field_validator("assessment_id")
    @classmethod
    def assessment_id_is_canonical(cls, value: str) -> str:
        return _require_canonical_id(value, name="claim assessment reference ID")

    @field_validator("assessment_fingerprint")
    @classmethod
    def canonicalize_fingerprint(cls, value: str) -> str:
        return value.casefold()


class ObjectiveEligibilityAssessment(ManifestModel):
    """Shared evidence and exact-claim binding for empirical loss and metric roles."""

    assessment_id: str = Field(min_length=1)
    status: EligibilityStatus
    scope: AssessmentScope
    required_split_unit: ExperimentalUnitLevel
    data_source_ids: tuple[str, ...] = ()
    supporting_claim_assessments: tuple[ClaimAssessmentReference, ...] = ()
    evidence_notes: tuple[str, ...] = ()
    assumptions: tuple[str, ...] = ()
    blockers: tuple[str, ...] = ()

    @field_validator("assessment_id")
    @classmethod
    def assessment_id_is_canonical(cls, value: str) -> str:
        return _require_canonical_id(value, name="objective assessment ID")

    @property
    def fingerprint(self) -> str:
        payload = self.model_dump(mode="python", exclude={"scope"})
        payload["scope_fingerprint"] = self.scope.fingerprint
        payload["data_source_ids"] = sorted(self.data_source_ids)
        payload["supporting_claim_assessments"] = sorted(
            (
                {
                    "assessment_id": reference.assessment_id,
                    "assessment_fingerprint": reference.assessment_fingerprint,
                }
                for reference in self.supporting_claim_assessments
            ),
            key=lambda reference: (
                reference["assessment_id"],
                reference["assessment_fingerprint"],
            ),
        )
        payload["evidence_notes"] = sorted(self.evidence_notes)
        payload["assumptions"] = sorted(self.assumptions)
        payload["blockers"] = sorted(self.blockers)
        return canonical_fingerprint(payload)

    @model_validator(mode="after")
    def status_has_honest_evidence(self) -> ObjectiveEligibilityAssessment:
        _require_unique_strings(self.data_source_ids, name="objective data source IDs")
        _require_unique_strings(self.evidence_notes, name="objective evidence notes")
        _require_unique_strings(self.assumptions, name="objective assumptions")
        _require_unique_strings(self.blockers, name="objective blockers")
        reference_ids = tuple(item.assessment_id for item in self.supporting_claim_assessments)
        _require_unique_strings(reference_ids, name="supporting claim assessment IDs")
        supported = self.status in {
            EligibilityStatus.ELIGIBLE,
            EligibilityStatus.CONDITIONALLY_ELIGIBLE,
        }
        if supported and (
            not self.data_source_ids
            or not self.supporting_claim_assessments
            or not self.evidence_notes
        ):
            raise ValueError(
                "supported objective eligibility requires data, exact claims, and evidence notes"
            )
        if self.status is EligibilityStatus.ELIGIBLE and self.blockers:
            raise ValueError("eligible objective assessments cannot retain blockers")
        if self.status is EligibilityStatus.CONDITIONALLY_ELIGIBLE and not self.assumptions:
            raise ValueError("conditional objective assessments require assumptions")
        if self.status in {EligibilityStatus.INELIGIBLE, EligibilityStatus.NOT_ASSESSED} and (
            not self.blockers
        ):
            raise ValueError("ineligible or unassessed objectives require an explicit blocker")
        return self


class LossEligibilityAssessment(ObjectiveEligibilityAssessment):
    loss_kind: LossKind


class MetricEligibilityAssessment(ObjectiveEligibilityAssessment):
    metric_id: str = Field(min_length=1, pattern=r"^[a-z][a-z0-9_.-]*$")
    metric_family: MetricFamily
    partition_purpose: MetricPartitionPurpose

    @field_validator("metric_id")
    @classmethod
    def metric_id_is_canonical(cls, value: str) -> str:
        return _require_canonical_id(value, name="metric ID")


class DatasetAssessmentReference(ManifestModel):
    dataset_manifest_fingerprint: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    assessment_id: str = Field(min_length=1)
    assessment_fingerprint: str = Field(pattern=r"^[a-fA-F0-9]{64}$")

    @field_validator("assessment_id")
    @classmethod
    def assessment_id_is_canonical(cls, value: str) -> str:
        return _require_canonical_id(value, name="dataset assessment reference ID")

    @field_validator("dataset_manifest_fingerprint", "assessment_fingerprint")
    @classmethod
    def canonicalize_fingerprints(cls, value: str) -> str:
        return value.casefold()


class DatasetAssessmentResolution(ManifestModel):
    reference: DatasetAssessmentReference
    assessment_kind: AssessmentKind
    scientific_status: EligibilityStatus
    data_source_ids: tuple[str, ...]
    effective_permission: EffectiveDataUsePermission | None
    workflow_status: EligibilityStatus
    workflow_reasons: tuple[WorkflowEligibilityReason, ...] = Field(min_length=1)
    scientific_assumptions: tuple[str, ...] = ()
    scientific_blockers: tuple[str, ...] = ()
    legal_conditions: tuple[str, ...] = ()
    applicable_policy_ids: tuple[str, ...] = ()
    use_allowed_without_additional_review: bool

    @model_validator(mode="after")
    def workflow_result_is_derived_not_asserted(self) -> DatasetAssessmentResolution:
        _require_unique_strings(
            self.data_source_ids,
            name="resolution data source IDs",
        )
        if tuple(sorted(self.data_source_ids)) != self.data_source_ids:
            raise ValueError("resolution data source IDs must be sorted")
        if (self.effective_permission is None) is not (not self.data_source_ids):
            raise ValueError("effective permission must be absent exactly when sources are absent")
        if (
            self.effective_permission is not None
            and self.data_source_ids != self.effective_permission.source_ids
        ):
            raise ValueError("resolution sources must match effective-permission sources exactly")
        expected_status = _workflow_status(
            self.scientific_status,
            self.effective_permission,
        )
        if self.workflow_status is not expected_status:
            raise ValueError("workflow status must be derived from science and permission")
        if self.workflow_reasons != _workflow_reasons(
            self.scientific_status,
            self.effective_permission,
        ):
            raise ValueError("workflow reasons must be derived from science and permission")
        expected_allowed = (
            self.scientific_status is EligibilityStatus.ELIGIBLE
            and self.effective_permission is not None
            and self.effective_permission.status is PermissionStatus.PERMITTED
        )
        if self.use_allowed_without_additional_review is not expected_allowed:
            raise ValueError("workflow allow flag must be derived from science and permission")
        expected_conditions = (
            self.effective_permission.conditions if self.effective_permission is not None else ()
        )
        expected_policy_ids = (
            self.effective_permission.applicable_policy_ids
            if self.effective_permission is not None
            else ()
        )
        if self.legal_conditions != expected_conditions:
            raise ValueError("resolution legal conditions must match effective permission")
        if self.applicable_policy_ids != expected_policy_ids:
            raise ValueError("resolution policy IDs must match effective permission")
        _require_unique_strings(
            self.scientific_assumptions,
            name="resolution scientific assumptions",
        )
        _require_unique_strings(
            self.scientific_blockers,
            name="resolution scientific blockers",
        )
        if (
            self.scientific_status is EligibilityStatus.CONDITIONALLY_ELIGIBLE
            and not self.scientific_assumptions
        ):
            raise ValueError("conditional scientific resolution requires assumptions")
        if (
            self.scientific_status
            in {
                EligibilityStatus.INELIGIBLE,
                EligibilityStatus.NOT_ASSESSED,
            }
            and not self.scientific_blockers
        ):
            raise ValueError("unsupported scientific resolution requires blockers")
        if self.scientific_status is EligibilityStatus.ELIGIBLE and self.scientific_blockers:
            raise ValueError("eligible scientific resolution cannot retain blockers")
        return self


def _workflow_status(
    scientific_status: EligibilityStatus,
    permission: EffectiveDataUsePermission | None,
) -> EligibilityStatus:
    if scientific_status is EligibilityStatus.INELIGIBLE:
        return EligibilityStatus.INELIGIBLE
    if permission is not None and permission.status is PermissionStatus.PROHIBITED:
        return EligibilityStatus.INELIGIBLE
    if scientific_status is EligibilityStatus.NOT_ASSESSED:
        return EligibilityStatus.NOT_ASSESSED
    if permission is None or permission.status is PermissionStatus.UNKNOWN:
        return EligibilityStatus.NOT_ASSESSED
    if (
        scientific_status is EligibilityStatus.CONDITIONALLY_ELIGIBLE
        or permission.status is PermissionStatus.CONDITIONAL
    ):
        return EligibilityStatus.CONDITIONALLY_ELIGIBLE
    return EligibilityStatus.ELIGIBLE


def _workflow_reasons(
    scientific_status: EligibilityStatus,
    permission: EffectiveDataUsePermission | None,
) -> tuple[WorkflowEligibilityReason, ...]:
    scientific_reason = {
        EligibilityStatus.ELIGIBLE: WorkflowEligibilityReason.SCIENTIFICALLY_ELIGIBLE,
        EligibilityStatus.CONDITIONALLY_ELIGIBLE: (
            WorkflowEligibilityReason.SCIENTIFIC_ASSUMPTIONS_PENDING
        ),
        EligibilityStatus.INELIGIBLE: WorkflowEligibilityReason.SCIENTIFIC_BLOCKERS,
        EligibilityStatus.NOT_ASSESSED: WorkflowEligibilityReason.SCIENTIFIC_NOT_ASSESSED,
    }[scientific_status]
    if permission is None:
        permission_reason = WorkflowEligibilityReason.NO_EXACT_DATA_SOURCES
    else:
        permission_reason = {
            PermissionStatus.PERMITTED: WorkflowEligibilityReason.USE_PERMITTED,
            PermissionStatus.CONDITIONAL: WorkflowEligibilityReason.USE_CONDITIONAL,
            PermissionStatus.PROHIBITED: WorkflowEligibilityReason.USE_PROHIBITED,
            PermissionStatus.UNKNOWN: WorkflowEligibilityReason.USE_UNKNOWN,
        }[permission.status]
    return (scientific_reason, permission_reason)


_PREDICTIVE_CLAIMS = frozenset(
    {
        ScientificClaim.POPULATION_DYNAMICS,
        ScientificClaim.INDIVIDUAL_LONGITUDINAL_DYNAMICS,
        ScientificClaim.INTERVENTION_EFFECT,
        ScientificClaim.LINEAGE_FATE,
        ScientificClaim.FUNCTIONAL_OUTCOME,
        ScientificClaim.RETROSPECTIVE_INTERVENTION_SELECTION,
    }
)

_LOSS_COMPATIBLE_CLAIMS: dict[LossKind, frozenset[ScientificClaim]] = {
    LossKind.MULTI_HORIZON_FUTURE: frozenset(
        {
            ScientificClaim.POPULATION_DYNAMICS,
            ScientificClaim.INDIVIDUAL_LONGITUDINAL_DYNAMICS,
        }
    ),
    LossKind.FUNCTIONAL_OUTCOME: frozenset({ScientificClaim.FUNCTIONAL_OUTCOME}),
    LossKind.INTERVENTION_EFFECT: frozenset({ScientificClaim.INTERVENTION_EFFECT}),
    LossKind.LINEAGE_TRANSITION: frozenset({ScientificClaim.LINEAGE_FATE}),
    LossKind.HELD_OUT_MODALITY: frozenset(
        {
            ScientificClaim.SAME_CELL_MULTIMODAL_FUSION,
            ScientificClaim.SAMPLE_LEVEL_MULTIMODAL_FUSION,
        }
    ),
    # These are model-level regularizers until a future manifest contract represents their
    # empirical evidence. A dataset must not claim support merely because a trainer can compute it.
    LossKind.MECHANISTIC_CONSISTENCY: frozenset(),
    LossKind.UNCERTAINTY_CALIBRATION: _PREDICTIVE_CLAIMS,
    LossKind.STATE_COMPLEXITY: frozenset(),
}

_METRIC_COMPATIBLE_CLAIMS: dict[MetricFamily, frozenset[ScientificClaim]] = {
    MetricFamily.PREDICTIVE_PROPER_SCORE: _PREDICTIVE_CLAIMS,
    MetricFamily.CALIBRATION: _PREDICTIVE_CLAIMS,
    MetricFamily.POPULATION_DISTRIBUTION: frozenset(
        {ScientificClaim.POPULATION_DYNAMICS, ScientificClaim.INTERVENTION_EFFECT}
    ),
    MetricFamily.INTERVENTION_EFFECT: frozenset({ScientificClaim.INTERVENTION_EFFECT}),
    # These families need target semantics not yet present in the 0.3 manifest (an exact
    # candidate/utility set, event-time/censoring, a held-out domain, or a paired-history task).
    # They remain explicitly fail-closed until the benchmark contract introduced in roadmap
    # item 8 can bind those requirements.
    MetricFamily.INTERVENTION_RANKING: frozenset(),
    MetricFamily.EVENT_OR_SURVIVAL: frozenset(),
    MetricFamily.LINEAGE: frozenset({ScientificClaim.LINEAGE_FATE}),
    MetricFamily.OOD_OR_SELECTIVE_RISK: frozenset(),
    MetricFamily.PREDICTIVE_SUFFICIENCY: frozenset(),
    MetricFamily.DECISION_UTILITY: frozenset(),
}


class DatasetManifest(ManifestModel):
    model_config = ConfigDict(
        json_schema_extra={
            "$comment": (
                "Experimental structural schema. Cross-field scientific eligibility requires "
                "validation with the cellstate.data.DatasetManifest Python model."
            )
        }
    )

    schema_version: DatasetManifestSchemaVersion = DATASET_MANIFEST_SCHEMA_VERSION
    dataset_id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    title: str = Field(min_length=1)
    description: str = Field(min_length=1)
    origin: PublicRealDataOrigin
    sources: tuple[SourceArtifact, ...] = Field(min_length=1)
    slice_spec: DatasetSliceSpec
    use_policies: tuple[DataUsePolicy, ...] = Field(min_length=1)
    experimental_design: ExperimentalDesign
    capabilities: DatasetCapabilities
    claim_assessments: tuple[ClaimAssessment, ...] = Field(min_length=1)
    loss_assessments: tuple[LossEligibilityAssessment, ...] = ()
    metric_assessments: tuple[MetricEligibilityAssessment, ...] = ()
    notes: tuple[str, ...] = ()

    @property
    def canonical_payload(self) -> dict[str, Any]:
        """Return the sole normalized payload used for manifest storage and identity."""

        payload = self.model_dump(mode="json")
        # New 0.3-compatible fields deliberately preserve prior fingerprints when omitted. This
        # lets reviewed 0.3 manifests and their content-addressed proofs migrate incrementally;
        # any explicit typed identity, permission scope, or endpoint contrast remains hashed.
        for unit in payload["experimental_design"]["units"]:
            if unit["identity"] is None:
                unit.pop("identity")
            if unit["id_field"] is None:
                unit.pop("id_field")
        sampling = payload["experimental_design"]["sampling"]
        if sampling["subject_identity"] is None:
            sampling.pop("subject_identity")
        if sampling["subject_id_field"] is None:
            sampling.pop("subject_id_field")
        if payload["experimental_design"]["randomized_endpoint_contrast"] is None:
            payload["experimental_design"].pop("randomized_endpoint_contrast")
        for modality in payload["capabilities"]["modalities"]:
            for field_name in ("alignment_unit", "alignment_identity"):
                if modality[field_name] is None:
                    modality.pop(field_name)
        for assessment in payload["claim_assessments"]:
            if assessment["execution_source_ids"] is None:
                assessment.pop("execution_source_ids")
        return payload

    @property
    def canonical_json_bytes(self) -> bytes:
        """Return exact compact UTF-8 bytes whose SHA-256 is the manifest fingerprint."""

        return encode_canonical_json_bytes(self.canonical_payload)

    @property
    def fingerprint(self) -> str:
        return canonical_fingerprint(self.canonical_payload)

    def claim_execution_source_ids(
        self,
        assessment: ClaimAssessment,
    ) -> tuple[str, ...]:
        """Resolve exact model-byte sources, including the explicit legacy migration rule."""

        if assessment.execution_source_ids is not None:
            return tuple(sorted(assessment.execution_source_ids))
        # Backward-compatible 0.3 behavior: before this field existed, the full evidence set was
        # also the permission scope. Explicit migration is required to separate review-only
        # sources; omission never silently changes an already-reviewed legal result.
        return tuple(sorted(assessment.evidence_source_ids))

    def permission_status(
        self,
        use_case: DataUseCase,
        *,
        source_ids: tuple[str, ...] | None = None,
    ) -> PermissionStatus:
        """Return the most restrictive status for a use across the selected sources."""

        return self.effective_permission(use_case, source_ids=source_ids).status

    def effective_permission(
        self,
        use_case: DataUseCase,
        *,
        source_ids: tuple[str, ...] | None = None,
    ) -> EffectiveDataUsePermission:
        """Resolve every applicable policy layer without conflating law and science."""

        if source_ids is not None and not source_ids:
            raise ValueError("permission query source IDs must not be empty")
        if source_ids is not None:
            _require_unique_strings(source_ids, name="permission query source IDs")
        selected = set(
            tuple(source.source_id for source in self.sources) if source_ids is None else source_ids
        )
        known = {source.source_id for source in self.sources}
        if unknown := selected - known:
            raise ValueError(f"unknown source IDs for permission query: {sorted(unknown)}")
        statuses: list[PermissionStatus] = []
        policy_ids: set[str] = set()
        conditions: set[str] = set()
        attribution_requirements: set[str] = set()
        for policy in self.use_policies:
            if selected & set(policy.source_ids):
                permission = next(item for item in policy.permissions if item.use_case is use_case)
                statuses.append(permission.status)
                policy_ids.add(policy.policy_id)
                conditions.update(permission.conditions)
                attribution_requirements.update(policy.attribution_requirements)
        precedence = {
            PermissionStatus.PERMITTED: 0,
            PermissionStatus.CONDITIONAL: 1,
            PermissionStatus.UNKNOWN: 2,
            PermissionStatus.PROHIBITED: 3,
        }
        status = max(statuses, key=precedence.__getitem__)
        return EffectiveDataUsePermission(
            use_case=use_case,
            source_ids=tuple(sorted(selected)),
            status=status,
            applicable_policy_ids=tuple(sorted(policy_ids)),
            conditions=tuple(sorted(conditions)),
            attribution_requirements=tuple(sorted(attribution_requirements)),
        )

    def resolve_assessment(
        self,
        reference: DatasetAssessmentReference,
        *,
        use_case: DataUseCase,
    ) -> DatasetAssessmentResolution:
        """Resolve one exact assessment plus its independently computed legal permission."""

        if reference.dataset_manifest_fingerprint != self.fingerprint:
            raise ValueError("assessment reference does not bind this dataset manifest")
        records: tuple[
            tuple[AssessmentKind, ClaimAssessment | ObjectiveEligibilityAssessment], ...
        ] = (
            *((AssessmentKind.CLAIM, item) for item in self.claim_assessments),
            *((AssessmentKind.LOSS, item) for item in self.loss_assessments),
            *((AssessmentKind.METRIC, item) for item in self.metric_assessments),
        )
        matches = tuple(
            item for item in records if item[1].assessment_id == reference.assessment_id
        )
        if not matches:
            raise ValueError("assessment reference names an unknown assessment ID")
        assessment_kind, assessment = matches[0]
        if reference.assessment_fingerprint != assessment.fingerprint:
            raise ValueError("assessment reference fingerprint does not match its assessment")
        assessment_source_ids = (
            self.claim_execution_source_ids(assessment)
            if isinstance(assessment, ClaimAssessment)
            else assessment.data_source_ids
        )
        data_source_ids = tuple(sorted(assessment_source_ids))
        permission = (
            self.effective_permission(use_case, source_ids=data_source_ids)
            if data_source_ids
            else None
        )
        return DatasetAssessmentResolution(
            reference=reference,
            assessment_kind=assessment_kind,
            scientific_status=assessment.status,
            data_source_ids=data_source_ids,
            effective_permission=permission,
            workflow_status=_workflow_status(assessment.status, permission),
            workflow_reasons=_workflow_reasons(assessment.status, permission),
            scientific_assumptions=assessment.assumptions,
            scientific_blockers=assessment.blockers,
            legal_conditions=permission.conditions if permission is not None else (),
            applicable_policy_ids=(
                permission.applicable_policy_ids if permission is not None else ()
            ),
            use_allowed_without_additional_review=(
                assessment.status is EligibilityStatus.ELIGIBLE
                and permission is not None
                and permission.status is PermissionStatus.PERMITTED
            ),
        )

    @model_validator(mode="after")
    def references_and_claims_are_coherent(self) -> DatasetManifest:
        source_ids = [source.source_id for source in self.sources]
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("dataset source IDs must be unique")
        known_sources = set(source_ids)
        sources_by_id = {source.source_id: source for source in self.sources}
        scientific_sources = tuple(
            source for source in self.sources if source.kind is not SourceKind.DOCUMENTATION
        )

        def is_linked_to_origin(source: SourceArtifact) -> bool:
            direct_link = (
                source.accession == self.origin.study_accession
                and source.release == self.origin.release
            )
            if source.parent_study_accession is None:
                return direct_link
            return (
                source.parent_study_accession == self.origin.study_accession
                and source.parent_study_release == self.origin.release
            )

        if not scientific_sources or any(
            not is_linked_to_origin(source) for source in scientific_sources
        ):
            raise ValueError(
                "every raw, processed, or metadata source must link to the origin accession "
                "and release, directly or through an explicit parent study"
            )

        covered_policy_sources: list[str] = []
        policy_ids = [policy.policy_id for policy in self.use_policies]
        _require_unique_strings(tuple(policy_ids), name="data-use policy IDs")
        for policy in self.use_policies:
            covered_policy_sources.extend(policy.source_ids)
        covered_sources = set(covered_policy_sources)
        if unknown := covered_sources - known_sources:
            raise ValueError(f"data-use policies reference unknown sources: {sorted(unknown)}")
        if missing := known_sources - covered_sources:
            raise ValueError(f"data-use policies must cover every source: {sorted(missing)}")

        referenced_sources: set[str] = set(self.experimental_design.sampling.source_ids)
        referenced_sources.update(self.slice_spec.selection_source_ids)
        for stage in self.slice_spec.selection_stages:
            referenced_sources.update(stage.source_ids)
        for unit in self.experimental_design.units:
            referenced_sources.update(unit.source_ids)
        declared_endpoint_contrast = self.experimental_design.randomized_endpoint_contrast
        if declared_endpoint_contrast is not None:
            referenced_sources.update(declared_endpoint_contrast.source_ids)
            referenced_sources.update(declared_endpoint_contrast.matched_control.source_ids)
        for modality in self.capabilities.modalities:
            referenced_sources.update(modality.source_ids)
            if modality.collection_time_window is not None:
                referenced_sources.update(modality.collection_time_window.source_ids)
        referenced_sources.update(self.capabilities.interventions.source_ids)
        referenced_sources.update(self.capabilities.timing.source_ids)
        referenced_sources.update(self.capabilities.lineage.source_ids)
        referenced_sources.update(self.capabilities.spatial.source_ids)
        for output in self.capabilities.functional.outputs:
            referenced_sources.update(output.source_ids)
            if output.measurement_time_window is not None:
                referenced_sources.update(output.measurement_time_window.source_ids)
        for variable in self.capabilities.environment.variables:
            referenced_sources.update(variable.source_ids)
        for assessment in self.claim_assessments:
            referenced_sources.update(assessment.evidence_source_ids)
            if assessment.execution_source_ids is not None:
                referenced_sources.update(assessment.execution_source_ids)
            if assessment.scope.inference_cutoff_window is not None:
                referenced_sources.update(assessment.scope.inference_cutoff_window.source_ids)
            for window in assessment.scope.horizon_windows:
                referenced_sources.update(window.source_ids)
        for objective_assessment in (*self.loss_assessments, *self.metric_assessments):
            referenced_sources.update(objective_assessment.data_source_ids)
        if unknown := referenced_sources - known_sources:
            raise ValueError(f"manifest references unknown source artifacts: {sorted(unknown)}")

        def reject_documentation_only(ids: tuple[str, ...], *, name: str) -> None:
            if ids and all(
                sources_by_id[source_id].kind is SourceKind.DOCUMENTATION for source_id in ids
            ):
                raise ValueError(f"{name} cannot rely only on documentation artifacts")

        for unit in self.experimental_design.units:
            reject_documentation_only(unit.source_ids, name="experimental-unit evidence")
        reject_documentation_only(
            self.experimental_design.sampling.source_ids,
            name="sampling evidence",
        )
        reject_documentation_only(
            self.slice_spec.selection_source_ids,
            name="dataset-slice selection evidence",
        )
        if declared_endpoint_contrast is not None:
            reject_documentation_only(
                declared_endpoint_contrast.source_ids,
                name="randomized endpoint-contrast evidence",
            )
            reject_documentation_only(
                declared_endpoint_contrast.matched_control.source_ids,
                name="matched-control evidence",
            )
        for stage in self.slice_spec.selection_stages:
            reject_documentation_only(
                stage.source_ids,
                name=f"dataset-slice stage {stage.stage_id} evidence",
            )
        for modality in self.capabilities.modalities:
            kinds = {sources_by_id[source_id].kind for source_id in modality.source_ids}
            if not kinds <= {SourceKind.RAW, SourceKind.PROCESSED}:
                raise ValueError("modality evidence must reference raw or processed artifacts")
            if modality.raw_available is not (SourceKind.RAW in kinds) or (
                modality.processed_available is not (SourceKind.PROCESSED in kinds)
            ):
                raise ValueError("modality availability must match referenced source kinds")
        units_by_level = {unit.level: unit for unit in self.experimental_design.units}

        def is_same_or_ancestor(
            candidate: ExperimentalUnitLevel,
            level: ExperimentalUnitLevel,
        ) -> bool:
            current: ExperimentalUnitLevel | None = level
            while current is not None:
                if current is candidate:
                    return True
                current = units_by_level[current].parent_level
            return False

        alignment_levels = {
            SubjectAlignment.SAME_CELL: ExperimentalUnitLevel.CELL,
            SubjectAlignment.SAME_SAMPLE: ExperimentalUnitLevel.SAMPLE,
            SubjectAlignment.SAME_CLONE: ExperimentalUnitLevel.CLONE,
            SubjectAlignment.SAME_SPATIAL_REGION: ExperimentalUnitLevel.SPATIAL_REGION,
        }
        for modality in self.capabilities.modalities:
            if modality.subject_alignment is SubjectAlignment.UNPAIRED:
                continue
            alignment_level = (
                self.experimental_design.sampling.subject_unit
                if modality.subject_alignment is SubjectAlignment.SAME_POPULATION
                else alignment_levels[modality.subject_alignment]
            )
            alignment_unit = units_by_level.get(alignment_level)
            if alignment_unit is None:
                raise ValueError("modality alignment references an undeclared unit level")
            if modality.alignment_unit is not None:
                if modality.alignment_unit is not alignment_level:
                    raise ValueError(
                        "modality alignment unit must match its declared subject alignment"
                    )
            else:
                modality_identity = (
                    modality.alignment_identity
                    if modality.alignment_identity is not None
                    else _source_field_identity(modality.alignment_key_field or "")
                )
                if modality_identity != alignment_unit.resolved_identity:
                    raise ValueError(
                        "modality alignment key must match its declared unit's exact identity"
                    )
        for ids, name in (
            (self.capabilities.interventions.source_ids, "intervention evidence"),
            (self.capabilities.timing.source_ids, "timing evidence"),
            (self.capabilities.lineage.source_ids, "lineage evidence"),
            (self.capabilities.spatial.source_ids, "spatial evidence"),
        ):
            reject_documentation_only(ids, name=name)
        population_unit_levels = set(ExperimentalUnitLevel) - {
            ExperimentalUnitLevel.CELL,
            ExperimentalUnitLevel.CLONE,
            ExperimentalUnitLevel.SPATIAL_REGION,
        }
        for output in self.capabilities.functional.outputs:
            reject_documentation_only(output.source_ids, name="functional evidence")
            aggregation_unit = units_by_level.get(output.aggregation_level)
            if aggregation_unit is None:
                raise ValueError("functional aggregation references an undeclared unit level")
            if output.alignment_key_field != aggregation_unit.id_field:
                raise ValueError(
                    "functional alignment key must match its aggregation-unit ID field"
                )
            expected_level = alignment_levels.get(output.subject_alignment)
            if expected_level is not None and output.aggregation_level is not expected_level:
                raise ValueError(
                    "functional subject alignment is incompatible with its aggregation level"
                )
            if output.subject_alignment is SubjectAlignment.SAME_POPULATION and (
                output.aggregation_level not in population_unit_levels
                or not is_same_or_ancestor(
                    output.aggregation_level,
                    self.experimental_design.sampling.subject_unit,
                )
            ):
                raise ValueError(
                    "population-aligned functional output must aggregate at the sampling unit "
                    "or one of its declared ancestors"
                )
        for variable in self.capabilities.environment.variables:
            reject_documentation_only(variable.source_ids, name="environment evidence")
        supported_statuses = {
            EligibilityStatus.ELIGIBLE,
            EligibilityStatus.CONDITIONALLY_ELIGIBLE,
        }
        for assessment in self.claim_assessments:
            if assessment.status in supported_statuses:
                reject_documentation_only(
                    assessment.evidence_source_ids,
                    name="supported claim evidence",
                )
                execution_source_ids = self.claim_execution_source_ids(assessment)
                if not execution_source_ids:
                    raise ValueError("supported claims require executable non-documentation data")
                if not set(execution_source_ids) <= set(assessment.evidence_source_ids):
                    raise ValueError(
                        "claim execution sources must be a subset of its scientific evidence"
                    )
                if assessment.execution_source_ids is not None and any(
                    sources_by_id[source_id].kind is SourceKind.DOCUMENTATION
                    for source_id in execution_source_ids
                ):
                    raise ValueError("claim execution sources cannot be documentation artifacts")
        for objective_assessment in (*self.loss_assessments, *self.metric_assessments):
            if objective_assessment.status in supported_statuses:
                reject_documentation_only(
                    objective_assessment.data_source_ids,
                    name="supported objective data",
                )

        all_assessment_ids = (
            *(assessment.assessment_id for assessment in self.claim_assessments),
            *(assessment.assessment_id for assessment in self.loss_assessments),
            *(assessment.assessment_id for assessment in self.metric_assessments),
        )
        _require_unique_strings(all_assessment_ids, name="dataset assessment IDs")
        claim_keys = tuple(
            (assessment.claim, assessment.scope.fingerprint)
            for assessment in self.claim_assessments
        )
        if len(claim_keys) != len(set(claim_keys)):
            raise ValueError("claim and canonical assessment scope pairs must be unique")
        loss_keys = tuple(
            (
                assessment.loss_kind,
                assessment.scope.fingerprint,
                assessment.required_split_unit,
            )
            for assessment in self.loss_assessments
        )
        if len(loss_keys) != len(set(loss_keys)):
            raise ValueError("loss kind, scope, and required split tuples must be unique")
        metric_keys = tuple(
            (
                assessment.metric_id,
                assessment.scope.fingerprint,
                assessment.required_split_unit,
                assessment.partition_purpose,
            )
            for assessment in self.metric_assessments
        )
        if len(metric_keys) != len(set(metric_keys)):
            raise ValueError("metric, scope, split, and partition-purpose tuples must be unique")
        sampling = self.experimental_design.sampling
        if (
            sampling.time_field is not None
            and not self.capabilities.timing.observation_times_recorded
        ):
            raise ValueError("sampling time field requires observation timing capability")
        timing_windows_by_id = {
            window.window_id: window for window in self.capabilities.timing.observation_windows
        }
        if sampling.time_window_id is not None and (
            sampling.time_window_id not in timing_windows_by_id
        ):
            raise ValueError("sampling time-window ID must resolve to timing capability")
        if declared_endpoint_contrast is not None:
            interventions = self.capabilities.interventions
            if (
                interventions.assignment is not AssignmentMechanism.RANDOMIZED
                or not interventions.matched_controls_present
            ):
                raise ValueError(
                    "randomized endpoint contrast requires randomized assignment and controls"
                )
            if (
                not self.capabilities.timing.intervention_times_recorded
                or not self.capabilities.timing.event_ordering_recorded
                or declared_endpoint_contrast.endpoint_time_seconds
                not in self.capabilities.timing.timepoints_seconds
            ):
                raise ValueError(
                    "randomized endpoint contrast requires recorded assignment ordering and its "
                    "exact observed endpoint time"
                )

        def require_registered_window(window: TemporalWindow, *, name: str) -> None:
            registered = timing_windows_by_id.get(window.window_id)
            if registered is None or registered.fingerprint != window.fingerprint:
                raise ValueError(f"{name} must exactly match a registered observation window")

        for modality in self.capabilities.modalities:
            if modality.collection_time_window is not None:
                require_registered_window(
                    modality.collection_time_window,
                    name="modality collection-time window",
                )
        for output in self.capabilities.functional.outputs:
            if output.measurement_time_window is not None:
                require_registered_window(
                    output.measurement_time_window,
                    name="functional measurement-time window",
                )
        for assessment in self.claim_assessments:
            if assessment.scope.inference_cutoff_window is not None:
                require_registered_window(
                    assessment.scope.inference_cutoff_window,
                    name="assessment inference-cutoff window",
                )
            for window in assessment.scope.horizon_windows:
                require_registered_window(window, name="assessment horizon window")
        if any(variable.time_resolved for variable in self.capabilities.environment.variables) and (
            not self.capabilities.timing.environment_times_recorded
        ):
            raise ValueError("time-resolved environment variables require environment timing")
        if self.capabilities.lineage.division_times_recorded and (
            not self.capabilities.timing.observation_times_recorded
        ):
            raise ValueError("division times require observation timing capability")
        if sampling.linkage in {SubjectLinkage.SAME_CLONE, SubjectLinkage.LINEAGE} and (
            self.capabilities.lineage.resolution is LineageResolution.NONE
        ):
            raise ValueError("lineage-linked sampling requires lineage capability")

        modality_by_key = {item.modality.key: item for item in self.capabilities.modalities}
        intervention_keys = set(_term_keys(self.capabilities.interventions.kinds))
        functional_by_id = {item.readout_id: item for item in self.capabilities.functional.outputs}
        environment_by_key = {
            item.variable.key: item for item in self.capabilities.environment.variables
        }
        biological_system_keys = set(_term_keys(self.origin.biological_systems))

        supported = tuple(
            item for item in self.claim_assessments if item.status in supported_statuses
        )
        for assessment in supported:
            scope = assessment.scope
            if scope.subject_kind is not sampling.subject_kind:
                raise ValueError("claim subject kind must match the dataset sampling subject")
            references = (
                (
                    set(_term_keys(scope.biological_systems)),
                    biological_system_keys,
                    "biological system",
                ),
                (set(_term_keys(scope.modalities)), set(modality_by_key), "modality"),
                (set(_term_keys(scope.intervention_kinds)), intervention_keys, "intervention kind"),
                (
                    set(scope.functional_readout_ids),
                    set(functional_by_id),
                    "functional readout",
                ),
                (
                    set(_term_keys(scope.environment_variables)),
                    set(environment_by_key),
                    "environment variable",
                ),
            )
            for requested, available, name in references:
                if unknown := requested - available:
                    raise ValueError(
                        f"claim scope references unsupported {name}: {sorted(unknown)}"
                    )
            evidence_sources = set(assessment.evidence_source_ids)

            def require_scoped_evidence(
                source_ids: tuple[str, ...],
                *,
                name: str,
                claim_sources: set[str] = evidence_sources,
            ) -> None:
                if claim_sources.isdisjoint(source_ids):
                    raise ValueError(f"claim evidence does not support its scoped {name}")

            if scope.system_boundary is SystemBoundary.CELL_AND_SOLUBLE_ENVIRONMENT and (
                not scope.environment_variables
            ):
                raise ValueError(
                    "a soluble-environment boundary requires scoped environment variables"
                )
            if scope.system_boundary in {
                SystemBoundary.CELL_AND_NEIGHBORS,
                SystemBoundary.SPATIAL_TISSUE_NICHE,
            }:
                spatial = self.capabilities.spatial
                admissible_resolutions = (
                    {SpatialResolution.NEIGHBOR_GRAPH}
                    if scope.system_boundary is SystemBoundary.CELL_AND_NEIGHBORS
                    else {
                        SpatialResolution.CELL_COORDINATES,
                        SpatialResolution.NEIGHBOR_GRAPH,
                        SpatialResolution.IMAGE_VOLUME,
                    }
                )
                if spatial.resolution not in admissible_resolutions:
                    raise ValueError(
                        "a spatial or neighborhood system boundary requires cell-resolved "
                        "spatial capability"
                    )
                require_scoped_evidence(spatial.source_ids, name="spatial boundary")
                spatial_alignment = (
                    spatial.subject_alignment,
                    spatial.alignment_group,
                    spatial.alignment_key_field,
                )
                scoped_modalities = (modality_by_key[key] for key in _term_keys(scope.modalities))
                if any(value is None for value in spatial_alignment) or not any(
                    (
                        modality.subject_alignment,
                        modality.alignment_group,
                        modality.alignment_key_field,
                    )
                    == spatial_alignment
                    for modality in scoped_modalities
                ):
                    raise ValueError(
                        "a spatial system boundary requires a scoped modality aligned to its "
                        "spatial evidence"
                    )
            if scope.system_boundary is SystemBoundary.CLONE:
                if self.capabilities.lineage.resolution is LineageResolution.NONE:
                    raise ValueError("a clone boundary requires lineage or clone identity evidence")
                require_scoped_evidence(
                    self.capabilities.lineage.source_ids,
                    name="clone boundary",
                )

            for modality_key in _term_keys(scope.modalities):
                require_scoped_evidence(
                    modality_by_key[modality_key].source_ids,
                    name=f"modality {modality_key}",
                )
            if scope.intervention_kinds:
                require_scoped_evidence(
                    self.capabilities.interventions.source_ids,
                    name="interventions",
                )
                if declared_endpoint_contrast is not None:
                    require_scoped_evidence(
                        declared_endpoint_contrast.source_ids,
                        name="randomized endpoint contrast",
                    )
                    require_scoped_evidence(
                        declared_endpoint_contrast.matched_control.source_ids,
                        name="matched-control definition",
                    )
            for readout_id in scope.functional_readout_ids:
                require_scoped_evidence(
                    functional_by_id[readout_id].source_ids,
                    name=f"functional readout {readout_id}",
                )
            for environment_key in _term_keys(scope.environment_variables):
                require_scoped_evidence(
                    environment_by_key[environment_key].source_ids,
                    name=f"environment variable {environment_key}",
                )
            if scope.horizons_seconds or scope.horizon_windows:
                require_scoped_evidence(
                    self.capabilities.timing.source_ids,
                    name="timing",
                )
            if assessment.claim in {
                ScientificClaim.POPULATION_DYNAMICS,
                ScientificClaim.INDIVIDUAL_LONGITUDINAL_DYNAMICS,
                ScientificClaim.LINEAGE_FATE,
            }:
                require_scoped_evidence(sampling.source_ids, name="sampling structure")
            if assessment.claim is ScientificClaim.LINEAGE_FATE:
                require_scoped_evidence(
                    self.capabilities.lineage.source_ids,
                    name="lineage",
                )
            if assessment.claim is ScientificClaim.SPATIAL_CONTEXT:
                require_scoped_evidence(
                    self.capabilities.spatial.source_ids,
                    name="spatial structure",
                )
            self._validate_supported_claim(assessment, modality_by_key, functional_by_id)

        self._validate_objective_assessments(
            known_sources=known_sources,
        )
        return self

    def _validate_objective_assessments(self, *, known_sources: set[str]) -> None:
        claims_by_id = {
            assessment.assessment_id: assessment for assessment in self.claim_assessments
        }
        units_by_level = {unit.level: unit for unit in self.experimental_design.units}
        functional_by_id = {
            readout.readout_id: readout for readout in self.capabilities.functional.outputs
        }

        def is_same_or_ancestor(
            candidate: ExperimentalUnitLevel,
            level: ExperimentalUnitLevel,
        ) -> bool:
            current: ExperimentalUnitLevel | None = level
            while current is not None:
                if current is candidate:
                    return True
                current = units_by_level[current].parent_level
            return False

        supported_statuses = {
            EligibilityStatus.ELIGIBLE,
            EligibilityStatus.CONDITIONALLY_ELIGIBLE,
        }
        for assessment in (*self.loss_assessments, *self.metric_assessments):
            if unknown := set(assessment.data_source_ids) - known_sources:
                raise ValueError(
                    f"objective assessment references unknown data sources: {sorted(unknown)}"
                )
            resolved_claims: list[ClaimAssessment] = []
            for reference in assessment.supporting_claim_assessments:
                claim = claims_by_id.get(reference.assessment_id)
                if claim is None:
                    raise ValueError("objective references an unknown claim assessment ID")
                if reference.assessment_fingerprint != claim.fingerprint:
                    raise ValueError("objective claim-assessment fingerprint does not match")
                if assessment.scope.fingerprint != claim.scope.fingerprint:
                    raise ValueError("objective and supporting claim scopes must match exactly")
                resolved_claims.append(claim)

            if assessment.status not in supported_statuses:
                continue
            if any(claim.status not in supported_statuses for claim in resolved_claims):
                raise ValueError("supported objectives require supported claim assessments")
            if assessment.status is EligibilityStatus.ELIGIBLE and any(
                claim.status is EligibilityStatus.CONDITIONALLY_ELIGIBLE
                for claim in resolved_claims
            ):
                raise ValueError(
                    "eligible objectives cannot be stronger than conditional supporting claims"
                )
            claim_sources = {
                source_id
                for claim in resolved_claims
                for source_id in self.claim_execution_source_ids(claim)
            }
            if set(assessment.data_source_ids) != claim_sources:
                raise ValueError(
                    "objective data sources must exactly cover all supporting claim evidence "
                    "used as execution data"
                )
            split_unit = assessment.required_split_unit
            if split_unit not in units_by_level:
                raise ValueError("objective split requirement references an undeclared unit")
            scoped_readout_levels = {
                functional_by_id[readout_id].aggregation_level
                for readout_id in assessment.scope.functional_readout_ids
            }
            protected_levels = {
                self.experimental_design.default_split_unit,
                self.experimental_design.sampling.subject_unit,
                *scoped_readout_levels,
                *(
                    level
                    for level in (
                        self.experimental_design.biological_replicate_unit,
                        self.experimental_design.randomization_unit,
                    )
                    if level is not None
                ),
            }
            if any(not is_same_or_ancestor(split_unit, level) for level in protected_levels):
                raise ValueError(
                    "objective required split unit cannot be finer than a protected unit"
                )
            claim_kinds = {claim.claim for claim in resolved_claims}
            if isinstance(assessment, LossEligibilityAssessment):
                compatible = _LOSS_COMPATIBLE_CLAIMS[assessment.loss_kind]
                if not claim_kinds or not claim_kinds <= compatible:
                    raise ValueError("loss kind has no compatible supporting scientific claim")
            elif isinstance(assessment, MetricEligibilityAssessment):
                compatible = _METRIC_COMPATIBLE_CLAIMS[assessment.metric_family]
                if not claim_kinds or not claim_kinds <= compatible:
                    raise ValueError("metric family has no compatible supporting scientific claim")
                if assessment.metric_family is MetricFamily.POPULATION_DISTRIBUTION and (
                    assessment.scope.subject_kind is not SamplingSubjectKind.POPULATION
                    or assessment.scope.system_boundary is not SystemBoundary.POPULATION
                ):
                    raise ValueError(
                        "population-distribution metrics require an intervention-effect or "
                        "population-dynamics claim, population subject, and population boundary"
                    )

    def _validate_supported_claim(
        self,
        eligibility: ClaimAssessment,
        modality_by_key: dict[str, ModalitySpec],
        functional_by_id: dict[str, FunctionalReadout],
    ) -> None:
        claim = eligibility.claim
        scope = eligibility.scope
        sampling = self.experimental_design.sampling
        timing = self.capabilities.timing
        scoped_modalities = [modality_by_key[key] for key in _term_keys(scope.modalities)]
        scoped_outputs = [functional_by_id[key] for key in scope.functional_readout_ids]
        has_horizon = bool(scope.horizons_seconds or scope.horizon_windows)
        temporal_observation_count = len(timing.timepoints_seconds) + len(
            timing.observation_windows
        )

        if eligibility.identification_basis is IdentificationBasis.TRANSPORTED_UNDER_ASSUMPTIONS:
            raise ValueError(
                "supported transported claims require a structured transport scope with typed "
                "source and target domains"
            )

        explicitly_gated_claims = {
            ScientificClaim.ASSAY_MEASUREMENT_MODEL,
            ScientificClaim.SNAPSHOT_STATE_PRIOR,
            ScientificClaim.SAME_CELL_MULTIMODAL_FUSION,
            ScientificClaim.SAMPLE_LEVEL_MULTIMODAL_FUSION,
            ScientificClaim.POPULATION_DYNAMICS,
            ScientificClaim.INDIVIDUAL_LONGITUDINAL_DYNAMICS,
            ScientificClaim.INTERVENTION_EFFECT,
            ScientificClaim.COUNTERFACTUAL_GENERALIZATION,
            ScientificClaim.LINEAGE_FATE,
            ScientificClaim.SPATIAL_CONTEXT,
            ScientificClaim.FUNCTIONAL_OUTCOME,
            ScientificClaim.RETROSPECTIVE_INTERVENTION_SELECTION,
        }
        if claim not in explicitly_gated_claims:
            raise ValueError("supported scientific claim has no implemented evidence gate")

        def require_observed_horizon(
            *,
            name: str,
            allow_randomized_endpoint_contrast: bool = False,
        ) -> None:
            if not has_horizon:
                raise ValueError(f"{name} requires a prediction horizon")
            if scope.horizon_windows:
                cutoff_window = scope.inference_cutoff_window
                if cutoff_window is None:
                    raise ValueError(f"{name} interval targets require an interval cutoff")
                registered = {window.window_id: window for window in timing.observation_windows}
                required_windows = (cutoff_window, *scope.horizon_windows)
                if any(
                    window.window_id not in registered
                    or registered[window.window_id].fingerprint != window.fingerprint
                    for window in required_windows
                ):
                    raise ValueError(f"{name} interval lies outside observed temporal support")
                return
            if scope.inference_cutoff_seconds is None:
                raise ValueError(
                    f"{name} requires a fixed cutoff until field-clock comparability is modeled"
                )
            endpoint_contrast = self.experimental_design.randomized_endpoint_contrast
            if allow_randomized_endpoint_contrast and endpoint_contrast is not None:
                cutoff = scope.inference_cutoff_seconds
                endpoint_times = tuple(cutoff + horizon for horizon in scope.horizons_seconds)
                if (
                    cutoff == endpoint_contrast.assignment_time_seconds
                    and endpoint_times
                    and all(
                        endpoint_time == endpoint_contrast.endpoint_time_seconds
                        for endpoint_time in endpoint_times
                    )
                    and endpoint_contrast.endpoint_time_seconds in timing.timepoints_seconds
                ):
                    # This path identifies an assignment-to-endpoint contrast. It deliberately
                    # does not reinterpret assignment time as an observed molecular baseline.
                    return
            if not timing.timepoints_seconds:
                raise ValueError(f"{name} horizon exceeds observed temporal support")
            cutoff = scope.inference_cutoff_seconds
            minimum_time = min(timing.timepoints_seconds)
            maximum_time = max(timing.timepoints_seconds)
            if cutoff + 1e-12 < minimum_time or cutoff > maximum_time + 1e-12:
                raise ValueError(f"{name} inference cutoff lies outside observed temporal support")
            if maximum_time + 1e-12 < cutoff + max(scope.horizons_seconds):
                raise ValueError(f"{name} horizon exceeds observed temporal support")

        if claim is ScientificClaim.ASSAY_MEASUREMENT_MODEL:
            if not scoped_modalities or any(
                not modality.raw_available for modality in scoped_modalities
            ):
                raise ValueError(
                    "assay measurement models require scoped raw measurement modalities"
                )
            sources_by_id = {source.source_id: source for source in self.sources}
            evidence_sources = set(eligibility.evidence_source_ids)
            if any(
                evidence_sources.isdisjoint(
                    source_id
                    for source_id in modality.source_ids
                    if sources_by_id[source_id].kind is SourceKind.RAW
                )
                for modality in scoped_modalities
            ):
                raise ValueError(
                    "assay measurement model evidence must cite each scoped modality's raw source"
                )

        if claim is ScientificClaim.SNAPSHOT_STATE_PRIOR and not scoped_modalities:
            raise ValueError("snapshot state priors require a scoped measurement modality")

        if claim in {
            ScientificClaim.SAME_CELL_MULTIMODAL_FUSION,
            ScientificClaim.SAMPLE_LEVEL_MULTIMODAL_FUSION,
        }:
            required_alignment = (
                SubjectAlignment.SAME_CELL
                if claim is ScientificClaim.SAME_CELL_MULTIMODAL_FUSION
                else SubjectAlignment.SAME_SAMPLE
            )
            if len(scoped_modalities) < 2 or any(
                modality.subject_alignment is not required_alignment
                for modality in scoped_modalities
            ):
                raise ValueError(
                    "multimodal fusion requires two scoped, correctly aligned modalities"
                )
            units_by_level = {unit.level: unit for unit in self.experimental_design.units}

            def exact_modality_alignment(
                modality: ModalitySpec,
            ) -> tuple[str | None, str]:
                if modality.alignment_unit is not None:
                    identity = units_by_level[modality.alignment_unit].resolved_identity
                elif modality.alignment_identity is not None:
                    identity = modality.alignment_identity
                else:
                    assert modality.alignment_key_field is not None
                    identity = _source_field_identity(modality.alignment_key_field)
                return (modality.alignment_group, identity.fingerprint)

            pairing = {exact_modality_alignment(modality) for modality in scoped_modalities}
            if len(pairing) != 1:
                raise ValueError(
                    "multimodal fusion modalities must share one alignment group and key"
                )

        if claim is ScientificClaim.POPULATION_DYNAMICS and (
            scope.subject_kind is not SamplingSubjectKind.POPULATION
            or sampling.mode is not SamplingMode.REPEATED_POPULATION_DESTRUCTIVE
            or sampling.linkage is not SubjectLinkage.SAME_POPULATION
            or temporal_observation_count < 2
            or not has_horizon
            or not scoped_modalities
        ):
            raise ValueError(
                "population dynamics requires repeated linked populations and horizons"
            )
        if claim is ScientificClaim.POPULATION_DYNAMICS:
            require_observed_horizon(name="population dynamics")

        if claim is ScientificClaim.INDIVIDUAL_LONGITUDINAL_DYNAMICS and (
            scope.subject_kind is not SamplingSubjectKind.INDIVIDUAL_CELL
            or sampling.mode
            not in {
                SamplingMode.PARTIAL_NONDESTRUCTIVE,
                SamplingMode.LONGITUDINAL_NONDESTRUCTIVE,
            }
            or sampling.linkage is not SubjectLinkage.SAME_CELL
            or temporal_observation_count < 2
            or not has_horizon
            or not scoped_modalities
            or not all(
                not modality.destructive
                and modality.subject_alignment is SubjectAlignment.SAME_CELL
                for modality in scoped_modalities
            )
        ):
            raise ValueError("individual dynamics requires nondestructive same-cell evidence")
        if claim is ScientificClaim.INDIVIDUAL_LONGITUDINAL_DYNAMICS:
            require_observed_horizon(name="individual dynamics")

        if claim in {
            ScientificClaim.INTERVENTION_EFFECT,
            ScientificClaim.COUNTERFACTUAL_GENERALIZATION,
            ScientificClaim.RETROSPECTIVE_INTERVENTION_SELECTION,
        }:
            if claim is ScientificClaim.COUNTERFACTUAL_GENERALIZATION:
                raise ValueError(
                    "counterfactual generalization is unsupported until structured transport "
                    "scope is represented"
                )
            intervention = self.capabilities.interventions
            expected_assignment = {
                IdentificationBasis.RANDOMIZED_WITHIN_STUDY: AssignmentMechanism.RANDOMIZED,
                IdentificationBasis.QUASI_EXPERIMENTAL: AssignmentMechanism.QUASI_EXPERIMENTAL,
            }.get(eligibility.identification_basis)
            assignment_matches_basis = (
                expected_assignment is not None and intervention.assignment is expected_assignment
            )
            if (
                not assignment_matches_basis
                or not scope.intervention_kinds
                or not has_horizon
                or not (scope.modalities or scope.functional_readout_ids)
                or not intervention.matched_controls_present
                or self.experimental_design.biological_replicate_unit is None
                or self.experimental_design.randomization_unit is None
                or (
                    self.experimental_design.matched_control_field is None
                    and self.experimental_design.randomized_endpoint_contrast is None
                )
                or not intervention.start_stop_recorded
                or not timing.intervention_times_recorded
                or not timing.event_ordering_recorded
            ):
                raise ValueError(
                    "intervention claims require scoped timed exposure, recorded event ordering, "
                    "controls, and replication"
                )
            require_observed_horizon(
                name="intervention claim",
                allow_randomized_endpoint_contrast=True,
            )
            if claim is ScientificClaim.RETROSPECTIVE_INTERVENTION_SELECTION and (
                not scope.functional_readout_ids
                or not (intervention.targets_recorded or intervention.doses_recorded)
            ):
                raise ValueError(
                    "retrospective intervention selection requires candidates and a utility"
                )

        if claim is ScientificClaim.LINEAGE_FATE and (
            scope.subject_kind is not SamplingSubjectKind.CLONE
            or sampling.mode is not SamplingMode.LINEAGE_LINKED_ENDPOINT
            or sampling.linkage not in {SubjectLinkage.SAME_CLONE, SubjectLinkage.LINEAGE}
            or self.capabilities.lineage.resolution is LineageResolution.NONE
            or temporal_observation_count < 2
            or not has_horizon
            or not scoped_outputs
        ):
            raise ValueError("lineage fate requires linked lineage, future time, and fate output")
        if claim is ScientificClaim.LINEAGE_FATE:
            require_observed_horizon(name="lineage fate")

        if claim is ScientificClaim.SPATIAL_CONTEXT and (
            self.capabilities.spatial.resolution
            in {SpatialResolution.NONE, SpatialResolution.SAMPLE_REGION}
            or not scoped_modalities
            or scope.system_boundary
            not in {
                SystemBoundary.CELL_AND_NEIGHBORS,
                SystemBoundary.SPATIAL_TISSUE_NICHE,
            }
        ):
            raise ValueError(
                "spatial context requires cell-resolved evidence and a spatial boundary"
            )
        if claim is ScientificClaim.SPATIAL_CONTEXT:
            spatial = self.capabilities.spatial
            spatial_alignment = (
                spatial.subject_alignment,
                spatial.alignment_group,
                spatial.alignment_key_field,
            )
            if any(value is None for value in spatial_alignment) or not any(
                (
                    modality.subject_alignment,
                    modality.alignment_group,
                    modality.alignment_key_field,
                )
                == spatial_alignment
                for modality in scoped_modalities
            ):
                raise ValueError(
                    "spatial context requires a scoped modality aligned to the spatial "
                    "evidence group and key"
                )

        functional_prediction_claims = {
            ScientificClaim.FUNCTIONAL_OUTCOME,
            ScientificClaim.LINEAGE_FATE,
            ScientificClaim.RETROSPECTIVE_INTERVENTION_SELECTION,
        }
        intervention_claim_with_function = claim in {
            ScientificClaim.INTERVENTION_EFFECT,
            ScientificClaim.COUNTERFACTUAL_GENERALIZATION,
        } and bool(scoped_outputs)
        if claim in functional_prediction_claims or intervention_claim_with_function:
            if (
                not has_horizon
                or not scoped_outputs
                or (claim in functional_prediction_claims and not scoped_modalities)
            ):
                raise ValueError(
                    "future functional claims require conditioning modalities, outputs, "
                    "and horizons"
                )
            expected_output_alignment = {
                SamplingSubjectKind.INDIVIDUAL_CELL: SubjectAlignment.SAME_CELL,
                SamplingSubjectKind.CLONE: SubjectAlignment.SAME_CLONE,
                SamplingSubjectKind.SAMPLE: SubjectAlignment.SAME_SAMPLE,
                SamplingSubjectKind.POPULATION: SubjectAlignment.SAME_POPULATION,
                SamplingSubjectKind.SPATIAL_REGION: SubjectAlignment.SAME_SPATIAL_REGION,
            }[scope.subject_kind]
            covered_interval_targets: set[str] = set()
            interval_targets_by_fingerprint = {
                window.fingerprint: window for window in scope.horizon_windows
            }
            for output in scoped_outputs:
                if output.subject_alignment is not expected_output_alignment:
                    raise ValueError(
                        "functional output alignment must match the scoped prediction subject"
                    )
                if (
                    eligibility.status is EligibilityStatus.ELIGIBLE
                    and output.status is ReadoutStatus.DERIVED
                ):
                    raise ValueError(
                        "derived functional targets must remain conditionally eligible"
                    )
                if scope.horizon_windows:
                    measurement_window = output.measurement_time_window
                    if (
                        measurement_window is None
                        or measurement_window.fingerprint not in interval_targets_by_fingerprint
                    ):
                        raise ValueError(
                            "functional measurement window must exactly match a scoped target "
                            "window"
                        )
                    covered_interval_targets.add(measurement_window.fingerprint)
                elif (
                    scope.inference_cutoff_seconds is not None
                    and output.measurement_time_seconds is not None
                ):
                    cutoff = scope.inference_cutoff_seconds
                    measurement = output.measurement_time_seconds
                    if measurement + 1e-12 < cutoff + max(scope.horizons_seconds):
                        raise ValueError("functional measurement does not cover the claim horizon")
                    if (
                        not timing.timepoints_seconds
                        or measurement + 1e-12 < min(timing.timepoints_seconds)
                        or measurement > max(timing.timepoints_seconds) + 1e-12
                    ):
                        raise ValueError(
                            "functional measurement lies outside declared temporal support"
                        )
                elif (
                    scope.inference_cutoff_field is not None
                    and output.measurement_time_field is not None
                ):
                    raise ValueError(
                        "field-clock future outcomes are unsupported until lag comparability "
                        "is modeled"
                    )
                else:
                    raise ValueError(
                        "functional outcome and inference cutoff need comparable time forms"
                    )
            if set(interval_targets_by_fingerprint) != covered_interval_targets:
                raise ValueError("every scoped target window requires a functional output")
