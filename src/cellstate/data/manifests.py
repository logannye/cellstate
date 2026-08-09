"""Experimental contracts for public-real biological dataset evidence.

These models describe source evidence and the claims it may support. They do not download,
normalize, impute, pair, or biologically interpret measurements.
"""

from __future__ import annotations

from datetime import date, datetime
from enum import StrEnum
from typing import Literal
from urllib.parse import urlparse

from pydantic import ConfigDict, Field, field_validator, model_validator

from cellstate.domain.common import OntologyTerm, SchemaModel, canonical_fingerprint, require_finite
from cellstate.domain.query import SystemBoundary

DatasetManifestSchemaVersion = Literal["0.1-experimental"]
DATASET_MANIFEST_SCHEMA_VERSION: DatasetManifestSchemaVersion = "0.1-experimental"


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


class ExperimentalUnitSpec(ManifestModel):
    level: ExperimentalUnitLevel
    id_field: str = Field(min_length=1)
    source_ids: tuple[str, ...] = Field(min_length=1)
    parent_level: ExperimentalUnitLevel | None = None

    @model_validator(mode="after")
    def unit_spec_is_coherent(self) -> ExperimentalUnitSpec:
        if self.parent_level is self.level:
            raise ValueError("an experimental-unit level cannot parent itself")
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
    subject_id_field: str = Field(min_length=1)
    source_ids: tuple[str, ...] = Field(min_length=1)
    mode: SamplingMode
    linkage: SubjectLinkage
    time_field: str | None = None
    source_time_units: str | None = None
    canonical_time_units: Literal["s"] = "s"
    attrition_field: str | None = None

    @model_validator(mode="after")
    def sampling_and_linkage_are_coherent(self) -> SamplingDesign:
        _require_unique_strings(self.source_ids, name="sampling source IDs")
        if (self.time_field is None) != (self.source_time_units is None):
            raise ValueError("sampling time field and source units must be declared together")
        repeated_modes = {
            SamplingMode.REPEATED_POPULATION_DESTRUCTIVE,
            SamplingMode.LINEAGE_LINKED_ENDPOINT,
            SamplingMode.PARTIAL_NONDESTRUCTIVE,
            SamplingMode.LONGITUDINAL_NONDESTRUCTIVE,
        }
        if self.mode in repeated_modes and self.time_field is None:
            raise ValueError("repeated, lineage, or longitudinal sampling requires a time field")
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


class ExperimentalDesign(ManifestModel):
    units: tuple[ExperimentalUnitSpec, ...] = Field(min_length=1)
    sampling: SamplingDesign
    default_split_unit: ExperimentalUnitLevel
    biological_replicate_unit: ExperimentalUnitLevel | None = None
    randomization_unit: ExperimentalUnitLevel | None = None
    matched_control_field: str | None = None
    batch_fields: tuple[str, ...] = ()

    @model_validator(mode="after")
    def unit_graph_and_split_are_valid(self) -> ExperimentalDesign:
        by_level = {unit.level: unit for unit in self.units}
        if len(by_level) != len(self.units):
            raise ValueError("experimental-unit levels must be unique")
        id_fields = [unit.id_field for unit in self.units]
        _require_unique_strings(tuple(id_fields), name="experimental-unit ID fields")
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
        if subject_unit.id_field != self.sampling.subject_id_field:
            raise ValueError("sampling subject ID field must match its declared unit")
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
        _require_unique_strings(self.batch_fields, name="batch fields")
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
    raw_available: bool = False
    processed_available: bool = False
    destructive: bool
    feature_identifier_namespace: str | None = None

    @model_validator(mode="after")
    def representation_and_alignment_are_explicit(self) -> ModalitySpec:
        if not self.raw_available and not self.processed_available:
            raise ValueError("a modality must expose raw or processed measurements")
        _require_unique_strings(self.source_ids, name="modality source IDs")
        alignment_values = (self.alignment_group, self.alignment_key_field)
        if self.subject_alignment is SubjectAlignment.UNPAIRED:
            if any(value is not None for value in alignment_values):
                raise ValueError("unpaired modality must not declare an alignment key")
        elif any(value is None for value in alignment_values):
            raise ValueError("paired modality requires an alignment group and key field")
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
        assertions_present = bool(self.timepoints_seconds) or any(
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
        if self.timepoints_seconds and not self.observation_times_recorded:
            raise ValueError("declared timepoints require recorded observation times")
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


class FunctionalReadout(ManifestModel):
    output: OntologyTerm
    source_ids: tuple[str, ...] = Field(min_length=1)
    units: str = Field(min_length=1)
    aggregation_level: ExperimentalUnitLevel
    subject_alignment: SubjectAlignment
    alignment_group: str = Field(min_length=1)
    alignment_key_field: str = Field(min_length=1)
    status: ReadoutStatus
    measurement_time_seconds: float | None = None
    measurement_time_field: str | None = None

    @field_validator("measurement_time_seconds")
    @classmethod
    def fixed_measurement_time_is_finite(cls, value: float | None) -> float | None:
        if value is not None:
            return require_finite(value, name="functional measurement time")
        return value

    @model_validator(mode="after")
    def measurement_time_is_explicit(self) -> FunctionalReadout:
        _require_unique_strings(self.source_ids, name="functional source IDs")
        if (self.measurement_time_seconds is None) == (self.measurement_time_field is None):
            raise ValueError(
                "functional readout requires exactly one fixed or field measurement time"
            )
        if self.subject_alignment is SubjectAlignment.UNPAIRED:
            raise ValueError("functional readout requires an explicit subject alignment")
        return self


class FunctionalCapability(ManifestModel):
    outputs: tuple[FunctionalReadout, ...] = ()

    @field_validator("outputs")
    @classmethod
    def outputs_are_unique(
        cls, outputs: tuple[FunctionalReadout, ...]
    ) -> tuple[FunctionalReadout, ...]:
        _require_unique_terms(tuple(output.output for output in outputs), name="functional outputs")
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


class ClaimStatus(StrEnum):
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


class ClaimScope(ManifestModel):
    subject_kind: SamplingSubjectKind
    system_boundary: SystemBoundary
    biological_systems: tuple[OntologyTerm, ...] = Field(min_length=1)
    modalities: tuple[OntologyTerm, ...] = ()
    intervention_kinds: tuple[OntologyTerm, ...] = ()
    functional_outputs: tuple[OntologyTerm, ...] = ()
    environment_variables: tuple[OntologyTerm, ...] = ()
    horizons_seconds: tuple[float, ...] = ()
    inference_cutoff_seconds: float | None = None
    inference_cutoff_field: str | None = None

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

    @model_validator(mode="after")
    def scope_members_and_cutoff_are_coherent(self) -> ClaimScope:
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
            ("claim functional outputs", self.functional_outputs),
            ("claim environment variables", self.environment_variables),
        ):
            _require_unique_terms(terms, name=name)
        cutoff_values = (self.inference_cutoff_seconds, self.inference_cutoff_field)
        if self.horizons_seconds and sum(value is not None for value in cutoff_values) != 1:
            raise ValueError("claims with horizons require exactly one inference cutoff")
        if not self.horizons_seconds and any(value is not None for value in cutoff_values):
            raise ValueError("a claim without horizons must not declare an inference cutoff")
        return self


class ClaimEligibility(ManifestModel):
    claim: ScientificClaim
    status: ClaimStatus
    identification_basis: IdentificationBasis
    scope: ClaimScope
    evidence_source_ids: tuple[str, ...] = ()
    evidence_notes: tuple[str, ...] = ()
    assumptions: tuple[str, ...] = ()
    blockers: tuple[str, ...] = ()

    @model_validator(mode="after")
    def status_has_honest_evidence(self) -> ClaimEligibility:
        _require_unique_strings(self.evidence_source_ids, name="claim evidence source IDs")
        _require_unique_strings(self.evidence_notes, name="claim evidence notes")
        _require_unique_strings(self.assumptions, name="claim assumptions")
        _require_unique_strings(self.blockers, name="claim blockers")
        supported = self.status in {ClaimStatus.ELIGIBLE, ClaimStatus.CONDITIONALLY_ELIGIBLE}
        if self.status is ClaimStatus.ELIGIBLE:
            if not self.evidence_source_ids or not self.evidence_notes:
                raise ValueError("eligible claims require source-backed evidence")
            if self.blockers:
                raise ValueError("eligible claims cannot retain blockers")
        elif self.status is ClaimStatus.CONDITIONALLY_ELIGIBLE:
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
            and self.status is not ClaimStatus.CONDITIONALLY_ELIGIBLE
        ):
            raise ValueError("quasi-experimental or transported claims must remain conditional")
        if (
            self.claim is ScientificClaim.COUNTERFACTUAL_GENERALIZATION
            and self.status is not ClaimStatus.CONDITIONALLY_ELIGIBLE
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
    use_policies: tuple[DataUsePolicy, ...] = Field(min_length=1)
    experimental_design: ExperimentalDesign
    capabilities: DatasetCapabilities
    claim_eligibility: tuple[ClaimEligibility, ...] = Field(min_length=1)
    notes: tuple[str, ...] = ()

    @property
    def fingerprint(self) -> str:
        # ``canonical_fingerprint`` predates date-valued contracts; JSON mode gives dates and
        # datetimes their canonical ISO representation before applying the shared hash routine.
        return canonical_fingerprint(self.model_dump(mode="json"))

    def permission_status(
        self,
        use_case: DataUseCase,
        *,
        source_ids: tuple[str, ...] | None = None,
    ) -> PermissionStatus:
        """Return the most restrictive status for a use across the selected sources."""

        if source_ids is not None and not source_ids:
            raise ValueError("permission query source IDs must not be empty")
        selected = set(
            tuple(source.source_id for source in self.sources) if source_ids is None else source_ids
        )
        known = {source.source_id for source in self.sources}
        if unknown := selected - known:
            raise ValueError(f"unknown source IDs for permission query: {sorted(unknown)}")
        statuses: list[PermissionStatus] = []
        for policy in self.use_policies:
            if selected & set(policy.source_ids):
                statuses.append(
                    next(
                        permission.status
                        for permission in policy.permissions
                        if permission.use_case is use_case
                    )
                )
        precedence = {
            PermissionStatus.PERMITTED: 0,
            PermissionStatus.CONDITIONAL: 1,
            PermissionStatus.UNKNOWN: 2,
            PermissionStatus.PROHIBITED: 3,
        }
        return max(statuses, key=precedence.__getitem__)

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
        if (
            len(covered_policy_sources) != len(set(covered_policy_sources))
            or set(covered_policy_sources) != known_sources
        ):
            raise ValueError("data-use policies must cover every source exactly once")

        referenced_sources: set[str] = set(self.experimental_design.sampling.source_ids)
        for unit in self.experimental_design.units:
            referenced_sources.update(unit.source_ids)
        for modality in self.capabilities.modalities:
            referenced_sources.update(modality.source_ids)
        referenced_sources.update(self.capabilities.interventions.source_ids)
        referenced_sources.update(self.capabilities.timing.source_ids)
        referenced_sources.update(self.capabilities.lineage.source_ids)
        referenced_sources.update(self.capabilities.spatial.source_ids)
        for output in self.capabilities.functional.outputs:
            referenced_sources.update(output.source_ids)
        for variable in self.capabilities.environment.variables:
            referenced_sources.update(variable.source_ids)
        for eligibility in self.claim_eligibility:
            referenced_sources.update(eligibility.evidence_source_ids)
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
            if modality.alignment_key_field != alignment_unit.id_field:
                raise ValueError("modality alignment key must match its declared unit ID field")
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
        for eligibility in self.claim_eligibility:
            if eligibility.status in {ClaimStatus.ELIGIBLE, ClaimStatus.CONDITIONALLY_ELIGIBLE}:
                reject_documentation_only(
                    eligibility.evidence_source_ids,
                    name="supported claim evidence",
                )

        claims = [eligibility.claim for eligibility in self.claim_eligibility]
        if len(claims) != len(set(claims)):
            raise ValueError("each scientific claim may be assessed only once")
        sampling = self.experimental_design.sampling
        if (
            sampling.time_field is not None
            and not self.capabilities.timing.observation_times_recorded
        ):
            raise ValueError("sampling time field requires observation timing capability")
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
        functional_by_key = {item.output.key: item for item in self.capabilities.functional.outputs}
        environment_by_key = {
            item.variable.key: item for item in self.capabilities.environment.variables
        }
        biological_system_keys = set(_term_keys(self.origin.biological_systems))

        supported = tuple(
            item
            for item in self.claim_eligibility
            if item.status in {ClaimStatus.ELIGIBLE, ClaimStatus.CONDITIONALLY_ELIGIBLE}
        )
        for eligibility in supported:
            scope = eligibility.scope
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
                    set(_term_keys(scope.functional_outputs)),
                    set(functional_by_key),
                    "functional output",
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
            evidence_sources = set(eligibility.evidence_source_ids)

            def require_scoped_evidence(
                source_ids: tuple[str, ...],
                *,
                name: str,
                claim_sources: set[str] = evidence_sources,
            ) -> None:
                if claim_sources.isdisjoint(source_ids):
                    raise ValueError(f"claim evidence does not support its scoped {name}")

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
            for output_key in _term_keys(scope.functional_outputs):
                require_scoped_evidence(
                    functional_by_key[output_key].source_ids,
                    name=f"functional output {output_key}",
                )
            for environment_key in _term_keys(scope.environment_variables):
                require_scoped_evidence(
                    environment_by_key[environment_key].source_ids,
                    name=f"environment variable {environment_key}",
                )
            if scope.horizons_seconds:
                require_scoped_evidence(
                    self.capabilities.timing.source_ids,
                    name="timing",
                )
            if eligibility.claim in {
                ScientificClaim.POPULATION_DYNAMICS,
                ScientificClaim.INDIVIDUAL_LONGITUDINAL_DYNAMICS,
                ScientificClaim.LINEAGE_FATE,
            }:
                require_scoped_evidence(sampling.source_ids, name="sampling structure")
            if eligibility.claim is ScientificClaim.LINEAGE_FATE:
                require_scoped_evidence(
                    self.capabilities.lineage.source_ids,
                    name="lineage",
                )
            if eligibility.claim is ScientificClaim.SPATIAL_CONTEXT:
                require_scoped_evidence(
                    self.capabilities.spatial.source_ids,
                    name="spatial structure",
                )
            self._validate_supported_claim(eligibility, modality_by_key, functional_by_key)
        return self

    def _validate_supported_claim(
        self,
        eligibility: ClaimEligibility,
        modality_by_key: dict[str, ModalitySpec],
        functional_by_key: dict[str, FunctionalReadout],
    ) -> None:
        claim = eligibility.claim
        scope = eligibility.scope
        sampling = self.experimental_design.sampling
        timing = self.capabilities.timing
        scoped_modalities = [modality_by_key[key] for key in _term_keys(scope.modalities)]
        scoped_outputs = [functional_by_key[key] for key in _term_keys(scope.functional_outputs)]

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

        def require_observed_horizon(*, name: str) -> None:
            if not scope.horizons_seconds:
                raise ValueError(f"{name} requires a prediction horizon")
            if scope.inference_cutoff_seconds is None:
                raise ValueError(
                    f"{name} requires a fixed cutoff until field-clock comparability is modeled"
                )
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
            pairing = {
                (modality.alignment_group, modality.alignment_key_field)
                for modality in scoped_modalities
            }
            if len(pairing) != 1:
                raise ValueError(
                    "multimodal fusion modalities must share one alignment group and key"
                )

        if claim is ScientificClaim.POPULATION_DYNAMICS and (
            scope.subject_kind is not SamplingSubjectKind.POPULATION
            or sampling.mode is not SamplingMode.REPEATED_POPULATION_DESTRUCTIVE
            or sampling.linkage is not SubjectLinkage.SAME_POPULATION
            or len(timing.timepoints_seconds) < 2
            or not scope.horizons_seconds
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
            or len(timing.timepoints_seconds) < 2
            or not scope.horizons_seconds
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
                or not scope.horizons_seconds
                or not (scope.modalities or scope.functional_outputs)
                or not intervention.matched_controls_present
                or self.experimental_design.biological_replicate_unit is None
                or self.experimental_design.randomization_unit is None
                or self.experimental_design.matched_control_field is None
                or not intervention.start_stop_recorded
                or not timing.intervention_times_recorded
                or not timing.event_ordering_recorded
            ):
                raise ValueError(
                    "intervention claims require scoped timed exposure, recorded event ordering, "
                    "controls, and replication"
                )
            require_observed_horizon(name="intervention claim")
            if claim is ScientificClaim.RETROSPECTIVE_INTERVENTION_SELECTION and (
                not scope.functional_outputs
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
            or len(timing.timepoints_seconds) < 2
            or not scope.horizons_seconds
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
                not scope.horizons_seconds
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
            for output in scoped_outputs:
                if output.subject_alignment is not expected_output_alignment:
                    raise ValueError(
                        "functional output alignment must match the scoped prediction subject"
                    )
                if (
                    eligibility.status is ClaimStatus.ELIGIBLE
                    and output.status is ReadoutStatus.DERIVED
                ):
                    raise ValueError(
                        "derived functional targets must remain conditionally eligible"
                    )
                if (
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
