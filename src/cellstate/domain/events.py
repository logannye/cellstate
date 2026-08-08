"""Typed, time-stamped records used to reconstruct a causal cellular history."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import Field, JsonValue, field_validator, model_validator

from .common import (
    SCHEMA_VERSION,
    ArtifactRef,
    OntologyTerm,
    Quantity,
    SchemaModel,
    SchemaVersion,
    require_finite,
)


class MissingnessStatus(StrEnum):
    OBSERVED = "observed"
    NOT_MEASURED = "not_measured"
    MISSING = "missing"
    BELOW_DETECTION = "below_detection"
    CENSORED = "censored"
    ASSAY_FAILURE = "assay_failure"


class CensoringDirection(StrEnum):
    BELOW = "below"
    ABOVE = "above"
    INTERVAL = "interval"


class EvidenceRole(StrEnum):
    DIRECT = "direct"
    SIBLING = "sibling"
    ANCESTOR = "ancestor"
    POPULATION = "population"


class MissingnessReport(SchemaModel):
    status: MissingnessStatus = MissingnessStatus.OBSERVED
    reason: str | None = None
    detection_limit: Quantity | None = None
    censoring_direction: CensoringDirection | None = None
    interval_lower: Quantity | None = None
    interval_upper: Quantity | None = None

    @model_validator(mode="after")
    def censoring_has_explicit_bounds(self) -> MissingnessReport:
        interval_bounds = (self.interval_lower, self.interval_upper)
        if self.status is MissingnessStatus.BELOW_DETECTION:
            if self.detection_limit is None:
                raise ValueError("a below-detection record requires a detection limit")
            if self.censoring_direction is not None or any(
                bound is not None for bound in interval_bounds
            ):
                raise ValueError("below-detection status already defines its censoring direction")
        elif self.status is MissingnessStatus.CENSORED:
            if self.censoring_direction is None:
                raise ValueError("a censored record requires a censoring direction")
            if self.censoring_direction in {
                CensoringDirection.BELOW,
                CensoringDirection.ABOVE,
            }:
                if self.detection_limit is None:
                    raise ValueError("one-sided censoring requires a detection limit")
                if any(bound is not None for bound in interval_bounds):
                    raise ValueError("one-sided censoring must not also carry interval bounds")
            else:
                if self.detection_limit is not None:
                    raise ValueError("interval censoring must use explicit lower and upper bounds")
                if self.interval_lower is None or self.interval_upper is None:
                    raise ValueError("interval censoring requires lower and upper bounds")
                if self.interval_lower.units != self.interval_upper.units:
                    raise ValueError("interval censoring bounds must use the same units")
                if self.interval_lower.value >= self.interval_upper.value:
                    raise ValueError("interval censoring lower bound must be below its upper bound")
        elif (
            self.detection_limit is not None
            or self.censoring_direction is not None
            or any(bound is not None for bound in interval_bounds)
        ):
            raise ValueError("censoring bounds are only valid for censored observations")
        return self


class MeasurementUncertainty(SchemaModel):
    distribution: str = "unspecified"
    standard_error: float | None = Field(default=None, ge=0)
    parameters: dict[str, JsonValue] = Field(default_factory=dict)


class QualityReport(SchemaModel):
    score: float = Field(default=1.0, ge=0, le=1)
    flags: tuple[str, ...] = ()


class AssayMetadata(SchemaModel):
    assay_id: str = Field(min_length=1)
    batch: str | None = None
    instrument: str | None = None
    operator: str | None = None
    library_size: int | None = Field(default=None, ge=0)
    capture_efficiency: float | None = Field(default=None, ge=0, le=1)
    staining_panel: str | None = None
    segmentation_confidence: float | None = Field(default=None, ge=0, le=1)
    plate_position: str | None = None
    processing_delay_seconds: float | None = Field(default=None, ge=0)
    detection_limits: dict[str, Quantity] = Field(default_factory=dict)
    attributes: dict[str, JsonValue] = Field(default_factory=dict)


class EventBase(SchemaModel):
    schema_version: SchemaVersion = SCHEMA_VERSION
    event_id: str = Field(min_length=1)
    subject_id: str = Field(min_length=1)
    time_seconds: float
    source: str | None = None

    @field_validator("time_seconds")
    @classmethod
    def finite_time(cls, value: float) -> float:
        return require_finite(value, name="event time")


class ObservationEvent(EventBase):
    kind: Literal["observation"] = "observation"
    modality: OntologyTerm
    evidence_role: EvidenceRole = EvidenceRole.DIRECT
    source_subject_id: str | None = None
    value: ArtifactRef | JsonValue | None
    units: str | None = None
    uncertainty: MeasurementUncertainty = Field(default_factory=MeasurementUncertainty)
    quality: QualityReport = Field(default_factory=QualityReport)
    missingness: MissingnessReport = Field(default_factory=MissingnessReport)
    assay: AssayMetadata

    @model_validator(mode="after")
    def missingness_matches_value(self) -> ObservationEvent:
        if self.missingness.status is MissingnessStatus.OBSERVED and self.value is None:
            raise ValueError("an observed measurement must carry a value; zero is a valid value")
        if self.missingness.status is not MissingnessStatus.OBSERVED and self.value is not None:
            raise ValueError(
                "a missing, failed, or censored measurement must encode evidence in its status "
                "and bounds, not as an imputed value"
            )
        bounds = (
            self.missingness.detection_limit,
            self.missingness.interval_lower,
            self.missingness.interval_upper,
        )
        if self.units is not None and any(
            bound is not None and bound.units != self.units for bound in bounds
        ):
            raise ValueError("observation units and censoring-bound units must agree")
        if self.evidence_role is EvidenceRole.DIRECT:
            if self.source_subject_id not in {None, self.subject_id}:
                raise ValueError("direct evidence must come from the history subject")
        elif self.source_subject_id is None:
            raise ValueError("lineage or population evidence requires source_subject_id")
        return self


class PerturbationStatus(StrEnum):
    UNKNOWN = "unknown"
    INFERRED = "inferred"
    MEASURED = "measured"
    FAILED = "failed"


class ActualPerturbation(SchemaModel):
    status: PerturbationStatus
    efficiency: float | None = Field(default=None, ge=0, le=1)
    evidence_event_ids: tuple[str, ...] = ()

    @model_validator(mode="after")
    def realization_is_coherent(self) -> ActualPerturbation:
        if self.status in {PerturbationStatus.MEASURED, PerturbationStatus.INFERRED}:
            if self.efficiency is None:
                raise ValueError("measured or inferred perturbation requires efficiency")
            if not self.evidence_event_ids:
                raise ValueError("measured or inferred perturbation requires evidence event IDs")
        elif self.status is PerturbationStatus.FAILED:
            if self.efficiency != 0:
                raise ValueError("a failed perturbation must have zero efficiency")
            if not self.evidence_event_ids:
                raise ValueError("a failed perturbation requires evidence event IDs")
        elif self.efficiency is not None:
            raise ValueError("an unknown perturbation realization cannot claim an efficiency")
        if len(self.evidence_event_ids) != len(set(self.evidence_event_ids)):
            raise ValueError("perturbation evidence event IDs must be unique")
        return self


class InterventionEvent(EventBase):
    kind: Literal["intervention"] = "intervention"
    intervention_type: OntologyTerm
    target: OntologyTerm | None = None
    mechanism: OntologyTerm | None = None
    dose: Quantity | None = None
    duration_seconds: float = Field(default=0, ge=0)
    delivery_method: str | None = None
    estimated_efficiency: float | None = Field(default=None, ge=0, le=1)
    reversible: bool | None = None
    actual_perturbation: ActualPerturbation | None = None

    @field_validator("duration_seconds")
    @classmethod
    def finite_duration(cls, value: float) -> float:
        return require_finite(value, name="intervention duration")


class EnvironmentEvent(EventBase):
    kind: Literal["environment"] = "environment"
    variables: dict[str, Quantity | JsonValue]
    spatial_region: str | None = None

    @field_validator("variables")
    @classmethod
    def canonical_variable_keys(
        cls, variables: dict[str, Quantity | JsonValue]
    ) -> dict[str, Quantity | JsonValue]:
        canonical: dict[str, Quantity | JsonValue] = {}
        for key, value in variables.items():
            if not key:
                raise ValueError("environment variable keys must be nonempty")
            normalized = key.casefold()
            if normalized in canonical:
                raise ValueError("environment variable keys must be unique case-insensitively")
            canonical[normalized] = value
        return canonical


class DivisionEvent(EventBase):
    kind: Literal["division"] = "division"
    child_ids: tuple[str, ...] = Field(min_length=2)
    partition_metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("child_ids")
    @classmethod
    def unique_children(cls, child_ids: tuple[str, ...]) -> tuple[str, ...]:
        if len(child_ids) != len(set(child_ids)):
            raise ValueError("division child IDs must be unique")
        return child_ids

    @model_validator(mode="after")
    def parent_is_not_its_own_child(self) -> DivisionEvent:
        if self.subject_id in self.child_ids:
            raise ValueError("a division parent cannot also be one of its children")
        return self


class ContactEvent(EventBase):
    kind: Literal["contact"] = "contact"
    other_subject_id: str = Field(min_length=1)
    duration_seconds: float = Field(default=0, ge=0)
    distance: Quantity | None = None
    contact_area: Quantity | None = None
    communication_capacity: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def physical_values_are_coherent(self) -> ContactEvent:
        if self.other_subject_id == self.subject_id:
            raise ValueError("a contact event requires a distinct other subject")
        require_finite(self.duration_seconds, name="contact duration")
        if self.communication_capacity is not None:
            require_finite(self.communication_capacity, name="communication capacity")
        for name, quantity in (("distance", self.distance), ("contact area", self.contact_area)):
            if quantity is not None and quantity.value < 0:
                raise ValueError(f"{name} must be nonnegative")
        return self


CellEvent = Annotated[
    ObservationEvent | InterventionEvent | EnvironmentEvent | DivisionEvent | ContactEvent,
    Field(discriminator="kind"),
]


class EngineeredConstruct(SchemaModel):
    construct_id: str = Field(min_length=1)
    sequence_ref: ArtifactRef | None = None
    copy_number: float | None = Field(default=None, ge=0)
    integration_site: str | None = None
    promoter_configuration: str | None = None
    expression_status: str | None = None


class StaticContext(SchemaModel):
    species: OntologyTerm
    donor_id: str | None = None
    germline_genotype: ArtifactRef | None = None
    somatic_genotype: ArtifactRef | None = None
    copy_number: ArtifactRef | None = None
    sex: str | None = None
    age: Quantity | None = None
    tissue_origin: OntologyTerm | None = None
    disease_context: tuple[OntologyTerm, ...] = ()
    engineered_constructs: tuple[EngineeredConstruct, ...] = ()
    clone_id: str | None = None
    manufacturing_provenance: str | None = None
    attributes: dict[str, JsonValue] = Field(default_factory=dict)


class LineageHistory(SchemaModel):
    parent_cell_id: str | None = None
    division_times_seconds: tuple[float, ...] = ()
    sibling_cell_ids: tuple[str, ...] = ()
    clone_id: str | None = None
    lineage_barcode: str | None = None
    inherited_measurement_ids: tuple[str, ...] = ()
    generation_number: int | None = Field(default=None, ge=0)

    @field_validator("division_times_seconds")
    @classmethod
    def sorted_finite_divisions(cls, times: tuple[float, ...]) -> tuple[float, ...]:
        for value in times:
            require_finite(value, name="division time")
        return tuple(sorted(times))


class SpatialNode(SchemaModel):
    node_id: str = Field(min_length=1)
    node_type: str = Field(min_length=1)
    belief_id: str | None = None
    features: dict[str, JsonValue] = Field(default_factory=dict)


class SpatialEdge(SchemaModel):
    source_id: str = Field(min_length=1)
    target_id: str = Field(min_length=1)
    relationship: str = Field(min_length=1)
    distance: Quantity | None = None
    contact_area: Quantity | None = None
    contact_duration: Quantity | None = None
    communication_capacity: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def physical_values_are_coherent(self) -> SpatialEdge:
        if self.source_id == self.target_id:
            raise ValueError("spatial edges cannot be self-edges")
        if self.communication_capacity is not None:
            require_finite(self.communication_capacity, name="communication capacity")
        for name, quantity in (
            ("distance", self.distance),
            ("contact area", self.contact_area),
            ("contact duration", self.contact_duration),
        ):
            if quantity is not None and quantity.value < 0:
                raise ValueError(f"{name} must be nonnegative")
        return self


class SpatialGraph(SchemaModel):
    nodes: tuple[SpatialNode, ...] = ()
    edges: tuple[SpatialEdge, ...] = ()

    @model_validator(mode="after")
    def graph_references_known_nodes(self) -> SpatialGraph:
        node_ids = [node.node_id for node in self.nodes]
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("spatial node IDs must be unique")
        known = set(node_ids)
        for edge in self.edges:
            if edge.source_id not in known or edge.target_id not in known:
                raise ValueError("spatial edges must reference declared nodes")
        return self


class PopulationContext(SchemaModel):
    same_sample_subject_ids: tuple[str, ...] = ()
    biological_replicate_ids: tuple[str, ...] = ()
    sibling_subject_ids: tuple[str, ...] = ()
    matched_control_ids: tuple[str, ...] = ()
    reference_atlas_ids: tuple[str, ...] = ()
    prior_experiment_ids: tuple[str, ...] = ()
    donor_matched_subject_ids: tuple[str, ...] = ()
    attributes: dict[str, JsonValue] = Field(default_factory=dict)
