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
from .subjects import BeliefSubject, IdentityBasis, SubjectKind


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
    ANCESTOR = "ancestor"
    DESCENDANT = "descendant"
    SIBLING = "sibling"
    CLONE_AGGREGATE = "clone_aggregate"
    MATCHED_POPULATION = "matched_population"
    GENERAL_POPULATION = "general_population"
    SPATIAL_NEIGHBOR = "spatial_neighbor"
    EXTERNAL_REFERENCE = "external_reference"


class EvidenceLink(SchemaModel):
    """An explicit, confidence-qualified link from sampled source to belief target."""

    source_subject: BeliefSubject
    target_subject: BeliefSubject
    role: EvidenceRole
    linkage_basis: IdentityBasis
    linkage_confidence: float = Field(gt=0, le=1)
    linkage_details: str = Field(min_length=1)
    sampling_unit_id: str = Field(min_length=1)
    assay_row_id: str | None = Field(default=None, min_length=1)
    randomization_unit_id: str | None = Field(default=None, min_length=1)
    biological_replicate_id: str | None = Field(default=None, min_length=1)

    @field_validator("linkage_confidence")
    @classmethod
    def finite_confidence(cls, value: float) -> float:
        return require_finite(value, name="evidence linkage confidence")

    @model_validator(mode="after")
    def role_has_a_defensible_linkage(self) -> EvidenceLink:
        source = self.source_subject
        target = self.target_subject
        same_subject = source == target

        if self.role is EvidenceRole.DIRECT:
            if not same_subject:
                raise ValueError("direct evidence requires identical source and target subjects")
            if self.linkage_basis not in {
                IdentityBasis.OBSERVED_IDENTITY,
                IdentityBasis.DIRECT_TRACKING,
                IdentityBasis.VIABILITY_PRESERVING_SAMPLING,
                IdentityBasis.DECLARED_MEMBERSHIP,
                IdentityBasis.EXPERIMENTAL_UNIT,
                IdentityBasis.SPATIAL_REGION,
                IdentityBasis.OBSERVED_NEIGHBORHOOD_GRAPH,
            }:
                raise ValueError(
                    "direct evidence requires an observed identity or membership basis"
                )
            if self.linkage_confidence != 1:
                raise ValueError("direct evidence requires linkage confidence 1")
            return self

        if same_subject:
            raise ValueError("non-direct evidence requires distinct source and target subjects")

        lineage_roles = {
            EvidenceRole.ANCESTOR,
            EvidenceRole.DESCENDANT,
            EvidenceRole.SIBLING,
        }
        lineage_bases = {
            IdentityBasis.OBSERVED_PARENTAGE,
            IdentityBasis.HERITABLE_BARCODE,
            IdentityBasis.PHYLOGENY,
            IdentityBasis.PROBABILISTIC_LINEAGE,
        }
        if self.role in lineage_roles:
            if source.kind is not SubjectKind.INDIVIDUAL_CELL:
                raise ValueError("lineage evidence source must be an individual cell")
            if target.kind not in {SubjectKind.INDIVIDUAL_CELL, SubjectKind.CLONE_LINEAGE}:
                raise ValueError("lineage evidence target must be an individual or lineage")
            if self.linkage_basis not in lineage_bases:
                raise ValueError("lineage evidence requires an explicit lineage linkage basis")
        elif self.role is EvidenceRole.CLONE_AGGREGATE:
            if source.kind is not SubjectKind.CLONE_LINEAGE:
                raise ValueError("clone-aggregate evidence requires a clone/lineage source")
            if self.linkage_basis not in lineage_bases:
                raise ValueError("clone-aggregate evidence requires a lineage linkage basis")
        elif self.role is EvidenceRole.MATCHED_POPULATION:
            if source.kind is not SubjectKind.POPULATION:
                raise ValueError("matched-population evidence requires a population source")
            if self.linkage_basis is not IdentityBasis.MATCHED_EXPERIMENTAL_DESIGN:
                raise ValueError("matched-population evidence requires matched-design linkage")
        elif self.role is EvidenceRole.GENERAL_POPULATION:
            if target.kind not in {SubjectKind.POPULATION, SubjectKind.SPATIAL_NICHE}:
                raise ValueError("general-population evidence requires an aggregate target")
            if self.linkage_basis not in {
                IdentityBasis.DECLARED_MEMBERSHIP,
                IdentityBasis.PROBABILISTIC_MEMBERSHIP,
                IdentityBasis.TRANSPORT_ASSUMPTION,
            }:
                raise ValueError("population evidence requires membership or transport linkage")
        elif self.role is EvidenceRole.SPATIAL_NEIGHBOR:
            if self.linkage_basis not in {
                IdentityBasis.SPATIAL_PROXIMITY,
                IdentityBasis.OBSERVED_NEIGHBORHOOD_GRAPH,
            }:
                raise ValueError("spatial-neighbor evidence requires spatial linkage")
        elif self.role is EvidenceRole.EXTERNAL_REFERENCE and self.linkage_basis not in {
            IdentityBasis.EXTERNAL_REFERENCE,
            IdentityBasis.TRANSPORT_ASSUMPTION,
        }:
            raise ValueError("external-reference evidence requires external or transport linkage")
        return self


class CollectionEffect(StrEnum):
    NONDESTRUCTIVE = "nondestructive"
    VIABILITY_PRESERVING_WITH_KNOWN_EFFECT = "viability_preserving_with_known_effect"
    PARTIALLY_DESTRUCTIVE_POPULATION_SAMPLING = "partially_destructive_population_sampling"
    TERMINAL_DESTRUCTIVE = "terminal_destructive"


class ObservationCollection(SchemaModel):
    """Biological effect of collecting one observation from its sampled source."""

    effect: CollectionEffect
    effect_description: str | None = Field(default=None, min_length=1)
    sampling_fraction: float | None = Field(default=None, gt=0, le=1)

    @field_validator("sampling_fraction")
    @classmethod
    def finite_fraction(cls, value: float | None) -> float | None:
        if value is not None:
            return require_finite(value, name="sampling fraction")
        return value

    @model_validator(mode="after")
    def declared_effect_has_required_details(self) -> ObservationCollection:
        if self.effect is CollectionEffect.VIABILITY_PRESERVING_WITH_KNOWN_EFFECT:
            if self.effect_description is None:
                raise ValueError("a known collection effect requires a description")
        elif self.effect_description is not None:
            raise ValueError("effect_description is only valid for a known viability effect")

        if self.effect is CollectionEffect.PARTIALLY_DESTRUCTIVE_POPULATION_SAMPLING:
            if self.sampling_fraction is None:
                raise ValueError("partial population sampling requires a sampling fraction")
        elif self.sampling_fraction is not None:
            raise ValueError("sampling_fraction is only valid for partial population sampling")
        return self


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
    subject: BeliefSubject
    time_seconds: float
    source: str | None = None

    @field_validator("time_seconds")
    @classmethod
    def finite_time(cls, value: float) -> float:
        return require_finite(value, name="event time")

    @property
    def subject_id(self) -> str:
        """Convenience identifier; serialized v2 events retain the full typed subject."""

        return self.subject.subject_id


class ObservationEvent(EventBase):
    kind: Literal["observation"] = "observation"
    modality: OntologyTerm
    evidence_link: EvidenceLink
    collection: ObservationCollection
    duration_seconds: float = Field(ge=0)
    value: ArtifactRef | JsonValue | None
    units: str | None = None
    uncertainty: MeasurementUncertainty = Field(default_factory=MeasurementUncertainty)
    quality: QualityReport = Field(default_factory=QualityReport)
    missingness: MissingnessReport = Field(default_factory=MissingnessReport)
    assay: AssayMetadata

    @field_validator("duration_seconds")
    @classmethod
    def finite_duration(cls, value: float) -> float:
        return require_finite(value, name="observation duration")

    @model_validator(mode="after")
    def missingness_matches_value(self) -> ObservationEvent:
        require_finite(self.end_time_seconds, name="observation end time")
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
        if self.evidence_link.target_subject != self.subject:
            raise ValueError("observation evidence target must equal the event subject")
        if (
            self.collection.effect is CollectionEffect.PARTIALLY_DESTRUCTIVE_POPULATION_SAMPLING
            and self.subject.kind not in {SubjectKind.POPULATION, SubjectKind.SPATIAL_NICHE}
        ):
            raise ValueError("partial population sampling requires a population or niche target")
        if (
            self.collection.effect is CollectionEffect.TERMINAL_DESTRUCTIVE
            and self.evidence_link.role is EvidenceRole.DIRECT
            and self.subject.kind is not SubjectKind.INDIVIDUAL_CELL
        ):
            raise ValueError(
                "terminal direct collection is reserved for an individual sampled entity; "
                "use partial population sampling for an aggregate"
            )
        return self

    @property
    def end_time_seconds(self) -> float:
        return self.time_seconds + self.duration_seconds

    @property
    def evidence_role(self) -> EvidenceRole:
        return self.evidence_link.role

    @property
    def source_subject_id(self) -> str:
        return self.evidence_link.source_subject.subject_id


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


class ScheduleKind(StrEnum):
    SINGLE = "single"
    CONTINUOUS = "continuous"
    REPEATED = "repeated"
    PULSED = "pulsed"


class ReversibilityStatus(StrEnum):
    """What is actually known about reversal after an intervention ends.

    This is deliberately separate from a schedule's washout interval.  Removing an
    exposure does not by itself establish that its biological effects are reversible.
    """

    REVERSIBLE = "reversible"
    IRREVERSIBLE = "irreversible"
    UNKNOWN = "unknown"


class InterventionSchedule(SchemaModel):
    """An explicit realized or intended administration schedule."""

    kind: ScheduleKind
    administration_count: int = Field(ge=1)
    interval_seconds: float | None = Field(default=None, gt=0)
    washout_seconds: float = Field(ge=0)

    @field_validator("interval_seconds", "washout_seconds")
    @classmethod
    def finite_schedule_values(cls, value: float | None) -> float | None:
        if value is not None:
            return require_finite(value, name="intervention schedule value")
        return value

    @model_validator(mode="after")
    def schedule_shape_is_coherent(self) -> InterventionSchedule:
        repeated = self.kind in {ScheduleKind.REPEATED, ScheduleKind.PULSED}
        if repeated:
            if self.administration_count < 2 or self.interval_seconds is None:
                raise ValueError("repeated or pulsed schedules require count >= 2 and an interval")
        elif self.administration_count != 1 or self.interval_seconds is not None:
            raise ValueError("single or continuous schedules require count 1 and no interval")
        return self


class AssignmentMechanism(StrEnum):
    """How exposure was assigned to the experimental unit receiving an action."""

    OBSERVATIONAL = "observational"
    ASSIGNED_NONRANDOM = "assigned_nonrandom"
    QUASI_EXPERIMENTAL = "quasi_experimental"
    RANDOMIZED = "randomized"


class MatchedControl(SchemaModel):
    """Explicit linkage to the control unit used for an assigned intervention."""

    subject_id: str = Field(min_length=1)
    assignment_unit_id: str = Field(min_length=1)
    condition: OntologyTerm
    matching_basis: str = Field(min_length=1)
    contemporaneous: bool


class InterventionEvent(EventBase):
    kind: Literal["intervention"] = "intervention"
    intervention_spec_id: str = Field(min_length=1)
    intervention_type: OntologyTerm
    target: OntologyTerm | None = None
    mechanism: OntologyTerm | None = None
    dose: Quantity
    duration_seconds: float = Field(ge=0)
    schedule: InterventionSchedule
    delivery_method: str = Field(min_length=1)
    estimated_efficiency: float | None = Field(default=None, ge=0, le=1)
    reversibility_status: ReversibilityStatus
    assignment_mechanism: AssignmentMechanism
    assignment_unit_kind: str = Field(min_length=1)
    assignment_unit_id: str = Field(min_length=1)
    randomization_unit_kind: str | None = Field(default=None, min_length=1)
    randomization_unit_id: str | None = Field(default=None, min_length=1)
    matched_control: MatchedControl | None
    actual_perturbation: ActualPerturbation | None = None

    @field_validator("duration_seconds")
    @classmethod
    def finite_duration(cls, value: float) -> float:
        return require_finite(value, name="intervention duration")

    @model_validator(mode="after")
    def interval_and_assignment_are_coherent(self) -> InterventionEvent:
        require_finite(self.time_seconds + self.duration_seconds, name="intervention end time")
        randomization_fields = (self.randomization_unit_kind, self.randomization_unit_id)
        if (randomization_fields[0] is None) is not (randomization_fields[1] is None):
            raise ValueError(
                "randomization unit kind and ID must be declared together or both omitted"
            )
        if (
            self.assignment_mechanism is AssignmentMechanism.RANDOMIZED
            and self.randomization_unit_id is None
        ):
            raise ValueError("randomized assignment requires an explicit randomization unit")
        if self.matched_control is not None:
            if self.matched_control.subject_id == self.subject_id:
                raise ValueError("a matched control must be a distinct biological subject")
            if self.matched_control.assignment_unit_id == self.assignment_unit_id:
                raise ValueError("a matched control must use a distinct assignment unit")
        return self


class EnvironmentTemporalMode(StrEnum):
    FIXED = "fixed"
    PIECEWISE_CONSTANT = "piecewise_constant"
    TIME_VARYING = "time_varying"


class EnvironmentEvent(EventBase):
    kind: Literal["environment"] = "environment"
    variables: dict[str, Quantity | JsonValue] = Field(min_length=1)
    duration_seconds: float = Field(ge=0)
    temporal_mode: EnvironmentTemporalMode
    spatial_region: str | None = None

    @field_validator("duration_seconds")
    @classmethod
    def finite_duration(cls, value: float) -> float:
        return require_finite(value, name="environment duration")

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

    @model_validator(mode="after")
    def interval_is_finite(self) -> EnvironmentEvent:
        require_finite(self.time_seconds + self.duration_seconds, name="environment end time")
        return self


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
