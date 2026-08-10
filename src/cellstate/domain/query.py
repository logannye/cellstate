"""The formal, bounded question that determines what a sufficient state must retain."""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from typing import Annotated, Literal, TypeAlias

from pydantic import Field, JsonValue, field_validator, model_validator

from .common import (
    SCHEMA_VERSION,
    OntologyTerm,
    Quantity,
    SchemaModel,
    SchemaVersion,
    canonical_fingerprint,
    require_finite,
)
from .events import (
    AssignmentMechanism,
    CensoringDirection,
    CollectionEffect,
    EnvironmentEvent,
    EnvironmentTemporalMode,
    EvidenceRole,
    InterventionEvent,
    InterventionSchedule,
    MissingnessStatus,
    ObservationCollection,
    ObservationEvent,
    PerturbationStatus,
    ReversibilityStatus,
    ScheduleKind,
)
from .subjects import SubjectSpecification, TargetAggregation


class SystemBoundary(StrEnum):
    ISOLATED_CELL = "isolated_cell"
    CELL_AND_SOLUBLE_ENVIRONMENT = "cell_and_soluble_environment"
    CELL_AND_NEIGHBORS = "cell_and_neighbors"
    CLONE = "clone"
    POPULATION = "population"
    SPATIAL_TISSUE_NICHE = "spatial_tissue_niche"


class Timescale(StrEnum):
    FAST = "fast"
    INTERMEDIATE = "intermediate"
    SLOW = "slow"


class PredictionHorizon(SchemaModel):
    name: str = Field(min_length=1)
    duration_seconds: float = Field(gt=0)
    timescale: Timescale

    @field_validator("duration_seconds")
    @classmethod
    def finite_duration(cls, value: float) -> float:
        return require_finite(value, name="prediction horizon")


class VersionedReference(SchemaModel):
    """Immutable identity for a protocol, model, or target transformation."""

    reference_id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    fingerprint: str = Field(pattern=r"^[a-fA-F0-9]{64}$")


class FutureAssayObservationEndpoint(SchemaModel):
    kind: Literal["future_assay_observation"] = "future_assay_observation"
    assay_id: str = Field(min_length=1)
    protocol_reference: VersionedReference


class LatentQuantityEndpoint(SchemaModel):
    kind: Literal["latent_quantity"] = "latent_quantity"
    model_reference: VersionedReference


class VersionedTransformEndpoint(SchemaModel):
    kind: Literal["versioned_transform"] = "versioned_transform"
    source_term: OntologyTerm
    transformation_reference: VersionedReference


TargetEndpoint = Annotated[
    FutureAssayObservationEndpoint | LatentQuantityEndpoint | VersionedTransformEndpoint,
    Field(discriminator="kind"),
]


class TargetMissingnessPolicy(StrEnum):
    REQUIRE_OBSERVED = "require_observed"
    MODEL_EXPLICITLY = "model_explicitly"
    EXCLUDE_WITH_DECLARED_REASON = "exclude_with_declared_reason"
    NOT_APPLICABLE = "not_applicable"


class TargetMissingnessSemantics(SchemaModel):
    policy: TargetMissingnessPolicy
    reportable_statuses: tuple[MissingnessStatus, ...]

    @model_validator(mode="after")
    def policy_matches_statuses(self) -> TargetMissingnessSemantics:
        if len(self.reportable_statuses) != len(set(self.reportable_statuses)):
            raise ValueError("target missingness statuses must be unique")
        if self.policy is TargetMissingnessPolicy.NOT_APPLICABLE:
            if self.reportable_statuses:
                raise ValueError("not-applicable target missingness cannot declare statuses")
        elif not self.reportable_statuses:
            raise ValueError("target missingness policy requires explicit reportable statuses")
        if self.policy is TargetMissingnessPolicy.REQUIRE_OBSERVED and self.reportable_statuses != (
            MissingnessStatus.OBSERVED,
        ):
            raise ValueError("require-observed targets may report only observed status")
        return self


class TargetCensoringPolicy(StrEnum):
    REJECT_CENSORED = "reject_censored"
    MODEL_WITH_ASSAY_LIMITS = "model_with_assay_limits"
    MODEL_WITH_RECORDED_BOUNDS = "model_with_recorded_bounds"
    NOT_APPLICABLE = "not_applicable"


class TargetCensoringSemantics(SchemaModel):
    policy: TargetCensoringPolicy
    allowed_directions: tuple[CensoringDirection, ...]

    @model_validator(mode="after")
    def policy_matches_directions(self) -> TargetCensoringSemantics:
        if len(self.allowed_directions) != len(set(self.allowed_directions)):
            raise ValueError("target censoring directions must be unique")
        modeled = self.policy in {
            TargetCensoringPolicy.MODEL_WITH_ASSAY_LIMITS,
            TargetCensoringPolicy.MODEL_WITH_RECORDED_BOUNDS,
        }
        if modeled is not bool(self.allowed_directions):
            raise ValueError(
                "modeled censoring requires directions; rejected/not-applicable censoring forbids "
                "them"
            )
        return self


class OutputSpec(SchemaModel):
    term: OntologyTerm
    units: str = Field(min_length=1)
    aggregation: TargetAggregation
    endpoint: TargetEndpoint
    value_schema_reference: VersionedReference
    missingness: TargetMissingnessSemantics
    censoring: TargetCensoringSemantics
    supported_horizon_names: tuple[str, ...] = Field(min_length=1)
    weight: float = Field(gt=0)
    functional: bool

    @field_validator("weight")
    @classmethod
    def finite_weight(cls, value: float) -> float:
        return require_finite(value, name="output weight")

    @field_validator("supported_horizon_names")
    @classmethod
    def supported_horizons_are_unique(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if len(values) != len(set(values)):
            raise ValueError("target supported horizon names must be unique")
        return values

    @model_validator(mode="after")
    def value_schema_is_separate_from_endpoint(self) -> OutputSpec:
        endpoint_reference = (
            self.endpoint.protocol_reference
            if isinstance(self.endpoint, FutureAssayObservationEndpoint)
            else (
                self.endpoint.model_reference
                if isinstance(self.endpoint, LatentQuantityEndpoint)
                else self.endpoint.transformation_reference
            )
        )
        if self.value_schema_reference.reference_id == endpoint_reference.reference_id:
            raise ValueError(
                "output value-schema reference must be distinct from its endpoint protocol, "
                "model, or transform reference"
            )
        return self


class NumericDomain(SchemaModel):
    """A closed, finite numeric domain with mandatory units."""

    kind: Literal["numeric"] = "numeric"
    minimum: float
    maximum: float
    units: str = Field(min_length=1)

    @field_validator("minimum", "maximum")
    @classmethod
    def finite_bound(cls, value: float) -> float:
        return require_finite(value, name="numeric-domain bound")

    @model_validator(mode="after")
    def bounds_are_ordered(self) -> NumericDomain:
        if self.minimum > self.maximum:
            raise ValueError("numeric-domain minimum must not exceed maximum")
        return self

    def contains(self, value: Quantity) -> bool:
        return value.units == self.units and self.minimum <= value.value <= self.maximum


CategoricalValue: TypeAlias = str | int | float | bool


class CategoricalDomain(SchemaModel):
    """An explicit finite set of permitted categorical values."""

    kind: Literal["categorical"] = "categorical"
    values: tuple[CategoricalValue, ...] = Field(min_length=1)

    @field_validator("values")
    @classmethod
    def finite_unique_values(
        cls, values: tuple[CategoricalValue, ...]
    ) -> tuple[CategoricalValue, ...]:
        identities: list[tuple[type[object], object]] = []
        for value in values:
            if isinstance(value, float):
                require_finite(value, name="categorical-domain value")
            identity = (type(value), value)
            if identity in identities:
                raise ValueError("categorical-domain values must be unique")
            identities.append(identity)
        return values

    def contains(self, value: JsonValue) -> bool:
        return any(
            type(value) is type(candidate) and value == candidate for candidate in self.values
        )


ValueDomain = Annotated[NumericDomain | CategoricalDomain, Field(discriminator="kind")]


class ScalarRange(SchemaModel):
    """A closed finite range for values whose units are fixed by the field name."""

    minimum: float
    maximum: float

    @field_validator("minimum", "maximum")
    @classmethod
    def finite_bound(cls, value: float) -> float:
        return require_finite(value, name="range bound")

    @model_validator(mode="after")
    def bounds_are_ordered(self) -> ScalarRange:
        if self.minimum > self.maximum:
            raise ValueError("range minimum must not exceed maximum")
        return self

    def contains(self, value: float) -> bool:
        return self.minimum <= value <= self.maximum


class IntegerRange(SchemaModel):
    minimum: int = Field(ge=1)
    maximum: int = Field(ge=1)

    @model_validator(mode="after")
    def bounds_are_ordered(self) -> IntegerRange:
        if self.minimum > self.maximum:
            raise ValueError("integer-range minimum must not exceed maximum")
        return self

    def contains(self, value: int) -> bool:
        return self.minimum <= value <= self.maximum


class ScheduleDomain(SchemaModel):
    """Finite support for intervention administration and washout schedules."""

    allowed_kinds: tuple[ScheduleKind, ...] = Field(min_length=1)
    administration_count: IntegerRange
    interval_seconds: ScalarRange | None
    washout_seconds: ScalarRange

    @field_validator("allowed_kinds")
    @classmethod
    def kinds_are_unique(cls, kinds: tuple[ScheduleKind, ...]) -> tuple[ScheduleKind, ...]:
        if len(kinds) != len(set(kinds)):
            raise ValueError("allowed schedule kinds must be unique")
        return kinds

    @model_validator(mode="after")
    def ranges_can_represent_allowed_kinds(self) -> ScheduleDomain:
        repeated = bool({ScheduleKind.REPEATED, ScheduleKind.PULSED} & set(self.allowed_kinds))
        singleton = bool({ScheduleKind.SINGLE, ScheduleKind.CONTINUOUS} & set(self.allowed_kinds))
        if repeated and self.interval_seconds is None:
            raise ValueError("repeated or pulsed support requires an interval range")
        if not repeated and self.interval_seconds is not None:
            raise ValueError("an interval range is invalid without repeated or pulsed support")
        if repeated and self.administration_count.maximum < 2:
            raise ValueError("repeated or pulsed support requires administration count >= 2")
        if singleton and not self.administration_count.contains(1):
            raise ValueError("single or continuous support requires administration count 1")
        return self

    def contains(self, schedule: InterventionSchedule) -> bool:
        if schedule.kind not in self.allowed_kinds:
            return False
        if not self.administration_count.contains(schedule.administration_count):
            return False
        if not self.washout_seconds.contains(schedule.washout_seconds):
            return False
        if schedule.interval_seconds is None:
            return self.interval_seconds is None or schedule.kind in {
                ScheduleKind.SINGLE,
                ScheduleKind.CONTINUOUS,
            }
        return self.interval_seconds is not None and self.interval_seconds.contains(
            schedule.interval_seconds
        )


class RealizationEvidenceRequirement(SchemaModel):
    """Per-action evidence expected for the latent realized perturbation.

    Gaps are support/readiness findings, not malformed-event errors. In particular, UNKNOWN may be
    an allowed status even with a nonzero evidence minimum when observations quantify uncertainty
    without identifying an efficiency.
    """

    allowed_statuses: tuple[PerturbationStatus, ...] = Field(min_length=1)
    allowed_modalities: tuple[OntologyTerm, ...]
    minimum_evidence_events: int = Field(ge=0)

    @field_validator("allowed_statuses")
    @classmethod
    def statuses_are_unique(
        cls, values: tuple[PerturbationStatus, ...]
    ) -> tuple[PerturbationStatus, ...]:
        if len(values) != len(set(values)):
            raise ValueError("realization-evidence statuses must be unique")
        return values

    @field_validator("allowed_modalities")
    @classmethod
    def modalities_are_unique(cls, values: tuple[OntologyTerm, ...]) -> tuple[OntologyTerm, ...]:
        if len({value.key for value in values}) != len(values):
            raise ValueError("realization-evidence modalities must be unique")
        return values

    @model_validator(mode="after")
    def evidence_minimum_has_a_declared_channel(self) -> RealizationEvidenceRequirement:
        if self.minimum_evidence_events > 0 and not self.allowed_modalities:
            raise ValueError("a nonzero realization-evidence minimum requires an allowed modality")
        return self

    def gaps(
        self,
        event: InterventionEvent,
        observations_by_id: Mapping[str, ObservationEvent],
    ) -> tuple[str, ...]:
        """Return deterministic evidence gaps without rejecting uncertain realization state."""

        realization = event.actual_perturbation
        if realization is None:
            return ("realization_not_assessed",)
        gaps: list[str] = []
        if realization.status not in self.allowed_statuses:
            gaps.append(f"unsupported_realization_status:{realization.status.value}")
        evidence_ids = realization.evidence_event_ids
        if len(evidence_ids) < self.minimum_evidence_events:
            gaps.append(
                "insufficient_realization_evidence:"
                f"{len(evidence_ids)}<{self.minimum_evidence_events}"
            )
        allowed_modalities = {modality.key for modality in self.allowed_modalities}
        for evidence_id in evidence_ids:
            observation = observations_by_id.get(evidence_id)
            if observation is None:
                gaps.append(f"unknown_realization_evidence:{evidence_id}")
            elif observation.modality.key not in allowed_modalities:
                gaps.append(
                    f"unsupported_realization_modality:{evidence_id}:{observation.modality.key}"
                )
            elif observation.time_seconds < event.time_seconds:
                gaps.append(f"pre_intervention_realization_evidence:{evidence_id}")
        return tuple(gaps)


class InterventionSpec(SchemaModel):
    """A bounded action family; matching names or units alone never establishes support."""

    spec_id: str = Field(min_length=1)
    kind: OntologyTerm
    target: OntologyTerm | None = None
    mechanisms: tuple[OntologyTerm, ...] = ()
    dose_domain: NumericDomain
    duration_seconds: ScalarRange
    schedule: ScheduleDomain
    delivery_methods: tuple[str, ...] = Field(min_length=1)
    allowed_reversibility_statuses: tuple[ReversibilityStatus, ...] = Field(min_length=1)
    allowed_assignment_mechanisms: tuple[AssignmentMechanism, ...] = Field(min_length=1)
    assignment_unit_kind: str = Field(min_length=1)
    randomization_unit_kind: str | None = Field(default=None, min_length=1)
    require_randomization_unit: bool
    require_matched_control: bool
    realization_evidence: RealizationEvidenceRequirement

    @field_validator("mechanisms")
    @classmethod
    def mechanisms_are_unique(
        cls, mechanisms: tuple[OntologyTerm, ...]
    ) -> tuple[OntologyTerm, ...]:
        if len({mechanism.key for mechanism in mechanisms}) != len(mechanisms):
            raise ValueError("intervention mechanisms must be unique")
        return mechanisms

    @field_validator("delivery_methods")
    @classmethod
    def delivery_methods_are_unique(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        normalized = [value.casefold() for value in values]
        if any(not value for value in values) or len(normalized) != len(set(normalized)):
            raise ValueError("delivery methods must be nonempty and unique case-insensitively")
        return values

    @field_validator("allowed_reversibility_statuses")
    @classmethod
    def reversibility_statuses_are_unique(
        cls, values: tuple[ReversibilityStatus, ...]
    ) -> tuple[ReversibilityStatus, ...]:
        if len(values) != len(set(values)):
            raise ValueError("allowed reversibility statuses must be unique")
        return tuple(status for status in ReversibilityStatus if status in values)

    @field_validator("allowed_assignment_mechanisms")
    @classmethod
    def assignment_mechanisms_are_unique(
        cls, values: tuple[AssignmentMechanism, ...]
    ) -> tuple[AssignmentMechanism, ...]:
        if len(values) != len(set(values)):
            raise ValueError("allowed assignment mechanisms must be unique")
        return values

    @model_validator(mode="after")
    def assignment_requirements_are_coherent(self) -> InterventionSpec:
        if self.require_randomization_unit and self.randomization_unit_kind is None:
            raise ValueError(
                "a required randomization unit must declare its experimental-unit kind"
            )
        if AssignmentMechanism.RANDOMIZED in self.allowed_assignment_mechanisms and (
            not self.require_randomization_unit or self.randomization_unit_kind is None
        ):
            raise ValueError(
                "randomized intervention support requires a declared randomization unit"
            )
        return self

    def contains(self, event: InterventionEvent) -> bool:
        """Return whether a concrete intervention is inside every declared support bound."""

        if (
            event.intervention_spec_id != self.spec_id
            or event.intervention_type.key != self.kind.key
        ):
            return False
        if (event.target.key if event.target else None) != (
            self.target.key if self.target else None
        ):
            return False
        mechanism_key = event.mechanism.key if event.mechanism is not None else None
        declared_mechanisms = {mechanism.key for mechanism in self.mechanisms}
        if mechanism_key not in declared_mechanisms and (
            mechanism_key is not None or self.mechanisms
        ):
            return False
        if event.assignment_mechanism not in self.allowed_assignment_mechanisms:
            return False
        if event.assignment_unit_kind.casefold() != self.assignment_unit_kind.casefold():
            return False
        event_randomization_kind = event.randomization_unit_kind
        if self.require_randomization_unit and event_randomization_kind is None:
            return False
        if event_randomization_kind is not None and (
            self.randomization_unit_kind is None
            or event_randomization_kind.casefold() != self.randomization_unit_kind.casefold()
        ):
            return False
        if self.require_matched_control and event.matched_control is None:
            return False
        return (
            self.dose_domain.contains(event.dose)
            and self.duration_seconds.contains(event.duration_seconds)
            and self.schedule.contains(event.schedule)
            and event.delivery_method.casefold()
            in {method.casefold() for method in self.delivery_methods}
            and event.reversibility_status in self.allowed_reversibility_statuses
        )

    def realization_evidence_gaps(
        self,
        event: InterventionEvent,
        observations_by_id: Mapping[str, ObservationEvent],
    ) -> tuple[str, ...]:
        if not self.contains(event):
            return ("intervention_outside_declared_action_domain",)
        return self.realization_evidence.gaps(event, observations_by_id)


class MissingHistoryPolicy(StrEnum):
    REJECT = "reject"
    REPRESENT_AS_UNKNOWN = "represent_as_unknown"
    USE_DECLARED_DEFAULT = "use_declared_default"


class EnvironmentVariableSpec(SchemaModel):
    variable: OntologyTerm
    domain: ValueDomain
    duration_seconds: ScalarRange
    required: bool
    allowed_temporal_modes: tuple[EnvironmentTemporalMode, ...] = Field(min_length=1)
    missing_history_policy: MissingHistoryPolicy
    default_value: Quantity | CategoricalValue | None = None

    @field_validator("allowed_temporal_modes")
    @classmethod
    def temporal_modes_are_unique(
        cls, values: tuple[EnvironmentTemporalMode, ...]
    ) -> tuple[EnvironmentTemporalMode, ...]:
        if len(values) != len(set(values)):
            raise ValueError("allowed environment temporal modes must be unique")
        return values

    @model_validator(mode="after")
    def default_is_explicit_and_supported(self) -> EnvironmentVariableSpec:
        needs_default = self.missing_history_policy is MissingHistoryPolicy.USE_DECLARED_DEFAULT
        if needs_default != (self.default_value is not None):
            raise ValueError("only use-declared-default policy may carry a default value")
        if self.default_value is not None and not self.contains(self.default_value):
            raise ValueError("environment default value is outside its declared domain")
        return self

    @property
    def units(self) -> str | None:
        return self.domain.units if isinstance(self.domain, NumericDomain) else None

    def contains(self, value: Quantity | JsonValue) -> bool:
        if isinstance(self.domain, NumericDomain):
            if not isinstance(value, Quantity):
                try:
                    value = Quantity.model_validate(value)
                except (TypeError, ValueError):
                    return False
            return self.domain.contains(value)
        if isinstance(value, Quantity):
            return False
        return self.domain.contains(value)

    def contains_event_value(self, event: EnvironmentEvent, key: str) -> bool:
        value = next(
            (
                candidate
                for candidate_key, candidate in event.variables.items()
                if candidate_key.casefold() == key.casefold()
            ),
            None,
        )
        return (
            value is not None
            and self.duration_seconds.contains(event.duration_seconds)
            and event.temporal_mode in self.allowed_temporal_modes
            and self.contains(value)
        )


class PrecisionRequirement(SchemaModel):
    target: OntologyTerm
    horizon_name: str = Field(min_length=1)
    metric: str = Field(min_length=1)
    maximum_error: float = Field(gt=0)
    confidence: float | None = Field(default=None, gt=0, le=1)
    units: str | None = None

    @field_validator("maximum_error", "confidence")
    @classmethod
    def finite_values(cls, value: float | None) -> float | None:
        if value is not None:
            return require_finite(value, name="precision requirement")
        return value


class AssayPurpose(StrEnum):
    """Declared use of an assay inside one query."""

    TARGET_ENDPOINT = "target_endpoint"
    MEASUREMENT_SELECTION = "measurement_selection"


class AssaySpec(SchemaModel):
    assay_id: str = Field(min_length=1)
    modality: OntologyTerm
    protocol_reference: VersionedReference
    collection: ObservationCollection
    purposes: tuple[AssayPurpose, ...] = Field(min_length=1)
    cost: float | None = Field(default=None, ge=0)
    cost_units: str | None = Field(default=None, min_length=1)
    turnaround_seconds: float | None = Field(default=None, ge=0)

    @field_validator("cost", "turnaround_seconds")
    @classmethod
    def finite_values(cls, value: float | None) -> float | None:
        if value is not None:
            return require_finite(value, name="assay cost or turnaround")
        return value

    @field_validator("purposes")
    @classmethod
    def purposes_are_unique(cls, values: tuple[AssayPurpose, ...]) -> tuple[AssayPurpose, ...]:
        if len(values) != len(set(values)):
            raise ValueError("assay purposes must be unique")
        return tuple(purpose for purpose in AssayPurpose if purpose in values)

    @model_validator(mode="after")
    def economics_match_declared_purpose(self) -> AssaySpec:
        economics = (self.cost, self.cost_units, self.turnaround_seconds)
        used_for_selection = AssayPurpose.MEASUREMENT_SELECTION in self.purposes
        if used_for_selection and any(value is None for value in economics):
            raise ValueError(
                "measurement-selection assays require explicit cost, cost units, and turnaround"
            )
        if not used_for_selection and any(value is not None for value in economics):
            raise ValueError(
                "target-only assays must omit cost, cost units, and turnaround economics"
            )
        return self


class EvidencePolicy(SchemaModel):
    """Cutoff-safe observation channels admissible for state estimation."""

    lookback_seconds: float | None = Field(default=None, ge=0)
    include_at_cutoff: bool
    allowed_modalities: tuple[OntologyTerm, ...] = Field(min_length=1)
    allowed_evidence_roles: tuple[EvidenceRole, ...] = Field(min_length=1)
    minimum_observed_measurements: int = Field(ge=0)

    @field_validator("lookback_seconds")
    @classmethod
    def finite_lookback(cls, value: float | None) -> float | None:
        if value is not None:
            return require_finite(value, name="evidence lookback")
        return value

    @field_validator("allowed_modalities")
    @classmethod
    def modalities_are_unique(
        cls, modalities: tuple[OntologyTerm, ...]
    ) -> tuple[OntologyTerm, ...]:
        if len({modality.key for modality in modalities}) != len(modalities):
            raise ValueError("allowed evidence modalities must be unique")
        return modalities

    @field_validator("allowed_evidence_roles")
    @classmethod
    def evidence_roles_are_unique(cls, roles: tuple[EvidenceRole, ...]) -> tuple[EvidenceRole, ...]:
        if len(roles) != len(set(roles)):
            raise ValueError("allowed evidence roles must be unique")
        return roles


class AcceptanceThresholds(SchemaModel):
    """Numerical scientific-validity gates compiled with a query."""

    maximum_ood_score: float = Field(ge=0, le=1)
    maximum_history_information_gain: float = Field(ge=0)
    minimum_calibration_coverage: float = Field(gt=0, le=1)
    maximum_calibration_error: float = Field(ge=0)
    maximum_counterfactual_uncertainty: float = Field(ge=0)
    maximum_decision_uncertainty: float = Field(ge=0)
    minimum_identifiability: float = Field(gt=0, le=1)

    @field_validator(
        "maximum_ood_score",
        "maximum_history_information_gain",
        "minimum_calibration_coverage",
        "maximum_calibration_error",
        "maximum_counterfactual_uncertainty",
        "maximum_decision_uncertainty",
        "minimum_identifiability",
    )
    @classmethod
    def finite_threshold(cls, value: float) -> float:
        return require_finite(value, name="query acceptance threshold")


class ConstraintCategory(StrEnum):
    SAFETY = "safety"
    MANUFACTURING = "manufacturing"
    FEASIBILITY = "feasibility"


class ConstraintStrength(StrEnum):
    HARD = "hard"
    SOFT = "soft"


class ConstraintSpec(SchemaModel):
    constraint_id: str = Field(min_length=1)
    category: ConstraintCategory
    description: str = Field(min_length=1)
    strength: ConstraintStrength


class QueryConstraints(SchemaModel):
    """Non-metric constraints whose violation requires abstention."""

    maximum_intervention_combination_order: int = Field(ge=1)
    allowed_combinations: tuple[tuple[str, ...], ...] = ()
    forbidden_combinations: tuple[tuple[str, ...], ...] = ()
    require_complete_intervention_history: bool
    require_complete_environment_history: bool
    require_complete_lineage_history: bool
    require_complete_neighborhood_history: bool
    allow_transport: bool
    maximum_total_assay_cost: float | None = Field(default=None, ge=0)
    assay_cost_units: str | None = Field(default=None, min_length=1)
    maximum_assay_delay_seconds: float | None = Field(default=None, ge=0)
    named_constraints: tuple[ConstraintSpec, ...] = ()

    @field_validator("allowed_combinations", "forbidden_combinations")
    @classmethod
    def combinations_are_canonical(
        cls, combinations: tuple[tuple[str, ...], ...]
    ) -> tuple[tuple[str, ...], ...]:
        canonical: list[tuple[str, ...]] = []
        for combination in combinations:
            if len(combination) < 2:
                raise ValueError("an intervention combination must contain at least two specs")
            if any(not spec_id for spec_id in combination):
                raise ValueError("intervention combination spec IDs must be nonempty")
            normalized = tuple(sorted(combination))
            if len(normalized) != len(set(normalized)):
                raise ValueError("an intervention combination cannot repeat a spec ID")
            if normalized in canonical:
                raise ValueError("intervention combinations must be unique")
            canonical.append(normalized)
        return tuple(sorted(canonical))

    @field_validator("named_constraints")
    @classmethod
    def named_constraint_ids_are_unique(
        cls, constraints: tuple[ConstraintSpec, ...]
    ) -> tuple[ConstraintSpec, ...]:
        ids = [constraint.constraint_id for constraint in constraints]
        if len(ids) != len(set(ids)):
            raise ValueError("named constraint IDs must be unique")
        return constraints

    @field_validator("maximum_total_assay_cost", "maximum_assay_delay_seconds")
    @classmethod
    def finite_limits(cls, value: float | None) -> float | None:
        if value is not None:
            return require_finite(value, name="query constraint")
        return value

    @model_validator(mode="after")
    def combination_rules_are_consistent(self) -> QueryConstraints:
        assay_economics = (
            self.maximum_total_assay_cost,
            self.assay_cost_units,
            self.maximum_assay_delay_seconds,
        )
        if any(value is None for value in assay_economics) and any(
            value is not None for value in assay_economics
        ):
            raise ValueError(
                "query assay budget, cost units, and delay limit must be declared together or "
                "all omitted"
            )
        overlap = set(self.allowed_combinations) & set(self.forbidden_combinations)
        if overlap:
            raise ValueError("an intervention combination cannot be both allowed and forbidden")
        if any(
            len(combination) > self.maximum_intervention_combination_order
            for combination in (*self.allowed_combinations, *self.forbidden_combinations)
        ):
            raise ValueError("combination rule exceeds maximum intervention combination order")
        return self


class StateQuery(SchemaModel):
    """A task-specific definition of predictive and interventional sufficiency."""

    schema_version: SchemaVersion = SCHEMA_VERSION
    subject: SubjectSpecification
    system_boundary: SystemBoundary
    temporal_resolution_seconds: float = Field(gt=0)
    prediction_horizons: tuple[PredictionHorizon, ...] = Field(min_length=1)
    target_outputs: tuple[OutputSpec, ...] = Field(min_length=1)
    intervention_space: tuple[InterventionSpec, ...] = ()
    environment_space: tuple[EnvironmentVariableSpec, ...] = ()
    precision_requirements: tuple[PrecisionRequirement, ...] = ()
    available_assays: tuple[AssaySpec, ...] = ()
    evidence_policy: EvidencePolicy
    acceptance_thresholds: AcceptanceThresholds
    constraints: QueryConstraints

    @field_validator("temporal_resolution_seconds")
    @classmethod
    def finite_temporal_resolution(cls, value: float) -> float:
        return require_finite(value, name="query temporal resolution")

    @field_validator("prediction_horizons")
    @classmethod
    def unique_horizons(
        cls, horizons: tuple[PredictionHorizon, ...]
    ) -> tuple[PredictionHorizon, ...]:
        names = [horizon.name for horizon in horizons]
        if len(names) != len(set(names)):
            raise ValueError("prediction horizon names must be unique")
        return tuple(sorted(horizons, key=lambda item: item.duration_seconds))

    @model_validator(mode="after")
    def references_and_members_are_unique(self) -> StateQuery:
        if any(
            horizon.duration_seconds < self.temporal_resolution_seconds
            for horizon in self.prediction_horizons
        ):
            raise ValueError("query temporal resolution cannot exceed a prediction horizon")
        target_keys = [output.term.key for output in self.target_outputs]
        if len(target_keys) != len(set(target_keys)):
            raise ValueError("target outputs must be unique")
        for output in self.target_outputs:
            if output.aggregation.subject_kind is not self.subject.kind:
                raise ValueError("target aggregation must match the query subject kind")
            if output.aggregation.experimental_unit != self.subject.experimental_unit_kind:
                raise ValueError(
                    "target aggregation experimental unit must match the query subject unit kind"
                )

        spec_ids = [item.spec_id for item in self.intervention_space]
        if len(spec_ids) != len(set(spec_ids)):
            raise ValueError("intervention specification IDs must be unique")
        known_spec_ids = set(spec_ids)
        referenced_spec_ids = {
            spec_id
            for combination in (
                *self.constraints.allowed_combinations,
                *self.constraints.forbidden_combinations,
            )
            for spec_id in combination
        }
        if unknown_spec_ids := referenced_spec_ids - known_spec_ids:
            raise ValueError(
                "intervention combination references undeclared specification IDs: "
                f"{sorted(unknown_spec_ids)}"
            )

        environment_keys = [item.variable.key for item in self.environment_space]
        if len(environment_keys) != len(set(environment_keys)):
            raise ValueError("environment variable specifications must be unique")
        assay_ids = [assay.assay_id for assay in self.available_assays]
        if len(assay_ids) != len(set(assay_ids)):
            raise ValueError("available assay IDs must be unique")
        known_assay_ids = set(assay_ids)
        measurement_selection_assays = tuple(
            assay
            for assay in self.available_assays
            if AssayPurpose.MEASUREMENT_SELECTION in assay.purposes
        )
        assay_budget_declared = self.constraints.maximum_total_assay_cost is not None
        if bool(measurement_selection_assays) != assay_budget_declared:
            if measurement_selection_assays:
                raise ValueError(
                    "queries with measurement-selection assays require explicit assay budget, "
                    "cost units, and delay limit"
                )
            raise ValueError(
                "queries without measurement-selection assays must omit assay budget, cost "
                "units, and delay limit"
            )

        for assay in self.available_assays:
            if AssayPurpose.MEASUREMENT_SELECTION in assay.purposes:
                assert assay.cost is not None
                assert assay.cost_units is not None
                assert assay.turnaround_seconds is not None
                assert self.constraints.maximum_total_assay_cost is not None
                assert self.constraints.assay_cost_units is not None
                assert self.constraints.maximum_assay_delay_seconds is not None
                if assay.cost_units != self.constraints.assay_cost_units:
                    raise ValueError("assay cost units must match the query budget units")
                if assay.cost > self.constraints.maximum_total_assay_cost:
                    raise ValueError("an available assay exceeds the query cost constraint")
                if assay.turnaround_seconds > self.constraints.maximum_assay_delay_seconds:
                    raise ValueError("an available assay exceeds the query delay constraint")
            if (
                assay.collection.effect
                is CollectionEffect.PARTIALLY_DESTRUCTIVE_POPULATION_SAMPLING
                and self.subject.kind.value not in {"population", "spatial_niche"}
            ):
                raise ValueError("partial population assays require an aggregate query subject")

        horizon_names = {horizon.name for horizon in self.prediction_horizons}
        target_endpoint_assay_ids: set[str] = set()
        assays_by_id = {assay.assay_id: assay for assay in self.available_assays}
        for output in self.target_outputs:
            if unknown_horizons := set(output.supported_horizon_names) - horizon_names:
                raise ValueError(
                    f"target {output.term.key!r} references undeclared supported horizons: "
                    f"{sorted(unknown_horizons)}"
                )
            if (
                isinstance(output.endpoint, FutureAssayObservationEndpoint)
                and output.endpoint.assay_id not in known_assay_ids
            ):
                raise ValueError(
                    f"target {output.term.key!r} references an undeclared future assay"
                )
            if isinstance(output.endpoint, FutureAssayObservationEndpoint):
                target_assay = assays_by_id[output.endpoint.assay_id]
                if AssayPurpose.TARGET_ENDPOINT not in target_assay.purposes:
                    raise ValueError(
                        f"target {output.term.key!r} future assay must declare target-endpoint "
                        "purpose"
                    )
                if output.endpoint.protocol_reference != target_assay.protocol_reference:
                    raise ValueError(
                        f"target {output.term.key!r} future assay protocol does not match its "
                        "declared assay specification"
                    )
                target_endpoint_assay_ids.add(target_assay.assay_id)
        unused_target_assays = {
            assay.assay_id
            for assay in self.available_assays
            if AssayPurpose.TARGET_ENDPOINT in assay.purposes
        } - target_endpoint_assay_ids
        if unused_target_assays:
            raise ValueError(
                "target-endpoint assay purposes must be referenced by a query target: "
                f"{sorted(unused_target_assays)}"
            )
        precision_keys: list[tuple[str, str, str]] = []
        for requirement in self.precision_requirements:
            if requirement.target.key not in target_keys:
                raise ValueError("precision requirement references an undeclared target")
            if requirement.horizon_name not in horizon_names:
                raise ValueError("precision requirement references an undeclared horizon")
            target = next(
                output
                for output in self.target_outputs
                if output.term.key == requirement.target.key
            )
            if requirement.horizon_name not in target.supported_horizon_names:
                raise ValueError(
                    "precision requirement references a horizon unsupported by its target"
                )
            if requirement.units is not None and requirement.units != target.units:
                raise ValueError("precision requirement units must match target units")
            precision_keys.append(
                (requirement.target.key, requirement.horizon_name, requirement.metric)
            )
        if len(precision_keys) != len(set(precision_keys)):
            raise ValueError(
                "precision requirements must be unique per target, horizon, and metric"
            )
        return self

    def matching_intervention_specs(self, event: InterventionEvent) -> tuple[InterventionSpec, ...]:
        """Return query action specifications that contain the complete event."""

        return tuple(spec for spec in self.intervention_space if spec.contains(event))

    def contains_intervention(self, event: InterventionEvent) -> bool:
        return len(self.matching_intervention_specs(event)) == 1

    def realization_evidence_gaps(
        self,
        event: InterventionEvent,
        observations_by_id: Mapping[str, ObservationEvent],
    ) -> tuple[str, ...]:
        """Assess per-action realization evidence without making uncertainty invalid input."""

        matching = self.matching_intervention_specs(event)
        if len(matching) != 1:
            return ("intervention_outside_declared_action_domain",)
        return matching[0].realization_evidence.gaps(event, observations_by_id)

    def contains_intervention_combination(self, events: tuple[InterventionEvent, ...]) -> bool:
        """Check members plus explicit allowed/forbidden combination policy.

        An empty ``allowed_combinations`` collection permits any supported combination up to the
        declared maximum order, except combinations listed as forbidden.
        """

        if not events or len(events) > self.constraints.maximum_intervention_combination_order:
            return False
        if not all(self.contains_intervention(event) for event in events):
            return False
        spec_ids = tuple(sorted(event.intervention_spec_id for event in events))
        if len(spec_ids) != len(set(spec_ids)):
            return False
        if len(spec_ids) == 1:
            return True
        if spec_ids in self.constraints.forbidden_combinations:
            return False
        allowed = self.constraints.allowed_combinations
        return not allowed or spec_ids in allowed

    def environment_spec(self, key: str) -> EnvironmentVariableSpec | None:
        normalized = key.casefold()
        return next(
            (spec for spec in self.environment_space if spec.variable.key.casefold() == normalized),
            None,
        )

    def contains_environment_value(self, key: str, value: Quantity | JsonValue) -> bool:
        spec = self.environment_spec(key)
        return spec is not None and spec.contains(value)

    def contains_environment_event(self, event: EnvironmentEvent) -> bool:
        return not any(
            (spec := self.environment_spec(key)) is None
            or not spec.contains_event_value(event, key)
            for key in event.variables
        )

    def contains_environment_events(self, events: tuple[EnvironmentEvent, ...]) -> bool:
        """Check event domains and coverage of required environment variables."""

        if not all(self.contains_environment_event(event) for event in events):
            return False
        supplied = {key.casefold() for event in events for key in event.variables}
        required = {
            spec.variable.key.casefold() for spec in self.environment_space if spec.required
        }
        return required <= supplied

    @property
    def fingerprint(self) -> str:
        return canonical_fingerprint(self)
