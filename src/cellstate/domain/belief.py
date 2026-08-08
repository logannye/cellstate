"""Serializable posterior belief, structured marginals, dynamics, and diagnostics."""

from __future__ import annotations

import math
from enum import StrEnum
from typing import Annotated, Literal, cast
from uuid import UUID, uuid4

import numpy as np
from pydantic import Field, JsonValue, field_validator, model_validator

from .common import (
    SCHEMA_VERSION,
    ArtifactRef,
    EvidenceStatus,
    ProvenanceRecord,
    SchemaModel,
    SchemaVersion,
    SupportStatus,
    require_finite,
)
from .events import InterventionEvent
from .query import StateQuery, Timescale


class DistributionSupport(StrEnum):
    REAL = "real"
    NONNEGATIVE = "nonnegative"
    UNIT_INTERVAL = "unit_interval"
    SIMPLEX = "simplex"
    DISCRETE = "discrete"
    MIXED = "mixed"


class ParametricDistribution(SchemaModel):
    kind: Literal["parametric"] = "parametric"
    family: str = Field(min_length=1)
    dimensions: tuple[str, ...] = Field(min_length=1)
    support: DistributionSupport = DistributionSupport.REAL
    mean: tuple[float, ...]
    covariance: tuple[tuple[float, ...], ...]
    parameters: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def compatible_shapes(self) -> ParametricDistribution:
        size = len(self.dimensions)
        if len(set(self.dimensions)) != size:
            raise ValueError("distribution dimensions must be unique")
        if len(self.mean) != size or len(self.covariance) != size:
            raise ValueError("distribution mean/covariance shape must match dimensions")
        for row in self.covariance:
            if len(row) != size:
                raise ValueError("distribution covariance must be square")
        for value in (*self.mean, *(entry for row in self.covariance for entry in row)):
            require_finite(value, name="distribution parameter")
        for row_index in range(size):
            if self.covariance[row_index][row_index] < 0:
                raise ValueError("covariance diagonal must be nonnegative")
            for column_index in range(size):
                if not math.isclose(
                    self.covariance[row_index][column_index],
                    self.covariance[column_index][row_index],
                    rel_tol=1e-8,
                    abs_tol=1e-10,
                ):
                    raise ValueError("covariance must be symmetric")
        if np.linalg.eigvalsh(np.asarray(self.covariance, dtype=float)).min() < -1e-10:
            raise ValueError("covariance must be positive semidefinite")
        return self


class SampleDistribution(SchemaModel):
    kind: Literal["samples"] = "samples"
    dimensions: tuple[str, ...] = Field(min_length=1)
    support: DistributionSupport = DistributionSupport.REAL
    samples: ArtifactRef
    sample_count: int = Field(gt=0)
    weights: ArtifactRef | None = None

    @model_validator(mode="after")
    def artifacts_match_declared_samples(self) -> SampleDistribution:
        if len(self.dimensions) != len(set(self.dimensions)):
            raise ValueError("sample-distribution dimensions must be unique")
        expected_shape = (self.sample_count, len(self.dimensions))
        if self.samples.shape != expected_shape:
            raise ValueError(
                "posterior sample artifact shape must be (sample_count, state_dimension_count)"
            )
        if self.samples.dimensions != ("sample", "state_dimension"):
            raise ValueError("posterior sample artifact axes must be ('sample', 'state_dimension')")
        if self.weights is not None and (
            self.weights.shape != (self.sample_count,) or self.weights.dimensions != ("sample",)
        ):
            raise ValueError("posterior weight artifact must align one weight per sample")
        return self


class UnavailableDistribution(SchemaModel):
    kind: Literal["unavailable"] = "unavailable"
    dimensions: tuple[str, ...] = ()
    reason_code: str = Field(min_length=1)
    message: str = Field(min_length=1)


StateDistribution = Annotated[
    ParametricDistribution | SampleDistribution | UnavailableDistribution,
    Field(discriminator="kind"),
]


class StateFactor(StrEnum):
    STABLE_IDENTITY = "stable_identity"
    SLOW_MEMORY = "slow_memory"
    REGULATORY = "regulatory"
    SIGNALING = "signaling"
    METABOLIC = "metabolic"
    PHYSICAL = "physical"
    DAMAGE_STRESS = "damage_stress"
    FUNCTIONAL_CAPACITY = "functional_capacity"


class FactorBelief(SchemaModel):
    factor: StateFactor
    timescales: frozenset[Timescale] = Field(min_length=1)
    evidence_status: EvidenceStatus
    posterior: StateDistribution
    evidence_event_ids: tuple[str, ...] = ()
    shared_latent_dimensions: tuple[str, ...] = ()
    modality_private_dimensions: tuple[str, ...] = ()

    @model_validator(mode="after")
    def evidence_semantics(self) -> FactorBelief:
        unavailable = isinstance(self.posterior, UnavailableDistribution)
        if unavailable and self.evidence_status is not EvidenceStatus.UNIDENTIFIABLE:
            raise ValueError("an unavailable factor posterior must be marked unidentifiable")
        if self.evidence_status is EvidenceStatus.OBSERVED and not self.evidence_event_ids:
            raise ValueError("directly observed factors require evidence event IDs")
        return self


class ContextBelief(SchemaModel):
    active_interventions: tuple[InterventionEvent, ...] = ()
    soluble_environment: dict[str, JsonValue] = Field(default_factory=dict)
    physical_environment: dict[str, JsonValue] = Field(default_factory=dict)
    neighborhood: dict[str, JsonValue] = Field(default_factory=dict)
    spatial_position: dict[str, JsonValue] = Field(default_factory=dict)
    unsupported_dimensions: tuple[str, ...] = ()


class EvaluatedScalar(SchemaModel):
    status: SupportStatus
    value: float | None = None
    units: str | None = None
    reason: str | None = None

    @model_validator(mode="after")
    def status_matches_value(self) -> EvaluatedScalar:
        if self.status is SupportStatus.SUPPORTED:
            if self.value is None:
                raise ValueError("a supported scalar requires a value")
            require_finite(self.value, name="evaluated scalar")
        elif self.value is not None:
            raise ValueError("not-evaluated or unsupported scalars must not use numeric sentinels")
        return self


class EventHazard(SchemaModel):
    event: str = Field(min_length=1)
    rate: EvaluatedScalar

    @model_validator(mode="after")
    def nonnegative_rate(self) -> EventHazard:
        if self.rate.value is not None and self.rate.value < 0:
            raise ValueError("event hazards must be nonnegative")
        return self


class FateProbability(SchemaModel):
    fate: str = Field(min_length=1)
    horizon_seconds: float = Field(gt=0)
    probability: EvaluatedScalar

    @model_validator(mode="after")
    def unit_interval_probability(self) -> FateProbability:
        if self.probability.value is not None and not 0 <= self.probability.value <= 1:
            raise ValueError("fate probabilities must lie in [0, 1]")
        return self


class DynamicSummary(SchemaModel):
    velocity: StateDistribution
    stability: EvaluatedScalar
    division_hazard: EvaluatedScalar
    death_hazard: EvaluatedScalar
    transition_hazards: tuple[EventHazard, ...] = ()
    fate_probabilities: tuple[FateProbability, ...] = ()
    bifurcation_proximity: EvaluatedScalar
    recovery_timescale: EvaluatedScalar


class UncertaintyKind(StrEnum):
    MEASUREMENT = "measurement"
    BIOLOGICAL = "biological_stochasticity"
    PARAMETER = "parameter"
    MODEL = "model"
    COUNTERFACTUAL = "counterfactual"


class UncertaintyComponent(SchemaModel):
    kind: UncertaintyKind
    status: SupportStatus
    magnitude: float | None = Field(default=None, ge=0)
    metric: str | None = None
    notes: str | None = None

    @model_validator(mode="after")
    def status_matches_magnitude(self) -> UncertaintyComponent:
        if self.status is SupportStatus.SUPPORTED and self.magnitude is None:
            raise ValueError("supported uncertainty components require a magnitude")
        if self.status is not SupportStatus.SUPPORTED and self.magnitude is not None:
            raise ValueError("unsupported uncertainty must not use a numeric sentinel")
        return self


class UncertaintyBreakdown(SchemaModel):
    components: tuple[UncertaintyComponent, ...]

    @field_validator("components")
    @classmethod
    def exactly_one_of_each_kind(
        cls, components: tuple[UncertaintyComponent, ...]
    ) -> tuple[UncertaintyComponent, ...]:
        kinds = [component.kind for component in components]
        expected = set(UncertaintyKind)
        if set(kinds) != expected or len(kinds) != len(expected):
            raise ValueError("uncertainty must contain each uncertainty kind exactly once")
        return components


class SufficiencyReport(SchemaModel):
    status: SupportStatus
    state_only_loss: float | None = None
    state_plus_history_loss: float | None = None
    history_information_gain: float | None = None
    metric: str | None = None
    notes: tuple[str, ...] = ()

    @model_validator(mode="after")
    def evaluated_together(self) -> SufficiencyReport:
        values = (self.state_only_loss, self.state_plus_history_loss, self.history_information_gain)
        if self.status is SupportStatus.SUPPORTED and any(value is None for value in values):
            raise ValueError(
                "an evaluated sufficiency report requires both losses and information gain"
            )
        if self.status is not SupportStatus.SUPPORTED and any(
            value is not None for value in values
        ):
            raise ValueError("an unevaluated sufficiency report must not contain numeric sentinels")
        if self.status is SupportStatus.SUPPORTED:
            state_only = require_finite(cast(float, self.state_only_loss), name="state-only loss")
            state_plus_history = require_finite(
                cast(float, self.state_plus_history_loss),
                name="state-plus-history loss",
            )
            gain = require_finite(
                cast(float, self.history_information_gain),
                name="history information gain",
            )
            if not math.isclose(
                gain,
                state_only - state_plus_history,
                rel_tol=1e-9,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    "history information gain must equal state-only loss minus "
                    "state-plus-history loss"
                )
        return self


class OODReport(SchemaModel):
    status: SupportStatus
    score: float | None = Field(default=None, ge=0, le=1)
    unsupported_conditions: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    @model_validator(mode="after")
    def status_matches_score(self) -> OODReport:
        if self.status is SupportStatus.SUPPORTED and self.score is None:
            raise ValueError("a supported OOD report requires a score")
        if self.status is not SupportStatus.SUPPORTED and self.score is not None:
            raise ValueError("an unevaluated OOD report must not contain a numeric sentinel")
        return self


class ObservabilityReport(SchemaModel):
    observed: tuple[str, ...] = ()
    inferred_with_support: tuple[str, ...] = ()
    unidentifiable: tuple[str, ...] = ()
    unsupported_by_model: tuple[str, ...] = ()
    out_of_query_scope: tuple[str, ...] = ()

    @model_validator(mode="after")
    def disjoint_categories(self) -> ObservabilityReport:
        groups = [
            set(self.observed),
            set(self.inferred_with_support),
            set(self.unidentifiable),
            set(self.unsupported_by_model),
            set(self.out_of_query_scope),
        ]
        for index, left in enumerate(groups):
            for right in groups[index + 1 :]:
                if left & right:
                    raise ValueError("observability categories must be disjoint")
        return self


class BeliefDiagnostics(SchemaModel):
    ood: OODReport
    sufficiency: SufficiencyReport
    observability: ObservabilityReport
    constraint_residuals: dict[str, float] = Field(default_factory=dict)


class MeasurementRecommendation(SchemaModel):
    status: SupportStatus
    assay_id: str | None = None
    expected_value_of_information: float | None = Field(default=None, ge=0)
    cost: float | None = Field(default=None, ge=0)
    rationale: str | None = None

    @model_validator(mode="after")
    def complete_if_supported(self) -> MeasurementRecommendation:
        values = (self.assay_id, self.expected_value_of_information, self.cost)
        if self.status is SupportStatus.SUPPORTED and any(value is None for value in values):
            raise ValueError(
                "a supported measurement recommendation requires assay, value, and cost"
            )
        if self.status is not SupportStatus.SUPPORTED and any(
            value is not None for value in values
        ):
            raise ValueError("an unavailable recommendation must not contain sentinel values")
        return self


class BeliefStatus(StrEnum):
    COMPLETE = "complete"
    PARTIAL = "partial"
    UNAVAILABLE = "unavailable"


class CellStateBelief(SchemaModel):
    """Query-conditioned posterior over the causally relevant current state."""

    schema_version: SchemaVersion = SCHEMA_VERSION
    belief_id: UUID = Field(default_factory=uuid4)
    subject_id: str = Field(min_length=1)
    as_of_seconds: float
    query: StateQuery
    query_fingerprint: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    history_fingerprint: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    context_fingerprint: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    status: BeliefStatus
    joint_posterior: StateDistribution
    factors: tuple[FactorBelief, ...]
    context: ContextBelief
    dynamics: DynamicSummary
    uncertainty: UncertaintyBreakdown
    diagnostics: BeliefDiagnostics
    next_measurement: MeasurementRecommendation
    provenance: ProvenanceRecord

    @field_validator("as_of_seconds")
    @classmethod
    def finite_as_of(cls, value: float) -> float:
        return require_finite(value, name="belief time")

    @model_validator(mode="after")
    def coherent_status_and_factors(self) -> CellStateBelief:
        factor_names = [factor.factor for factor in self.factors]
        expected = set(StateFactor)
        if set(factor_names) != expected or len(factor_names) != len(expected):
            raise ValueError("a belief must contain each structured state factor exactly once")
        unavailable = isinstance(self.joint_posterior, UnavailableDistribution)
        if self.status is BeliefStatus.UNAVAILABLE and not unavailable:
            raise ValueError("an unavailable belief requires an unavailable joint posterior")
        if self.status is not BeliefStatus.UNAVAILABLE and unavailable:
            raise ValueError("a usable belief requires a joint posterior distribution")
        if self.provenance.query_fingerprint != self.query_fingerprint:
            raise ValueError("provenance/query fingerprints must agree")
        if self.query.fingerprint != self.query_fingerprint:
            raise ValueError("embedded query and query fingerprint must agree")
        if self.provenance.history_fingerprint != self.history_fingerprint:
            raise ValueError("provenance/history fingerprints must agree")
        if self.provenance.context_fingerprint != self.context_fingerprint:
            raise ValueError("provenance/context fingerprints must agree")
        source_event_ids = set(self.provenance.source_event_ids)
        joint_dimensions = set(self.joint_posterior.dimensions)
        for factor in self.factors:
            if not set(factor.evidence_event_ids) <= source_event_ids:
                raise ValueError("factor evidence IDs must appear in belief provenance")
            if (
                not isinstance(factor.posterior, UnavailableDistribution)
                and not set(factor.posterior.dimensions) <= joint_dimensions
            ):
                raise ValueError("factor dimensions must be a subset of the joint posterior")
            if isinstance(self.joint_posterior, ParametricDistribution) and isinstance(
                factor.posterior, ParametricDistribution
            ):
                indices = [
                    self.joint_posterior.dimensions.index(dimension)
                    for dimension in factor.posterior.dimensions
                ]
                expected_mean = np.asarray(self.joint_posterior.mean)[indices]
                expected_covariance = np.asarray(self.joint_posterior.covariance)[
                    np.ix_(indices, indices)
                ]
                if not np.allclose(
                    expected_mean, factor.posterior.mean, atol=1e-8
                ) or not np.allclose(expected_covariance, factor.posterior.covariance, atol=1e-8):
                    raise ValueError("parametric factor posterior must match the joint marginal")
        if self.next_measurement.status is SupportStatus.SUPPORTED:
            candidate_assays = {assay.assay_id for assay in self.query.available_assays}
            if self.next_measurement.assay_id not in candidate_assays:
                raise ValueError("recommended assay must be declared by the state query")
        if self.status is BeliefStatus.COMPLETE:
            incomplete_factors = any(
                isinstance(factor.posterior, UnavailableDistribution)
                or factor.evidence_status is EvidenceStatus.UNIDENTIFIABLE
                for factor in self.factors
            )
            incomplete_uncertainty = any(
                component.status is not SupportStatus.SUPPORTED
                for component in self.uncertainty.components
            )
            incomplete_diagnostics = (
                self.diagnostics.ood.status is not SupportStatus.SUPPORTED
                or self.diagnostics.sufficiency.status is not SupportStatus.SUPPORTED
            )
            if (
                incomplete_factors
                or incomplete_uncertainty
                or incomplete_diagnostics
                or self.context.unsupported_dimensions
            ):
                raise ValueError(
                    "a complete belief cannot contain unsupported or unevaluated state"
                )
        return self
