"""Compiled, query-specific definition of the active latent state."""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field, field_validator, model_validator

from .common import (
    SCHEMA_VERSION,
    SchemaModel,
    SchemaVersion,
    canonical_fingerprint,
    require_finite,
)
from .events import EvidenceRole
from .query import (
    AcceptanceThresholds,
    AssaySpec,
    EnvironmentVariableSpec,
    EvidencePolicy,
    InterventionSpec,
    OutputSpec,
    PrecisionRequirement,
    PredictionHorizon,
    QueryConstraints,
    StateQuery,
    SystemBoundary,
    Timescale,
)
from .subjects import SubjectSpecification


class StateFactor(StrEnum):
    STABLE_IDENTITY = "stable_identity"
    SLOW_MEMORY = "slow_memory"
    REGULATORY = "regulatory"
    SIGNALING = "signaling"
    METABOLIC = "metabolic"
    PHYSICAL = "physical"
    DAMAGE_STRESS = "damage_stress"
    FUNCTIONAL_CAPACITY = "functional_capacity"


class StateFactorSpecification(SchemaModel):
    """The dimensions of one factor that are active for a compiled query."""

    factor: StateFactor
    dimensions: tuple[str, ...] = Field(min_length=1)
    timescales: frozenset[Timescale] = Field(min_length=1)
    required_for_outputs: tuple[str, ...] = Field(min_length=1)
    rationale: str = Field(min_length=1)

    @field_validator("dimensions", "required_for_outputs")
    @classmethod
    def unique_members(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if len(values) != len(set(values)):
            raise ValueError("compiled factor members must be unique")
        return values


class ExcludedStateFactor(SchemaModel):
    """A known factor intentionally excluded by query compilation."""

    factor: StateFactor
    rationale: str = Field(min_length=1)


class CompiledStateSpecification(SchemaModel):
    """Auditable output of query compilation, embedded in every belief and forecast."""

    schema_version: SchemaVersion = SCHEMA_VERSION
    query_fingerprint: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    subject: SubjectSpecification
    compiler_id: str = Field(min_length=1)
    compiler_version: str = Field(min_length=1)
    compiler_fingerprint: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    active_factors: tuple[StateFactorSpecification, ...] = Field(min_length=1)
    excluded_factors: tuple[ExcludedStateFactor, ...]
    system_boundary: SystemBoundary
    temporal_resolution_seconds: float = Field(gt=0)
    target_outputs: tuple[OutputSpec, ...] = Field(min_length=1)
    prediction_horizons: tuple[PredictionHorizon, ...] = Field(min_length=1)
    intervention_space: tuple[InterventionSpec, ...]
    environment_space: tuple[EnvironmentVariableSpec, ...]
    precision_requirements: tuple[PrecisionRequirement, ...]
    available_assays: tuple[AssaySpec, ...]
    evidence_policy: EvidencePolicy
    constraints: QueryConstraints
    target_output_keys: tuple[str, ...] = Field(min_length=1)
    horizon_names: tuple[str, ...] = Field(min_length=1)
    admissible_evidence_roles: tuple[EvidenceRole, ...] = Field(min_length=1)
    acceptance_thresholds: AcceptanceThresholds
    context_modulator_dimensions: tuple[str, ...] = ()
    intervention_realization_dimensions: tuple[str, ...] = ()
    observation_nuisance_dimensions: tuple[str, ...] = ()

    @field_validator("temporal_resolution_seconds")
    @classmethod
    def finite_temporal_resolution(cls, value: float) -> float:
        return require_finite(value, name="compiled temporal resolution")

    @field_validator(
        "context_modulator_dimensions",
        "intervention_realization_dimensions",
        "observation_nuisance_dimensions",
    )
    @classmethod
    def unique_dimensions(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if len(values) != len(set(values)):
            raise ValueError("compiled state dimensions must be unique")
        return values

    @model_validator(mode="after")
    def factors_and_dimensions_are_unambiguous(self) -> CompiledStateSpecification:
        factors = [item.factor for item in self.active_factors]
        if len(factors) != len(set(factors)):
            raise ValueError("a compiled state may activate each factor at most once")
        excluded = [item.factor for item in self.excluded_factors]
        if len(excluded) != len(set(excluded)):
            raise ValueError("a compiled state may exclude each factor at most once")
        if set(factors) & set(excluded) or set(factors) | set(excluded) != set(StateFactor):
            raise ValueError(
                "active and excluded factors must partition the known state-factor families"
            )
        for name, values in (
            ("compiled target outputs", self.target_output_keys),
            ("compiled horizons", self.horizon_names),
            ("compiled evidence roles", self.admissible_evidence_roles),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"{name} must be unique")

        if self.target_output_keys != tuple(output.term.key for output in self.target_outputs):
            raise ValueError("compiled target keys must exactly project the embedded outputs")
        if self.horizon_names != tuple(horizon.name for horizon in self.prediction_horizons):
            raise ValueError("compiled horizon names must exactly project the embedded horizons")
        if self.admissible_evidence_roles != self.evidence_policy.allowed_evidence_roles:
            raise ValueError(
                "compiled evidence roles must exactly project the embedded evidence policy"
            )
        target_keys = set(self.target_output_keys)
        for factor in self.active_factors:
            if not set(factor.required_for_outputs) <= target_keys:
                raise ValueError("compiled factors cannot require undeclared target outputs")

        embedded_query = StateQuery(
            subject=self.subject,
            system_boundary=self.system_boundary,
            temporal_resolution_seconds=self.temporal_resolution_seconds,
            prediction_horizons=self.prediction_horizons,
            target_outputs=self.target_outputs,
            intervention_space=self.intervention_space,
            environment_space=self.environment_space,
            precision_requirements=self.precision_requirements,
            available_assays=self.available_assays,
            evidence_policy=self.evidence_policy,
            acceptance_thresholds=self.acceptance_thresholds,
            constraints=self.constraints,
        )
        if embedded_query.fingerprint != self.query_fingerprint:
            raise ValueError(
                "compiled query semantics must exactly reproduce the bound query fingerprint"
            )
        groups = [
            set(dimension for item in self.active_factors for dimension in item.dimensions),
            set(self.context_modulator_dimensions),
            set(self.intervention_realization_dimensions),
            set(self.observation_nuisance_dimensions),
        ]
        for index, left in enumerate(groups):
            for right in groups[index + 1 :]:
                if left & right:
                    raise ValueError("compiled state dimension groups must be disjoint")
        return self

    @property
    def active_factor_map(self) -> dict[StateFactor, StateFactorSpecification]:
        return {item.factor: item for item in self.active_factors}

    @property
    def joint_dimensions(self) -> tuple[str, ...]:
        """Causally relevant dimensions represented by the joint state posterior."""

        return (
            *(dimension for item in self.active_factors for dimension in item.dimensions),
            *self.context_modulator_dimensions,
            *self.intervention_realization_dimensions,
            *self.observation_nuisance_dimensions,
        )

    @property
    def fingerprint(self) -> str:
        return canonical_fingerprint(self)


__all__ = [
    "CompiledStateSpecification",
    "ExcludedStateFactor",
    "StateFactor",
    "StateFactorSpecification",
]
