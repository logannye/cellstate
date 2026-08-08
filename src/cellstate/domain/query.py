"""The formal question that determines what a sufficient state must retain."""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field, field_validator, model_validator

from .common import (
    SCHEMA_VERSION,
    OntologyTerm,
    SchemaModel,
    SchemaVersion,
    canonical_fingerprint,
    require_finite,
)


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


class OutputSpec(SchemaModel):
    term: OntologyTerm
    units: str = Field(min_length=1)
    weight: float = Field(default=1.0, gt=0)
    functional: bool = True

    @field_validator("weight")
    @classmethod
    def finite_weight(cls, value: float) -> float:
        return require_finite(value, name="output weight")


class InterventionSpec(SchemaModel):
    kind: OntologyTerm
    target: OntologyTerm | None = None
    mechanisms: tuple[OntologyTerm, ...] = ()
    dose_units: str | None = None


class EnvironmentVariableSpec(SchemaModel):
    variable: OntologyTerm
    units: str | None = None
    required: bool = False


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


class AssaySpec(SchemaModel):
    assay_id: str = Field(min_length=1)
    modality: OntologyTerm
    cost: float = Field(default=1.0, ge=0)
    turnaround_seconds: float | None = Field(default=None, ge=0)

    @field_validator("cost", "turnaround_seconds")
    @classmethod
    def finite_values(cls, value: float | None) -> float | None:
        if value is not None:
            return require_finite(value, name="assay cost or turnaround")
        return value


class StateQuery(SchemaModel):
    """A task-specific definition of predictive and interventional sufficiency."""

    schema_version: SchemaVersion = SCHEMA_VERSION
    system_boundary: SystemBoundary
    prediction_horizons: tuple[PredictionHorizon, ...] = Field(min_length=1)
    target_outputs: tuple[OutputSpec, ...] = Field(min_length=1)
    intervention_space: tuple[InterventionSpec, ...] = ()
    environment_space: tuple[EnvironmentVariableSpec, ...] = ()
    precision_requirements: tuple[PrecisionRequirement, ...] = ()
    available_assays: tuple[AssaySpec, ...] = ()

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
        target_keys = [output.term.key for output in self.target_outputs]
        if len(target_keys) != len(set(target_keys)):
            raise ValueError("target outputs must be unique")
        intervention_keys = [
            (item.kind.key, item.target.key if item.target is not None else None)
            for item in self.intervention_space
        ]
        if len(intervention_keys) != len(set(intervention_keys)):
            raise ValueError("intervention specifications must be unique")
        environment_keys = [item.variable.key for item in self.environment_space]
        if len(environment_keys) != len(set(environment_keys)):
            raise ValueError("environment variable specifications must be unique")
        assay_ids = [assay.assay_id for assay in self.available_assays]
        if len(assay_ids) != len(set(assay_ids)):
            raise ValueError("available assay IDs must be unique")
        horizon_names = {horizon.name for horizon in self.prediction_horizons}
        precision_keys: list[tuple[str, str, str]] = []
        for requirement in self.precision_requirements:
            if requirement.target.key not in target_keys:
                raise ValueError("precision requirement references an undeclared target")
            if requirement.horizon_name not in horizon_names:
                raise ValueError("precision requirement references an undeclared horizon")
            precision_keys.append(
                (requirement.target.key, requirement.horizon_name, requirement.metric)
            )
        if len(precision_keys) != len(set(precision_keys)):
            raise ValueError(
                "precision requirements must be unique per target, horizon, and metric"
            )
        return self

    @property
    def fingerprint(self) -> str:
        return canonical_fingerprint(self)
