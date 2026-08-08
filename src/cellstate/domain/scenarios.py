"""Contracts for controlled evolution and intervention selection."""

from __future__ import annotations

import math
from enum import StrEnum
from uuid import UUID

import numpy as np
from pydantic import Field, field_validator, model_validator

from .belief import (
    DynamicSummary,
    FactorBelief,
    ParametricDistribution,
    StateDistribution,
    StateFactor,
    UnavailableDistribution,
    UncertaintyBreakdown,
)
from .common import (
    SCHEMA_VERSION,
    EvidenceStatus,
    OntologyTerm,
    ProvenanceRecord,
    Quantity,
    SchemaModel,
    SchemaVersion,
    SupportStatus,
    require_finite,
)
from .events import EnvironmentEvent, InterventionEvent
from .query import OutputSpec, StateQuery


class EvolutionScenario(SchemaModel):
    schema_version: SchemaVersion = SCHEMA_VERSION
    scenario_id: str = Field(min_length=1)
    horizon_name: str = Field(min_length=1)
    subject_id: str = Field(min_length=1)
    start_time_seconds: float
    end_time_seconds: float
    interventions: tuple[InterventionEvent, ...] = ()
    environments: tuple[EnvironmentEvent, ...] = ()
    inherit_active_interventions: bool | None = None
    inherit_current_environment: bool | None = None

    @field_validator("start_time_seconds", "end_time_seconds")
    @classmethod
    def finite_times(cls, value: float) -> float:
        return require_finite(value, name="scenario time")

    @model_validator(mode="after")
    def valid_interval_and_events(self) -> EvolutionScenario:
        if self.end_time_seconds <= self.start_time_seconds:
            raise ValueError("scenario end must be later than its start")
        events = (*self.interventions, *self.environments)
        event_ids = [event.event_id for event in events]
        if len(event_ids) != len(set(event_ids)):
            raise ValueError("scenario event IDs must be unique")
        for event in events:
            if event.subject_id != self.subject_id:
                raise ValueError("scenario events must refer to the scenario subject")
            if not self.start_time_seconds <= event.time_seconds <= self.end_time_seconds:
                raise ValueError("scenario event lies outside the scenario interval")
        return self


class ObjectiveDirection(StrEnum):
    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"
    TARGET = "target"


class ObjectiveTerm(SchemaModel):
    target: OntologyTerm
    direction: ObjectiveDirection
    weight: float = Field(default=1.0, gt=0)
    target_value: Quantity | None = None

    @model_validator(mode="after")
    def target_has_value(self) -> ObjectiveTerm:
        if self.direction is ObjectiveDirection.TARGET and self.target_value is None:
            raise ValueError("target objectives require target_value")
        return self


class InterventionObjective(SchemaModel):
    objective_id: str = Field(min_length=1)
    horizon_name: str = Field(min_length=1)
    terms: tuple[ObjectiveTerm, ...] = Field(min_length=1)
    risk_aversion: float = Field(default=0, ge=0)

    @field_validator("terms")
    @classmethod
    def unique_targets(cls, terms: tuple[ObjectiveTerm, ...]) -> tuple[ObjectiveTerm, ...]:
        keys = [term.target.key for term in terms]
        if len(keys) != len(set(keys)):
            raise ValueError("an intervention objective may name each target only once")
        return terms


class TargetPrediction(SchemaModel):
    target: OutputSpec
    units: str = Field(min_length=1)
    horizon_seconds: float = Field(gt=0)
    status: SupportStatus
    distribution: StateDistribution
    notes: tuple[str, ...] = ()

    @model_validator(mode="after")
    def status_matches_distribution(self) -> TargetPrediction:
        if self.units != self.target.units:
            raise ValueError("target prediction units must match the query output specification")
        unavailable = isinstance(self.distribution, UnavailableDistribution)
        if self.status is SupportStatus.SUPPORTED and unavailable:
            raise ValueError("a supported target prediction requires a distribution")
        if self.status is not SupportStatus.SUPPORTED and not unavailable:
            raise ValueError(
                "an unavailable target prediction requires an unavailable distribution"
            )
        return self


class StateForecast(SchemaModel):
    schema_version: SchemaVersion = SCHEMA_VERSION
    forecast_id: str = Field(min_length=1)
    parent_belief_id: UUID
    scenario_id: str = Field(min_length=1)
    scenario_fingerprint: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    query: StateQuery
    query_fingerprint: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    horizon_name: str = Field(min_length=1)
    horizon_seconds: float = Field(gt=0)
    subject_id: str = Field(min_length=1)
    start_time_seconds: float
    end_time_seconds: float
    joint_posterior: StateDistribution
    factors: tuple[FactorBelief, ...]
    target_predictions: tuple[TargetPrediction, ...]
    dynamics: DynamicSummary
    uncertainty: UncertaintyBreakdown
    provenance: ProvenanceRecord

    @model_validator(mode="after")
    def query_horizon_and_predictions_are_complete(self) -> StateForecast:
        if self.query.fingerprint != self.query_fingerprint:
            raise ValueError("forecast query and query fingerprint must agree")
        if self.provenance.query_fingerprint != self.query_fingerprint:
            raise ValueError("forecast provenance/query fingerprints must agree")
        if not math.isclose(
            self.end_time_seconds - self.start_time_seconds,
            self.horizon_seconds,
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            raise ValueError("forecast horizon must equal its time interval")
        horizons = {horizon.name: horizon for horizon in self.query.prediction_horizons}
        if self.horizon_name not in horizons:
            raise ValueError("forecast horizon is not declared by its query")
        if not math.isclose(
            horizons[self.horizon_name].duration_seconds,
            self.horizon_seconds,
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            raise ValueError("forecast duration does not match its named query horizon")
        expected_by_key = {target.term.key: target for target in self.query.target_outputs}
        expected_targets = set(expected_by_key)
        prediction_targets = [prediction.target.term.key for prediction in self.target_predictions]
        if set(prediction_targets) != expected_targets or len(prediction_targets) != len(
            expected_targets
        ):
            raise ValueError("forecast must predict each query target exactly once")
        if any(
            prediction.target != expected_by_key[prediction.target.term.key]
            for prediction in self.target_predictions
        ):
            raise ValueError("forecast target specifications must match the query")
        if any(
            not math.isclose(
                prediction.horizon_seconds,
                self.horizon_seconds,
                rel_tol=1e-9,
                abs_tol=1e-9,
            )
            for prediction in self.target_predictions
        ):
            raise ValueError("target predictions must use the forecast horizon")
        factor_names = [factor.factor for factor in self.factors]
        if set(factor_names) != set(StateFactor) or len(factor_names) != len(StateFactor):
            raise ValueError("forecast must contain every structured state factor exactly once")
        source_event_ids = set(self.provenance.source_event_ids)
        joint_dimensions = set(self.joint_posterior.dimensions)
        for factor in self.factors:
            if factor.evidence_status is EvidenceStatus.OBSERVED:
                raise ValueError("future forecast factors cannot be directly observed")
            if not set(factor.evidence_event_ids) <= source_event_ids:
                raise ValueError("forecast factor evidence IDs must appear in provenance")
            if (
                not isinstance(factor.posterior, UnavailableDistribution)
                and not set(factor.posterior.dimensions) <= joint_dimensions
            ):
                raise ValueError("forecast factor dimensions must be a subset of the joint")
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
                    raise ValueError("forecast factor posterior must match the joint marginal")
        return self


class CandidateEvaluation(SchemaModel):
    scenario_id: str = Field(min_length=1)
    expected_utility: float | None = None
    uncertainty_penalty: float | None = Field(default=None, ge=0)
    selection_score: float | None = None
    supported: bool
    notes: tuple[str, ...] = ()

    @model_validator(mode="after")
    def score_is_risk_adjusted_utility(self) -> CandidateEvaluation:
        values = (self.expected_utility, self.uncertainty_penalty, self.selection_score)
        if self.supported and any(value is None for value in values):
            raise ValueError("a supported candidate evaluation requires all utility values")
        if not self.supported and any(value is not None for value in values):
            raise ValueError("an unsupported candidate must not contain numeric utility sentinels")
        if not self.supported:
            return self
        assert self.expected_utility is not None
        assert self.uncertainty_penalty is not None
        assert self.selection_score is not None
        if not math.isclose(
            self.selection_score,
            self.expected_utility - self.uncertainty_penalty,
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "selection score must equal expected utility minus uncertainty penalty"
            )
        return self


class ScenarioReference(SchemaModel):
    scenario_id: str = Field(min_length=1)
    fingerprint: str = Field(pattern=r"^[a-fA-F0-9]{64}$")


class InterventionPlan(SchemaModel):
    schema_version: SchemaVersion = SCHEMA_VERSION
    plan_id: str = Field(min_length=1)
    parent_belief_id: UUID
    query_fingerprint: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    horizon_name: str = Field(min_length=1)
    objective_id: str = Field(min_length=1)
    objective_fingerprint: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    candidates: tuple[ScenarioReference, ...] = Field(min_length=1)
    selected_scenario_id: str | None
    evaluations: tuple[CandidateEvaluation, ...]
    rationale: str = Field(min_length=1)
    seed: int = Field(ge=0)
    provenance: ProvenanceRecord

    @model_validator(mode="after")
    def selection_is_bound_to_candidates(self) -> InterventionPlan:
        candidate_ids = [candidate.scenario_id for candidate in self.candidates]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("plan candidate scenario IDs must be unique")
        evaluation_ids = [evaluation.scenario_id for evaluation in self.evaluations]
        if len(evaluation_ids) != len(set(evaluation_ids)) or set(evaluation_ids) != set(
            candidate_ids
        ):
            raise ValueError("plan must evaluate every candidate scenario exactly once")
        evaluations = {evaluation.scenario_id: evaluation for evaluation in self.evaluations}
        if self.selected_scenario_id is not None and (
            self.selected_scenario_id not in evaluations
            or not evaluations[self.selected_scenario_id].supported
        ):
            raise ValueError("a selected scenario must be a supported candidate evaluation")
        supported = [evaluation for evaluation in self.evaluations if evaluation.supported]
        if supported:
            if self.selected_scenario_id is None:
                raise ValueError("a plan with supported candidates must select one")
            selected_score = evaluations[self.selected_scenario_id].selection_score
            best_score = max(
                evaluation.selection_score
                for evaluation in supported
                if evaluation.selection_score is not None
            )
            assert selected_score is not None
            if not math.isclose(selected_score, best_score, rel_tol=1e-9, abs_tol=1e-12):
                raise ValueError("the selected scenario must have the highest selection score")
        if self.provenance.query_fingerprint != self.query_fingerprint:
            raise ValueError("plan provenance/query fingerprints must agree")
        return self
