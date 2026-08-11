"""Contracts for controlled evolution, causal forecasts, and intervention selection."""

from __future__ import annotations

import math
from enum import StrEnum
from uuid import UUID

import numpy as np
from pydantic import Field, field_validator, model_validator

from .belief import (
    BeliefDiagnostics,
    CausalSupportReport,
    ContextBelief,
    DynamicSummary,
    FactorBelief,
    InterventionRealizationBelief,
    NuisanceBelief,
    ParametricDistribution,
    QueryReadinessReport,
    UnavailableDistribution,
    UncertaintyBreakdown,
    _validate_causal_support_against_query,
    _validate_identified_evidence_provenance,
)
from .common import (
    SCHEMA_VERSION,
    CausalStatus,
    EvidenceStatus,
    OntologyTerm,
    ProvenanceRecord,
    Quantity,
    SchemaModel,
    SchemaVersion,
    SupportStatus,
    require_finite,
)
from .distributions import StateDistribution
from .events import EnvironmentEvent, InterventionEvent, PerturbationStatus
from .query import OutputSpec, StateQuery
from .specification import CompiledStateSpecification
from .subjects import BeliefSubject


class EvolutionScenario(SchemaModel):
    schema_version: SchemaVersion = SCHEMA_VERSION
    scenario_id: str = Field(min_length=1)
    horizon_name: str = Field(min_length=1)
    subject: BeliefSubject
    start_time_seconds: float
    end_time_seconds: float
    interventions: tuple[InterventionEvent, ...] = ()
    environments: tuple[EnvironmentEvent, ...] = ()
    inherit_active_interventions: bool | None = None
    inherit_current_environment: bool | None = None

    @property
    def subject_id(self) -> str:
        return self.subject.subject_id

    @field_validator("start_time_seconds", "end_time_seconds")
    @classmethod
    def finite_times(cls, value: float) -> float:
        return require_finite(value, name="scenario time")

    @model_validator(mode="after")
    def valid_interval_and_events(self) -> EvolutionScenario:
        if self.end_time_seconds <= self.start_time_seconds:
            raise ValueError("scenario end must be later than its start")
        events: tuple[InterventionEvent | EnvironmentEvent, ...] = (
            *self.interventions,
            *self.environments,
        )
        event_ids = [event.event_id for event in events]
        if len(event_ids) != len(set(event_ids)):
            raise ValueError("scenario event IDs must be unique")
        for event in events:
            if event.subject != self.subject:
                raise ValueError("scenario events must refer to the typed scenario subject")
            event_end = event.time_seconds + event.duration_seconds
            if event.time_seconds < self.start_time_seconds or event_end > self.end_time_seconds:
                raise ValueError("scenario event interval lies outside the scenario interval")
        for intervention in self.interventions:
            if intervention.estimated_efficiency is not None:
                raise ValueError("a planned intervention cannot claim a realized future efficiency")
            if (
                intervention.actual_perturbation is not None
                and intervention.actual_perturbation.status is not PerturbationStatus.UNKNOWN
            ):
                raise ValueError(
                    "a planned intervention cannot carry retrospective realization evidence"
                )
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


class TransportStatus(StrEnum):
    WITHIN_SUPPORT = "within_support"
    TRANSPORTED = "transported"
    EXTRAPOLATED = "extrapolated"
    NOT_EVALUATED = "not_evaluated"
    UNSUPPORTED = "unsupported"


class TransportReport(SchemaModel):
    status: TransportStatus
    source_domain: str | None = Field(default=None, min_length=1)
    target_domain: str | None = Field(default=None, min_length=1)
    assumptions: tuple[str, ...] = ()
    evidence_ids: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    @model_validator(mode="after")
    def transported_claims_name_domains_and_assumptions(self) -> TransportReport:
        if any(
            value is not None and not value.strip()
            for value in (self.source_domain, self.target_domain)
        ):
            raise ValueError("transport domains must be nonblank")
        if any(not assumption.strip() for assumption in self.assumptions):
            raise ValueError("transport assumptions must be nonblank")
        if len(self.assumptions) != len(set(self.assumptions)):
            raise ValueError("transport assumptions must be unique")
        if any(not evidence_id.strip() for evidence_id in self.evidence_ids):
            raise ValueError("transport evidence IDs must be nonblank")
        if self.status in {TransportStatus.TRANSPORTED, TransportStatus.EXTRAPOLATED}:
            if self.source_domain is None or self.target_domain is None or not self.assumptions:
                raise ValueError(
                    "transported or extrapolated results require source, target, and assumptions"
                )
            if self.status is TransportStatus.TRANSPORTED and not self.evidence_ids:
                raise ValueError("transported results require transport evidence")
        elif self.status is TransportStatus.WITHIN_SUPPORT and self.assumptions:
            raise ValueError("within-support predictions must not claim transport assumptions")
        if len(self.evidence_ids) != len(set(self.evidence_ids)):
            raise ValueError("transport evidence IDs must be unique")
        return self


def _validate_causal_transport(
    causal_status: CausalStatus,
    transport: TransportReport,
    assumptions: tuple[str, ...],
    *,
    source_scope: str | None = None,
    target_scope: str | None = None,
) -> None:
    if causal_status is CausalStatus.IDENTIFIED_POPULATION_EFFECT:
        if transport.status is not TransportStatus.WITHIN_SUPPORT:
            raise ValueError("an identified population effect requires within-support transport")
    elif causal_status is CausalStatus.TRANSPORTED_UNDER_ASSUMPTIONS:
        if transport.status is not TransportStatus.TRANSPORTED:
            raise ValueError("transported causal status requires a transported result")
        if not transport.evidence_ids:
            raise ValueError("transported causal status requires transport evidence")
        if set(assumptions) != set(transport.assumptions):
            raise ValueError("causal and transport assumptions must agree")
        if (source_scope is not None or target_scope is not None) and (
            transport.source_domain != source_scope or transport.target_domain != target_scope
        ):
            raise ValueError("causal scopes and transport domains must agree")
    elif (
        causal_status is CausalStatus.MECHANISTIC_EXTRAPOLATION
        and transport.status is not TransportStatus.EXTRAPOLATED
    ):
        raise ValueError("mechanistic extrapolation requires extrapolated transport status")


class TargetPrediction(SchemaModel):
    target: OutputSpec
    units: str = Field(min_length=1)
    horizon_seconds: float = Field(gt=0)
    status: SupportStatus
    distribution: StateDistribution
    causal_status: CausalStatus
    transport: TransportReport
    causal_assumptions: tuple[str, ...] = ()
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
        if (
            self.causal_status
            in {
                CausalStatus.TRANSPORTED_UNDER_ASSUMPTIONS,
                CausalStatus.MECHANISTIC_EXTRAPOLATION,
            }
            and not self.causal_assumptions
        ):
            raise ValueError("transported or extrapolated causal claims require assumptions")
        _validate_causal_transport(
            self.causal_status,
            self.transport,
            self.causal_assumptions,
        )
        return self


def _require_parametric_marginal(
    joint: StateDistribution,
    marginal: StateDistribution,
    *,
    label: str,
) -> None:
    if not isinstance(joint, ParametricDistribution) or not isinstance(
        marginal, ParametricDistribution
    ):
        return
    indices = [joint.dimensions.index(dimension) for dimension in marginal.dimensions]
    expected_mean = np.asarray(joint.mean)[indices]
    expected_covariance = np.asarray(joint.covariance)[np.ix_(indices, indices)]
    if not np.allclose(expected_mean, marginal.mean, rtol=0, atol=1e-8) or not np.allclose(
        expected_covariance, marginal.covariance, rtol=0, atol=1e-8
    ):
        raise ValueError(f"parametric {label} posterior must match the forecast joint marginal")


class StateForecast(SchemaModel):
    schema_version: SchemaVersion = SCHEMA_VERSION
    forecast_id: str = Field(min_length=1)
    parent_belief_id: UUID
    scenario_id: str = Field(min_length=1)
    scenario_fingerprint: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    query: StateQuery
    query_fingerprint: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    state_specification: CompiledStateSpecification
    horizon_name: str = Field(min_length=1)
    horizon_seconds: float = Field(gt=0)
    subject: BeliefSubject
    start_time_seconds: float
    end_time_seconds: float
    joint_posterior: StateDistribution
    factors: tuple[FactorBelief, ...]
    context: ContextBelief
    intervention_realizations: tuple[InterventionRealizationBelief, ...] = ()
    nuisance: NuisanceBelief | None = None
    target_predictions: tuple[TargetPrediction, ...]
    dynamics: DynamicSummary
    uncertainty: UncertaintyBreakdown
    diagnostics: BeliefDiagnostics
    readiness: QueryReadinessReport
    causal_status: CausalStatus
    transport: TransportReport
    provenance: ProvenanceRecord

    @property
    def subject_id(self) -> str:
        return self.subject.subject_id

    @model_validator(mode="after")
    def query_horizon_and_predictions_are_complete(self) -> StateForecast:
        if not self.subject.is_compatible_with(self.query.subject):
            raise ValueError("forecast subject must satisfy the query subject specification")
        if self.query.fingerprint != self.query_fingerprint:
            raise ValueError("forecast query and query fingerprint must agree")
        if self.provenance.query_fingerprint != self.query_fingerprint:
            raise ValueError("forecast provenance/query fingerprints must agree")
        if self.state_specification.query_fingerprint != self.query_fingerprint:
            raise ValueError("forecast state specification/query fingerprints must agree")
        if self.state_specification.subject != self.query.subject:
            raise ValueError("forecast state specification must use the query subject")
        if self.state_specification.acceptance_thresholds != self.query.acceptance_thresholds:
            raise ValueError("forecast compiled thresholds must match the query")
        if set(self.state_specification.target_output_keys) != {
            output.term.key for output in self.query.target_outputs
        }:
            raise ValueError("forecast compiled targets must match the query")
        if set(self.state_specification.horizon_names) != {
            horizon.name for horizon in self.query.prediction_horizons
        }:
            raise ValueError("forecast compiled horizons must match the query")
        if set(self.state_specification.admissible_evidence_roles) != set(
            self.query.evidence_policy.allowed_evidence_roles
        ):
            raise ValueError("forecast compiled evidence roles must match the query")
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

        expected_by_key = {
            target.term.key: target
            for target in self.query.target_outputs
            if self.horizon_name in target.supported_horizon_names
        }
        expected_targets = set(expected_by_key)
        prediction_targets = [prediction.target.term.key for prediction in self.target_predictions]
        if set(prediction_targets) != expected_targets or len(prediction_targets) != len(
            expected_targets
        ):
            raise ValueError(
                "forecast must predict each query target exactly once when supported at this "
                "horizon"
            )
        if any(
            prediction.target != expected_by_key[prediction.target.term.key]
            for prediction in self.target_predictions
        ):
            raise ValueError("forecast target specifications must match the query")
        if any(
            prediction.causal_status not in {self.causal_status, CausalStatus.UNSUPPORTED}
            for prediction in self.target_predictions
        ):
            raise ValueError(
                "target causal status cannot be stronger than or differ from the forecast branch"
            )
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

        active = self.state_specification.active_factor_map
        factors = {factor.factor: factor for factor in self.factors}
        if len(factors) != len(self.factors) or set(factors) != set(active):
            raise ValueError("forecast factors must equal the query-compiled active factors")
        expected_joint = set(self.state_specification.joint_dimensions)
        if set(self.joint_posterior.dimensions) != expected_joint:
            raise ValueError("forecast joint dimensions must equal the compiled state")
        source_event_ids = set(self.provenance.source_event_ids)
        for factor_kind, factor in factors.items():
            if factor.evidence_status is EvidenceStatus.OBSERVED:
                raise ValueError("future forecast factors cannot be directly observed")
            if set(factor.posterior.dimensions) != set(active[factor_kind].dimensions):
                raise ValueError("forecast factor dimensions must match the compiled factor")
            if factor.timescales != active[factor_kind].timescales:
                raise ValueError("forecast factor timescales must match the compiled factor")
            if not set(factor.evidence_event_ids) <= source_event_ids:
                raise ValueError("forecast factor evidence IDs must appear in provenance")
            _require_parametric_marginal(
                self.joint_posterior, factor.posterior, label=factor.factor.value
            )

        context_dimensions = set(self.state_specification.context_modulator_dimensions)
        if context_dimensions:
            if (
                self.context.latent_context_posterior is None
                or set(self.context.latent_context_posterior.dimensions) != context_dimensions
            ):
                raise ValueError("forecast context posterior must match the compiled state")
            _require_parametric_marginal(
                self.joint_posterior,
                self.context.latent_context_posterior,
                label="context",
            )
        elif self.context.latent_context_posterior is not None:
            raise ValueError("forecast must not emit out-of-query context state")

        realization_ids = [
            realization.intervention_event_id for realization in self.intervention_realizations
        ]
        if len(realization_ids) != len(set(realization_ids)):
            raise ValueError("forecast realization blocks must name unique interventions")
        if not set(realization_ids) <= source_event_ids:
            raise ValueError("forecast realization blocks must reference provenance events")
        realization_dimensions = [
            dimension
            for realization in self.intervention_realizations
            for dimension in realization.posterior.dimensions
        ]
        if len(realization_dimensions) != len(set(realization_dimensions)) or set(
            realization_dimensions
        ) != set(self.state_specification.intervention_realization_dimensions):
            raise ValueError("forecast realization blocks must partition the compiled state")
        for realization in self.intervention_realizations:
            if not set(realization.evidence_event_ids) <= source_event_ids:
                raise ValueError("forecast realization evidence must appear in provenance")
            _require_parametric_marginal(
                self.joint_posterior,
                realization.posterior,
                label=f"intervention realization {realization.intervention_event_id}",
            )

        nuisance_dimensions = set(self.state_specification.observation_nuisance_dimensions)
        if nuisance_dimensions:
            if (
                self.nuisance is None
                or set(self.nuisance.posterior.dimensions) != nuisance_dimensions
            ):
                raise ValueError("forecast Xi posterior must match the compiled nuisance state")
            if not set(self.nuisance.evidence_event_ids) <= source_event_ids:
                raise ValueError("forecast nuisance evidence must appear in provenance")
            _require_parametric_marginal(
                self.joint_posterior,
                self.nuisance.posterior,
                label="Xi nuisance",
            )
        elif self.nuisance is not None:
            raise ValueError("forecast must not emit out-of-query nuisance state")

        if set(self.diagnostics.identifiability.dimension_status) != expected_joint:
            raise ValueError("forecast identifiability must classify every active dimension")
        if self.diagnostics.support.outcome is not self.readiness.support:
            raise ValueError("forecast support/readiness outcomes must agree")
        if self.diagnostics.sufficiency.outcome is not self.readiness.sufficiency:
            raise ValueError("forecast sufficiency/readiness outcomes must agree")
        if self.diagnostics.identifiability.outcome is not self.readiness.identifiability:
            raise ValueError("forecast identifiability/readiness outcomes must agree")
        if self.diagnostics.decision_uncertainty.outcome is not self.readiness.decision_uncertainty:
            raise ValueError("forecast decision uncertainty/readiness outcomes must agree")
        if self.diagnostics.calibration.outcome is not self.readiness.calibration:
            raise ValueError("forecast calibration/readiness outcomes must agree")
        if self.diagnostics.causal_support.outcome is not self.readiness.causal:
            raise ValueError("forecast causal-support/readiness outcomes must agree")
        if self.diagnostics.causal_support.causal_status is not self.causal_status:
            raise ValueError("forecast causal status must match its causal-support report")
        _validate_causal_transport(
            self.causal_status,
            self.transport,
            self.diagnostics.causal_support.transport_assumptions,
            source_scope=self.diagnostics.causal_support.source_scope,
            target_scope=self.diagnostics.causal_support.target_scope,
        )
        for prediction in self.target_predictions:
            if prediction.causal_status is not CausalStatus.TRANSPORTED_UNDER_ASSUMPTIONS:
                continue
            _validate_causal_transport(
                prediction.causal_status,
                prediction.transport,
                self.diagnostics.causal_support.transport_assumptions,
                source_scope=self.diagnostics.causal_support.source_scope,
                target_scope=self.diagnostics.causal_support.target_scope,
            )
        if (
            not set(self.diagnostics.causal_support.evidence_ids)
            <= self.provenance.scientific_evidence_ids
        ):
            raise ValueError("forecast causal evidence must appear in provenance")
        _validate_causal_support_against_query(
            self.diagnostics.causal_support,
            self.query,
            required_target_horizons={
                (prediction.target.term.key, self.horizon_name)
                for prediction in self.target_predictions
            },
        )
        if any(
            estimand.scenario_id != self.scenario_id
            or estimand.scenario_fingerprint != self.scenario_fingerprint
            for estimand in self.diagnostics.causal_support.estimands
        ):
            raise ValueError(
                "forecast causal estimands must bind the exact scenario ID and fingerprint"
            )
        _validate_identified_evidence_provenance(
            self.diagnostics.causal_support,
            self.provenance,
        )
        if self.readiness.control_requested is not bool(self.query.intervention_space):
            raise ValueError("forecast readiness must reflect the query intervention space")
        if (
            self.causal_status is CausalStatus.TRANSPORTED_UNDER_ASSUMPTIONS
            and not self.query.constraints.allow_transport
        ):
            raise ValueError("query constraints forbid a transported forecast")
        thresholds = self.query.acceptance_thresholds
        if self.diagnostics.support.maximum_ood_score != thresholds.maximum_ood_score:
            raise ValueError("forecast support must use the query OOD threshold")
        if (
            self.diagnostics.sufficiency.maximum_history_information_gain
            != thresholds.maximum_history_information_gain
        ):
            raise ValueError("forecast sufficiency must use the query threshold")
        if (
            self.diagnostics.identifiability.minimum_identifiability_score
            != thresholds.minimum_identifiability
        ):
            raise ValueError("forecast identifiability must use the query threshold")
        if (
            self.diagnostics.decision_uncertainty.maximum_decision_uncertainty
            != thresholds.maximum_decision_uncertainty
            or self.diagnostics.decision_uncertainty.maximum_counterfactual_uncertainty
            != thresholds.maximum_counterfactual_uncertainty
        ):
            raise ValueError("forecast decision uncertainty must use the query thresholds")
        if (
            self.diagnostics.calibration.minimum_coverage != thresholds.minimum_calibration_coverage
            or self.diagnostics.calibration.maximum_calibration_error
            != thresholds.maximum_calibration_error
        ):
            raise ValueError("forecast calibration must use the query thresholds")
        if self.causal_status is CausalStatus.UNSUPPORTED and self.readiness.valid_for_control:
            raise ValueError("a causally unsupported forecast cannot be valid for control")
        return self


class CandidateEvaluation(SchemaModel):
    scenario_id: str = Field(min_length=1)
    expected_utility: float | None = None
    uncertainty_penalty: float | None = Field(default=None, ge=0)
    selection_score: float | None = None
    supported: bool
    causal_status: CausalStatus
    causal_support: CausalSupportReport
    transport: TransportReport
    readiness: QueryReadinessReport
    notes: tuple[str, ...] = ()

    @model_validator(mode="after")
    def score_is_risk_adjusted_utility(self) -> CandidateEvaluation:
        values = (self.expected_utility, self.uncertainty_penalty, self.selection_score)
        if self.supported and any(value is None for value in values):
            raise ValueError("a supported candidate evaluation requires all utility values")
        if not self.supported and any(value is not None for value in values):
            raise ValueError("an unsupported candidate must not contain numeric utility sentinels")
        if self.causal_support.causal_status is not self.causal_status:
            raise ValueError("candidate causal status must match its support report")
        if self.causal_support.outcome is not self.readiness.causal:
            raise ValueError("candidate causal support must match readiness")
        if any(
            estimand.scenario_id != self.scenario_id or estimand.scenario_fingerprint is None
            for estimand in self.causal_support.estimands
        ):
            raise ValueError(
                "candidate causal estimands must bind the candidate scenario and fingerprint"
            )
        _validate_causal_transport(
            self.causal_status,
            self.transport,
            self.causal_support.transport_assumptions,
        )
        if not self.supported:
            return self
        if not self.readiness.valid_for_control:
            raise ValueError("a supported intervention candidate must be valid for control")
        if self.causal_status not in {
            CausalStatus.IDENTIFIED_POPULATION_EFFECT,
            CausalStatus.TRANSPORTED_UNDER_ASSUMPTIONS,
        }:
            raise ValueError(
                "a selectable candidate requires identified or transported causal support"
            )
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


class PlanStatus(StrEnum):
    SELECTED = "selected"
    ABSTAINED = "abstained"


class InterventionPlan(SchemaModel):
    schema_version: SchemaVersion = SCHEMA_VERSION
    plan_id: str = Field(min_length=1)
    parent_belief_id: UUID
    query_fingerprint: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    horizon_name: str = Field(min_length=1)
    objective_id: str = Field(min_length=1)
    objective_fingerprint: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    candidates: tuple[ScenarioReference, ...] = Field(min_length=1)
    status: PlanStatus
    selected_scenario_id: str | None
    evaluations: tuple[CandidateEvaluation, ...]
    readiness: QueryReadinessReport
    causal_status: CausalStatus
    causal_support: CausalSupportReport
    transport: TransportReport
    abstention_reasons: tuple[str, ...] = ()
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
        candidate_fingerprints = {
            candidate.scenario_id: candidate.fingerprint for candidate in self.candidates
        }
        for evaluation in self.evaluations:
            if any(
                estimand.scenario_fingerprint != candidate_fingerprints[evaluation.scenario_id]
                for estimand in evaluation.causal_support.estimands
            ):
                raise ValueError(
                    "candidate causal estimands must bind the exact candidate fingerprint"
                )
        if self.causal_support.causal_status is not self.causal_status:
            raise ValueError("plan causal status must match its support report")
        if self.causal_support.outcome is not self.readiness.causal:
            raise ValueError("plan causal support must match readiness")
        _validate_causal_transport(
            self.causal_status,
            self.transport,
            self.causal_support.transport_assumptions,
        )
        if not set(self.causal_support.evidence_ids) <= self.provenance.scientific_evidence_ids:
            raise ValueError("plan causal evidence must appear in provenance")
        _validate_identified_evidence_provenance(self.causal_support, self.provenance)
        for evaluation in self.evaluations:
            _validate_identified_evidence_provenance(
                evaluation.causal_support,
                self.provenance,
            )

        if self.status is PlanStatus.ABSTAINED:
            if self.selected_scenario_id is not None:
                raise ValueError("an abstaining plan cannot select a scenario")
            if not self.abstention_reasons:
                raise ValueError("an abstaining plan requires explicit reasons")
        else:
            if self.abstention_reasons:
                raise ValueError("a selected plan cannot also report abstention reasons")
            if not self.readiness.valid_for_control:
                raise ValueError("an intervention plan cannot select while control is invalid")
            if self.causal_status not in {
                CausalStatus.IDENTIFIED_POPULATION_EFFECT,
                CausalStatus.TRANSPORTED_UNDER_ASSUMPTIONS,
            }:
                raise ValueError(
                    "a selected plan requires identified or transported causal support"
                )
            if self.selected_scenario_id is None:
                raise ValueError("a selected plan requires a selected scenario")

        if self.selected_scenario_id is not None and (
            self.selected_scenario_id not in evaluations
            or not evaluations[self.selected_scenario_id].supported
        ):
            raise ValueError("a selected scenario must be a supported candidate evaluation")
        if self.selected_scenario_id is not None:
            selected_evaluation = evaluations[self.selected_scenario_id]
            if selected_evaluation.causal_status is not self.causal_status:
                raise ValueError("selected plan causal status must match its candidate evaluation")
            if selected_evaluation.causal_support != self.causal_support:
                raise ValueError("selected plan causal support must equal its candidate evaluation")
            if selected_evaluation.transport != self.transport:
                raise ValueError("selected plan transport must equal its candidate evaluation")
        supported = [evaluation for evaluation in self.evaluations if evaluation.supported]
        if self.status is PlanStatus.SELECTED:
            if not supported:
                raise ValueError("a selected plan requires at least one supported candidate")
            assert self.selected_scenario_id is not None
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
        transports = (self.transport, *(evaluation.transport for evaluation in self.evaluations))
        if any(
            not set(transport.evidence_ids) <= self.provenance.scientific_evidence_ids
            for transport in transports
        ):
            raise ValueError("plan transport evidence must appear in provenance")
        return self


__all__ = [
    "CandidateEvaluation",
    "EvolutionScenario",
    "InterventionObjective",
    "InterventionPlan",
    "ObjectiveDirection",
    "ObjectiveTerm",
    "PlanStatus",
    "ScenarioReference",
    "StateForecast",
    "TargetPrediction",
    "TransportReport",
    "TransportStatus",
]
