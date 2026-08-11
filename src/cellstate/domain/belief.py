"""Serializable v2 posterior belief, dynamics, diagnostics, and readiness."""

from __future__ import annotations

import math
from enum import StrEnum
from typing import cast
from uuid import UUID, uuid4

import numpy as np
from pydantic import Field, JsonValue, field_validator, model_validator

from .common import (
    SCHEMA_VERSION,
    CausalStatus,
    CriterionOutcome,
    EvidenceStatus,
    OntologyTerm,
    ProvenanceRecord,
    SchemaModel,
    SchemaVersion,
    SupportStatus,
    require_finite,
)
from .distributions import (
    DistributionSupport,
    ParametricDistribution,
    SampleDistribution,
    StateDistribution,
    UnavailableDistribution,
)
from .events import AssignmentMechanism, InterventionEvent
from .query import StateQuery, Timescale
from .specification import CompiledStateSpecification, StateFactor
from .subjects import BeliefSubject, TargetAggregation


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


class InterventionRealizationBelief(SchemaModel):
    """Posterior over the realized intracellular effect of an intended action."""

    intervention_event_id: str = Field(min_length=1)
    evidence_status: EvidenceStatus
    posterior: StateDistribution
    evidence_event_ids: tuple[str, ...] = ()

    @model_validator(mode="after")
    def evidence_semantics(self) -> InterventionRealizationBelief:
        if isinstance(self.posterior, UnavailableDistribution):
            if self.evidence_status is not EvidenceStatus.UNIDENTIFIABLE:
                raise ValueError(
                    "an unavailable intervention-realization posterior must be unidentifiable"
                )
        elif not self.posterior.dimensions:
            raise ValueError("an intervention-realization posterior requires dimensions")
        if self.evidence_status is EvidenceStatus.OBSERVED and not self.evidence_event_ids:
            raise ValueError("an observed intervention realization requires evidence event IDs")
        if len(self.evidence_event_ids) != len(set(self.evidence_event_ids)):
            raise ValueError("intervention-realization evidence IDs must be unique")
        return self


class NuisanceBelief(SchemaModel):
    """Posterior over Xi: observation nuisance not automatically promoted to biology."""

    posterior: StateDistribution
    evidence_event_ids: tuple[str, ...] = ()

    @field_validator("evidence_event_ids")
    @classmethod
    def unique_evidence(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if len(values) != len(set(values)):
            raise ValueError("nuisance evidence IDs must be unique")
        return values


class ContextBelief(SchemaModel):
    active_interventions: tuple[InterventionEvent, ...] = ()
    soluble_environment: dict[str, JsonValue] = Field(default_factory=dict)
    physical_environment: dict[str, JsonValue] = Field(default_factory=dict)
    neighborhood: dict[str, JsonValue] = Field(default_factory=dict)
    spatial_position: dict[str, JsonValue] = Field(default_factory=dict)
    latent_context_posterior: StateDistribution | None = None
    unsupported_dimensions: tuple[str, ...] = ()

    @model_validator(mode="after")
    def deterministic_members_are_canonical(self) -> ContextBelief:
        intervention_ids = [event.event_id for event in self.active_interventions]
        if len(intervention_ids) != len(set(intervention_ids)):
            raise ValueError("active context intervention IDs must be unique")
        if any(not dimension.strip() for dimension in self.unsupported_dimensions):
            raise ValueError("unsupported context dimensions must be nonblank")
        if len(self.unsupported_dimensions) != len(set(self.unsupported_dimensions)):
            raise ValueError("unsupported context dimensions must be unique")
        for context_name, values in (
            ("soluble environment", self.soluble_environment),
            ("physical environment", self.physical_environment),
            ("neighborhood", self.neighborhood),
            ("spatial position", self.spatial_position),
        ):
            normalized = [key.casefold() for key in values]
            if any(not key.strip() for key in values) or len(normalized) != len(set(normalized)):
                raise ValueError(
                    f"{context_name} keys must be nonblank and case-insensitively unique"
                )
        return self


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

    @model_validator(mode="after")
    def hazards_and_timescale_are_nonnegative(self) -> DynamicSummary:
        for name, scalar in (
            ("division hazard", self.division_hazard),
            ("death hazard", self.death_hazard),
            ("recovery timescale", self.recovery_timescale),
        ):
            if scalar.value is not None and scalar.value < 0:
                raise ValueError(f"{name} must be nonnegative")
        return self


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


class EvaluationStatus(StrEnum):
    """Whether a diagnostic was calculated, separate from whether it passed."""

    EVALUATED = "evaluated"
    NOT_EVALUATED = "not_evaluated"
    UNSUPPORTED = "unsupported"


def _validate_evaluation_outcome(
    evaluation_status: EvaluationStatus, outcome: CriterionOutcome
) -> None:
    expected = {
        EvaluationStatus.NOT_EVALUATED: CriterionOutcome.NOT_EVALUATED,
        EvaluationStatus.UNSUPPORTED: CriterionOutcome.UNSUPPORTED,
    }
    if evaluation_status in expected and outcome is not expected[evaluation_status]:
        raise ValueError("diagnostic availability and scientific outcome disagree")
    if evaluation_status is EvaluationStatus.EVALUATED and outcome not in {
        CriterionOutcome.PASSED,
        CriterionOutcome.FAILED,
    }:
        raise ValueError("an evaluated diagnostic must explicitly pass or fail")


class SufficiencyReport(SchemaModel):
    evaluation_status: EvaluationStatus
    outcome: CriterionOutcome
    state_only_loss: float | None = None
    state_plus_history_loss: float | None = None
    history_information_gain: float | None = None
    markov_sufficiency_score: float | None = Field(default=None, ge=0, le=1)
    maximum_history_information_gain: float = Field(ge=0)
    metric: str | None = None
    residual_predictive_history_features: tuple[str, ...] = ()
    suspected_missing_state_variables: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    @model_validator(mode="after")
    def evaluated_together(self) -> SufficiencyReport:
        _validate_evaluation_outcome(self.evaluation_status, self.outcome)
        values = (
            self.state_only_loss,
            self.state_plus_history_loss,
            self.history_information_gain,
            self.markov_sufficiency_score,
        )
        if self.evaluation_status is EvaluationStatus.EVALUATED and any(
            value is None for value in values
        ):
            raise ValueError(
                "an evaluated sufficiency report requires losses, score, and threshold"
            )
        if self.evaluation_status is not EvaluationStatus.EVALUATED and any(
            value is not None for value in values
        ):
            raise ValueError("an unevaluated sufficiency report must not contain numeric sentinels")
        if self.evaluation_status is EvaluationStatus.EVALUATED:
            state_only = require_finite(cast(float, self.state_only_loss), name="state-only loss")
            state_plus_history = require_finite(
                cast(float, self.state_plus_history_loss), name="state-plus-history loss"
            )
            gain = require_finite(
                cast(float, self.history_information_gain), name="history information gain"
            )
            threshold = require_finite(
                self.maximum_history_information_gain, name="maximum history information gain"
            )
            if not math.isclose(gain, state_only - state_plus_history, rel_tol=1e-9, abs_tol=1e-12):
                raise ValueError(
                    "history information gain must equal state-only loss minus "
                    "state-plus-history loss"
                )
            expected = CriterionOutcome.PASSED if gain <= threshold else CriterionOutcome.FAILED
            if self.outcome is not expected:
                raise ValueError("sufficiency outcome must agree with its declared threshold")
        return self


class SupportReport(SchemaModel):
    evaluation_status: EvaluationStatus
    outcome: CriterionOutcome
    in_distribution_score: float | None = Field(default=None, ge=0, le=1)
    ood_score: float | None = Field(default=None, ge=0, le=1)
    maximum_ood_score: float = Field(ge=0, le=1)
    unsupported_subjects: tuple[str, ...] = ()
    unsupported_aggregations: tuple[str, ...] = ()
    unsupported_genotypes: tuple[str, ...] = ()
    unsupported_modalities: tuple[str, ...] = ()
    unsupported_interventions: tuple[str, ...] = ()
    unsupported_doses: tuple[str, ...] = ()
    unsupported_schedules: tuple[str, ...] = ()
    unsupported_delivery_methods: tuple[str, ...] = ()
    unsupported_environments: tuple[str, ...] = ()
    unsupported_combinations: tuple[str, ...] = ()
    unsupported_horizons: tuple[str, ...] = ()
    unsupported_outputs: tuple[str, ...] = ()
    unsupported_causal_classes: tuple[str, ...] = ()
    extrapolation_level: str | None = None
    abstention_required: bool
    notes: tuple[str, ...] = ()

    @model_validator(mode="after")
    def status_matches_scores(self) -> SupportReport:
        _validate_evaluation_outcome(self.evaluation_status, self.outcome)
        values = (self.in_distribution_score, self.ood_score)
        if self.evaluation_status is EvaluationStatus.EVALUATED and any(
            value is None for value in values
        ):
            raise ValueError("an evaluated support report requires ID, OOD, and threshold scores")
        if self.evaluation_status is not EvaluationStatus.EVALUATED and any(
            value is not None for value in values
        ):
            raise ValueError("an unevaluated support report must not contain numeric sentinels")
        if self.evaluation_status is EvaluationStatus.EVALUATED:
            blockers = any(
                (
                    self.unsupported_subjects,
                    self.unsupported_aggregations,
                    self.unsupported_genotypes,
                    self.unsupported_modalities,
                    self.unsupported_interventions,
                    self.unsupported_doses,
                    self.unsupported_schedules,
                    self.unsupported_delivery_methods,
                    self.unsupported_environments,
                    self.unsupported_combinations,
                    self.unsupported_horizons,
                    self.unsupported_outputs,
                    self.unsupported_causal_classes,
                )
            )
            passed = cast(float, self.ood_score) <= self.maximum_ood_score and not blockers
            expected = CriterionOutcome.PASSED if passed else CriterionOutcome.FAILED
            if self.outcome is not expected:
                raise ValueError("support outcome must agree with OOD threshold and blockers")
        if self.abstention_required is (self.outcome is CriterionOutcome.PASSED):
            raise ValueError("support abstention must be the inverse of a passing outcome")
        return self


class OODReport(SupportReport):
    """Backward import name for the expanded v2 model-support report."""


class DimensionIdentifiability(StrEnum):
    DIRECTLY_OBSERVED = "directly_observed"
    INFERRED_WITH_SUPPORT = "inferred_with_support"
    WEAKLY_IDENTIFIED = "weakly_identified"
    INTERVENTIONALLY_UNIDENTIFIED = "interventionally_unidentified"
    UNIDENTIFIABLE = "unidentifiable"
    OUTSIDE_MODEL_SUPPORT = "outside_model_support"
    OUT_OF_QUERY_SCOPE = "out_of_query_scope"


class ObservabilityReport(SchemaModel):
    """Compatibility summary retained for callers migrating from schema v1."""

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


class IdentifiabilityReport(SchemaModel):
    evaluation_status: EvaluationStatus
    outcome: CriterionOutcome
    dimension_status: dict[str, DimensionIdentifiability] = Field(default_factory=dict)
    identifiability_score: float | None = Field(default=None, ge=0, le=1)
    minimum_identifiability_score: float = Field(ge=0, le=1)
    observationally_equivalent_hypotheses: tuple[str, ...] = ()
    interventionally_distinct_equivalent_hypotheses: tuple[str, ...] = ()
    recommended_disambiguating_assays: tuple[str, ...] = ()
    recommended_disambiguating_interventions: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    @model_validator(mode="after")
    def outcome_matches_score(self) -> IdentifiabilityReport:
        _validate_evaluation_outcome(self.evaluation_status, self.outcome)
        values = (self.identifiability_score,)
        if self.evaluation_status is EvaluationStatus.EVALUATED and any(
            value is None for value in values
        ):
            raise ValueError("evaluated identifiability requires a score and threshold")
        if self.evaluation_status is not EvaluationStatus.EVALUATED and any(
            value is not None for value in values
        ):
            raise ValueError("unevaluated identifiability must not contain numeric sentinels")
        if self.evaluation_status is EvaluationStatus.EVALUATED:
            expected = (
                CriterionOutcome.PASSED
                if cast(float, self.identifiability_score) >= self.minimum_identifiability_score
                else CriterionOutcome.FAILED
            )
            if self.outcome is not expected:
                raise ValueError("identifiability outcome must agree with its threshold")
        return self


class DecisionUncertaintyReport(SchemaModel):
    evaluation_status: EvaluationStatus
    outcome: CriterionOutcome
    decision_uncertainty: float | None = Field(default=None, ge=0)
    maximum_decision_uncertainty: float = Field(ge=0)
    counterfactual_uncertainty: float | None = Field(default=None, ge=0)
    maximum_counterfactual_uncertainty: float = Field(ge=0)
    metric: str | None = None
    notes: tuple[str, ...] = ()

    @model_validator(mode="after")
    def outcome_matches_thresholds(self) -> DecisionUncertaintyReport:
        _validate_evaluation_outcome(self.evaluation_status, self.outcome)
        values = (
            self.decision_uncertainty,
            self.counterfactual_uncertainty,
        )
        if self.evaluation_status is EvaluationStatus.EVALUATED and any(
            value is None for value in values
        ):
            raise ValueError("evaluated decision uncertainty requires values and thresholds")
        if self.evaluation_status is not EvaluationStatus.EVALUATED and any(
            value is not None for value in values
        ):
            raise ValueError("unevaluated decision uncertainty cannot use numeric sentinels")
        if self.evaluation_status is EvaluationStatus.EVALUATED:
            passed = (
                cast(float, self.decision_uncertainty) <= self.maximum_decision_uncertainty
                and cast(float, self.counterfactual_uncertainty)
                <= self.maximum_counterfactual_uncertainty
            )
            expected = CriterionOutcome.PASSED if passed else CriterionOutcome.FAILED
            if self.outcome is not expected:
                raise ValueError("decision-uncertainty outcome must agree with its thresholds")
        return self


class CalibrationReport(SchemaModel):
    evaluation_status: EvaluationStatus
    outcome: CriterionOutcome
    empirical_coverage: float | None = Field(default=None, ge=0, le=1)
    minimum_coverage: float = Field(ge=0, le=1)
    calibration_error: float | None = Field(default=None, ge=0)
    maximum_calibration_error: float = Field(ge=0)
    metric: str | None = None
    notes: tuple[str, ...] = ()

    @model_validator(mode="after")
    def outcome_matches_coverage(self) -> CalibrationReport:
        _validate_evaluation_outcome(self.evaluation_status, self.outcome)
        values = (
            self.empirical_coverage,
            self.calibration_error,
        )
        if self.evaluation_status is EvaluationStatus.EVALUATED and any(
            value is None for value in values
        ):
            raise ValueError("evaluated calibration requires coverage and its threshold")
        if self.evaluation_status is not EvaluationStatus.EVALUATED and any(
            value is not None for value in values
        ):
            raise ValueError("unevaluated calibration cannot use numeric sentinels")
        if self.evaluation_status is EvaluationStatus.EVALUATED:
            expected = (
                CriterionOutcome.PASSED
                if (
                    cast(float, self.empirical_coverage) >= self.minimum_coverage
                    and cast(float, self.calibration_error) <= self.maximum_calibration_error
                )
                else CriterionOutcome.FAILED
            )
            if self.outcome is not expected:
                raise ValueError("calibration outcome must agree with its declared thresholds")
        return self


class CausalEstimandBinding(SchemaModel):
    """Typed intervention/environment contrast for one query target and horizon."""

    target: OntologyTerm
    horizon_name: str = Field(min_length=1)
    aggregation: TargetAggregation
    intervention_spec_ids: tuple[str, ...] = ()
    environment_variable_keys: tuple[str, ...] = ()
    comparator: str = Field(min_length=1)
    scenario_id: str | None = Field(default=None, min_length=1)
    scenario_fingerprint: str | None = Field(
        default=None,
        pattern=r"^[a-fA-F0-9]{64}$",
    )
    decision_set_fingerprint: str | None = Field(
        default=None,
        pattern=r"^[a-fA-F0-9]{64}$",
    )

    @model_validator(mode="after")
    def contrast_is_explicit(self) -> CausalEstimandBinding:
        if not self.comparator.strip():
            raise ValueError("a causal-estimand comparator must be nonblank")
        if any(not spec_id.strip() for spec_id in self.intervention_spec_ids):
            raise ValueError("causal-estimand intervention spec IDs must be nonblank")
        if not self.intervention_spec_ids and not self.environment_variable_keys:
            raise ValueError("a causal estimand must name an intervention or environment contrast")
        if len(self.intervention_spec_ids) != len(set(self.intervention_spec_ids)):
            raise ValueError("causal-estimand intervention spec IDs must be unique")
        if (self.scenario_id is None) is not (self.scenario_fingerprint is None):
            raise ValueError(
                "causal-estimand scenario ID and fingerprint must be declared together"
            )
        if self.scenario_id is not None and self.decision_set_fingerprint is not None:
            raise ValueError(
                "a causal estimand cannot bind one scenario and a complete decision set"
            )
        normalized_environment = [key.casefold() for key in self.environment_variable_keys]
        if any(not key.strip() for key in self.environment_variable_keys) or len(
            normalized_environment
        ) != len(set(normalized_environment)):
            raise ValueError("causal-estimand environment variables must be nonempty and unique")
        return self


class CausalSupportReport(SchemaModel):
    evaluation_status: EvaluationStatus
    outcome: CriterionOutcome
    causal_status: CausalStatus
    identification_basis: str | None = None
    identification_design: AssignmentMechanism | None = None
    estimands: tuple[CausalEstimandBinding, ...] = ()
    evidence_ids: tuple[str, ...] = ()
    evidence_fingerprints: dict[str, str] = Field(default_factory=dict)
    source_scope: str | None = None
    target_scope: str | None = None
    transport_assumptions: tuple[str, ...] = ()
    blockers: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    @model_validator(mode="after")
    def outcome_is_auditable(self) -> CausalSupportReport:
        _validate_evaluation_outcome(self.evaluation_status, self.outcome)
        optional_text = (
            self.identification_basis,
            self.source_scope,
            self.target_scope,
        )
        if any(value is not None and not value.strip() for value in optional_text):
            raise ValueError("causal identification basis and scopes must be nonblank")
        if any(not evidence_id.strip() for evidence_id in self.evidence_ids):
            raise ValueError("causal-support evidence IDs must be nonblank")
        if any(not assumption.strip() for assumption in self.transport_assumptions):
            raise ValueError("causal transport assumptions must be nonblank")
        if len(self.transport_assumptions) != len(set(self.transport_assumptions)):
            raise ValueError("causal transport assumptions must be unique")
        if len(self.evidence_ids) != len(set(self.evidence_ids)):
            raise ValueError("causal-support evidence IDs must be unique")
        if set(self.evidence_fingerprints) != set(self.evidence_ids):
            raise ValueError(
                "causal support requires exactly one fingerprint per evidence artifact"
            )
        if any(
            len(fingerprint) != 64
            or any(character not in "0123456789abcdefABCDEF" for character in fingerprint)
            for fingerprint in self.evidence_fingerprints.values()
        ):
            raise ValueError("causal evidence fingerprints must be SHA-256 hex digests")
        identified = self.causal_status in {
            CausalStatus.IDENTIFIED_POPULATION_EFFECT,
            CausalStatus.TRANSPORTED_UNDER_ASSUMPTIONS,
        }
        if identified:
            if not self.estimands or self.identification_design not in {
                AssignmentMechanism.RANDOMIZED,
                AssignmentMechanism.QUASI_EXPERIMENTAL,
            }:
                raise ValueError(
                    "identified causal support requires a typed estimand and an eligible "
                    "randomized or quasi-experimental design"
                )
            estimand_keys = [(item.target.key, item.horizon_name) for item in self.estimands]
            if len(estimand_keys) != len(set(estimand_keys)):
                raise ValueError("causal estimands must be unique by target and horizon")
        elif self.estimands or self.identification_design is not None:
            raise ValueError(
                "typed causal estimands and identification designs require identified status"
            )
        if self.evaluation_status is EvaluationStatus.EVALUATED:
            if (
                self.identification_basis is None
                or self.source_scope is None
                or self.target_scope is None
            ):
                raise ValueError(
                    "evaluated causal support requires an identification basis and scopes"
                )
            expected = (
                CriterionOutcome.PASSED
                if identified and not self.blockers
                else CriterionOutcome.FAILED
            )
            if self.outcome is not expected:
                raise ValueError(
                    "causal outcome must agree with identification status and blockers"
                )
            if self.outcome is CriterionOutcome.PASSED and not self.evidence_ids:
                raise ValueError("passing causal support requires identification evidence")
        elif self.causal_status is not CausalStatus.UNSUPPORTED:
            raise ValueError("unevaluated causal support must not claim causal identification")
        if self.causal_status is CausalStatus.TRANSPORTED_UNDER_ASSUMPTIONS:
            if (
                self.source_scope is None
                or self.target_scope is None
                or not self.transport_assumptions
            ):
                raise ValueError("transported causal support requires scopes and assumptions")
        elif self.transport_assumptions:
            raise ValueError("transport assumptions require transported causal status")
        return self


def _validate_causal_support_against_query(
    report: CausalSupportReport,
    query: StateQuery,
    *,
    required_target_horizons: set[tuple[str, str]] | None = None,
) -> None:
    """Fail closed unless an identified claim is the exact query estimand."""

    if report.causal_status not in {
        CausalStatus.IDENTIFIED_POPULATION_EFFECT,
        CausalStatus.TRANSPORTED_UNDER_ASSUMPTIONS,
    }:
        return
    assert report.estimands
    assert report.identification_design is not None
    outputs = {output.term.key: output for output in query.target_outputs}
    specs = {spec.spec_id: spec for spec in query.intervention_space}
    environment_keys = {spec.variable.key.casefold() for spec in query.environment_space}
    for estimand in report.estimands:
        output = outputs.get(estimand.target.key)
        if output is None:
            raise ValueError("causal estimand target is absent from the state query")
        if estimand.horizon_name not in output.supported_horizon_names:
            raise ValueError("causal estimand horizon is unsupported by its query target")
        if estimand.aggregation != output.aggregation:
            raise ValueError("causal estimand aggregation must match the query target")

        unknown_specs = set(estimand.intervention_spec_ids) - set(specs)
        if unknown_specs:
            raise ValueError(
                "causal estimand references interventions absent from the query: "
                f"{sorted(unknown_specs)}"
            )
        unknown_environment = {
            key.casefold() for key in estimand.environment_variable_keys
        } - environment_keys
        if unknown_environment:
            raise ValueError(
                "causal estimand references environment variables absent from the query: "
                f"{sorted(unknown_environment)}"
            )

        for spec_id in estimand.intervention_spec_ids:
            spec = specs[spec_id]
            if report.identification_design is AssignmentMechanism.RANDOMIZED:
                eligible = (
                    AssignmentMechanism.RANDOMIZED in spec.allowed_assignment_mechanisms
                    and spec.require_randomization_unit
                    and spec.randomization_unit_kind is not None
                )
            else:
                eligible = (
                    spec.require_matched_control
                    and AssignmentMechanism.QUASI_EXPERIMENTAL in spec.allowed_assignment_mechanisms
                )
            if not eligible:
                raise ValueError(
                    f"query intervention {spec_id!r} does not support the declared causal design"
                )

    actual_target_horizons = {
        (estimand.target.key, estimand.horizon_name) for estimand in report.estimands
    }
    expected_target_horizons = required_target_horizons
    if expected_target_horizons is None:
        expected_target_horizons = {
            (output.term.key, horizon_name)
            for output in query.target_outputs
            for horizon_name in output.supported_horizon_names
        }
    if actual_target_horizons != expected_target_horizons:
        raise ValueError(
            "causal estimands must exactly cover the required query target and horizon scope"
        )


def _validate_identified_evidence_provenance(
    report: CausalSupportReport,
    provenance: ProvenanceRecord,
) -> None:
    """Require identified effects to cite content-addressed validation claim artifacts.

    Ordinary history events condition a belief but do not by themselves identify a population
    effect. Local experimental identification will require a future typed assignment/control/
    outcome graph; until that contract exists, it fails closed here.
    """

    if report.outcome is not CriterionOutcome.PASSED:
        return
    if report.causal_status not in {
        CausalStatus.IDENTIFIED_POPULATION_EFFECT,
        CausalStatus.TRANSPORTED_UNDER_ASSUMPTIONS,
    }:
        return
    if (
        provenance.support_envelope_id is None
        or provenance.support_envelope_fingerprint is None
        or provenance.training_support_id is None
        or provenance.training_support_fingerprint is None
    ):
        raise ValueError(
            "identified causal support requires content-addressed support envelope and "
            "training support provenance"
        )
    evidence_ids = set(report.evidence_ids)
    if not evidence_ids <= set(provenance.validation_evidence_ids):
        raise ValueError(
            "identified causal support must cite external validation claim artifacts, not "
            "ordinary history events"
        )
    expected = {
        evidence_id: provenance.validation_evidence_fingerprints[evidence_id]
        for evidence_id in evidence_ids
    }
    if report.evidence_fingerprints != expected:
        raise ValueError(
            "causal evidence fingerprints must match the content-addressed validation artifacts"
        )


class QueryReadinessReport(SchemaModel):
    support: CriterionOutcome
    sufficiency: CriterionOutcome
    identifiability: CriterionOutcome
    decision_uncertainty: CriterionOutcome
    calibration: CriterionOutcome
    causal: CriterionOutcome
    measurement_model: CriterionOutcome
    control_requested: bool
    valid_for_prediction: bool
    valid_for_control: bool
    valid_for_measurement_selection: bool
    abstention_required: bool
    reasons: tuple[str, ...] = ()

    @model_validator(mode="after")
    def flags_are_derived_fail_closed(self) -> QueryReadinessReport:
        prediction = all(
            outcome is CriterionOutcome.PASSED
            for outcome in (
                self.support,
                self.sufficiency,
                self.identifiability,
                self.calibration,
            )
        )
        control = (
            self.control_requested
            and prediction
            and self.decision_uncertainty is CriterionOutcome.PASSED
            and self.causal is CriterionOutcome.PASSED
        )
        measurement = (
            self.support is CriterionOutcome.PASSED
            and self.measurement_model is CriterionOutcome.PASSED
        )
        abstention = not prediction or (self.control_requested and not control)
        if self.valid_for_prediction is not prediction:
            raise ValueError("prediction-readiness flag does not match diagnostic outcomes")
        if self.valid_for_control is not control:
            raise ValueError("control-readiness flag does not match diagnostic outcomes")
        if self.valid_for_measurement_selection is not measurement:
            raise ValueError("measurement-readiness flag does not match diagnostic outcomes")
        if self.abstention_required is not abstention:
            raise ValueError("abstention flag does not match requested-task readiness")
        if abstention and not self.reasons:
            raise ValueError("an abstaining readiness report requires explicit reasons")
        return self


class BeliefDiagnostics(SchemaModel):
    support: SupportReport
    sufficiency: SufficiencyReport
    identifiability: IdentifiabilityReport
    decision_uncertainty: DecisionUncertaintyReport
    calibration: CalibrationReport
    causal_support: CausalSupportReport
    constraint_residuals: dict[str, float] = Field(default_factory=dict)


class BeliefStatus(StrEnum):
    """Structural availability only; it does not assert scientific validity."""

    COMPLETE = "complete"
    PARTIAL = "partial"
    UNAVAILABLE = "unavailable"


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
        raise ValueError(f"parametric {label} posterior must match the joint marginal")


class CellStateBelief(SchemaModel):
    """Query-conditioned posterior over the causally relevant current state."""

    schema_version: SchemaVersion = SCHEMA_VERSION
    belief_id: UUID = Field(default_factory=uuid4)
    subject: BeliefSubject
    as_of_seconds: float
    query: StateQuery
    query_fingerprint: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    history_fingerprint: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    context_fingerprint: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    state_specification: CompiledStateSpecification
    status: BeliefStatus
    joint_posterior: StateDistribution
    factors: tuple[FactorBelief, ...]
    context: ContextBelief
    intervention_realizations: tuple[InterventionRealizationBelief, ...] = ()
    nuisance: NuisanceBelief | None = None
    dynamics: DynamicSummary
    uncertainty: UncertaintyBreakdown
    diagnostics: BeliefDiagnostics
    readiness: QueryReadinessReport
    provenance: ProvenanceRecord

    @property
    def subject_id(self) -> str:
        return self.subject.subject_id

    @field_validator("as_of_seconds")
    @classmethod
    def finite_as_of(cls, value: float) -> float:
        return require_finite(value, name="belief time")

    @model_validator(mode="after")
    def coherent_contract(self) -> CellStateBelief:
        if not self.subject.is_compatible_with(self.query.subject):
            raise ValueError("belief subject must satisfy the query subject specification")
        if self.state_specification.subject != self.query.subject:
            raise ValueError("compiled state subject must match the query subject")
        if self.state_specification.query_fingerprint != self.query_fingerprint:
            raise ValueError("compiled state/query fingerprints must agree")
        if self.state_specification.acceptance_thresholds != self.query.acceptance_thresholds:
            raise ValueError("compiled acceptance thresholds must match the query")
        if set(self.state_specification.target_output_keys) != {
            output.term.key for output in self.query.target_outputs
        }:
            raise ValueError("compiled targets must match the query outputs")
        if set(self.state_specification.horizon_names) != {
            horizon.name for horizon in self.query.prediction_horizons
        }:
            raise ValueError("compiled horizons must match the query horizons")
        if set(self.state_specification.admissible_evidence_roles) != set(
            self.query.evidence_policy.allowed_evidence_roles
        ):
            raise ValueError("compiled evidence roles must match the query evidence policy")
        if self.provenance.query_fingerprint != self.query_fingerprint:
            raise ValueError("provenance/query fingerprints must agree")
        if self.query.fingerprint != self.query_fingerprint:
            raise ValueError("embedded query and query fingerprint must agree")
        if self.provenance.history_fingerprint != self.history_fingerprint:
            raise ValueError("provenance/history fingerprints must agree")
        if self.provenance.context_fingerprint != self.context_fingerprint:
            raise ValueError("provenance/context fingerprints must agree")

        active = self.state_specification.active_factor_map
        factors = {factor.factor: factor for factor in self.factors}
        if len(factors) != len(self.factors) or set(factors) != set(active):
            raise ValueError("belief factors must equal the query-compiled active factors")

        expected_joint_dimensions = set(self.state_specification.joint_dimensions)
        if set(self.joint_posterior.dimensions) != expected_joint_dimensions:
            raise ValueError("joint posterior dimensions must equal the compiled state")

        source_event_ids = set(self.provenance.source_event_ids)
        for factor_kind, factor in factors.items():
            specification = active[factor_kind]
            if set(factor.posterior.dimensions) != set(specification.dimensions):
                raise ValueError("factor posterior dimensions must match the compiled factor")
            if factor.timescales != specification.timescales:
                raise ValueError("factor timescales must match the compiled factor")
            if not set(factor.evidence_event_ids) <= source_event_ids:
                raise ValueError("factor evidence IDs must appear in belief provenance")
            _require_parametric_marginal(
                self.joint_posterior, factor.posterior, label=factor.factor.value
            )

        context_dimensions = set(self.state_specification.context_modulator_dimensions)
        if context_dimensions:
            if self.context.latent_context_posterior is None:
                raise ValueError("compiled context modulators require a context posterior")
            if set(self.context.latent_context_posterior.dimensions) != context_dimensions:
                raise ValueError("context posterior dimensions must match the compiled state")
            _require_parametric_marginal(
                self.joint_posterior,
                self.context.latent_context_posterior,
                label="context",
            )
        elif self.context.latent_context_posterior is not None:
            raise ValueError("an out-of-query context posterior must not be emitted")

        realization_ids = [item.intervention_event_id for item in self.intervention_realizations]
        if len(realization_ids) != len(set(realization_ids)):
            raise ValueError("intervention-realization blocks must name unique interventions")
        if not set(realization_ids) <= source_event_ids:
            raise ValueError(
                "intervention-realization blocks must reference interventions in provenance"
            )
        realization_dimensions = [
            dimension
            for realization in self.intervention_realizations
            for dimension in realization.posterior.dimensions
        ]
        if len(realization_dimensions) != len(set(realization_dimensions)) or set(
            realization_dimensions
        ) != set(self.state_specification.intervention_realization_dimensions):
            raise ValueError(
                "intervention-realization blocks must partition the compiled realization state"
            )
        for realization in self.intervention_realizations:
            if not set(realization.evidence_event_ids) <= source_event_ids:
                raise ValueError("intervention-realization evidence must appear in provenance")
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
                raise ValueError("Xi nuisance posterior must match the compiled nuisance state")
            if not set(self.nuisance.evidence_event_ids) <= source_event_ids:
                raise ValueError("nuisance evidence IDs must appear in provenance")
            _require_parametric_marginal(
                self.joint_posterior,
                self.nuisance.posterior,
                label="Xi nuisance",
            )
        elif self.nuisance is not None:
            raise ValueError("an out-of-query nuisance posterior must not be emitted")

        if set(self.diagnostics.identifiability.dimension_status) != expected_joint_dimensions:
            raise ValueError("identifiability must classify every active state dimension")
        if self.readiness.support is not self.diagnostics.support.outcome:
            raise ValueError("readiness/support outcomes must agree")
        if self.readiness.sufficiency is not self.diagnostics.sufficiency.outcome:
            raise ValueError("readiness/sufficiency outcomes must agree")
        if self.readiness.identifiability is not self.diagnostics.identifiability.outcome:
            raise ValueError("readiness/identifiability outcomes must agree")
        if self.readiness.decision_uncertainty is not self.diagnostics.decision_uncertainty.outcome:
            raise ValueError("readiness/decision-uncertainty outcomes must agree")
        if self.readiness.calibration is not self.diagnostics.calibration.outcome:
            raise ValueError("readiness/calibration outcomes must agree")
        if self.readiness.causal is not self.diagnostics.causal_support.outcome:
            raise ValueError("readiness/causal-support outcomes must agree")
        if (
            not set(self.diagnostics.causal_support.evidence_ids)
            <= self.provenance.scientific_evidence_ids
        ):
            raise ValueError("causal-support evidence must appear in belief provenance")
        _validate_causal_support_against_query(self.diagnostics.causal_support, self.query)
        if any(
            estimand.scenario_id is not None or estimand.decision_set_fingerprint is not None
            for estimand in self.diagnostics.causal_support.estimands
        ):
            raise ValueError(
                "current-state causal support cannot bind a future scenario or decision set"
            )
        _validate_identified_evidence_provenance(
            self.diagnostics.causal_support,
            self.provenance,
        )
        if self.readiness.control_requested is not bool(self.query.intervention_space):
            raise ValueError(
                "readiness must reflect whether the query requests intervention control"
            )

        thresholds = self.query.acceptance_thresholds
        if self.diagnostics.support.maximum_ood_score != thresholds.maximum_ood_score:
            raise ValueError("support report must use the query OOD threshold")
        if (
            self.diagnostics.sufficiency.maximum_history_information_gain
            != thresholds.maximum_history_information_gain
        ):
            raise ValueError("sufficiency report must use the query history threshold")
        if (
            self.diagnostics.identifiability.minimum_identifiability_score
            != thresholds.minimum_identifiability
        ):
            raise ValueError("identifiability report must use the query threshold")
        decision = self.diagnostics.decision_uncertainty
        if (
            decision.maximum_decision_uncertainty != thresholds.maximum_decision_uncertainty
            or decision.maximum_counterfactual_uncertainty
            != thresholds.maximum_counterfactual_uncertainty
        ):
            raise ValueError("decision uncertainty report must use the query thresholds")
        calibration = self.diagnostics.calibration
        if (
            calibration.minimum_coverage != thresholds.minimum_calibration_coverage
            or calibration.maximum_calibration_error != thresholds.maximum_calibration_error
        ):
            raise ValueError("calibration report must use the query thresholds")
        if (
            self.diagnostics.causal_support.causal_status
            is CausalStatus.TRANSPORTED_UNDER_ASSUMPTIONS
            and not self.query.constraints.allow_transport
        ):
            raise ValueError("query constraints forbid transported causal support")

        joint_unavailable = isinstance(self.joint_posterior, UnavailableDistribution)
        if self.status is BeliefStatus.UNAVAILABLE and not joint_unavailable:
            raise ValueError("an unavailable belief requires an unavailable joint posterior")
        if self.status is not BeliefStatus.UNAVAILABLE and joint_unavailable:
            raise ValueError("a structurally usable belief requires a joint posterior")
        if self.status is BeliefStatus.COMPLETE:
            posterior_blocks: tuple[StateDistribution, ...] = (
                *(factor.posterior for factor in self.factors),
                *(item.posterior for item in self.intervention_realizations),
                *(
                    (self.context.latent_context_posterior,)
                    if self.context.latent_context_posterior is not None
                    else ()
                ),
                *((self.nuisance.posterior,) if self.nuisance is not None else ()),
            )
            if any(isinstance(block, UnavailableDistribution) for block in posterior_blocks):
                raise ValueError("a structurally complete belief requires every active posterior")
        return self


__all__ = [
    "BeliefDiagnostics",
    "BeliefStatus",
    "CalibrationReport",
    "CausalSupportReport",
    "CellStateBelief",
    "ContextBelief",
    "DecisionUncertaintyReport",
    "DimensionIdentifiability",
    "DistributionSupport",
    "DynamicSummary",
    "EvaluatedScalar",
    "EvaluationStatus",
    "EventHazard",
    "FactorBelief",
    "FateProbability",
    "IdentifiabilityReport",
    "InterventionRealizationBelief",
    "NuisanceBelief",
    "OODReport",
    "ObservabilityReport",
    "ParametricDistribution",
    "QueryReadinessReport",
    "SampleDistribution",
    "StateFactor",
    "SufficiencyReport",
    "SupportReport",
    "UnavailableDistribution",
    "UncertaintyBreakdown",
    "UncertaintyComponent",
    "UncertaintyKind",
]
