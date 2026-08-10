"""Standalone decision-oriented next-measurement contracts."""

from __future__ import annotations

import math
from collections.abc import Sequence
from enum import StrEnum
from uuid import UUID

from pydantic import Field, field_validator, model_validator

from .belief import (
    CausalSupportReport,
    QueryReadinessReport,
    _validate_identified_evidence_provenance,
)
from .common import (
    SCHEMA_VERSION,
    CausalStatus,
    CriterionOutcome,
    ProvenanceRecord,
    SchemaModel,
    SchemaVersion,
    SupportStatus,
    canonical_fingerprint,
    require_finite,
)
from .events import CollectionEffect
from .scenarios import (
    EvolutionScenario,
    InterventionObjective,
    ScenarioReference,
    TransportReport,
    TransportStatus,
)


class MeasurementValueBasis(StrEnum):
    """What a supported numeric measurement value actually represents."""

    INTERVENTION_DECISION_EVSI = "intervention_decision_evsi"


class MeasurementInformationScope(StrEnum):
    """Decision-relevant information scope; private latent entropy is deliberately absent."""

    QUERY_TARGETS = "query_targets"
    INTERVENTION_OUTCOMES = "intervention_outcomes"
    DECISION_REGRET = "decision_regret"


class MeasurementDecisionStatus(StrEnum):
    RECOMMENDED = "recommended"
    ABSTAINED = "abstained"
    NOT_EVALUATED = "not_evaluated"
    UNSUPPORTED = "unsupported"


class MeasurementEvidenceCriterion(StrEnum):
    """The four scientific calculations required for decision-oriented EVSI."""

    ASSAY_OUTCOME_MODEL = "assay_outcome_model"
    HYPOTHETICAL_UPDATE = "hypothetical_update"
    EXACT_CANDIDATE_COUNTERFACTUAL_REPLANNING = "exact_candidate_counterfactual_replanning"
    DECISION_UTILITY = "decision_utility"


class MeasurementEvidenceTrace(SchemaModel):
    """Content-addressed support for one EVSI criterion on one exact request scope."""

    criterion: MeasurementEvidenceCriterion
    outcome: CriterionOutcome
    scope_fingerprint: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    evidence_ids: tuple[str, ...] = ()
    evidence_fingerprints: dict[str, str] = Field(default_factory=dict)
    reasons: tuple[str, ...] = ()

    @model_validator(mode="after")
    def evidence_matches_outcome(self) -> MeasurementEvidenceTrace:
        if any(not evidence_id.strip() for evidence_id in self.evidence_ids):
            raise ValueError("measurement criterion evidence IDs must be nonblank")
        if len(self.evidence_ids) != len(set(self.evidence_ids)):
            raise ValueError("measurement criterion evidence IDs must be unique")
        if set(self.evidence_fingerprints) != set(self.evidence_ids):
            raise ValueError("measurement criterion evidence requires one fingerprint per artifact")
        if any(
            len(fingerprint) != 64
            or any(character not in "0123456789abcdefABCDEF" for character in fingerprint)
            for fingerprint in self.evidence_fingerprints.values()
        ):
            raise ValueError("measurement criterion fingerprints must be SHA-256 hex digests")
        if any(not reason.strip() for reason in self.reasons):
            raise ValueError("measurement criterion reasons must be nonblank")

        if self.outcome is CriterionOutcome.PASSED and not self.evidence_ids:
            raise ValueError("a passing measurement criterion requires content-addressed evidence")
        if self.outcome is CriterionOutcome.FAILED and (not self.evidence_ids or not self.reasons):
            raise ValueError(
                "a failed measurement criterion requires evidence and explicit reasons"
            )
        if self.outcome in {
            CriterionOutcome.NOT_EVALUATED,
            CriterionOutcome.UNSUPPORTED,
        }:
            if self.evidence_ids or self.evidence_fingerprints:
                raise ValueError(
                    "an unavailable measurement criterion cannot claim evidence sentinels"
                )
            if not self.reasons:
                raise ValueError("an unavailable measurement criterion requires explicit reasons")
        return self


class AssayReference(SchemaModel):
    assay_id: str = Field(min_length=1)
    fingerprint: str = Field(pattern=r"^[a-fA-F0-9]{64}$")

    @field_validator("assay_id")
    @classmethod
    def nonblank_assay_id(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("assay reference ID must be nonblank")
        return value


def measurement_decision_set_fingerprint(
    candidates: Sequence[EvolutionScenario | ScenarioReference],
) -> str:
    """Bind causal support to the exact ordered, content-addressed decision set."""

    references = tuple(
        candidate
        if isinstance(candidate, ScenarioReference)
        else ScenarioReference(
            scenario_id=candidate.scenario_id,
            fingerprint=canonical_fingerprint(candidate),
        )
        for candidate in candidates
    )
    return canonical_fingerprint(
        {"ordered_candidates": [reference.model_dump(mode="json") for reference in references]}
    )


class MeasurementDecisionRequest(SchemaModel):
    """One auditable decision problem for buying information before acting."""

    schema_version: SchemaVersion = SCHEMA_VERSION
    request_id: str = Field(min_length=1)
    parent_belief_id: UUID
    query_fingerprint: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    objective: InterventionObjective
    candidates: tuple[EvolutionScenario, ...] = Field(min_length=2)
    candidate_assay_ids: tuple[str, ...] = Field(min_length=1)
    collection_time_seconds: float
    decision_deadline_seconds: float
    minimum_net_decision_value: float = Field(default=0, ge=0)
    utility_units: str = Field(min_length=1)
    assay_cost_to_utility_rate: float = Field(gt=0)
    delay_penalty_per_second: float = Field(ge=0)
    destructiveness_penalties: dict[CollectionEffect, float]

    @field_validator("request_id", "utility_units")
    @classmethod
    def required_text_is_nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("measurement-decision identifiers and units must be nonblank")
        return value

    @field_validator(
        "collection_time_seconds",
        "decision_deadline_seconds",
        "minimum_net_decision_value",
        "assay_cost_to_utility_rate",
        "delay_penalty_per_second",
    )
    @classmethod
    def finite_values(cls, value: float) -> float:
        return require_finite(value, name="measurement-decision value")

    @field_validator("candidate_assay_ids")
    @classmethod
    def assay_ids_are_unique(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if any(not value.strip() for value in values):
            raise ValueError("candidate assay IDs must be nonempty")
        if len(values) != len(set(values)):
            raise ValueError("candidate assay IDs must be unique")
        return values

    @model_validator(mode="after")
    def decision_problem_is_coherent(self) -> MeasurementDecisionRequest:
        scenario_ids = [candidate.scenario_id for candidate in self.candidates]
        if len(scenario_ids) != len(set(scenario_ids)):
            raise ValueError("measurement-decision candidate scenario IDs must be unique")
        if any(
            candidate.horizon_name != self.objective.horizon_name for candidate in self.candidates
        ):
            raise ValueError("measurement candidates must use the objective horizon")
        first_subject = self.candidates[0].subject
        if any(candidate.subject != first_subject for candidate in self.candidates[1:]):
            raise ValueError("measurement candidates must use one typed subject")
        if self.decision_deadline_seconds < self.collection_time_seconds:
            raise ValueError("measurement decision deadline cannot predate collection")
        if set(self.destructiveness_penalties) != set(CollectionEffect):
            raise ValueError("measurement request must price every collection effect explicitly")
        if any(
            not math.isfinite(penalty) or penalty < 0
            for penalty in self.destructiveness_penalties.values()
        ):
            raise ValueError("destructiveness penalties must be finite and nonnegative")
        if self.destructiveness_penalties[CollectionEffect.NONDESTRUCTIVE] != 0:
            raise ValueError("nondestructive collection must have zero destruction penalty")
        return self

    @property
    def fingerprint(self) -> str:
        return canonical_fingerprint(self)


class AssayEvaluation(SchemaModel):
    """Decision value and scientific support for one candidate assay."""

    assay: AssayReference
    status: SupportStatus
    collection_effect: CollectionEffect
    value_basis: MeasurementValueBasis | None = None
    information_scope: MeasurementInformationScope | None = None
    baseline_best_expected_utility: float | None = None
    expected_post_measurement_best_utility: float | None = None
    gross_expected_decision_value: float | None = Field(default=None, ge=0)
    expected_information_gain: float | None = Field(default=None, ge=0)
    information_gain_metric: str | None = Field(default=None, min_length=1)
    expected_posterior_uncertainty_reduction: float | None = Field(default=None, ge=0)
    uncertainty_reduction_metric: str | None = Field(default=None, min_length=1)
    expected_intervention_ranking_change: float | None = Field(default=None, ge=0)
    ranking_change_metric: str | None = Field(default=None, min_length=1)
    raw_assay_cost: float | None = Field(default=None, ge=0)
    assay_cost_units: str | None = Field(default=None, min_length=1)
    assay_cost_penalty: float | None = Field(default=None, ge=0)
    expected_delay_seconds: float | None = Field(default=None, ge=0)
    delay_cost: float | None = Field(default=None, ge=0)
    destructiveness_cost: float | None = Field(default=None, ge=0)
    net_expected_decision_value: float | None = None
    evidence_traces: tuple[MeasurementEvidenceTrace, ...] = Field(
        min_length=4,
        max_length=4,
    )
    measurement_model_evidence_ids: tuple[str, ...] = ()
    measurement_model_evidence_fingerprints: dict[str, str] = Field(default_factory=dict)
    reasons: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    @property
    def canonical_net_decision_value(self) -> float | None:
        """Recompute the decision score used for thresholding and selection."""

        if self.status is not SupportStatus.SUPPORTED:
            return None
        assert self.baseline_best_expected_utility is not None
        assert self.expected_post_measurement_best_utility is not None
        assert self.assay_cost_penalty is not None
        assert self.delay_cost is not None
        assert self.destructiveness_cost is not None
        return (
            self.expected_post_measurement_best_utility
            - self.baseline_best_expected_utility
            - self.assay_cost_penalty
            - self.delay_cost
            - self.destructiveness_cost
        )

    @model_validator(mode="before")
    @classmethod
    def explanatory_text_is_nonblank(cls, value: object) -> object:
        if not isinstance(value, dict):
            return value
        optional_text = (
            "information_gain_metric",
            "uncertainty_reduction_metric",
            "ranking_change_metric",
            "assay_cost_units",
        )
        if any(
            value.get(field_name) is not None and not str(value[field_name]).strip()
            for field_name in optional_text
        ):
            raise ValueError("assay evaluation metrics and units must be nonblank")
        for field_name in ("reasons", "notes"):
            entries = value.get(field_name, ())
            if any(not str(entry).strip() for entry in entries):
                raise ValueError("assay evaluation reasons and notes must be nonblank")
        return value

    @model_validator(mode="after")
    def status_and_value_are_coherent(self) -> AssayEvaluation:
        expected_criteria = tuple(MeasurementEvidenceCriterion)
        actual_criteria = tuple(trace.criterion for trace in self.evidence_traces)
        if actual_criteria != expected_criteria:
            raise ValueError(
                "assay evaluation requires each EVSI evidence criterion once in canonical order"
            )
        if self.status is SupportStatus.SUPPORTED and any(
            trace.outcome is not CriterionOutcome.PASSED for trace in self.evidence_traces
        ):
            raise ValueError("a supported numeric assay evaluation requires four passing traces")
        if any(not evidence_id.strip() for evidence_id in self.measurement_model_evidence_ids):
            raise ValueError("measurement-model evidence IDs must be nonblank")
        if len(self.measurement_model_evidence_ids) != len(
            set(self.measurement_model_evidence_ids)
        ):
            raise ValueError("measurement-model evidence IDs must be unique")
        numeric_values = (
            self.baseline_best_expected_utility,
            self.expected_post_measurement_best_utility,
            self.gross_expected_decision_value,
            self.expected_information_gain,
            self.expected_posterior_uncertainty_reduction,
            self.expected_intervention_ranking_change,
            self.raw_assay_cost,
            self.assay_cost_penalty,
            self.expected_delay_seconds,
            self.delay_cost,
            self.destructiveness_cost,
            self.net_expected_decision_value,
        )
        metric_values = (
            self.value_basis,
            self.information_scope,
            self.information_gain_metric,
            self.uncertainty_reduction_metric,
            self.ranking_change_metric,
            self.assay_cost_units,
        )
        if self.status is SupportStatus.SUPPORTED:
            if any(value is None for value in (*numeric_values, *metric_values)):
                raise ValueError("a supported assay evaluation requires every value and metric")
            if self.value_basis is not MeasurementValueBasis.INTERVENTION_DECISION_EVSI:
                raise ValueError("supported measurement value must be intervention EVSI")
            if not self.measurement_model_evidence_ids:
                raise ValueError(
                    "supported assay evaluation requires measurement-model validation evidence"
                )
            if self.reasons:
                raise ValueError("a supported assay evaluation cannot contain support blockers")
            outcome_trace = self.evidence_traces[0]
            if (
                self.measurement_model_evidence_ids != outcome_trace.evidence_ids
                or self.measurement_model_evidence_fingerprints
                != outcome_trace.evidence_fingerprints
            ):
                raise ValueError(
                    "measurement-model evidence must match the assay-outcome evidence trace"
                )
        else:
            if any(value is not None for value in (*numeric_values, *metric_values)):
                raise ValueError(
                    "unsupported or unevaluated assays must not use numeric or metric sentinels"
                )
            if self.measurement_model_evidence_ids or self.measurement_model_evidence_fingerprints:
                raise ValueError(
                    "unsupported or unevaluated assays cannot claim measurement-model evidence"
                )
            if not self.reasons:
                raise ValueError("an unavailable assay evaluation requires explicit reasons")

        if set(self.measurement_model_evidence_fingerprints) != set(
            self.measurement_model_evidence_ids
        ):
            raise ValueError(
                "assay evaluation requires one fingerprint per measurement-model artifact"
            )
        if any(
            len(fingerprint) != 64
            or any(character not in "0123456789abcdefABCDEF" for character in fingerprint)
            for fingerprint in self.measurement_model_evidence_fingerprints.values()
        ):
            raise ValueError("measurement-model fingerprints must be SHA-256 hex digests")
        if any(value is not None and not math.isfinite(value) for value in numeric_values):
            raise ValueError("assay evaluation values must be finite")

        if self.status is SupportStatus.SUPPORTED:
            assert self.baseline_best_expected_utility is not None
            assert self.expected_post_measurement_best_utility is not None
            assert self.gross_expected_decision_value is not None
            assert self.assay_cost_penalty is not None
            assert self.delay_cost is not None
            assert self.destructiveness_cost is not None
            assert self.net_expected_decision_value is not None
            gross = (
                self.expected_post_measurement_best_utility - self.baseline_best_expected_utility
            )
            if not math.isclose(
                self.gross_expected_decision_value,
                gross,
                rel_tol=0,
                abs_tol=1e-9,
            ):
                raise ValueError(
                    "gross EVSI must equal post-measurement minus baseline best utility"
                )
            net = (
                self.gross_expected_decision_value
                - self.assay_cost_penalty
                - self.delay_cost
                - self.destructiveness_cost
            )
            if not math.isclose(
                self.net_expected_decision_value,
                net,
                rel_tol=0,
                abs_tol=1e-9,
            ):
                raise ValueError("net measurement value must equal EVSI minus all penalties")
        return self


class MeasurementRecommendation(SchemaModel):
    """Auditable assay decision; deliberately separate from the belief itself."""

    schema_version: SchemaVersion = SCHEMA_VERSION
    recommendation_id: str = Field(min_length=1)
    status: MeasurementDecisionStatus
    parent_belief_id: UUID
    query_fingerprint: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    request_id: str = Field(min_length=1)
    request_fingerprint: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    objective_id: str = Field(min_length=1)
    objective_fingerprint: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    candidates: tuple[ScenarioReference, ...] = Field(min_length=2)
    assays: tuple[AssayReference, ...] = Field(min_length=1)
    selected_assay_id: str | None = None
    evaluations: tuple[AssayEvaluation, ...]
    readiness: QueryReadinessReport
    causal_status: CausalStatus
    causal_support: CausalSupportReport
    transport: TransportReport
    minimum_net_decision_value: float = Field(ge=0)
    utility_units: str = Field(min_length=1)
    abstention_reasons: tuple[str, ...] = ()
    rationale: str = Field(min_length=1)
    seed: int = Field(ge=0)
    provenance: ProvenanceRecord

    @field_validator(
        "recommendation_id",
        "request_id",
        "objective_id",
        "utility_units",
        "rationale",
    )
    @classmethod
    def required_text_is_nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError(
                "measurement result identifiers, units, and rationale must be nonblank"
            )
        return value

    @field_validator("abstention_reasons")
    @classmethod
    def abstention_reasons_are_nonblank(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if any(not value.strip() for value in values):
            raise ValueError("measurement abstention reasons must be nonblank")
        return values

    @model_validator(mode="after")
    def decision_is_complete_and_auditable(self) -> MeasurementRecommendation:
        if not self.rationale.strip():
            raise ValueError("measurement rationale must be nonblank")
        candidate_ids = [candidate.scenario_id for candidate in self.candidates]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("measurement candidate scenario IDs must be unique")
        assay_ids = [assay.assay_id for assay in self.assays]
        if len(assay_ids) != len(set(assay_ids)):
            raise ValueError("measurement candidate assay IDs must be unique")
        evaluation_ids = [evaluation.assay.assay_id for evaluation in self.evaluations]
        if evaluation_ids != assay_ids:
            raise ValueError(
                "measurement result must evaluate every requested assay once and in order"
            )
        if any(
            evaluation.assay != assay
            for evaluation, assay in zip(self.evaluations, self.assays, strict=True)
        ):
            raise ValueError("assay evaluation references must match the requested assays")
        if self.provenance.query_fingerprint != self.query_fingerprint:
            raise ValueError("measurement provenance/query fingerprints must agree")
        if self.causal_support.causal_status is not self.causal_status:
            raise ValueError("measurement causal status must match its support report")
        if self.causal_support.outcome is not self.readiness.causal:
            raise ValueError("measurement causal support must match readiness")

        for evaluation in self.evaluations:
            trace_evidence_ids = {
                evidence_id
                for trace in evaluation.evidence_traces
                for evidence_id in trace.evidence_ids
            }
            if not trace_evidence_ids <= set(self.provenance.validation_evidence_ids):
                raise ValueError(
                    "measurement criterion evidence must cite validation artifacts in provenance"
                )
            for trace in evaluation.evidence_traces:
                expected_trace_fingerprints = {
                    evidence_id: self.provenance.validation_evidence_fingerprints[evidence_id]
                    for evidence_id in trace.evidence_ids
                }
                if trace.evidence_fingerprints != expected_trace_fingerprints:
                    raise ValueError(
                        "measurement criterion evidence fingerprints must match provenance"
                    )
            if evaluation.status is not SupportStatus.SUPPORTED:
                continue
            evidence_ids = set(evaluation.measurement_model_evidence_ids)
            if not evidence_ids <= set(self.provenance.validation_evidence_ids):
                raise ValueError(
                    "measurement-model support must cite validation artifacts in provenance"
                )
            expected = {
                evidence_id: self.provenance.validation_evidence_fingerprints[evidence_id]
                for evidence_id in evidence_ids
            }
            if evaluation.measurement_model_evidence_fingerprints != expected:
                raise ValueError("measurement-model evidence fingerprints must match provenance")

        supported = [
            evaluation
            for evaluation in self.evaluations
            if evaluation.status is SupportStatus.SUPPORTED
        ]
        if supported:
            if not self.readiness.valid_for_measurement_selection:
                raise ValueError("numeric measurement value is not scientifically ready")
            if (
                self.causal_status
                not in {
                    CausalStatus.IDENTIFIED_POPULATION_EFFECT,
                    CausalStatus.TRANSPORTED_UNDER_ASSUMPTIONS,
                }
                or self.causal_support.outcome is not CriterionOutcome.PASSED
            ):
                raise ValueError("intervention-oriented EVSI requires passing causal support")
            if (
                self.causal_status is CausalStatus.IDENTIFIED_POPULATION_EFFECT
                and self.transport.status is not TransportStatus.WITHIN_SUPPORT
            ):
                raise ValueError("identified EVSI must remain within validated support")
            if (
                self.transport.source_domain != self.causal_support.source_scope
                or self.transport.target_domain != self.causal_support.target_scope
            ):
                raise ValueError("measurement causal scopes and transport domains must agree")
            if self.causal_status is CausalStatus.TRANSPORTED_UNDER_ASSUMPTIONS and (
                self.transport.status is not TransportStatus.TRANSPORTED
                or set(self.transport.assumptions) != set(self.causal_support.transport_assumptions)
                or self.transport.source_domain != self.causal_support.source_scope
                or self.transport.target_domain != self.causal_support.target_scope
                or not self.transport.evidence_ids
            ):
                raise ValueError(
                    "transported EVSI requires matching domains, assumptions, and evidence"
                )
            expected_decision_set = measurement_decision_set_fingerprint(self.candidates)
            if any(
                estimand.decision_set_fingerprint != expected_decision_set
                for estimand in self.causal_support.estimands
            ):
                raise ValueError(
                    "measurement EVSI causal support must bind the exact ordered decision set"
                )
            _validate_identified_evidence_provenance(
                self.causal_support,
                self.provenance,
            )
            if not set(self.transport.evidence_ids) <= self.provenance.scientific_evidence_ids:
                raise ValueError("measurement transport evidence must be present in provenance")
        if self.status is MeasurementDecisionStatus.RECOMMENDED:
            if self.selected_assay_id is None:
                raise ValueError("a measurement recommendation requires a selected assay")
            if self.abstention_reasons:
                raise ValueError("a recommendation cannot also report abstention reasons")
            selected = next(
                (
                    evaluation
                    for evaluation in supported
                    if evaluation.assay.assay_id == self.selected_assay_id
                ),
                None,
            )
            if selected is None or selected.canonical_net_decision_value is None:
                raise ValueError("selected assay must have a supported numeric evaluation")
            best_value = max(
                evaluation.canonical_net_decision_value
                for evaluation in supported
                if evaluation.canonical_net_decision_value is not None
            )
            first_best = next(
                evaluation.assay.assay_id
                for evaluation in supported
                if evaluation.canonical_net_decision_value == best_value
            )
            if self.selected_assay_id != first_best:
                raise ValueError("selected assay must be the first maximum-net candidate")
            if selected.canonical_net_decision_value <= self.minimum_net_decision_value:
                raise ValueError("selected assay must strictly exceed the net-value threshold")
        else:
            if self.selected_assay_id is not None:
                raise ValueError("an unavailable measurement decision cannot select an assay")
            if not self.abstention_reasons:
                raise ValueError("an unavailable measurement decision requires explicit reasons")
            if self.status is MeasurementDecisionStatus.NOT_EVALUATED and any(
                evaluation.status is not SupportStatus.NOT_EVALUATED
                for evaluation in self.evaluations
            ):
                raise ValueError(
                    "a not-evaluated decision requires not-evaluated assay evaluations"
                )
            if self.status is MeasurementDecisionStatus.UNSUPPORTED and (
                supported
                or not any(
                    evaluation.status is SupportStatus.UNSUPPORTED
                    for evaluation in self.evaluations
                )
            ):
                raise ValueError(
                    "an unsupported decision requires at least one unsupported assay and no "
                    "numeric evaluations"
                )
            if self.status is MeasurementDecisionStatus.ABSTAINED and not supported:
                raise ValueError("an abstaining decision requires a supported numeric evaluation")
            if self.status is MeasurementDecisionStatus.ABSTAINED and any(
                evaluation.canonical_net_decision_value is not None
                and evaluation.canonical_net_decision_value > self.minimum_net_decision_value
                for evaluation in supported
            ):
                raise ValueError(
                    "an assay above the declared threshold cannot yield an abstaining decision"
                )
        return self


__all__ = [
    "AssayEvaluation",
    "AssayReference",
    "MeasurementDecisionRequest",
    "MeasurementDecisionStatus",
    "MeasurementEvidenceCriterion",
    "MeasurementEvidenceTrace",
    "MeasurementInformationScope",
    "MeasurementRecommendation",
    "MeasurementValueBasis",
    "measurement_decision_set_fingerprint",
]
