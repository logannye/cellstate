from __future__ import annotations

from collections.abc import Callable
from typing import Any
from uuid import uuid4

import pytest
from conftest import (
    SYNTHETIC_TEST_OPTIONS,
    assay_spec_factory,
    intervention_factory,
    intervention_spec_factory,
    query_factory,
    request_factory,
)

from cellstate.api import _measurement_candidate_regime_fingerprint, recommend_next_measurement
from cellstate.domain.belief import (
    CausalEstimandBinding,
    CausalSupportReport,
    CellStateBelief,
    EvaluationStatus,
    QueryReadinessReport,
    SupportReport,
)
from cellstate.domain.common import (
    CausalStatus,
    CriterionOutcome,
    ProvenanceRecord,
    SupportStatus,
    canonical_fingerprint,
)
from cellstate.domain.events import (
    AssignmentMechanism,
    CollectionEffect,
    ObservationCollection,
    ReversibilityStatus,
)
from cellstate.domain.measurements import (
    AssayEvaluation,
    AssayReference,
    MeasurementDecisionRequest,
    MeasurementDecisionStatus,
    MeasurementEvidenceCriterion,
    MeasurementEvidenceTrace,
    MeasurementInformationScope,
    MeasurementRecommendation,
    MeasurementValueBasis,
    measurement_decision_set_fingerprint,
)
from cellstate.domain.query import (
    AssayPurpose,
    FutureAssayObservationEndpoint,
    StateQuery,
)
from cellstate.domain.request import InferenceOptions
from cellstate.domain.scenarios import (
    EvolutionScenario,
    InterventionObjective,
    ObjectiveDirection,
    ObjectiveTerm,
    ScenarioReference,
    TransportReport,
    TransportStatus,
)
from cellstate.errors import CapabilityError, ContractViolationError
from cellstate.ports import (
    EstimatorDescriptor,
    MeasurementCapabilityReport,
    ModelArtifactKind,
    measurement_capability_scope_fingerprint,
)
from cellstate.reference import LinearGaussianMeasurementPolicy


def _objective(belief: CellStateBelief) -> InterventionObjective:
    return InterventionObjective(
        objective_id="maximize-function",
        horizon_name="acute",
        terms=(
            ObjectiveTerm(
                target=belief.query.target_outputs[0].term,
                direction=ObjectiveDirection.MAXIMIZE,
            ),
        ),
    )


def _baseline(belief: CellStateBelief, scenario_id: str = "baseline") -> EvolutionScenario:
    return EvolutionScenario(
        scenario_id=scenario_id,
        horizon_name="acute",
        subject=belief.subject,
        start_time_seconds=belief.as_of_seconds,
        end_time_seconds=belief.as_of_seconds + 60,
    )


def _penalties() -> dict[CollectionEffect, float]:
    return {
        CollectionEffect.NONDESTRUCTIVE: 0,
        CollectionEffect.VIABILITY_PRESERVING_WITH_KNOWN_EFFECT: 1,
        CollectionEffect.PARTIALLY_DESTRUCTIVE_POPULATION_SAMPLING: 10,
        CollectionEffect.TERMINAL_DESTRUCTIVE: 100,
    }


def _request(
    belief: CellStateBelief,
    *,
    candidates: tuple[EvolutionScenario, ...] | None = None,
    assay_ids: tuple[str, ...] = ("signal-panel",),
    collection_time_seconds: float | None = None,
    decision_deadline_seconds: float | None = None,
) -> MeasurementDecisionRequest:
    resolved_deadline = (
        belief.as_of_seconds + 5 if decision_deadline_seconds is None else decision_deadline_seconds
    )
    if candidates is None:
        treated = EvolutionScenario(
            scenario_id="treated",
            horizon_name="acute",
            subject=belief.subject,
            start_time_seconds=belief.as_of_seconds,
            end_time_seconds=belief.as_of_seconds + 60,
            interventions=(
                intervention_factory(
                    event_id="future-treatment",
                    subject=belief.subject,
                    time_seconds=belief.as_of_seconds + 5,
                    duration_seconds=10,
                    estimated_efficiency=None,
                ),
            ),
        )
        candidates = (_baseline(belief), treated)
    return MeasurementDecisionRequest(
        request_id="measurement-decision",
        parent_belief_id=belief.belief_id,
        query_fingerprint=belief.query_fingerprint,
        objective=_objective(belief),
        candidates=candidates,
        candidate_assay_ids=assay_ids,
        collection_time_seconds=(
            belief.as_of_seconds if collection_time_seconds is None else collection_time_seconds
        ),
        decision_deadline_seconds=resolved_deadline,
        minimum_net_decision_value=0,
        utility_units="synthetic_utility",
        assay_cost_to_utility_rate=0.5,
        delay_penalty_per_second=0.1,
        destructiveness_penalties=_penalties(),
    )


class _MutatingPolicy:
    def __init__(
        self,
        base: LinearGaussianMeasurementPolicy,
        *,
        mutate_capability: Callable[[MeasurementCapabilityReport], Any] | None = None,
        mutate_recommendation: Callable[[MeasurementRecommendation], Any] | None = None,
    ) -> None:
        self.base = base
        self._mutate_capability = mutate_capability
        self._mutate_recommendation = mutate_recommendation

    @property
    def descriptor(self) -> EstimatorDescriptor:
        return self.base.descriptor

    def capabilities(
        self,
        belief: CellStateBelief,
        request: MeasurementDecisionRequest,
    ) -> Any:
        report = self.base.capabilities(belief, request)
        return self._mutate_capability(report) if self._mutate_capability else report

    def recommend(
        self,
        belief: CellStateBelief,
        request: MeasurementDecisionRequest,
        *,
        options: InferenceOptions,
    ) -> Any:
        result = self.base.recommend(belief, request, options=options)
        return self._mutate_recommendation(result) if self._mutate_recommendation else result


def test_reference_policy_crosses_public_boundary_as_honest_abstention(model) -> None:
    belief = model.estimate(request_factory(), options=SYNTHETIC_TEST_OPTIONS)
    request = _request(belief)

    result = recommend_next_measurement(
        belief,
        request=request,
        policy=LinearGaussianMeasurementPolicy(model),
        options=InferenceOptions(seed=17),
    )

    assert result.status is MeasurementDecisionStatus.NOT_EVALUATED
    assert result.seed == result.provenance.seed == 17
    assert not belief.readiness.valid_for_control
    assert all(
        evaluation.status is SupportStatus.NOT_EVALUATED
        and evaluation.net_expected_decision_value is None
        for evaluation in result.evaluations
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (
            lambda belief, request: request.model_copy(update={"parent_belief_id": uuid4()}),
            "different belief",
        ),
        (
            lambda belief, request: request.model_copy(update={"query_fingerprint": "0" * 64}),
            "different query",
        ),
        (
            lambda belief, request: request.model_copy(
                update={"collection_time_seconds": belief.as_of_seconds - 1}
            ),
            "cannot predate",
        ),
        (
            lambda belief, request: request.model_copy(
                update={"decision_deadline_seconds": belief.as_of_seconds + 61}
            ),
            "beyond a candidate horizon",
        ),
        (
            lambda belief, request: request.model_copy(
                update={"candidate_assay_ids": ("unknown",)}
            ),
            "outside the query",
        ),
    ),
)
def test_measurement_request_requires_exact_belief_time_and_assay_binding(
    model,
    mutation: Callable[[CellStateBelief, MeasurementDecisionRequest], MeasurementDecisionRequest],
    message: str,
) -> None:
    belief = model.estimate(request_factory(), options=SYNTHETIC_TEST_OPTIONS)
    request = mutation(belief, _request(belief))

    with pytest.raises(ContractViolationError, match=message):
        recommend_next_measurement(
            belief,
            request=request,
            policy=LinearGaussianMeasurementPolicy(model),
        )


def test_measurement_request_rejects_target_endpoint_only_assay(model) -> None:
    belief = model.estimate(request_factory(), options=SYNTHETIC_TEST_OPTIONS)
    original_assay = belief.query.available_assays[0]
    target_only_assay = original_assay.model_copy(
        update={
            "purposes": (AssayPurpose.TARGET_ENDPOINT,),
            "cost": None,
            "cost_units": None,
            "turnaround_seconds": None,
        }
    )
    target_output = belief.query.target_outputs[0].model_copy(
        update={
            "endpoint": FutureAssayObservationEndpoint(
                assay_id=target_only_assay.assay_id,
                protocol_reference=target_only_assay.protocol_reference,
            )
        }
    )
    no_measurement_budget = belief.query.constraints.model_copy(
        update={
            "maximum_total_assay_cost": None,
            "assay_cost_units": None,
            "maximum_assay_delay_seconds": None,
        }
    )
    query = StateQuery.model_validate(
        {
            **belief.query.model_dump(mode="python"),
            "target_outputs": (target_output,),
            "available_assays": (target_only_assay,),
            "constraints": no_measurement_budget,
        }
    )
    belief = belief.model_copy(update={"query": query, "query_fingerprint": query.fingerprint})

    with pytest.raises(ContractViolationError, match="do not declare measurement-selection"):
        recommend_next_measurement(
            belief,
            request=_request(belief),
            policy=LinearGaussianMeasurementPolicy(model),
        )


def test_candidate_changes_wait_until_measurement_decision_deadline(model) -> None:
    belief = model.estimate(request_factory(), options=SYNTHETIC_TEST_OPTIONS)
    action = intervention_factory(
        event_id="premature-drug",
        subject=belief.subject,
        time_seconds=belief.as_of_seconds + 1,
        duration_seconds=10,
        estimated_efficiency=None,
    )
    candidate = EvolutionScenario(
        scenario_id="premature-action",
        horizon_name="acute",
        subject=belief.subject,
        start_time_seconds=belief.as_of_seconds,
        end_time_seconds=belief.as_of_seconds + 60,
        interventions=(action,),
    )
    baseline = _baseline(belief)

    with pytest.raises(ContractViolationError, match="cannot begin before"):
        recommend_next_measurement(
            belief,
            request=_request(belief, candidates=(baseline, candidate)),
            policy=LinearGaussianMeasurementPolicy(model),
        )


def test_candidate_dependent_inheritance_cannot_change_state_before_decision(model) -> None:
    belief = model.estimate(request_factory(), options=SYNTHETIC_TEST_OPTIONS)
    active = intervention_factory(
        event_id="active-drug",
        subject=belief.subject,
        time_seconds=belief.as_of_seconds - 1,
        duration_seconds=20,
        estimated_efficiency=None,
    )
    belief = belief.model_copy(
        update={"context": belief.context.model_copy(update={"active_interventions": (active,)})}
    )
    inherit = _baseline(belief, "inherit").model_copy(update={"inherit_active_interventions": True})
    clear = _baseline(belief, "clear").model_copy(update={"inherit_active_interventions": False})

    with pytest.raises(ContractViolationError, match="inheritance would change treatment"):
        recommend_next_measurement(
            belief,
            request=_request(belief, candidates=(inherit, clear)),
            policy=LinearGaussianMeasurementPolicy(model),
        )


@pytest.mark.parametrize(
    ("effect", "message"),
    (
        (CollectionEffect.TERMINAL_DESTRUCTIVE, "terminal destructive"),
        (
            CollectionEffect.PARTIALLY_DESTRUCTIVE_POPULATION_SAMPLING,
            "partial population sampling",
        ),
    ),
)
def test_same_cell_destructive_collection_is_impossible(model, effect, message: str) -> None:
    belief = model.estimate(request_factory(), options=SYNTHETIC_TEST_OPTIONS)
    assay = assay_spec_factory(
        collection=ObservationCollection(
            effect=effect,
            sampling_fraction=(
                0.1
                if effect is CollectionEffect.PARTIALLY_DESTRUCTIVE_POPULATION_SAMPLING
                else None
            ),
        ),
    )
    query = belief.query.model_copy(update={"available_assays": (assay,)})
    belief = belief.model_copy(update={"query": query, "query_fingerprint": query.fingerprint})

    with pytest.raises(ContractViolationError, match=message):
        recommend_next_measurement(
            belief,
            request=_request(belief),
            policy=LinearGaussianMeasurementPolicy(model),
        )


def test_assay_result_must_arrive_before_deadline(model) -> None:
    belief = model.estimate(request_factory(), options=SYNTHETIC_TEST_OPTIONS)
    assay = assay_spec_factory(turnaround_seconds=10)
    query = belief.query.model_copy(update={"available_assays": (assay,)})
    belief = belief.model_copy(update={"query": query, "query_fingerprint": query.fingerprint})

    with pytest.raises(ContractViolationError, match="cannot return before"):
        recommend_next_measurement(
            belief,
            request=_request(belief),
            policy=LinearGaussianMeasurementPolicy(model),
        )


def test_one_cost_conversion_rate_requires_one_assay_currency(model) -> None:
    belief = model.estimate(request_factory(), options=SYNTHETIC_TEST_OPTIONS)
    first = assay_spec_factory(assay_id="first")
    second = assay_spec_factory(assay_id="second").model_copy(
        update={"cost_units": "different_currency"}
    )
    query = belief.query.model_copy(update={"available_assays": (first, second)})
    belief = belief.model_copy(update={"query": query, "query_fingerprint": query.fingerprint})

    with pytest.raises(ContractViolationError, match="different cost units"):
        recommend_next_measurement(
            belief,
            request=_request(belief, assay_ids=("first", "second")),
            policy=LinearGaussianMeasurementPolicy(model),
        )


def test_measurement_capability_report_is_revalidated_and_exact(model) -> None:
    belief = model.estimate(request_factory(), options=SYNTHETIC_TEST_OPTIONS)
    request = _request(belief)
    base = LinearGaussianMeasurementPolicy(model)

    wrong_scope = _MutatingPolicy(
        base,
        mutate_capability=lambda report: report.model_copy(update={"scope_fingerprint": "0" * 64}),
    )
    with pytest.raises(ContractViolationError, match="exact requested scope"):
        recommend_next_measurement(belief, request=request, policy=wrong_scope)

    incomplete = _MutatingPolicy(
        base,
        mutate_capability=lambda report: report.model_copy(update={"assay_support": {}}),
    )
    with pytest.raises(ContractViolationError, match="invalid capability report"):
        recommend_next_measurement(belief, request=request, policy=incomplete)

    unsupported = _MutatingPolicy(
        base,
        mutate_capability=lambda report: report.model_copy(
            update={"supported": False, "blockers": ("no-outcome-model",)}
        ),
    )
    with pytest.raises(CapabilityError, match="no-outcome-model"):
        recommend_next_measurement(belief, request=request, policy=unsupported)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (
            lambda result: result.model_copy(update={"parent_belief_id": uuid4()}),
            "not exactly bound",
        ),
        (
            lambda result: result.model_copy(update={"minimum_net_decision_value": 1}),
            "not exactly bound",
        ),
        (
            lambda result: result.model_copy(
                update={
                    "provenance": result.provenance.model_copy(
                        update={"history_fingerprint": "0" * 64}
                    )
                }
            ),
            "provenance does not match",
        ),
        (
            lambda result: result.model_copy(update={"seed": result.seed + 1}),
            "wrong inference seed",
        ),
        (
            lambda result: result.model_copy(
                update={
                    "evaluations": (
                        result.evaluations[0].model_copy(
                            update={
                                "collection_effect": (
                                    CollectionEffect.VIABILITY_PRESERVING_WITH_KNOWN_EFFECT
                                )
                            }
                        ),
                    )
                }
            ),
            "changed its collection effect",
        ),
    ),
)
def test_measurement_result_binding_provenance_seed_and_collection_are_exact(
    model,
    mutation: Callable[[MeasurementRecommendation], MeasurementRecommendation],
    message: str,
) -> None:
    belief = model.estimate(request_factory(), options=SYNTHETIC_TEST_OPTIONS)
    request = _request(belief)
    policy = _MutatingPolicy(
        LinearGaussianMeasurementPolicy(model),
        mutate_recommendation=mutation,
    )

    with pytest.raises(ContractViolationError, match=message):
        recommend_next_measurement(belief, request=request, policy=policy)


_MODEL_FINGERPRINT = "9" * 64
_ASSAY_EVIDENCE = "assay-validation"
_ASSAY_EVIDENCE_FINGERPRINT = "a" * 64
_CAUSAL_EVIDENCE = "causal-validation"
_CAUSAL_EVIDENCE_FINGERPRINT = "b" * 64


def _synthetic_supported_descriptor() -> EstimatorDescriptor:
    return EstimatorDescriptor(
        model_id="validated-measurement-policy",
        model_version="1",
        model_fingerprint=_MODEL_FINGERPRINT,
        posterior_schema_id="cellstate/validated-measurement-v1",
        description="Synthetic validated contract fixture.",
        artifact_kind=ModelArtifactKind.SYNTHETIC_TEST_MODEL,
        support_envelope_id="support-envelope",
        support_envelope_fingerprint="1" * 64,
        training_support_id="training-support",
        training_support_fingerprint="2" * 64,
        validation_evidence_ids=(_ASSAY_EVIDENCE, _CAUSAL_EVIDENCE),
        validation_evidence_fingerprints={
            _ASSAY_EVIDENCE: _ASSAY_EVIDENCE_FINGERPRINT,
            _CAUSAL_EVIDENCE: _CAUSAL_EVIDENCE_FINGERPRINT,
        },
    )


def _causal_support(
    belief: CellStateBelief,
    candidates: tuple[EvolutionScenario, ...] | None = None,
) -> CausalSupportReport:
    target = belief.query.target_outputs[0]
    return CausalSupportReport(
        evaluation_status=EvaluationStatus.EVALUATED,
        outcome=CriterionOutcome.PASSED,
        causal_status=CausalStatus.IDENTIFIED_POPULATION_EFFECT,
        identification_basis="randomized validation study",
        identification_design=AssignmentMechanism.RANDOMIZED,
        estimands=(
            CausalEstimandBinding(
                target=target.term,
                horizon_name="acute",
                aggregation=target.aggregation,
                intervention_spec_ids=("drug",),
                comparator="randomized control",
                decision_set_fingerprint=(
                    measurement_decision_set_fingerprint(candidates)
                    if candidates is not None
                    else None
                ),
            ),
        ),
        evidence_ids=(_CAUSAL_EVIDENCE,),
        evidence_fingerprints={_CAUSAL_EVIDENCE: _CAUSAL_EVIDENCE_FINGERPRINT},
        source_scope="validation population",
        target_scope="query population",
    )


def _measurement_ready_belief(model) -> CellStateBelief:
    randomized_spec = intervention_spec_factory(
        allowed_assignment_mechanisms=(AssignmentMechanism.RANDOMIZED,),
        randomization_unit_kind="well",
        require_randomization_unit=True,
    )
    query = query_factory().model_copy(update={"intervention_space": (randomized_spec,)})
    base = model.estimate(
        request_factory(query=query),
        options=SYNTHETIC_TEST_OPTIONS,
    )
    causal_support = _causal_support(base)
    readiness_payload = base.readiness.model_dump(mode="python")
    readiness_payload.update(
        {
            "support": CriterionOutcome.PASSED,
            "causal": CriterionOutcome.PASSED,
            "measurement_model": CriterionOutcome.PASSED,
            "valid_for_measurement_selection": True,
        }
    )
    readiness = QueryReadinessReport.model_validate(readiness_payload)
    support = SupportReport(
        evaluation_status=EvaluationStatus.EVALUATED,
        outcome=CriterionOutcome.PASSED,
        in_distribution_score=1,
        ood_score=0,
        maximum_ood_score=base.query.acceptance_thresholds.maximum_ood_score,
        abstention_required=False,
    )
    descriptor = _synthetic_supported_descriptor()
    provenance_payload = base.provenance.model_dump(mode="python")
    provenance_payload.update(
        {
            "model_id": descriptor.model_id,
            "model_version": descriptor.model_version,
            "model_fingerprint": descriptor.model_fingerprint,
            "posterior_schema_id": descriptor.posterior_schema_id,
            "support_envelope_id": descriptor.support_envelope_id,
            "support_envelope_fingerprint": descriptor.support_envelope_fingerprint,
            "training_support_id": descriptor.training_support_id,
            "training_support_fingerprint": descriptor.training_support_fingerprint,
            "validation_evidence_ids": descriptor.validation_evidence_ids,
            "validation_evidence_fingerprints": descriptor.validation_evidence_fingerprints,
        }
    )
    provenance = ProvenanceRecord.model_validate(provenance_payload)
    payload = base.model_dump(mode="python")
    payload.update(
        {
            "diagnostics": base.diagnostics.model_copy(
                update={"support": support, "causal_support": causal_support}
            ),
            "readiness": readiness,
            "provenance": provenance,
        }
    )
    return CellStateBelief.model_validate(payload)


class _SupportedPolicy:
    descriptor = _synthetic_supported_descriptor()

    def __init__(
        self,
        *,
        capability_assay_status: SupportStatus = SupportStatus.SUPPORTED,
        information_scope: MeasurementInformationScope = (
            MeasurementInformationScope.INTERVENTION_OUTCOMES
        ),
        wrong_economics: bool = False,
        wrong_contrast: bool = False,
        transported: bool = False,
        trace_scope_fingerprint: str | None = None,
    ) -> None:
        self.capability_assay_status = capability_assay_status
        self.information_scope = information_scope
        self.wrong_economics = wrong_economics
        self.wrong_contrast = wrong_contrast
        self.transported = transported
        self.trace_scope_fingerprint = trace_scope_fingerprint

    def capabilities(
        self,
        belief: CellStateBelief,
        request: MeasurementDecisionRequest,
    ) -> MeasurementCapabilityReport:
        return MeasurementCapabilityReport(
            supported=True,
            scope_fingerprint=measurement_capability_scope_fingerprint(belief, request),
            assay_support={
                assay_id: self.capability_assay_status for assay_id in request.candidate_assay_ids
            },
            collection_effect_support={
                effect: (
                    SupportStatus.SUPPORTED
                    if effect is CollectionEffect.NONDESTRUCTIVE
                    else SupportStatus.UNSUPPORTED
                )
                for effect in CollectionEffect
            },
            assay_outcome_model=CriterionOutcome.PASSED,
            hypothetical_update=CriterionOutcome.PASSED,
            counterfactual_replanning=CriterionOutcome.PASSED,
            decision_utility=CriterionOutcome.PASSED,
        )

    def recommend(
        self,
        belief: CellStateBelief,
        request: MeasurementDecisionRequest,
        *,
        options: InferenceOptions,
    ) -> MeasurementRecommendation:
        assay = next(
            assay
            for assay in belief.query.available_assays
            if assay.assay_id == request.candidate_assay_ids[0]
        )
        assay_reference = AssayReference(
            assay_id=assay.assay_id,
            fingerprint=canonical_fingerprint(assay),
        )
        raw_cost = assay.cost + (1 if self.wrong_economics else 0)
        cost_penalty = assay.cost * request.assay_cost_to_utility_rate
        expected_delay = (
            request.collection_time_seconds - belief.as_of_seconds + assay.turnaround_seconds
        )
        delay_cost = expected_delay * request.delay_penalty_per_second
        destructiveness_cost = request.destructiveness_penalties[assay.collection.effect]
        gross = 5.0
        net = gross - cost_penalty - delay_cost - destructiveness_cost
        scope_fingerprint = self.trace_scope_fingerprint or (
            measurement_capability_scope_fingerprint(belief, request)
        )
        evaluation = AssayEvaluation(
            assay=assay_reference,
            status=SupportStatus.SUPPORTED,
            collection_effect=assay.collection.effect,
            value_basis=MeasurementValueBasis.INTERVENTION_DECISION_EVSI,
            information_scope=self.information_scope,
            baseline_best_expected_utility=1,
            expected_post_measurement_best_utility=6,
            gross_expected_decision_value=gross,
            expected_information_gain=0.4,
            information_gain_metric="expected_target_log_score_gain",
            expected_posterior_uncertainty_reduction=0.3,
            uncertainty_reduction_metric="decision_regret_reduction",
            expected_intervention_ranking_change=0.2,
            ranking_change_metric="top_choice_change_probability",
            raw_assay_cost=raw_cost,
            assay_cost_units=assay.cost_units,
            assay_cost_penalty=cost_penalty,
            expected_delay_seconds=expected_delay,
            delay_cost=delay_cost,
            destructiveness_cost=destructiveness_cost,
            net_expected_decision_value=net,
            evidence_traces=tuple(
                MeasurementEvidenceTrace(
                    criterion=criterion,
                    outcome=CriterionOutcome.PASSED,
                    scope_fingerprint=scope_fingerprint,
                    evidence_ids=(_ASSAY_EVIDENCE,),
                    evidence_fingerprints={_ASSAY_EVIDENCE: _ASSAY_EVIDENCE_FINGERPRINT},
                )
                for criterion in MeasurementEvidenceCriterion
            ),
            measurement_model_evidence_ids=(_ASSAY_EVIDENCE,),
            measurement_model_evidence_fingerprints={_ASSAY_EVIDENCE: _ASSAY_EVIDENCE_FINGERPRINT},
        )
        causal_support = _causal_support(belief, request.candidates)
        if self.wrong_contrast:
            causal_support = causal_support.model_copy(
                update={
                    "estimands": (
                        causal_support.estimands[0].model_copy(
                            update={"intervention_spec_ids": ("other",)}
                        ),
                    )
                }
            )
        transport = TransportReport(
            status=TransportStatus.WITHIN_SUPPORT,
            source_domain=causal_support.source_scope,
            target_domain=causal_support.target_scope,
        )
        if self.transported:
            assumptions = ("stable response mechanism",)
            causal_support = causal_support.model_copy(
                update={
                    "causal_status": CausalStatus.TRANSPORTED_UNDER_ASSUMPTIONS,
                    "source_scope": "validated source population",
                    "target_scope": "query target population",
                    "transport_assumptions": assumptions,
                }
            )
            transport = TransportReport(
                status=TransportStatus.TRANSPORTED,
                source_domain="validated source population",
                target_domain="query target population",
                assumptions=assumptions,
                evidence_ids=(_CAUSAL_EVIDENCE,),
            )
        provenance = belief.provenance.model_copy(update={"seed": options.seed})
        return MeasurementRecommendation(
            recommendation_id="supported-measurement",
            status=MeasurementDecisionStatus.RECOMMENDED,
            parent_belief_id=belief.belief_id,
            query_fingerprint=belief.query_fingerprint,
            request_id=request.request_id,
            request_fingerprint=request.fingerprint,
            objective_id=request.objective.objective_id,
            objective_fingerprint=canonical_fingerprint(request.objective),
            candidates=tuple(
                ScenarioReference(
                    scenario_id=candidate.scenario_id,
                    fingerprint=canonical_fingerprint(candidate),
                )
                for candidate in request.candidates
            ),
            assays=(assay_reference,),
            selected_assay_id=assay.assay_id,
            evaluations=(evaluation,),
            readiness=belief.readiness,
            causal_status=causal_support.causal_status,
            causal_support=causal_support,
            transport=transport,
            minimum_net_decision_value=request.minimum_net_decision_value,
            utility_units=request.utility_units,
            rationale="Positive validated EVSI.",
            seed=options.seed,
            provenance=provenance,
        )


class _UnavailableAssayPolicy(_SupportedPolicy):
    def __init__(
        self,
        support_status: SupportStatus,
        *,
        bind_reason: bool = True,
    ) -> None:
        super().__init__(capability_assay_status=support_status)
        self.support_status = support_status
        self.bind_reason = bind_reason

    def recommend(
        self,
        belief: CellStateBelief,
        request: MeasurementDecisionRequest,
        *,
        options: InferenceOptions,
    ) -> MeasurementRecommendation:
        numeric = super().recommend(belief, request, options=options)
        assay_id = numeric.assays[0].assay_id
        reason = (
            f"assay_support:{assay_id}:{self.support_status.value}"
            if self.bind_reason
            else "assay support unavailable"
        )
        unavailable = AssayEvaluation(
            assay=numeric.assays[0],
            status=self.support_status,
            collection_effect=numeric.evaluations[0].collection_effect,
            evidence_traces=numeric.evaluations[0].evidence_traces,
            reasons=(reason,),
        )
        decision_status = (
            MeasurementDecisionStatus.UNSUPPORTED
            if self.support_status is SupportStatus.UNSUPPORTED
            else MeasurementDecisionStatus.NOT_EVALUATED
        )
        return numeric.model_copy(
            update={
                "status": decision_status,
                "selected_assay_id": None,
                "evaluations": (unavailable,),
                "abstention_reasons": ("assay support blocks EVSI calculation",),
                "rationale": "Validated EVSI components exist, but assay support is unavailable.",
            }
        )


def _action_candidates(belief: CellStateBelief) -> tuple[EvolutionScenario, ...]:
    deadline = belief.as_of_seconds + 5
    intervention = intervention_factory(
        event_id="randomized-drug",
        subject=belief.subject,
        time_seconds=deadline,
        duration_seconds=10,
        estimated_efficiency=None,
        assignment_mechanism=AssignmentMechanism.RANDOMIZED,
        randomization_unit_kind="well",
        randomization_unit_id="randomized-well-1",
    )
    treated = EvolutionScenario(
        scenario_id="treated",
        horizon_name="acute",
        subject=belief.subject,
        start_time_seconds=belief.as_of_seconds,
        end_time_seconds=belief.as_of_seconds + 60,
        interventions=(intervention,),
    )
    return (_baseline(belief), treated)


def test_supported_measurement_selection_does_not_require_control_readiness(model) -> None:
    belief = _measurement_ready_belief(model)
    request = _request(belief, candidates=_action_candidates(belief))

    result = recommend_next_measurement(
        belief,
        request=request,
        policy=_SupportedPolicy(),
    )

    assert result.status is MeasurementDecisionStatus.RECOMMENDED
    assert result.selected_assay_id == "signal-panel"
    assert belief.readiness.valid_for_measurement_selection
    assert not belief.readiness.valid_for_control


def test_evsi_traces_match_exact_scope_and_capability_outcomes(model) -> None:
    belief = _measurement_ready_belief(model)
    request = _request(belief, candidates=_action_candidates(belief))

    with pytest.raises(ContractViolationError, match="exact request scope"):
        recommend_next_measurement(
            belief,
            request=request,
            policy=_SupportedPolicy(trace_scope_fingerprint="0" * 64),
        )

    mismatched_capability = _MutatingPolicy(
        _SupportedPolicy(),
        mutate_capability=lambda report: report.model_copy(
            update={"hypothetical_update": CriterionOutcome.FAILED}
        ),
    )
    with pytest.raises(ContractViolationError, match="disagrees with capability outcomes"):
        recommend_next_measurement(
            belief,
            request=request,
            policy=mismatched_capability,
        )


@pytest.mark.parametrize(
    ("support_status", "decision_status"),
    (
        (SupportStatus.NOT_EVALUATED, MeasurementDecisionStatus.NOT_EVALUATED),
        (SupportStatus.UNSUPPORTED, MeasurementDecisionStatus.UNSUPPORTED),
    ),
)
def test_passed_evsi_components_can_retain_explicit_assay_support_blockers(
    model,
    support_status: SupportStatus,
    decision_status: MeasurementDecisionStatus,
) -> None:
    belief = _measurement_ready_belief(model)
    request = _request(belief, candidates=_action_candidates(belief))

    result = recommend_next_measurement(
        belief,
        request=request,
        policy=_UnavailableAssayPolicy(support_status),
    )

    assert result.status is decision_status
    assert result.evaluations[0].status is support_status
    assert all(
        trace.outcome is CriterionOutcome.PASSED for trace in result.evaluations[0].evidence_traces
    )
    assert result.evaluations[0].net_expected_decision_value is None


def test_unavailable_assay_evaluation_must_name_its_capability_blocker(model) -> None:
    belief = _measurement_ready_belief(model)
    request = _request(belief, candidates=_action_candidates(belief))

    with pytest.raises(ContractViolationError, match="reasons do not bind"):
        recommend_next_measurement(
            belief,
            request=request,
            policy=_UnavailableAssayPolicy(
                SupportStatus.UNSUPPORTED,
                bind_reason=False,
            ),
        )


@pytest.mark.parametrize(
    ("policy", "message"),
    (
        (
            _SupportedPolicy(capability_assay_status=SupportStatus.NOT_EVALUATED),
            "status disagrees",
        ),
        (
            _SupportedPolicy(information_scope=MeasurementInformationScope.QUERY_TARGETS),
            "cannot count as decision EVSI",
        ),
        (_SupportedPolicy(wrong_economics=True), "changed its cost"),
        (_SupportedPolicy(wrong_contrast=True), "exact decision estimand"),
    ),
)
def test_numeric_evsi_cannot_exceed_scientific_or_economic_support(
    model,
    policy: _SupportedPolicy,
    message: str,
) -> None:
    belief = _measurement_ready_belief(model)
    request = _request(belief, candidates=_action_candidates(belief))

    with pytest.raises(ContractViolationError, match=message):
        recommend_next_measurement(
            belief,
            request=request,
            policy=policy,
        )


def test_numeric_evsi_must_match_the_actual_candidate_contrast(model) -> None:
    belief = _measurement_ready_belief(model)
    request = _request(
        belief,
        candidates=(_baseline(belief), _baseline(belief, "duplicate-baseline")),
    )

    with pytest.raises(ContractViolationError, match="semantically distinct"):
        recommend_next_measurement(
            belief,
            request=request,
            policy=_SupportedPolicy(),
        )


def test_case_and_assignment_ids_cannot_fake_a_distinct_candidate_regime(model) -> None:
    belief = _measurement_ready_belief(model)
    treated = _action_candidates(belief)[1]
    intervention = treated.interventions[0]
    duplicate = treated.model_copy(
        update={
            "scenario_id": "treated-case-variant",
            "interventions": (
                intervention.model_copy(
                    update={
                        "event_id": "same-drug-case-variant",
                        "delivery_method": intervention.delivery_method.upper(),
                        "randomization_unit_id": "randomized-well-2",
                    }
                ),
            ),
        }
    )
    request = _request(belief, candidates=(treated, duplicate))

    with pytest.raises(ContractViolationError, match="semantically distinct"):
        recommend_next_measurement(
            belief,
            request=request,
            policy=_SupportedPolicy(),
        )


def test_reversibility_status_is_bound_into_candidate_regime_fingerprint(model) -> None:
    belief = _measurement_ready_belief(model)
    treated = _action_candidates(belief)[1]
    intervention = treated.interventions[0]
    unknown_reversibility = treated.model_copy(
        update={
            "scenario_id": "unknown-reversibility",
            "interventions": (
                intervention.model_copy(
                    update={
                        "event_id": "unknown-reversibility-event",
                        "reversibility_status": ReversibilityStatus.UNKNOWN,
                    }
                ),
            ),
        }
    )

    assert _measurement_candidate_regime_fingerprint(
        belief, treated
    ) != _measurement_candidate_regime_fingerprint(belief, unknown_reversibility)


def test_transported_evsi_respects_the_query_transport_constraint(model) -> None:
    belief = _measurement_ready_belief(model)
    request = _request(belief, candidates=_action_candidates(belief))

    with pytest.raises(ContractViolationError, match="do not allow transported"):
        recommend_next_measurement(
            belief,
            request=request,
            policy=_SupportedPolicy(transported=True),
        )
