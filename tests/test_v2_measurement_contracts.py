from __future__ import annotations

from uuid import uuid4

import pytest
from conftest import (
    assay_spec_factory,
    intervention_factory,
    query_factory,
    subject_factory,
)
from pydantic import ValidationError

from cellstate.domain.belief import (
    CausalEstimandBinding,
    CausalSupportReport,
    EvaluationStatus,
    QueryReadinessReport,
)
from cellstate.domain.common import (
    CausalStatus,
    CriterionOutcome,
    ProvenanceRecord,
    SupportStatus,
    canonical_fingerprint,
)
from cellstate.domain.events import AssignmentMechanism, CollectionEffect
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
from cellstate.domain.scenarios import (
    EvolutionScenario,
    InterventionObjective,
    ObjectiveDirection,
    ObjectiveTerm,
    ScenarioReference,
    TransportReport,
    TransportStatus,
)


def _scenario(scenario_id: str = "drug-arm") -> EvolutionScenario:
    subject = subject_factory()
    return EvolutionScenario(
        scenario_id=scenario_id,
        horizon_name="acute",
        subject=subject,
        start_time_seconds=10,
        end_time_seconds=70,
        interventions=(
            intervention_factory(
                event_id=f"{scenario_id}-drug",
                subject=subject,
                time_seconds=20,
                duration_seconds=20,
                estimated_efficiency=None,
            ),
        ),
    )


def _objective() -> InterventionObjective:
    return InterventionObjective(
        objective_id="maximize-capacity",
        horizon_name="acute",
        terms=(
            ObjectiveTerm(
                target=query_factory().target_outputs[0].term,
                direction=ObjectiveDirection.MAXIMIZE,
            ),
        ),
    )


def _penalties() -> dict[CollectionEffect, float]:
    return {
        CollectionEffect.NONDESTRUCTIVE: 0,
        CollectionEffect.VIABILITY_PRESERVING_WITH_KNOWN_EFFECT: 0.25,
        CollectionEffect.PARTIALLY_DESTRUCTIVE_POPULATION_SAMPLING: 0.5,
        CollectionEffect.TERMINAL_DESTRUCTIVE: 100,
    }


def _request() -> MeasurementDecisionRequest:
    treated = _scenario()
    baseline = EvolutionScenario(
        scenario_id="baseline",
        horizon_name="acute",
        subject=treated.subject,
        start_time_seconds=10,
        end_time_seconds=70,
    )
    return MeasurementDecisionRequest(
        request_id="measure-1",
        parent_belief_id=uuid4(),
        query_fingerprint=query_factory().fingerprint,
        objective=_objective(),
        candidates=(baseline, treated),
        candidate_assay_ids=("signal-panel",),
        collection_time_seconds=10,
        decision_deadline_seconds=15,
        minimum_net_decision_value=0,
        utility_units="synthetic_utility",
        assay_cost_to_utility_rate=0.5,
        delay_penalty_per_second=0.01,
        destructiveness_penalties=_penalties(),
    )


def _assay_reference(assay_id: str = "signal-panel") -> AssayReference:
    assay = assay_spec_factory(assay_id=assay_id)
    return AssayReference(assay_id=assay_id, fingerprint=canonical_fingerprint(assay))


def _evidence_traces(
    *,
    outcome: CriterionOutcome = CriterionOutcome.PASSED,
    scope_fingerprint: str = "3" * 64,
) -> tuple[MeasurementEvidenceTrace, ...]:
    return tuple(
        MeasurementEvidenceTrace(
            criterion=criterion,
            outcome=outcome,
            scope_fingerprint=scope_fingerprint,
            evidence_ids=("assay-validation",) if outcome is CriterionOutcome.PASSED else (),
            evidence_fingerprints=(
                {"assay-validation": "a" * 64} if outcome is CriterionOutcome.PASSED else {}
            ),
            reasons=("criterion unavailable",)
            if outcome in {CriterionOutcome.NOT_EVALUATED, CriterionOutcome.UNSUPPORTED}
            else (),
        )
        for criterion in MeasurementEvidenceCriterion
    )


def _supported_evaluation(
    assay_id: str = "signal-panel",
    *,
    net_value: float = 1.25,
) -> AssayEvaluation:
    assay_cost = 1.0
    delay_cost = 0.25
    destruction_cost = 0.5
    gross = net_value + assay_cost + delay_cost + destruction_cost
    return AssayEvaluation(
        assay=_assay_reference(assay_id),
        status=SupportStatus.SUPPORTED,
        collection_effect=CollectionEffect.NONDESTRUCTIVE,
        value_basis=MeasurementValueBasis.INTERVENTION_DECISION_EVSI,
        information_scope=MeasurementInformationScope.INTERVENTION_OUTCOMES,
        baseline_best_expected_utility=2.0,
        expected_post_measurement_best_utility=2.0 + gross,
        gross_expected_decision_value=gross,
        expected_information_gain=0.4,
        information_gain_metric="expected_query_target_log_score_gain",
        expected_posterior_uncertainty_reduction=0.3,
        uncertainty_reduction_metric="decision_relevant_variance_reduction",
        expected_intervention_ranking_change=0.2,
        ranking_change_metric="top_choice_change_probability",
        raw_assay_cost=2.0,
        assay_cost_units="synthetic_credit",
        assay_cost_penalty=assay_cost,
        expected_delay_seconds=5.0,
        delay_cost=delay_cost,
        destructiveness_cost=destruction_cost,
        net_expected_decision_value=net_value,
        evidence_traces=_evidence_traces(),
        measurement_model_evidence_ids=("assay-validation",),
        measurement_model_evidence_fingerprints={"assay-validation": "a" * 64},
    )


def _causal_support(request: MeasurementDecisionRequest) -> CausalSupportReport:
    query = query_factory()
    target = query.target_outputs[0]
    return CausalSupportReport(
        evaluation_status=EvaluationStatus.EVALUATED,
        outcome=CriterionOutcome.PASSED,
        causal_status=CausalStatus.IDENTIFIED_POPULATION_EFFECT,
        identification_basis="randomized external validation claim",
        identification_design=AssignmentMechanism.RANDOMIZED,
        estimands=(
            CausalEstimandBinding(
                target=target.term,
                horizon_name="acute",
                aggregation=target.aggregation,
                intervention_spec_ids=("drug",),
                comparator="randomized control wells",
                decision_set_fingerprint=measurement_decision_set_fingerprint(request.candidates),
            ),
        ),
        evidence_ids=("causal-validation",),
        evidence_fingerprints={"causal-validation": "b" * 64},
        source_scope="randomized source study",
        target_scope="declared query population",
    )


def _measurement_ready_but_control_uncertain() -> QueryReadinessReport:
    return QueryReadinessReport(
        support=CriterionOutcome.PASSED,
        sufficiency=CriterionOutcome.FAILED,
        identifiability=CriterionOutcome.FAILED,
        decision_uncertainty=CriterionOutcome.FAILED,
        calibration=CriterionOutcome.FAILED,
        causal=CriterionOutcome.PASSED,
        measurement_model=CriterionOutcome.PASSED,
        control_requested=True,
        valid_for_prediction=False,
        valid_for_control=False,
        valid_for_measurement_selection=True,
        abstention_required=True,
        reasons=("measurement may resolve current decision uncertainty",),
    )


def _provenance(query_fingerprint: str) -> ProvenanceRecord:
    return ProvenanceRecord(
        model_id="measurement-policy",
        model_version="test-v1",
        model_fingerprint="c" * 64,
        posterior_schema_id="cellstate/measurement-test-v1",
        query_fingerprint=query_fingerprint,
        history_fingerprint="d" * 64,
        history_structure_fingerprint="e" * 64,
        context_fingerprint="f" * 64,
        support_envelope_id="measurement-envelope",
        support_envelope_fingerprint="1" * 64,
        training_support_id="measurement-training",
        training_support_fingerprint="2" * 64,
        validation_evidence_ids=("assay-validation", "causal-validation"),
        validation_evidence_fingerprints={
            "assay-validation": "a" * 64,
            "causal-validation": "b" * 64,
        },
        seed=7,
    )


def _recommendation(*evaluations: AssayEvaluation) -> MeasurementRecommendation:
    request = _request()
    if not evaluations:
        evaluations = (_supported_evaluation(),)
    assays = tuple(evaluation.assay for evaluation in evaluations)
    return MeasurementRecommendation(
        recommendation_id="recommendation-1",
        status=MeasurementDecisionStatus.RECOMMENDED,
        parent_belief_id=request.parent_belief_id,
        query_fingerprint=request.query_fingerprint,
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
        assays=assays,
        selected_assay_id=assays[0].assay_id,
        evaluations=evaluations,
        readiness=_measurement_ready_but_control_uncertain(),
        causal_status=CausalStatus.IDENTIFIED_POPULATION_EFFECT,
        causal_support=_causal_support(request),
        transport=TransportReport(
            status=TransportStatus.WITHIN_SUPPORT,
            source_domain="randomized source study",
            target_domain="declared query population",
        ),
        minimum_net_decision_value=request.minimum_net_decision_value,
        utility_units=request.utility_units,
        rationale="Highest positive net expected decision value.",
        seed=7,
        provenance=_provenance(request.query_fingerprint),
    )


def test_measurement_request_binds_order_timing_and_every_collection_penalty() -> None:
    request = _request()
    assert request.fingerprint == canonical_fingerprint(request)

    payload = request.model_dump(mode="python")
    payload["candidate_assay_ids"] = ("signal-panel", "signal-panel")
    with pytest.raises(ValidationError, match="assay IDs must be unique"):
        MeasurementDecisionRequest.model_validate(payload)

    payload = request.model_dump(mode="python")
    payload["candidates"] = payload["candidates"][:1]
    with pytest.raises(ValidationError, match="at least 2"):
        MeasurementDecisionRequest.model_validate(payload)

    payload = request.model_dump(mode="python")
    payload["candidate_assay_ids"] = ("",)
    with pytest.raises(ValidationError, match="assay IDs must be nonempty"):
        MeasurementDecisionRequest.model_validate(payload)

    payload = request.model_dump(mode="python")
    payload["decision_deadline_seconds"] = 9
    with pytest.raises(ValidationError, match="cannot predate collection"):
        MeasurementDecisionRequest.model_validate(payload)

    payload = request.model_dump(mode="python")
    del payload["destructiveness_penalties"][CollectionEffect.TERMINAL_DESTRUCTIVE]
    with pytest.raises(ValidationError, match="price every collection effect"):
        MeasurementDecisionRequest.model_validate(payload)


def test_assay_evaluation_requires_real_evsi_and_exact_penalty_math() -> None:
    evaluation = _supported_evaluation()
    assert evaluation.value_basis is MeasurementValueBasis.INTERVENTION_DECISION_EVSI

    payload = evaluation.model_dump(mode="python")
    payload["gross_expected_decision_value"] = 99
    with pytest.raises(ValidationError, match="gross EVSI"):
        AssayEvaluation.model_validate(payload)

    payload = evaluation.model_dump(mode="python")
    payload["net_expected_decision_value"] = 99
    with pytest.raises(ValidationError, match="EVSI minus all penalties"):
        AssayEvaluation.model_validate(payload)

    payload = evaluation.model_dump(mode="python")
    payload["measurement_model_evidence_ids"] = (
        "assay-validation",
        "assay-validation",
    )
    with pytest.raises(ValidationError, match="evidence IDs must be unique"):
        AssayEvaluation.model_validate(payload)

    payload = evaluation.model_dump(mode="python")
    payload["information_gain_metric"] = " "
    with pytest.raises(ValidationError, match="metrics and units must be nonblank"):
        AssayEvaluation.model_validate(payload)

    with pytest.raises(ValidationError, match="must not use numeric"):
        AssayEvaluation(
            assay=_assay_reference(),
            status=SupportStatus.NOT_EVALUATED,
            collection_effect=CollectionEffect.NONDESTRUCTIVE,
            gross_expected_decision_value=0,
            evidence_traces=_evidence_traces(outcome=CriterionOutcome.NOT_EVALUATED),
            reasons=("not evaluated",),
        )


def test_evsi_evidence_traces_are_complete_content_addressed_and_fail_closed() -> None:
    with pytest.raises(ValidationError, match="requires content-addressed evidence"):
        MeasurementEvidenceTrace(
            criterion=MeasurementEvidenceCriterion.ASSAY_OUTCOME_MODEL,
            outcome=CriterionOutcome.PASSED,
            scope_fingerprint="3" * 64,
        )

    with pytest.raises(ValidationError, match="requires explicit reasons"):
        MeasurementEvidenceTrace(
            criterion=MeasurementEvidenceCriterion.HYPOTHETICAL_UPDATE,
            outcome=CriterionOutcome.NOT_EVALUATED,
            scope_fingerprint="3" * 64,
        )

    with pytest.raises(ValidationError, match="cannot claim evidence sentinels"):
        MeasurementEvidenceTrace(
            criterion=MeasurementEvidenceCriterion.DECISION_UTILITY,
            outcome=CriterionOutcome.UNSUPPORTED,
            scope_fingerprint="3" * 64,
            evidence_ids=("assay-validation",),
            evidence_fingerprints={"assay-validation": "a" * 64},
            reasons=("utility validation is unavailable",),
        )

    payload = _supported_evaluation().model_dump(mode="python")
    payload["evidence_traces"] = tuple(reversed(payload["evidence_traces"]))
    with pytest.raises(ValidationError, match="once in canonical order"):
        AssayEvaluation.model_validate(payload)

    payload = _supported_evaluation().model_dump(mode="python")
    traces = list(payload["evidence_traces"])
    traces[1] = MeasurementEvidenceTrace(
        criterion=MeasurementEvidenceCriterion.HYPOTHETICAL_UPDATE,
        outcome=CriterionOutcome.NOT_EVALUATED,
        scope_fingerprint="3" * 64,
        reasons=("hypothetical update was not evaluated",),
    ).model_dump(mode="python")
    payload["evidence_traces"] = traces
    with pytest.raises(ValidationError, match="requires four passing traces"):
        AssayEvaluation.model_validate(payload)


def test_measurement_selection_can_be_ready_while_control_is_not() -> None:
    recommendation = _recommendation()
    assert recommendation.status is MeasurementDecisionStatus.RECOMMENDED
    assert recommendation.readiness.valid_for_measurement_selection
    assert not recommendation.readiness.valid_for_control


def test_recommendation_can_mix_numeric_and_assay_blocked_evaluations() -> None:
    supported = _supported_evaluation("signal-panel")
    blocked = AssayEvaluation(
        assay=_assay_reference("functional-challenge"),
        status=SupportStatus.UNSUPPORTED,
        collection_effect=CollectionEffect.NONDESTRUCTIVE,
        evidence_traces=_evidence_traces(),
        reasons=("assay_support:functional-challenge:unsupported",),
    )

    recommendation = _recommendation(supported, blocked)

    assert recommendation.status is MeasurementDecisionStatus.RECOMMENDED
    assert recommendation.selected_assay_id == "signal-panel"
    assert tuple(evaluation.status for evaluation in recommendation.evaluations) == (
        SupportStatus.SUPPORTED,
        SupportStatus.UNSUPPORTED,
    )


def test_recommendation_uses_strict_threshold_first_tie_and_complete_assay_coverage() -> None:
    first = _supported_evaluation("signal-panel", net_value=1.25)
    second = _supported_evaluation("functional-challenge", net_value=1.25)
    recommendation = _recommendation(first, second)
    assert recommendation.selected_assay_id == "signal-panel"

    payload = recommendation.model_dump(mode="python")
    payload["selected_assay_id"] = "functional-challenge"
    with pytest.raises(ValidationError, match="first maximum-net"):
        MeasurementRecommendation.model_validate(payload)

    payload = recommendation.model_dump(mode="python")
    payload["evaluations"] = payload["evaluations"][:-1]
    with pytest.raises(ValidationError, match="every requested assay"):
        MeasurementRecommendation.model_validate(payload)

    payload = recommendation.model_dump(mode="python")
    payload["minimum_net_decision_value"] = 1.25
    with pytest.raises(ValidationError, match="strictly exceed"):
        MeasurementRecommendation.model_validate(payload)

    payload = recommendation.model_dump(mode="python")
    payload["transport"] = {"status": TransportStatus.NOT_EVALUATED}
    with pytest.raises(ValidationError, match="within validated support"):
        MeasurementRecommendation.model_validate(payload)

    payload = recommendation.model_dump(mode="python")
    payload["causal_support"]["estimands"][0]["decision_set_fingerprint"] = "0" * 64
    with pytest.raises(ValidationError, match="exact ordered decision set"):
        MeasurementRecommendation.model_validate(payload)

    payload = recommendation.model_dump(mode="python")
    payload["transport"]["source_domain"] = "unrelated source"
    with pytest.raises(ValidationError, match="scopes and transport domains must agree"):
        MeasurementRecommendation.model_validate(payload)

    payload = recommendation.model_dump(mode="python")
    payload["transport"]["evidence_ids"] = ("unrecorded-transport-evidence",)
    with pytest.raises(ValidationError, match="transport evidence must be present"):
        MeasurementRecommendation.model_validate(payload)

    payload = recommendation.model_dump(mode="python")
    payload["candidates"] = payload["candidates"][:1]
    with pytest.raises(ValidationError, match="at least 2"):
        MeasurementRecommendation.model_validate(payload)

    near_threshold = _supported_evaluation().model_dump(mode="python")
    penalties = (
        near_threshold["assay_cost_penalty"]
        + near_threshold["delay_cost"]
        + near_threshold["destructiveness_cost"]
    )
    canonical_gross = penalties - 5e-10
    near_threshold["expected_post_measurement_best_utility"] = (
        near_threshold["baseline_best_expected_utility"] + canonical_gross
    )
    near_threshold["gross_expected_decision_value"] = canonical_gross
    near_threshold["net_expected_decision_value"] = 4e-10
    evaluation = AssayEvaluation.model_validate(near_threshold)
    with pytest.raises(ValidationError, match="strictly exceed"):
        _recommendation(evaluation)

    lower = _supported_evaluation("signal-panel", net_value=1_000_000_000.0)
    higher = _supported_evaluation("functional-challenge", net_value=1_000_000_000.5)
    with pytest.raises(ValidationError, match="first maximum-net"):
        _recommendation(lower, higher)


def test_not_evaluated_measurement_decision_has_no_numeric_sentinels() -> None:
    request = _request()
    assay = _assay_reference()
    evaluation = AssayEvaluation(
        assay=assay,
        status=SupportStatus.NOT_EVALUATED,
        collection_effect=CollectionEffect.NONDESTRUCTIVE,
        evidence_traces=_evidence_traces(outcome=CriterionOutcome.NOT_EVALUATED),
        reasons=("assay outcome and counterfactual update models are unavailable",),
    )
    recommendation = MeasurementRecommendation(
        recommendation_id="not-evaluated",
        status=MeasurementDecisionStatus.NOT_EVALUATED,
        parent_belief_id=request.parent_belief_id,
        query_fingerprint=request.query_fingerprint,
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
        assays=(assay,),
        evaluations=(evaluation,),
        readiness=QueryReadinessReport(
            support=CriterionOutcome.PASSED,
            sufficiency=CriterionOutcome.FAILED,
            identifiability=CriterionOutcome.FAILED,
            decision_uncertainty=CriterionOutcome.NOT_EVALUATED,
            calibration=CriterionOutcome.FAILED,
            causal=CriterionOutcome.NOT_EVALUATED,
            measurement_model=CriterionOutcome.UNSUPPORTED,
            control_requested=True,
            valid_for_prediction=False,
            valid_for_control=False,
            valid_for_measurement_selection=False,
            abstention_required=True,
            reasons=("measurement model unavailable",),
        ),
        causal_status=CausalStatus.UNSUPPORTED,
        causal_support=CausalSupportReport(
            evaluation_status=EvaluationStatus.NOT_EVALUATED,
            outcome=CriterionOutcome.NOT_EVALUATED,
            causal_status=CausalStatus.UNSUPPORTED,
        ),
        transport=TransportReport(status=TransportStatus.NOT_EVALUATED),
        minimum_net_decision_value=0,
        utility_units=request.utility_units,
        abstention_reasons=("calibrated decision EVSI is unavailable",),
        rationale="No covariance proxy is reported as value of information.",
        seed=7,
        provenance=_provenance(request.query_fingerprint),
    )
    assert recommendation.selected_assay_id is None
    assert recommendation.evaluations[0].gross_expected_decision_value is None

    payload = recommendation.model_dump(mode="python")
    payload["status"] = MeasurementDecisionStatus.ABSTAINED
    with pytest.raises(ValidationError, match="requires a supported numeric evaluation"):
        MeasurementRecommendation.model_validate(payload)


def test_unsupported_decision_represents_unavailable_assay_support() -> None:
    base = _recommendation()
    unsupported_evaluation = AssayEvaluation(
        assay=base.assays[0],
        status=SupportStatus.UNSUPPORTED,
        collection_effect=CollectionEffect.NONDESTRUCTIVE,
        evidence_traces=_evidence_traces(outcome=CriterionOutcome.UNSUPPORTED),
        reasons=("assay is outside validated support",),
    )
    payload = base.model_dump(mode="python")
    payload.update(
        {
            "status": MeasurementDecisionStatus.UNSUPPORTED,
            "selected_assay_id": None,
            "evaluations": (unsupported_evaluation.model_dump(mode="python"),),
            "abstention_reasons": ("no requested assay is supported",),
        }
    )
    unsupported = MeasurementRecommendation.model_validate(payload)
    assert unsupported.status is MeasurementDecisionStatus.UNSUPPORTED

    payload["status"] = MeasurementDecisionStatus.NOT_EVALUATED
    with pytest.raises(ValidationError, match="requires not-evaluated assay evaluations"):
        MeasurementRecommendation.model_validate(payload)

    payload["status"] = MeasurementDecisionStatus.UNSUPPORTED
    payload["evaluations"] = (
        unsupported_evaluation.model_copy(
            update={
                "status": SupportStatus.NOT_EVALUATED,
                "evidence_traces": _evidence_traces(outcome=CriterionOutcome.NOT_EVALUATED),
            }
        ).model_dump(mode="python"),
    )
    with pytest.raises(ValidationError, match="at least one unsupported assay"):
        MeasurementRecommendation.model_validate(payload)


def test_measurement_scientific_evidence_and_transport_text_cannot_be_vacuous() -> None:
    request = _request()

    provenance_payload = _provenance(request.query_fingerprint).model_dump(mode="python")
    provenance_payload["validation_evidence_ids"] = (" ",)
    provenance_payload["validation_evidence_fingerprints"] = {" ": "a" * 64}
    with pytest.raises(ValidationError, match="validation evidence IDs must be nonblank"):
        ProvenanceRecord.model_validate(provenance_payload)

    causal_payload = _causal_support(request).model_dump(mode="python")
    causal_payload["identification_basis"] = " "
    with pytest.raises(ValidationError, match="basis and scopes must be nonblank"):
        CausalSupportReport.model_validate(causal_payload)

    estimand_payload = causal_payload["estimands"][0]
    estimand_payload["comparator"] = " "
    with pytest.raises(ValidationError, match="comparator must be nonblank"):
        CausalEstimandBinding.model_validate(estimand_payload)

    with pytest.raises(ValidationError, match="assumptions must be nonblank"):
        TransportReport(
            status=TransportStatus.TRANSPORTED,
            source_domain="source",
            target_domain="target",
            assumptions=(" ",),
        )


def test_measurement_trace_evidence_fingerprints_must_match_provenance() -> None:
    payload = _recommendation().model_dump(mode="python")
    evaluations = list(payload["evaluations"])
    traces = list(evaluations[0]["evidence_traces"])
    traces[1]["evidence_fingerprints"] = {"assay-validation": "9" * 64}
    evaluations[0]["evidence_traces"] = traces
    payload["evaluations"] = evaluations
    with pytest.raises(ValidationError, match="fingerprints must match provenance"):
        MeasurementRecommendation.model_validate(payload)
