from __future__ import annotations

import pytest
from conftest import (
    SYNTHETIC_TEST_OPTIONS,
    intervention_factory,
    request_factory,
    subject_factory,
)
from pydantic import ValidationError

from cellstate import estimate_cell_state
from cellstate.domain.belief import CellStateBelief
from cellstate.domain.common import CriterionOutcome, OntologyTerm, SupportStatus
from cellstate.domain.events import CollectionEffect
from cellstate.domain.measurements import (
    MeasurementDecisionRequest,
    MeasurementDecisionStatus,
    MeasurementEvidenceCriterion,
)
from cellstate.domain.request import InferenceOptions
from cellstate.domain.scenarios import (
    EvolutionScenario,
    InterventionObjective,
    ObjectiveDirection,
    ObjectiveTerm,
)
from cellstate.errors import CapabilityError
from cellstate.ports import (
    MeasurementCapabilityReport,
    MeasurementPolicy,
    measurement_capability_scope_fingerprint,
)
from cellstate.reference import LinearGaussianMeasurementPolicy


def _decision_request(
    belief: CellStateBelief,
    *,
    assay_ids: tuple[str, ...] = ("signal-panel",),
) -> MeasurementDecisionRequest:
    objective = InterventionObjective(
        objective_id="maximize-function",
        horizon_name="acute",
        terms=(
            ObjectiveTerm(
                target=OntologyTerm(label="functional capacity"),
                direction=ObjectiveDirection.MAXIMIZE,
            ),
        ),
    )
    baseline = EvolutionScenario(
        scenario_id="baseline",
        horizon_name="acute",
        subject=subject_factory(),
        start_time_seconds=belief.as_of_seconds,
        end_time_seconds=belief.as_of_seconds + 60,
    )
    treated = EvolutionScenario(
        scenario_id="treated",
        horizon_name="acute",
        subject=baseline.subject,
        start_time_seconds=belief.as_of_seconds,
        end_time_seconds=belief.as_of_seconds + 60,
        interventions=(
            intervention_factory(
                event_id="future-treatment",
                subject=baseline.subject,
                time_seconds=belief.as_of_seconds + 5,
                duration_seconds=10,
                estimated_efficiency=None,
            ),
        ),
    )
    return MeasurementDecisionRequest(
        request_id="measurement-decision",
        parent_belief_id=belief.belief_id,
        query_fingerprint=belief.query_fingerprint,
        objective=objective,
        candidates=(baseline, treated),
        candidate_assay_ids=assay_ids,
        collection_time_seconds=belief.as_of_seconds,
        decision_deadline_seconds=belief.as_of_seconds + 5,
        minimum_net_decision_value=0,
        utility_units="synthetic_utility",
        assay_cost_to_utility_rate=1,
        delay_penalty_per_second=0,
        destructiveness_penalties={
            CollectionEffect.NONDESTRUCTIVE: 0,
            CollectionEffect.VIABILITY_PRESERVING_WITH_KNOWN_EFFECT: 1,
            CollectionEffect.PARTIALLY_DESTRUCTIVE_POPULATION_SAMPLING: 10,
            CollectionEffect.TERMINAL_DESTRUCTIVE: 100,
        },
    )


def test_reference_measurement_capabilities_are_exact_and_scientifically_honest(
    model,
) -> None:
    belief = estimate_cell_state(
        request_factory(),
        estimator=model,
        options=SYNTHETIC_TEST_OPTIONS,
    )
    request = _decision_request(belief)
    policy = LinearGaussianMeasurementPolicy(model)

    assert isinstance(policy, MeasurementPolicy)
    assert policy.descriptor == model.descriptor
    report = policy.capabilities(belief, request)
    assert report.supported
    assert report.scope_fingerprint == measurement_capability_scope_fingerprint(belief, request)
    assert report.assay_support == {"signal-panel": SupportStatus.NOT_EVALUATED}
    assert (
        report.collection_effect_support[CollectionEffect.NONDESTRUCTIVE]
        is SupportStatus.NOT_EVALUATED
    )
    assert all(
        status is SupportStatus.UNSUPPORTED
        for effect, status in report.collection_effect_support.items()
        if effect is not CollectionEffect.NONDESTRUCTIVE
    )
    assert {
        report.assay_outcome_model,
        report.hypothetical_update,
        report.counterfactual_replanning,
        report.decision_utility,
    } == {CriterionOutcome.NOT_EVALUATED}


def test_reference_measurement_policy_returns_complete_seeded_abstention(model) -> None:
    belief = estimate_cell_state(
        request_factory(),
        estimator=model,
        options=SYNTHETIC_TEST_OPTIONS,
    )
    request = _decision_request(belief)
    policy = LinearGaussianMeasurementPolicy(model)

    first = policy.recommend(belief, request, options=InferenceOptions(seed=17))
    repeated = policy.recommend(belief, request, options=InferenceOptions(seed=17))
    changed_seed = policy.recommend(belief, request, options=InferenceOptions(seed=18))

    assert first == repeated
    assert first.recommendation_id != changed_seed.recommendation_id
    assert first.status is MeasurementDecisionStatus.NOT_EVALUATED
    assert first.selected_assay_id is None
    assert first.parent_belief_id == belief.belief_id
    assert first.query_fingerprint == belief.query_fingerprint
    assert first.request_fingerprint == request.fingerprint
    assert first.seed == first.provenance.seed == 17
    assert first.causal_status is belief.diagnostics.causal_support.causal_status
    assert first.causal_support == belief.diagnostics.causal_support
    assert first.provenance.model_id == belief.provenance.model_id
    assert first.provenance.model_fingerprint == belief.provenance.model_fingerprint
    assert first.provenance.source_event_ids == belief.provenance.source_event_ids
    assert len(first.evaluations) == len(request.candidate_assay_ids) == 1

    evaluation = first.evaluations[0]
    assert evaluation.status is SupportStatus.NOT_EVALUATED
    assert evaluation.value_basis is None
    assert evaluation.expected_information_gain is None
    assert evaluation.expected_posterior_uncertainty_reduction is None
    assert evaluation.net_expected_decision_value is None
    assert evaluation.measurement_model_evidence_ids == ()
    assert tuple(trace.criterion for trace in evaluation.evidence_traces) == tuple(
        MeasurementEvidenceCriterion
    )
    assert all(
        trace.outcome is CriterionOutcome.NOT_EVALUATED
        and trace.scope_fingerprint == measurement_capability_scope_fingerprint(belief, request)
        and not trace.evidence_ids
        and not trace.evidence_fingerprints
        and trace.reasons
        for trace in evaluation.evidence_traces
    )
    assert "covariance reduction is not" in " ".join(evaluation.notes).casefold()


def test_reference_measurement_policy_reports_unknown_assay_blocker(model) -> None:
    belief = estimate_cell_state(
        request_factory(),
        estimator=model,
        options=SYNTHETIC_TEST_OPTIONS,
    )
    request = _decision_request(belief, assay_ids=("unknown-assay",))
    policy = LinearGaussianMeasurementPolicy(model)

    report = policy.capabilities(belief, request)
    assert not report.supported
    assert report.assay_support == {"unknown-assay": SupportStatus.UNSUPPORTED}
    assert report.blockers == ("unknown_assay:unknown-assay",)
    with pytest.raises(CapabilityError, match="outside the query"):
        policy.recommend(belief, request, options=SYNTHETIC_TEST_OPTIONS)


def test_measurement_capability_report_requires_complete_explicit_scope() -> None:
    payload = {
        "supported": True,
        "scope_fingerprint": "1" * 64,
        "assay_support": {"assay": SupportStatus.NOT_EVALUATED},
        "collection_effect_support": {
            effect: SupportStatus.NOT_EVALUATED for effect in CollectionEffect
        },
        "assay_outcome_model": CriterionOutcome.NOT_EVALUATED,
        "hypothetical_update": CriterionOutcome.NOT_EVALUATED,
        "counterfactual_replanning": CriterionOutcome.NOT_EVALUATED,
        "decision_utility": CriterionOutcome.NOT_EVALUATED,
    }
    with pytest.raises(ValidationError, match="cannot have blockers"):
        MeasurementCapabilityReport.model_validate({**payload, "blockers": ("structural blocker",)})
    with pytest.raises(ValidationError, match="requires blockers"):
        MeasurementCapabilityReport.model_validate({**payload, "supported": False})
    with pytest.raises(ValidationError, match="every requested assay"):
        MeasurementCapabilityReport.model_validate({**payload, "assay_support": {}})
    with pytest.raises(ValidationError, match="IDs must be nonempty"):
        MeasurementCapabilityReport.model_validate(
            {**payload, "assay_support": {"": SupportStatus.NOT_EVALUATED}}
        )
    with pytest.raises(ValidationError, match="every collection effect"):
        MeasurementCapabilityReport.model_validate(
            {
                **payload,
                "collection_effect_support": {
                    CollectionEffect.NONDESTRUCTIVE: SupportStatus.NOT_EVALUATED
                },
            }
        )
