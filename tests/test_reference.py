from __future__ import annotations

import numpy as np
import pytest
from conftest import (
    SYNTHETIC_TEST_OPTIONS,
    environment_factory,
    environment_spec_factory,
    intervention_factory,
    observation_factory,
    query_factory,
    request_factory,
    subject_factory,
)

from cellstate import (
    CellHistory,
    CensoringDirection,
    EstimateCellStateRequest,
    EvidenceRole,
    EvolutionScenario,
    InterventionObjective,
    MissingnessReport,
    MissingnessStatus,
    ObjectiveDirection,
    ObjectiveTerm,
    OntologyTerm,
    Quantity,
    StateQuery,
    SystemBoundary,
    choose_intervention,
    estimate_cell_state,
    evolve_cell_state,
)
from cellstate.domain import ParametricDistribution, PlanStatus, SupportStatus
from cellstate.domain.belief import EvaluationStatus
from cellstate.domain.common import CausalStatus
from cellstate.domain.subjects import IdentityBasis
from cellstate.errors import CapabilityError, ContractViolationError
from cellstate.reference import LinearGaussianPlanner, sample_posterior


def _trace(belief: object) -> float:
    posterior = belief.joint_posterior
    assert isinstance(posterior, ParametricDistribution)
    return float(np.trace(np.asarray(posterior.covariance)))


def test_estimation_returns_structured_auditable_belief(model, estimate_request) -> None:
    belief = estimate_cell_state(
        estimate_request,
        estimator=model,
        options=SYNTHETIC_TEST_OPTIONS,
    )
    assert isinstance(belief.joint_posterior, ParametricDistribution)
    assert len(belief.factors) == len(belief.state_specification.active_factors) == 4
    assert belief.query_fingerprint == estimate_request.query.fingerprint
    assert belief.history_fingerprint == estimate_request.history.fingerprint
    assert belief.provenance.source_event_ids == ("obs-0",)
    assert belief.diagnostics.sufficiency.evaluation_status is EvaluationStatus.NOT_EVALUATED
    assert "next_measurement" not in belief.model_fields_set


def test_observation_reduces_uncertainty_relative_to_missing(model) -> None:
    observed_request = request_factory()
    missing = observation_factory(
        event_id="obs-0",
        time_seconds=0,
        modality="transcriptome",
        value=None,
        missingness=MissingnessReport(status=MissingnessStatus.MISSING),
    )
    missing_request = request_factory(
        history=CellHistory(subject=subject_factory(), events=(missing,))
    )
    assert _trace(
        estimate_cell_state(
            observed_request,
            estimator=model,
            options=SYNTHETIC_TEST_OPTIONS,
        )
    ) < _trace(
        estimate_cell_state(
            missing_request,
            estimator=model,
            options=SYNTHETIC_TEST_OPTIONS,
        )
    )


def test_recursive_update_matches_single_pass(model, query) -> None:
    first_history = CellHistory(
        subject=subject_factory(),
        events=(observation_factory(event_id="rna", time_seconds=0),),
    )
    first_request = request_factory(history=first_history, as_of_seconds=5, query=query)
    first_belief = estimate_cell_state(
        first_request,
        estimator=model,
        options=SYNTHETIC_TEST_OPTIONS,
    )

    second_observation = observation_factory(
        event_id="signal", time_seconds=10, modality="phosphosignaling", value=0.8
    )
    full_history = CellHistory(
        subject=subject_factory(),
        events=(*first_history.events, second_observation),
        completeness=first_request.history.completeness,
    )
    recursive_request = EstimateCellStateRequest(
        query=query,
        history=full_history,
        as_of_seconds=10,
        static_context=first_request.static_context,
        previous_belief=first_belief,
    )
    batch_request = request_factory(history=full_history, as_of_seconds=10, query=query)
    recursive = estimate_cell_state(
        recursive_request,
        estimator=model,
        options=SYNTHETIC_TEST_OPTIONS,
    )
    batch = estimate_cell_state(
        batch_request,
        estimator=model,
        options=SYNTHETIC_TEST_OPTIONS,
    )
    assert isinstance(recursive.joint_posterior, ParametricDistribution)
    assert isinstance(batch.joint_posterior, ParametricDistribution)
    assert np.allclose(recursive.joint_posterior.mean, batch.joint_posterior.mean)
    assert np.allclose(recursive.joint_posterior.covariance, batch.joint_posterior.covariance)


def test_controlled_forecast_changes_posterior(model, estimate_request) -> None:
    belief = estimate_cell_state(
        estimate_request,
        estimator=model,
        options=SYNTHETIC_TEST_OPTIONS,
    )
    baseline_scenario = EvolutionScenario(
        scenario_id="baseline",
        horizon_name="acute",
        subject=subject_factory(),
        start_time_seconds=10,
        end_time_seconds=70,
    )
    drug = intervention_factory(
        event_id="drug",
        time_seconds=10,
        duration_seconds=10,
        estimated_efficiency=None,
    )
    drug_scenario = baseline_scenario.model_copy(
        update={"scenario_id": "drug", "interventions": (drug,)}
    )
    baseline = evolve_cell_state(
        belief,
        scenario=baseline_scenario,
        evolution_model=model,
        options=SYNTHETIC_TEST_OPTIONS,
    )
    treated = evolve_cell_state(
        belief,
        scenario=drug_scenario,
        evolution_model=model,
        options=SYNTHETIC_TEST_OPTIONS,
    )
    assert isinstance(baseline.joint_posterior, ParametricDistribution)
    assert isinstance(treated.joint_posterior, ParametricDistribution)
    assert not np.allclose(baseline.joint_posterior.mean, treated.joint_posterior.mean)
    assert len(treated.target_predictions) == 1
    assert treated.target_predictions[0].status is SupportStatus.SUPPORTED
    assert treated.target_predictions[0].horizon_seconds == 60


def test_posterior_sampling_is_seeded(model, estimate_request) -> None:
    belief = estimate_cell_state(
        estimate_request,
        estimator=model,
        options=SYNTHETIC_TEST_OPTIONS,
    )
    first = sample_posterior(belief, 4, seed=7)
    second = sample_posterior(belief, 4, seed=7)
    third = sample_posterior(belief, 4, seed=8)
    assert first == second
    assert first != third
    with pytest.raises(ValueError, match="positive"):
        sample_posterior(belief, 0, seed=7)


def test_unknown_observed_modality_is_rejected(model) -> None:
    event = observation_factory(modality="unknown_modality")
    base_query = query_factory()
    query = base_query.model_copy(
        update={
            "evidence_policy": base_query.evidence_policy.model_copy(
                update={
                    "allowed_modalities": (
                        *base_query.evidence_policy.allowed_modalities,
                        OntologyTerm(label="unknown modality"),
                    )
                }
            )
        }
    )
    request = request_factory(
        history=CellHistory(subject=subject_factory(), events=(event,)),
        query=query,
    )
    with pytest.raises(CapabilityError, match="unknown_modality"):
        estimate_cell_state(
            request,
            estimator=model,
            options=SYNTHETIC_TEST_OPTIONS,
        )


def test_censored_likelihood_is_not_silently_approximated(model) -> None:
    event = observation_factory(
        event_id="censored",
        time_seconds=0,
        modality="transcriptome",
        value=None,
        units="relative",
        missingness=MissingnessReport(
            status=MissingnessStatus.CENSORED,
            censoring_direction=CensoringDirection.BELOW,
            detection_limit=Quantity(value=0.1, units="relative"),
        ),
    )
    request = request_factory(history=CellHistory(subject=subject_factory(), events=(event,)))
    with pytest.raises(CapabilityError, match="censored"):
        estimate_cell_state(
            request,
            estimator=model,
            options=SYNTHETIC_TEST_OPTIONS,
        )


def test_intervention_without_efficacy_uses_reference_prior_and_abstains(model) -> None:
    intervention = intervention_factory(
        event_id="drug",
        time_seconds=0,
        duration_seconds=1,
        estimated_efficiency=None,
    )
    request = request_factory(
        history=CellHistory(subject=subject_factory(), events=(intervention,))
    )
    belief = estimate_cell_state(
        request,
        estimator=model,
        options=SYNTHETIC_TEST_OPTIONS,
    )
    assert belief.readiness.abstention_required
    assert not belief.readiness.valid_for_control
    assert belief.diagnostics.causal_support.causal_status is CausalStatus.UNSUPPORTED
    assert belief.intervention_realizations == ()


def test_unknown_environment_is_rejected(model) -> None:
    environment = environment_factory(
        event_id="environment",
        time_seconds=0,
        duration_seconds=10,
        variables={"unknown": Quantity(value=1, units="relative")},
    )
    base_query = query_factory()
    query = StateQuery.model_validate(
        {
            **base_query.model_dump(),
            "system_boundary": SystemBoundary.CELL_AND_SOLUBLE_ENVIRONMENT,
            "environment_space": (environment_spec_factory(variable="unknown"),),
            "evidence_policy": base_query.evidence_policy.model_copy(
                update={"lookback_seconds": 10.0}
            ),
        }
    )
    request = request_factory(
        history=CellHistory(subject=subject_factory(), events=(environment,)), query=query
    )
    with pytest.raises(CapabilityError, match="unknown"):
        estimate_cell_state(
            request,
            estimator=model,
            options=SYNTHETIC_TEST_OPTIONS,
        )


def test_reference_planner_abstains_without_scientific_support(model, estimate_request) -> None:
    belief = estimate_cell_state(
        estimate_request,
        estimator=model,
        options=SYNTHETIC_TEST_OPTIONS,
    )
    baseline = EvolutionScenario(
        scenario_id="baseline",
        horizon_name="acute",
        subject=subject_factory(),
        start_time_seconds=10,
        end_time_seconds=70,
    )
    drug = intervention_factory(
        event_id="drug-plan",
        time_seconds=10,
        duration_seconds=10,
        estimated_efficiency=None,
    )
    stimulated = baseline.model_copy(update={"scenario_id": "stimulated", "interventions": (drug,)})
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
    plan = choose_intervention(
        belief,
        objective=objective,
        candidates=(baseline, stimulated),
        planner=LinearGaussianPlanner(model),
        options=SYNTHETIC_TEST_OPTIONS,
    )
    assert plan.status is PlanStatus.ABSTAINED
    assert plan.selected_scenario_id is None
    assert plan.abstention_reasons
    assert all(not evaluation.supported for evaluation in plan.evaluations)


def test_same_time_recursive_update_assimilates_new_evidence(model, query) -> None:
    empty_request = request_factory(
        history=CellHistory(subject=subject_factory()), as_of_seconds=0, query=query
    )
    prior_belief = estimate_cell_state(
        empty_request,
        estimator=model,
        options=SYNTHETIC_TEST_OPTIONS,
    )
    observation = observation_factory(event_id="same-time", time_seconds=0, value=10)
    history = CellHistory(
        subject=subject_factory(),
        events=(observation,),
        completeness=empty_request.history.completeness,
    )
    recursive_request = EstimateCellStateRequest(
        query=query,
        history=history,
        as_of_seconds=0,
        static_context=empty_request.static_context,
        previous_belief=prior_belief,
    )
    batch_request = request_factory(history=history, as_of_seconds=0, query=query)
    recursive = estimate_cell_state(
        recursive_request,
        estimator=model,
        options=SYNTHETIC_TEST_OPTIONS,
    )
    batch = estimate_cell_state(
        batch_request,
        estimator=model,
        options=SYNTHETIC_TEST_OPTIONS,
    )
    assert isinstance(recursive.joint_posterior, ParametricDistribution)
    assert isinstance(batch.joint_posterior, ParametricDistribution)
    assert np.allclose(recursive.joint_posterior.mean, batch.joint_posterior.mean)


def test_reference_rejects_instantaneous_intervention_no_op(model, estimate_request) -> None:
    belief = estimate_cell_state(
        estimate_request,
        estimator=model,
        options=SYNTHETIC_TEST_OPTIONS,
    )
    impulse = intervention_factory(
        event_id="impulse",
        time_seconds=10,
        duration_seconds=0,
        estimated_efficiency=None,
    )
    scenario = EvolutionScenario(
        scenario_id="impulse",
        horizon_name="acute",
        subject=subject_factory(),
        start_time_seconds=10,
        end_time_seconds=70,
        interventions=(impulse,),
    )
    with pytest.raises(CapabilityError, match="instantaneous"):
        evolve_cell_state(
            belief,
            scenario=scenario,
            evolution_model=model,
            options=SYNTHETIC_TEST_OPTIONS,
        )


def test_reference_rejects_lineage_evidence_it_cannot_model(model) -> None:
    sibling = observation_factory(
        evidence_role=EvidenceRole.SIBLING,
        source_subject=subject_factory("sibling-1"),
        linkage_basis=IdentityBasis.HERITABLE_BARCODE,
    )
    base_query = query_factory()
    query = base_query.model_copy(
        update={
            "evidence_policy": base_query.evidence_policy.model_copy(
                update={
                    "allowed_evidence_roles": (
                        EvidenceRole.DIRECT,
                        EvidenceRole.SIBLING,
                    )
                }
            )
        }
    )
    request = request_factory(
        history=CellHistory(subject=subject_factory(), events=(sibling,)),
        query=query,
    )
    with pytest.raises(CapabilityError, match="lineage/population"):
        estimate_cell_state(
            request,
            estimator=model,
            options=SYNTHETIC_TEST_OPTIONS,
        )


def test_forecast_requires_explicit_environment_persistence(model, query) -> None:
    environment_query = StateQuery.model_validate(
        {
            **query.model_dump(),
            "system_boundary": SystemBoundary.CELL_AND_SOLUBLE_ENVIRONMENT,
            "environment_space": (environment_spec_factory(),),
            "evidence_policy": query.evidence_policy.model_copy(update={"lookback_seconds": 10.0}),
        }
    )
    environment = environment_factory(
        event_id="environment",
        time_seconds=0,
        duration_seconds=11,
        variables={"nutrient": Quantity(value=1, units="relative")},
    )
    history = CellHistory(
        subject=subject_factory(), events=(environment, observation_factory(event_id="rna"))
    )
    belief = estimate_cell_state(
        request_factory(history=history, query=environment_query),
        estimator=model,
        options=SYNTHETIC_TEST_OPTIONS,
    )
    scenario = EvolutionScenario(
        scenario_id="future",
        horizon_name="acute",
        subject=subject_factory(),
        start_time_seconds=10,
        end_time_seconds=70,
    )
    with pytest.raises(ContractViolationError, match="explicitly inherit or clear"):
        evolve_cell_state(
            belief,
            scenario=scenario,
            evolution_model=model,
            options=SYNTHETIC_TEST_OPTIONS,
        )
    inherited = scenario.model_copy(update={"inherit_current_environment": True})
    forecast = evolve_cell_state(
        belief,
        scenario=inherited,
        evolution_model=model,
        options=SYNTHETIC_TEST_OPTIONS,
    )
    assert isinstance(forecast.joint_posterior, ParametricDistribution)
