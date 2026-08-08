from __future__ import annotations

import numpy as np
import pytest
from conftest import observation_factory, query_factory, request_factory

from cellstate import (
    AssayMetadata,
    CellHistory,
    CensoringDirection,
    EnvironmentEvent,
    EnvironmentVariableSpec,
    EstimateCellStateRequest,
    EvidenceRole,
    EvolutionScenario,
    InterventionEvent,
    InterventionObjective,
    MissingnessReport,
    MissingnessStatus,
    ObjectiveDirection,
    ObjectiveTerm,
    ObservationEvent,
    OntologyTerm,
    Quantity,
    StateQuery,
    SystemBoundary,
    choose_intervention,
    estimate_cell_state,
    evolve_cell_state,
)
from cellstate.domain import ParametricDistribution, SupportStatus
from cellstate.errors import CapabilityError, UnsupportedInterventionError, UnsupportedModalityError
from cellstate.reference import LinearGaussianPlanner, sample_posterior


def _trace(belief: object) -> float:
    posterior = belief.joint_posterior
    assert isinstance(posterior, ParametricDistribution)
    return float(np.trace(np.asarray(posterior.covariance)))


def test_estimation_returns_structured_auditable_belief(model, estimate_request) -> None:
    belief = estimate_cell_state(estimate_request, estimator=model)
    assert isinstance(belief.joint_posterior, ParametricDistribution)
    assert len(belief.factors) == 8
    assert belief.query_fingerprint == estimate_request.query.fingerprint
    assert belief.history_fingerprint == estimate_request.history.fingerprint
    assert belief.provenance.source_event_ids == ("obs-0",)
    assert belief.diagnostics.sufficiency.status is SupportStatus.NOT_EVALUATED
    assert belief.next_measurement.status is SupportStatus.NOT_EVALUATED


def test_observation_reduces_uncertainty_relative_to_missing(model) -> None:
    observed_request = request_factory()
    missing = ObservationEvent(
        event_id="obs-0",
        subject_id="cell-1",
        time_seconds=0,
        modality=OntologyTerm(label="transcriptome"),
        value=None,
        missingness=MissingnessReport(status=MissingnessStatus.MISSING),
        assay=AssayMetadata(assay_id="rna"),
    )
    missing_request = request_factory(history=CellHistory(subject_id="cell-1", events=(missing,)))
    assert _trace(estimate_cell_state(observed_request, estimator=model)) < _trace(
        estimate_cell_state(missing_request, estimator=model)
    )


def test_recursive_update_matches_single_pass(model, query) -> None:
    first_history = CellHistory(
        subject_id="cell-1", events=(observation_factory(event_id="rna", time_seconds=0),)
    )
    first_request = request_factory(history=first_history, as_of_seconds=5, query=query)
    first_belief = estimate_cell_state(first_request, estimator=model)

    second_observation = observation_factory(
        event_id="signal", time_seconds=10, modality="phosphosignaling", value=0.8
    )
    full_history = CellHistory(
        subject_id="cell-1",
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
    recursive = estimate_cell_state(recursive_request, estimator=model)
    batch = estimate_cell_state(batch_request, estimator=model)
    assert isinstance(recursive.joint_posterior, ParametricDistribution)
    assert isinstance(batch.joint_posterior, ParametricDistribution)
    assert np.allclose(recursive.joint_posterior.mean, batch.joint_posterior.mean)
    assert np.allclose(recursive.joint_posterior.covariance, batch.joint_posterior.covariance)


def test_controlled_forecast_changes_posterior(model, estimate_request) -> None:
    belief = estimate_cell_state(estimate_request, estimator=model)
    baseline_scenario = EvolutionScenario(
        scenario_id="baseline",
        horizon_name="acute",
        subject_id="cell-1",
        start_time_seconds=10,
        end_time_seconds=70,
    )
    drug = InterventionEvent(
        event_id="drug",
        subject_id="cell-1",
        time_seconds=10,
        duration_seconds=10,
        intervention_type=OntologyTerm(label="drug"),
        dose=Quantity(value=1, units="relative"),
        estimated_efficiency=1,
    )
    drug_scenario = baseline_scenario.model_copy(
        update={"scenario_id": "drug", "interventions": (drug,)}
    )
    baseline = evolve_cell_state(belief, scenario=baseline_scenario, evolution_model=model)
    treated = evolve_cell_state(belief, scenario=drug_scenario, evolution_model=model)
    assert isinstance(baseline.joint_posterior, ParametricDistribution)
    assert isinstance(treated.joint_posterior, ParametricDistribution)
    assert not np.allclose(baseline.joint_posterior.mean, treated.joint_posterior.mean)
    assert len(treated.target_predictions) == 1
    assert treated.target_predictions[0].status is SupportStatus.SUPPORTED
    assert treated.target_predictions[0].horizon_seconds == 60


def test_posterior_sampling_is_seeded(model, estimate_request) -> None:
    belief = estimate_cell_state(estimate_request, estimator=model)
    first = sample_posterior(belief, 4, seed=7)
    second = sample_posterior(belief, 4, seed=7)
    third = sample_posterior(belief, 4, seed=8)
    assert first == second
    assert first != third
    with pytest.raises(ValueError, match="positive"):
        sample_posterior(belief, 0, seed=7)


def test_unknown_observed_modality_is_rejected(model) -> None:
    event = observation_factory(modality="unknown_modality")
    request = request_factory(history=CellHistory(subject_id="cell-1", events=(event,)))
    with pytest.raises(UnsupportedModalityError):
        estimate_cell_state(request, estimator=model)


def test_censored_likelihood_is_not_silently_approximated(model) -> None:
    event = ObservationEvent(
        event_id="censored",
        subject_id="cell-1",
        time_seconds=0,
        modality=OntologyTerm(label="transcriptome"),
        value=None,
        units="relative",
        missingness=MissingnessReport(
            status=MissingnessStatus.CENSORED,
            censoring_direction=CensoringDirection.BELOW,
            detection_limit=Quantity(value=0.1, units="relative"),
        ),
        assay=AssayMetadata(assay_id="rna"),
    )
    request = request_factory(history=CellHistory(subject_id="cell-1", events=(event,)))
    with pytest.raises(CapabilityError, match="censored"):
        estimate_cell_state(request, estimator=model)


def test_intervention_without_efficacy_is_rejected(model) -> None:
    intervention = InterventionEvent(
        event_id="drug",
        subject_id="cell-1",
        time_seconds=0,
        duration_seconds=1,
        intervention_type=OntologyTerm(label="drug"),
    )
    request = request_factory(history=CellHistory(subject_id="cell-1", events=(intervention,)))
    with pytest.raises(UnsupportedInterventionError, match="efficacy"):
        estimate_cell_state(request, estimator=model)


def test_unknown_environment_is_rejected(model) -> None:
    environment = EnvironmentEvent(
        event_id="environment",
        subject_id="cell-1",
        time_seconds=0,
        variables={"unknown": 1.0},
    )
    base_query = query_factory()
    query = StateQuery.model_validate(
        {
            **base_query.model_dump(),
            "system_boundary": SystemBoundary.CELL_AND_SOLUBLE_ENVIRONMENT,
            "environment_space": (EnvironmentVariableSpec(variable=OntologyTerm(label="unknown")),),
        }
    )
    request = request_factory(
        history=CellHistory(subject_id="cell-1", events=(environment,)), query=query
    )
    with pytest.raises(CapabilityError, match="unknown"):
        estimate_cell_state(request, estimator=model)


def test_reference_planner_selects_supported_candidate(model, estimate_request) -> None:
    belief = estimate_cell_state(estimate_request, estimator=model)
    baseline = EvolutionScenario(
        scenario_id="baseline",
        horizon_name="acute",
        subject_id="cell-1",
        start_time_seconds=10,
        end_time_seconds=70,
    )
    drug = InterventionEvent(
        event_id="drug-plan",
        subject_id="cell-1",
        time_seconds=10,
        duration_seconds=10,
        intervention_type=OntologyTerm(label="drug"),
        dose=Quantity(value=1, units="relative"),
        estimated_efficiency=1,
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
    )
    assert plan.selected_scenario_id == "stimulated"
    assert all(evaluation.supported for evaluation in plan.evaluations)


def test_same_time_recursive_update_assimilates_new_evidence(model, query) -> None:
    empty_request = request_factory(
        history=CellHistory(subject_id="cell-1"), as_of_seconds=0, query=query
    )
    prior_belief = estimate_cell_state(empty_request, estimator=model)
    observation = observation_factory(event_id="same-time", time_seconds=0, value=10)
    history = CellHistory(
        subject_id="cell-1",
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
    recursive = estimate_cell_state(recursive_request, estimator=model)
    batch = estimate_cell_state(batch_request, estimator=model)
    assert isinstance(recursive.joint_posterior, ParametricDistribution)
    assert isinstance(batch.joint_posterior, ParametricDistribution)
    assert np.allclose(recursive.joint_posterior.mean, batch.joint_posterior.mean)


def test_reference_rejects_instantaneous_intervention_no_op(model, estimate_request) -> None:
    belief = estimate_cell_state(estimate_request, estimator=model)
    impulse = InterventionEvent(
        event_id="impulse",
        subject_id="cell-1",
        time_seconds=10,
        duration_seconds=0,
        intervention_type=OntologyTerm(label="drug"),
        dose=Quantity(value=1, units="relative"),
        estimated_efficiency=1,
    )
    scenario = EvolutionScenario(
        scenario_id="impulse",
        horizon_name="acute",
        subject_id="cell-1",
        start_time_seconds=10,
        end_time_seconds=70,
        interventions=(impulse,),
    )
    with pytest.raises(UnsupportedInterventionError, match="instantaneous"):
        evolve_cell_state(belief, scenario=scenario, evolution_model=model)


def test_reference_rejects_lineage_evidence_it_cannot_model(model) -> None:
    sibling = ObservationEvent.model_validate(
        {
            **observation_factory().model_dump(),
            "evidence_role": EvidenceRole.SIBLING,
            "source_subject_id": "sibling-1",
        }
    )
    request = request_factory(history=CellHistory(subject_id="cell-1", events=(sibling,)))
    with pytest.raises(CapabilityError, match="lineage/population"):
        estimate_cell_state(request, estimator=model)


def test_forecast_requires_explicit_environment_persistence(model, query) -> None:
    environment_query = StateQuery.model_validate(
        {
            **query.model_dump(),
            "system_boundary": SystemBoundary.CELL_AND_SOLUBLE_ENVIRONMENT,
            "environment_space": (
                EnvironmentVariableSpec(
                    variable=OntologyTerm(label="nutrient"),
                    units="relative",
                    required=True,
                ),
            ),
        }
    )
    environment = EnvironmentEvent(
        event_id="environment",
        subject_id="cell-1",
        time_seconds=0,
        variables={"nutrient": Quantity(value=1, units="relative")},
    )
    history = CellHistory(
        subject_id="cell-1", events=(environment, observation_factory(event_id="rna"))
    )
    belief = estimate_cell_state(
        request_factory(history=history, query=environment_query), estimator=model
    )
    scenario = EvolutionScenario(
        scenario_id="future",
        horizon_name="acute",
        subject_id="cell-1",
        start_time_seconds=10,
        end_time_seconds=70,
    )
    with pytest.raises(CapabilityError, match="inherit, clear, or replace"):
        evolve_cell_state(belief, scenario=scenario, evolution_model=model)
    inherited = scenario.model_copy(update={"inherit_current_environment": True})
    forecast = evolve_cell_state(belief, scenario=inherited, evolution_model=model)
    assert isinstance(forecast.joint_posterior, ParametricDistribution)
