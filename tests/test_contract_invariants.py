from __future__ import annotations

from collections.abc import Callable
from uuid import uuid4

import numpy as np
import pytest
from conftest import (
    SYNTHETIC_TEST_OPTIONS,
    bootstrap_interval_factory,
    environment_factory,
    environment_spec_factory,
    intervention_factory,
    intervention_spec_factory,
    observation_factory,
    query_factory,
    request_factory,
    subject_factory,
)
from pydantic import ValidationError

from cellstate import (
    ActualPerturbation,
    AssayMetadata,
    CellHistory,
    CensoringDirection,
    EstimateCellStateRequest,
    EvolutionScenario,
    HistoryCompleteness,
    InferenceOptions,
    InterventionObjective,
    MeasurementUncertainty,
    MissingnessReport,
    MissingnessStatus,
    ObjectiveDirection,
    ObjectiveTerm,
    ObservationEvent,
    OntologyTerm,
    PerturbationStatus,
    PrecisionRequirement,
    Quantity,
    RecordCompleteness,
    ReversibilityStatus,
    StateQuery,
    StaticContext,
    choose_intervention,
    estimate_cell_state,
    evolve_cell_state,
)
from cellstate.domain import (
    ArtifactRef,
    CandidateEvaluation,
    CellStateBelief,
    ContactEvent,
    DivisionEvent,
    EvidenceStatus,
    InterventionPlan,
    LineageHistory,
    PopulationContext,
    SampleDistribution,
    SpatialEdge,
    StateFactor,
    StateForecast,
    SufficiencyReport,
)
from cellstate.domain.belief import (
    BeliefStatus,
    CausalEstimandBinding,
    CausalSupportReport,
    DimensionIdentifiability,
    EvaluationStatus,
    QueryReadinessReport,
)
from cellstate.domain.common import CausalStatus, CriterionOutcome
from cellstate.domain.events import AssignmentMechanism
from cellstate.domain.scenarios import PlanStatus, TransportReport, TransportStatus
from cellstate.errors import CapabilityError, ContractViolationError, PosteriorCompatibilityError
from cellstate.ports import CapabilityReport
from cellstate.reference import (
    LinearGaussianConfig,
    LinearGaussianPlanner,
    LinearGaussianReference,
    LinearObservationConfig,
    minimal_reference_config,
)


@pytest.mark.parametrize(
    "report",
    [
        MissingnessReport.model_construct(status=MissingnessStatus.BELOW_DETECTION),
        MissingnessReport.model_construct(
            status=MissingnessStatus.CENSORED,
            censoring_direction=CensoringDirection.BELOW,
        ),
        MissingnessReport.model_construct(
            status=MissingnessStatus.CENSORED,
            censoring_direction=CensoringDirection.INTERVAL,
            interval_lower=Quantity(value=2, units="relative"),
            interval_upper=Quantity(value=1, units="relative"),
        ),
    ],
)
def test_censoring_requires_interpretable_bounds(report: MissingnessReport) -> None:
    with pytest.raises(ValidationError):
        MissingnessReport.model_validate(report.model_dump())


def test_interval_censoring_is_explicit_and_cannot_carry_an_imputed_value() -> None:
    report = MissingnessReport(
        status=MissingnessStatus.CENSORED,
        censoring_direction=CensoringDirection.INTERVAL,
        interval_lower=Quantity(value=0.1, units="relative"),
        interval_upper=Quantity(value=0.3, units="relative"),
    )
    with pytest.raises(ValidationError, match="not as an imputed value"):
        observation_factory(
            event_id="censored",
            time_seconds=0,
            modality="transcriptome",
            value=0.2,
            units="relative",
            missingness=report,
        )
    with pytest.raises(ValidationError, match="units"):
        observation_factory(
            event_id="censored",
            time_seconds=0,
            modality="transcriptome",
            value=None,
            units="kg",
            missingness=report,
        )


def test_lineage_and_spatial_contracts_reject_self_edges_and_negative_geometry() -> None:
    with pytest.raises(ValidationError, match="children"):
        DivisionEvent(
            event_id="division",
            subject=subject_factory(),
            time_seconds=1,
            child_ids=("cell-1", "child-2"),
        )
    with pytest.raises(ValidationError, match="parent"):
        CellHistory(
            subject=subject_factory(),
            lineage=LineageHistory(parent_cell_id="cell-1"),
        )
    with pytest.raises(ValidationError, match="nonnegative"):
        SpatialEdge(
            source_id="a",
            target_id="b",
            relationship="contact",
            distance=Quantity(value=-1, units="um"),
        )


def test_history_requires_realization_evidence_to_be_present() -> None:
    intervention = intervention_factory(
        event_id="edit",
        time_seconds=0,
        intervention_type="genetic edit",
        intervention_spec_id="genetic-edit",
        actual_perturbation=ActualPerturbation(
            status=PerturbationStatus.MEASURED,
            efficiency=0.8,
            evidence_event_ids=("missing-assay",),
        ),
    )
    with pytest.raises(ValidationError, match="same history"):
        CellHistory(subject=subject_factory(), events=(intervention,))


def test_query_rejects_dangling_precision_and_duplicate_members() -> None:
    query = query_factory()
    with pytest.raises(ValidationError, match="undeclared target"):
        StateQuery.model_validate(
            {
                **query.model_dump(),
                "precision_requirements": (
                    PrecisionRequirement(
                        target=OntologyTerm(label="survival"),
                        horizon_name="acute",
                        metric="absolute_error",
                        maximum_error=0.1,
                    ),
                ),
            }
        )
    with pytest.raises(ValidationError, match="undeclared horizon"):
        StateQuery.model_validate(
            {
                **query.model_dump(),
                "precision_requirements": (
                    PrecisionRequirement(
                        target=OntologyTerm(label="functional capacity"),
                        horizon_name="week",
                        metric="absolute_error",
                        maximum_error=0.1,
                    ),
                ),
            }
        )
    with pytest.raises(ValidationError, match="target outputs"):
        StateQuery.model_validate(
            {**query.model_dump(), "target_outputs": (*query.target_outputs, *query.target_outputs)}
        )
    duplicate_intervention = intervention_spec_factory()
    with pytest.raises(ValidationError, match="specification IDs"):
        StateQuery.model_validate(
            {
                **query.model_dump(),
                "intervention_space": (duplicate_intervention, duplicate_intervention),
            }
        )


def test_sufficiency_report_enforces_its_defining_identity() -> None:
    with pytest.raises(ValidationError, match="must equal"):
        SufficiencyReport(
            evaluation_status=EvaluationStatus.EVALUATED,
            outcome=CriterionOutcome.PASSED,
            state_only_loss=1,
            state_plus_history_loss=0.5,
            history_information_gain=0.2,
            history_information_gain_interval=bootstrap_interval_factory(0.2),
            markov_sufficiency_score=0.8,
            maximum_history_information_gain=0.3,
            retained_unit_fraction=1.0,
        )
    with pytest.raises(ValidationError, match="finite"):
        SufficiencyReport(
            evaluation_status=EvaluationStatus.EVALUATED,
            outcome=CriterionOutcome.PASSED,
            state_only_loss=float("inf"),
            state_plus_history_loss=0.5,
            history_information_gain=float("inf"),
            markov_sufficiency_score=0.8,
            maximum_history_information_gain=0.3,
            retained_unit_fraction=1.0,
        )


def test_belief_rejects_factor_provenance_and_marginal_contradictions(
    model: LinearGaussianReference,
) -> None:
    belief = estimate_cell_state(
        request_factory(),
        estimator=model,
        options=SYNTHETIC_TEST_OPTIONS,
    )
    evidence_payload = belief.model_dump(mode="python")
    factor = next(item for item in evidence_payload["factors"] if item["factor"] == "slow_memory")
    factor["evidence_event_ids"] = ("unknown-event",)
    with pytest.raises(ValidationError, match="provenance"):
        CellStateBelief.model_validate(evidence_payload)

    marginal_payload = belief.model_dump(mode="python")
    factor = next(item for item in marginal_payload["factors"] if item["factor"] == "slow_memory")
    factor["posterior"]["mean"] = (999.0,)
    with pytest.raises(ValidationError, match="joint marginal"):
        CellStateBelief.model_validate(marginal_payload)


def test_structural_completeness_does_not_override_scientific_readiness(
    model: LinearGaussianReference,
) -> None:
    belief = estimate_cell_state(
        request_factory(),
        estimator=model,
        options=SYNTHETIC_TEST_OPTIONS,
    )
    embedded_measurement = belief.model_dump(mode="python")
    embedded_measurement["next_measurement"] = {"status": "not_evaluated"}
    with pytest.raises(ValidationError, match="Extra inputs"):
        CellStateBelief.model_validate(embedded_measurement)

    complete_payload = belief.model_dump(mode="python")
    complete_payload["status"] = "complete"
    structurally_complete = CellStateBelief.model_validate(complete_payload)
    assert structurally_complete.status is BeliefStatus.COMPLETE
    assert structurally_complete.readiness.abstention_required
    assert not structurally_complete.readiness.valid_for_prediction
    assert not structurally_complete.readiness.valid_for_control


def _acute_scenario(**updates: object) -> EvolutionScenario:
    payload: dict[str, object] = {
        "scenario_id": "baseline",
        "horizon_name": "acute",
        "subject": subject_factory(),
        "start_time_seconds": 10,
        "end_time_seconds": 70,
    }
    payload.update(updates)
    return EvolutionScenario.model_validate(payload)


def _acute_objective(**updates: object) -> InterventionObjective:
    payload: dict[str, object] = {
        "objective_id": "objective",
        "horizon_name": "acute",
        "terms": (
            ObjectiveTerm(
                target=OntologyTerm(label="functional capacity"),
                direction=ObjectiveDirection.MAXIMIZE,
            ),
        ),
    }
    payload.update(updates)
    return InterventionObjective.model_validate(payload)


@pytest.mark.parametrize(
    ("objective_update", "scenario_update", "message"),
    [
        ({"horizon_name": "week"}, {}, "objective horizon"),
        (
            {
                "terms": (
                    ObjectiveTerm(
                        target=OntologyTerm(label="survival"),
                        direction=ObjectiveDirection.MAXIMIZE,
                    ),
                )
            },
            {},
            "undeclared query targets",
        ),
        ({}, {"horizon_name": "other"}, "objective horizon"),
        ({}, {"start_time_seconds": 9, "end_time_seconds": 69}, "belief time"),
        ({}, {"end_time_seconds": 71}, "candidate duration"),
    ],
)
def test_planning_requires_one_comparable_query_horizon(
    model: LinearGaussianReference,
    objective_update: dict[str, object],
    scenario_update: dict[str, object],
    message: str,
) -> None:
    belief = estimate_cell_state(
        request_factory(),
        estimator=model,
        options=SYNTHETIC_TEST_OPTIONS,
    )
    with pytest.raises(ContractViolationError, match=message):
        choose_intervention(
            belief,
            objective=_acute_objective(**objective_update),
            candidates=(_acute_scenario(**scenario_update),),
            planner=LinearGaussianPlanner(model),
            options=SYNTHETIC_TEST_OPTIONS,
        )


def test_planning_rejects_duplicate_candidates(model: LinearGaussianReference) -> None:
    belief = estimate_cell_state(
        request_factory(),
        estimator=model,
        options=SYNTHETIC_TEST_OPTIONS,
    )
    candidate = _acute_scenario()
    with pytest.raises(ContractViolationError, match="IDs"):
        choose_intervention(
            belief,
            objective=_acute_objective(),
            candidates=(candidate, candidate),
            planner=LinearGaussianPlanner(model),
            options=SYNTHETIC_TEST_OPTIONS,
        )


def test_forecast_contract_covers_query_target_and_horizon_exactly(
    model: LinearGaussianReference,
) -> None:
    belief = estimate_cell_state(
        request_factory(),
        estimator=model,
        options=SYNTHETIC_TEST_OPTIONS,
    )
    forecast = evolve_cell_state(
        belief,
        scenario=_acute_scenario(),
        evolution_model=model,
        options=SYNTHETIC_TEST_OPTIONS,
    )
    target_payload = forecast.model_dump(mode="python")
    target_payload["target_predictions"][0]["target"]["weight"] = 2
    with pytest.raises(ValidationError, match="specifications"):
        StateForecast.model_validate(target_payload)

    horizon_payload = forecast.model_dump(mode="python")
    horizon_payload["target_predictions"][0]["horizon_seconds"] = 30
    with pytest.raises(ValidationError, match="forecast horizon"):
        StateForecast.model_validate(horizon_payload)

    missing_payload = forecast.model_dump(mode="python")
    missing_payload["target_predictions"] = ()
    with pytest.raises(ValidationError, match="each query target"):
        StateForecast.model_validate(missing_payload)


def _different_reference_model() -> LinearGaussianReference:
    config = minimal_reference_config().model_copy(update={"drift_vector": (0.1, 0.0, 0.0, 0.0)})
    return LinearGaussianReference(config)


def test_recursive_and_evolution_paths_reject_incompatible_posteriors(
    model: LinearGaussianReference,
) -> None:
    request = request_factory()
    belief = estimate_cell_state(
        request,
        estimator=model,
        options=SYNTHETIC_TEST_OPTIONS,
    )
    other_model = _different_reference_model()
    recursive = EstimateCellStateRequest(
        query=request.query,
        history=request.history,
        as_of_seconds=request.as_of_seconds,
        static_context=request.static_context,
        previous_belief=belief,
    )
    with pytest.raises(PosteriorCompatibilityError):
        estimate_cell_state(
            recursive,
            estimator=other_model,
            options=SYNTHETIC_TEST_OPTIONS,
        )
    with pytest.raises(PosteriorCompatibilityError):
        evolve_cell_state(
            belief,
            scenario=_acute_scenario(),
            evolution_model=other_model,
            options=SYNTHETIC_TEST_OPTIONS,
        )
    with pytest.raises(PosteriorCompatibilityError):
        other_model.evolve(belief, _acute_scenario(), options=SYNTHETIC_TEST_OPTIONS)
    other_belief = estimate_cell_state(
        request,
        estimator=other_model,
        options=SYNTHETIC_TEST_OPTIONS,
    )
    assert other_belief.belief_id != belief.belief_id


@pytest.mark.parametrize(
    ("query_transform", "message"),
    [
        (
            lambda query: StateQuery.model_validate(
                {
                    **query.model_dump(),
                    "environment_space": (environment_spec_factory(units="mM", required=False),),
                }
            ),
            "mM",
        ),
        (
            lambda query: StateQuery.model_validate(
                {
                    **query.model_dump(),
                    "intervention_space": (intervention_spec_factory(dose_units="molar"),),
                }
            ),
            "molar",
        ),
    ],
)
def test_capability_preflight_checks_declared_units(
    model: LinearGaussianReference,
    query_transform: Callable[[StateQuery], StateQuery],
    message: str,
) -> None:
    query = query_transform(query_factory())
    with pytest.raises(CapabilityError, match=message):
        estimate_cell_state(
            request_factory(query=query),
            estimator=model,
            options=SYNTHETIC_TEST_OPTIONS,
        )


def test_reference_observation_model_rejects_wrong_units(
    model: LinearGaussianReference,
) -> None:
    event = observation_factory().model_copy(update={"units": "kg"})
    request = request_factory(history=CellHistory(subject=subject_factory(), events=(event,)))
    with pytest.raises(CapabilityError, match="units"):
        estimate_cell_state(
            request,
            estimator=model,
            options=SYNTHETIC_TEST_OPTIONS,
        )


@pytest.mark.parametrize(
    ("completeness", "message"),
    [
        (
            HistoryCompleteness(
                interventions=RecordCompleteness.UNKNOWN,
                environments=RecordCompleteness.COMPLETE,
                lineage=RecordCompleteness.COMPLETE,
                neighborhood=RecordCompleteness.COMPLETE,
            ),
            "intervention record",
        ),
        (
            HistoryCompleteness(
                interventions=RecordCompleteness.COMPLETE,
                environments=RecordCompleteness.INCOMPLETE,
                lineage=RecordCompleteness.COMPLETE,
                neighborhood=RecordCompleteness.COMPLETE,
            ),
            "environment record",
        ),
        (
            HistoryCompleteness(
                interventions=RecordCompleteness.COMPLETE,
                environments=RecordCompleteness.COMPLETE,
                lineage=RecordCompleteness.UNKNOWN,
                neighborhood=RecordCompleteness.COMPLETE,
            ),
            "lineage record",
        ),
        (
            HistoryCompleteness(
                interventions=RecordCompleteness.COMPLETE,
                environments=RecordCompleteness.COMPLETE,
                lineage=RecordCompleteness.COMPLETE,
                neighborhood=RecordCompleteness.INCOMPLETE,
            ),
            "neighborhood/contact record",
        ),
    ],
)
def test_reference_never_interprets_unknown_causal_history_as_no_event(
    model: LinearGaussianReference,
    completeness: HistoryCompleteness,
    message: str,
) -> None:
    base = request_factory()
    history = base.history.model_copy(update={"completeness": completeness})
    request = EstimateCellStateRequest(
        query=base.query,
        history=history,
        as_of_seconds=base.as_of_seconds,
        static_context=base.static_context,
    )
    with pytest.raises(CapabilityError, match=message):
        estimate_cell_state(
            request,
            estimator=model,
            options=SYNTHETIC_TEST_OPTIONS,
        )


def test_recursive_request_detects_changed_content_under_the_same_event_id(
    model: LinearGaussianReference,
) -> None:
    first = request_factory()
    belief = estimate_cell_state(
        first,
        estimator=model,
        options=SYNTHETIC_TEST_OPTIONS,
    )
    changed = observation_factory(event_id="obs-0", value=10)
    history = CellHistory(
        subject=subject_factory(),
        events=(changed,),
        completeness=first.history.completeness,
    )
    with pytest.raises(ValidationError, match="change previously assimilated"):
        EstimateCellStateRequest(
            query=first.query,
            history=history,
            as_of_seconds=first.as_of_seconds,
            static_context=first.static_context,
            previous_belief=belief,
        )


def test_recursive_diagnostics_retain_all_previously_assimilated_evidence(
    model: LinearGaussianReference,
) -> None:
    first = request_factory(as_of_seconds=5)
    previous = estimate_cell_state(
        first,
        estimator=model,
        options=SYNTHETIC_TEST_OPTIONS,
    )
    recursive = EstimateCellStateRequest(
        query=first.query,
        history=first.history,
        as_of_seconds=10,
        static_context=first.static_context,
        previous_belief=previous,
    )
    recursive_belief = estimate_cell_state(
        recursive,
        estimator=model,
        options=SYNTHETIC_TEST_OPTIONS,
    )
    batch_belief = estimate_cell_state(
        request_factory(as_of_seconds=10),
        estimator=model,
        options=SYNTHETIC_TEST_OPTIONS,
    )

    recursive_factors = {factor.factor: factor for factor in recursive_belief.factors}
    batch_factors = {factor.factor: factor for factor in batch_belief.factors}
    assert recursive_factors[StateFactor.SLOW_MEMORY].evidence_event_ids == ("obs-0",)
    assert recursive_factors[StateFactor.SLOW_MEMORY].evidence_status is (
        batch_factors[StateFactor.SLOW_MEMORY].evidence_status
    )
    assert recursive_belief.diagnostics.identifiability == batch_belief.diagnostics.identifiability


def test_environment_keys_are_canonical_and_same_time_conflicts_are_rejected(
    model: LinearGaussianReference,
) -> None:
    base_query = query_factory()
    environment_query = StateQuery.model_validate(
        {
            **base_query.model_dump(),
            "system_boundary": "cell_and_soluble_environment",
            "environment_space": (environment_spec_factory(),),
            "evidence_policy": base_query.evidence_policy.model_copy(
                update={"lookback_seconds": 10.0}
            ),
        }
    )
    canonical = environment_factory(
        event_id="canonical",
        time_seconds=0,
        duration_seconds=10,
        variables={"Nutrient": Quantity(value=1, units="relative")},
    )
    assert tuple(canonical.variables) == ("nutrient",)
    with pytest.raises(ValidationError, match="unique case-insensitively"):
        environment_factory(
            event_id="collision",
            time_seconds=0,
            variables={
                "Nutrient": Quantity(value=1, units="relative"),
                "nutrient": Quantity(value=1, units="relative"),
            },
        )

    first = canonical.model_copy(update={"event_id": "a"})
    second = environment_factory(
        event_id="b",
        time_seconds=0,
        duration_seconds=10,
        variables={"nutrient": Quantity(value=2, units="relative")},
    )
    history = CellHistory(subject=subject_factory(), events=(first, second))
    with pytest.raises(ValidationError, match="conflicting overlapping intervals"):
        request_factory(history=history, query=environment_query)

    regional = canonical.model_copy(update={"event_id": "regional", "spatial_region": "niche-A"})
    with pytest.raises(CapabilityError, match="spatially regional"):
        estimate_cell_state(
            request_factory(
                history=CellHistory(subject=subject_factory(), events=(regional,)),
                query=environment_query,
            ),
            estimator=model,
            options=SYNTHETIC_TEST_OPTIONS,
        )


def test_reference_rejects_context_it_does_not_condition_on(
    model: LinearGaussianReference,
) -> None:
    base = request_factory()
    donor_context = StaticContext(
        species=base.static_context.species,
        donor_id="donor-1",
    )
    donor_request = EstimateCellStateRequest(
        query=base.query,
        history=base.history,
        as_of_seconds=base.as_of_seconds,
        static_context=donor_context,
    )
    with pytest.raises(CapabilityError, match="static context"):
        estimate_cell_state(
            donor_request,
            estimator=model,
            options=SYNTHETIC_TEST_OPTIONS,
        )

    population_request = EstimateCellStateRequest(
        query=base.query,
        history=base.history,
        as_of_seconds=base.as_of_seconds,
        static_context=base.static_context,
        population_context=PopulationContext(same_sample_subject_ids=("cell-2",)),
    )
    with pytest.raises(CapabilityError, match="population context"):
        estimate_cell_state(
            population_request,
            estimator=model,
            options=SYNTHETIC_TEST_OPTIONS,
        )


@pytest.mark.parametrize(
    ("event", "message"),
    [
        (
            DivisionEvent(
                event_id="division",
                subject=subject_factory(),
                time_seconds=1,
                child_ids=("child-1", "child-2"),
            ),
            "division",
        ),
        (
            ContactEvent(
                event_id="contact",
                subject=subject_factory(),
                time_seconds=1,
                other_subject_id="cell-2",
            ),
            "contact",
        ),
    ],
)
def test_reference_rejects_unmodeled_discrete_and_contact_events(
    model: LinearGaussianReference,
    event: object,
    message: str,
) -> None:
    history = CellHistory.model_validate({"subject": subject_factory(), "events": (event,)})
    with pytest.raises(CapabilityError, match=message):
        estimate_cell_state(
            request_factory(history=history),
            estimator=model,
            options=SYNTHETIC_TEST_OPTIONS,
        )


def test_scenarios_cannot_leave_the_query_control_space(
    model: LinearGaussianReference,
) -> None:
    belief = estimate_cell_state(
        request_factory(),
        estimator=model,
        options=SYNTHETIC_TEST_OPTIONS,
    )
    cytokine = intervention_factory(
        event_id="cytokine",
        time_seconds=10,
        duration_seconds=1,
        intervention_type="cytokine",
        estimated_efficiency=None,
    )
    with pytest.raises(ContractViolationError, match="bounded action space"):
        evolve_cell_state(
            belief,
            scenario=_acute_scenario(interventions=(cytokine,)),
            evolution_model=model,
            options=SYNTHETIC_TEST_OPTIONS,
        )
    environment = environment_factory(
        event_id="environment",
        time_seconds=10,
        variables={"nutrient": Quantity(value=1, units="relative")},
    )
    with pytest.raises(ContractViolationError, match="environment space"):
        evolve_cell_state(
            belief,
            scenario=_acute_scenario(environments=(environment,)),
            evolution_model=model,
            options=SYNTHETIC_TEST_OPTIONS,
        )


def test_target_objective_units_must_match_query_output(
    model: LinearGaussianReference,
) -> None:
    belief = estimate_cell_state(
        request_factory(),
        estimator=model,
        options=SYNTHETIC_TEST_OPTIONS,
    )
    objective = InterventionObjective(
        objective_id="target",
        horizon_name="acute",
        terms=(
            ObjectiveTerm(
                target=OntologyTerm(label="functional capacity"),
                direction=ObjectiveDirection.TARGET,
                target_value=Quantity(value=0.5, units="kg"),
            ),
        ),
    )
    with pytest.raises(ContractViolationError, match="incompatible units"):
        choose_intervention(
            belief,
            objective=objective,
            candidates=(_acute_scenario(),),
            planner=LinearGaussianPlanner(model),
            options=SYNTHETIC_TEST_OPTIONS,
        )


def test_sample_posterior_artifacts_have_an_explicit_axis_convention() -> None:
    samples = ArtifactRef(
        uri="memory://samples",
        sha256="a" * 64,
        media_type="application/x-npy",
        shape=(2, 1),
        dimensions=("sample", "state_dimension"),
    )
    posterior = SampleDistribution(
        dimensions=("state",),
        samples=samples,
        sample_count=2,
    )
    assert posterior.sample_count == 2
    with pytest.raises(ValidationError, match="artifact shape"):
        SampleDistribution(
            dimensions=("state",),
            samples=samples.model_copy(update={"shape": (3, 1)}),
            sample_count=2,
        )
    with pytest.raises(ValidationError, match="axis labels"):
        ArtifactRef(
            uri="memory://bad",
            sha256="b" * 64,
            media_type="application/x-npy",
            shape=(2, 1),
            dimensions=("sample",),
        )


def test_capability_and_candidate_scores_cannot_contradict_their_labels() -> None:
    with pytest.raises(ValidationError, match="cannot declare"):
        CapabilityReport(
            supported=True,
            scope_fingerprint="a" * 64,
            unsupported_outputs=("survival",),
        )

    query = query_factory()
    target = query.target_outputs[0]
    causal_support = CausalSupportReport(
        evaluation_status=EvaluationStatus.EVALUATED,
        outcome=CriterionOutcome.PASSED,
        causal_status=CausalStatus.IDENTIFIED_POPULATION_EFFECT,
        identification_basis="randomized intervention",
        identification_design=AssignmentMechanism.RANDOMIZED,
        estimands=(
            CausalEstimandBinding(
                target=target.term,
                horizon_name="acute",
                aggregation=target.aggregation,
                intervention_spec_ids=("drug",),
                comparator="randomized control wells",
                scenario_id="candidate",
                scenario_fingerprint="d" * 64,
            ),
        ),
        evidence_ids=("trial-1",),
        evidence_fingerprints={"trial-1": "b" * 64},
        source_scope="reference population",
        target_scope="query population",
    )
    readiness = QueryReadinessReport(
        support=CriterionOutcome.PASSED,
        sufficiency=CriterionOutcome.PASSED,
        identifiability=CriterionOutcome.PASSED,
        decision_uncertainty=CriterionOutcome.PASSED,
        calibration=CriterionOutcome.PASSED,
        causal=CriterionOutcome.PASSED,
        measurement_model=CriterionOutcome.PASSED,
        control_requested=True,
        valid_for_prediction=True,
        valid_for_control=True,
        valid_for_measurement_selection=True,
        abstention_required=False,
    )
    transport = TransportReport(status=TransportStatus.WITHIN_SUPPORT)
    candidate_context = {
        "causal_status": CausalStatus.IDENTIFIED_POPULATION_EFFECT,
        "causal_support": causal_support,
        "transport": transport,
        "readiness": readiness,
    }
    with pytest.raises(ValidationError, match="selection score"):
        CandidateEvaluation(
            scenario_id="candidate",
            expected_utility=1,
            uncertainty_penalty=0.2,
            selection_score=1,
            supported=True,
            **candidate_context,
        )
    with pytest.raises(ValidationError, match="requires all utility"):
        CandidateEvaluation(
            scenario_id="candidate",
            supported=True,
            **candidate_context,
        )
    with pytest.raises(ValidationError, match="numeric utility sentinels"):
        CandidateEvaluation(
            scenario_id="candidate",
            expected_utility=0,
            uncertainty_penalty=0,
            selection_score=0,
            supported=False,
            **candidate_context,
        )


def test_realized_perturbation_evidence_must_be_an_observed_measurement() -> None:
    intervention = intervention_factory(
        event_id="edit",
        time_seconds=0,
        duration_seconds=1,
        actual_perturbation=ActualPerturbation(
            status=PerturbationStatus.MEASURED,
            efficiency=0.5,
            evidence_event_ids=("edit",),
        ),
    )
    with pytest.raises(ValidationError, match="observed measurement events"):
        CellHistory(subject=subject_factory(), events=(intervention,))

    early_measurement = observation_factory(event_id="target-engagement", time_seconds=0)
    measured_later = intervention.model_copy(
        update={
            "event_id": "later-edit",
            "time_seconds": 10,
            "actual_perturbation": ActualPerturbation(
                status=PerturbationStatus.MEASURED,
                efficiency=0.5,
                evidence_event_ids=("target-engagement",),
            ),
        }
    )
    with pytest.raises(ValidationError, match="cannot predate"):
        CellHistory(subject=subject_factory(), events=(early_measurement, measured_later))


@pytest.mark.parametrize(
    "event",
    [
        observation_factory().model_copy(
            update={"assay": AssayMetadata(assay_id="rna", batch="batch-1")}
        ),
        observation_factory().model_copy(
            update={
                "uncertainty": MeasurementUncertainty(
                    distribution="normal", parameters={"variance": 0.25}
                )
            }
        ),
    ],
)
def test_reference_rejects_measurement_details_it_cannot_condition_on(
    model: LinearGaussianReference,
    event: ObservationEvent,
) -> None:
    request = request_factory(history=CellHistory(subject=subject_factory(), events=(event,)))
    with pytest.raises(CapabilityError):
        estimate_cell_state(
            request,
            estimator=model,
            options=SYNTHETIC_TEST_OPTIONS,
        )


def test_observed_status_requires_a_direct_measurement_at_the_belief_time(
    model: LinearGaussianReference,
) -> None:
    old_event = observation_factory(modality="functional_readout", time_seconds=0, value=0.5)
    old_belief = estimate_cell_state(
        request_factory(
            history=CellHistory(subject=subject_factory(), events=(old_event,)),
            as_of_seconds=10,
        ),
        estimator=model,
        options=SYNTHETIC_TEST_OPTIONS,
    )
    old_factor = next(
        factor for factor in old_belief.factors if factor.factor is StateFactor.FUNCTIONAL_CAPACITY
    )
    assert old_factor.evidence_status is not EvidenceStatus.OBSERVED
    assert (
        old_belief.diagnostics.identifiability.dimension_status["functional_capacity"]
        is not DimensionIdentifiability.DIRECTLY_OBSERVED
    )

    current_event = observation_factory(modality="functional_readout", time_seconds=10, value=0.5)
    current_belief = estimate_cell_state(
        request_factory(
            history=CellHistory(subject=subject_factory(), events=(current_event,)),
            as_of_seconds=10,
        ),
        estimator=model,
        options=SYNTHETIC_TEST_OPTIONS,
    )
    current_factor = next(
        factor
        for factor in current_belief.factors
        if factor.factor is StateFactor.FUNCTIONAL_CAPACITY
    )
    assert current_factor.evidence_status is EvidenceStatus.OBSERVED
    assert (
        current_belief.diagnostics.identifiability.dimension_status["functional_capacity"]
        is DimensionIdentifiability.DIRECTLY_OBSERVED
    )


def test_confounded_readout_does_not_claim_identifiable_factors() -> None:
    base = minimal_reference_config()
    confounded = LinearObservationConfig(
        modality_key="transcriptome",
        units="relative",
        matrix=((1.0, 1.0, 0.0, 0.0),),
        noise_covariance=((0.25,),),
    )
    config = LinearGaussianConfig.model_validate(
        {
            **base.model_dump(mode="python"),
            "observation_models": (confounded, *base.observation_models[1:]),
        }
    )
    model = LinearGaussianReference(config)
    event = observation_factory(time_seconds=10)
    belief = estimate_cell_state(
        request_factory(
            history=CellHistory(subject=subject_factory(), events=(event,)),
            as_of_seconds=10,
        ),
        estimator=model,
        options=SYNTHETIC_TEST_OPTIONS,
    )
    factors = {factor.factor: factor for factor in belief.factors}
    assert factors[StateFactor.SLOW_MEMORY].evidence_status is EvidenceStatus.UNIDENTIFIABLE
    assert factors[StateFactor.SIGNALING].evidence_status is EvidenceStatus.UNIDENTIFIABLE
    dimension_status = belief.diagnostics.identifiability.dimension_status
    assert dimension_status["memory"] is DimensionIdentifiability.UNIDENTIFIABLE
    assert dimension_status["signaling"] is DimensionIdentifiability.UNIDENTIFIABLE


def test_reference_prior_epoch_is_invariant_to_non_evidence_records(
    model: LinearGaussianReference,
) -> None:
    empty = estimate_cell_state(
        request_factory(history=CellHistory(subject=subject_factory()), as_of_seconds=10),
        estimator=model,
        options=SYNTHETIC_TEST_OPTIONS,
    )
    not_measured = observation_factory(
        event_id="not-measured",
        time_seconds=5,
        modality="transcriptome",
        value=None,
        units="relative",
        missingness=MissingnessReport(status=MissingnessStatus.NOT_MEASURED),
    )
    with_missing_record = estimate_cell_state(
        request_factory(
            history=CellHistory(subject=subject_factory(), events=(not_measured,)),
            as_of_seconds=10,
        ),
        estimator=model,
        options=SYNTHETIC_TEST_OPTIONS,
    )
    assert np.allclose(
        empty.joint_posterior.mean,
        with_missing_record.joint_posterior.mean,
        rtol=1e-12,
        atol=1e-12,
    )
    assert np.allclose(
        empty.joint_posterior.covariance,
        with_missing_record.joint_posterior.covariance,
        rtol=1e-12,
        atol=1e-12,
    )


def test_reference_configuration_rejects_empty_factor_timescales() -> None:
    payload = minimal_reference_config().model_dump(mode="python")
    payload["factor_timescales"][StateFactor.SLOW_MEMORY] = frozenset()
    with pytest.raises(ValidationError, match="timescale assignments must be nonempty"):
        LinearGaussianConfig.model_validate(payload)


def test_reference_configuration_requires_canonical_species_keys() -> None:
    config = minimal_reference_config()
    assert "ncbitaxon:9606" in config.supported_species_keys
    payload = config.model_dump(mode="python")
    payload["supported_species_keys"] = ("NCBITaxon:9606",)
    with pytest.raises(ValidationError, match="canonical ontology identity"):
        LinearGaussianConfig.model_validate(payload)


def test_reference_preflight_rejects_target_specific_intervention_semantics(
    model: LinearGaussianReference,
) -> None:
    base = query_factory()
    query = StateQuery.model_validate(
        {
            **base.model_dump(mode="python"),
            "intervention_space": (
                intervention_spec_factory(
                    target=OntologyTerm(label="kinase-X"),
                ),
            ),
        }
    )
    with pytest.raises(CapabilityError, match="target"):
        estimate_cell_state(
            request_factory(query=query),
            estimator=model,
            options=SYNTHETIC_TEST_OPTIONS,
        )


@pytest.mark.parametrize(
    ("unsupported_field", "message"),
    [
        ({"delivery_method": "viral"}, "drug"),
        ({"reversibility_status": ReversibilityStatus.IRREVERSIBLE}, "reversibility"),
    ],
)
def test_reference_preflight_rejects_intervention_semantics_it_does_not_model(
    model: LinearGaussianReference,
    unsupported_field: dict[str, object],
    message: str,
) -> None:
    delivery = str(unsupported_field.get("delivery_method", "synthetic_reference"))
    reversibility_status = unsupported_field.get(
        "reversibility_status", ReversibilityStatus.REVERSIBLE
    )
    assert isinstance(reversibility_status, ReversibilityStatus)
    custom_query = StateQuery.model_validate(
        {
            **query_factory().model_dump(),
            "intervention_space": (
                intervention_spec_factory(
                    delivery_methods=(delivery,),
                    allowed_reversibility_statuses=(reversibility_status,),
                ),
            ),
        }
    )
    with pytest.raises(CapabilityError, match=message):
        estimate_cell_state(
            request_factory(query=custom_query),
            estimator=model,
            options=SYNTHETIC_TEST_OPTIONS,
        )


def test_forecasts_make_ongoing_intervention_persistence_explicit(
    model: LinearGaussianReference,
) -> None:
    ongoing = intervention_factory(
        event_id="ongoing-drug",
        time_seconds=0,
        duration_seconds=100,
        estimated_efficiency=1,
    )
    history = CellHistory(
        subject=subject_factory(),
        events=(observation_factory(), ongoing),
    )
    belief = estimate_cell_state(
        request_factory(history=history),
        estimator=model,
        options=SYNTHETIC_TEST_OPTIONS,
    )
    assert tuple(event.event_id for event in belief.context.active_interventions) == (
        "ongoing-drug",
    )

    with pytest.raises(ContractViolationError, match="active interventions"):
        evolve_cell_state(
            belief,
            scenario=_acute_scenario(),
            evolution_model=model,
            options=SYNTHETIC_TEST_OPTIONS,
        )

    continued = evolve_cell_state(
        belief,
        scenario=_acute_scenario(inherit_active_interventions=True),
        evolution_model=model,
        options=SYNTHETIC_TEST_OPTIONS,
    )
    cleared = evolve_cell_state(
        belief,
        scenario=_acute_scenario(
            scenario_id="cleared",
            inherit_active_interventions=False,
        ),
        evolution_model=model,
        options=SYNTHETIC_TEST_OPTIONS,
    )
    assert tuple(event.event_id for event in continued.context.active_interventions) == (
        "ongoing-drug",
    )
    assert cleared.context.active_interventions == ()
    assert continued.joint_posterior != cleared.joint_posterior


def test_reference_plan_abstains_without_candidate_scientific_support(
    model: LinearGaussianReference,
) -> None:
    belief = estimate_cell_state(
        request_factory(),
        estimator=model,
        options=SYNTHETIC_TEST_OPTIONS,
    )
    intervention = intervention_factory(
        event_id="drug",
        time_seconds=10,
        duration_seconds=10,
        estimated_efficiency=None,
    )
    baseline = _acute_scenario()
    stimulated = _acute_scenario(scenario_id="stimulated", interventions=(intervention,))
    plan = choose_intervention(
        belief,
        objective=_acute_objective(),
        candidates=(baseline, stimulated),
        planner=LinearGaussianPlanner(model),
        options=SYNTHETIC_TEST_OPTIONS,
    )
    assert plan.status is PlanStatus.ABSTAINED
    assert plan.selected_scenario_id is None
    assert plan.abstention_reasons
    assert all(not evaluation.supported for evaluation in plan.evaluations)
    payload = plan.model_dump(mode="python")
    payload["selected_scenario_id"] = "baseline"
    with pytest.raises(ValidationError, match="abstaining plan cannot select"):
        InterventionPlan.model_validate(payload)


def test_public_boundaries_reject_backend_results_not_bound_to_inputs(
    model: LinearGaussianReference,
) -> None:
    request = request_factory()

    class WrongSubjectEstimator:
        descriptor = model.descriptor
        query_compiler = model.query_compiler

        def capabilities(self, request, state_specification):
            return model.capabilities(request, state_specification)

        def estimate(self, request: EstimateCellStateRequest, *, options: InferenceOptions):
            return model.estimate(request, options=options).model_copy(
                update={"subject": subject_factory("wrong-cell")}
            )

    with pytest.raises(ContractViolationError, match="wrong typed subject"):
        estimate_cell_state(
            request,
            estimator=WrongSubjectEstimator(),
            options=SYNTHETIC_TEST_OPTIONS,
        )

    belief = estimate_cell_state(
        request,
        estimator=model,
        options=SYNTHETIC_TEST_OPTIONS,
    )

    class WrongParentEvolution:
        descriptor = model.descriptor

        def capabilities(self, belief, scenario):
            return model.capabilities(belief, scenario)

        def evolve(
            self,
            belief: CellStateBelief,
            scenario: EvolutionScenario,
            *,
            options: InferenceOptions,
        ):
            return model.evolve(belief, scenario, options=options).model_copy(
                update={"parent_belief_id": uuid4()}
            )

    with pytest.raises(ContractViolationError, match="not bound"):
        evolve_cell_state(
            belief,
            scenario=_acute_scenario(),
            evolution_model=WrongParentEvolution(),
            options=SYNTHETIC_TEST_OPTIONS,
        )

    planner = LinearGaussianPlanner(model)

    class WrongObjectivePlanner:
        descriptor = planner.descriptor

        def capabilities(self, belief, objective, candidates):
            return planner.capabilities(belief, objective, candidates)

        def choose(
            self,
            belief: CellStateBelief,
            objective: InterventionObjective,
            candidates: tuple[EvolutionScenario, ...],
            *,
            options: InferenceOptions,
        ):
            return planner.choose(belief, objective, candidates, options=options).model_copy(
                update={"objective_id": "wrong-objective"}
            )

    with pytest.raises(ContractViolationError, match="not bound"):
        choose_intervention(
            belief,
            objective=_acute_objective(),
            candidates=(_acute_scenario(),),
            planner=WrongObjectivePlanner(),
            options=SYNTHETIC_TEST_OPTIONS,
        )


def test_estimator_cannot_label_intervention_provenance_as_observed_factor_evidence(
    model: LinearGaussianReference,
) -> None:
    intervention = intervention_factory(
        event_id="historical-drug",
        time_seconds=0,
        duration_seconds=1,
        estimated_efficiency=1,
    )
    history = CellHistory(
        subject=subject_factory(),
        events=(observation_factory(), intervention),
    )
    request = request_factory(history=history)

    class WrongEvidenceEstimator:
        descriptor = model.descriptor
        query_compiler = model.query_compiler

        def capabilities(self, request, state_specification):
            return model.capabilities(request, state_specification)

        def estimate(self, request: EstimateCellStateRequest, *, options: InferenceOptions):
            belief = model.estimate(request, options=options)
            factors = tuple(
                factor.model_copy(
                    update={
                        "evidence_status": EvidenceStatus.OBSERVED,
                        "evidence_event_ids": ("historical-drug",),
                    }
                )
                if factor.factor is StateFactor.SLOW_MEMORY
                else factor
                for factor in belief.factors
            )
            return belief.model_copy(update={"factors": factors})

    with pytest.raises(ContractViolationError, match="observed factor evidence"):
        estimate_cell_state(
            request,
            estimator=WrongEvidenceEstimator(),
            options=SYNTHETIC_TEST_OPTIONS,
        )


def test_plan_and_forecast_validators_cover_derived_artifacts(
    model: LinearGaussianReference,
) -> None:
    belief = estimate_cell_state(
        request_factory(),
        estimator=model,
        options=SYNTHETIC_TEST_OPTIONS,
    )
    scenario = _acute_scenario()
    forecast = evolve_cell_state(
        belief,
        scenario=scenario,
        evolution_model=model,
        options=SYNTHETIC_TEST_OPTIONS,
    )
    forecast_payload = forecast.model_dump(mode="python")
    factor = next(item for item in forecast_payload["factors"] if item["factor"] == "slow_memory")
    factor["posterior"]["mean"] = (999.0,)
    with pytest.raises(ValidationError, match="joint marginal"):
        StateForecast.model_validate(forecast_payload)

    observed_payload = forecast.model_dump(mode="python")
    observed_factor = next(
        item for item in observed_payload["factors"] if item["factor"] == "slow_memory"
    )
    observed_factor["evidence_status"] = "observed"
    with pytest.raises(ValidationError, match="future forecast factors"):
        StateForecast.model_validate(observed_payload)

    plan = choose_intervention(
        belief,
        objective=_acute_objective(),
        candidates=(scenario,),
        planner=LinearGaussianPlanner(model),
        options=SYNTHETIC_TEST_OPTIONS,
    )
    plan_payload = plan.model_dump(mode="python")
    plan_payload["evaluations"] = ()
    with pytest.raises(ValidationError, match="every candidate"):
        InterventionPlan.model_validate(plan_payload)


def test_schema_boundary_maps_and_json_values_are_deeply_immutable(
    model: LinearGaussianReference,
) -> None:
    belief = estimate_cell_state(
        request_factory(),
        estimator=model,
        options=SYNTHETIC_TEST_OPTIONS,
    )
    source_event_id = belief.provenance.source_event_ids[0]

    with pytest.raises(TypeError, match="nested schema mappings are frozen"):
        belief.provenance.source_event_fingerprints[source_event_id] = "0" * 64
    with pytest.raises(TypeError, match="nested schema mappings are frozen"):
        belief.context.physical_environment["ghost"] = "fabricated"

    copied_context = belief.context.model_copy(
        update={
            "physical_environment": {
                "nested": {"values": ["first", "second"]},
            }
        }
    )
    nested = copied_context.physical_environment["nested"]
    assert isinstance(nested, dict)
    with pytest.raises(TypeError, match="nested schema mappings are frozen"):
        nested["extra"] = "fabricated"
    values = nested["values"]
    assert isinstance(values, list)
    with pytest.raises(TypeError, match="nested schema lists are frozen"):
        values.append("fabricated")
