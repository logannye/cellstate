from __future__ import annotations

from uuid import uuid4

import pytest
from conftest import intervention_spec_factory
from pydantic import ValidationError

from cellstate.domain.belief import (
    BeliefDiagnostics,
    BeliefStatus,
    CalibrationReport,
    CausalEstimandBinding,
    CausalSupportReport,
    CellStateBelief,
    ContextBelief,
    DecisionUncertaintyReport,
    DimensionIdentifiability,
    DynamicSummary,
    EvaluatedScalar,
    EvaluationStatus,
    EventHazard,
    FactorBelief,
    FateProbability,
    IdentifiabilityReport,
    InterventionRealizationBelief,
    NuisanceBelief,
    ObservabilityReport,
    QueryReadinessReport,
    SufficiencyReport,
    SupportReport,
    UncertaintyBreakdown,
    UncertaintyComponent,
    UncertaintyKind,
)
from cellstate.domain.common import (
    CausalStatus,
    CriterionOutcome,
    EvidenceStatus,
    OntologyTerm,
    ProvenanceRecord,
    Quantity,
    SupportStatus,
)
from cellstate.domain.distributions import (
    DistributionSupport,
    ParametricDistribution,
    UnavailableDistribution,
)
from cellstate.domain.events import (
    ActualPerturbation,
    AssignmentMechanism,
    EnvironmentEvent,
    EnvironmentTemporalMode,
    EvidenceRole,
    InterventionEvent,
    InterventionSchedule,
    PerturbationStatus,
    ReversibilityStatus,
    ScheduleKind,
)
from cellstate.domain.query import (
    AcceptanceThresholds,
    EvidencePolicy,
    LatentQuantityEndpoint,
    OutputSpec,
    PredictionHorizon,
    QueryConstraints,
    StateQuery,
    SystemBoundary,
    TargetCensoringPolicy,
    TargetCensoringSemantics,
    TargetMissingnessPolicy,
    TargetMissingnessSemantics,
    Timescale,
    VersionedReference,
)
from cellstate.domain.scenarios import (
    CandidateEvaluation,
    EvolutionScenario,
    InterventionObjective,
    InterventionPlan,
    ObjectiveDirection,
    ObjectiveTerm,
    PlanStatus,
    ScenarioReference,
    StateForecast,
    TargetPrediction,
    TransportReport,
    TransportStatus,
)
from cellstate.domain.specification import (
    CompiledStateSpecification,
    ExcludedStateFactor,
    StateFactor,
    StateFactorSpecification,
)
from cellstate.domain.subjects import (
    AggregationStatistic,
    BeliefSubject,
    IdentityBasis,
    SubjectKind,
    SubjectSpecification,
    TargetAggregation,
)


def _thresholds() -> AcceptanceThresholds:
    return AcceptanceThresholds(
        maximum_ood_score=0.2,
        maximum_history_information_gain=0.1,
        minimum_calibration_coverage=0.8,
        maximum_calibration_error=0.1,
        maximum_counterfactual_uncertainty=0.5,
        maximum_decision_uncertainty=1.0,
        minimum_identifiability=0.5,
    )


def _subject_specification() -> SubjectSpecification:
    return SubjectSpecification(
        kind=SubjectKind.POPULATION,
        biological_system=OntologyTerm(label="K562"),
        membership_semantics="cells in one experimental well",
        experimental_unit_kind="well",
        allowed_identity_bases=(IdentityBasis.EXPERIMENTAL_UNIT,),
    )


def _subject() -> BeliefSubject:
    return BeliefSubject(
        subject_id="well-A1",
        kind=SubjectKind.POPULATION,
        biological_system=OntologyTerm(label="K562"),
        membership_semantics="cells in one experimental well",
        experimental_unit_kind="well",
        experimental_unit_id="well-A1",
        identity_basis=IdentityBasis.EXPERIMENTAL_UNIT,
        member_ids=("cell-1", "cell-2"),
    )


def _query() -> StateQuery:
    subject = _subject_specification()
    return StateQuery(
        subject=subject,
        system_boundary=SystemBoundary.POPULATION,
        temporal_resolution_seconds=1,
        prediction_horizons=(
            PredictionHorizon(name="acute", duration_seconds=60, timescale=Timescale.FAST),
        ),
        target_outputs=(
            OutputSpec(
                term=OntologyTerm(label="functional capacity"),
                units="relative",
                aggregation=TargetAggregation(
                    subject_kind=SubjectKind.POPULATION,
                    statistic=AggregationStatistic.MEAN,
                    experimental_unit="well",
                ),
                endpoint=LatentQuantityEndpoint(
                    model_reference=VersionedReference(
                        reference_id="functional-capacity-latent-model",
                        version="test-v1",
                        fingerprint="9" * 64,
                    )
                ),
                value_schema_reference=VersionedReference(
                    reference_id="functional-capacity-value-schema",
                    version="test-v1",
                    fingerprint="8" * 64,
                ),
                missingness=TargetMissingnessSemantics(
                    policy=TargetMissingnessPolicy.NOT_APPLICABLE,
                    reportable_statuses=(),
                ),
                censoring=TargetCensoringSemantics(
                    policy=TargetCensoringPolicy.NOT_APPLICABLE,
                    allowed_directions=(),
                ),
                supported_horizon_names=("acute",),
                weight=1,
                functional=True,
            ),
        ),
        evidence_policy=EvidencePolicy(
            include_at_cutoff=True,
            allowed_modalities=(OntologyTerm(label="transcriptome"),),
            allowed_evidence_roles=(EvidenceRole.DIRECT,),
            minimum_observed_measurements=1,
        ),
        acceptance_thresholds=_thresholds(),
        constraints=QueryConstraints(
            maximum_intervention_combination_order=1,
            require_complete_intervention_history=True,
            require_complete_environment_history=True,
            require_complete_lineage_history=False,
            require_complete_neighborhood_history=False,
            allow_transport=False,
        ),
    )


def _specification(query: StateQuery) -> CompiledStateSpecification:
    active = StateFactor.FUNCTIONAL_CAPACITY
    return CompiledStateSpecification(
        query_fingerprint=query.fingerprint,
        subject=query.subject,
        compiler_id="test-compiler",
        compiler_version="2.0",
        compiler_fingerprint="a" * 64,
        active_factors=(
            StateFactorSpecification(
                factor=active,
                dimensions=("capacity",),
                timescales=frozenset({Timescale.FAST}),
                required_for_outputs=("functional_capacity",),
                rationale="direct causal ancestor of the requested output",
            ),
        ),
        excluded_factors=tuple(
            ExcludedStateFactor(factor=factor, rationale="outside this narrow query")
            for factor in StateFactor
            if factor is not active
        ),
        system_boundary=query.system_boundary,
        temporal_resolution_seconds=query.temporal_resolution_seconds,
        target_outputs=query.target_outputs,
        prediction_horizons=query.prediction_horizons,
        intervention_space=query.intervention_space,
        environment_space=query.environment_space,
        precision_requirements=query.precision_requirements,
        available_assays=query.available_assays,
        evidence_policy=query.evidence_policy,
        constraints=query.constraints,
        target_output_keys=("functional_capacity",),
        horizon_names=("acute",),
        admissible_evidence_roles=(EvidenceRole.DIRECT,),
        acceptance_thresholds=query.acceptance_thresholds,
        context_modulator_dimensions=("context",),
        intervention_realization_dimensions=("realization",),
        observation_nuisance_dimensions=("batch",),
    )


def _normal(dimensions: tuple[str, ...], mean: tuple[float, ...]) -> ParametricDistribution:
    size = len(dimensions)
    return ParametricDistribution(
        family="normal",
        dimensions=dimensions,
        mean=mean,
        covariance=tuple(
            tuple(1.0 if row == column else 0.0 for column in range(size)) for row in range(size)
        ),
    )


def _unsupported_scalar() -> EvaluatedScalar:
    return EvaluatedScalar(status=SupportStatus.UNSUPPORTED, reason="not implemented")


def _dynamics() -> DynamicSummary:
    unavailable = UnavailableDistribution(reason_code="not_implemented", message="not implemented")
    scalar = _unsupported_scalar()
    return DynamicSummary(
        velocity=unavailable,
        stability=scalar,
        division_hazard=scalar,
        death_hazard=scalar,
        bifurcation_proximity=scalar,
        recovery_timescale=scalar,
    )


def _uncertainty() -> UncertaintyBreakdown:
    return UncertaintyBreakdown(
        components=tuple(
            UncertaintyComponent(kind=kind, status=SupportStatus.UNSUPPORTED)
            for kind in UncertaintyKind
        )
    )


def _passing_diagnostics(dimensions: tuple[str, ...]) -> BeliefDiagnostics:
    return BeliefDiagnostics(
        support=SupportReport(
            evaluation_status=EvaluationStatus.EVALUATED,
            outcome=CriterionOutcome.PASSED,
            in_distribution_score=0.95,
            ood_score=0.05,
            maximum_ood_score=0.2,
            extrapolation_level="none",
            abstention_required=False,
        ),
        sufficiency=SufficiencyReport(
            evaluation_status=EvaluationStatus.EVALUATED,
            outcome=CriterionOutcome.PASSED,
            state_only_loss=1.0,
            state_plus_history_loss=0.95,
            history_information_gain=0.05,
            markov_sufficiency_score=0.95,
            maximum_history_information_gain=0.1,
            metric="negative_log_likelihood",
        ),
        identifiability=IdentifiabilityReport(
            evaluation_status=EvaluationStatus.EVALUATED,
            outcome=CriterionOutcome.PASSED,
            dimension_status={
                dimension: DimensionIdentifiability.INFERRED_WITH_SUPPORT
                for dimension in dimensions
            },
            identifiability_score=0.9,
            minimum_identifiability_score=0.5,
        ),
        decision_uncertainty=DecisionUncertaintyReport(
            evaluation_status=EvaluationStatus.EVALUATED,
            outcome=CriterionOutcome.FAILED,
            decision_uncertainty=2.0,
            maximum_decision_uncertainty=1.0,
            counterfactual_uncertainty=0.1,
            maximum_counterfactual_uncertainty=0.5,
        ),
        calibration=CalibrationReport(
            evaluation_status=EvaluationStatus.EVALUATED,
            outcome=CriterionOutcome.PASSED,
            empirical_coverage=0.9,
            minimum_coverage=0.8,
            calibration_error=0.05,
            maximum_calibration_error=0.1,
        ),
        causal_support=CausalSupportReport(
            evaluation_status=EvaluationStatus.NOT_EVALUATED,
            outcome=CriterionOutcome.NOT_EVALUATED,
            causal_status=CausalStatus.UNSUPPORTED,
        ),
    )


def _prediction_ready() -> QueryReadinessReport:
    return QueryReadinessReport(
        support=CriterionOutcome.PASSED,
        sufficiency=CriterionOutcome.PASSED,
        identifiability=CriterionOutcome.PASSED,
        decision_uncertainty=CriterionOutcome.FAILED,
        calibration=CriterionOutcome.PASSED,
        causal=CriterionOutcome.NOT_EVALUATED,
        measurement_model=CriterionOutcome.UNSUPPORTED,
        control_requested=False,
        valid_for_prediction=True,
        valid_for_control=False,
        valid_for_measurement_selection=False,
        abstention_required=False,
    )


def _provenance(
    query_fingerprint: str,
    *,
    source_event_ids: tuple[str, ...] = (),
    validation_evidence_ids: tuple[str, ...] = (),
) -> ProvenanceRecord:
    supported = bool(validation_evidence_ids)
    return ProvenanceRecord(
        model_id="test",
        model_version="2.0",
        model_fingerprint="b" * 64,
        posterior_schema_id="test/v2",
        query_fingerprint=query_fingerprint,
        history_fingerprint="c" * 64,
        history_structure_fingerprint="d" * 64,
        context_fingerprint="e" * 64,
        source_event_ids=source_event_ids,
        source_event_fingerprints={event_id: "9" * 64 for event_id in source_event_ids},
        support_envelope_id="test-envelope" if supported else None,
        support_envelope_fingerprint="6" * 64 if supported else None,
        training_support_id="test-training" if supported else None,
        training_support_fingerprint="5" * 64 if supported else None,
        validation_evidence_ids=validation_evidence_ids,
        validation_evidence_fingerprints={
            evidence_id: "8" * 64 for evidence_id in validation_evidence_ids
        },
        seed=0,
    )


def _belief() -> CellStateBelief:
    query = _query()
    specification = _specification(query)
    joint = _normal(specification.joint_dimensions, (1.0, 2.0, 3.0, 4.0))
    return CellStateBelief(
        subject=_subject(),
        as_of_seconds=10,
        query=query,
        query_fingerprint=query.fingerprint,
        history_fingerprint="c" * 64,
        context_fingerprint="e" * 64,
        state_specification=specification,
        status=BeliefStatus.COMPLETE,
        joint_posterior=joint,
        factors=(
            FactorBelief(
                factor=StateFactor.FUNCTIONAL_CAPACITY,
                timescales=frozenset({Timescale.FAST}),
                evidence_status=EvidenceStatus.INFERRED,
                posterior=_normal(("capacity",), (1.0,)),
            ),
        ),
        context=ContextBelief(
            latent_context_posterior=_normal(("context",), (2.0,)),
        ),
        intervention_realizations=(
            InterventionRealizationBelief(
                intervention_event_id="prior-drug",
                evidence_status=EvidenceStatus.INFERRED,
                posterior=_normal(("realization",), (3.0,)),
            ),
        ),
        nuisance=NuisanceBelief(posterior=_normal(("batch",), (4.0,))),
        dynamics=_dynamics(),
        uncertainty=_uncertainty(),
        diagnostics=_passing_diagnostics(specification.joint_dimensions),
        readiness=_prediction_ready(),
        provenance=_provenance(query.fingerprint, source_event_ids=("prior-drug",)),
    )


def _unsupported_diagnostics(dimensions: tuple[str, ...]) -> BeliefDiagnostics:
    return BeliefDiagnostics(
        support=SupportReport(
            evaluation_status=EvaluationStatus.UNSUPPORTED,
            outcome=CriterionOutcome.UNSUPPORTED,
            maximum_ood_score=0.2,
            unsupported_subjects=("outside validated population",),
            abstention_required=True,
        ),
        sufficiency=SufficiencyReport(
            evaluation_status=EvaluationStatus.NOT_EVALUATED,
            outcome=CriterionOutcome.NOT_EVALUATED,
            maximum_history_information_gain=0.1,
        ),
        identifiability=IdentifiabilityReport(
            evaluation_status=EvaluationStatus.NOT_EVALUATED,
            outcome=CriterionOutcome.NOT_EVALUATED,
            dimension_status={
                dimension: DimensionIdentifiability.UNIDENTIFIABLE for dimension in dimensions
            },
            minimum_identifiability_score=0.5,
        ),
        decision_uncertainty=DecisionUncertaintyReport(
            evaluation_status=EvaluationStatus.NOT_EVALUATED,
            outcome=CriterionOutcome.NOT_EVALUATED,
            maximum_decision_uncertainty=1.0,
            maximum_counterfactual_uncertainty=0.5,
        ),
        calibration=CalibrationReport(
            evaluation_status=EvaluationStatus.NOT_EVALUATED,
            outcome=CriterionOutcome.NOT_EVALUATED,
            minimum_coverage=0.8,
            maximum_calibration_error=0.1,
        ),
        causal_support=CausalSupportReport(
            evaluation_status=EvaluationStatus.UNSUPPORTED,
            outcome=CriterionOutcome.UNSUPPORTED,
            causal_status=CausalStatus.UNSUPPORTED,
            blockers=("no identified intervention effect",),
        ),
    )


def _identified_causal_support(
    *,
    scenario_id: str | None = None,
    scenario_fingerprint: str | None = None,
) -> CausalSupportReport:
    query = _query()
    target = query.target_outputs[0]
    return CausalSupportReport(
        evaluation_status=EvaluationStatus.EVALUATED,
        outcome=CriterionOutcome.PASSED,
        causal_status=CausalStatus.IDENTIFIED_POPULATION_EFFECT,
        identification_basis="randomized intervention assignment",
        identification_design=AssignmentMechanism.RANDOMIZED,
        estimands=(
            CausalEstimandBinding(
                target=target.term,
                horizon_name="acute",
                aggregation=target.aggregation,
                intervention_spec_ids=("drug",),
                comparator="matched vehicle-control wells",
                scenario_id=scenario_id,
                scenario_fingerprint=scenario_fingerprint,
            ),
        ),
        evidence_ids=("trial-1",),
        evidence_fingerprints={"trial-1": "8" * 64},
        source_scope="K562 wells in the randomized study",
        target_scope="K562 wells under the declared query",
    )


def _control_ready() -> QueryReadinessReport:
    return QueryReadinessReport(
        support=CriterionOutcome.PASSED,
        sufficiency=CriterionOutcome.PASSED,
        identifiability=CriterionOutcome.PASSED,
        decision_uncertainty=CriterionOutcome.PASSED,
        calibration=CriterionOutcome.PASSED,
        causal=CriterionOutcome.PASSED,
        measurement_model=CriterionOutcome.UNSUPPORTED,
        control_requested=True,
        valid_for_prediction=True,
        valid_for_control=True,
        valid_for_measurement_selection=False,
        abstention_required=False,
    )


def _supported_candidate(
    scenario_id: str,
    *,
    scenario_fingerprint: str,
    score: float,
) -> CandidateEvaluation:
    penalty = 0.25
    return CandidateEvaluation(
        scenario_id=scenario_id,
        expected_utility=score + penalty,
        uncertainty_penalty=penalty,
        selection_score=score,
        supported=True,
        causal_status=CausalStatus.IDENTIFIED_POPULATION_EFFECT,
        causal_support=_identified_causal_support(
            scenario_id=scenario_id,
            scenario_fingerprint=scenario_fingerprint,
        ),
        transport=TransportReport(status=TransportStatus.WITHIN_SUPPORT),
        readiness=_control_ready(),
    )


def _selected_plan() -> InterventionPlan:
    query_fingerprint = "f" * 64
    best = _supported_candidate("best", scenario_fingerprint="2" * 64, score=3.0)
    alternative = _supported_candidate(
        "alternative",
        scenario_fingerprint="3" * 64,
        score=2.0,
    )
    return InterventionPlan(
        plan_id="selected-plan",
        parent_belief_id=uuid4(),
        query_fingerprint=query_fingerprint,
        horizon_name="acute",
        objective_id="maximize-capacity",
        objective_fingerprint="1" * 64,
        candidates=(
            ScenarioReference(scenario_id="best", fingerprint="2" * 64),
            ScenarioReference(scenario_id="alternative", fingerprint="3" * 64),
        ),
        status=PlanStatus.SELECTED,
        selected_scenario_id="best",
        evaluations=(best, alternative),
        readiness=_control_ready(),
        causal_status=CausalStatus.IDENTIFIED_POPULATION_EFFECT,
        causal_support=best.causal_support,
        transport=TransportReport(status=TransportStatus.WITHIN_SUPPORT),
        rationale="The selected candidate has the highest risk-adjusted expected utility.",
        seed=0,
        provenance=_provenance(
            query_fingerprint,
            validation_evidence_ids=("trial-1",),
        ),
    )


def _forecast() -> StateForecast:
    belief = _belief()
    target = belief.query.target_outputs[0]
    return StateForecast(
        forecast_id="forecast-1",
        parent_belief_id=belief.belief_id,
        scenario_id="no-action",
        scenario_fingerprint="6" * 64,
        query=belief.query,
        query_fingerprint=belief.query_fingerprint,
        state_specification=belief.state_specification,
        horizon_name="acute",
        horizon_seconds=60,
        subject=belief.subject,
        start_time_seconds=belief.as_of_seconds,
        end_time_seconds=belief.as_of_seconds + 60,
        joint_posterior=belief.joint_posterior,
        factors=belief.factors,
        context=belief.context,
        intervention_realizations=belief.intervention_realizations,
        nuisance=belief.nuisance,
        target_predictions=(
            TargetPrediction(
                target=target,
                units=target.units,
                horizon_seconds=60,
                status=SupportStatus.SUPPORTED,
                distribution=_normal(("functional_capacity",), (1.5,)),
                causal_status=CausalStatus.UNSUPPORTED,
                transport=TransportReport(status=TransportStatus.UNSUPPORTED),
            ),
        ),
        dynamics=belief.dynamics,
        uncertainty=belief.uncertainty,
        diagnostics=belief.diagnostics,
        readiness=belief.readiness,
        causal_status=CausalStatus.UNSUPPORTED,
        transport=TransportReport(status=TransportStatus.UNSUPPORTED),
        provenance=belief.provenance,
    )


def _transported_forecast() -> StateForecast:
    base = _forecast()
    query_payload = base.query.model_dump(mode="python")
    query_payload["intervention_space"] = (
        intervention_spec_factory(
            allowed_assignment_mechanisms=(AssignmentMechanism.RANDOMIZED,),
            randomization_unit_kind="well",
            require_randomization_unit=True,
        ).model_dump(mode="python"),
    )
    query_payload["constraints"]["allow_transport"] = True
    query = StateQuery.model_validate(query_payload)
    specification = _specification(query)
    target = query.target_outputs[0]
    scenario_id = "transported-candidate"
    scenario_fingerprint = "6" * 64
    assumptions = ("conditional exchangeability",)
    causal_support = CausalSupportReport(
        evaluation_status=EvaluationStatus.EVALUATED,
        outcome=CriterionOutcome.PASSED,
        causal_status=CausalStatus.TRANSPORTED_UNDER_ASSUMPTIONS,
        identification_basis="randomized source-population trial",
        identification_design=AssignmentMechanism.RANDOMIZED,
        estimands=(
            CausalEstimandBinding(
                target=target.term,
                horizon_name="acute",
                aggregation=target.aggregation,
                intervention_spec_ids=("drug",),
                comparator="matched vehicle-control wells",
                scenario_id=scenario_id,
                scenario_fingerprint=scenario_fingerprint,
            ),
        ),
        evidence_ids=("trial-1",),
        evidence_fingerprints={"trial-1": "8" * 64},
        source_scope="audited source population",
        target_scope="audited target population",
        transport_assumptions=assumptions,
    )
    diagnostics_payload = _passing_diagnostics(specification.joint_dimensions).model_dump(
        mode="python"
    )
    diagnostics_payload["decision_uncertainty"] = DecisionUncertaintyReport(
        evaluation_status=EvaluationStatus.EVALUATED,
        outcome=CriterionOutcome.PASSED,
        decision_uncertainty=0.1,
        maximum_decision_uncertainty=1.0,
        counterfactual_uncertainty=0.1,
        maximum_counterfactual_uncertainty=0.5,
    ).model_dump(mode="python")
    diagnostics_payload["causal_support"] = causal_support.model_dump(mode="python")
    transport = TransportReport(
        status=TransportStatus.TRANSPORTED,
        source_domain=causal_support.source_scope,
        target_domain=causal_support.target_scope,
        assumptions=assumptions,
        evidence_ids=("trial-1",),
    )
    prediction = TargetPrediction(
        target=target,
        units=target.units,
        horizon_seconds=60,
        status=SupportStatus.SUPPORTED,
        distribution=base.target_predictions[0].distribution,
        causal_status=CausalStatus.TRANSPORTED_UNDER_ASSUMPTIONS,
        transport=transport,
        causal_assumptions=assumptions,
    )
    payload = base.model_dump(mode="python")
    payload.update(
        {
            "scenario_id": scenario_id,
            "scenario_fingerprint": scenario_fingerprint,
            "query": query.model_dump(mode="python"),
            "query_fingerprint": query.fingerprint,
            "state_specification": specification.model_dump(mode="python"),
            "target_predictions": (prediction.model_dump(mode="python"),),
            "diagnostics": diagnostics_payload,
            "readiness": _control_ready().model_dump(mode="python"),
            "causal_status": CausalStatus.TRANSPORTED_UNDER_ASSUMPTIONS,
            "transport": transport.model_dump(mode="python"),
            "provenance": _provenance(
                query.fingerprint,
                source_event_ids=("prior-drug",),
                validation_evidence_ids=("trial-1",),
            ).model_dump(mode="python"),
        }
    )
    return StateForecast.model_validate(payload)


def test_compiled_factors_explicitly_partition_active_and_out_of_query() -> None:
    query = _query()
    payload = _specification(query).model_dump(mode="python")
    payload["excluded_factors"] = payload["excluded_factors"][:-1]
    with pytest.raises(ValidationError, match="partition"):
        CompiledStateSpecification.model_validate(payload)


def test_belief_uses_only_active_factors_and_binds_r_and_xi_to_joint() -> None:
    belief = _belief()
    assert tuple(factor.factor for factor in belief.factors) == (StateFactor.FUNCTIONAL_CAPACITY,)
    assert belief.readiness.valid_for_prediction
    assert not belief.readiness.valid_for_control

    extra_factor = belief.factors[0].model_copy(update={"factor": StateFactor.SIGNALING})
    payload = belief.model_dump(mode="python")
    payload["factors"] = (*belief.factors, extra_factor)
    with pytest.raises(ValidationError, match="compiled active factors"):
        CellStateBelief.model_validate(payload)

    xi_payload = belief.model_dump(mode="python")
    xi_payload["nuisance"]["posterior"]["mean"] = (999.0,)
    with pytest.raises(ValidationError, match="Xi nuisance posterior"):
        CellStateBelief.model_validate(xi_payload)


def test_belief_binds_realization_and_causal_evidence_to_provenance() -> None:
    belief = _belief()

    realization_payload = belief.model_dump(mode="python")
    realization_payload["intervention_realizations"][0]["intervention_event_id"] = "ghost-action"
    with pytest.raises(ValidationError, match="interventions in provenance"):
        CellStateBelief.model_validate(realization_payload)

    causal_payload = belief.model_dump(mode="python")
    causal_payload["diagnostics"]["causal_support"] = _identified_causal_support().model_dump(
        mode="python"
    )
    causal_payload["readiness"]["causal"] = CriterionOutcome.PASSED
    with pytest.raises(ValidationError, match="causal-support evidence"):
        CellStateBelief.model_validate(causal_payload)

    causal_payload["provenance"]["validation_evidence_ids"] = ("trial-1",)
    causal_payload["provenance"]["validation_evidence_fingerprints"] = {"trial-1": "8" * 64}
    causal_payload["provenance"]["support_envelope_id"] = "test-envelope"
    causal_payload["provenance"]["support_envelope_fingerprint"] = "6" * 64
    causal_payload["provenance"]["training_support_id"] = "test-training"
    causal_payload["provenance"]["training_support_fingerprint"] = "5" * 64
    with pytest.raises(ValidationError, match="interventions absent from the query"):
        CellStateBelief.model_validate(causal_payload)

    query = _query().model_copy(
        update={
            "intervention_space": (
                intervention_spec_factory(
                    allowed_assignment_mechanisms=(AssignmentMechanism.RANDOMIZED,),
                    randomization_unit_kind="well",
                    require_randomization_unit=True,
                ),
            )
        }
    )
    specification = _specification(query)
    causal_payload["query"] = query.model_dump(mode="python")
    causal_payload["query_fingerprint"] = query.fingerprint
    causal_payload["state_specification"] = specification.model_dump(mode="python")
    causal_payload["provenance"]["query_fingerprint"] = query.fingerprint
    decision = DecisionUncertaintyReport(
        evaluation_status=EvaluationStatus.EVALUATED,
        outcome=CriterionOutcome.PASSED,
        decision_uncertainty=0.1,
        maximum_decision_uncertainty=1.0,
        counterfactual_uncertainty=0.1,
        maximum_counterfactual_uncertainty=0.5,
    )
    causal_payload["diagnostics"]["decision_uncertainty"] = decision.model_dump(mode="python")
    causal_payload["readiness"] = _control_ready().model_dump(mode="python")
    supported = CellStateBelief.model_validate(causal_payload)
    assert supported.diagnostics.causal_support.outcome is CriterionOutcome.PASSED

    unsupported_envelope = supported.model_dump(mode="python")
    unsupported_envelope["provenance"]["support_envelope_id"] = None
    unsupported_envelope["provenance"]["support_envelope_fingerprint"] = None
    unsupported_envelope["provenance"]["training_support_id"] = None
    unsupported_envelope["provenance"]["training_support_fingerprint"] = None
    with pytest.raises(ValidationError, match="support envelope and training support"):
        CellStateBelief.model_validate(unsupported_envelope)

    local_only = supported.model_dump(mode="python")
    local_only["provenance"]["validation_evidence_ids"] = ()
    local_only["provenance"]["validation_evidence_fingerprints"] = {}
    local_only["provenance"]["source_event_ids"] = ("prior-drug", "trial-1")
    local_only["provenance"]["source_event_fingerprints"] = {
        "prior-drug": "9" * 64,
        "trial-1": "8" * 64,
    }
    with pytest.raises(ValidationError, match="external validation claim artifacts"):
        CellStateBelief.model_validate(local_only)

    fingerprint_mismatch = supported.model_dump(mode="python")
    fingerprint_mismatch["diagnostics"]["causal_support"]["evidence_fingerprints"] = {
        "trial-1": "7" * 64
    }
    with pytest.raises(ValidationError, match="content-addressed validation artifacts"):
        CellStateBelief.model_validate(fingerprint_mismatch)

    wrong_target = supported.model_dump(mode="python")
    wrong_target["diagnostics"]["causal_support"]["estimands"][0]["target"] = {
        "label": "unrequested fate",
    }
    with pytest.raises(ValidationError, match="target is absent"):
        CellStateBelief.model_validate(wrong_target)

    assigned_nonrandom = supported.model_dump(mode="python")
    nonrandom_query = _query().model_copy(
        update={
            "intervention_space": (
                intervention_spec_factory(
                    allowed_assignment_mechanisms=(AssignmentMechanism.ASSIGNED_NONRANDOM,),
                    require_matched_control=True,
                ),
            )
        }
    )
    assigned_nonrandom["query"] = nonrandom_query.model_dump(mode="python")
    assigned_nonrandom["query_fingerprint"] = nonrandom_query.fingerprint
    assigned_nonrandom["state_specification"] = _specification(nonrandom_query).model_dump(
        mode="python"
    )
    assigned_nonrandom["provenance"]["query_fingerprint"] = nonrandom_query.fingerprint
    with pytest.raises(ValidationError, match="does not support the declared causal design"):
        CellStateBelief.model_validate(assigned_nonrandom)


def test_evaluation_availability_is_not_a_scientific_pass() -> None:
    with pytest.raises(ValidationError, match="availability"):
        SupportReport(
            evaluation_status=EvaluationStatus.NOT_EVALUATED,
            outcome=CriterionOutcome.PASSED,
            maximum_ood_score=0.2,
            abstention_required=False,
        )
    with pytest.raises(ValidationError, match="agree"):
        SupportReport(
            evaluation_status=EvaluationStatus.EVALUATED,
            outcome=CriterionOutcome.PASSED,
            in_distribution_score=0.1,
            ood_score=0.9,
            maximum_ood_score=0.2,
            abstention_required=False,
        )


def test_transport_and_causal_claims_require_explicit_assumptions() -> None:
    target = _query().target_outputs[0]
    with pytest.raises(ValidationError, match="assumptions"):
        TargetPrediction(
            target=target,
            units=target.units,
            horizon_seconds=60,
            status=SupportStatus.SUPPORTED,
            distribution=_normal(("functional_capacity",), (1.0,)),
            causal_status=CausalStatus.TRANSPORTED_UNDER_ASSUMPTIONS,
            transport=TransportReport(
                status=TransportStatus.TRANSPORTED,
                source_domain="study",
                target_domain="deployment",
                assumptions=("conditional exchangeability",),
                evidence_ids=("transport-study",),
            ),
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        pytest.param("top-domain", "scopes and transport domains", id="top-domain"),
        pytest.param("target-domain", "scopes and transport domains", id="target-domain"),
        pytest.param("top-evidence", "transport evidence", id="top-evidence"),
        pytest.param("target-evidence", "transport evidence", id="target-evidence"),
    ),
)
def test_transported_forecast_binds_every_transport_claim_to_scope_and_evidence(
    mutation: str,
    message: str,
) -> None:
    payload = _transported_forecast().model_dump(mode="python")
    if mutation == "top-domain":
        payload["transport"]["source_domain"] = "unrelated source population"
    elif mutation == "target-domain":
        payload["target_predictions"][0]["transport"]["target_domain"] = (
            "unrelated target population"
        )
    elif mutation == "top-evidence":
        payload["transport"]["evidence_ids"] = ()
    elif mutation == "target-evidence":
        payload["target_predictions"][0]["transport"]["evidence_ids"] = ()
    else:  # pragma: no cover - the parameter table is closed above
        raise AssertionError(f"unknown mutation {mutation}")

    with pytest.raises(ValidationError, match=message):
        StateForecast.model_validate(payload)


def test_plan_can_abstain_without_fabricating_a_selected_candidate() -> None:
    readiness = QueryReadinessReport(
        support=CriterionOutcome.UNSUPPORTED,
        sufficiency=CriterionOutcome.NOT_EVALUATED,
        identifiability=CriterionOutcome.NOT_EVALUATED,
        decision_uncertainty=CriterionOutcome.NOT_EVALUATED,
        calibration=CriterionOutcome.NOT_EVALUATED,
        causal=CriterionOutcome.UNSUPPORTED,
        measurement_model=CriterionOutcome.UNSUPPORTED,
        control_requested=True,
        valid_for_prediction=False,
        valid_for_control=False,
        valid_for_measurement_selection=False,
        abstention_required=True,
        reasons=("candidate lies outside model support",),
    )
    causal = CausalSupportReport(
        evaluation_status=EvaluationStatus.UNSUPPORTED,
        outcome=CriterionOutcome.UNSUPPORTED,
        causal_status=CausalStatus.UNSUPPORTED,
        blockers=("no identified effect",),
    )
    transport = TransportReport(status=TransportStatus.UNSUPPORTED)
    evaluation = CandidateEvaluation(
        scenario_id="candidate",
        supported=False,
        causal_status=CausalStatus.UNSUPPORTED,
        causal_support=causal,
        transport=transport,
        readiness=readiness,
        notes=("unsupported",),
    )
    query_fingerprint = "f" * 64
    plan = InterventionPlan(
        plan_id="abstain",
        parent_belief_id=uuid4(),
        query_fingerprint=query_fingerprint,
        horizon_name="acute",
        objective_id="objective",
        objective_fingerprint="1" * 64,
        candidates=(ScenarioReference(scenario_id="candidate", fingerprint="2" * 64),),
        status=PlanStatus.ABSTAINED,
        selected_scenario_id=None,
        evaluations=(evaluation,),
        readiness=readiness,
        causal_status=CausalStatus.UNSUPPORTED,
        causal_support=causal,
        transport=transport,
        abstention_reasons=("no supported candidate",),
        rationale="No action is scientifically supported.",
        seed=0,
        provenance=_provenance(query_fingerprint),
    )
    assert plan.selected_scenario_id is None

    payload = plan.model_dump(mode="python")
    payload["status"] = PlanStatus.SELECTED
    payload["selected_scenario_id"] = "candidate"
    payload["abstention_reasons"] = ()
    with pytest.raises(ValidationError):
        InterventionPlan.model_validate(payload)


def test_factor_realization_and_nuisance_evidence_cannot_overclaim_observation() -> None:
    unavailable = UnavailableDistribution(
        dimensions=("capacity",),
        reason_code="not_identified",
        message="the available evidence cannot identify this dimension",
    )
    with pytest.raises(ValidationError, match="unavailable factor posterior"):
        FactorBelief(
            factor=StateFactor.FUNCTIONAL_CAPACITY,
            timescales=frozenset({Timescale.FAST}),
            evidence_status=EvidenceStatus.INFERRED,
            posterior=unavailable,
        )
    with pytest.raises(ValidationError, match="directly observed factors"):
        FactorBelief(
            factor=StateFactor.FUNCTIONAL_CAPACITY,
            timescales=frozenset({Timescale.FAST}),
            evidence_status=EvidenceStatus.OBSERVED,
            posterior=_normal(("capacity",), (1.0,)),
        )
    with pytest.raises(ValidationError, match="observed intervention realization"):
        InterventionRealizationBelief(
            intervention_event_id="drug",
            evidence_status=EvidenceStatus.OBSERVED,
            posterior=_normal(("realization",), (1.0,)),
        )
    with pytest.raises(ValidationError, match="must be unique"):
        InterventionRealizationBelief(
            intervention_event_id="drug",
            evidence_status=EvidenceStatus.INFERRED,
            posterior=_normal(("realization",), (1.0,)),
            evidence_event_ids=("obs", "obs"),
        )
    with pytest.raises(ValidationError, match="nuisance evidence IDs must be unique"):
        NuisanceBelief(
            posterior=_normal(("batch",), (0.0,)),
            evidence_event_ids=("obs", "obs"),
        )


def test_dynamic_summaries_reject_numeric_sentinels_and_invalid_risk_values() -> None:
    with pytest.raises(ValidationError, match="supported scalar requires a value"):
        EvaluatedScalar(status=SupportStatus.SUPPORTED)
    with pytest.raises(ValidationError, match="must not use numeric sentinels"):
        EvaluatedScalar(status=SupportStatus.UNSUPPORTED, value=0.0)
    with pytest.raises(ValidationError, match="hazards must be nonnegative"):
        EventHazard(
            event="division",
            rate=EvaluatedScalar(status=SupportStatus.SUPPORTED, value=-0.1, units="1/s"),
        )
    with pytest.raises(ValidationError, match="probabilities must lie"):
        FateProbability(
            fate="death",
            horizon_seconds=60,
            probability=EvaluatedScalar(status=SupportStatus.SUPPORTED, value=1.01),
        )
    with pytest.raises(ValidationError, match="each uncertainty kind exactly once"):
        UncertaintyBreakdown(
            components=(
                UncertaintyComponent(
                    kind=UncertaintyKind.MEASUREMENT,
                    status=SupportStatus.SUPPORTED,
                    magnitude=0.1,
                ),
            )
        )


@pytest.mark.parametrize(
    ("field", "units"),
    (
        pytest.param("division_hazard", "1/s", id="division-hazard"),
        pytest.param("death_hazard", "1/s", id="death-hazard"),
        pytest.param("recovery_timescale", "s", id="recovery-timescale"),
    ),
)
def test_dynamic_summaries_reject_negative_hazards_and_timescales(field: str, units: str) -> None:
    payload = _dynamics().model_dump(mode="python")
    payload[field] = EvaluatedScalar(
        status=SupportStatus.SUPPORTED,
        value=-0.1,
        units=units,
    ).model_dump(mode="python")

    with pytest.raises(ValidationError, match=field.replace("_", " ")):
        DynamicSummary.model_validate(payload)


@pytest.mark.parametrize(
    ("support", "mean", "message"),
    (
        pytest.param(
            DistributionSupport.NONNEGATIVE,
            (-0.1,),
            "negative mean",
            id="nonnegative",
        ),
        pytest.param(
            DistributionSupport.UNIT_INTERVAL,
            (1.1,),
            "unit-interval",
            id="unit-interval",
        ),
        pytest.param(
            DistributionSupport.SIMPLEX,
            (-0.1, 1.1),
            "simplex distribution means must lie",
            id="simplex-component",
        ),
        pytest.param(
            DistributionSupport.SIMPLEX,
            (0.2, 0.2),
            "sum to one",
            id="simplex-total",
        ),
    ),
)
def test_parametric_distribution_mean_respects_declared_support(
    support: DistributionSupport,
    mean: tuple[float, ...],
    message: str,
) -> None:
    size = len(mean)
    with pytest.raises(ValidationError, match=message):
        ParametricDistribution(
            family="test",
            dimensions=tuple(f"dimension-{index}" for index in range(size)),
            support=support,
            mean=mean,
            covariance=tuple(
                tuple(1.0 if row == column else 0.0 for column in range(size))
                for row in range(size)
            ),
        )


def test_sufficiency_reports_bind_loss_improvement_to_the_declared_threshold() -> None:
    failed = SufficiencyReport(
        evaluation_status=EvaluationStatus.EVALUATED,
        outcome=CriterionOutcome.FAILED,
        state_only_loss=1.0,
        state_plus_history_loss=0.7,
        history_information_gain=0.3,
        markov_sufficiency_score=0.4,
        maximum_history_information_gain=0.1,
    )
    assert failed.outcome is CriterionOutcome.FAILED

    with pytest.raises(ValidationError, match="must equal"):
        SufficiencyReport(
            evaluation_status=EvaluationStatus.EVALUATED,
            outcome=CriterionOutcome.PASSED,
            state_only_loss=1.0,
            state_plus_history_loss=0.95,
            history_information_gain=0.01,
            markov_sufficiency_score=0.9,
            maximum_history_information_gain=0.1,
        )
    with pytest.raises(ValidationError, match="must not contain numeric sentinels"):
        SufficiencyReport(
            evaluation_status=EvaluationStatus.NOT_EVALUATED,
            outcome=CriterionOutcome.NOT_EVALUATED,
            state_only_loss=0.0,
            maximum_history_information_gain=0.1,
        )


def test_support_blockers_force_failure_and_abstention_even_when_ood_is_low() -> None:
    report = SupportReport(
        evaluation_status=EvaluationStatus.EVALUATED,
        outcome=CriterionOutcome.FAILED,
        in_distribution_score=0.99,
        ood_score=0.01,
        maximum_ood_score=0.2,
        unsupported_interventions=("unseen drug combination",),
        abstention_required=True,
    )
    assert report.abstention_required

    payload = report.model_dump(mode="python")
    payload["abstention_required"] = False
    with pytest.raises(ValidationError, match="inverse"):
        SupportReport.model_validate(payload)


def test_diagnostic_scores_cannot_disagree_with_scientific_outcomes() -> None:
    with pytest.raises(ValidationError, match="identifiability outcome"):
        IdentifiabilityReport(
            evaluation_status=EvaluationStatus.EVALUATED,
            outcome=CriterionOutcome.PASSED,
            identifiability_score=0.2,
            minimum_identifiability_score=0.5,
        )
    with pytest.raises(ValidationError, match="decision-uncertainty outcome"):
        DecisionUncertaintyReport(
            evaluation_status=EvaluationStatus.EVALUATED,
            outcome=CriterionOutcome.PASSED,
            decision_uncertainty=2.0,
            maximum_decision_uncertainty=1.0,
            counterfactual_uncertainty=0.1,
            maximum_counterfactual_uncertainty=0.5,
        )
    with pytest.raises(ValidationError, match="calibration outcome"):
        CalibrationReport(
            evaluation_status=EvaluationStatus.EVALUATED,
            outcome=CriterionOutcome.PASSED,
            empirical_coverage=0.7,
            minimum_coverage=0.8,
            calibration_error=0.05,
            maximum_calibration_error=0.1,
        )
    with pytest.raises(ValidationError, match="must be disjoint"):
        ObservabilityReport(observed=("capacity",), unidentifiable=("capacity",))


def test_causal_support_requires_auditable_identification_and_transport_evidence() -> None:
    identified = _identified_causal_support()
    with pytest.raises(ValidationError, match="identification basis and scopes"):
        CausalSupportReport(
            evaluation_status=EvaluationStatus.EVALUATED,
            outcome=CriterionOutcome.PASSED,
            causal_status=CausalStatus.IDENTIFIED_POPULATION_EFFECT,
            identification_design=identified.identification_design,
            estimands=identified.estimands,
            evidence_ids=("trial-1",),
            evidence_fingerprints={"trial-1": "8" * 64},
        )
    with pytest.raises(ValidationError, match="identification evidence"):
        CausalSupportReport(
            evaluation_status=EvaluationStatus.EVALUATED,
            outcome=CriterionOutcome.PASSED,
            causal_status=CausalStatus.IDENTIFIED_POPULATION_EFFECT,
            identification_basis="randomization",
            identification_design=identified.identification_design,
            estimands=identified.estimands,
            source_scope="study population",
            target_scope="query population",
        )
    with pytest.raises(ValidationError, match="must not claim causal identification"):
        CausalSupportReport(
            evaluation_status=EvaluationStatus.NOT_EVALUATED,
            outcome=CriterionOutcome.NOT_EVALUATED,
            causal_status=CausalStatus.IDENTIFIED_POPULATION_EFFECT,
            identification_design=identified.identification_design,
            estimands=identified.estimands,
        )
    with pytest.raises(ValidationError, match="transported causal support"):
        CausalSupportReport(
            evaluation_status=EvaluationStatus.EVALUATED,
            outcome=CriterionOutcome.PASSED,
            causal_status=CausalStatus.TRANSPORTED_UNDER_ASSUMPTIONS,
            identification_basis="randomization in source",
            identification_design=identified.identification_design,
            estimands=identified.estimands,
            evidence_ids=("trial-1",),
            evidence_fingerprints={"trial-1": "8" * 64},
            source_scope="source population",
            target_scope="target population",
        )


def test_readiness_flags_are_independent_and_derived_fail_closed() -> None:
    report = QueryReadinessReport(
        support=CriterionOutcome.PASSED,
        sufficiency=CriterionOutcome.FAILED,
        identifiability=CriterionOutcome.PASSED,
        decision_uncertainty=CriterionOutcome.NOT_EVALUATED,
        calibration=CriterionOutcome.PASSED,
        causal=CriterionOutcome.NOT_EVALUATED,
        measurement_model=CriterionOutcome.PASSED,
        control_requested=False,
        valid_for_prediction=False,
        valid_for_control=False,
        valid_for_measurement_selection=True,
        abstention_required=True,
        reasons=("raw history still adds predictive information",),
    )
    assert report.valid_for_measurement_selection
    assert not report.valid_for_prediction

    payload = report.model_dump(mode="python")
    payload["valid_for_measurement_selection"] = False
    with pytest.raises(ValidationError, match="measurement-readiness"):
        QueryReadinessReport.model_validate(payload)

    payload = report.model_dump(mode="python")
    payload["reasons"] = ()
    with pytest.raises(ValidationError, match="requires explicit reasons"):
        QueryReadinessReport.model_validate(payload)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        pytest.param("subject", "belief subject", id="typed-subject"),
        pytest.param("compiled-query", "compiled query semantics", id="compiled-query-binding"),
        pytest.param("compiled-target", "compiled target keys", id="target-binding"),
        pytest.param("compiled-horizon", "compiled horizon names", id="horizon-binding"),
        pytest.param("provenance-history", "provenance/history", id="history-binding"),
        pytest.param("joint-dimension", "joint posterior dimensions", id="joint-state"),
        pytest.param("factor-dimension", "factor posterior dimensions", id="factor-state"),
        pytest.param("factor-timescale", "factor timescales", id="factor-timescale"),
        pytest.param("factor-evidence", "factor evidence IDs", id="factor-evidence"),
        pytest.param("context", "context posterior dimensions", id="context-state"),
        pytest.param("realization", "partition the compiled realization", id="realization-state"),
        pytest.param("nuisance-evidence", "nuisance evidence IDs", id="nuisance-evidence"),
        pytest.param("identifiability", "classify every active", id="identifiability-coverage"),
        pytest.param("readiness", "readiness/support", id="readiness-binding"),
        pytest.param("threshold", "query OOD threshold", id="threshold-binding"),
    ),
)
def test_belief_rejects_unbound_or_internally_inconsistent_state_blocks(
    mutation: str,
    message: str,
) -> None:
    payload = _belief().model_dump(mode="python")
    if mutation == "subject":
        payload["subject"]["membership_semantics"] = "a different experimental population"
    elif mutation == "compiled-query":
        payload["state_specification"]["query_fingerprint"] = "0" * 64
    elif mutation == "compiled-target":
        payload["state_specification"]["target_output_keys"] = ("different_target",)
    elif mutation == "compiled-horizon":
        payload["state_specification"]["horizon_names"] = ("different_horizon",)
    elif mutation == "provenance-history":
        payload["provenance"]["history_fingerprint"] = "0" * 64
    elif mutation == "joint-dimension":
        payload["joint_posterior"]["dimensions"] = (
            "capacity",
            "context",
            "realization",
            "wrong_nuisance",
        )
    elif mutation == "factor-dimension":
        payload["factors"][0]["posterior"]["dimensions"] = ("wrong_capacity",)
    elif mutation == "factor-timescale":
        payload["factors"][0]["timescales"] = frozenset({Timescale.SLOW})
    elif mutation == "factor-evidence":
        payload["factors"][0]["evidence_event_ids"] = ("unrecorded-observation",)
    elif mutation == "context":
        payload["context"]["latent_context_posterior"]["dimensions"] = ("wrong_context",)
    elif mutation == "realization":
        payload["intervention_realizations"] = ()
    elif mutation == "nuisance-evidence":
        payload["nuisance"]["evidence_event_ids"] = ("unrecorded-observation",)
    elif mutation == "identifiability":
        payload["diagnostics"]["identifiability"]["dimension_status"].pop("batch")
    elif mutation == "readiness":
        payload["readiness"]["support"] = CriterionOutcome.FAILED
        payload["readiness"]["valid_for_prediction"] = False
        payload["readiness"]["abstention_required"] = True
        payload["readiness"]["reasons"] = ("support failed",)
    elif mutation == "threshold":
        payload["diagnostics"]["support"]["maximum_ood_score"] = 0.3
    else:  # pragma: no cover - the parameter table is closed above
        raise AssertionError(f"unknown mutation {mutation}")

    with pytest.raises(ValidationError, match=message):
        CellStateBelief.model_validate(payload)


def test_belief_rejects_marginals_that_disagree_with_the_authoritative_joint() -> None:
    payload = _belief().model_dump(mode="python")
    payload["factors"][0]["posterior"]["mean"] = (99.0,)
    with pytest.raises(ValidationError, match="parametric functional_capacity posterior"):
        CellStateBelief.model_validate(payload)

    payload = _belief().model_dump(mode="python")
    payload["intervention_realizations"][0]["posterior"]["covariance"] = ((2.0,),)
    with pytest.raises(ValidationError, match="intervention realization"):
        CellStateBelief.model_validate(payload)


@pytest.mark.parametrize("artifact_kind", ("belief", "forecast"))
@pytest.mark.parametrize("component", ("mean", "covariance"))
def test_marginal_consistency_uses_scale_independent_tolerance(
    artifact_kind: str,
    component: str,
) -> None:
    artifact = _belief() if artifact_kind == "belief" else _forecast()
    payload = artifact.model_dump(mode="python")
    contract = CellStateBelief if artifact_kind == "belief" else StateForecast
    message = (
        "parametric functional_capacity posterior"
        if artifact_kind == "belief"
        else "forecast joint marginal"
    )

    if component == "mean":
        joint_mean = list(payload["joint_posterior"]["mean"])
        joint_mean[0] = 1_000_000_000.0
        payload["joint_posterior"]["mean"] = tuple(joint_mean)
        payload["factors"][0]["posterior"]["mean"] = (1_000_001_000.0,)
    else:
        joint_covariance = [list(row) for row in payload["joint_posterior"]["covariance"]]
        joint_covariance[0][0] = 1_000_000_000.0
        payload["joint_posterior"]["covariance"] = tuple(tuple(row) for row in joint_covariance)
        payload["factors"][0]["posterior"]["covariance"] = ((1_000_001_000.0,),)

    with pytest.raises(ValidationError, match=message):
        contract.model_validate(payload)


def test_structural_belief_status_does_not_smuggle_unavailable_posteriors() -> None:
    payload = _belief().model_dump(mode="python")
    payload["status"] = BeliefStatus.UNAVAILABLE
    with pytest.raises(ValidationError, match="requires an unavailable joint"):
        CellStateBelief.model_validate(payload)

    payload = _belief().model_dump(mode="python")
    payload["factors"][0]["evidence_status"] = EvidenceStatus.UNIDENTIFIABLE
    payload["factors"][0]["posterior"] = UnavailableDistribution(
        dimensions=("capacity",),
        reason_code="not_identified",
        message="capacity was not identified",
    ).model_dump(mode="python")
    with pytest.raises(ValidationError, match="complete belief requires every active posterior"):
        CellStateBelief.model_validate(payload)


def test_evolution_scenario_forbids_retrospective_information_about_future_actions() -> None:
    subject = _subject()
    intervention = InterventionEvent(
        event_id="planned-drug",
        subject=subject,
        time_seconds=20,
        intervention_spec_id="drug",
        intervention_type=OntologyTerm(label="drug"),
        dose=Quantity(value=1, units="relative"),
        duration_seconds=10,
        schedule=InterventionSchedule(
            kind=ScheduleKind.SINGLE,
            administration_count=1,
            washout_seconds=0,
        ),
        delivery_method="media",
        estimated_efficiency=None,
        reversibility_status=ReversibilityStatus.REVERSIBLE,
        assignment_mechanism=AssignmentMechanism.ASSIGNED_NONRANDOM,
        assignment_unit_kind="well",
        assignment_unit_id=subject.experimental_unit_id,
        matched_control=None,
    )
    environment = EnvironmentEvent(
        event_id="planned-environment",
        subject=subject,
        time_seconds=30,
        variables={"oxygen": Quantity(value=5, units="percent")},
        duration_seconds=10,
        temporal_mode=EnvironmentTemporalMode.FIXED,
    )
    scenario = EvolutionScenario(
        scenario_id="candidate",
        horizon_name="acute",
        subject=subject,
        start_time_seconds=10,
        end_time_seconds=70,
        interventions=(intervention,),
        environments=(environment,),
        inherit_active_interventions=False,
        inherit_current_environment=False,
    )
    assert scenario.subject_id == subject.subject_id

    payload = scenario.model_dump(mode="python")
    payload["interventions"][0]["estimated_efficiency"] = 0.9
    with pytest.raises(ValidationError, match="realized future efficiency"):
        EvolutionScenario.model_validate(payload)

    payload = scenario.model_dump(mode="python")
    payload["interventions"][0]["actual_perturbation"] = ActualPerturbation(
        status=PerturbationStatus.MEASURED,
        efficiency=0.9,
        evidence_event_ids=("future-observation",),
    ).model_dump(mode="python")
    with pytest.raises(ValidationError, match="retrospective realization evidence"):
        EvolutionScenario.model_validate(payload)

    payload = scenario.model_dump(mode="python")
    payload["environments"][0]["time_seconds"] = 65
    with pytest.raises(ValidationError, match="outside the scenario interval"):
        EvolutionScenario.model_validate(payload)


def test_objectives_and_transport_reports_are_explicit_and_unambiguous() -> None:
    target = OntologyTerm(label="functional capacity")
    with pytest.raises(ValidationError, match="target objectives require"):
        ObjectiveTerm(target=target, direction=ObjectiveDirection.TARGET)

    term = ObjectiveTerm(target=target, direction=ObjectiveDirection.MAXIMIZE)
    with pytest.raises(ValidationError, match="each target only once"):
        InterventionObjective(
            objective_id="duplicate-target",
            horizon_name="acute",
            terms=(term, term),
        )

    with pytest.raises(ValidationError, match="source, target, and assumptions"):
        TransportReport(status=TransportStatus.TRANSPORTED, source_domain="study")
    with pytest.raises(ValidationError, match="transport evidence"):
        TransportReport(
            status=TransportStatus.TRANSPORTED,
            source_domain="study",
            target_domain="deployment",
            assumptions=("conditional exchangeability",),
        )
    with pytest.raises(ValidationError, match="must not claim transport assumptions"):
        TransportReport(
            status=TransportStatus.WITHIN_SUPPORT,
            assumptions=("conditional exchangeability",),
        )
    with pytest.raises(ValidationError, match="evidence IDs must be unique"):
        TransportReport(
            status=TransportStatus.TRANSPORTED,
            source_domain="study",
            target_domain="deployment",
            assumptions=("conditional exchangeability",),
            evidence_ids=("study-1", "study-1"),
        )


def test_target_prediction_support_causal_status_and_transport_must_cohere() -> None:
    target = _query().target_outputs[0]
    unavailable = UnavailableDistribution(reason_code="unsupported", message="outside support")
    with pytest.raises(ValidationError, match="units must match"):
        TargetPrediction(
            target=target,
            units="wrong-unit",
            horizon_seconds=60,
            status=SupportStatus.UNSUPPORTED,
            distribution=unavailable,
            causal_status=CausalStatus.UNSUPPORTED,
            transport=TransportReport(status=TransportStatus.UNSUPPORTED),
        )
    with pytest.raises(ValidationError, match="supported target prediction"):
        TargetPrediction(
            target=target,
            units=target.units,
            horizon_seconds=60,
            status=SupportStatus.SUPPORTED,
            distribution=unavailable,
            causal_status=CausalStatus.UNSUPPORTED,
            transport=TransportReport(status=TransportStatus.UNSUPPORTED),
        )
    with pytest.raises(ValidationError, match="requires a transported result"):
        TargetPrediction(
            target=target,
            units=target.units,
            horizon_seconds=60,
            status=SupportStatus.SUPPORTED,
            distribution=_normal(("functional_capacity",), (1.0,)),
            causal_status=CausalStatus.TRANSPORTED_UNDER_ASSUMPTIONS,
            causal_assumptions=("conditional exchangeability",),
            transport=TransportReport(status=TransportStatus.WITHIN_SUPPORT),
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        pytest.param("interval", "horizon must equal", id="interval"),
        pytest.param("named-horizon", "duration does not match", id="named-horizon"),
        pytest.param("target-count", "each query target exactly once", id="target-count"),
        pytest.param(
            "unsupported-target-horizon",
            "when supported at this horizon",
            id="unsupported-target-horizon",
        ),
        pytest.param("target-horizon", "use the forecast horizon", id="target-horizon"),
        pytest.param(
            "target-causal-branch",
            "stronger than or differ from the forecast branch",
            id="target-causal-branch",
        ),
        pytest.param("observed-factor", "cannot be directly observed", id="future-observation"),
        pytest.param("factor-marginal", "forecast joint marginal", id="joint-marginal"),
        pytest.param(
            "duplicate-realization",
            "realization blocks must name unique interventions",
            id="duplicate-realization",
        ),
        pytest.param(
            "ghost-realization",
            "realization blocks must reference provenance events",
            id="ghost-realization",
        ),
        pytest.param(
            "nuisance-provenance",
            "forecast nuisance evidence must appear in provenance",
            id="nuisance-provenance",
        ),
        pytest.param("causal-status", "must match its causal-support", id="causal-status"),
        pytest.param("support-readiness", "support/readiness", id="support-readiness"),
        pytest.param("threshold", "query OOD threshold", id="threshold"),
    ),
)
def test_forecast_is_fully_bound_to_query_state_horizon_and_diagnostics(
    mutation: str,
    message: str,
) -> None:
    payload = _forecast().model_dump(mode="python")
    if mutation == "interval":
        payload["end_time_seconds"] = 80
    elif mutation == "named-horizon":
        payload["horizon_seconds"] = 30
        payload["end_time_seconds"] = 40
        payload["target_predictions"][0]["horizon_seconds"] = 30
    elif mutation == "target-count":
        payload["target_predictions"] = ()
    elif mutation == "unsupported-target-horizon":
        query_payload = payload["query"]
        late_horizon = dict(query_payload["prediction_horizons"][0])
        late_horizon.update({"name": "late", "duration_seconds": 120.0})
        query_payload["prediction_horizons"] = (
            *query_payload["prediction_horizons"],
            late_horizon,
        )
        query_payload["target_outputs"][0]["supported_horizon_names"] = ("late",)
        mutated_query = StateQuery.model_validate(query_payload)
        payload["query"] = mutated_query.model_dump(mode="python")
        payload["query_fingerprint"] = mutated_query.fingerprint
        payload["state_specification"]["query_fingerprint"] = mutated_query.fingerprint
        payload["state_specification"]["prediction_horizons"] = mutated_query.prediction_horizons
        payload["state_specification"]["target_outputs"] = mutated_query.target_outputs
        payload["state_specification"]["horizon_names"] = ("acute", "late")
        payload["provenance"]["query_fingerprint"] = mutated_query.fingerprint
    elif mutation == "target-horizon":
        payload["target_predictions"][0]["horizon_seconds"] = 30
    elif mutation == "target-causal-branch":
        payload["target_predictions"][0]["causal_status"] = CausalStatus.PREDICTIVE_ASSOCIATION
        payload["target_predictions"][0]["transport"] = TransportReport(
            status=TransportStatus.WITHIN_SUPPORT
        ).model_dump(mode="python")
    elif mutation == "observed-factor":
        payload["factors"][0]["evidence_status"] = EvidenceStatus.OBSERVED
        payload["factors"][0]["evidence_event_ids"] = ("future-observation",)
        payload["provenance"]["source_event_ids"] = ("future-observation",)
        payload["provenance"]["source_event_fingerprints"] = {"future-observation": "9" * 64}
    elif mutation == "factor-marginal":
        payload["factors"][0]["posterior"]["mean"] = (99.0,)
    elif mutation == "duplicate-realization":
        payload["intervention_realizations"] = (
            payload["intervention_realizations"][0],
            payload["intervention_realizations"][0],
        )
    elif mutation == "ghost-realization":
        payload["intervention_realizations"][0]["intervention_event_id"] = "ghost-action"
    elif mutation == "nuisance-provenance":
        payload["nuisance"]["evidence_event_ids"] = ("ghost-observation",)
    elif mutation == "causal-status":
        payload["causal_status"] = CausalStatus.PREDICTIVE_ASSOCIATION
    elif mutation == "support-readiness":
        payload["readiness"]["support"] = CriterionOutcome.FAILED
        payload["readiness"]["valid_for_prediction"] = False
        payload["readiness"]["abstention_required"] = True
        payload["readiness"]["reasons"] = ("support failed",)
    elif mutation == "threshold":
        payload["diagnostics"]["support"]["maximum_ood_score"] = 0.3
    else:  # pragma: no cover - the parameter table is closed above
        raise AssertionError(f"unknown mutation {mutation}")

    with pytest.raises(ValidationError, match=message):
        StateForecast.model_validate(payload)


def test_candidate_evaluation_requires_causal_control_readiness_and_real_score_math() -> None:
    candidate = _supported_candidate(
        "candidate",
        scenario_fingerprint="4" * 64,
        score=2.0,
    )
    assert candidate.selection_score == 2.0

    payload = candidate.model_dump(mode="python")
    payload["selection_score"] = 999.0
    with pytest.raises(ValidationError, match="expected utility minus uncertainty penalty"):
        CandidateEvaluation.model_validate(payload)

    payload = candidate.model_dump(mode="python")
    payload["supported"] = False
    with pytest.raises(ValidationError, match="must not contain numeric utility sentinels"):
        CandidateEvaluation.model_validate(payload)

    payload = candidate.model_dump(mode="python")
    payload["readiness"]["causal"] = CriterionOutcome.FAILED
    payload["readiness"]["valid_for_control"] = False
    payload["readiness"]["abstention_required"] = True
    payload["readiness"]["reasons"] = ("causal validation failed",)
    with pytest.raises(ValidationError, match="causal support must match readiness"):
        CandidateEvaluation.model_validate(payload)

    payload = candidate.model_dump(mode="python")
    payload["transport"] = TransportReport(status=TransportStatus.NOT_EVALUATED).model_dump(
        mode="python"
    )
    with pytest.raises(ValidationError, match="requires within-support transport"):
        CandidateEvaluation.model_validate(payload)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        pytest.param(
            "duplicate-candidate", "candidate scenario IDs must be unique", id="duplicate"
        ),
        pytest.param("missing-evaluation", "evaluate every candidate", id="evaluation-coverage"),
        pytest.param("abstain-and-select", "abstaining plan cannot select", id="abstain-select"),
        pytest.param(
            "selected-with-reasons", "cannot also report abstention", id="selected-reasons"
        ),
        pytest.param("unsupported-selection", "supported candidate evaluation", id="unsupported"),
        pytest.param(
            "selected-causal-status",
            "causal status must match its candidate evaluation",
            id="selected-causal-status",
        ),
        pytest.param(
            "selected-transport-status",
            "requires within-support transport",
            id="selected-transport-status",
        ),
        pytest.param(
            "selected-transport-details",
            "transport must equal its candidate evaluation",
            id="selected-transport-details",
        ),
        pytest.param(
            "transport-provenance",
            "transport evidence must appear in provenance",
            id="transport-provenance",
        ),
        pytest.param("not-best", "highest selection score", id="argmax"),
        pytest.param("provenance", "provenance/query", id="provenance"),
    ),
)
def test_selected_plan_cannot_bypass_candidate_coverage_abstention_or_argmax(
    mutation: str,
    message: str,
) -> None:
    payload = _selected_plan().model_dump(mode="python")
    if mutation == "duplicate-candidate":
        payload["candidates"][1]["scenario_id"] = "best"
    elif mutation == "missing-evaluation":
        payload["evaluations"] = payload["evaluations"][:-1]
    elif mutation == "abstain-and-select":
        payload["status"] = PlanStatus.ABSTAINED
        payload["abstention_reasons"] = ("forced abstention",)
    elif mutation == "selected-with-reasons":
        payload["abstention_reasons"] = ("contradictory reason",)
    elif mutation == "unsupported-selection":
        payload["evaluations"][0]["supported"] = False
        payload["evaluations"][0]["expected_utility"] = None
        payload["evaluations"][0]["uncertainty_penalty"] = None
        payload["evaluations"][0]["selection_score"] = None
    elif mutation == "selected-causal-status":
        assumptions = ("conditional exchangeability",)
        payload["evaluations"][0]["causal_status"] = CausalStatus.TRANSPORTED_UNDER_ASSUMPTIONS
        payload["evaluations"][0]["causal_support"]["causal_status"] = (
            CausalStatus.TRANSPORTED_UNDER_ASSUMPTIONS
        )
        payload["evaluations"][0]["causal_support"]["transport_assumptions"] = assumptions
        payload["evaluations"][0]["transport"] = TransportReport(
            status=TransportStatus.TRANSPORTED,
            source_domain="source study",
            target_domain="query population",
            assumptions=assumptions,
            evidence_ids=("trial-1",),
        ).model_dump(mode="python")
    elif mutation == "selected-transport-status":
        payload["transport"]["status"] = TransportStatus.NOT_EVALUATED
    elif mutation == "selected-transport-details":
        payload["transport"]["source_domain"] = "fabricated source domain"
    elif mutation == "transport-provenance":
        payload["transport"]["evidence_ids"] = ("ghost-transport-evidence",)
        payload["evaluations"][0]["transport"]["evidence_ids"] = ("ghost-transport-evidence",)
    elif mutation == "not-best":
        payload["selected_scenario_id"] = "alternative"
        payload["causal_support"] = payload["evaluations"][1]["causal_support"]
    elif mutation == "provenance":
        payload["provenance"]["query_fingerprint"] = "0" * 64
    else:  # pragma: no cover - the parameter table is closed above
        raise AssertionError(f"unknown mutation {mutation}")

    with pytest.raises(ValidationError, match=message):
        InterventionPlan.model_validate(payload)
