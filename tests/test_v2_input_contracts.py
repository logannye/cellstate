from __future__ import annotations

import pytest
from pydantic import ValidationError

from cellstate.domain.common import OntologyTerm, Quantity
from cellstate.domain.events import (
    AssayMetadata,
    AssignmentMechanism,
    CensoringDirection,
    CollectionEffect,
    EnvironmentEvent,
    EnvironmentTemporalMode,
    EvidenceLink,
    EvidenceRole,
    InterventionEvent,
    InterventionSchedule,
    MissingnessStatus,
    ObservationCollection,
    ObservationEvent,
    PerturbationStatus,
    ReversibilityStatus,
    ScheduleKind,
    StaticContext,
)
from cellstate.domain.history import CellHistory
from cellstate.domain.query import (
    AcceptanceThresholds,
    AssayPurpose,
    AssaySpec,
    CategoricalDomain,
    EnvironmentVariableSpec,
    EvidencePolicy,
    IntegerRange,
    InterventionSpec,
    LatentQuantityEndpoint,
    MissingHistoryPolicy,
    NumericDomain,
    OutputSpec,
    PredictionHorizon,
    QueryConstraints,
    RealizationEvidenceRequirement,
    ScalarRange,
    ScheduleDomain,
    StateQuery,
    SystemBoundary,
    TargetCensoringPolicy,
    TargetCensoringSemantics,
    TargetMissingnessPolicy,
    TargetMissingnessSemantics,
    Timescale,
    VersionedReference,
)
from cellstate.domain.request import EstimateCellStateRequest, InferenceOptions
from cellstate.domain.subjects import (
    AggregationStatistic,
    BeliefSubject,
    IdentityBasis,
    SubjectKind,
    SubjectSpecification,
    TargetAggregation,
)


def _system() -> OntologyTerm:
    return OntologyTerm(label="K562 cultured cells", identifier="CVCL:0004")


def _population(*, subject_id: str = "well-a") -> BeliefSubject:
    return BeliefSubject(
        subject_id=subject_id,
        kind=SubjectKind.POPULATION,
        biological_system=_system(),
        membership_semantics="cells in one declared culture well",
        experimental_unit_kind="well",
        experimental_unit_id=subject_id,
        identity_basis=IdentityBasis.EXPERIMENTAL_UNIT,
    )


def _individual(*, subject_id: str = "cell-a") -> BeliefSubject:
    return BeliefSubject(
        subject_id=subject_id,
        kind=SubjectKind.INDIVIDUAL_CELL,
        biological_system=_system(),
        membership_semantics="one directly tracked viable cell",
        experimental_unit_kind="well",
        experimental_unit_id="well-a",
        identity_basis=IdentityBasis.DIRECT_TRACKING,
    )


def _subject_spec(subject: BeliefSubject) -> SubjectSpecification:
    return SubjectSpecification(
        kind=subject.kind,
        biological_system=subject.biological_system,
        membership_semantics=subject.membership_semantics,
        experimental_unit_kind=subject.experimental_unit_kind,
        allowed_identity_bases=(subject.identity_basis,),
    )


def _schedule_domain() -> ScheduleDomain:
    return ScheduleDomain(
        allowed_kinds=(ScheduleKind.SINGLE,),
        administration_count=IntegerRange(minimum=1, maximum=1),
        interval_seconds=None,
        washout_seconds=ScalarRange(minimum=0, maximum=3_600),
    )


def _intervention_spec() -> InterventionSpec:
    return InterventionSpec(
        spec_id="drug-a",
        kind=OntologyTerm(label="small molecule"),
        target=OntologyTerm(label="BCR-ABL"),
        mechanisms=(OntologyTerm(label="kinase inhibition"),),
        dose_domain=NumericDomain(minimum=0.1, maximum=10, units="uM"),
        duration_seconds=ScalarRange(minimum=60, maximum=3_600),
        schedule=_schedule_domain(),
        delivery_methods=("culture medium",),
        allowed_reversibility_statuses=(ReversibilityStatus.REVERSIBLE,),
        allowed_assignment_mechanisms=(AssignmentMechanism.ASSIGNED_NONRANDOM,),
        assignment_unit_kind="well",
        randomization_unit_kind=None,
        require_randomization_unit=False,
        require_matched_control=False,
        realization_evidence=RealizationEvidenceRequirement(
            allowed_statuses=tuple(PerturbationStatus),
            allowed_modalities=(OntologyTerm(label="transcriptome"),),
            minimum_evidence_events=0,
        ),
    )


def _query(subject: BeliefSubject, **updates: object) -> StateQuery:
    statistic = (
        AggregationStatistic.INDIVIDUAL
        if subject.kind is SubjectKind.INDIVIDUAL_CELL
        else AggregationStatistic.DISTRIBUTION
    )
    payload: dict[str, object] = {
        "subject": _subject_spec(subject),
        "system_boundary": (
            SystemBoundary.ISOLATED_CELL
            if subject.kind is SubjectKind.INDIVIDUAL_CELL
            else SystemBoundary.POPULATION
        ),
        "temporal_resolution_seconds": 1,
        "prediction_horizons": (
            PredictionHorizon(
                name="acute",
                duration_seconds=3_600,
                timescale=Timescale.FAST,
            ),
        ),
        "target_outputs": (
            OutputSpec(
                term=OntologyTerm(label="RNA distribution"),
                units="counts",
                aggregation=TargetAggregation(
                    subject_kind=subject.kind,
                    statistic=statistic,
                    experimental_unit=subject.experimental_unit_kind,
                ),
                endpoint=LatentQuantityEndpoint(
                    model_reference=VersionedReference(
                        reference_id="population-rna-latent-model",
                        version="test-v1",
                        fingerprint="8" * 64,
                    )
                ),
                value_schema_reference=VersionedReference(
                    reference_id="population-rna-count-vector-schema",
                    version="test-v1",
                    fingerprint="9" * 64,
                ),
                missingness=TargetMissingnessSemantics(
                    policy=TargetMissingnessPolicy.MODEL_EXPLICITLY,
                    reportable_statuses=tuple(MissingnessStatus),
                ),
                censoring=TargetCensoringSemantics(
                    policy=TargetCensoringPolicy.MODEL_WITH_RECORDED_BOUNDS,
                    allowed_directions=tuple(CensoringDirection),
                ),
                supported_horizon_names=("acute",),
                weight=1,
                functional=False,
            ),
        ),
        "intervention_space": (_intervention_spec(),),
        "environment_space": (),
        "available_assays": (),
        "evidence_policy": EvidencePolicy(
            lookback_seconds=600,
            include_at_cutoff=True,
            allowed_modalities=(OntologyTerm(label="transcriptome"),),
            allowed_evidence_roles=(
                EvidenceRole.DIRECT,
                EvidenceRole.GENERAL_POPULATION,
            ),
            minimum_observed_measurements=1,
        ),
        "acceptance_thresholds": AcceptanceThresholds(
            maximum_ood_score=0.2,
            maximum_history_information_gain=0.01,
            minimum_calibration_coverage=0.9,
            maximum_calibration_error=0.05,
            maximum_counterfactual_uncertainty=0.2,
            maximum_decision_uncertainty=0.2,
            minimum_identifiability=0.8,
        ),
        "constraints": QueryConstraints(
            maximum_intervention_combination_order=1,
            require_complete_intervention_history=False,
            require_complete_environment_history=False,
            require_complete_lineage_history=False,
            require_complete_neighborhood_history=False,
            allow_transport=False,
        ),
    }
    payload.update(updates)
    return StateQuery.model_validate(payload)


def _observation(
    *,
    target: BeliefSubject,
    source: BeliefSubject | None = None,
    event_id: str = "obs",
    time_seconds: float = 0,
    duration_seconds: float = 0,
    role: EvidenceRole = EvidenceRole.DIRECT,
    basis: IdentityBasis | None = None,
    collection: ObservationCollection | None = None,
) -> ObservationEvent:
    resolved_source = source or target
    return ObservationEvent(
        event_id=event_id,
        subject=target,
        time_seconds=time_seconds,
        duration_seconds=duration_seconds,
        modality=OntologyTerm(label="transcriptome"),
        evidence_link=EvidenceLink(
            source_subject=resolved_source,
            target_subject=target,
            role=role,
            linkage_basis=basis or resolved_source.identity_basis,
            linkage_confidence=1,
            linkage_details="recorded experimental linkage",
            sampling_unit_id=resolved_source.experimental_unit_id,
        ),
        collection=collection or ObservationCollection(effect=CollectionEffect.NONDESTRUCTIVE),
        value=1,
        units="counts",
        assay=AssayMetadata(assay_id="rna"),
    )


def _request(
    *,
    subject: BeliefSubject,
    query: StateQuery,
    events: tuple[ObservationEvent, ...],
    cutoff: float,
) -> EstimateCellStateRequest:
    return EstimateCellStateRequest(
        query=query,
        history=CellHistory(subject=subject, events=events),
        as_of_seconds=cutoff,
        static_context=StaticContext(species=OntologyTerm(label="Homo sapiens")),
    )


def test_destructive_cells_can_inform_population_without_fake_identity() -> None:
    population = _population()
    sampled_cell = _individual(subject_id="sampled-cell")
    observation = _observation(
        target=population,
        source=sampled_cell,
        role=EvidenceRole.GENERAL_POPULATION,
        basis=IdentityBasis.DECLARED_MEMBERSHIP,
        collection=ObservationCollection(
            effect=CollectionEffect.PARTIALLY_DESTRUCTIVE_POPULATION_SAMPLING,
            sampling_fraction=0.01,
        ),
    )
    request = _request(
        subject=population,
        query=_query(population),
        events=(observation,),
        cutoff=0,
    )

    assert request.history.subject.kind is SubjectKind.POPULATION
    assert observation.source_subject_id == "sampled-cell"
    assert observation.evidence_link.target_subject == population


def test_terminal_individual_collection_closes_the_entire_history() -> None:
    individual = _individual()
    terminal = _observation(
        target=individual,
        event_id="terminal",
        time_seconds=1,
        collection=ObservationCollection(effect=CollectionEffect.TERMINAL_DESTRUCTIVE),
    )
    later = _observation(target=individual, event_id="impossible", time_seconds=2)

    with pytest.raises(ValidationError, match="closes an individual-cell history"):
        CellHistory(subject=individual, events=(terminal, later))

    later_intervention = InterventionEvent(
        event_id="impossible-treatment",
        subject=individual,
        time_seconds=3,
        intervention_spec_id="drug-a",
        intervention_type=OntologyTerm(label="small molecule"),
        target=OntologyTerm(label="BCR-ABL"),
        mechanism=OntologyTerm(label="kinase inhibition"),
        dose=Quantity(value=1, units="uM"),
        duration_seconds=600,
        schedule=InterventionSchedule(
            kind=ScheduleKind.SINGLE,
            administration_count=1,
            washout_seconds=0,
        ),
        delivery_method="culture medium",
        reversibility_status=ReversibilityStatus.REVERSIBLE,
        assignment_mechanism=AssignmentMechanism.ASSIGNED_NONRANDOM,
        assignment_unit_kind="well",
        assignment_unit_id=individual.experimental_unit_id,
        matched_control=None,
    )
    with pytest.raises(ValidationError, match="later events are invalid"):
        CellHistory(subject=individual, events=(terminal, later_intervention))


def test_viability_preserving_individual_evidence_remains_longitudinal() -> None:
    individual = _individual()
    collection = ObservationCollection(
        effect=CollectionEffect.VIABILITY_PRESERVING_WITH_KNOWN_EFFECT,
        effect_description="less than one percent cytoplasm removed",
    )
    first = _observation(
        target=individual,
        event_id="live-1",
        time_seconds=1,
        collection=collection,
    )
    second = _observation(
        target=individual,
        event_id="live-2",
        time_seconds=2,
        collection=collection,
    )

    request = _request(
        subject=individual,
        query=_query(individual),
        events=(first, second),
        cutoff=2,
    )
    assert len(request.history.events) == 2


def test_inferred_lineage_membership_requires_confidence() -> None:
    with pytest.raises(ValidationError, match="requires explicit confidence"):
        BeliefSubject(
            subject_id="clone-a",
            kind=SubjectKind.CLONE_LINEAGE,
            biological_system=_system(),
            membership_semantics="probabilistic barcode cluster",
            experimental_unit_kind="clone",
            experimental_unit_id="clone-a",
            identity_basis=IdentityBasis.PROBABILISTIC_LINEAGE,
        )


def test_evidence_link_rejects_implicit_subject_casts() -> None:
    with pytest.raises(ValidationError, match="identical source and target"):
        EvidenceLink(
            source_subject=_individual(subject_id="cell-a"),
            target_subject=_individual(subject_id="cell-b"),
            role=EvidenceRole.DIRECT,
            linkage_basis=IdentityBasis.DIRECT_TRACKING,
            linkage_confidence=1,
            linkage_details="invalid identity cast",
            sampling_unit_id="well-a",
        )


def test_query_membership_checks_every_intervention_bound() -> None:
    population = _population()
    query = _query(population)
    supported = InterventionEvent(
        event_id="drug",
        subject=population,
        time_seconds=0,
        intervention_spec_id="drug-a",
        intervention_type=OntologyTerm(label="small molecule"),
        target=OntologyTerm(label="BCR-ABL"),
        mechanism=OntologyTerm(label="kinase inhibition"),
        dose=Quantity(value=1, units="uM"),
        duration_seconds=600,
        schedule=InterventionSchedule(
            kind=ScheduleKind.SINGLE,
            administration_count=1,
            washout_seconds=0,
        ),
        delivery_method="culture medium",
        reversibility_status=ReversibilityStatus.REVERSIBLE,
        assignment_mechanism=AssignmentMechanism.ASSIGNED_NONRANDOM,
        assignment_unit_kind="well",
        assignment_unit_id=population.experimental_unit_id,
        matched_control=None,
    )

    assert query.contains_intervention(supported)
    assert not query.contains_intervention(
        supported.model_copy(update={"dose": Quantity(value=100, units="uM")})
    )
    assert not query.contains_intervention(
        supported.model_copy(update={"delivery_method": "electroporation"})
    )
    assert not query.contains_intervention(
        supported.model_copy(update={"duration_seconds": 10_000})
    )


def test_environment_membership_checks_domain_duration_and_temporal_mode() -> None:
    population = _population()
    environment_spec = EnvironmentVariableSpec(
        variable=OntologyTerm(label="medium"),
        domain=CategoricalDomain(values=("RPMI", "DMEM")),
        duration_seconds=ScalarRange(minimum=0, maximum=3_600),
        required=True,
        allowed_temporal_modes=(EnvironmentTemporalMode.FIXED,),
        missing_history_policy=MissingHistoryPolicy.REJECT,
    )
    query = _query(population, environment_space=(environment_spec,))
    supported = EnvironmentEvent(
        event_id="medium",
        subject=population,
        time_seconds=0,
        variables={"medium": "RPMI"},
        duration_seconds=600,
        temporal_mode=EnvironmentTemporalMode.FIXED,
    )
    assert query.contains_environment_events((supported,))
    assert not query.contains_environment_event(
        supported.model_copy(update={"duration_seconds": 7_200})
    )
    assert not query.contains_environment_event(
        supported.model_copy(update={"variables": {"medium": "unknown"}})
    )


def test_request_enforces_observation_end_cutoff_window_role_and_modality() -> None:
    individual = _individual()
    query = _query(individual)
    crossing = _observation(
        target=individual,
        time_seconds=9,
        duration_seconds=2,
    )
    with pytest.raises(ValidationError, match="after as_of_seconds"):
        _request(subject=individual, query=query, events=(crossing,), cutoff=10)

    stale = _observation(target=individual, time_seconds=-1_000)
    with pytest.raises(ValidationError, match="outside the query evidence window"):
        _request(subject=individual, query=query, events=(stale,), cutoff=10)


def test_query_rejects_target_aggregation_unit_cast() -> None:
    population = _population()
    target = (
        _query(population)
        .target_outputs[0]
        .model_copy(
            update={
                "aggregation": TargetAggregation(
                    subject_kind=SubjectKind.POPULATION,
                    statistic=AggregationStatistic.DISTRIBUTION,
                    experimental_unit="cell",
                )
            }
        )
    )
    with pytest.raises(ValidationError, match="experimental unit"):
        _query(population, target_outputs=(target,))


def test_assay_cost_delay_and_collection_effect_are_explicitly_bounded() -> None:
    population = _population()
    constraints = _query(population).constraints.model_copy(
        update={
            "maximum_total_assay_cost": 100,
            "assay_cost_units": "USD",
            "maximum_assay_delay_seconds": 86_400,
        }
    )
    too_expensive = AssaySpec(
        assay_id="rna",
        modality=OntologyTerm(label="transcriptome"),
        protocol_reference=VersionedReference(
            reference_id="rna-protocol",
            version="test-v1",
            fingerprint="4" * 64,
        ),
        collection=ObservationCollection(
            effect=CollectionEffect.PARTIALLY_DESTRUCTIVE_POPULATION_SAMPLING,
            sampling_fraction=0.01,
        ),
        purposes=(AssayPurpose.MEASUREMENT_SELECTION,),
        cost=101,
        cost_units="USD",
        turnaround_seconds=1,
    )
    with pytest.raises(ValidationError, match="cost constraint"):
        _query(population, available_assays=(too_expensive,), constraints=constraints)


def test_v1_generic_subject_payload_is_not_silently_accepted() -> None:
    payload = _query(_population()).model_dump(mode="python")
    payload["schema_version"] = "1.0"
    payload.pop("subject")
    payload["subject_id"] = "well-a"
    with pytest.raises(ValidationError):
        StateQuery.model_validate(payload)


def test_scientific_invalidity_cannot_be_overridden_at_the_public_boundary() -> None:
    with pytest.raises(ValidationError):
        InferenceOptions(allow_scientifically_invalid_for_testing=True)
