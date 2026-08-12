from __future__ import annotations

import pytest

from cellstate import (
    CellHistory,
    EstimateCellStateRequest,
    HistoryCompleteness,
    InferenceOptions,
    OntologyTerm,
    OutputSpec,
    PredictionHorizon,
    Quantity,
    RecordCompleteness,
    StateQuery,
    StaticContext,
    SystemBoundary,
    Timescale,
)
from cellstate.domain.common import BootstrapInterval
from cellstate.domain.events import (
    ActualPerturbation,
    AssayMetadata,
    AssignmentMechanism,
    CollectionEffect,
    EnvironmentEvent,
    EnvironmentTemporalMode,
    EvidenceLink,
    EvidenceRole,
    InterventionEvent,
    InterventionSchedule,
    MatchedControl,
    MeasurementUncertainty,
    MissingnessReport,
    ObservationCollection,
    ObservationEvent,
    PerturbationStatus,
    QualityReport,
    ReversibilityStatus,
    ScheduleKind,
)
from cellstate.domain.query import (
    AcceptanceThresholds,
    AssayPurpose,
    AssaySpec,
    EnvironmentVariableSpec,
    EvidencePolicy,
    IntegerRange,
    InterventionSpec,
    LatentQuantityEndpoint,
    NumericDomain,
    QueryConstraints,
    RealizationEvidenceRequirement,
    ScalarRange,
    ScheduleDomain,
    TargetCensoringPolicy,
    TargetCensoringSemantics,
    TargetMissingnessPolicy,
    TargetMissingnessSemantics,
    VersionedReference,
)
from cellstate.domain.subjects import (
    AggregationStatistic,
    BeliefSubject,
    IdentityBasis,
    SubjectKind,
    SubjectSpecification,
    TargetAggregation,
)
from cellstate.reference import LinearGaussianReference, minimal_reference_config

SYNTHETIC_TEST_OPTIONS = InferenceOptions()


def bootstrap_interval_factory(
    point_estimate: float = 0.0,
    *,
    half_width: float = 1.0,
    cluster_counts: tuple[int, ...] = (12, 6),
) -> BootstrapInterval:
    """A structurally valid interval for contract tests.

    Its numbers are not a measurement.  Tests that care about the interval's statistical behavior
    use the estimator itself; this exists so contract tests can construct an evaluated report
    without reimplementing the bootstrap.
    """

    return BootstrapInterval(
        point_estimate=point_estimate,
        lower=point_estimate - half_width,
        upper=point_estimate + half_width,
        percentile_lower=point_estimate - half_width / 2,
        percentile_upper=point_estimate + half_width / 2,
        small_cluster_scale=2.0,
        standard_error=half_width / 2,
        confidence_level=0.95,
        resample_count=2000,
        evaluation_unit_count=48,
        resampling_scheme="multiway_clustered",
        interval_method="equal_tailed_percentile_small_cluster_scaled",
        dependence_dimension_ids=("compound", "plate"),
        cluster_counts=cluster_counts,
        degenerate_resample_count=0,
        seed=0,
        rng_algorithm="numpy-pcg64dxsm-v1",
        implementation_version="1.0.0",
    )


def subject_factory(
    subject_id: str = "cell-1",
    *,
    experimental_unit_id: str = "well-1",
) -> BeliefSubject:
    return BeliefSubject(
        subject_id=subject_id,
        kind=SubjectKind.INDIVIDUAL_CELL,
        biological_system=OntologyTerm(label="synthetic reference cell"),
        membership_semantics="one directly tracked synthetic cell",
        experimental_unit_kind="well",
        experimental_unit_id=experimental_unit_id,
        identity_basis=IdentityBasis.DIRECT_TRACKING,
    )


def subject_specification_factory(subject: BeliefSubject | None = None) -> SubjectSpecification:
    resolved = subject or subject_factory()
    return SubjectSpecification(
        kind=resolved.kind,
        biological_system=resolved.biological_system,
        membership_semantics=resolved.membership_semantics,
        experimental_unit_kind=resolved.experimental_unit_kind,
        allowed_identity_bases=(resolved.identity_basis,),
    )


def target_aggregation_factory(
    subject: BeliefSubject | None = None,
) -> TargetAggregation:
    resolved = subject or subject_factory()
    return TargetAggregation(
        subject_kind=resolved.kind,
        statistic=AggregationStatistic.INDIVIDUAL,
        experimental_unit=resolved.experimental_unit_kind,
    )


def evidence_link_factory(
    *,
    target: BeliefSubject | None = None,
    source: BeliefSubject | None = None,
    role: EvidenceRole = EvidenceRole.DIRECT,
    linkage_basis: IdentityBasis | None = None,
    linkage_confidence: float = 1,
) -> EvidenceLink:
    resolved_target = target or subject_factory()
    resolved_source = source or resolved_target
    basis = linkage_basis
    if basis is None:
        basis = (
            resolved_target.identity_basis
            if role is EvidenceRole.DIRECT
            else IdentityBasis.HERITABLE_BARCODE
        )
    return EvidenceLink(
        source_subject=resolved_source,
        target_subject=resolved_target,
        role=role,
        linkage_basis=basis,
        linkage_confidence=linkage_confidence,
        linkage_details="explicit synthetic test linkage",
        sampling_unit_id=resolved_source.experimental_unit_id,
    )


def collection_factory(
    effect: CollectionEffect = CollectionEffect.NONDESTRUCTIVE,
) -> ObservationCollection:
    return ObservationCollection(effect=effect)


def assay_spec_factory(
    *,
    assay_id: str = "signal-panel",
    modality: str = "phosphosignaling",
    cost: float = 2,
    turnaround_seconds: float = 0,
    collection: ObservationCollection | None = None,
) -> AssaySpec:
    return AssaySpec(
        assay_id=assay_id,
        modality=OntologyTerm(label=modality),
        protocol_reference=VersionedReference(
            reference_id=f"{assay_id}-protocol",
            version="test-v1",
            fingerprint="4" * 64,
        ),
        collection=collection or collection_factory(),
        purposes=(AssayPurpose.MEASUREMENT_SELECTION,),
        cost=cost,
        cost_units="synthetic_credit",
        turnaround_seconds=turnaround_seconds,
    )


def schedule_domain_factory() -> ScheduleDomain:
    return ScheduleDomain(
        allowed_kinds=(ScheduleKind.SINGLE,),
        administration_count=IntegerRange(minimum=1, maximum=1),
        interval_seconds=None,
        washout_seconds=ScalarRange(minimum=0, maximum=10_000),
    )


def intervention_spec_factory(
    *,
    spec_id: str = "drug",
    kind: str = "drug",
    target: OntologyTerm | None = None,
    mechanisms: tuple[OntologyTerm, ...] = (),
    dose_units: str = "relative",
    minimum_dose: float = 0,
    maximum_dose: float = 100,
    minimum_duration_seconds: float = 0,
    maximum_duration_seconds: float = 10_000,
    delivery_methods: tuple[str, ...] = ("synthetic_reference",),
    allowed_reversibility_statuses: tuple[ReversibilityStatus, ...] = (
        ReversibilityStatus.REVERSIBLE,
    ),
    allowed_assignment_mechanisms: tuple[AssignmentMechanism, ...] = (
        AssignmentMechanism.ASSIGNED_NONRANDOM,
    ),
    assignment_unit_kind: str = "well",
    randomization_unit_kind: str | None = None,
    require_randomization_unit: bool = False,
    require_matched_control: bool = False,
) -> InterventionSpec:
    return InterventionSpec(
        spec_id=spec_id,
        kind=OntologyTerm(label=kind),
        target=target,
        mechanisms=mechanisms,
        dose_domain=NumericDomain(
            minimum=minimum_dose,
            maximum=maximum_dose,
            units=dose_units,
        ),
        duration_seconds=ScalarRange(
            minimum=minimum_duration_seconds,
            maximum=maximum_duration_seconds,
        ),
        schedule=schedule_domain_factory(),
        delivery_methods=delivery_methods,
        allowed_reversibility_statuses=allowed_reversibility_statuses,
        allowed_assignment_mechanisms=allowed_assignment_mechanisms,
        assignment_unit_kind=assignment_unit_kind,
        randomization_unit_kind=randomization_unit_kind,
        require_randomization_unit=require_randomization_unit,
        require_matched_control=require_matched_control,
        realization_evidence=RealizationEvidenceRequirement(
            allowed_statuses=tuple(PerturbationStatus),
            allowed_modalities=(
                OntologyTerm(label="transcriptome"),
                OntologyTerm(label="phosphosignaling"),
            ),
            minimum_evidence_events=0,
        ),
    )


def intervention_factory(
    *,
    event_id: str = "drug",
    subject: BeliefSubject | None = None,
    time_seconds: float = 0,
    duration_seconds: float = 1,
    intervention_type: str = "drug",
    intervention_spec_id: str | None = None,
    target: OntologyTerm | None = None,
    mechanism: OntologyTerm | None = None,
    dose: float = 1,
    dose_units: str = "relative",
    delivery_method: str = "synthetic_reference",
    reversibility_status: ReversibilityStatus = ReversibilityStatus.REVERSIBLE,
    estimated_efficiency: float | None = 1,
    assignment_mechanism: AssignmentMechanism = AssignmentMechanism.ASSIGNED_NONRANDOM,
    assignment_unit_kind: str = "well",
    assignment_unit_id: str | None = None,
    randomization_unit_kind: str | None = None,
    randomization_unit_id: str | None = None,
    matched_control: MatchedControl | None = None,
    actual_perturbation: ActualPerturbation | None = None,
) -> InterventionEvent:
    resolved_subject = subject or subject_factory()
    return InterventionEvent(
        event_id=event_id,
        subject=resolved_subject,
        time_seconds=time_seconds,
        intervention_spec_id=intervention_spec_id or intervention_type,
        intervention_type=OntologyTerm(label=intervention_type),
        target=target,
        mechanism=mechanism,
        dose=Quantity(value=dose, units=dose_units),
        duration_seconds=duration_seconds,
        schedule=InterventionSchedule(
            kind=ScheduleKind.SINGLE,
            administration_count=1,
            washout_seconds=0,
        ),
        delivery_method=delivery_method,
        reversibility_status=reversibility_status,
        estimated_efficiency=estimated_efficiency,
        assignment_mechanism=assignment_mechanism,
        assignment_unit_kind=assignment_unit_kind,
        assignment_unit_id=assignment_unit_id or resolved_subject.experimental_unit_id,
        randomization_unit_kind=randomization_unit_kind,
        randomization_unit_id=randomization_unit_id,
        matched_control=matched_control,
        actual_perturbation=actual_perturbation,
    )


def environment_spec_factory(
    *,
    variable: str = "nutrient",
    units: str = "relative",
    required: bool = True,
    minimum: float = -100,
    maximum: float = 100,
    maximum_duration_seconds: float = 10_000,
) -> EnvironmentVariableSpec:
    return EnvironmentVariableSpec(
        variable=OntologyTerm(label=variable),
        domain=NumericDomain(minimum=minimum, maximum=maximum, units=units),
        duration_seconds=ScalarRange(minimum=0, maximum=maximum_duration_seconds),
        required=required,
        allowed_temporal_modes=(EnvironmentTemporalMode.FIXED,),
        missing_history_policy="reject",
    )


def environment_factory(
    *,
    event_id: str = "environment",
    subject: BeliefSubject | None = None,
    time_seconds: float = 0,
    duration_seconds: float = 0,
    variables: dict[str, object] | None = None,
    spatial_region: str | None = None,
) -> EnvironmentEvent:
    return EnvironmentEvent(
        event_id=event_id,
        subject=subject or subject_factory(),
        time_seconds=time_seconds,
        variables=variables or {"nutrient": Quantity(value=1, units="relative")},
        duration_seconds=duration_seconds,
        temporal_mode=EnvironmentTemporalMode.FIXED,
        spatial_region=spatial_region,
    )


def query_factory() -> StateQuery:
    subject = subject_factory()
    return StateQuery(
        subject=subject_specification_factory(subject),
        system_boundary=SystemBoundary.ISOLATED_CELL,
        temporal_resolution_seconds=1,
        prediction_horizons=(
            PredictionHorizon(name="acute", duration_seconds=60, timescale=Timescale.FAST),
        ),
        target_outputs=(
            OutputSpec(
                term=OntologyTerm(label="functional capacity"),
                units="relative",
                aggregation=target_aggregation_factory(subject),
                endpoint=LatentQuantityEndpoint(
                    model_reference=VersionedReference(
                        reference_id="linear-gaussian-reference",
                        version="0.2.0",
                        fingerprint="7" * 64,
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
        intervention_space=(intervention_spec_factory(),),
        available_assays=(assay_spec_factory(),),
        evidence_policy=EvidencePolicy(
            lookback_seconds=None,
            include_at_cutoff=True,
            allowed_modalities=tuple(
                OntologyTerm(label=modality)
                for modality in (
                    "transcriptome",
                    "phosphosignaling",
                    "functional_readout",
                )
            ),
            allowed_evidence_roles=(EvidenceRole.DIRECT,),
            minimum_observed_measurements=0,
        ),
        acceptance_thresholds=AcceptanceThresholds(
            maximum_ood_score=1,
            maximum_history_information_gain=1,
            minimum_calibration_coverage=0.5,
            maximum_calibration_error=1,
            maximum_counterfactual_uncertainty=1,
            maximum_decision_uncertainty=1,
            minimum_identifiability=0.01,
        ),
        constraints=QueryConstraints(
            maximum_intervention_combination_order=1,
            require_complete_intervention_history=False,
            require_complete_environment_history=False,
            require_complete_lineage_history=False,
            require_complete_neighborhood_history=False,
            allow_transport=False,
            maximum_total_assay_cost=100,
            assay_cost_units="synthetic_credit",
            maximum_assay_delay_seconds=100_000,
        ),
    )


def observation_factory(
    *,
    event_id: str = "obs-0",
    subject: BeliefSubject | None = None,
    source_subject: BeliefSubject | None = None,
    evidence_role: EvidenceRole = EvidenceRole.DIRECT,
    linkage_basis: IdentityBasis | None = None,
    time_seconds: float = 0,
    duration_seconds: float = 0,
    modality: str = "transcriptome",
    value: object = 0.5,
    units: str | None = "relative",
    missingness: MissingnessReport | None = None,
    assay: AssayMetadata | None = None,
    uncertainty: MeasurementUncertainty | None = None,
    quality: QualityReport | None = None,
    collection: ObservationCollection | None = None,
) -> ObservationEvent:
    target = subject or subject_factory()
    return ObservationEvent(
        event_id=event_id,
        subject=target,
        time_seconds=time_seconds,
        duration_seconds=duration_seconds,
        modality=OntologyTerm(label=modality),
        evidence_link=evidence_link_factory(
            target=target,
            source=source_subject,
            role=evidence_role,
            linkage_basis=linkage_basis,
        ),
        collection=collection or collection_factory(),
        value=value,
        units=units,
        missingness=missingness or MissingnessReport(),
        assay=assay or AssayMetadata(assay_id=f"{modality}-assay"),
        uncertainty=uncertainty or MeasurementUncertainty(),
        quality=quality or QualityReport(),
    )


def request_factory(
    *,
    history: CellHistory | None = None,
    as_of_seconds: float = 10,
    query: StateQuery | None = None,
) -> EstimateCellStateRequest:
    resolved_history = history or CellHistory(
        subject=subject_factory(), events=(observation_factory(),)
    )
    existing = resolved_history.completeness
    resolved_history = resolved_history.model_copy(
        update={
            "completeness": HistoryCompleteness(
                observations=existing.observations,
                interventions=RecordCompleteness.COMPLETE,
                environments=RecordCompleteness.COMPLETE,
                lineage=RecordCompleteness.COMPLETE,
                neighborhood=RecordCompleteness.COMPLETE,
            )
        }
    )
    return EstimateCellStateRequest(
        query=query or query_factory(),
        history=resolved_history,
        as_of_seconds=as_of_seconds,
        static_context=StaticContext(species=OntologyTerm(label="Homo sapiens")),
    )


@pytest.fixture
def query() -> StateQuery:
    return query_factory()


@pytest.fixture
def model() -> LinearGaussianReference:
    return LinearGaussianReference(minimal_reference_config())


@pytest.fixture
def estimate_request() -> EstimateCellStateRequest:
    return request_factory()
