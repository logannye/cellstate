"""Minimal estimate -> evolve example using the non-biological reference backend."""

from cellstate import (
    ActualPerturbation,
    AssayMetadata,
    CellHistory,
    EstimateCellStateRequest,
    EvolutionScenario,
    HistoryCompleteness,
    InferenceOptions,
    OntologyTerm,
    OutputSpec,
    PerturbationStatus,
    PredictionHorizon,
    Quantity,
    StateQuery,
    StaticContext,
    SystemBoundary,
    Timescale,
    estimate_cell_state,
    evolve_cell_state,
)
from cellstate.domain import RecordCompleteness
from cellstate.domain.common import canonical_fingerprint
from cellstate.domain.events import (
    AssignmentMechanism,
    CollectionEffect,
    EnvironmentEvent,
    EnvironmentTemporalMode,
    EvidenceLink,
    EvidenceRole,
    InterventionEvent,
    InterventionSchedule,
    ObservationCollection,
    ObservationEvent,
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

CELL = "cell-001"
CONTRACT_DEMO_OPTIONS = InferenceOptions()
SUBJECT = BeliefSubject(
    subject_id=CELL,
    kind=SubjectKind.INDIVIDUAL_CELL,
    biological_system=OntologyTerm(label="synthetic reference cell"),
    membership_semantics="one directly tracked synthetic cell",
    experimental_unit_kind="well",
    experimental_unit_id="well-001",
    identity_basis=IdentityBasis.DIRECT_TRACKING,
)

DIRECT_LINK = EvidenceLink(
    source_subject=SUBJECT,
    target_subject=SUBJECT,
    role=EvidenceRole.DIRECT,
    linkage_basis=IdentityBasis.DIRECT_TRACKING,
    linkage_confidence=1,
    linkage_details="same synthetic cell under direct tracking",
    sampling_unit_id="well-001",
)
NONDESTRUCTIVE = ObservationCollection(effect=CollectionEffect.NONDESTRUCTIVE)

query = StateQuery(
    subject=SubjectSpecification(
        kind=SUBJECT.kind,
        biological_system=SUBJECT.biological_system,
        membership_semantics=SUBJECT.membership_semantics,
        experimental_unit_kind=SUBJECT.experimental_unit_kind,
        allowed_identity_bases=(IdentityBasis.DIRECT_TRACKING,),
    ),
    system_boundary=SystemBoundary.CELL_AND_SOLUBLE_ENVIRONMENT,
    temporal_resolution_seconds=1,
    prediction_horizons=(
        PredictionHorizon(name="acute", duration_seconds=60, timescale=Timescale.FAST),
        PredictionHorizon(
            name="intermediate", duration_seconds=3600, timescale=Timescale.INTERMEDIATE
        ),
    ),
    target_outputs=(
        OutputSpec(
            term=OntologyTerm(label="functional capacity"),
            units="relative",
            aggregation=TargetAggregation(
                subject_kind=SubjectKind.INDIVIDUAL_CELL,
                statistic=AggregationStatistic.INDIVIDUAL,
                experimental_unit="well",
            ),
            endpoint=LatentQuantityEndpoint(
                model_reference=VersionedReference(
                    reference_id="linear-gaussian-reference-output-decoder",
                    version="0.2.0",
                    fingerprint=canonical_fingerprint(
                        {
                            "reference_id": "linear-gaussian-reference-output-decoder",
                            "version": "0.2.0",
                        }
                    ),
                )
            ),
            value_schema_reference=VersionedReference(
                reference_id="functional-capacity-scalar-schema",
                version="demo-v1",
                fingerprint=canonical_fingerprint(
                    {
                        "reference_id": "functional-capacity-scalar-schema",
                        "version": "demo-v1",
                        "shape": [],
                        "units": "relative",
                    }
                ),
            ),
            missingness=TargetMissingnessSemantics(
                policy=TargetMissingnessPolicy.NOT_APPLICABLE,
                reportable_statuses=(),
            ),
            censoring=TargetCensoringSemantics(
                policy=TargetCensoringPolicy.NOT_APPLICABLE,
                allowed_directions=(),
            ),
            supported_horizon_names=("acute", "intermediate"),
            weight=1,
            functional=True,
        ),
    ),
    intervention_space=(
        InterventionSpec(
            spec_id="drug",
            kind=OntologyTerm(label="drug"),
            dose_domain=NumericDomain(minimum=0, maximum=10, units="relative"),
            duration_seconds=ScalarRange(minimum=0, maximum=3600),
            schedule=ScheduleDomain(
                allowed_kinds=(ScheduleKind.SINGLE,),
                administration_count=IntegerRange(minimum=1, maximum=1),
                interval_seconds=None,
                washout_seconds=ScalarRange(minimum=0, maximum=3600),
            ),
            delivery_methods=("synthetic_reference",),
            allowed_reversibility_statuses=(ReversibilityStatus.REVERSIBLE,),
            allowed_assignment_mechanisms=(AssignmentMechanism.ASSIGNED_NONRANDOM,),
            assignment_unit_kind="well",
            randomization_unit_kind=None,
            require_randomization_unit=False,
            require_matched_control=False,
            realization_evidence=RealizationEvidenceRequirement(
                allowed_statuses=tuple(PerturbationStatus),
                allowed_modalities=(
                    OntologyTerm(label="transcriptome"),
                    OntologyTerm(label="phosphosignaling"),
                ),
                minimum_evidence_events=0,
            ),
        ),
    ),
    environment_space=(
        EnvironmentVariableSpec(
            variable=OntologyTerm(label="nutrient"),
            domain=NumericDomain(minimum=0, maximum=1, units="relative"),
            duration_seconds=ScalarRange(minimum=0, maximum=3600),
            required=True,
            allowed_temporal_modes=(EnvironmentTemporalMode.FIXED,),
            missing_history_policy="reject",
        ),
    ),
    available_assays=(
        AssaySpec(
            assay_id="phospho-panel",
            modality=OntologyTerm(label="phosphosignaling"),
            protocol_reference=VersionedReference(
                reference_id="synthetic-phospho-panel-protocol",
                version="demo-v1",
                fingerprint="3" * 64,
            ),
            collection=NONDESTRUCTIVE,
            purposes=(AssayPurpose.MEASUREMENT_SELECTION,),
            cost=2,
            cost_units="synthetic_credit",
            turnaround_seconds=0,
        ),
        AssaySpec(
            assay_id="functional-challenge",
            modality=OntologyTerm(label="functional readout"),
            protocol_reference=VersionedReference(
                reference_id="synthetic-functional-challenge-protocol",
                version="demo-v1",
                fingerprint="2" * 64,
            ),
            collection=NONDESTRUCTIVE,
            purposes=(AssayPurpose.MEASUREMENT_SELECTION,),
            cost=5,
            cost_units="synthetic_credit",
            turnaround_seconds=0,
        ),
    ),
    evidence_policy=EvidencePolicy(
        lookback_seconds=60.0,
        include_at_cutoff=True,
        allowed_modalities=(
            OntologyTerm(label="transcriptome"),
            OntologyTerm(label="phosphosignaling"),
        ),
        allowed_evidence_roles=(EvidenceRole.DIRECT,),
        minimum_observed_measurements=1,
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
        require_complete_intervention_history=True,
        require_complete_environment_history=True,
        require_complete_lineage_history=True,
        require_complete_neighborhood_history=True,
        allow_transport=False,
        maximum_total_assay_cost=10,
        assay_cost_units="synthetic_credit",
        maximum_assay_delay_seconds=3600,
    ),
)

history = CellHistory(
    subject=SUBJECT,
    completeness=HistoryCompleteness(
        observations=RecordCompleteness.INCOMPLETE,
        interventions=RecordCompleteness.COMPLETE,
        environments=RecordCompleteness.COMPLETE,
        lineage=RecordCompleteness.COMPLETE,
        neighborhood=RecordCompleteness.COMPLETE,
    ),
    events=(
        EnvironmentEvent(
            event_id="env-0",
            subject=SUBJECT,
            time_seconds=0,
            duration_seconds=60,
            temporal_mode=EnvironmentTemporalMode.FIXED,
            variables={"nutrient": Quantity(value=0.8, units="relative")},
        ),
        ObservationEvent(
            event_id="rna-0",
            subject=SUBJECT,
            time_seconds=0,
            duration_seconds=0,
            modality=OntologyTerm(label="transcriptome"),
            evidence_link=DIRECT_LINK,
            collection=NONDESTRUCTIVE,
            value=0.4,
            units="relative",
            assay=AssayMetadata(assay_id="rna-panel"),
        ),
        InterventionEvent(
            event_id="drug-1",
            subject=SUBJECT,
            time_seconds=20,
            intervention_spec_id="drug",
            duration_seconds=30,
            intervention_type=OntologyTerm(label="drug"),
            dose=Quantity(value=1, units="relative"),
            schedule=InterventionSchedule(
                kind=ScheduleKind.SINGLE,
                administration_count=1,
                washout_seconds=0,
            ),
            delivery_method="synthetic_reference",
            reversibility_status=ReversibilityStatus.REVERSIBLE,
            assignment_mechanism=AssignmentMechanism.ASSIGNED_NONRANDOM,
            assignment_unit_kind="well",
            assignment_unit_id=SUBJECT.experimental_unit_id,
            matched_control=None,
            actual_perturbation=ActualPerturbation(
                status=PerturbationStatus.INFERRED,
                efficiency=0.75,
                evidence_event_ids=("signal-45",),
            ),
        ),
        ObservationEvent(
            event_id="signal-45",
            subject=SUBJECT,
            time_seconds=45,
            duration_seconds=0,
            modality=OntologyTerm(label="phosphosignaling"),
            evidence_link=DIRECT_LINK,
            collection=NONDESTRUCTIVE,
            value=0.7,
            units="relative",
            assay=AssayMetadata(assay_id="phospho-panel"),
        ),
    ),
)

request = EstimateCellStateRequest(
    query=query,
    history=history,
    as_of_seconds=60,
    static_context=StaticContext(
        species=OntologyTerm(label="Homo sapiens", identifier="NCBITaxon:9606")
    ),
)

model = LinearGaussianReference(minimal_reference_config())
belief = estimate_cell_state(
    request,
    estimator=model,
    options=CONTRACT_DEMO_OPTIONS,
)

scenario = EvolutionScenario(
    scenario_id="washout-follow-up",
    horizon_name="acute",
    subject=SUBJECT,
    start_time_seconds=60,
    end_time_seconds=120,
    environments=(
        EnvironmentEvent(
            event_id="env-future",
            subject=SUBJECT,
            time_seconds=60,
            duration_seconds=60,
            temporal_mode=EnvironmentTemporalMode.FIXED,
            variables={"nutrient": Quantity(value=1, units="relative")},
        ),
    ),
)
forecast = evolve_cell_state(
    belief,
    scenario=scenario,
    evolution_model=model,
    options=CONTRACT_DEMO_OPTIONS,
)

print(belief.model_dump_json(indent=2))
print(f"Forecast posterior family: {forecast.joint_posterior.kind}")
