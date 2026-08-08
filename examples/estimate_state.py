"""Minimal estimate -> evolve example using the non-biological reference backend."""

from cellstate import (
    ActualPerturbation,
    AssayMetadata,
    AssaySpec,
    CellHistory,
    EnvironmentEvent,
    EnvironmentVariableSpec,
    EstimateCellStateRequest,
    EvolutionScenario,
    HistoryCompleteness,
    InterventionEvent,
    InterventionSpec,
    ObservationEvent,
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
from cellstate.reference import LinearGaussianReference, minimal_reference_config

CELL = "cell-001"

query = StateQuery(
    system_boundary=SystemBoundary.CELL_AND_SOLUBLE_ENVIRONMENT,
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
            functional=True,
        ),
    ),
    intervention_space=(InterventionSpec(kind=OntologyTerm(label="drug"), dose_units="relative"),),
    environment_space=(
        EnvironmentVariableSpec(
            variable=OntologyTerm(label="nutrient"),
            units="relative",
            required=True,
        ),
    ),
    available_assays=(
        AssaySpec(
            assay_id="phospho-panel",
            modality=OntologyTerm(label="phosphosignaling"),
            cost=2.0,
        ),
        AssaySpec(
            assay_id="functional-challenge",
            modality=OntologyTerm(label="functional readout"),
            cost=5.0,
        ),
    ),
)

history = CellHistory(
    subject_id=CELL,
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
            subject_id=CELL,
            time_seconds=0,
            variables={"nutrient": Quantity(value=0.8, units="relative")},
        ),
        ObservationEvent(
            event_id="rna-0",
            subject_id=CELL,
            time_seconds=0,
            modality=OntologyTerm(label="transcriptome"),
            value=0.4,
            units="relative",
            assay=AssayMetadata(assay_id="rna-panel"),
        ),
        InterventionEvent(
            event_id="drug-1",
            subject_id=CELL,
            time_seconds=20,
            duration_seconds=30,
            intervention_type=OntologyTerm(label="drug"),
            dose=Quantity(value=1.0, units="relative"),
            actual_perturbation=ActualPerturbation(
                status=PerturbationStatus.INFERRED,
                efficiency=0.75,
                evidence_event_ids=("rna-0",),
            ),
        ),
        ObservationEvent(
            event_id="signal-45",
            subject_id=CELL,
            time_seconds=45,
            modality=OntologyTerm(label="phosphosignaling"),
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
belief = estimate_cell_state(request, estimator=model)

scenario = EvolutionScenario(
    scenario_id="washout-follow-up",
    horizon_name="acute",
    subject_id=CELL,
    start_time_seconds=60,
    end_time_seconds=120,
    environments=(
        EnvironmentEvent(
            event_id="env-future",
            subject_id=CELL,
            time_seconds=60,
            variables={"nutrient": Quantity(value=1.0, units="relative")},
        ),
    ),
)
forecast = evolve_cell_state(belief, scenario=scenario, evolution_model=model)

print(belief.model_dump_json(indent=2))
print(f"Forecast posterior family: {forecast.joint_posterior.kind}")
