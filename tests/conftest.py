from __future__ import annotations

import pytest

from cellstate import (
    AssayMetadata,
    AssaySpec,
    CellHistory,
    EstimateCellStateRequest,
    HistoryCompleteness,
    InterventionSpec,
    ObservationEvent,
    OntologyTerm,
    OutputSpec,
    PredictionHorizon,
    RecordCompleteness,
    StateQuery,
    StaticContext,
    SystemBoundary,
    Timescale,
)
from cellstate.reference import LinearGaussianReference, minimal_reference_config


def query_factory() -> StateQuery:
    return StateQuery(
        system_boundary=SystemBoundary.ISOLATED_CELL,
        prediction_horizons=(
            PredictionHorizon(name="acute", duration_seconds=60, timescale=Timescale.FAST),
        ),
        target_outputs=(
            OutputSpec(term=OntologyTerm(label="functional capacity"), units="relative"),
        ),
        intervention_space=(InterventionSpec(kind=OntologyTerm(label="drug")),),
        available_assays=(
            AssaySpec(
                assay_id="signal-panel",
                modality=OntologyTerm(label="phosphosignaling"),
                cost=2,
            ),
        ),
    )


def observation_factory(
    *,
    event_id: str = "obs-0",
    time_seconds: float = 0,
    modality: str = "transcriptome",
    value: object = 0.5,
) -> ObservationEvent:
    return ObservationEvent(
        event_id=event_id,
        subject_id="cell-1",
        time_seconds=time_seconds,
        modality=OntologyTerm(label=modality),
        value=value,
        units="relative",
        assay=AssayMetadata(assay_id=f"{modality}-assay"),
    )


def request_factory(
    *,
    history: CellHistory | None = None,
    as_of_seconds: float = 10,
    query: StateQuery | None = None,
) -> EstimateCellStateRequest:
    resolved_history = history or CellHistory(subject_id="cell-1", events=(observation_factory(),))
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
