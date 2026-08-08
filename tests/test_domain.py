from __future__ import annotations

import pytest
from conftest import observation_factory, query_factory, request_factory
from pydantic import ValidationError

from cellstate import (
    ActualPerturbation,
    AssayMetadata,
    AssaySpec,
    CellHistory,
    EstimateCellStateRequest,
    EvidenceRole,
    MissingnessReport,
    MissingnessStatus,
    ObservationEvent,
    OntologyTerm,
    PerturbationStatus,
    PredictionHorizon,
    StateQuery,
    Timescale,
)
from cellstate.domain import (
    EvolutionScenario,
    ParametricDistribution,
    SpatialEdge,
    SpatialGraph,
    SpatialNode,
)


def test_observed_zero_is_not_missing() -> None:
    observation = observation_factory(value=0)
    assert observation.value == 0
    assert observation.missingness.status is MissingnessStatus.OBSERVED


@pytest.mark.parametrize(
    ("status", "value"),
    [
        (MissingnessStatus.OBSERVED, None),
        (MissingnessStatus.MISSING, 0),
        (MissingnessStatus.NOT_MEASURED, 1),
        (MissingnessStatus.ASSAY_FAILURE, 1),
    ],
)
def test_observation_rejects_incoherent_missingness(
    status: MissingnessStatus, value: object
) -> None:
    with pytest.raises(ValidationError):
        ObservationEvent(
            event_id="bad",
            subject_id="cell-1",
            time_seconds=0,
            modality=OntologyTerm(label="transcriptome"),
            value=value,
            missingness=MissingnessReport(status=status),
            assay=AssayMetadata(assay_id="rna"),
        )


def test_history_is_canonical_and_has_stable_fingerprint() -> None:
    early = observation_factory(event_id="early", time_seconds=1)
    late = observation_factory(event_id="late", time_seconds=2)
    first = CellHistory(subject_id="cell-1", events=(late, early))
    second = CellHistory(subject_id="cell-1", events=(early, late))
    assert [event.event_id for event in first.events] == ["early", "late"]
    assert first.fingerprint == second.fingerprint
    assert first.through(1) == (early,)
    assert first.between(1, 2) == (late,)


def test_history_rejects_duplicate_ids_and_mixed_subjects() -> None:
    duplicate = observation_factory(event_id="same")
    with pytest.raises(ValidationError, match="event IDs"):
        CellHistory(subject_id="cell-1", events=(duplicate, duplicate))
    other = duplicate.model_copy(update={"event_id": "other", "subject_id": "cell-2"})
    with pytest.raises(ValidationError, match="history subject"):
        CellHistory(subject_id="cell-1", events=(duplicate, other))


def test_request_rejects_future_evidence() -> None:
    future = observation_factory(time_seconds=11)
    history = CellHistory(subject_id="cell-1", events=(future,))
    with pytest.raises(ValidationError, match="after as_of_seconds"):
        request_factory(history=history, as_of_seconds=10)


def test_query_rejects_duplicate_horizon_and_assay_names() -> None:
    query = query_factory()
    horizon = PredictionHorizon(name="same", duration_seconds=1, timescale=Timescale.FAST)
    with pytest.raises(ValidationError, match="horizon names"):
        StateQuery(
            system_boundary=query.system_boundary,
            prediction_horizons=(horizon, horizon.model_copy(update={"duration_seconds": 2})),
            target_outputs=query.target_outputs,
        )
    assay = AssaySpec(assay_id="same", modality=OntologyTerm(label="rna"))
    with pytest.raises(ValidationError, match="assay IDs"):
        query.model_copy(update={"available_assays": (assay, assay)}).model_validate(
            query.model_copy(update={"available_assays": (assay, assay)}).model_dump()
        )


def test_spatial_graph_rejects_dangling_edges() -> None:
    with pytest.raises(ValidationError, match="declared nodes"):
        SpatialGraph(
            nodes=(SpatialNode(node_id="a", node_type="cell"),),
            edges=(SpatialEdge(source_id="a", target_id="b", relationship="contact"),),
        )


def test_scenario_rejects_invalid_interval() -> None:
    with pytest.raises(ValidationError, match="later"):
        EvolutionScenario(
            scenario_id="bad",
            horizon_name="acute",
            subject_id="cell-1",
            start_time_seconds=10,
            end_time_seconds=10,
        )


def test_strict_models_reject_unknown_fields() -> None:
    payload = request_factory().model_dump()
    payload["unexpected"] = True
    with pytest.raises(ValidationError, match="Extra inputs"):
        EstimateCellStateRequest.model_validate(payload)


def test_schema_versions_are_enforced() -> None:
    payload = query_factory().model_dump()
    payload["schema_version"] = "2.0"
    with pytest.raises(ValidationError, match=r"1\.0"):
        StateQuery.model_validate(payload)


def test_parametric_distribution_rejects_non_psd_covariance() -> None:
    with pytest.raises(ValidationError, match="positive semidefinite"):
        ParametricDistribution(
            family="multivariate_normal",
            dimensions=("a", "b"),
            mean=(0, 0),
            covariance=((1, 2), (2, 1)),
        )


def test_lineage_evidence_names_its_source_subject() -> None:
    with pytest.raises(ValidationError, match="source_subject_id"):
        observation_factory().model_copy(
            update={"evidence_role": EvidenceRole.SIBLING}
        ).model_validate(
            {
                **observation_factory().model_dump(),
                "evidence_role": "sibling",
            }
        )
    sibling = ObservationEvent.model_validate(
        {
            **observation_factory().model_dump(),
            "evidence_role": "sibling",
            "source_subject_id": "sibling-1",
        }
    )
    assert sibling.subject_id == "cell-1"
    assert sibling.source_subject_id == "sibling-1"


@pytest.mark.parametrize(
    ("status", "efficiency", "evidence"),
    [
        (PerturbationStatus.MEASURED, None, ("assay",)),
        (PerturbationStatus.INFERRED, 0.5, ()),
        (PerturbationStatus.FAILED, 1.0, ()),
        (PerturbationStatus.UNKNOWN, 0.5, ()),
    ],
)
def test_realized_perturbation_rejects_contradictory_states(
    status: PerturbationStatus, efficiency: float | None, evidence: tuple[str, ...]
) -> None:
    with pytest.raises(ValidationError):
        ActualPerturbation(
            status=status,
            efficiency=efficiency,
            evidence_event_ids=evidence,
        )
    failed = ActualPerturbation(
        status=PerturbationStatus.FAILED,
        efficiency=0,
        evidence_event_ids=("assay",),
    )
    assert failed.efficiency == 0
