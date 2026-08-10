from __future__ import annotations

import pytest
from conftest import (
    SYNTHETIC_TEST_OPTIONS,
    observation_factory,
    query_factory,
    request_factory,
    subject_factory,
)
from pydantic import ValidationError

from cellstate import (
    CellHistory,
    EstimateCellStateRequest,
    EvolutionScenario,
    InterventionObjective,
    ObjectiveDirection,
    ObjectiveTerm,
    OntologyTerm,
    PrecisionRequirement,
    choose_intervention,
    estimate_cell_state,
    evolve_cell_state,
)
from cellstate.domain import DivisionEvent
from cellstate.engine import ModelRegistry, build_event_graph
from cellstate.errors import CapabilityError, ContractViolationError, UnsupportedModalityError
from cellstate.reference import LinearGaussianPlanner


def test_event_graph_tracks_canonical_temporal_precedence() -> None:
    history = CellHistory(
        subject=subject_factory(),
        events=(
            observation_factory(event_id="later", time_seconds=2),
            observation_factory(event_id="earlier", time_seconds=1),
        ),
    )
    graph = build_event_graph(history)
    assert graph.event_ids == ("earlier", "later")
    assert len(graph.edges) == 1
    assert graph.edges[0].relationship == "strict_temporal_order"


def test_event_graph_does_not_order_simultaneous_events_and_preserves_division() -> None:
    first = observation_factory(event_id="a", time_seconds=1)
    simultaneous = observation_factory(event_id="b", time_seconds=1)
    division = DivisionEvent(
        event_id="division",
        subject=subject_factory(),
        time_seconds=2,
        child_ids=("child-a", "child-b"),
    )
    graph = build_event_graph(
        CellHistory(subject=subject_factory(), events=(first, simultaneous, division))
    )
    edge_pairs = {(edge.source_event_id, edge.target_event_id) for edge in graph.edges}
    assert ("a", "b") not in edge_pairs
    assert edge_pairs == {("a", "division"), ("b", "division")}
    assert {edge.child_subject_id for edge in graph.lineage_edges} == {"child-a", "child-b"}


def test_model_registry_is_case_insensitive_and_rejects_duplicates() -> None:
    registry = ModelRegistry((("RNA", object()),))
    assert registry.supports("rna")
    assert registry.get("rNa") is not None
    assert registry.keys == ("rna",)
    with pytest.raises(ValueError, match="already registered"):
        registry.register("rna", object())
    with pytest.raises(UnsupportedModalityError):
        registry.get("protein")


def test_capability_preflight_cannot_be_bypassed_for_unsupported_target(model) -> None:
    base_query = query_factory()
    query = base_query.model_copy(
        update={
            "target_outputs": (
                base_query.target_outputs[0].model_copy(
                    update={"term": OntologyTerm(label="unknown output")}
                ),
            )
        }
    )
    request = request_factory(query=query)
    with pytest.raises(CapabilityError, match="unknown_output"):
        estimate_cell_state(
            request,
            estimator=model,
            options=SYNTHETIC_TEST_OPTIONS,
        )


def test_public_api_rejects_mismatched_scenario_and_empty_candidates(model) -> None:
    request = request_factory()
    belief = estimate_cell_state(
        request,
        estimator=model,
        options=SYNTHETIC_TEST_OPTIONS,
    )
    wrong_subject = EvolutionScenario(
        scenario_id="wrong-subject",
        horizon_name="acute",
        subject=subject_factory("cell-2"),
        start_time_seconds=10,
        end_time_seconds=70,
    )
    with pytest.raises(ContractViolationError, match="same typed subject"):
        evolve_cell_state(
            belief,
            scenario=wrong_subject,
            evolution_model=model,
            options=SYNTHETIC_TEST_OPTIONS,
        )
    wrong_time = wrong_subject.model_copy(
        update={
            "scenario_id": "wrong-time",
            "subject": subject_factory(),
            "start_time_seconds": 9,
        }
    )
    with pytest.raises(ContractViolationError, match="belief time"):
        evolve_cell_state(
            belief,
            scenario=wrong_time,
            evolution_model=model,
            options=SYNTHETIC_TEST_OPTIONS,
        )
    with pytest.raises(ValueError, match="at least one"):
        choose_intervention(
            belief,
            objective=InterventionObjective(
                objective_id="objective",
                horizon_name="acute",
                terms=(
                    ObjectiveTerm(
                        target=OntologyTerm(label="functional capacity"),
                        direction=ObjectiveDirection.MAXIMIZE,
                    ),
                ),
            ),
            candidates=(),
            planner=LinearGaussianPlanner(model),
            options=SYNTHETIC_TEST_OPTIONS,
        )


def test_recursive_request_rejects_subject_time_and_query_mismatches(model) -> None:
    request = request_factory()
    belief = estimate_cell_state(
        request,
        estimator=model,
        options=SYNTHETIC_TEST_OPTIONS,
    )
    with pytest.raises(ValidationError, match="cannot be later"):
        EstimateCellStateRequest(
            query=request.query,
            history=request.history,
            as_of_seconds=5,
            static_context=request.static_context,
            previous_belief=belief,
        )
    other_history = CellHistory(
        subject=subject_factory("cell-2"),
        events=(
            observation_factory(
                event_id="other",
                subject=subject_factory("cell-2"),
            ),
        ),
    )
    with pytest.raises(ValidationError, match="same subject"):
        EstimateCellStateRequest(
            query=request.query,
            history=other_history,
            as_of_seconds=10,
            static_context=request.static_context,
            previous_belief=belief,
        )
    different_query_payload = query_factory().model_dump()
    different_query_payload["system_boundary"] = "population"
    different_query = type(request.query).model_validate(different_query_payload)
    with pytest.raises(ValidationError, match="same state query"):
        EstimateCellStateRequest(
            query=different_query,
            history=request.history,
            as_of_seconds=10,
            static_context=request.static_context,
            previous_belief=belief,
        )
    changed_context = request.static_context.model_copy(update={"donor_id": "other-donor"})
    with pytest.raises(ValidationError, match="same static/population context"):
        EstimateCellStateRequest(
            query=request.query,
            history=request.history,
            as_of_seconds=10,
            static_context=changed_context,
            previous_belief=belief,
        )


def test_capability_preflight_covers_boundary_and_precision(model) -> None:
    query_payload = query_factory().model_dump()
    query_payload["system_boundary"] = "population"
    population_query = type(query_factory()).model_validate(query_payload)
    with pytest.raises(CapabilityError, match="population"):
        estimate_cell_state(
            request_factory(query=population_query),
            estimator=model,
            options=SYNTHETIC_TEST_OPTIONS,
        )

    precision_query = query_factory().model_copy(
        update={
            "precision_requirements": (
                PrecisionRequirement(
                    target=OntologyTerm(label="functional capacity"),
                    horizon_name="acute",
                    metric="absolute_error",
                    maximum_error=0.1,
                ),
            )
        }
    )
    with pytest.raises(CapabilityError, match="absolute_error"):
        estimate_cell_state(
            request_factory(query=precision_query),
            estimator=model,
            options=SYNTHETIC_TEST_OPTIONS,
        )
