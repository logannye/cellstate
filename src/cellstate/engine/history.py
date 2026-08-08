"""Construction of a minimal causal event graph from a canonical timeline."""

from __future__ import annotations

from itertools import groupby, pairwise

from pydantic import Field

from cellstate.domain.common import SchemaModel
from cellstate.domain.events import DivisionEvent
from cellstate.domain.history import CellHistory


class EventEdge(SchemaModel):
    source_event_id: str
    target_event_id: str
    relationship: str = Field(min_length=1)


class EventGraph(SchemaModel):
    event_ids: tuple[str, ...]
    edges: tuple[EventEdge, ...]
    lineage_edges: tuple[LineageEdge, ...] = ()


class LineageEdge(SchemaModel):
    parent_subject_id: str
    child_subject_id: str
    division_event_id: str | None = None


def build_event_graph(history: CellHistory) -> EventGraph:
    """Build strict-time ordering and explicit division edges without inventing causality."""

    events = history.events
    time_groups = [
        tuple(group) for _, group in groupby(events, key=lambda event: event.time_seconds)
    ]
    edges = [
        EventEdge(
            source_event_id=left.event_id,
            target_event_id=right.event_id,
            relationship="strict_temporal_order",
        )
        for left_group, right_group in pairwise(time_groups)
        for left in left_group
        for right in right_group
    ]
    lineage_edges = [
        LineageEdge(
            parent_subject_id=event.subject_id,
            child_subject_id=child_id,
            division_event_id=event.event_id,
        )
        for event in events
        if isinstance(event, DivisionEvent)
        for child_id in event.child_ids
    ]
    if history.lineage is not None and history.lineage.parent_cell_id is not None:
        lineage_edges.append(
            LineageEdge(
                parent_subject_id=history.lineage.parent_cell_id,
                child_subject_id=history.subject_id,
            )
        )
    return EventGraph(
        event_ids=tuple(event.event_id for event in events),
        edges=tuple(edges),
        lineage_edges=tuple(lineage_edges),
    )
