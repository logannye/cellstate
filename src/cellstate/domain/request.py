"""Aggregate request and inference controls for state estimation."""

from __future__ import annotations

from pydantic import Field, field_validator, model_validator

from .belief import CellStateBelief
from .common import (
    SCHEMA_VERSION,
    SchemaModel,
    SchemaVersion,
    canonical_fingerprint,
    require_finite,
)
from .events import PopulationContext, StaticContext
from .history import CellHistory
from .query import StateQuery


class InferenceOptions(SchemaModel):
    seed: int = Field(default=0, ge=0)
    strict_capabilities: bool = True


class EstimateCellStateRequest(SchemaModel):
    schema_version: SchemaVersion = SCHEMA_VERSION
    query: StateQuery
    history: CellHistory
    as_of_seconds: float
    static_context: StaticContext
    previous_belief: CellStateBelief | None = None
    population_context: PopulationContext | None = None

    @property
    def context_fingerprint(self) -> str:
        return canonical_fingerprint(
            {
                "static_context": self.static_context.model_dump(mode="json"),
                "population_context": (
                    self.population_context.model_dump(mode="json")
                    if self.population_context is not None
                    else None
                ),
            }
        )

    @field_validator("as_of_seconds")
    @classmethod
    def finite_time(cls, value: float) -> float:
        return require_finite(value, name="request as-of time")

    @model_validator(mode="after")
    def temporally_and_recursively_consistent(self) -> EstimateCellStateRequest:
        future = [
            event.event_id
            for event in self.history.events
            if event.time_seconds > self.as_of_seconds
        ]
        if future:
            raise ValueError(f"history contains events after as_of_seconds: {future}")
        if self.history.lineage is not None and any(
            time > self.as_of_seconds for time in self.history.lineage.division_times_seconds
        ):
            raise ValueError("lineage history contains a division after as_of_seconds")
        previous = self.previous_belief
        if previous is not None:
            if previous.subject_id != self.history.subject_id:
                raise ValueError("previous belief and request must refer to the same subject")
            if previous.as_of_seconds > self.as_of_seconds:
                raise ValueError("previous belief cannot be later than the requested state")
            if previous.query_fingerprint != self.query.fingerprint:
                raise ValueError("recursive updates require the same state query")
            if previous.context_fingerprint != self.context_fingerprint:
                raise ValueError("recursive updates require the same static/population context")
            if (
                previous.provenance.history_structure_fingerprint
                != self.history.structure_fingerprint
            ):
                raise ValueError(
                    "recursive updates cannot change lineage, neighborhood, or history "
                    "completeness without a smoothing/migration backend"
                )
            current_event_ids = {event.event_id for event in self.history.events}
            previous_event_ids = set(previous.provenance.source_event_ids)
            if missing := previous_event_ids - current_event_ids:
                raise ValueError(
                    f"recursive history omits events used by the previous belief: {sorted(missing)}"
                )
            events_by_id = {event.event_id: event for event in self.history.events}
            changed_events = [
                event_id
                for event_id, fingerprint in previous.provenance.source_event_fingerprints.items()
                if canonical_fingerprint(events_by_id[event_id]) != fingerprint
            ]
            if changed_events:
                raise ValueError(
                    "recursive updates cannot change previously assimilated event content: "
                    f"{sorted(changed_events)}"
                )
            late_arrivals = [
                event.event_id
                for event in self.history.events
                if event.event_id not in previous_event_ids
                and event.time_seconds < previous.as_of_seconds
            ]
            if late_arrivals:
                raise ValueError(
                    "recursive updates cannot add events before the previous belief time without "
                    f"a smoothing backend: {late_arrivals}"
                )
        return self
