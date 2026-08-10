"""Canonical cellular event history and completeness declarations."""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field, field_validator, model_validator

from .common import SCHEMA_VERSION, SchemaModel, SchemaVersion, canonical_fingerprint
from .events import (
    CellEvent,
    CollectionEffect,
    EvidenceRole,
    InterventionEvent,
    LineageHistory,
    MissingnessStatus,
    ObservationEvent,
    SpatialGraph,
)
from .subjects import BeliefSubject, SubjectKind


class RecordCompleteness(StrEnum):
    COMPLETE = "complete"
    INCOMPLETE = "incomplete"
    UNKNOWN = "unknown"


class HistoryCompleteness(SchemaModel):
    observations: RecordCompleteness = RecordCompleteness.UNKNOWN
    interventions: RecordCompleteness = RecordCompleteness.UNKNOWN
    environments: RecordCompleteness = RecordCompleteness.UNKNOWN
    lineage: RecordCompleteness = RecordCompleteness.UNKNOWN
    neighborhood: RecordCompleteness = RecordCompleteness.UNKNOWN


class CellHistory(SchemaModel):
    """A frozen-top-level timeline; empty is not complete unless declared complete."""

    schema_version: SchemaVersion = SCHEMA_VERSION
    subject: BeliefSubject
    events: tuple[CellEvent, ...] = ()
    lineage: LineageHistory | None = None
    neighborhood: SpatialGraph | None = None
    completeness: HistoryCompleteness = Field(default_factory=HistoryCompleteness)

    @field_validator("events")
    @classmethod
    def canonical_event_order(cls, events: tuple[CellEvent, ...]) -> tuple[CellEvent, ...]:
        return tuple(sorted(events, key=lambda event: (event.time_seconds, event.event_id)))

    @model_validator(mode="after")
    def consistent_subject_and_ids(self) -> CellHistory:
        event_ids = [event.event_id for event in self.events]
        if len(event_ids) != len(set(event_ids)):
            raise ValueError("event IDs must be unique within a history")
        mismatched = [event.event_id for event in self.events if event.subject != self.subject]
        if mismatched:
            raise ValueError(
                f"events must refer to history subject {self.subject_id!r}: {mismatched}"
            )
        if self.lineage is not None:
            if self.lineage.parent_cell_id == self.subject_id:
                raise ValueError("a lineage parent must differ from the history subject")
            if self.subject_id in self.lineage.sibling_cell_ids:
                raise ValueError("a lineage subject cannot also be its own sibling")
            if len(self.lineage.sibling_cell_ids) != len(set(self.lineage.sibling_cell_ids)):
                raise ValueError("lineage sibling IDs must be unique")
        known_event_ids = set(event_ids)
        events_by_id = {event.event_id: event for event in self.events}
        for event in self.events:
            if isinstance(event, InterventionEvent) and event.actual_perturbation is not None:
                missing_evidence = (
                    set(event.actual_perturbation.evidence_event_ids) - known_event_ids
                )
                if missing_evidence:
                    raise ValueError(
                        "actual perturbation evidence must reference events in the same history: "
                        f"{sorted(missing_evidence)}"
                    )
                invalid_evidence: list[str] = []
                for evidence_id in event.actual_perturbation.evidence_event_ids:
                    evidence = events_by_id[evidence_id]
                    if (
                        not isinstance(evidence, ObservationEvent)
                        or evidence.missingness.status is not MissingnessStatus.OBSERVED
                    ):
                        invalid_evidence.append(evidence_id)
                if invalid_evidence:
                    raise ValueError(
                        "actual perturbation evidence must reference observed measurement events: "
                        f"{sorted(invalid_evidence)}"
                    )
                premature_evidence = [
                    evidence_id
                    for evidence_id in event.actual_perturbation.evidence_event_ids
                    if events_by_id[evidence_id].time_seconds < event.time_seconds
                ]
                if premature_evidence:
                    raise ValueError(
                        "realized perturbation evidence cannot predate the intervention: "
                        f"{sorted(premature_evidence)}"
                    )
        if self.subject.kind is SubjectKind.INDIVIDUAL_CELL:
            terminal_direct_observations = [
                event
                for event in self.events
                if isinstance(event, ObservationEvent)
                and event.evidence_role is EvidenceRole.DIRECT
                and event.collection.effect is CollectionEffect.TERMINAL_DESTRUCTIVE
            ]
            for terminal in terminal_direct_observations:
                later_events = [
                    event.event_id
                    for event in self.events
                    if event.event_id != terminal.event_id
                    and not (
                        isinstance(event, ObservationEvent)
                        and event.evidence_role is not EvidenceRole.DIRECT
                    )
                    and (
                        event.time_seconds + getattr(event, "duration_seconds", 0)
                        > terminal.end_time_seconds
                    )
                ]
                if later_events:
                    raise ValueError(
                        "a terminal destructive observation closes an individual-cell history; "
                        f"later events are invalid: {sorted(later_events)}"
                    )
        return self

    @property
    def subject_id(self) -> str:
        """Convenience identifier; v2 serializes the complete typed subject."""

        return self.subject.subject_id

    @property
    def fingerprint(self) -> str:
        return canonical_fingerprint(self)

    @property
    def structure_fingerprint(self) -> str:
        """Fingerprint lineage, neighborhood, completeness, and subject apart from events."""

        return canonical_fingerprint(
            {
                "schema_version": self.schema_version,
                "subject": self.subject,
                "lineage": self.lineage,
                "neighborhood": self.neighborhood,
                "completeness": self.completeness,
            }
        )

    def through(self, time_seconds: float) -> tuple[CellEvent, ...]:
        return tuple(
            event
            for event in self.events
            if (
                event.end_time_seconds
                if isinstance(event, ObservationEvent)
                else event.time_seconds
            )
            <= time_seconds
        )

    def between(self, start: float, end: float) -> tuple[CellEvent, ...]:
        return tuple(
            event
            for event in self.events
            if start
            < (
                event.end_time_seconds
                if isinstance(event, ObservationEvent)
                else event.time_seconds
            )
            <= end
        )
