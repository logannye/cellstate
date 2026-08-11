"""Aggregate request and inference controls for state estimation."""

from __future__ import annotations

import math
from itertools import pairwise

from pydantic import Field, field_validator, model_validator

from .belief import CellStateBelief
from .common import (
    SCHEMA_VERSION,
    SchemaModel,
    SchemaVersion,
    canonical_fingerprint,
    require_finite,
)
from .events import (
    EnvironmentEvent,
    InterventionEvent,
    MissingnessStatus,
    ObservationEvent,
    PopulationContext,
    StaticContext,
)
from .history import CellHistory, RecordCompleteness
from .query import MissingHistoryPolicy, StateQuery


class InferenceOptions(SchemaModel):
    seed: int = Field(default=0, ge=0)


def _environment_intervals(
    events: tuple[EnvironmentEvent, ...],
    key: str,
    *,
    upper_bound: float,
) -> tuple[tuple[float, float, str, str], ...]:
    """Return positive-duration value intervals for one canonical environment key."""

    normalized = key.casefold()
    intervals: list[tuple[float, float, str, str]] = []
    for event in events:
        value = next(
            (
                candidate
                for candidate_key, candidate in event.variables.items()
                if candidate_key.casefold() == normalized
            ),
            None,
        )
        if value is None or event.duration_seconds <= 0 or event.time_seconds >= upper_bound:
            continue
        interval_end = min(event.time_seconds + event.duration_seconds, upper_bound)
        if interval_end <= event.time_seconds:
            continue
        intervals.append(
            (
                event.time_seconds,
                interval_end,
                canonical_fingerprint({"value": value}),
                event.event_id,
            )
        )
    return tuple(intervals)


def _conflicting_environment_intervals(
    intervals: tuple[tuple[float, float, str, str], ...],
    *,
    lower_bound: float | None,
    upper_bound: float,
) -> tuple[str, str] | None:
    for index, left in enumerate(intervals):
        for right in intervals[index + 1 :]:
            interval_floor = -math.inf if lower_bound is None else lower_bound
            overlap_start = max(left[0], right[0], interval_floor)
            overlap_end = min(left[1], right[1], upper_bound)
            if overlap_start < overlap_end and left[2] != right[2]:
                return left[3], right[3]
    return None


def _interval_is_covered(
    intervals: tuple[tuple[float, float, str, str], ...],
    *,
    lower_bound: float,
    upper_bound: float,
) -> bool:
    if math.isclose(lower_bound, upper_bound, rel_tol=0, abs_tol=1e-12):
        return any(start <= lower_bound <= end and end > start for start, end, _, _ in intervals)

    cursor = lower_bound
    for start, end, _, _ in sorted(intervals):
        clipped_start = max(start, lower_bound)
        clipped_end = min(end, upper_bound)
        if clipped_end <= clipped_start:
            continue
        if clipped_start > cursor and not math.isclose(
            clipped_start, cursor, rel_tol=0, abs_tol=1e-12
        ):
            return False
        cursor = max(cursor, clipped_end)
        if cursor > upper_bound or math.isclose(cursor, upper_bound, rel_tol=0, abs_tol=1e-12):
            return True
    return False


def _overlapping_intervention_regimens(
    events: tuple[InterventionEvent, ...],
) -> tuple[tuple[InterventionEvent, ...], ...]:
    """Partition lifetime actions into the distinct regimens active at the same time."""

    def effective_end(event: InterventionEvent) -> float:
        return event.time_seconds + event.duration_seconds + (event.schedule.washout_seconds or 0)

    boundaries = sorted(
        {boundary for event in events for boundary in (event.time_seconds, effective_end(event))}
    )
    sample_times = {
        *boundaries,
        *(left + (right - left) / 2 for left, right in pairwise(boundaries)),
    }
    regimens: dict[tuple[str, ...], tuple[InterventionEvent, ...]] = {}
    for time in sample_times:
        active = tuple(
            event
            for event in events
            if (effective_end(event) == event.time_seconds and event.time_seconds == time)
            or (
                effective_end(event) > event.time_seconds
                and event.time_seconds <= time < effective_end(event)
            )
        )
        key = tuple(sorted(event.event_id for event in active))
        if key:
            regimens[key] = active
    return tuple(regimens.values())


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
        if not self.query.subject.supports(self.history.subject):
            raise ValueError("history subject does not satisfy the query subject specification")
        future = [
            event.event_id
            for event in self.history.events
            if (
                event.end_time_seconds
                if isinstance(event, ObservationEvent)
                else event.time_seconds
            )
            > self.as_of_seconds
        ]
        if future:
            raise ValueError(f"history contains events after as_of_seconds: {future}")

        observations = [
            event for event in self.history.events if isinstance(event, ObservationEvent)
        ]
        cutoff_equal = [
            event.event_id
            for event in observations
            if not self.query.evidence_policy.include_at_cutoff
            and event.end_time_seconds == self.as_of_seconds
        ]
        if cutoff_equal:
            raise ValueError(
                f"query evidence policy excludes observations ending at the cutoff: {cutoff_equal}"
            )
        lookback = self.query.evidence_policy.lookback_seconds
        if lookback is not None:
            lower_bound = self.as_of_seconds - lookback
            outside_window = [
                event.event_id for event in observations if event.end_time_seconds < lower_bound
            ]
            if outside_window:
                raise ValueError(
                    "history contains observations outside the query evidence window: "
                    f"{outside_window}"
                )
        allowed_modalities = {
            modality.key for modality in self.query.evidence_policy.allowed_modalities
        }
        unsupported_modalities = [
            event.event_id for event in observations if event.modality.key not in allowed_modalities
        ]
        if unsupported_modalities:
            raise ValueError(
                "history contains observation modalities outside the query evidence policy: "
                f"{unsupported_modalities}"
            )
        allowed_roles = set(self.query.evidence_policy.allowed_evidence_roles)
        unsupported_roles = [
            event.event_id for event in observations if event.evidence_role not in allowed_roles
        ]
        if unsupported_roles:
            raise ValueError(
                "history contains evidence roles outside the query evidence policy: "
                f"{unsupported_roles}"
            )
        observed_count = sum(
            event.missingness.status is MissingnessStatus.OBSERVED for event in observations
        )
        if observed_count < self.query.evidence_policy.minimum_observed_measurements:
            raise ValueError(
                "history does not meet the query minimum observed evidence requirement"
            )

        environment_events = tuple(
            event for event in self.history.events if isinstance(event, EnvironmentEvent)
        )
        unsupported_environment = [
            event.event_id
            for event in environment_events
            if not self.query.contains_environment_event(event)
        ]
        if unsupported_environment:
            raise ValueError(
                "history contains environment events outside the query domain: "
                f"{unsupported_environment}"
            )
        reject_if_missing = tuple(
            specification
            for specification in self.query.environment_space
            if specification.required
            and specification.missing_history_policy is MissingHistoryPolicy.REJECT
        )
        coverage_lower_bound: float | None = None
        if reject_if_missing:
            if lookback is not None:
                coverage_lower_bound = self.as_of_seconds - lookback
            elif self.previous_belief is not None:
                coverage_lower_bound = self.previous_belief.as_of_seconds
            else:
                raise ValueError(
                    "required environment coverage needs a finite query evidence lookback or "
                    "a previous-belief conditioning interval"
                )

        for specification in self.query.environment_space:
            key = specification.variable.key
            intervals = _environment_intervals(
                environment_events,
                key,
                upper_bound=self.as_of_seconds,
            )
            conflict = _conflicting_environment_intervals(
                intervals,
                lower_bound=coverage_lower_bound,
                upper_bound=self.as_of_seconds,
            )
            if conflict is not None:
                raise ValueError(
                    f"environment variable {key!r} has conflicting overlapping intervals: "
                    f"{list(conflict)}"
                )
            if (
                specification in reject_if_missing
                and coverage_lower_bound is not None
                and not _interval_is_covered(
                    intervals,
                    lower_bound=coverage_lower_bound,
                    upper_bound=self.as_of_seconds,
                )
            ):
                raise ValueError(
                    f"history does not cover required environment variable {key!r} across "
                    "the complete conditioning interval"
                )

        interventions = tuple(
            event for event in self.history.events if isinstance(event, InterventionEvent)
        )
        unsupported_interventions = [
            event.event_id for event in interventions if not self.query.contains_intervention(event)
        ]
        if unsupported_interventions:
            raise ValueError(
                "history contains interventions outside the query domain: "
                f"{unsupported_interventions}"
            )
        invalid_regimens = [
            tuple(event.event_id for event in regimen)
            for regimen in _overlapping_intervention_regimens(interventions)
            if not self.query.contains_intervention_combination(regimen)
        ]
        if invalid_regimens:
            raise ValueError(
                "history contains overlapping intervention regimens outside the query's bounded "
                f"combination policy: {invalid_regimens}"
            )

        completeness_requirements = (
            (
                self.query.constraints.require_complete_intervention_history,
                self.history.completeness.interventions,
                "intervention",
            ),
            (
                self.query.constraints.require_complete_environment_history,
                self.history.completeness.environments,
                "environment",
            ),
            (
                self.query.constraints.require_complete_lineage_history,
                self.history.completeness.lineage,
                "lineage",
            ),
            (
                self.query.constraints.require_complete_neighborhood_history,
                self.history.completeness.neighborhood,
                "neighborhood",
            ),
        )
        incomplete = [
            name
            for required, status, name in completeness_requirements
            if required and status is not RecordCompleteness.COMPLETE
        ]
        if incomplete:
            raise ValueError(
                "query requires complete history records for: " + ", ".join(incomplete)
            )
        if self.history.lineage is not None and any(
            time > self.as_of_seconds for time in self.history.lineage.division_times_seconds
        ):
            raise ValueError("lineage history contains a division after as_of_seconds")
        previous = self.previous_belief
        if previous is not None:
            if previous.subject != self.history.subject:
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
                and (
                    event.end_time_seconds
                    if isinstance(event, ObservationEvent)
                    else event.time_seconds
                )
                < previous.as_of_seconds
            ]
            if late_arrivals:
                raise ValueError(
                    "recursive updates cannot add events before the previous belief time without "
                    f"a smoothing backend: {late_arrivals}"
                )
        return self
