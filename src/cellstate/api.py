"""Fail-closed public boundaries for state, forecast, measurement, and control operations."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from pydantic import JsonValue

from .domain.belief import (
    BeliefStatus,
    CausalSupportReport,
    CellStateBelief,
    ContextBelief,
    _validate_causal_support_against_query,
    _validate_identified_evidence_provenance,
)
from .domain.common import (
    CausalStatus,
    CriterionOutcome,
    EvidenceStatus,
    ProvenanceRecord,
    Quantity,
    SchemaModel,
    SupportStatus,
    canonical_fingerprint,
)
from .domain.events import (
    CollectionEffect,
    EnvironmentEvent,
    EvidenceRole,
    InterventionEvent,
    MissingnessStatus,
    ObservationEvent,
)
from .domain.measurements import (
    MeasurementDecisionRequest,
    MeasurementDecisionStatus,
    MeasurementEvidenceCriterion,
    MeasurementInformationScope,
    MeasurementRecommendation,
    measurement_decision_set_fingerprint,
)
from .domain.query import AssayPurpose, AssaySpec, StateQuery
from .domain.request import (
    EstimateCellStateRequest,
    InferenceOptions,
    _conflicting_environment_intervals,
    _interval_is_covered,
)
from .domain.scenarios import (
    EvolutionScenario,
    InterventionObjective,
    InterventionPlan,
    StateForecast,
    TransportReport,
    TransportStatus,
)
from .domain.specification import CompiledStateSpecification
from .domain.subjects import SubjectKind
from .errors import CapabilityError, ContractViolationError, PosteriorCompatibilityError
from .ports import (
    CapabilityReport,
    CellStateEstimator,
    EstimatorDescriptor,
    InterventionPlanner,
    MeasurementCapabilityReport,
    MeasurementPolicy,
    ModelArtifactKind,
    QueryCompilerDescriptor,
    StateEvolutionModel,
    estimation_capability_scope_fingerprint,
    evolution_capability_scope_fingerprint,
    measurement_capability_scope_fingerprint,
    planning_capability_scope_fingerprint,
)


def _validated_public_model_descriptor(
    model: CellStateEstimator | StateEvolutionModel | InterventionPlanner | MeasurementPolicy,
) -> EstimatorDescriptor:
    """Reject biological runtimes until the admission registry is executable.

    The experimental bundle contract currently verifies only declarations.  It deliberately does
    not yet resolve model/code/evidence bytes, load the exact entry point, or derive query-specific
    conditional operation prerequisites.  A caller-constructed biological descriptor therefore
    cannot be accepted as an execution receipt merely because it satisfies a Python protocol.
    """

    try:
        descriptor = EstimatorDescriptor.model_validate(model.descriptor.model_dump(mode="python"))
    except (AttributeError, TypeError, ValueError) as error:
        raise ContractViolationError("model exposes an invalid estimator descriptor") from error
    if descriptor.artifact_kind is ModelArtifactKind.BIOLOGICAL_MODEL:
        raise CapabilityError(
            "biological runtime execution is disabled until the content-addressed admission "
            "registry resolves implementation, model, training, and validation bytes and "
            "verifies query-specific operation prerequisites"
        )
    return descriptor


def _require_compatible_posterior(
    belief: CellStateBelief,
    model: CellStateEstimator | StateEvolutionModel | InterventionPlanner | MeasurementPolicy,
) -> None:
    _require_provenance_compatible(belief.provenance, model)


def _require_provenance_compatible(
    provenance: ProvenanceRecord,
    model: CellStateEstimator | StateEvolutionModel | InterventionPlanner | MeasurementPolicy,
) -> None:
    descriptor = _validated_public_model_descriptor(model)
    expected = (
        descriptor.model_id,
        descriptor.model_version,
        descriptor.model_fingerprint,
        descriptor.posterior_schema_id,
        descriptor.support_envelope_id,
        descriptor.support_envelope_fingerprint,
        descriptor.training_support_id,
        descriptor.training_support_fingerprint,
        descriptor.validation_evidence_ids,
        descriptor.validation_evidence_fingerprints,
    )
    actual = (
        provenance.model_id,
        provenance.model_version,
        provenance.model_fingerprint,
        provenance.posterior_schema_id,
        provenance.support_envelope_id,
        provenance.support_envelope_fingerprint,
        provenance.training_support_id,
        provenance.training_support_fingerprint,
        provenance.validation_evidence_ids,
        provenance.validation_evidence_fingerprints,
    )
    if actual != expected:
        raise PosteriorCompatibilityError(
            "belief posterior was produced by an incompatible model, configuration, "
            "posterior schema, or training support"
        )


def _validated_capability_report(
    returned: CapabilityReport,
    *,
    expected_scope_fingerprint: str,
    operation: str,
) -> CapabilityReport:
    try:
        report = CapabilityReport.model_validate(returned.model_dump(mode="python"))
    except (AttributeError, TypeError, ValueError) as error:
        raise ContractViolationError(
            f"{operation} backend returned an invalid capability report"
        ) from error
    if report.scope_fingerprint != expected_scope_fingerprint:
        raise ContractViolationError(
            f"{operation} capability report is not bound to the exact requested scope"
        )
    if not report.supported or report.blockers:
        details = (*report.blockers, *report.notes)
        raise CapabilityError(
            f"{operation} capability preflight failed: "
            + ("; ".join(details) if details else "backend declared the scope unsupported")
        )
    return report


def _intervention_is_active(event: InterventionEvent, time_seconds: float) -> bool:
    """Return whether an interval or point intervention is active at one instant."""

    if event.duration_seconds == 0:
        return math.isclose(time_seconds, event.time_seconds, rel_tol=0, abs_tol=1e-12)
    return event.time_seconds <= time_seconds < event.time_seconds + event.duration_seconds


def _environment_is_active(event: EnvironmentEvent, time_seconds: float) -> bool:
    """Treat zero-duration environment records as points, never persistent assignments."""

    if event.duration_seconds == 0:
        return math.isclose(time_seconds, event.time_seconds, rel_tol=0, abs_tol=1e-12)
    return event.time_seconds <= time_seconds < event.time_seconds + event.duration_seconds


def _serialized_environment_value(value: Quantity | JsonValue) -> JsonValue:
    if isinstance(value, Quantity):
        return value.model_dump(mode="json")
    return value


def _derived_soluble_environment(
    events: Sequence[object],
    query: StateQuery,
    time_seconds: float,
    *,
    inherited: Mapping[str, JsonValue] | None = None,
) -> dict[str, JsonValue]:
    """Reduce exact active environment records into the canonical endpoint context."""

    latest: dict[str, tuple[float, JsonValue]] = {
        key.casefold(): (-math.inf, value) for key, value in (inherited or {}).items()
    }
    for event in events:
        if not isinstance(event, EnvironmentEvent) or not _environment_is_active(
            event, time_seconds
        ):
            continue
        for key, value in event.variables.items():
            normalized = key.casefold()
            if normalized not in latest or event.time_seconds >= latest[normalized][0]:
                latest[normalized] = (
                    event.time_seconds,
                    _serialized_environment_value(value),
                )
    for specification in query.environment_space:
        key = specification.variable.key.casefold()
        if key not in latest and specification.default_value is not None:
            latest[key] = (
                -math.inf,
                _serialized_environment_value(specification.default_value),
            )
    return {key: value for key, (_, value) in sorted(latest.items())}


def _require_derived_context(
    context: ContextBelief,
    *,
    active_interventions: tuple[InterventionEvent, ...],
    soluble_environment: Mapping[str, JsonValue],
    result_name: str,
    inherited_context: ContextBelief | None = None,
) -> None:
    """Bind deterministic context fields to the exact source events crossing the API."""

    if context.active_interventions != active_interventions:
        raise ContractViolationError(
            f"{result_name} active interventions are not derived from the exact source events"
        )
    if context.soluble_environment != dict(soluble_environment):
        raise ContractViolationError(
            f"{result_name} soluble environment is not derived from the exact source events"
        )
    inherited_maps = inherited_context or ContextBelief()
    for field_name in ("physical_environment", "neighborhood", "spatial_position"):
        if getattr(context, field_name) != getattr(inherited_maps, field_name):
            raise ContractViolationError(
                f"{result_name} {field_name.replace('_', ' ')} has no exact source derivation"
            )
    if (
        inherited_context is not None
        and context.unsupported_dimensions != inherited_context.unsupported_dimensions
    ):
        raise ContractViolationError(
            f"{result_name} changed inherited unsupported context dimensions"
        )


def _validated_measurement_capability_report(
    returned: MeasurementCapabilityReport,
    *,
    expected_scope_fingerprint: str,
    candidate_assay_ids: Sequence[str],
) -> MeasurementCapabilityReport:
    """Revalidate measurement preflight and require positive, exact support maps."""

    try:
        report = MeasurementCapabilityReport.model_validate(returned.model_dump(mode="python"))
    except (AttributeError, TypeError, ValueError) as error:
        raise ContractViolationError(
            "measurement policy returned an invalid capability report"
        ) from error
    if report.scope_fingerprint != expected_scope_fingerprint:
        raise ContractViolationError(
            "measurement capability report is not bound to the exact requested scope"
        )
    if tuple(report.assay_support) != tuple(candidate_assay_ids):
        raise ContractViolationError(
            "measurement capability report must cover the exact ordered candidate assay set"
        )
    if set(report.collection_effect_support) != set(CollectionEffect):
        raise ContractViolationError(
            "measurement capability report must explicitly assess every collection effect"
        )
    if not report.supported:
        details = (*report.blockers, *report.notes)
        raise CapabilityError(
            "measurement recommendation capability preflight failed: "
            + ("; ".join(details) if details else "policy declared the scope unsupported")
        )
    return report


def _validate_measurement_request_against_belief(
    belief: CellStateBelief,
    request: MeasurementDecisionRequest,
) -> dict[str, AssaySpec]:
    """Bind a measurement decision to one belief and its bounded future decision problem."""

    if request.parent_belief_id != belief.belief_id:
        raise ContractViolationError("measurement request is bound to a different belief")
    if request.query_fingerprint != belief.query_fingerprint:
        raise ContractViolationError("measurement request is bound to a different query")
    if request.collection_time_seconds < belief.as_of_seconds:
        raise ContractViolationError("measurement collection cannot predate the belief time")

    regime_fingerprints = {
        _measurement_candidate_regime_fingerprint(belief, candidate)
        for candidate in request.candidates
    }
    if len(regime_fingerprints) < 2:
        raise ContractViolationError(
            "measurement selection requires at least two semantically distinct candidate regimes"
        )

    horizons = {horizon.name: horizon for horizon in belief.query.prediction_horizons}
    horizon = horizons.get(request.objective.horizon_name)
    if horizon is None:
        raise ContractViolationError(
            "measurement objective horizon is not declared by the belief query"
        )
    outputs = {output.term.key: output for output in belief.query.target_outputs}
    for term in request.objective.terms:
        output = outputs.get(term.target.key)
        if output is None:
            raise ContractViolationError(
                f"measurement objective contains undeclared query target {term.target.key!r}"
            )
        if request.objective.horizon_name not in output.supported_horizon_names:
            raise ContractViolationError(
                f"measurement objective target {term.target.key!r} does not support horizon "
                f"{request.objective.horizon_name!r}"
            )
        if term.target_value is not None and term.target_value.units != output.units:
            raise ContractViolationError(
                f"measurement objective target value for {term.target.key!r} uses "
                "incompatible units"
            )

    if request.decision_deadline_seconds > belief.as_of_seconds:
        if (
            belief.context.active_interventions
            and len({candidate.inherit_active_interventions for candidate in request.candidates})
            > 1
        ):
            raise ContractViolationError(
                "candidate-dependent intervention inheritance would change treatment before the "
                "measurement decision deadline"
            )
        if (
            belief.context.soluble_environment
            and len({candidate.inherit_current_environment for candidate in request.candidates}) > 1
        ):
            raise ContractViolationError(
                "candidate-dependent environment inheritance would change context before the "
                "measurement decision deadline"
            )

    expected_duration = horizon.duration_seconds
    for candidate in request.candidates:
        if candidate.subject != belief.subject:
            raise ContractViolationError(
                "all measurement-decision candidates must refer to the belief's typed subject"
            )
        if candidate.start_time_seconds != belief.as_of_seconds:
            raise ContractViolationError(
                "all measurement-decision candidates must start at the belief time"
            )
        if candidate.horizon_name != request.objective.horizon_name or not math.isclose(
            candidate.end_time_seconds - candidate.start_time_seconds,
            expected_duration,
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            raise ContractViolationError(
                "measurement-decision candidate interval does not match the objective horizon"
            )
        if request.decision_deadline_seconds > candidate.end_time_seconds:
            raise ContractViolationError(
                "measurement decision deadline lies beyond a candidate horizon"
            )
        _validate_scenario_against_query(
            candidate,
            belief.query,
            belief.context.soluble_environment,
            belief.context.active_interventions,
        )
        candidate_events = (*candidate.interventions, *candidate.environments)
        if any(
            event.time_seconds < request.decision_deadline_seconds for event in candidate_events
        ):
            raise ContractViolationError(
                "candidate-dependent actions and environment changes cannot begin before the "
                "measurement decision deadline"
            )
        if {event.event_id for event in candidate_events} & set(belief.provenance.source_event_ids):
            raise ContractViolationError(
                "measurement candidate event IDs must not reuse belief source event IDs"
            )

    assays_by_id = {assay.assay_id: assay for assay in belief.query.available_assays}
    unknown_assays = set(request.candidate_assay_ids) - set(assays_by_id)
    if unknown_assays:
        raise ContractViolationError(
            f"measurement request names assays outside the query: {sorted(unknown_assays)}"
        )
    requested_assays = {
        assay_id: assays_by_id[assay_id] for assay_id in request.candidate_assay_ids
    }
    ineligible_assays = tuple(
        assay.assay_id
        for assay in requested_assays.values()
        if AssayPurpose.MEASUREMENT_SELECTION not in assay.purposes
    )
    if ineligible_assays:
        raise ContractViolationError(
            "measurement request names assays that do not declare measurement-selection "
            f"purpose: {list(ineligible_assays)}"
        )
    if len({assay.cost_units for assay in requested_assays.values()}) != 1:
        raise ContractViolationError(
            "one assay-cost conversion rate cannot compare assays with different cost units"
        )
    for assay in requested_assays.values():
        assert assay.cost is not None
        assert assay.cost_units is not None
        assert assay.turnaround_seconds is not None
        result_time = request.collection_time_seconds + assay.turnaround_seconds
        if result_time > request.decision_deadline_seconds:
            raise ContractViolationError(
                f"assay {assay.assay_id!r} cannot return before the decision deadline"
            )
        if assay.collection.effect is CollectionEffect.TERMINAL_DESTRUCTIVE:
            raise ContractViolationError(
                "a terminal destructive assay cannot inform a later decision for the same "
                "belief subject; aggregate subsampling must declare a partial sampling fraction"
            )
        if (
            belief.subject.kind is SubjectKind.INDIVIDUAL_CELL
            and assay.collection.effect
            is CollectionEffect.PARTIALLY_DESTRUCTIVE_POPULATION_SAMPLING
        ):
            raise ContractViolationError(
                "partial population sampling cannot be applied to an individual-cell belief"
            )
    return requested_assays


def _measurement_candidate_regime_fingerprint(
    belief: CellStateBelief,
    candidate: EvolutionScenario,
) -> str:
    """Fingerprint biological controls while excluding provenance-only event identifiers."""

    interventions = [
        {
            "time_seconds": event.time_seconds,
            "intervention_spec_id": event.intervention_spec_id,
            "intervention_type": event.intervention_type.key,
            "target": event.target.key if event.target is not None else None,
            "mechanism": event.mechanism.key if event.mechanism is not None else None,
            "dose": event.dose.model_dump(mode="json"),
            "duration_seconds": event.duration_seconds,
            "schedule": event.schedule.model_dump(mode="json"),
            "delivery_method": event.delivery_method.casefold(),
            "reversibility_status": event.reversibility_status.value,
        }
        for event in candidate.interventions
    ]
    environments = [
        {
            "time_seconds": event.time_seconds,
            "duration_seconds": event.duration_seconds,
            "variables": {
                key.casefold(): value
                for key, value in event.model_dump(mode="json")["variables"].items()
            },
            "temporal_mode": event.temporal_mode.value,
            "spatial_region": event.spatial_region,
        }
        for event in candidate.environments
    ]
    return canonical_fingerprint(
        {
            "interventions": sorted(interventions, key=canonical_fingerprint),
            "environments": sorted(environments, key=canonical_fingerprint),
            "inherit_active_interventions": (
                candidate.inherit_active_interventions
                if belief.context.active_interventions
                else None
            ),
            "inherit_current_environment": (
                candidate.inherit_current_environment
                if belief.context.soluble_environment
                else None
            ),
        }
    )


def _measurement_contrast_scope(
    belief: CellStateBelief,
    request: MeasurementDecisionRequest,
) -> tuple[set[str], set[str]]:
    """Return the exact action/environment contrast varied by the candidate set."""

    intervention_spec_ids = {
        event.intervention_spec_id
        for candidate in request.candidates
        for event in candidate.interventions
    }
    environment_keys = {
        key.casefold()
        for candidate in request.candidates
        for event in candidate.environments
        for key in event.variables
    }

    active_inheritance = {
        candidate.inherit_active_interventions for candidate in request.candidates
    }
    if len(active_inheritance) > 1:
        intervention_spec_ids.update(
            event.intervention_spec_id for event in belief.context.active_interventions
        )
    environment_inheritance = {
        candidate.inherit_current_environment for candidate in request.candidates
    }
    if len(environment_inheritance) > 1:
        environment_keys.update(key.casefold() for key in belief.context.soluble_environment)
    return intervention_spec_ids, environment_keys


def _validate_measurement_causal_support(
    recommendation: MeasurementRecommendation,
    belief: CellStateBelief,
    request: MeasurementDecisionRequest,
) -> None:
    """Require numerical EVSI to rest on the exact target and candidate causal estimand."""

    required_target_horizons = {
        (term.target.key, request.objective.horizon_name) for term in request.objective.terms
    }
    try:
        _validate_causal_support_against_query(
            recommendation.causal_support,
            belief.query,
            required_target_horizons=required_target_horizons,
        )
        _validate_identified_evidence_provenance(
            recommendation.causal_support,
            recommendation.provenance,
        )
    except ValueError as error:
        raise ContractViolationError(
            "measurement EVSI causal support is not bound to the exact decision estimand"
        ) from error

    intervention_spec_ids, environment_keys = _measurement_contrast_scope(belief, request)
    expected_decision_set = measurement_decision_set_fingerprint(request.candidates)
    for estimand in recommendation.causal_support.estimands:
        if estimand.scenario_id is not None or estimand.scenario_fingerprint is not None:
            raise ContractViolationError(
                "measurement EVSI causal estimands must bind the complete candidate set, not "
                "one scenario"
            )
        if (
            set(estimand.intervention_spec_ids) != intervention_spec_ids
            or {key.casefold() for key in estimand.environment_variable_keys} != environment_keys
        ):
            raise ContractViolationError(
                "measurement EVSI causal contrast does not match the candidate actions and "
                "environment changes"
            )
        if estimand.decision_set_fingerprint != expected_decision_set:
            raise ContractViolationError(
                "measurement EVSI causal support does not bind the exact ordered decision set"
            )


def _compile_query(
    estimator: CellStateEstimator,
    query: StateQuery,
) -> CompiledStateSpecification:
    compiler = estimator.query_compiler
    try:
        descriptor = QueryCompilerDescriptor.model_validate(
            compiler.compiler_descriptor.model_dump(mode="python")
        )
        returned = compiler.compile(query)
        state_specification = CompiledStateSpecification.model_validate(
            returned.model_dump(mode="python")
        )
    except (AttributeError, TypeError, ValueError) as error:
        raise ContractViolationError(
            "estimator query compiler returned an invalid compiled state specification"
        ) from error

    expected_binding = (
        query.fingerprint,
        query.subject,
        descriptor.compiler_id,
        descriptor.compiler_version,
        descriptor.compiler_fingerprint,
        tuple(output.term.key for output in query.target_outputs),
        tuple(horizon.name for horizon in query.prediction_horizons),
        query.evidence_policy.allowed_evidence_roles,
        query.acceptance_thresholds,
    )
    actual_binding = (
        state_specification.query_fingerprint,
        state_specification.subject,
        state_specification.compiler_id,
        state_specification.compiler_version,
        state_specification.compiler_fingerprint,
        state_specification.target_output_keys,
        state_specification.horizon_names,
        state_specification.admissible_evidence_roles,
        state_specification.acceptance_thresholds,
    )
    if actual_binding != expected_binding:
        raise ContractViolationError(
            "compiled state specification is not exactly bound to the query and compiler"
        )
    return state_specification


def _validate_provenance_evidence(
    provenance: ProvenanceRecord,
    events_by_id: Mapping[str, SchemaModel],
    *,
    result_name: str,
) -> None:
    unknown_source_ids = set(provenance.source_event_ids) - set(events_by_id)
    if unknown_source_ids:
        raise ContractViolationError(
            f"{result_name} provenance names unknown source events: {sorted(unknown_source_ids)}"
        )
    for event_id, fingerprint in provenance.source_event_fingerprints.items():
        if canonical_fingerprint(events_by_id[event_id]) != fingerprint:
            raise ContractViolationError(
                f"{result_name} provenance fingerprint disagrees for event {event_id!r}"
            )


def _scenario_environment_intervals(
    scenario: EvolutionScenario,
    key: str,
    inherited_environment: Mapping[str, JsonValue],
) -> tuple[tuple[float, float, str, str], ...]:
    normalized = key.casefold()
    intervals: list[tuple[float, float, str, str]] = []
    for event in scenario.environments:
        value = next(
            (
                candidate
                for candidate_key, candidate in event.variables.items()
                if candidate_key.casefold() == normalized
            ),
            None,
        )
        if value is None or event.duration_seconds <= 0:
            continue
        intervals.append(
            (
                event.time_seconds,
                event.time_seconds + event.duration_seconds,
                canonical_fingerprint({"value": value}),
                event.event_id,
            )
        )
    if scenario.inherit_current_environment:
        inherited_value = next(
            (
                value
                for inherited_key, value in inherited_environment.items()
                if inherited_key.casefold() == normalized
            ),
            None,
        )
        if inherited_value is not None:
            first_assignment = min(
                (start for start, _, _, _ in intervals),
                default=scenario.end_time_seconds,
            )
            if first_assignment > scenario.start_time_seconds:
                intervals.append(
                    (
                        scenario.start_time_seconds,
                        first_assignment,
                        canonical_fingerprint({"value": inherited_value}),
                        "inherited-current-environment",
                    )
                )
    return tuple(intervals)


def _validate_scenario_against_query(
    scenario: EvolutionScenario,
    query: StateQuery,
    inherited_environment: Mapping[str, JsonValue],
    active_interventions: Sequence[InterventionEvent],
) -> None:
    if scenario.interventions and not query.contains_intervention_combination(
        scenario.interventions
    ):
        raise ContractViolationError(
            "scenario intervention set is outside the query's bounded action space"
        )

    if active_interventions and scenario.inherit_active_interventions is None:
        raise ContractViolationError(
            "scenario must explicitly inherit or clear active interventions from the belief"
        )
    if scenario.inherit_active_interventions:
        effective_interventions = (*active_interventions, *scenario.interventions)
        if effective_interventions and not query.contains_intervention_combination(
            effective_interventions
        ):
            raise ContractViolationError(
                "inherited and planned interventions exceed the query's bounded action space"
            )

    if not all(query.contains_environment_event(event) for event in scenario.environments):
        raise ContractViolationError(
            "scenario environment is outside the query's bounded environment space"
        )
    if inherited_environment and scenario.inherit_current_environment is None:
        raise ContractViolationError(
            "scenario must explicitly inherit or clear the current environment from the belief"
        )

    if scenario.inherit_current_environment:
        for key, value in inherited_environment.items():
            if not query.contains_environment_value(key, value):
                raise ContractViolationError(
                    f"inherited environment variable {key!r} is outside the query domain"
                )

    missing_environment: list[str] = []
    for specification in query.environment_space:
        key = specification.variable.key
        intervals = _scenario_environment_intervals(scenario, key, inherited_environment)
        conflict = _conflicting_environment_intervals(
            intervals,
            lower_bound=scenario.start_time_seconds,
            upper_bound=scenario.end_time_seconds,
        )
        if conflict is not None:
            raise ContractViolationError(
                f"scenario environment variable {key!r} has conflicting overlapping intervals: "
                f"{list(conflict)}"
            )
        inherits_current_value = scenario.inherit_current_environment and any(
            inherited_key.casefold() == key.casefold() for inherited_key in inherited_environment
        )
        if (specification.required or inherits_current_value) and not _interval_is_covered(
            intervals,
            lower_bound=scenario.start_time_seconds,
            upper_bound=scenario.end_time_seconds,
        ):
            missing_environment.append(key)
    if missing_environment:
        if scenario.inherit_current_environment:
            raise ContractViolationError(
                "inherited and planned environment does not cover required or inherited query "
                f"variables across the complete scenario interval: {missing_environment}"
            )
        raise ContractViolationError(
            "scenario does not supply all required query environment variables across the "
            f"complete scenario interval: {missing_environment}"
        )


def _validate_causal_support_against_scenario(
    report: CausalSupportReport,
    scenario: EvolutionScenario,
    active_interventions: Sequence[InterventionEvent],
) -> None:
    """Bind an identified scenario claim to its exact effective intervention contrast."""

    if not report.estimands:
        return
    inherited = (
        tuple(
            event
            for event in active_interventions
            if _intervention_is_active(event, scenario.start_time_seconds)
        )
        if scenario.inherit_active_interventions
        else ()
    )
    effective_interventions = (*inherited, *scenario.interventions)
    effective_spec_ids = {event.intervention_spec_id for event in effective_interventions}
    explicit_environment_keys = {
        key.casefold() for event in scenario.environments for key in event.variables
    }
    scenario_fingerprint = canonical_fingerprint(scenario)
    for estimand in report.estimands:
        if (
            estimand.scenario_id != scenario.scenario_id
            or estimand.scenario_fingerprint != scenario_fingerprint
        ):
            raise ContractViolationError(
                "causal estimand is not bound to the exact concrete scenario"
            )
        if set(estimand.intervention_spec_ids) != effective_spec_ids:
            raise ContractViolationError(
                "causal estimand intervention contrast does not match the scenario actions"
            )
        if (
            not {key.casefold() for key in estimand.environment_variable_keys}
            <= explicit_environment_keys
        ):
            raise ContractViolationError(
                "causal estimand environment contrast does not match explicit scenario changes"
            )


def _validate_planning_transport_against_query(
    report: CausalSupportReport,
    transport: TransportReport,
    query: StateQuery,
) -> None:
    """Reject causal transport claims that the exact query did not authorize."""

    if report.causal_status is CausalStatus.IDENTIFIED_POPULATION_EFFECT:
        if transport.status is not TransportStatus.WITHIN_SUPPORT:
            raise ValueError("identified intervention support must remain within support")
        return
    if report.causal_status is not CausalStatus.TRANSPORTED_UNDER_ASSUMPTIONS:
        return
    if not query.constraints.allow_transport:
        raise ValueError("query constraints forbid transported intervention support")
    if transport.status is not TransportStatus.TRANSPORTED:
        raise ValueError("transported intervention support requires transported status")
    if (
        transport.source_domain != report.source_scope
        or transport.target_domain != report.target_scope
    ):
        raise ValueError("intervention causal scopes and transport domains must agree")


def estimate_cell_state(
    request: EstimateCellStateRequest,
    *,
    estimator: CellStateEstimator,
    options: InferenceOptions | None = None,
) -> CellStateBelief:
    """Estimate a query-conditioned belief only after exact compilation and preflight."""

    descriptor = _validated_public_model_descriptor(estimator)
    resolved_options = options or InferenceOptions()
    state_specification = _compile_query(estimator, request.query)
    if request.previous_belief is not None:
        _require_compatible_posterior(request.previous_belief, estimator)
        if request.previous_belief.state_specification != state_specification:
            raise PosteriorCompatibilityError(
                "previous belief was produced under a different compiled state specification"
            )

    expected_scope = estimation_capability_scope_fingerprint(request, state_specification)
    _validated_capability_report(
        estimator.capabilities(request, state_specification),
        expected_scope_fingerprint=expected_scope,
        operation="state estimation",
    )

    returned_belief = estimator.estimate(request, options=resolved_options)
    try:
        belief = CellStateBelief.model_validate(returned_belief.model_dump(mode="python"))
    except (AttributeError, TypeError, ValueError) as error:
        raise ContractViolationError("estimator returned an invalid belief contract") from error

    if descriptor.artifact_kind is ModelArtifactKind.EMPIRICAL_OBSERVATION_MODEL and (
        belief.diagnostics.causal_support.causal_status
        in {
            CausalStatus.IDENTIFIED_POPULATION_EFFECT,
            CausalStatus.TRANSPORTED_UNDER_ASSUMPTIONS,
        }
    ):
        raise CapabilityError(
            "an empirical observation model cannot claim an identified or transported population "
            "effect; identification is gated by the content-addressed admission registry"
        )

    if belief.subject != request.history.subject:
        raise ContractViolationError("estimator returned a belief for the wrong typed subject")
    if not belief.subject.is_compatible_with(request.query.subject):
        raise ContractViolationError("estimator returned a subject outside the query estimand")
    if belief.as_of_seconds != request.as_of_seconds:
        raise ContractViolationError("estimator returned a belief at the wrong time")
    if belief.query != request.query or belief.query_fingerprint != request.query.fingerprint:
        raise ContractViolationError("estimator returned a belief for a different query")
    if belief.state_specification != state_specification:
        raise ContractViolationError(
            "estimator returned a belief under a different compiled state specification"
        )
    if belief.history_fingerprint != request.history.fingerprint:
        raise ContractViolationError("estimator returned a belief for a different history")
    if belief.context_fingerprint != request.context_fingerprint:
        raise ContractViolationError(
            "estimator returned a belief for different static/population context"
        )
    _require_provenance_compatible(belief.provenance, estimator)
    if belief.provenance.seed != resolved_options.seed:
        raise ContractViolationError("estimator returned provenance with the wrong inference seed")
    if belief.provenance.history_structure_fingerprint != request.history.structure_fingerprint:
        raise ContractViolationError(
            "estimator returned provenance for different history structure"
        )

    events_by_id = {event.event_id: event for event in request.history.events}
    expected_active_interventions = tuple(
        event
        for event in request.history.events
        if isinstance(event, InterventionEvent)
        and _intervention_is_active(event, request.as_of_seconds)
    )
    _require_derived_context(
        belief.context,
        active_interventions=expected_active_interventions,
        soluble_environment=_derived_soluble_environment(
            request.history.events,
            request.query,
            request.as_of_seconds,
        ),
        result_name="belief",
    )
    _validate_provenance_evidence(
        belief.provenance,
        events_by_id,
        result_name="belief",
    )
    omitted_event_ids = set(events_by_id) - set(belief.provenance.source_event_ids)
    if omitted_event_ids:
        raise ContractViolationError(
            f"belief provenance omits eligible request history events: {sorted(omitted_event_ids)}"
        )
    observations_by_id = {
        event_id: event
        for event_id, event in events_by_id.items()
        if isinstance(event, ObservationEvent)
    }
    realization_gaps: dict[str, tuple[str, ...]] = {}
    for event in request.history.events:
        if not isinstance(event, InterventionEvent):
            continue
        gaps = request.query.realization_evidence_gaps(event, observations_by_id)
        if gaps:
            realization_gaps[event.event_id] = gaps
    if realization_gaps and belief.readiness.causal is CriterionOutcome.PASSED:
        raise ContractViolationError(
            "belief cannot claim causal/control readiness while required historical "
            f"intervention-realization evidence is unresolved: {realization_gaps}"
        )
    for realization in belief.intervention_realizations:
        intervention = events_by_id.get(realization.intervention_event_id)
        if not isinstance(intervention, InterventionEvent):
            raise ContractViolationError(
                "an intervention-realization posterior must reference a historical "
                "intervention event"
            )
        realization_evidence = [
            events_by_id[event_id] for event_id in realization.evidence_event_ids
        ]
        if any(
            not isinstance(event, ObservationEvent)
            or event.missingness.status is not MissingnessStatus.OBSERVED
            or event.time_seconds < intervention.time_seconds
            for event in realization_evidence
        ):
            raise ContractViolationError(
                "intervention-realization evidence must reference observed measurements "
                "collected at or after intervention start"
            )
    for factor in belief.factors:
        if factor.evidence_status is not EvidenceStatus.OBSERVED:
            continue
        evidence = [events_by_id[event_id] for event_id in factor.evidence_event_ids]
        if any(
            not isinstance(event, ObservationEvent)
            or event.missingness.status is not MissingnessStatus.OBSERVED
            for event in evidence
        ):
            raise ContractViolationError(
                "observed factor evidence must reference observed measurement events"
            )
        if not any(
            isinstance(event, ObservationEvent)
            and event.evidence_role is EvidenceRole.DIRECT
            and event.evidence_link.target_subject == belief.subject
            and math.isclose(
                event.end_time_seconds,
                belief.as_of_seconds,
                rel_tol=0,
                abs_tol=1e-12,
            )
            for event in evidence
        ):
            raise ContractViolationError(
                "an observed factor requires direct linked evidence ending at the belief time"
            )

    if belief.status is BeliefStatus.UNAVAILABLE and belief.readiness.valid_for_prediction:
        raise ContractViolationError("an unavailable belief cannot claim prediction readiness")
    return belief


def evolve_cell_state(
    belief: CellStateBelief,
    *,
    scenario: EvolutionScenario,
    evolution_model: StateEvolutionModel,
    options: InferenceOptions | None = None,
) -> StateForecast:
    """Propagate a compatible posterior through one bounded controlled scenario."""

    resolved_options = options or InferenceOptions()
    if scenario.subject != belief.subject:
        raise ContractViolationError("scenario and belief must refer to the same typed subject")
    if scenario.start_time_seconds != belief.as_of_seconds:
        raise ContractViolationError("scenario must start at the belief time")
    _require_compatible_posterior(belief, evolution_model)
    _validate_scenario_against_query(
        scenario,
        belief.query,
        belief.context.soluble_environment,
        belief.context.active_interventions,
    )
    scenario_events = (*scenario.interventions, *scenario.environments)
    scenario_event_ids = [event.event_id for event in scenario_events]
    if set(scenario_event_ids) & set(belief.provenance.source_event_ids):
        raise ContractViolationError("scenario event IDs must not reuse belief source event IDs")
    horizons = {horizon.name: horizon for horizon in belief.query.prediction_horizons}
    if scenario.horizon_name not in horizons:
        raise ContractViolationError("scenario horizon is not declared by the belief query")
    if not math.isclose(
        scenario.end_time_seconds - scenario.start_time_seconds,
        horizons[scenario.horizon_name].duration_seconds,
        rel_tol=1e-9,
        abs_tol=1e-9,
    ):
        raise ContractViolationError("scenario duration does not match its named query horizon")

    expected_scope = evolution_capability_scope_fingerprint(belief, scenario)
    _validated_capability_report(
        evolution_model.capabilities(belief, scenario),
        expected_scope_fingerprint=expected_scope,
        operation="state evolution",
    )
    returned_forecast = evolution_model.evolve(belief, scenario, options=resolved_options)
    try:
        forecast = StateForecast.model_validate(returned_forecast.model_dump(mode="python"))
    except (AttributeError, TypeError, ValueError) as error:
        raise ContractViolationError("evolution model returned an invalid forecast") from error

    if forecast.subject != belief.subject or forecast.scenario_id != scenario.scenario_id:
        raise ContractViolationError("evolution model returned a forecast for the wrong scenario")
    if forecast.query != belief.query or forecast.query_fingerprint != belief.query_fingerprint:
        raise ContractViolationError("forecast is not bound to the belief query")
    if forecast.state_specification != belief.state_specification:
        raise ContractViolationError("forecast changed the compiled state specification")
    if (
        forecast.parent_belief_id != belief.belief_id
        or forecast.scenario_fingerprint != canonical_fingerprint(scenario)
    ):
        raise ContractViolationError("forecast is not bound to the input belief and scenario")
    if (
        forecast.start_time_seconds != scenario.start_time_seconds
        or forecast.end_time_seconds != scenario.end_time_seconds
        or forecast.horizon_name != scenario.horizon_name
    ):
        raise ContractViolationError("forecast time interval does not match the scenario")
    if (
        forecast.provenance.history_fingerprint != belief.history_fingerprint
        or forecast.provenance.context_fingerprint != belief.context_fingerprint
        or forecast.provenance.history_structure_fingerprint
        != belief.provenance.history_structure_fingerprint
    ):
        raise ContractViolationError("forecast provenance does not match the input belief")
    _require_provenance_compatible(forecast.provenance, evolution_model)
    if forecast.provenance.seed != resolved_options.seed:
        raise ContractViolationError("forecast provenance has the wrong inference seed")

    expected_event_fingerprints = {
        **belief.provenance.source_event_fingerprints,
        **{event.event_id: canonical_fingerprint(event) for event in scenario_events},
    }
    if forecast.provenance.source_event_fingerprints != expected_event_fingerprints:
        raise ContractViolationError("forecast provenance does not bind the complete evidence set")
    forecast_events: dict[str, SchemaModel] = {event.event_id: event for event in scenario_events}
    # Historical event payloads are not embedded in the belief; their content-addressed hashes
    # were already validated when the belief crossed the estimation boundary.
    if not set(forecast.provenance.source_event_ids) == set(expected_event_fingerprints):
        raise ContractViolationError("forecast provenance source IDs are incomplete")
    scenario_only_provenance = forecast.provenance.model_copy(
        update={
            "source_event_ids": tuple(forecast_events),
            "source_event_fingerprints": {
                event_id: forecast.provenance.source_event_fingerprints[event_id]
                for event_id in forecast_events
            },
        }
    )
    _validate_provenance_evidence(
        scenario_only_provenance,
        forecast_events,
        result_name="forecast",
    )

    inherited_active_interventions = (
        tuple(
            event
            for event in belief.context.active_interventions
            if _intervention_is_active(event, scenario.end_time_seconds)
        )
        if scenario.inherit_active_interventions
        else ()
    )
    scenario_active_interventions = tuple(
        event
        for event in scenario.interventions
        if _intervention_is_active(event, scenario.end_time_seconds)
    )
    inherited_environment = (
        belief.context.soluble_environment if scenario.inherit_current_environment else None
    )
    _require_derived_context(
        forecast.context,
        active_interventions=(
            *inherited_active_interventions,
            *scenario_active_interventions,
        ),
        soluble_environment=_derived_soluble_environment(
            scenario.environments,
            belief.query,
            scenario.end_time_seconds,
            inherited=inherited_environment,
        ),
        result_name="forecast",
        inherited_context=belief.context,
    )

    allowed_realization_ids = {
        *(item.intervention_event_id for item in belief.intervention_realizations),
        *(event.event_id for event in belief.context.active_interventions),
        *(event.event_id for event in scenario.interventions),
    }
    forecast_realization_ids = {
        item.intervention_event_id for item in forecast.intervention_realizations
    }
    if not forecast_realization_ids <= allowed_realization_ids:
        raise ContractViolationError(
            "forecast intervention realizations do not resolve to belief or scenario interventions"
        )

    evidence_ids = {
        *forecast.transport.evidence_ids,
        *(
            evidence_id
            for prediction in forecast.target_predictions
            for evidence_id in prediction.transport.evidence_ids
        ),
    }
    if not evidence_ids <= forecast.provenance.scientific_evidence_ids:
        raise ContractViolationError("forecast transport evidence is absent from provenance")
    if forecast.readiness.valid_for_prediction and any(
        prediction.status is not SupportStatus.SUPPORTED
        for prediction in forecast.target_predictions
    ):
        raise ContractViolationError(
            "a prediction-ready forecast cannot contain unsupported target predictions"
        )
    _validate_causal_support_against_scenario(
        forecast.diagnostics.causal_support,
        scenario,
        belief.context.active_interventions,
    )
    return forecast


def recommend_next_measurement(
    belief: CellStateBelief,
    *,
    request: MeasurementDecisionRequest,
    policy: MeasurementPolicy,
    options: InferenceOptions | None = None,
) -> MeasurementRecommendation:
    """Recommend or abstain over assays using validated intervention-decision EVSI."""

    requested_assays = _validate_measurement_request_against_belief(belief, request)
    _require_compatible_posterior(belief, policy)
    expected_scope = measurement_capability_scope_fingerprint(belief, request)
    capability = _validated_measurement_capability_report(
        policy.capabilities(belief, request),
        expected_scope_fingerprint=expected_scope,
        candidate_assay_ids=request.candidate_assay_ids,
    )
    resolved_options = options or InferenceOptions()
    returned_recommendation = policy.recommend(
        belief,
        request,
        options=resolved_options,
    )
    try:
        recommendation = MeasurementRecommendation.model_validate(
            returned_recommendation.model_dump(mode="python")
        )
    except (AttributeError, TypeError, ValueError) as error:
        raise ContractViolationError(
            "measurement policy returned an invalid recommendation contract"
        ) from error

    expected_candidates = tuple(
        (candidate.scenario_id, canonical_fingerprint(candidate))
        for candidate in request.candidates
    )
    actual_candidates = tuple(
        (candidate.scenario_id, candidate.fingerprint) for candidate in recommendation.candidates
    )
    expected_assays = tuple(
        (assay_id, canonical_fingerprint(requested_assays[assay_id]))
        for assay_id in request.candidate_assay_ids
    )
    actual_assays = tuple((assay.assay_id, assay.fingerprint) for assay in recommendation.assays)
    expected_binding = (
        belief.belief_id,
        belief.query_fingerprint,
        request.request_id,
        request.fingerprint,
        request.objective.objective_id,
        canonical_fingerprint(request.objective),
        expected_candidates,
        expected_assays,
        request.minimum_net_decision_value,
        request.utility_units,
    )
    actual_binding = (
        recommendation.parent_belief_id,
        recommendation.query_fingerprint,
        recommendation.request_id,
        recommendation.request_fingerprint,
        recommendation.objective_id,
        recommendation.objective_fingerprint,
        actual_candidates,
        actual_assays,
        recommendation.minimum_net_decision_value,
        recommendation.utility_units,
    )
    if actual_binding != expected_binding:
        raise ContractViolationError(
            "measurement recommendation is not exactly bound to the belief, request, "
            "objective, candidates, assays, and decision threshold"
        )
    if recommendation.readiness != belief.readiness:
        raise ContractViolationError(
            "measurement recommendation changed the input belief's readiness assessment"
        )

    provenance = recommendation.provenance
    expected_provenance_binding = (
        belief.history_fingerprint,
        belief.context_fingerprint,
        belief.provenance.history_structure_fingerprint,
        belief.provenance.source_event_ids,
        belief.provenance.source_event_fingerprints,
    )
    actual_provenance_binding = (
        provenance.history_fingerprint,
        provenance.context_fingerprint,
        provenance.history_structure_fingerprint,
        provenance.source_event_ids,
        provenance.source_event_fingerprints,
    )
    if actual_provenance_binding != expected_provenance_binding:
        raise ContractViolationError(
            "measurement recommendation provenance does not match the input belief"
        )
    _require_provenance_compatible(provenance, policy)
    if recommendation.seed != resolved_options.seed or provenance.seed != resolved_options.seed:
        raise ContractViolationError(
            "measurement recommendation provenance has the wrong inference seed"
        )

    scientific_criteria = {
        MeasurementEvidenceCriterion.ASSAY_OUTCOME_MODEL: capability.assay_outcome_model,
        MeasurementEvidenceCriterion.HYPOTHETICAL_UPDATE: capability.hypothetical_update,
        MeasurementEvidenceCriterion.EXACT_CANDIDATE_COUNTERFACTUAL_REPLANNING: (
            capability.counterfactual_replanning
        ),
        MeasurementEvidenceCriterion.DECISION_UTILITY: capability.decision_utility,
    }
    supported_evaluations = []
    for evaluation in recommendation.evaluations:
        assay = requested_assays[evaluation.assay.assay_id]
        if evaluation.collection_effect is not assay.collection.effect:
            raise ContractViolationError(
                f"assay {assay.assay_id!r} evaluation changed its collection effect"
            )
        for trace in evaluation.evidence_traces:
            if trace.scope_fingerprint != expected_scope:
                raise ContractViolationError(
                    "measurement criterion evidence is not bound to the exact request scope"
                )
            if trace.outcome is not scientific_criteria[trace.criterion]:
                raise ContractViolationError(
                    "measurement criterion evidence disagrees with capability outcomes"
                )
            trace_evidence_ids = set(trace.evidence_ids)
            if not trace_evidence_ids <= set(policy.descriptor.validation_evidence_ids):
                raise ContractViolationError(
                    "measurement criterion evidence is outside the policy descriptor"
                )
            expected_trace_fingerprints = {
                evidence_id: policy.descriptor.validation_evidence_fingerprints[evidence_id]
                for evidence_id in trace_evidence_ids
            }
            if trace.evidence_fingerprints != expected_trace_fingerprints:
                raise ContractViolationError(
                    "measurement criterion evidence fingerprints disagree with the policy "
                    "descriptor"
                )

        assay_support = capability.assay_support[assay.assay_id]
        collection_support = capability.collection_effect_support[assay.collection.effect]
        criterion_outcomes = tuple(
            scientific_criteria[trace.criterion] for trace in evaluation.evidence_traces
        )
        if (
            SupportStatus.UNSUPPORTED in {assay_support, collection_support}
            or CriterionOutcome.UNSUPPORTED in criterion_outcomes
            or CriterionOutcome.FAILED in criterion_outcomes
        ):
            expected_evaluation_status = SupportStatus.UNSUPPORTED
        elif (
            SupportStatus.NOT_EVALUATED in {assay_support, collection_support}
            or CriterionOutcome.NOT_EVALUATED in criterion_outcomes
        ):
            expected_evaluation_status = SupportStatus.NOT_EVALUATED
        else:
            expected_evaluation_status = SupportStatus.SUPPORTED
        if evaluation.status is not expected_evaluation_status:
            raise ContractViolationError(
                f"assay {assay.assay_id!r} evaluation status disagrees with its explicit "
                "assay, collection-effect, or criterion support"
            )

        explicit_support_reasons = {
            *(
                (f"assay_support:{assay.assay_id}:{assay_support.value}",)
                if assay_support is not SupportStatus.SUPPORTED
                else ()
            ),
            *(
                (
                    "collection_effect_support:"
                    f"{assay.collection.effect.value}:{collection_support.value}",
                )
                if collection_support is not SupportStatus.SUPPORTED
                else ()
            ),
        }
        if not explicit_support_reasons <= set(evaluation.reasons):
            raise ContractViolationError(
                f"assay {assay.assay_id!r} evaluation reasons do not bind its explicit "
                "assay and collection-effect support blockers"
            )
        if evaluation.status is not SupportStatus.SUPPORTED:
            continue
        supported_evaluations.append(evaluation)

        if not belief.readiness.valid_for_measurement_selection:
            raise ContractViolationError(
                "numeric assay value requires measurement-selection readiness"
            )
        if capability.assay_support[assay.assay_id] is not SupportStatus.SUPPORTED:
            raise ContractViolationError(
                f"numeric assay value exceeds declared support for {assay.assay_id!r}"
            )
        if (
            capability.collection_effect_support[assay.collection.effect]
            is not SupportStatus.SUPPORTED
        ):
            raise ContractViolationError(
                "numeric assay value exceeds declared collection-effect support"
            )
        if any(outcome is not CriterionOutcome.PASSED for outcome in scientific_criteria.values()):
            raise ContractViolationError(
                "numeric EVSI requires passing assay-outcome, hypothetical-update, "
                "counterfactual-replanning, and decision-utility support"
            )
        if evaluation.information_scope not in {
            MeasurementInformationScope.INTERVENTION_OUTCOMES,
            MeasurementInformationScope.DECISION_REGRET,
        }:
            raise ContractViolationError(
                "generic query-target or covariance information cannot count as decision EVSI"
            )

        assert assay.cost is not None
        assert assay.cost_units is not None
        assert assay.turnaround_seconds is not None
        expected_delay = (
            request.collection_time_seconds - belief.as_of_seconds + assay.turnaround_seconds
        )
        assert evaluation.raw_assay_cost is not None
        assert evaluation.assay_cost_penalty is not None
        assert evaluation.expected_delay_seconds is not None
        assert evaluation.delay_cost is not None
        assert evaluation.destructiveness_cost is not None
        if (
            any(
                not math.isclose(actual, expected, rel_tol=0, abs_tol=1e-9)
                for actual, expected in zip(
                    (
                        evaluation.raw_assay_cost,
                        evaluation.assay_cost_penalty,
                        evaluation.expected_delay_seconds,
                        evaluation.delay_cost,
                        evaluation.destructiveness_cost,
                    ),
                    (
                        assay.cost,
                        assay.cost * request.assay_cost_to_utility_rate,
                        expected_delay,
                        expected_delay * request.delay_penalty_per_second,
                        request.destructiveness_penalties[assay.collection.effect],
                    ),
                    strict=True,
                )
            )
            or evaluation.assay_cost_units != assay.cost_units
        ):
            raise ContractViolationError(
                f"assay {assay.assay_id!r} evaluation changed its cost, delay, or "
                "destructiveness economics"
            )

        evidence_ids = set(evaluation.measurement_model_evidence_ids)
        if not evidence_ids <= set(policy.descriptor.validation_evidence_ids):
            raise ContractViolationError(
                "numeric assay value cites measurement-model evidence outside the policy descriptor"
            )
        expected_evidence_fingerprints = {
            evidence_id: policy.descriptor.validation_evidence_fingerprints[evidence_id]
            for evidence_id in evidence_ids
        }
        if evaluation.measurement_model_evidence_fingerprints != expected_evidence_fingerprints:
            raise ContractViolationError(
                "measurement-model evidence fingerprints disagree with the policy descriptor"
            )

    if supported_evaluations:
        if (
            recommendation.causal_status
            not in {
                CausalStatus.IDENTIFIED_POPULATION_EFFECT,
                CausalStatus.TRANSPORTED_UNDER_ASSUMPTIONS,
            }
            or recommendation.causal_support.outcome is not CriterionOutcome.PASSED
        ):
            raise ContractViolationError(
                "numeric intervention-decision EVSI requires passing identified causal support"
            )
        if (
            recommendation.causal_status is CausalStatus.IDENTIFIED_POPULATION_EFFECT
            and recommendation.transport.status is not TransportStatus.WITHIN_SUPPORT
        ):
            raise ContractViolationError(
                "an identified within-domain EVSI requires within-support transport"
            )
        if recommendation.causal_status is CausalStatus.TRANSPORTED_UNDER_ASSUMPTIONS:
            if not belief.query.constraints.allow_transport:
                raise ContractViolationError(
                    "query constraints do not allow transported measurement EVSI"
                )
            if recommendation.transport.status is not TransportStatus.TRANSPORTED:
                raise ContractViolationError("transported EVSI requires a transported result")
            if set(recommendation.causal_support.transport_assumptions) != set(
                recommendation.transport.assumptions
            ):
                raise ContractViolationError(
                    "measurement causal and transport assumptions must agree"
                )
            if (
                recommendation.transport.source_domain != recommendation.causal_support.source_scope
                or recommendation.transport.target_domain
                != recommendation.causal_support.target_scope
            ):
                raise ContractViolationError(
                    "measurement causal scopes and transport domains must agree"
                )
        _validate_measurement_causal_support(recommendation, belief, request)
        scientific_evidence_ids = provenance.scientific_evidence_ids
        if (
            not {
                *recommendation.causal_support.evidence_ids,
                *recommendation.transport.evidence_ids,
            }
            <= scientific_evidence_ids
        ):
            raise ContractViolationError(
                "measurement causal or transport evidence is absent from provenance"
            )

    if recommendation.status is MeasurementDecisionStatus.RECOMMENDED:
        assert recommendation.selected_assay_id is not None
        selected_assay = requested_assays[recommendation.selected_assay_id]
        if (
            capability.assay_support[recommendation.selected_assay_id]
            is not SupportStatus.SUPPORTED
        ):
            raise ContractViolationError("selected assay is outside declared assay support")
        if (
            capability.collection_effect_support[selected_assay.collection.effect]
            is not SupportStatus.SUPPORTED
        ):
            raise ContractViolationError(
                "selected assay's collection effect is outside declared support"
            )
    return recommendation


def choose_intervention(
    belief: CellStateBelief,
    *,
    objective: InterventionObjective,
    candidates: Sequence[EvolutionScenario],
    planner: InterventionPlanner,
    options: InferenceOptions | None = None,
) -> InterventionPlan:
    """Choose or explicitly abstain over an exact, bounded candidate set."""

    if not candidates:
        raise ValueError("at least one intervention candidate is required")
    if any(candidate.subject != belief.subject for candidate in candidates):
        raise ContractViolationError("all candidates must refer to the belief's typed subject")
    _require_compatible_posterior(belief, planner)
    horizons = {horizon.name: horizon for horizon in belief.query.prediction_horizons}
    if objective.horizon_name not in horizons:
        raise ContractViolationError("objective horizon is not declared by the belief query")
    query_targets = {target.term.key for target in belief.query.target_outputs}
    if unknown := {term.target.key for term in objective.terms} - query_targets:
        raise ContractViolationError(
            f"objective contains undeclared query targets: {sorted(unknown)}"
        )
    outputs_by_key = {target.term.key: target for target in belief.query.target_outputs}
    for term in objective.terms:
        if objective.horizon_name not in outputs_by_key[term.target.key].supported_horizon_names:
            raise ContractViolationError(
                f"objective target {term.target.key!r} does not support horizon "
                f"{objective.horizon_name!r}"
            )
        if term.target_value is not None and (
            term.target_value.units != outputs_by_key[term.target.key].units
        ):
            raise ContractViolationError(
                f"objective target value for {term.target.key!r} uses incompatible units"
            )
    candidate_ids = [candidate.scenario_id for candidate in candidates]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ContractViolationError("candidate scenario IDs must be unique")

    expected_duration = horizons[objective.horizon_name].duration_seconds
    for candidate in candidates:
        if candidate.horizon_name != objective.horizon_name:
            raise ContractViolationError("all candidates must use the objective horizon")
        if candidate.start_time_seconds != belief.as_of_seconds:
            raise ContractViolationError("all candidates must start at the belief time")
        if not math.isclose(
            candidate.end_time_seconds - candidate.start_time_seconds,
            expected_duration,
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            raise ContractViolationError(
                "candidate duration does not match the objective's named query horizon"
            )
        _validate_scenario_against_query(
            candidate,
            belief.query,
            belief.context.soluble_environment,
            belief.context.active_interventions,
        )
        candidate_event_ids = {
            event.event_id for event in (*candidate.interventions, *candidate.environments)
        }
        if candidate_event_ids & set(belief.provenance.source_event_ids):
            raise ContractViolationError(
                "candidate event IDs must not reuse belief source event IDs"
            )

    resolved_options = options or InferenceOptions()
    expected_scope = planning_capability_scope_fingerprint(belief, objective, candidates)
    _validated_capability_report(
        planner.capabilities(belief, objective, candidates),
        expected_scope_fingerprint=expected_scope,
        operation="intervention planning",
    )
    returned_plan = planner.choose(
        belief,
        objective,
        candidates,
        options=resolved_options,
    )
    try:
        plan = InterventionPlan.model_validate(returned_plan.model_dump(mode="python"))
    except (AttributeError, TypeError, ValueError) as error:
        raise ContractViolationError("planner returned an invalid intervention plan") from error

    expected_candidates = tuple(
        (candidate.scenario_id, canonical_fingerprint(candidate)) for candidate in candidates
    )
    actual_candidates = tuple(
        (candidate.scenario_id, candidate.fingerprint) for candidate in plan.candidates
    )
    if (
        plan.parent_belief_id != belief.belief_id
        or plan.query_fingerprint != belief.query_fingerprint
        or plan.horizon_name != objective.horizon_name
        or plan.objective_id != objective.objective_id
        or plan.objective_fingerprint != canonical_fingerprint(objective)
        or actual_candidates != expected_candidates
    ):
        raise ContractViolationError(
            "planner returned a plan not bound to the belief, objective, and candidates"
        )
    if (
        plan.provenance.history_fingerprint != belief.history_fingerprint
        or plan.provenance.context_fingerprint != belief.context_fingerprint
        or plan.provenance.history_structure_fingerprint
        != belief.provenance.history_structure_fingerprint
        or plan.provenance.source_event_fingerprints != belief.provenance.source_event_fingerprints
    ):
        raise ContractViolationError("plan provenance does not match the input belief")
    _require_provenance_compatible(plan.provenance, planner)
    if plan.seed != resolved_options.seed or plan.provenance.seed != resolved_options.seed:
        raise ContractViolationError("plan provenance has the wrong inference seed")
    if not plan.readiness.control_requested:
        raise ContractViolationError("intervention plan readiness must evaluate control")

    required_objective_estimands = {
        (term.target.key, objective.horizon_name) for term in objective.terms
    }
    try:
        _validate_causal_support_against_query(
            plan.causal_support,
            belief.query,
            required_target_horizons=required_objective_estimands,
        )
        for evaluation in plan.evaluations:
            _validate_causal_support_against_query(
                evaluation.causal_support,
                belief.query,
                required_target_horizons=required_objective_estimands,
            )
        _validate_planning_transport_against_query(
            plan.causal_support,
            plan.transport,
            belief.query,
        )
        candidates_by_id = {candidate.scenario_id: candidate for candidate in candidates}
        for evaluation in plan.evaluations:
            _validate_causal_support_against_scenario(
                evaluation.causal_support,
                candidates_by_id[evaluation.scenario_id],
                belief.context.active_interventions,
            )
            _validate_planning_transport_against_query(
                evaluation.causal_support,
                evaluation.transport,
                belief.query,
            )
            if not evaluation.supported:
                continue
            if evaluation.causal_status not in {
                CausalStatus.IDENTIFIED_POPULATION_EFFECT,
                CausalStatus.TRANSPORTED_UNDER_ASSUMPTIONS,
            }:
                raise ValueError(
                    "a selectable intervention requires identified or transported causal support"
                )
    except ValueError as error:
        raise ContractViolationError(
            "plan causal or transport support is not authorized by the query and objective"
        ) from error

    provenance_ids = plan.provenance.scientific_evidence_ids
    plan_evidence_ids = {
        *plan.transport.evidence_ids,
        *plan.causal_support.evidence_ids,
        *(
            evidence_id
            for evaluation in plan.evaluations
            for evidence_id in (
                *evaluation.transport.evidence_ids,
                *evaluation.causal_support.evidence_ids,
            )
        ),
    }
    if not plan_evidence_ids <= provenance_ids:
        raise ContractViolationError("plan causal or transport evidence is absent from provenance")
    return plan
