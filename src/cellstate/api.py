"""Thin public application boundary for estimation, evolution, and planning."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from .domain.belief import CellStateBelief
from .domain.common import EvidenceStatus, ProvenanceRecord, Quantity, canonical_fingerprint
from .domain.events import EvidenceRole, MissingnessStatus, ObservationEvent
from .domain.query import StateQuery
from .domain.request import EstimateCellStateRequest, InferenceOptions
from .domain.scenarios import (
    EvolutionScenario,
    InterventionObjective,
    InterventionPlan,
    StateForecast,
)
from .errors import CapabilityError, ContractViolationError, PosteriorCompatibilityError
from .ports import CellStateEstimator, InterventionPlanner, StateEvolutionModel


def _require_compatible_posterior(
    belief: CellStateBelief,
    model: CellStateEstimator | StateEvolutionModel | InterventionPlanner,
) -> None:
    _require_provenance_compatible(belief.provenance, model)


def _require_provenance_compatible(
    provenance: ProvenanceRecord,
    model: CellStateEstimator | StateEvolutionModel | InterventionPlanner,
) -> None:
    descriptor = model.descriptor
    expected = (
        descriptor.model_id,
        descriptor.model_version,
        descriptor.model_fingerprint,
        descriptor.posterior_schema_id,
        descriptor.training_support_id,
    )
    actual = (
        provenance.model_id,
        provenance.model_version,
        provenance.model_fingerprint,
        provenance.posterior_schema_id,
        provenance.training_support_id,
    )
    if actual != expected:
        raise PosteriorCompatibilityError(
            "belief posterior was produced by an incompatible model, configuration, "
            "posterior schema, or training support"
        )


def _validate_scenario_against_query(
    scenario: EvolutionScenario,
    query: StateQuery,
    inherited_environment: Mapping[str, object],
) -> None:
    for intervention_event in scenario.interventions:
        matching = []
        for intervention_spec in query.intervention_space:
            if intervention_spec.kind.key != intervention_event.intervention_type.key:
                continue
            if intervention_spec.target is not None and (
                intervention_event.target is None
                or intervention_spec.target.key != intervention_event.target.key
            ):
                continue
            if intervention_spec.mechanisms and (
                intervention_event.mechanism is None
                or intervention_event.mechanism.key
                not in {mechanism.key for mechanism in intervention_spec.mechanisms}
            ):
                continue
            if intervention_spec.dose_units is not None and (
                intervention_event.dose is None
                or intervention_event.dose.units != intervention_spec.dose_units
            ):
                continue
            matching.append(intervention_spec)
        if not matching:
            raise ContractViolationError(
                f"scenario intervention {intervention_event.event_id!r} is outside the query "
                "intervention space"
            )
    environment_specs = {item.variable.key.casefold(): item for item in query.environment_space}
    for environment_event in scenario.environments:
        for key, value in environment_event.variables.items():
            environment_spec = environment_specs.get(key.casefold())
            if environment_spec is None:
                raise ContractViolationError(
                    f"scenario environment variable {key!r} is outside the query environment space"
                )
            if environment_spec.units is not None and (
                not isinstance(value, Quantity) or value.units != environment_spec.units
            ):
                raise ContractViolationError(
                    f"scenario environment variable {key!r} does not use query-declared units"
                )
    if scenario.inherit_current_environment is True:
        for key, inherited_value in inherited_environment.items():
            environment_spec = environment_specs.get(key.casefold())
            if environment_spec is None:
                raise ContractViolationError(
                    f"inherited environment variable {key!r} is outside the query environment space"
                )
            if environment_spec.units is not None:
                try:
                    quantity = (
                        inherited_value
                        if isinstance(inherited_value, Quantity)
                        else Quantity.model_validate(inherited_value)
                    )
                except (TypeError, ValueError) as error:
                    raise ContractViolationError(
                        f"inherited environment variable {key!r} lacks an interpretable quantity"
                    ) from error
                if quantity.units != environment_spec.units:
                    raise ContractViolationError(
                        f"inherited environment variable {key!r} uses incompatible units"
                    )


def estimate_cell_state(
    request: EstimateCellStateRequest,
    *,
    estimator: CellStateEstimator,
    options: InferenceOptions | None = None,
) -> CellStateBelief:
    """Estimate a query-conditioned posterior; no scientific default is installed."""

    resolved_options = options or InferenceOptions()
    if request.previous_belief is not None:
        _require_compatible_posterior(request.previous_belief, estimator)
    capability = estimator.capabilities(request.query)
    if resolved_options.strict_capabilities and not capability.supported:
        boundary = (
            (capability.unsupported_system_boundary,)
            if capability.unsupported_system_boundary is not None
            else ()
        )
        details = (
            *boundary,
            *capability.unsupported_modalities,
            *capability.unsupported_interventions,
            *capability.unsupported_environments,
            *capability.unsupported_outputs,
            *capability.unsupported_precision_requirements,
            *capability.notes,
        )
        raise CapabilityError("; ".join(details) or "estimator does not support this query")

    returned_belief = estimator.estimate(request, options=resolved_options)
    try:
        belief = CellStateBelief.model_validate(returned_belief.model_dump(mode="python"))
    except ValueError as error:
        raise ContractViolationError("estimator returned an invalid belief contract") from error
    if belief.subject_id != request.history.subject_id:
        raise ContractViolationError("estimator returned a belief for the wrong subject")
    if belief.as_of_seconds != request.as_of_seconds:
        raise ContractViolationError("estimator returned a belief at the wrong time")
    if belief.query_fingerprint != request.query.fingerprint:
        raise ContractViolationError("estimator returned a belief for a different query")
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
    unknown_source_ids = set(belief.provenance.source_event_ids) - set(events_by_id)
    if unknown_source_ids:
        raise ContractViolationError(
            f"estimator provenance names unknown source events: {sorted(unknown_source_ids)}"
        )
    for event_id, fingerprint in belief.provenance.source_event_fingerprints.items():
        if canonical_fingerprint(events_by_id[event_id]) != fingerprint:
            raise ContractViolationError(
                f"estimator provenance fingerprint disagrees for event {event_id!r}"
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
            and math.isclose(
                event.time_seconds,
                belief.as_of_seconds,
                rel_tol=0,
                abs_tol=1e-12,
            )
            for event in evidence
        ):
            raise ContractViolationError(
                "an observed factor requires a direct measurement at the belief time"
            )
    return belief


def evolve_cell_state(
    belief: CellStateBelief,
    *,
    scenario: EvolutionScenario,
    evolution_model: StateEvolutionModel,
    options: InferenceOptions | None = None,
) -> StateForecast:
    """Propagate posterior uncertainty through a planned controlled scenario."""

    if scenario.subject_id != belief.subject_id:
        raise ContractViolationError("scenario and belief must refer to the same subject")
    if scenario.start_time_seconds != belief.as_of_seconds:
        raise ContractViolationError("scenario must start at the belief time")
    _require_compatible_posterior(belief, evolution_model)
    if belief.context.active_interventions and scenario.inherit_active_interventions is None:
        raise ContractViolationError(
            "scenario must explicitly inherit or clear active interventions from the belief"
        )
    _validate_scenario_against_query(scenario, belief.query, belief.context.soluble_environment)
    scenario_event_ids = [
        event.event_id for event in (*scenario.interventions, *scenario.environments)
    ]
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
    resolved_options = options or InferenceOptions()
    returned_forecast = evolution_model.evolve(belief, scenario, options=resolved_options)
    try:
        forecast = StateForecast.model_validate(returned_forecast.model_dump(mode="python"))
    except ValueError as error:
        raise ContractViolationError("evolution model returned an invalid forecast") from error
    if forecast.subject_id != belief.subject_id or forecast.scenario_id != scenario.scenario_id:
        raise ContractViolationError("evolution model returned a forecast for the wrong scenario")
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
        forecast.query_fingerprint != belief.query_fingerprint
        or forecast.provenance.history_fingerprint != belief.history_fingerprint
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
        **{
            event.event_id: canonical_fingerprint(event)
            for event in (*scenario.interventions, *scenario.environments)
        },
    }
    if forecast.provenance.source_event_fingerprints != expected_event_fingerprints:
        raise ContractViolationError("forecast provenance does not bind the complete evidence set")
    return forecast


def choose_intervention(
    belief: CellStateBelief,
    *,
    objective: InterventionObjective,
    candidates: Sequence[EvolutionScenario],
    planner: InterventionPlanner,
    options: InferenceOptions | None = None,
) -> InterventionPlan:
    """Choose among explicit candidate scenarios under an explicit objective."""

    if not candidates:
        raise ValueError("at least one intervention candidate is required")
    if any(candidate.subject_id != belief.subject_id for candidate in candidates):
        raise ContractViolationError("all candidates must refer to the belief subject")
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
        if belief.context.active_interventions and candidate.inherit_active_interventions is None:
            raise ContractViolationError(
                "candidates must explicitly inherit or clear active interventions from the belief"
            )
        _validate_scenario_against_query(
            candidate, belief.query, belief.context.soluble_environment
        )
    resolved_options = options or InferenceOptions()
    returned_plan = planner.choose(
        belief,
        objective,
        candidates,
        options=resolved_options,
    )
    try:
        plan = InterventionPlan.model_validate(returned_plan.model_dump(mode="python"))
    except ValueError as error:
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
    return plan
