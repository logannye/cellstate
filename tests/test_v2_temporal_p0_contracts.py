from __future__ import annotations

from typing import Any

import pytest
from conftest import (
    environment_factory,
    environment_spec_factory,
    intervention_factory,
    intervention_spec_factory,
    observation_factory,
    query_factory,
    request_factory,
    subject_factory,
)
from pydantic import ValidationError

from cellstate import estimate_cell_state, evolve_cell_state
from cellstate.api import (
    _intervention_is_active,
    _validate_causal_support_against_scenario,
)
from cellstate.domain.belief import (
    CausalEstimandBinding,
    CausalSupportReport,
    DecisionUncertaintyReport,
    EvaluationStatus,
    QueryReadinessReport,
)
from cellstate.domain.common import (
    CausalStatus,
    CriterionOutcome,
    Quantity,
    canonical_fingerprint,
)
from cellstate.domain.events import (
    ActualPerturbation,
    AssignmentMechanism,
    CollectionEffect,
    EvidenceRole,
    PerturbationStatus,
)
from cellstate.domain.history import CellHistory
from cellstate.domain.query import EnvironmentVariableSpec, MissingHistoryPolicy
from cellstate.domain.request import EstimateCellStateRequest, InferenceOptions
from cellstate.domain.scenarios import EvolutionScenario
from cellstate.domain.subjects import IdentityBasis
from cellstate.errors import ContractViolationError
from cellstate.ports import CapabilityReport, ModelArtifactKind


def _environment_query(
    *,
    lookback_seconds: float | None = 10.0,
    missing_policy: MissingHistoryPolicy = MissingHistoryPolicy.REJECT,
) -> Any:
    query = query_factory()
    specification_payload = environment_spec_factory().model_dump(mode="python")
    specification_payload["missing_history_policy"] = missing_policy
    if missing_policy is MissingHistoryPolicy.USE_DECLARED_DEFAULT:
        specification_payload["default_value"] = Quantity(value=1, units="relative")
    specification = EnvironmentVariableSpec.model_validate(specification_payload)
    return query.model_copy(
        update={
            "environment_space": (specification,),
            "evidence_policy": query.evidence_policy.model_copy(
                update={"lookback_seconds": lookback_seconds}
            ),
        }
    )


def _history_with_environment(*events: Any) -> CellHistory:
    return CellHistory(
        subject=subject_factory(),
        events=(observation_factory(), *events),
    )


def test_zero_duration_intervention_is_active_only_at_its_exact_instant() -> None:
    intervention = intervention_factory(time_seconds=10, duration_seconds=0)

    assert _intervention_is_active(intervention, 10)
    assert not _intervention_is_active(intervention, 10 + 1e-9)


def test_inherited_point_intervention_is_part_of_scenario_causal_contrast() -> None:
    subject = subject_factory()
    target = query_factory().target_outputs[0]
    point = intervention_factory(
        event_id="point-drug",
        subject=subject,
        time_seconds=10,
        duration_seconds=0,
        intervention_spec_id="drug",
    )
    explicit = intervention_factory(
        event_id="future-stimulus",
        subject=subject,
        time_seconds=10,
        duration_seconds=60,
        intervention_type="stimulus",
        intervention_spec_id="stimulus",
        estimated_efficiency=None,
    )
    scenario = EvolutionScenario(
        scenario_id="point-plus-interval",
        horizon_name="acute",
        subject=subject,
        start_time_seconds=10,
        end_time_seconds=70,
        interventions=(explicit,),
        inherit_active_interventions=True,
    )
    estimand = CausalEstimandBinding(
        target=target.term,
        horizon_name="acute",
        aggregation=target.aggregation,
        intervention_spec_ids=("drug", "stimulus"),
        comparator="matched randomized controls",
        scenario_id=scenario.scenario_id,
        scenario_fingerprint=canonical_fingerprint(scenario),
    )
    report = CausalSupportReport(
        evaluation_status=EvaluationStatus.EVALUATED,
        outcome=CriterionOutcome.PASSED,
        causal_status=CausalStatus.IDENTIFIED_POPULATION_EFFECT,
        identification_basis="randomized intervention assignment",
        identification_design=AssignmentMechanism.RANDOMIZED,
        estimands=(estimand,),
        evidence_ids=("trial",),
        evidence_fingerprints={"trial": "8" * 64},
        source_scope="randomized source population",
        target_scope="query target population",
    )

    _validate_causal_support_against_scenario(report, scenario, (point,))

    omitted_point = report.model_copy(
        update={
            "estimands": (estimand.model_copy(update={"intervention_spec_ids": ("stimulus",)}),)
        }
    )
    with pytest.raises(ContractViolationError, match="intervention contrast"):
        _validate_causal_support_against_scenario(omitted_point, scenario, (point,))


def test_required_environment_union_must_cover_the_complete_conditioning_interval() -> None:
    query = _environment_query()
    first = environment_factory(event_id="medium-a", time_seconds=0, duration_seconds=4)
    second = environment_factory(event_id="medium-b", time_seconds=4, duration_seconds=6)
    request = request_factory(
        query=query,
        history=_history_with_environment(first, second),
    )
    assert request.as_of_seconds == 10

    gap = second.model_copy(update={"time_seconds": 5, "duration_seconds": 5})
    with pytest.raises(ValidationError, match="complete conditioning interval"):
        request_factory(
            query=query,
            history=_history_with_environment(first, gap),
        )

    point_presence = first.model_copy(update={"duration_seconds": 0})
    with pytest.raises(ValidationError, match="complete conditioning interval"):
        request_factory(
            query=query,
            history=_history_with_environment(point_presence),
        )


def test_environment_history_rejects_conflicting_overlaps_and_unbounded_reject_policy() -> None:
    query = _environment_query()
    first = environment_factory(event_id="medium-a", time_seconds=0, duration_seconds=7)
    conflicting = environment_factory(
        event_id="medium-b",
        time_seconds=5,
        duration_seconds=5,
        variables={"NUTRIENT": Quantity(value=2, units="relative")},
    )
    with pytest.raises(ValidationError, match="conflicting overlapping intervals"):
        request_factory(
            query=query,
            history=_history_with_environment(first, conflicting),
        )

    unbounded_query = _environment_query(lookback_seconds=None)
    covering = first.model_copy(update={"duration_seconds": 10})
    with pytest.raises(ValidationError, match="finite query evidence lookback"):
        request_factory(
            query=unbounded_query,
            history=_history_with_environment(covering),
        )


@pytest.mark.parametrize(
    "policy",
    (MissingHistoryPolicy.REPRESENT_AS_UNKNOWN, MissingHistoryPolicy.USE_DECLARED_DEFAULT),
)
def test_explicit_missing_environment_policies_do_not_invent_zero_exposure(
    policy: MissingHistoryPolicy,
) -> None:
    query = _environment_query(lookback_seconds=None, missing_policy=policy)
    request = request_factory(query=query)
    assert not any(event.kind == "environment" for event in request.history.events)
    specification = query.environment_space[0]
    if policy is MissingHistoryPolicy.REPRESENT_AS_UNKNOWN:
        assert specification.default_value is None
    else:
        assert specification.default_value == Quantity(value=1, units="relative")


def test_historical_combination_policy_applies_to_overlapping_regimens_not_lifetime_actions() -> (
    None
):
    query = query_factory()
    stimulus_spec = intervention_spec_factory(spec_id="stimulus", kind="stimulus")
    constraints = query.constraints.model_copy(
        update={
            "maximum_intervention_combination_order": 2,
            "forbidden_combinations": (("drug", "stimulus"),),
        }
    )
    query = query.model_copy(
        update={
            "intervention_space": (*query.intervention_space, stimulus_spec),
            "constraints": constraints,
        }
    )
    drug = intervention_factory(
        event_id="drug-first",
        intervention_spec_id="drug",
        intervention_type="drug",
        time_seconds=0,
        duration_seconds=2,
    )
    sequential_stimulus = intervention_factory(
        event_id="stimulus-second",
        intervention_spec_id="stimulus",
        intervention_type="stimulus",
        time_seconds=2,
        duration_seconds=2,
    )
    request = request_factory(
        query=query,
        history=CellHistory(
            subject=subject_factory(),
            events=(observation_factory(), drug, sequential_stimulus),
        ),
    )
    assert len(request.history.events) == 3

    overlapping_stimulus = sequential_stimulus.model_copy(update={"time_seconds": 1})
    with pytest.raises(ValidationError, match="overlapping intervention regimens"):
        request_factory(
            query=query,
            history=CellHistory(
                subject=subject_factory(),
                events=(observation_factory(), drug, overlapping_stimulus),
            ),
        )

    drug_with_washout = drug.model_copy(
        update={"schedule": drug.schedule.model_copy(update={"washout_seconds": 2})}
    )
    washout_overlap = sequential_stimulus.model_copy(update={"time_seconds": 3})
    with pytest.raises(ValidationError, match="overlapping intervention regimens"):
        request_factory(
            query=query,
            history=CellHistory(
                subject=subject_factory(),
                events=(observation_factory(), drug_with_washout, washout_overlap),
            ),
        )


def test_missing_realization_is_valid_uncertainty_and_returns_an_abstaining_belief(
    model: Any,
) -> None:
    intervention = intervention_factory(
        event_id="intended-only",
        actual_perturbation=None,
        estimated_efficiency=None,
    )
    request = request_factory(
        history=CellHistory(
            subject=subject_factory(),
            events=(observation_factory(), intervention),
        )
    )
    belief = estimate_cell_state(request, estimator=model)
    assert belief.readiness.abstention_required
    assert not belief.readiness.valid_for_control


class _CausalOverclaimEstimator:
    def __init__(self, model: Any) -> None:
        self._model = model
        self._descriptor = type(model.descriptor).model_validate(
            {
                **model.descriptor.model_dump(mode="python"),
                "artifact_kind": ModelArtifactKind.SYNTHETIC_TEST_MODEL,
                "support_envelope_id": "synthetic-envelope",
                "support_envelope_fingerprint": "a" * 64,
                "training_support_id": "synthetic-training",
                "training_support_fingerprint": "b" * 64,
                "validation_evidence_ids": ("trial-1",),
                "validation_evidence_fingerprints": {"trial-1": "c" * 64},
            }
        )

    @property
    def descriptor(self) -> Any:
        return self._descriptor

    @property
    def query_compiler(self) -> Any:
        return self._model.query_compiler

    def capabilities(self, request: Any, state_specification: Any) -> Any:
        delegated = self._model.capabilities(request, state_specification)
        return CapabilityReport(
            supported=True,
            scope_fingerprint=delegated.scope_fingerprint,
            notes=("synthetic biological overclaim adapter for boundary testing",),
        )

    def estimate(self, request: EstimateCellStateRequest, *, options: InferenceOptions) -> Any:
        belief = self._model.estimate(request, options=options)
        target = request.query.target_outputs[0]
        causal_support = CausalSupportReport(
            evaluation_status=EvaluationStatus.EVALUATED,
            outcome=CriterionOutcome.PASSED,
            causal_status=CausalStatus.IDENTIFIED_POPULATION_EFFECT,
            identification_basis="claimed randomized identification",
            identification_design=AssignmentMechanism.RANDOMIZED,
            estimands=(
                CausalEstimandBinding(
                    target=target.term,
                    horizon_name="acute",
                    aggregation=target.aggregation,
                    intervention_spec_ids=("drug",),
                    comparator="randomized control wells",
                ),
            ),
            evidence_ids=("trial-1",),
            evidence_fingerprints={"trial-1": "c" * 64},
            source_scope="synthetic source",
            target_scope="synthetic target",
        )
        decision = DecisionUncertaintyReport(
            evaluation_status=EvaluationStatus.EVALUATED,
            outcome=CriterionOutcome.PASSED,
            decision_uncertainty=0,
            maximum_decision_uncertainty=request.query.acceptance_thresholds.maximum_decision_uncertainty,
            counterfactual_uncertainty=0,
            maximum_counterfactual_uncertainty=request.query.acceptance_thresholds.maximum_counterfactual_uncertainty,
        )
        diagnostics = belief.diagnostics.model_copy(
            update={"causal_support": causal_support, "decision_uncertainty": decision}
        )
        readiness = QueryReadinessReport(
            support=belief.readiness.support,
            sufficiency=belief.readiness.sufficiency,
            identifiability=belief.readiness.identifiability,
            decision_uncertainty=CriterionOutcome.PASSED,
            calibration=belief.readiness.calibration,
            causal=CriterionOutcome.PASSED,
            measurement_model=belief.readiness.measurement_model,
            control_requested=True,
            valid_for_prediction=belief.readiness.valid_for_prediction,
            valid_for_control=belief.readiness.valid_for_prediction,
            valid_for_measurement_selection=belief.readiness.valid_for_measurement_selection,
            abstention_required=not belief.readiness.valid_for_prediction,
            reasons=(
                ()
                if belief.readiness.valid_for_prediction
                else ("non-causal readiness criteria still failed",)
            ),
        )
        provenance = belief.provenance.model_copy(
            update={
                "support_envelope_id": self._descriptor.support_envelope_id,
                "support_envelope_fingerprint": self._descriptor.support_envelope_fingerprint,
                "training_support_id": self._descriptor.training_support_id,
                "training_support_fingerprint": self._descriptor.training_support_fingerprint,
                "validation_evidence_ids": self._descriptor.validation_evidence_ids,
                "validation_evidence_fingerprints": (
                    self._descriptor.validation_evidence_fingerprints
                ),
            }
        )
        return belief.model_copy(
            update={
                "diagnostics": diagnostics,
                "readiness": readiness,
                "provenance": provenance,
            }
        )


def test_missing_realization_prevents_backend_from_overclaiming_causal_readiness(
    model: Any,
) -> None:
    config_payload = model.config.model_dump(mode="python")
    config_payload["control_assignment_mechanisms"]["drug"] = (AssignmentMechanism.RANDOMIZED,)
    config_payload["control_randomization_unit_kinds"]["drug"] = "well"
    config_payload["control_requires_randomization_unit"]["drug"] = True
    configured_model = type(model)(type(model.config).model_validate(config_payload))
    query = query_factory().model_copy(
        update={
            "intervention_space": (
                intervention_spec_factory(
                    allowed_assignment_mechanisms=(AssignmentMechanism.RANDOMIZED,),
                    randomization_unit_kind="well",
                    require_randomization_unit=True,
                ),
            )
        }
    )
    intervention = intervention_factory(
        event_id="intended-only",
        actual_perturbation=None,
        estimated_efficiency=None,
        assignment_mechanism=AssignmentMechanism.RANDOMIZED,
        randomization_unit_kind="well",
        randomization_unit_id="well-1",
    )
    request = request_factory(
        query=query,
        history=CellHistory(
            subject=subject_factory(),
            events=(observation_factory(), intervention),
        ),
    )
    with pytest.raises(ContractViolationError, match="realization evidence is unresolved"):
        estimate_cell_state(request, estimator=_CausalOverclaimEstimator(configured_model))


def test_inferred_realization_evidence_cannot_predate_the_intervention() -> None:
    observation = observation_factory(event_id="pre-action", time_seconds=0)
    intervention = intervention_factory(
        event_id="action",
        time_seconds=5,
        actual_perturbation=ActualPerturbation(
            status=PerturbationStatus.INFERRED,
            efficiency=0.5,
            evidence_event_ids=(observation.event_id,),
        ),
    )
    with pytest.raises(ValidationError, match="cannot predate the intervention"):
        CellHistory(
            subject=subject_factory(),
            events=(observation, intervention),
        )

    query = query_factory()
    assert query.realization_evidence_gaps(
        intervention,
        {observation.event_id: observation},
    ) == (f"pre_intervention_realization_evidence:{observation.event_id}",)


def test_terminal_collection_closes_direct_intervals_but_not_later_reference_evidence() -> None:
    subject = subject_factory()
    terminal = observation_factory(
        event_id="terminal",
        subject=subject,
        time_seconds=1,
        duration_seconds=2,
        collection=observation_factory().collection.model_copy(
            update={"effect": CollectionEffect.TERMINAL_DESTRUCTIVE}
        ),
    )
    overlapping_direct = observation_factory(
        event_id="overlap",
        subject=subject,
        time_seconds=2.5,
        duration_seconds=1,
    )
    with pytest.raises(ValidationError, match="later events are invalid"):
        CellHistory(subject=subject, events=(terminal, overlapping_direct))

    reference = observation_factory(
        event_id="external-reference",
        subject=subject,
        source_subject=subject_factory("reference-cell"),
        evidence_role=EvidenceRole.EXTERNAL_REFERENCE,
        linkage_basis=IdentityBasis.EXTERNAL_REFERENCE,
        time_seconds=5,
    )
    history = CellHistory(subject=subject, events=(terminal, reference))
    assert tuple(event.event_id for event in history.events) == ("terminal", "external-reference")


def _scenario_event(
    event_id: str,
    *,
    start: float,
    duration: float,
    value: float = 1,
) -> Any:
    return environment_factory(
        event_id=event_id,
        time_seconds=start,
        duration_seconds=duration,
        variables={"nutrient": Quantity(value=value, units="relative")},
    )


def test_scenario_environment_union_covers_full_horizon_and_rejects_gaps_or_conflicts(
    model: Any,
) -> None:
    query = _environment_query()
    historical_environment = environment_factory(time_seconds=0, duration_seconds=10)
    request = request_factory(
        query=query,
        history=_history_with_environment(historical_environment),
    )
    belief = estimate_cell_state(request, estimator=model)
    first = _scenario_event("future-a", start=10, duration=30)
    second = _scenario_event("future-b", start=40, duration=30)
    covered = EvolutionScenario(
        scenario_id="covered",
        horizon_name="acute",
        subject=belief.subject,
        start_time_seconds=10,
        end_time_seconds=70,
        environments=(first, second),
        inherit_current_environment=False,
    )
    forecast = evolve_cell_state(belief, scenario=covered, evolution_model=model)
    assert forecast.scenario_id == "covered"

    inherited_then_overridden = covered.model_copy(
        update={
            "scenario_id": "inherited-then-overridden",
            "inherit_current_environment": True,
            "environments": (_scenario_event("future-override", start=40, duration=30, value=2),),
        }
    )
    belief_with_current_environment = belief.model_copy(
        update={
            "context": belief.context.model_copy(
                update={"soluble_environment": {"nutrient": {"value": 1.0, "units": "relative"}}}
            )
        }
    )
    overridden_forecast = evolve_cell_state(
        belief_with_current_environment,
        scenario=inherited_then_overridden,
        evolution_model=model,
    )
    assert overridden_forecast.scenario_id == "inherited-then-overridden"

    gap = covered.model_copy(
        update={
            "scenario_id": "gap",
            "environments": (
                first,
                second.model_copy(update={"time_seconds": 41, "duration_seconds": 29}),
            ),
        }
    )
    with pytest.raises(ContractViolationError, match="complete scenario interval"):
        evolve_cell_state(belief, scenario=gap, evolution_model=model)

    conflicting = covered.model_copy(
        update={
            "scenario_id": "conflict",
            "environments": (
                first.model_copy(update={"duration_seconds": 40}),
                second.model_copy(
                    update={
                        "time_seconds": 35,
                        "duration_seconds": 35,
                        "variables": {"nutrient": Quantity(value=2, units="relative")},
                    }
                ),
            ),
        }
    )
    with pytest.raises(ContractViolationError, match="conflicting overlapping intervals"):
        evolve_cell_state(belief, scenario=conflicting, evolution_model=model)

    point_only = covered.model_copy(
        update={
            "scenario_id": "point-only",
            "environments": (first.model_copy(update={"duration_seconds": 0}),),
        }
    )
    with pytest.raises(ContractViolationError, match="complete scenario interval"):
        evolve_cell_state(belief, scenario=point_only, evolution_model=model)


def test_inherited_optional_environment_stops_at_first_assignment_and_requires_a_tail(
    model: Any,
) -> None:
    query = _environment_query()
    query = query.model_copy(
        update={
            "environment_space": (
                query.environment_space[0].model_copy(update={"required": False}),
            )
        }
    )
    belief = estimate_cell_state(request_factory(query=query), estimator=model)
    belief = belief.model_copy(
        update={
            "context": belief.context.model_copy(
                update={"soluble_environment": {"nutrient": {"value": 1.0, "units": "relative"}}}
            )
        }
    )
    incomplete_override = EvolutionScenario(
        scenario_id="optional-current-gap",
        horizon_name="acute",
        subject=belief.subject,
        start_time_seconds=10,
        end_time_seconds=70,
        environments=(_scenario_event("optional-override", start=40, duration=20, value=2),),
        inherit_current_environment=True,
    )
    with pytest.raises(ContractViolationError, match="required or inherited"):
        evolve_cell_state(
            belief,
            scenario=incomplete_override,
            evolution_model=model,
        )

    complete_override = incomplete_override.model_copy(
        update={
            "scenario_id": "optional-current-covered",
            "environments": (_scenario_event("optional-override", start=40, duration=30, value=2),),
        }
    )
    forecast = evolve_cell_state(
        belief,
        scenario=complete_override,
        evolution_model=model,
    )
    assert forecast.scenario_id == "optional-current-covered"
