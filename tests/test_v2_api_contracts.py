from __future__ import annotations

from collections.abc import Sequence
from typing import NoReturn
from uuid import uuid4

import pytest
from conftest import (
    environment_factory,
    intervention_factory,
    query_factory,
    subject_factory,
)

from cellstate import (
    EvolutionScenario,
    InferenceOptions,
    InterventionObjective,
    ObjectiveDirection,
    ObjectiveTerm,
    OntologyTerm,
    choose_intervention,
    estimate_cell_state,
    evolve_cell_state,
)
from cellstate.domain.belief import CellStateBelief, ContextBelief
from cellstate.domain.common import CausalStatus, ProvenanceRecord
from cellstate.domain.query import StateQuery
from cellstate.domain.request import EstimateCellStateRequest
from cellstate.domain.scenarios import InterventionPlan, StateForecast
from cellstate.domain.specification import (
    CompiledStateSpecification,
    ExcludedStateFactor,
    StateFactor,
    StateFactorSpecification,
)
from cellstate.errors import CapabilityError, ContractViolationError
from cellstate.ports import (
    CapabilityReport,
    EstimatorDescriptor,
    ModelArtifactKind,
    QueryCompilerDescriptor,
    estimation_capability_scope_fingerprint,
)

_MODEL_FINGERPRINT = "b" * 64
_COMPILER_FINGERPRINT = "a" * 64


def _descriptor() -> EstimatorDescriptor:
    return EstimatorDescriptor(
        model_id="api-contract-test",
        model_version="2.0",
        model_fingerprint=_MODEL_FINGERPRINT,
        posterior_schema_id="cellstate/test-v2",
        description="Contract-only fake used to exercise public preflights.",
        artifact_kind=ModelArtifactKind.CONTRACT_REFERENCE,
    )


def _state_specification(query: StateQuery) -> CompiledStateSpecification:
    active = StateFactor.FUNCTIONAL_CAPACITY
    return CompiledStateSpecification(
        query_fingerprint=query.fingerprint,
        subject=query.subject,
        compiler_id="api-test-compiler",
        compiler_version="2.0",
        compiler_fingerprint=_COMPILER_FINGERPRINT,
        active_factors=(
            StateFactorSpecification(
                factor=active,
                dimensions=("functional_capacity",),
                timescales=frozenset(horizon.timescale for horizon in query.prediction_horizons),
                required_for_outputs=tuple(output.term.key for output in query.target_outputs),
                rationale="Narrow contract-test state.",
            ),
        ),
        excluded_factors=tuple(
            ExcludedStateFactor(factor=factor, rationale="Outside this narrow test query.")
            for factor in StateFactor
            if factor is not active
        ),
        system_boundary=query.system_boundary,
        temporal_resolution_seconds=query.temporal_resolution_seconds,
        target_outputs=query.target_outputs,
        prediction_horizons=query.prediction_horizons,
        intervention_space=query.intervention_space,
        environment_space=query.environment_space,
        precision_requirements=query.precision_requirements,
        available_assays=query.available_assays,
        evidence_policy=query.evidence_policy,
        constraints=query.constraints,
        target_output_keys=tuple(output.term.key for output in query.target_outputs),
        horizon_names=tuple(horizon.name for horizon in query.prediction_horizons),
        admissible_evidence_roles=query.evidence_policy.allowed_evidence_roles,
        acceptance_thresholds=query.acceptance_thresholds,
    )


class _Compiler:
    def __init__(
        self,
        query: StateQuery,
        calls: list[str],
        *,
        wrong_query_binding: bool = False,
    ) -> None:
        self._query = query
        self._calls = calls
        self._wrong_query_binding = wrong_query_binding

    @property
    def compiler_descriptor(self) -> QueryCompilerDescriptor:
        return QueryCompilerDescriptor(
            compiler_id="api-test-compiler",
            compiler_version="2.0",
            compiler_fingerprint=_COMPILER_FINGERPRINT,
        )

    def compile(self, query: StateQuery) -> CompiledStateSpecification:
        self._calls.append("compile")
        assert query is self._query
        specification = _state_specification(query)
        if self._wrong_query_binding:
            return specification.model_copy(update={"query_fingerprint": "f" * 64})
        return specification


class _PreflightEstimator:
    def __init__(
        self,
        query: StateQuery,
        calls: list[str],
        *,
        blockers: bool = False,
        wrong_scope: bool = False,
        wrong_query_binding: bool = False,
    ) -> None:
        self._query = query
        self._calls = calls
        self._blockers = blockers
        self._wrong_scope = wrong_scope
        self._compiler = _Compiler(
            query,
            calls,
            wrong_query_binding=wrong_query_binding,
        )

    @property
    def descriptor(self) -> EstimatorDescriptor:
        return _descriptor()

    @property
    def query_compiler(self) -> _Compiler:
        return self._compiler

    def capabilities(
        self,
        request: EstimateCellStateRequest,
        state_specification: CompiledStateSpecification,
    ) -> CapabilityReport:
        self._calls.append("capabilities")
        assert request.query is self._query
        scope = estimation_capability_scope_fingerprint(request, state_specification)
        if self._wrong_scope:
            scope = "0" * 64
        if not self._blockers:
            return CapabilityReport(
                supported=False,
                scope_fingerprint=scope,
                notes=("deliberate preflight stop",),
            )
        return CapabilityReport(
            supported=False,
            scope_fingerprint=scope,
            unsupported_system_boundary="population",
            unsupported_subjects=("population subject",),
            unsupported_aggregations=("population mean",),
            unsupported_modalities=("transcriptome",),
            unsupported_interventions=("drug",),
            unsupported_doses=("drug:101 relative",),
            unsupported_schedules=("pulsed",),
            unsupported_delivery_methods=("electroporation",),
            unsupported_combinations=("drug+cytokine",),
            unsupported_environments=("hypoxia",),
            unsupported_outputs=("functional_capacity",),
            unsupported_horizons=("acute",),
            unsupported_precision_requirements=("absolute_error",),
            unsupported_causal_classes=(CausalStatus.IDENTIFIED_POPULATION_EFFECT,),
            unsupported_readiness_criteria=("calibration",),
            unsupported_constraints=("safety-1",),
            notes=("all blockers are surfaced",),
        )

    def estimate(
        self,
        request: EstimateCellStateRequest,
        *,
        options: InferenceOptions,
    ) -> NoReturn:
        raise AssertionError("estimate must not run after a strict preflight failure")


def test_estimation_compiles_before_exact_scope_capability_preflight(
    estimate_request: EstimateCellStateRequest,
) -> None:
    calls: list[str] = []
    estimator = _PreflightEstimator(estimate_request.query, calls)

    with pytest.raises(CapabilityError, match="deliberate preflight stop"):
        estimate_cell_state(estimate_request, estimator=estimator)

    assert calls == ["compile", "capabilities"]


def test_compiler_binding_is_verified_before_capability_preflight(
    estimate_request: EstimateCellStateRequest,
) -> None:
    calls: list[str] = []
    estimator = _PreflightEstimator(
        estimate_request.query,
        calls,
        wrong_query_binding=True,
    )

    with pytest.raises(ContractViolationError, match="invalid compiled state"):
        estimate_cell_state(estimate_request, estimator=estimator)

    assert calls == ["compile"]


def test_capability_scope_binding_cannot_be_disabled(
    estimate_request: EstimateCellStateRequest,
) -> None:
    calls: list[str] = []
    estimator = _PreflightEstimator(
        estimate_request.query,
        calls,
        wrong_scope=True,
    )

    with pytest.raises(ContractViolationError, match="exact requested scope"):
        estimate_cell_state(
            estimate_request,
            estimator=estimator,
            options=InferenceOptions(),
        )

    assert calls == ["compile", "capabilities"]


def test_strict_capability_failure_surfaces_every_blocker_category(
    estimate_request: EstimateCellStateRequest,
) -> None:
    estimator = _PreflightEstimator(
        estimate_request.query,
        [],
        blockers=True,
    )

    with pytest.raises(CapabilityError) as captured:
        estimate_cell_state(estimate_request, estimator=estimator)

    message = str(captured.value)
    for detail in (
        "population",
        "population subject",
        "population mean",
        "transcriptome",
        "drug",
        "drug:101 relative",
        "pulsed",
        "electroporation",
        "drug+cytokine",
        "hypoxia",
        "functional_capacity",
        "acute",
        "absolute_error",
        "identified_population_effect",
        "calibration",
        "safety-1",
        "all blockers are surfaced",
    ):
        assert detail in message


def _constructed_belief_for_preflight(query: StateQuery | None = None) -> CellStateBelief:
    query = query or query_factory()
    subject = subject_factory()
    provenance = ProvenanceRecord(
        model_id="api-contract-test",
        model_version="2.0",
        model_fingerprint=_MODEL_FINGERPRINT,
        posterior_schema_id="cellstate/test-v2",
        query_fingerprint=query.fingerprint,
        history_fingerprint="c" * 64,
        history_structure_fingerprint="d" * 64,
        context_fingerprint="e" * 64,
        seed=0,
    )
    return CellStateBelief.model_construct(
        belief_id=uuid4(),
        subject=subject,
        as_of_seconds=10,
        query=query,
        query_fingerprint=query.fingerprint,
        history_fingerprint=provenance.history_fingerprint,
        context_fingerprint=provenance.context_fingerprint,
        state_specification=_state_specification(query),
        context=ContextBelief(),
        provenance=provenance,
    )


class _WrongScopeEvolutionModel:
    descriptor = _descriptor()

    def capabilities(
        self,
        belief: CellStateBelief,
        scenario: EvolutionScenario,
    ) -> CapabilityReport:
        return CapabilityReport(supported=True, scope_fingerprint="0" * 64)

    def evolve(
        self,
        belief: CellStateBelief,
        scenario: EvolutionScenario,
        *,
        options: InferenceOptions,
    ) -> StateForecast:
        raise AssertionError("evolve must not run after a scope mismatch")


class _WrongScopePlanner:
    descriptor = _descriptor()

    def capabilities(
        self,
        belief: CellStateBelief,
        objective: InterventionObjective,
        candidates: Sequence[EvolutionScenario],
    ) -> CapabilityReport:
        return CapabilityReport(supported=True, scope_fingerprint="0" * 64)

    def choose(
        self,
        belief: CellStateBelief,
        objective: InterventionObjective,
        candidates: Sequence[EvolutionScenario],
        *,
        options: InferenceOptions,
    ) -> InterventionPlan:
        raise AssertionError("choose must not run after a scope mismatch")


def test_evolution_and_planning_require_exact_capability_scopes() -> None:
    belief = _constructed_belief_for_preflight()
    scenario = EvolutionScenario(
        scenario_id="baseline",
        horizon_name="acute",
        subject=belief.subject,
        start_time_seconds=10,
        end_time_seconds=70,
    )
    with pytest.raises(ContractViolationError, match="exact requested scope"):
        evolve_cell_state(
            belief,
            scenario=scenario,
            evolution_model=_WrongScopeEvolutionModel(),
        )

    objective = InterventionObjective(
        objective_id="capacity",
        horizon_name="acute",
        terms=(
            ObjectiveTerm(
                target=OntologyTerm(label="functional capacity"),
                direction=ObjectiveDirection.MAXIMIZE,
            ),
        ),
    )
    with pytest.raises(ContractViolationError, match="exact requested scope"):
        choose_intervention(
            belief,
            objective=objective,
            candidates=(scenario,),
            planner=_WrongScopePlanner(),
        )


def test_planning_objective_target_must_support_its_declared_horizon() -> None:
    base_query = query_factory()
    late_horizon = base_query.prediction_horizons[0].model_copy(
        update={"name": "late", "duration_seconds": 120.0}
    )
    query = base_query.model_copy(
        update={"prediction_horizons": (*base_query.prediction_horizons, late_horizon)}
    )
    belief = _constructed_belief_for_preflight(query)
    scenario = EvolutionScenario(
        scenario_id="late-candidate",
        horizon_name="late",
        subject=belief.subject,
        start_time_seconds=10,
        end_time_seconds=130,
    )
    objective = InterventionObjective(
        objective_id="unsupported-target-horizon",
        horizon_name="late",
        terms=(
            ObjectiveTerm(
                target=base_query.target_outputs[0].term,
                direction=ObjectiveDirection.MAXIMIZE,
            ),
        ),
    )
    with pytest.raises(ContractViolationError, match="does not support horizon"):
        choose_intervention(
            belief,
            objective=objective,
            candidates=(scenario,),
            planner=_WrongScopePlanner(),
        )


@pytest.mark.parametrize(
    "scenario",
    (
        EvolutionScenario(
            scenario_id="dose-outside-domain",
            horizon_name="acute",
            subject=subject_factory(),
            start_time_seconds=10,
            end_time_seconds=70,
            interventions=(
                intervention_factory(
                    event_id="excess-dose",
                    time_seconds=10,
                    dose=101,
                    estimated_efficiency=None,
                ),
            ),
        ),
        EvolutionScenario(
            scenario_id="combination-outside-domain",
            horizon_name="acute",
            subject=subject_factory(),
            start_time_seconds=10,
            end_time_seconds=70,
            interventions=(
                intervention_factory(
                    event_id="first-dose",
                    time_seconds=10,
                    estimated_efficiency=None,
                ),
                intervention_factory(
                    event_id="second-dose",
                    time_seconds=11,
                    estimated_efficiency=None,
                ),
            ),
        ),
        EvolutionScenario(
            scenario_id="environment-outside-domain",
            horizon_name="acute",
            subject=subject_factory(),
            start_time_seconds=10,
            end_time_seconds=70,
            environments=(
                environment_factory(
                    event_id="undeclared-environment",
                    time_seconds=10,
                    duration_seconds=1,
                ),
            ),
        ),
    ),
    ids=("dose", "combination", "environment"),
)
def test_scenarios_must_be_members_of_bounded_query_domains(
    scenario: EvolutionScenario,
) -> None:
    belief = _constructed_belief_for_preflight()
    with pytest.raises(ContractViolationError, match="bounded"):
        evolve_cell_state(
            belief,
            scenario=scenario,
            evolution_model=_WrongScopeEvolutionModel(),
        )
