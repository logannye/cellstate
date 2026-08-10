from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
from conftest import (
    SYNTHETIC_TEST_OPTIONS,
    environment_factory,
    environment_spec_factory,
    intervention_factory,
    observation_factory,
    query_factory,
    request_factory,
    subject_factory,
)
from pydantic import ValidationError

from cellstate import estimate_cell_state, evolve_cell_state
from cellstate.domain.belief import CellStateBelief, ContextBelief
from cellstate.domain.common import EvidenceStatus, OntologyTerm, Quantity, canonical_fingerprint
from cellstate.domain.events import (
    ActualPerturbation,
    CensoringDirection,
    CollectionEffect,
    EnvironmentEvent,
    EnvironmentTemporalMode,
    EvidenceLink,
    EvidenceRole,
    InterventionSchedule,
    MissingnessReport,
    MissingnessStatus,
    ObservationCollection,
    ObservationEvent,
    PerturbationStatus,
    ScheduleKind,
)
from cellstate.domain.history import CellHistory, LineageHistory
from cellstate.domain.query import (
    CategoricalDomain,
    EnvironmentVariableSpec,
    IntegerRange,
    MissingHistoryPolicy,
    NumericDomain,
    QueryConstraints,
    RealizationEvidenceRequirement,
    ScalarRange,
    ScheduleDomain,
)
from cellstate.domain.request import EstimateCellStateRequest, InferenceOptions
from cellstate.domain.scenarios import EvolutionScenario
from cellstate.domain.specification import CompiledStateSpecification
from cellstate.domain.subjects import BeliefSubject, IdentityBasis, SubjectKind
from cellstate.errors import ContractViolationError, PosteriorCompatibilityError
from cellstate.ports import CapabilityReport


def _aggregate_subject(
    kind: SubjectKind,
    subject_id: str,
    *,
    identity_basis: IdentityBasis | None = None,
) -> BeliefSubject:
    default_basis = {
        SubjectKind.CLONE_LINEAGE: IdentityBasis.HERITABLE_BARCODE,
        SubjectKind.POPULATION: IdentityBasis.EXPERIMENTAL_UNIT,
        SubjectKind.SPATIAL_NICHE: IdentityBasis.SPATIAL_REGION,
    }[kind]
    return BeliefSubject(
        subject_id=subject_id,
        kind=kind,
        biological_system=OntologyTerm(label="synthetic reference cell"),
        membership_semantics=f"declared {kind.value} membership",
        experimental_unit_kind="well",
        experimental_unit_id="well-1",
        identity_basis=identity_basis or default_basis,
    )


class _MutatingEstimator:
    def __init__(
        self,
        model: Any,
        mutate: Callable[[CellStateBelief], Any],
        *,
        descriptor: Any | None = None,
    ) -> None:
        self._model = model
        self._mutate = mutate
        self._descriptor = descriptor or model.descriptor

    @property
    def descriptor(self) -> Any:
        return self._descriptor

    @property
    def query_compiler(self) -> Any:
        return self._model.query_compiler

    def capabilities(
        self,
        request: EstimateCellStateRequest,
        state_specification: CompiledStateSpecification,
    ) -> CapabilityReport:
        return self._model.capabilities(request, state_specification)

    def estimate(
        self,
        request: EstimateCellStateRequest,
        *,
        options: InferenceOptions,
    ) -> Any:
        return self._mutate(self._model.estimate(request, options=options))


class _InvalidCapabilityEstimator(_MutatingEstimator):
    def capabilities(
        self,
        request: EstimateCellStateRequest,
        state_specification: CompiledStateSpecification,
    ) -> Any:
        del request, state_specification
        return object()


class _InvalidCompiler:
    compiler_descriptor = object()

    def compile(self, query: Any) -> Any:
        raise AssertionError(f"compile must not run for {query!r}")


class _InvalidCompilerEstimator(_MutatingEstimator):
    @property
    def query_compiler(self) -> _InvalidCompiler:
        return _InvalidCompiler()


def _replace_provenance(
    belief: CellStateBelief,
    **updates: object,
) -> CellStateBelief:
    return belief.model_copy(update={"provenance": belief.provenance.model_copy(update=updates)})


def _wrong_history_binding(belief: CellStateBelief) -> CellStateBelief:
    fingerprint = "1" * 64
    return _replace_provenance(
        belief.model_copy(update={"history_fingerprint": fingerprint}),
        history_fingerprint=fingerprint,
    )


def _wrong_context_binding(belief: CellStateBelief) -> CellStateBelief:
    fingerprint = "2" * 64
    return _replace_provenance(
        belief.model_copy(update={"context_fingerprint": fingerprint}),
        context_fingerprint=fingerprint,
    )


def _unknown_provenance_event(belief: CellStateBelief) -> CellStateBelief:
    event_ids = (*belief.provenance.source_event_ids, "ghost-event")
    event_fingerprints = {
        **belief.provenance.source_event_fingerprints,
        "ghost-event": "3" * 64,
    }
    return _replace_provenance(
        belief,
        source_event_ids=event_ids,
        source_event_fingerprints=event_fingerprints,
    )


def _tampered_provenance_event(belief: CellStateBelief) -> CellStateBelief:
    event_id = belief.provenance.source_event_ids[0]
    return _replace_provenance(
        belief,
        source_event_fingerprints={
            **belief.provenance.source_event_fingerprints,
            event_id: "4" * 64,
        },
    )


def _omitted_provenance_event(belief: CellStateBelief) -> CellStateBelief:
    omitted = belief.provenance.source_event_ids[0]
    provenance = belief.provenance.model_copy(
        update={
            "source_event_ids": tuple(
                event_id for event_id in belief.provenance.source_event_ids if event_id != omitted
            ),
            "source_event_fingerprints": {
                event_id: fingerprint
                for event_id, fingerprint in belief.provenance.source_event_fingerprints.items()
                if event_id != omitted
            },
        }
    )
    factors = tuple(
        factor.model_copy(
            update={
                "evidence_event_ids": tuple(
                    event_id for event_id in factor.evidence_event_ids if event_id != omitted
                )
            }
        )
        for factor in belief.factors
    )
    return belief.model_copy(update={"provenance": provenance, "factors": factors})


@pytest.mark.parametrize(
    ("mutate", "message"),
    (
        (lambda belief: belief.model_copy(update={"as_of_seconds": 11}), "wrong time"),
        (_wrong_history_binding, "different history"),
        (_wrong_context_binding, "different static/population context"),
        (
            lambda belief: _replace_provenance(belief, seed=99),
            "wrong inference seed",
        ),
        (
            lambda belief: _replace_provenance(
                belief,
                history_structure_fingerprint="5" * 64,
            ),
            "different history structure",
        ),
        (_unknown_provenance_event, "unknown source events"),
        (_tampered_provenance_event, "fingerprint disagrees"),
        (_omitted_provenance_event, "omits eligible request history events"),
    ),
    ids=(
        "time",
        "history",
        "context",
        "seed",
        "history-structure",
        "unknown-event",
        "tampered-event",
        "omitted-event",
    ),
)
def test_estimation_boundary_rejects_valid_but_misbound_results(
    model: Any,
    mutate: Callable[[CellStateBelief], CellStateBelief],
    message: str,
) -> None:
    request = request_factory()
    estimator = _MutatingEstimator(model, mutate)

    with pytest.raises(ContractViolationError, match=message):
        estimate_cell_state(
            request,
            estimator=estimator,
            options=SYNTHETIC_TEST_OPTIONS,
        )


def test_estimation_boundary_normalizes_backend_contracts_before_use(model: Any) -> None:
    request = request_factory()

    with pytest.raises(ContractViolationError, match="invalid capability report"):
        estimate_cell_state(
            request,
            estimator=_InvalidCapabilityEstimator(model, lambda belief: belief),
            options=SYNTHETIC_TEST_OPTIONS,
        )

    with pytest.raises(ContractViolationError, match="invalid compiled state"):
        estimate_cell_state(
            request,
            estimator=_InvalidCompilerEstimator(model, lambda belief: belief),
            options=SYNTHETIC_TEST_OPTIONS,
        )

    with pytest.raises(ContractViolationError, match="invalid belief contract"):
        estimate_cell_state(
            request,
            estimator=_MutatingEstimator(model, lambda belief: object()),
            options=SYNTHETIC_TEST_OPTIONS,
        )


def test_estimation_boundary_checks_model_identity_after_result_validation(model: Any) -> None:
    incompatible_descriptor = model.descriptor.model_copy(update={"model_fingerprint": "6" * 64})
    estimator = _MutatingEstimator(
        model,
        lambda belief: belief,
        descriptor=incompatible_descriptor,
    )

    with pytest.raises(PosteriorCompatibilityError, match="incompatible model"):
        estimate_cell_state(
            request_factory(),
            estimator=estimator,
            options=SYNTHETIC_TEST_OPTIONS,
        )


def test_valid_abstaining_belief_returns_as_a_structured_scientific_result(model: Any) -> None:
    request = request_factory()
    belief = estimate_cell_state(request, estimator=model)
    assert belief.readiness.abstention_required


def test_observed_factor_requires_direct_cutoff_evidence(model: Any) -> None:
    request = request_factory()

    def mark_stale_evidence_observed(belief: CellStateBelief) -> CellStateBelief:
        first, *rest = belief.factors
        observed = first.model_copy(
            update={
                "evidence_status": EvidenceStatus.OBSERVED,
                "evidence_event_ids": ("obs-0",),
            }
        )
        return belief.model_copy(update={"factors": (observed, *rest)})

    with pytest.raises(ContractViolationError, match="ending at the belief time"):
        estimate_cell_state(
            request,
            estimator=_MutatingEstimator(model, mark_stale_evidence_observed),
            options=SYNTHETIC_TEST_OPTIONS,
        )


def test_observed_factor_cannot_cite_missing_measurement(model: Any) -> None:
    missing = observation_factory(
        event_id="missing-observation",
        value=None,
        missingness=MissingnessReport(
            status=MissingnessStatus.MISSING,
            reason="instrument dropout",
        ),
    )
    request = request_factory(history=CellHistory(subject=subject_factory(), events=(missing,)))

    def cite_missing_measurement(belief: CellStateBelief) -> CellStateBelief:
        first, *rest = belief.factors
        observed = first.model_copy(
            update={
                "evidence_status": EvidenceStatus.OBSERVED,
                "evidence_event_ids": (missing.event_id,),
            }
        )
        return belief.model_copy(update={"factors": (observed, *rest)})

    with pytest.raises(ContractViolationError, match="observed measurement events"):
        estimate_cell_state(
            request,
            estimator=_MutatingEstimator(model, cite_missing_measurement),
            options=SYNTHETIC_TEST_OPTIONS,
        )


def test_recursive_updates_reject_omitted_and_late_evidence(model: Any) -> None:
    original = request_factory()
    belief = estimate_cell_state(
        original,
        estimator=model,
        options=SYNTHETIC_TEST_OPTIONS,
    )
    empty_history = original.history.model_copy(update={"events": ()})
    with pytest.raises(ValidationError, match="omits events"):
        EstimateCellStateRequest(
            query=original.query,
            history=empty_history,
            as_of_seconds=11,
            static_context=original.static_context,
            previous_belief=belief,
        )

    late = observation_factory(event_id="late-arrival", time_seconds=5)
    extended_history = original.history.model_copy(
        update={"events": (*original.history.events, late)}
    )
    with pytest.raises(ValidationError, match="smoothing backend"):
        EstimateCellStateRequest(
            query=original.query,
            history=extended_history,
            as_of_seconds=11,
            static_context=original.static_context,
            previous_belief=belief,
        )


def test_request_rejects_subject_cutoff_modality_role_and_evidence_count() -> None:
    request = request_factory()
    wrong_subject = _aggregate_subject(SubjectKind.POPULATION, "population-1")
    with pytest.raises(ValidationError, match="subject specification"):
        EstimateCellStateRequest(
            query=request.query,
            history=CellHistory(subject=wrong_subject),
            as_of_seconds=10,
            static_context=request.static_context,
        )

    cutoff_query = request.query.model_copy(
        update={
            "evidence_policy": request.query.evidence_policy.model_copy(
                update={"include_at_cutoff": False}
            )
        }
    )
    cutoff_observation = observation_factory(time_seconds=10)
    with pytest.raises(ValidationError, match="excludes observations ending at the cutoff"):
        request_factory(
            query=cutoff_query,
            history=CellHistory(subject=subject_factory(), events=(cutoff_observation,)),
        )

    unsupported_modality = observation_factory(modality="unregistered imaging channel")
    with pytest.raises(ValidationError, match="modalities outside"):
        request_factory(
            history=CellHistory(subject=subject_factory(), events=(unsupported_modality,))
        )

    external = observation_factory(
        source_subject=subject_factory("atlas-cell"),
        evidence_role=EvidenceRole.EXTERNAL_REFERENCE,
        linkage_basis=IdentityBasis.EXTERNAL_REFERENCE,
    )
    with pytest.raises(ValidationError, match="roles outside"):
        request_factory(history=CellHistory(subject=subject_factory(), events=(external,)))

    minimum_query = request.query.model_copy(
        update={
            "evidence_policy": request.query.evidence_policy.model_copy(
                update={"minimum_observed_measurements": 1}
            )
        }
    )
    missing = observation_factory(
        value=None,
        missingness=MissingnessReport(status=MissingnessStatus.ASSAY_FAILURE),
    )
    with pytest.raises(ValidationError, match="minimum observed"):
        request_factory(
            query=minimum_query,
            history=CellHistory(subject=subject_factory(), events=(missing,)),
        )


def test_request_rejects_unsupported_or_incomplete_causal_history() -> None:
    query = query_factory()
    unsupported_environment = environment_factory(variables={"oxygen": "hypoxic"})
    with pytest.raises(ValidationError, match="environment events outside"):
        request_factory(
            query=query,
            history=CellHistory(
                subject=subject_factory(),
                events=(observation_factory(), unsupported_environment),
            ),
        )

    required_environment_query = query.model_copy(
        update={
            "environment_space": (environment_spec_factory(),),
            "evidence_policy": query.evidence_policy.model_copy(update={"lookback_seconds": 10.0}),
        }
    )
    with pytest.raises(ValidationError, match="does not cover required"):
        request_factory(query=required_environment_query)

    unsupported_intervention = intervention_factory(dose=101)
    with pytest.raises(ValidationError, match="interventions outside"):
        request_factory(
            history=CellHistory(
                subject=subject_factory(),
                events=(observation_factory(), unsupported_intervention),
            )
        )

    realization_spec = query.intervention_space[0].model_copy(
        update={
            "realization_evidence": RealizationEvidenceRequirement(
                allowed_statuses=(PerturbationStatus.MEASURED,),
                allowed_modalities=(OntologyTerm(label="transcriptome"),),
                minimum_evidence_events=1,
            )
        }
    )
    realization_query = query.model_copy(update={"intervention_space": (realization_spec,)})
    intended_only = intervention_factory(actual_perturbation=None)
    realization_request = request_factory(
        query=realization_query,
        history=CellHistory(
            subject=subject_factory(),
            events=(observation_factory(), intended_only),
        ),
    )
    observations = {
        event.event_id: event
        for event in realization_request.history.events
        if isinstance(event, ObservationEvent)
    }
    assert realization_query.realization_evidence_gaps(intended_only, observations) == (
        "realization_not_assessed",
    )

    complete_query = query.model_copy(
        update={
            "constraints": query.constraints.model_copy(
                update={"require_complete_lineage_history": True}
            )
        }
    )
    with pytest.raises(ValidationError, match="complete history records"):
        EstimateCellStateRequest(
            query=complete_query,
            history=CellHistory(subject=subject_factory(), events=(observation_factory(),)),
            as_of_seconds=10,
            static_context=request_factory().static_context,
        )

    future_lineage = CellHistory(
        subject=subject_factory(),
        events=(observation_factory(),),
        lineage=LineageHistory(division_times_seconds=(11,)),
    )
    with pytest.raises(ValidationError, match="division after"):
        EstimateCellStateRequest(
            query=query,
            history=future_lineage,
            as_of_seconds=10,
            static_context=request_factory().static_context,
        )


@pytest.mark.parametrize(
    ("role", "source", "target", "basis", "message"),
    (
        (
            EvidenceRole.EXTERNAL_REFERENCE,
            subject_factory("same"),
            subject_factory("same"),
            IdentityBasis.EXTERNAL_REFERENCE,
            "distinct source and target",
        ),
        (
            EvidenceRole.ANCESTOR,
            _aggregate_subject(SubjectKind.POPULATION, "source-population"),
            subject_factory("target-cell"),
            IdentityBasis.OBSERVED_PARENTAGE,
            "source must be an individual",
        ),
        (
            EvidenceRole.DESCENDANT,
            subject_factory("source-cell"),
            _aggregate_subject(SubjectKind.POPULATION, "target-population"),
            IdentityBasis.OBSERVED_PARENTAGE,
            "target must be an individual or lineage",
        ),
        (
            EvidenceRole.SIBLING,
            subject_factory("source-cell"),
            subject_factory("target-cell"),
            IdentityBasis.EXTERNAL_REFERENCE,
            "explicit lineage linkage",
        ),
        (
            EvidenceRole.CLONE_AGGREGATE,
            subject_factory("source-cell"),
            subject_factory("target-cell"),
            IdentityBasis.OBSERVED_PARENTAGE,
            "clone/lineage source",
        ),
        (
            EvidenceRole.MATCHED_POPULATION,
            subject_factory("source-cell"),
            subject_factory("target-cell"),
            IdentityBasis.MATCHED_EXPERIMENTAL_DESIGN,
            "population source",
        ),
        (
            EvidenceRole.GENERAL_POPULATION,
            subject_factory("source-cell"),
            subject_factory("target-cell"),
            IdentityBasis.DECLARED_MEMBERSHIP,
            "aggregate target",
        ),
        (
            EvidenceRole.SPATIAL_NEIGHBOR,
            subject_factory("source-cell"),
            subject_factory("target-cell"),
            IdentityBasis.EXTERNAL_REFERENCE,
            "spatial linkage",
        ),
        (
            EvidenceRole.EXTERNAL_REFERENCE,
            subject_factory("source-cell"),
            subject_factory("target-cell"),
            IdentityBasis.OBSERVED_PARENTAGE,
            "external or transport linkage",
        ),
    ),
    ids=(
        "nondirect-same-subject",
        "lineage-source",
        "lineage-target",
        "lineage-basis",
        "clone-source",
        "matched-source",
        "population-target",
        "spatial-basis",
        "external-basis",
    ),
)
def test_evidence_roles_reject_unjustified_subject_casts(
    role: EvidenceRole,
    source: BeliefSubject,
    target: BeliefSubject,
    basis: IdentityBasis,
    message: str,
) -> None:
    with pytest.raises(ValidationError, match=message):
        EvidenceLink(
            source_subject=source,
            target_subject=target,
            role=role,
            linkage_basis=basis,
            linkage_confidence=0.9,
            linkage_details="explicit invalid-link test",
            sampling_unit_id="well-1",
        )


@pytest.mark.parametrize(
    "payload",
    (
        {"effect": CollectionEffect.VIABILITY_PRESERVING_WITH_KNOWN_EFFECT},
        {
            "effect": CollectionEffect.NONDESTRUCTIVE,
            "effect_description": "not applicable",
        },
        {"effect": CollectionEffect.PARTIALLY_DESTRUCTIVE_POPULATION_SAMPLING},
        {"effect": CollectionEffect.NONDESTRUCTIVE, "sampling_fraction": 0.1},
    ),
    ids=("missing-effect", "extra-effect", "missing-fraction", "extra-fraction"),
)
def test_collection_effects_require_exactly_their_scientific_metadata(
    payload: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        ObservationCollection.model_validate(payload)


@pytest.mark.parametrize(
    "payload",
    (
        {"status": MissingnessStatus.BELOW_DETECTION},
        {
            "status": MissingnessStatus.BELOW_DETECTION,
            "detection_limit": Quantity(value=1, units="count"),
            "censoring_direction": CensoringDirection.BELOW,
        },
        {"status": MissingnessStatus.CENSORED},
        {
            "status": MissingnessStatus.CENSORED,
            "censoring_direction": CensoringDirection.ABOVE,
        },
        {
            "status": MissingnessStatus.CENSORED,
            "censoring_direction": CensoringDirection.INTERVAL,
            "detection_limit": Quantity(value=1, units="count"),
        },
        {
            "status": MissingnessStatus.CENSORED,
            "censoring_direction": CensoringDirection.INTERVAL,
            "interval_lower": Quantity(value=1, units="count"),
        },
        {
            "status": MissingnessStatus.OBSERVED,
            "detection_limit": Quantity(value=1, units="count"),
        },
    ),
    ids=(
        "below-no-limit",
        "below-double-coded",
        "censored-no-direction",
        "one-sided-no-limit",
        "interval-with-limit",
        "interval-missing-bound",
        "observed-with-bounds",
    ),
)
def test_missingness_never_uses_implicit_or_contradictory_censoring(
    payload: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        MissingnessReport.model_validate(payload)


@pytest.mark.parametrize(
    "payload",
    (
        {"status": PerturbationStatus.MEASURED},
        {"status": PerturbationStatus.INFERRED, "efficiency": 0.5},
        {
            "status": PerturbationStatus.FAILED,
            "efficiency": 0.5,
            "evidence_event_ids": ("qc",),
        },
        {"status": PerturbationStatus.FAILED, "efficiency": 0},
        {"status": PerturbationStatus.UNKNOWN, "efficiency": 0.5},
        {
            "status": PerturbationStatus.MEASURED,
            "efficiency": 0.5,
            "evidence_event_ids": ("qc", "qc"),
        },
    ),
    ids=(
        "measured-no-efficiency",
        "inferred-no-evidence",
        "failed-nonzero",
        "failed-no-evidence",
        "unknown-with-efficiency",
        "duplicate-evidence",
    ),
)
def test_realized_perturbations_require_status_consistent_evidence(
    payload: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        ActualPerturbation.model_validate(payload)


def test_schedule_domains_reject_unrepresentable_shapes_and_events() -> None:
    with pytest.raises(ValidationError, match="requires an interval"):
        ScheduleDomain(
            allowed_kinds=(ScheduleKind.PULSED,),
            administration_count=IntegerRange(minimum=2, maximum=3),
            interval_seconds=None,
            washout_seconds=ScalarRange(minimum=0, maximum=60),
        )
    with pytest.raises(ValidationError, match="invalid without repeated"):
        ScheduleDomain(
            allowed_kinds=(ScheduleKind.SINGLE,),
            administration_count=IntegerRange(minimum=1, maximum=1),
            interval_seconds=ScalarRange(minimum=1, maximum=10),
            washout_seconds=ScalarRange(minimum=0, maximum=60),
        )
    with pytest.raises(ValidationError, match="count >= 2"):
        InterventionSchedule(
            kind=ScheduleKind.REPEATED,
            administration_count=1,
            interval_seconds=10,
            washout_seconds=0,
        )
    with pytest.raises(ValidationError, match="count 1"):
        InterventionSchedule(
            kind=ScheduleKind.CONTINUOUS,
            administration_count=2,
            washout_seconds=0,
        )


def test_environment_defaults_and_values_remain_inside_declared_domains() -> None:
    common = {
        "variable": OntologyTerm(label="medium"),
        "domain": CategoricalDomain(values=("RPMI", "DMEM")),
        "duration_seconds": ScalarRange(minimum=0, maximum=60),
        "required": True,
        "allowed_temporal_modes": (EnvironmentTemporalMode.FIXED,),
    }
    with pytest.raises(ValidationError, match="only use-declared-default"):
        EnvironmentVariableSpec(
            **common,
            missing_history_policy=MissingHistoryPolicy.REJECT,
            default_value="RPMI",
        )
    with pytest.raises(ValidationError, match="outside its declared domain"):
        EnvironmentVariableSpec(
            **common,
            missing_history_policy=MissingHistoryPolicy.USE_DECLARED_DEFAULT,
            default_value="MEM",
        )

    numeric = EnvironmentVariableSpec(
        variable=OntologyTerm(label="oxygen"),
        domain=NumericDomain(minimum=0, maximum=21, units="percent"),
        duration_seconds=ScalarRange(minimum=0, maximum=60),
        required=True,
        allowed_temporal_modes=(EnvironmentTemporalMode.FIXED,),
        missing_history_policy=MissingHistoryPolicy.REJECT,
    )
    assert numeric.contains({"value": 5, "units": "percent"})
    assert not numeric.contains("five percent")
    assert not numeric.contains(Quantity(value=5, units="fraction"))
    assert not EnvironmentVariableSpec(
        **common,
        missing_history_policy=MissingHistoryPolicy.REJECT,
    ).contains(Quantity(value=1, units="arbitrary"))


def test_scenarios_fail_before_backend_execution_when_domains_or_evidence_collide(
    model: Any,
) -> None:
    request = request_factory()
    belief = estimate_cell_state(
        request,
        estimator=model,
        options=SYNTHETIC_TEST_OPTIONS,
    )

    reused_evidence_id = EvolutionScenario(
        scenario_id="reused-evidence",
        horizon_name="acute",
        subject=belief.subject,
        start_time_seconds=10,
        end_time_seconds=70,
        interventions=(
            intervention_factory(
                event_id="obs-0",
                time_seconds=10,
                estimated_efficiency=None,
            ),
        ),
    )
    with pytest.raises(ContractViolationError, match="must not reuse"):
        evolve_cell_state(
            belief,
            scenario=reused_evidence_id,
            evolution_model=model,
            options=SYNTHETIC_TEST_OPTIONS,
        )

    unknown_horizon = EvolutionScenario(
        scenario_id="unknown-horizon",
        horizon_name="chronic",
        subject=belief.subject,
        start_time_seconds=10,
        end_time_seconds=70,
    )
    with pytest.raises(ContractViolationError, match="not declared"):
        evolve_cell_state(
            belief,
            scenario=unknown_horizon,
            evolution_model=model,
            options=SYNTHETIC_TEST_OPTIONS,
        )

    wrong_duration = unknown_horizon.model_copy(
        update={"scenario_id": "wrong-duration", "horizon_name": "acute", "end_time_seconds": 69}
    )
    with pytest.raises(ContractViolationError, match="duration does not match"):
        evolve_cell_state(
            belief,
            scenario=wrong_duration,
            evolution_model=model,
            options=SYNTHETIC_TEST_OPTIONS,
        )


def test_environment_scenarios_must_explicitly_cover_or_inherit_required_values(
    model: Any,
) -> None:
    base_query = query_factory()
    query = base_query.model_copy(
        update={
            "environment_space": (environment_spec_factory(),),
            "evidence_policy": base_query.evidence_policy.model_copy(
                update={"lookback_seconds": 10.0}
            ),
        }
    )
    environment = environment_factory(time_seconds=0, duration_seconds=10)
    request = request_factory(
        query=query,
        history=CellHistory(
            subject=subject_factory(),
            events=(observation_factory(), environment),
        ),
    )
    belief = estimate_cell_state(
        request,
        estimator=model,
        options=SYNTHETIC_TEST_OPTIONS,
    )
    clear_without_replacement = EvolutionScenario(
        scenario_id="clear-required-environment",
        horizon_name="acute",
        subject=belief.subject,
        start_time_seconds=10,
        end_time_seconds=70,
        inherit_current_environment=False,
    )
    with pytest.raises(ContractViolationError, match="does not supply all required"):
        evolve_cell_state(
            belief,
            scenario=clear_without_replacement,
            evolution_model=model,
            options=SYNTHETIC_TEST_OPTIONS,
        )

    no_inherited_value = belief.model_copy(
        update={"context": ContextBelief(soluble_environment={})}
    )
    inherit_missing = clear_without_replacement.model_copy(
        update={
            "scenario_id": "inherit-missing-environment",
            "inherit_current_environment": True,
        }
    )
    with pytest.raises(ContractViolationError, match="does not cover required"):
        evolve_cell_state(
            no_inherited_value,
            scenario=inherit_missing,
            evolution_model=model,
            options=SYNTHETIC_TEST_OPTIONS,
        )

    out_of_domain_context = belief.model_copy(
        update={
            "context": ContextBelief(
                soluble_environment={"nutrient": {"value": 1_000, "units": "relative"}}
            )
        }
    )
    inherit_invalid = inherit_missing.model_copy(
        update={"scenario_id": "inherit-invalid-environment"}
    )
    with pytest.raises(ContractViolationError, match="inherited environment variable"):
        evolve_cell_state(
            out_of_domain_context,
            scenario=inherit_invalid,
            evolution_model=model,
            options=SYNTHETIC_TEST_OPTIONS,
        )


def test_environment_event_key_normalization_is_part_of_domain_membership() -> None:
    event = EnvironmentEvent(
        event_id="medium",
        subject=subject_factory(),
        time_seconds=0,
        variables={"Nutrient": Quantity(value=1, units="relative")},
        duration_seconds=1,
        temporal_mode=EnvironmentTemporalMode.FIXED,
    )
    assert event.variables == {"nutrient": Quantity(value=1, units="relative")}
    assert canonical_fingerprint(event) == canonical_fingerprint(
        event.model_copy(update={"variables": {"nutrient": Quantity(value=1, units="relative")}})
    )


def _constraints(**updates: object) -> QueryConstraints:
    payload = query_factory().constraints.model_dump(mode="python")
    payload.update(updates)
    return QueryConstraints.model_validate(payload)


def test_combination_rules_are_canonical_bounded_and_noncontradictory() -> None:
    with pytest.raises(ValidationError, match="at least two"):
        _constraints(allowed_combinations=(("drug",),))
    with pytest.raises(ValidationError, match="repeat a spec ID"):
        _constraints(allowed_combinations=(("drug", "drug"),))
    with pytest.raises(ValidationError, match="both allowed and forbidden"):
        _constraints(
            maximum_intervention_combination_order=2,
            allowed_combinations=(("drug", "stimulus"),),
            forbidden_combinations=(("stimulus", "drug"),),
        )
    with pytest.raises(ValidationError, match="exceeds maximum"):
        _constraints(
            maximum_intervention_combination_order=2,
            allowed_combinations=(("a", "b", "c"),),
        )
