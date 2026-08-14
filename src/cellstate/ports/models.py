"""Stable high-level ports plus composable biological model-stage protocols."""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum
from typing import Protocol, TypeVar, runtime_checkable

from pydantic import Field, field_validator, model_validator

from cellstate.domain.belief import (
    BeliefDiagnostics,
    CellStateBelief,
    IdentifiabilityReport,
    InterventionRealizationBelief,
    SufficiencyReport,
    SupportReport,
    UncertaintyBreakdown,
)
from cellstate.domain.common import (
    CausalStatus,
    CriterionOutcome,
    SchemaModel,
    SupportStatus,
    canonical_fingerprint,
)
from cellstate.domain.distributions import StateDistribution
from cellstate.domain.events import CollectionEffect, ObservationEvent
from cellstate.domain.history import CellHistory
from cellstate.domain.measurements import MeasurementDecisionRequest, MeasurementRecommendation
from cellstate.domain.query import OutputSpec, StateQuery
from cellstate.domain.request import EstimateCellStateRequest, InferenceOptions
from cellstate.domain.scenarios import (
    EvolutionScenario,
    InterventionObjective,
    InterventionPlan,
    StateForecast,
)
from cellstate.domain.specification import CompiledStateSpecification
from cellstate.domain.subjects import BeliefSubject


class ModelArtifactKind(StrEnum):
    CONTRACT_REFERENCE = "contract_reference"
    BIOLOGICAL_MODEL = "biological_model"
    SYNTHETIC_TEST_MODEL = "synthetic_test_model"
    EMPIRICAL_OBSERVATION_MODEL = "empirical_observation_model"
    """Fitted on real public bytes; reports state and uncertainty; claims nothing causal.

    Authorized by ADR 0021.  It carries a biological model's provenance obligations -- the whole
    reason ``CONTRACT_REFERENCE`` is unusable for real data is that it is forbidden to cite what it
    was fit on -- and ``estimate_cell_state`` bars it from an identified or transported causal
    claim, so it cannot serve as a route around the admission registry that still gates
    ``BIOLOGICAL_MODEL``.
    """


class EstimatorDescriptor(SchemaModel):
    """Artifact identity without a scientifically meaningless global validation Boolean."""

    model_id: str = Field(min_length=1)
    model_version: str = Field(min_length=1)
    model_fingerprint: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    posterior_schema_id: str = Field(min_length=1)
    description: str = Field(min_length=1)
    artifact_kind: ModelArtifactKind
    support_envelope_id: str | None = None
    support_envelope_fingerprint: str | None = Field(default=None, pattern=r"^[a-fA-F0-9]{64}$")
    training_support_id: str | None = None
    training_support_fingerprint: str | None = Field(default=None, pattern=r"^[a-fA-F0-9]{64}$")
    validation_evidence_ids: tuple[str, ...] = ()
    validation_evidence_fingerprints: dict[str, str] = Field(default_factory=dict)

    @field_validator("validation_evidence_ids")
    @classmethod
    def unique_validation_evidence(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if len(values) != len(set(values)):
            raise ValueError("validation evidence IDs must be unique")
        return values

    @model_validator(mode="after")
    def references_match_artifact_kind(self) -> EstimatorDescriptor:
        required_text = (
            self.model_id,
            self.model_version,
            self.posterior_schema_id,
            self.description,
        )
        optional_text = (
            self.support_envelope_id,
            self.training_support_id,
        )
        if any(not value.strip() for value in required_text) or any(
            value is not None and not value.strip() for value in optional_text
        ):
            raise ValueError("model descriptor identifiers and description must be nonblank")
        if any(not evidence_id.strip() for evidence_id in self.validation_evidence_ids):
            raise ValueError("model validation evidence IDs must be nonblank")
        if set(self.validation_evidence_fingerprints) != set(self.validation_evidence_ids):
            raise ValueError(
                "a descriptor requires exactly one fingerprint per validation evidence artifact"
            )
        if (self.support_envelope_id is None) is not (self.support_envelope_fingerprint is None):
            raise ValueError("support envelope ID and fingerprint must be declared together")
        if (self.training_support_id is None) is not (self.training_support_fingerprint is None):
            raise ValueError("training support ID and fingerprint must be declared together")
        if self.artifact_kind is ModelArtifactKind.CONTRACT_REFERENCE and (
            self.support_envelope_id is not None
            or self.support_envelope_fingerprint is not None
            or self.training_support_id is not None
            or self.training_support_fingerprint is not None
            or self.validation_evidence_ids
            or self.validation_evidence_fingerprints
        ):
            raise ValueError("a contract reference cannot claim biological support evidence")
        if self.artifact_kind in {
            ModelArtifactKind.BIOLOGICAL_MODEL,
            ModelArtifactKind.SYNTHETIC_TEST_MODEL,
            ModelArtifactKind.EMPIRICAL_OBSERVATION_MODEL,
        } and (
            self.support_envelope_id is None
            or self.support_envelope_fingerprint is None
            or self.training_support_id is None
            or self.training_support_fingerprint is None
            or not self.validation_evidence_ids
            or not self.validation_evidence_fingerprints
        ):
            raise ValueError(
                "a supported model descriptor requires training, validation, and a support envelope"
            )
        return self


class QueryCompilerDescriptor(SchemaModel):
    compiler_id: str = Field(min_length=1)
    compiler_version: str = Field(min_length=1)
    compiler_fingerprint: str = Field(pattern=r"^[a-fA-F0-9]{64}$")


class CapabilityReport(SchemaModel):
    """Preflight calculation support for one exact query/request/scenario scope."""

    supported: bool
    scope_fingerprint: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    unsupported_system_boundary: str | None = None
    unsupported_subjects: tuple[str, ...] = ()
    unsupported_aggregations: tuple[str, ...] = ()
    unsupported_modalities: tuple[str, ...] = ()
    unsupported_interventions: tuple[str, ...] = ()
    unsupported_doses: tuple[str, ...] = ()
    unsupported_schedules: tuple[str, ...] = ()
    unsupported_delivery_methods: tuple[str, ...] = ()
    unsupported_combinations: tuple[str, ...] = ()
    unsupported_environments: tuple[str, ...] = ()
    unsupported_outputs: tuple[str, ...] = ()
    unsupported_horizons: tuple[str, ...] = ()
    unsupported_precision_requirements: tuple[str, ...] = ()
    unsupported_causal_classes: tuple[CausalStatus, ...] = ()
    unsupported_readiness_criteria: tuple[str, ...] = ()
    unsupported_constraints: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    @property
    def blockers(self) -> tuple[str, ...]:
        boundary = (
            (self.unsupported_system_boundary,)
            if self.unsupported_system_boundary is not None
            else ()
        )
        return (
            *boundary,
            *self.unsupported_subjects,
            *self.unsupported_aggregations,
            *self.unsupported_modalities,
            *self.unsupported_interventions,
            *self.unsupported_doses,
            *self.unsupported_schedules,
            *self.unsupported_delivery_methods,
            *self.unsupported_combinations,
            *self.unsupported_environments,
            *self.unsupported_outputs,
            *self.unsupported_horizons,
            *self.unsupported_precision_requirements,
            *(status.value for status in self.unsupported_causal_classes),
            *self.unsupported_readiness_criteria,
            *self.unsupported_constraints,
        )

    @model_validator(mode="after")
    def supported_cannot_have_blockers(self) -> CapabilityReport:
        if self.supported and self.blockers:
            raise ValueError("a supported capability report cannot declare unsupported features")
        return self


class MeasurementCapabilityReport(SchemaModel):
    """Calculation and scientific support for one exact measurement decision scope."""

    supported: bool
    scope_fingerprint: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    assay_support: dict[str, SupportStatus]
    collection_effect_support: dict[CollectionEffect, SupportStatus]
    assay_outcome_model: CriterionOutcome
    hypothetical_update: CriterionOutcome
    counterfactual_replanning: CriterionOutcome
    decision_utility: CriterionOutcome
    blockers: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    @model_validator(mode="after")
    def calculation_support_matches_blockers(self) -> MeasurementCapabilityReport:
        if self.supported and self.blockers:
            raise ValueError("a supported measurement capability report cannot have blockers")
        if not self.supported and not self.blockers:
            raise ValueError("an unsupported measurement capability report requires blockers")
        if not self.assay_support:
            raise ValueError("measurement capability must classify every requested assay")
        if any(not assay_id for assay_id in self.assay_support):
            raise ValueError("measurement capability assay IDs must be nonempty")
        if set(self.collection_effect_support) != set(CollectionEffect):
            raise ValueError("measurement capability must classify every collection effect")
        return self


def estimation_capability_scope_fingerprint(
    request: EstimateCellStateRequest,
    state_specification: CompiledStateSpecification,
) -> str:
    """Bind estimation preflight to the exact request and compiled state contract."""

    return canonical_fingerprint(
        {
            "operation": "estimate_cell_state",
            "request": request.model_dump(mode="json"),
            "state_specification": state_specification.model_dump(mode="json"),
        }
    )


def evolution_capability_scope_fingerprint(
    belief: CellStateBelief,
    scenario: EvolutionScenario,
) -> str:
    """Bind an evolution preflight to the exact posterior and controlled scenario."""

    return canonical_fingerprint(
        {
            "operation": "evolve_cell_state",
            "belief": belief.model_dump(mode="json"),
            "scenario": scenario.model_dump(mode="json"),
        }
    )


def planning_capability_scope_fingerprint(
    belief: CellStateBelief,
    objective: InterventionObjective,
    candidates: Sequence[EvolutionScenario],
) -> str:
    """Bind a planning preflight to the exact ordered candidate decision problem."""

    return canonical_fingerprint(
        {
            "operation": "choose_intervention",
            "belief": belief.model_dump(mode="json"),
            "objective": objective.model_dump(mode="json"),
            "candidates": [candidate.model_dump(mode="json") for candidate in candidates],
        }
    )


def measurement_capability_scope_fingerprint(
    belief: CellStateBelief,
    request: MeasurementDecisionRequest,
) -> str:
    """Bind measurement preflight to the exact posterior and decision problem."""

    return canonical_fingerprint(
        {
            "operation": "recommend_next_measurement",
            "belief": belief.model_dump(mode="json"),
            "request": request.model_dump(mode="json"),
        }
    )


@runtime_checkable
class QueryCompiler(Protocol):
    @property
    def compiler_descriptor(self) -> QueryCompilerDescriptor: ...

    def compile(self, query: StateQuery) -> CompiledStateSpecification: ...


@runtime_checkable
class CellStateEstimator(Protocol):
    @property
    def descriptor(self) -> EstimatorDescriptor: ...

    @property
    def query_compiler(self) -> QueryCompiler: ...

    def capabilities(
        self,
        request: EstimateCellStateRequest,
        state_specification: CompiledStateSpecification,
    ) -> CapabilityReport: ...

    def estimate(
        self,
        request: EstimateCellStateRequest,
        *,
        options: InferenceOptions,
    ) -> CellStateBelief: ...


@runtime_checkable
class StateEvolutionModel(Protocol):
    @property
    def descriptor(self) -> EstimatorDescriptor: ...

    def capabilities(
        self,
        belief: CellStateBelief,
        scenario: EvolutionScenario,
    ) -> CapabilityReport: ...

    def evolve(
        self,
        belief: CellStateBelief,
        scenario: EvolutionScenario,
        *,
        options: InferenceOptions,
    ) -> StateForecast: ...


@runtime_checkable
class InterventionPlanner(Protocol):
    @property
    def descriptor(self) -> EstimatorDescriptor: ...

    def capabilities(
        self,
        belief: CellStateBelief,
        objective: InterventionObjective,
        candidates: Sequence[EvolutionScenario],
    ) -> CapabilityReport: ...

    def choose(
        self,
        belief: CellStateBelief,
        objective: InterventionObjective,
        candidates: Sequence[EvolutionScenario],
        *,
        options: InferenceOptions,
    ) -> InterventionPlan: ...


PosteriorT = TypeVar("PosteriorT")
PosteriorCoT = TypeVar("PosteriorCoT", covariant=True)
EvidenceCoT = TypeVar("EvidenceCoT", covariant=True)
EvidenceT = TypeVar("EvidenceT")
EvidenceContraT = TypeVar("EvidenceContraT", contravariant=True)
ContextT = TypeVar("ContextT")
ContextContraT = TypeVar("ContextContraT", contravariant=True)
PopulationRequestContraT = TypeVar("PopulationRequestContraT", contravariant=True)
PopulationPreflightCoT = TypeVar("PopulationPreflightCoT", covariant=True)
PopulationResponseCoT = TypeVar("PopulationResponseCoT", covariant=True)


@runtime_checkable
class PopulationAssayResponseModel(
    Protocol[PopulationRequestContraT, PopulationPreflightCoT, PopulationResponseCoT]
):
    """Direct population-endpoint model, distinct from a cell-state estimator.

    Implementations map an explicit population context and assigned action to a future assay-
    response distribution.  The request, preflight, and response types are component-specific on
    purpose: this port neither consumes nor returns a hidden-state belief object and therefore
    cannot be registered as one of the four public cell-state runtime operations by interface
    alone.
    """

    @property
    def component_fingerprint(self) -> str: ...

    def preflight(self, request: PopulationRequestContraT) -> PopulationPreflightCoT: ...

    def sample_response(
        self,
        request: PopulationRequestContraT,
        *,
        sample_count: int,
        seed: int,
    ) -> PopulationResponseCoT: ...


class ObservationModel(Protocol[EvidenceCoT]):
    modality_key: str

    def likelihood(self, observation: ObservationEvent) -> EvidenceCoT: ...


class EvidenceTransferModel(Protocol[EvidenceT]):
    def transfer(
        self,
        evidence: EvidenceT,
        *,
        source_subject: BeliefSubject,
        target_subject: BeliefSubject,
    ) -> EvidenceT: ...


class ReferencePrior(Protocol[PosteriorCoT]):
    def condition(self, request: EstimateCellStateRequest) -> PosteriorCoT: ...


class TransitionKernel(Protocol[PosteriorT]):
    def propagate(
        self,
        posterior: PosteriorT,
        *,
        start_time_seconds: float,
        end_time_seconds: float,
        history: CellHistory,
    ) -> PosteriorT: ...


class FusionModel(Protocol[PosteriorT, EvidenceContraT]):
    def fuse(self, prior: PosteriorT, evidence: Sequence[EvidenceContraT]) -> PosteriorT: ...


class MechanisticConstraintDescriptor(SchemaModel):
    constraint_id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    source_ids: tuple[str, ...] = Field(min_length=1)
    confidence: float = Field(gt=0, le=1)
    residual_metric: str = Field(min_length=1)
    applicability_scope: str = Field(min_length=1)
    ablation_id: str = Field(min_length=1)


class SoftMechanisticConstraint(Protocol[PosteriorT]):
    descriptor: MechanisticConstraintDescriptor
    weight: float

    def apply(self, posterior: PosteriorT) -> tuple[PosteriorT, float]: ...


class DivisionInheritanceModel(Protocol[PosteriorT]):
    def branch(self, parent: PosteriorT, *, seed: int) -> tuple[PosteriorT, PosteriorT]: ...


class InterventionRealizationModel(Protocol):
    def infer(
        self,
        request: EstimateCellStateRequest,
        state_specification: CompiledStateSpecification,
    ) -> tuple[InterventionRealizationBelief, ...]: ...


class CellInteractionModel(Protocol[PosteriorT, ContextContraT]):
    def couple(self, posterior: PosteriorT, context: ContextContraT) -> PosteriorT: ...


class ExtracellularTransportModel(Protocol[ContextT]):
    def propagate(
        self,
        context: ContextT,
        *,
        start_time_seconds: float,
        end_time_seconds: float,
    ) -> ContextT: ...


class FunctionalDecoder(Protocol):
    def decode(
        self,
        posterior: StateDistribution,
        *,
        target: OutputSpec,
        horizon_seconds: float,
    ) -> StateDistribution: ...


class UncertaintyCalibrator(Protocol):
    def calibrate(
        self,
        distribution: StateDistribution,
        *,
        query: StateQuery,
    ) -> tuple[StateDistribution, UncertaintyBreakdown]: ...


class OODDetector(Protocol):
    def evaluate(
        self,
        request: EstimateCellStateRequest,
        state_specification: CompiledStateSpecification,
    ) -> SupportReport: ...


class SufficiencyEvaluator(Protocol):
    def evaluate(
        self,
        belief: CellStateBelief,
        request: EstimateCellStateRequest,
    ) -> SufficiencyReport: ...


class IdentifiabilityEvaluator(Protocol):
    def evaluate(
        self,
        belief: CellStateBelief,
        request: EstimateCellStateRequest,
    ) -> IdentifiabilityReport: ...


class DiagnosticEvaluator(Protocol):
    def evaluate(
        self, belief: CellStateBelief, request: EstimateCellStateRequest
    ) -> BeliefDiagnostics: ...


@runtime_checkable
class MeasurementPolicy(Protocol):
    @property
    def descriptor(self) -> EstimatorDescriptor: ...

    def capabilities(
        self,
        belief: CellStateBelief,
        request: MeasurementDecisionRequest,
    ) -> MeasurementCapabilityReport: ...

    def recommend(
        self,
        belief: CellStateBelief,
        request: MeasurementDecisionRequest,
        *,
        options: InferenceOptions,
    ) -> MeasurementRecommendation: ...


class CellStateModelBundle(Protocol):
    """Named composition surface for a full backend; members remain replaceable ports."""

    descriptor: EstimatorDescriptor
    query_compiler: QueryCompiler
    estimator: CellStateEstimator
    evolution_model: StateEvolutionModel
    planner: InterventionPlanner
    measurement_policy: MeasurementPolicy
