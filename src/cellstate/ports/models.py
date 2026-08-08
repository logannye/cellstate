"""Stable high-level ports plus composable model-stage protocols."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, TypeVar, runtime_checkable

from pydantic import Field, model_validator

from cellstate.domain.belief import (
    BeliefDiagnostics,
    CellStateBelief,
    MeasurementRecommendation,
)
from cellstate.domain.common import SchemaModel
from cellstate.domain.events import ObservationEvent
from cellstate.domain.history import CellHistory
from cellstate.domain.query import StateQuery
from cellstate.domain.request import EstimateCellStateRequest, InferenceOptions
from cellstate.domain.scenarios import (
    EvolutionScenario,
    InterventionObjective,
    InterventionPlan,
    StateForecast,
)


class EstimatorDescriptor(SchemaModel):
    model_id: str = Field(min_length=1)
    model_version: str = Field(min_length=1)
    model_fingerprint: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    posterior_schema_id: str = Field(min_length=1)
    description: str = Field(min_length=1)
    biologically_validated: bool = False
    training_support_id: str | None = None


class CapabilityReport(SchemaModel):
    supported: bool
    unsupported_system_boundary: str | None = None
    unsupported_modalities: tuple[str, ...] = ()
    unsupported_interventions: tuple[str, ...] = ()
    unsupported_environments: tuple[str, ...] = ()
    unsupported_outputs: tuple[str, ...] = ()
    unsupported_precision_requirements: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    @model_validator(mode="after")
    def supported_cannot_have_blockers(self) -> CapabilityReport:
        blockers = (
            self.unsupported_system_boundary is not None
            or bool(self.unsupported_modalities)
            or bool(self.unsupported_interventions)
            or bool(self.unsupported_environments)
            or bool(self.unsupported_outputs)
            or bool(self.unsupported_precision_requirements)
        )
        if self.supported and blockers:
            raise ValueError("a supported capability report cannot declare unsupported features")
        return self


@runtime_checkable
class CellStateEstimator(Protocol):
    @property
    def descriptor(self) -> EstimatorDescriptor: ...

    def capabilities(self, query: StateQuery) -> CapabilityReport: ...

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
EvidenceContraT = TypeVar("EvidenceContraT", contravariant=True)


class ObservationModel(Protocol[EvidenceCoT]):
    modality_key: str

    def likelihood(self, observation: ObservationEvent) -> EvidenceCoT: ...


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


class SoftMechanisticConstraint(Protocol[PosteriorT]):
    name: str
    weight: float

    def apply(self, posterior: PosteriorT) -> tuple[PosteriorT, float]: ...


class DivisionInheritanceModel(Protocol[PosteriorT]):
    def branch(self, parent: PosteriorT, *, seed: int) -> tuple[PosteriorT, PosteriorT]: ...


class DiagnosticEvaluator(Protocol):
    def evaluate(
        self, belief: CellStateBelief, request: EstimateCellStateRequest
    ) -> BeliefDiagnostics: ...


class MeasurementPolicy(Protocol):
    def recommend(
        self, belief: CellStateBelief, request: EstimateCellStateRequest
    ) -> MeasurementRecommendation: ...
