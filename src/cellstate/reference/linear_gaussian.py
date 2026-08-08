"""Executable linear-Gaussian reference backend.

This module exists to exercise the contracts, recursive filtering, controlled propagation,
uncertainty, and assay-selection pathways. It is not a validated model of cellular biology.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from hashlib import sha256
from itertools import pairwise
from typing import cast
from uuid import NAMESPACE_URL, uuid5

import numpy as np
from numpy.typing import NDArray
from pydantic import Field, field_validator, model_validator
from scipy.linalg import expm

from cellstate.domain.belief import (
    BeliefDiagnostics,
    BeliefStatus,
    CellStateBelief,
    ContextBelief,
    DynamicSummary,
    EvaluatedScalar,
    FactorBelief,
    MeasurementRecommendation,
    ObservabilityReport,
    OODReport,
    ParametricDistribution,
    StateFactor,
    SufficiencyReport,
    UnavailableDistribution,
    UncertaintyBreakdown,
    UncertaintyComponent,
    UncertaintyKind,
)
from cellstate.domain.common import (
    EvidenceStatus,
    ProvenanceRecord,
    Quantity,
    SchemaModel,
    SupportStatus,
    canonical_fingerprint,
    require_finite,
)
from cellstate.domain.events import (
    ContactEvent,
    DivisionEvent,
    EnvironmentEvent,
    EvidenceRole,
    InterventionEvent,
    MissingnessStatus,
    ObservationEvent,
)
from cellstate.domain.history import CellHistory, RecordCompleteness
from cellstate.domain.query import StateQuery, SystemBoundary, Timescale
from cellstate.domain.request import EstimateCellStateRequest, InferenceOptions
from cellstate.domain.scenarios import (
    CandidateEvaluation,
    EvolutionScenario,
    InterventionObjective,
    InterventionPlan,
    ObjectiveDirection,
    ScenarioReference,
    StateForecast,
    TargetPrediction,
)
from cellstate.errors import (
    CapabilityError,
    PosteriorCompatibilityError,
    UnsupportedInterventionError,
    UnsupportedModalityError,
)
from cellstate.ports import CapabilityReport, EstimatorDescriptor

FloatArray = NDArray[np.float64]


def _default_factor_timescales() -> dict[StateFactor, frozenset[Timescale]]:
    return {
        StateFactor.STABLE_IDENTITY: frozenset({Timescale.SLOW}),
        StateFactor.SLOW_MEMORY: frozenset({Timescale.SLOW}),
        StateFactor.REGULATORY: frozenset({Timescale.INTERMEDIATE}),
        StateFactor.SIGNALING: frozenset({Timescale.FAST}),
        StateFactor.METABOLIC: frozenset({Timescale.FAST, Timescale.INTERMEDIATE}),
        StateFactor.PHYSICAL: frozenset({Timescale.INTERMEDIATE}),
        StateFactor.DAMAGE_STRESS: frozenset({Timescale.INTERMEDIATE, Timescale.SLOW}),
        StateFactor.FUNCTIONAL_CAPACITY: frozenset(
            {Timescale.FAST, Timescale.INTERMEDIATE, Timescale.SLOW}
        ),
    }


class LinearObservationConfig(SchemaModel):
    modality_key: str = Field(min_length=1)
    units: str = Field(min_length=1)
    matrix: tuple[tuple[float, ...], ...] = Field(min_length=1)
    noise_covariance: tuple[tuple[float, ...], ...] = Field(min_length=1)
    direct_dimensions: tuple[str, ...] = ()


class LinearGaussianConfig(SchemaModel):
    """Matrices for a small continuous-time controlled Gaussian system."""

    model_id: str = "linear-gaussian-reference"
    model_version: str = "0.1.0"
    state_dimensions: tuple[str, ...] = Field(min_length=1)
    prior_time_seconds: float = 0.0
    prior_mean: tuple[float, ...]
    prior_covariance: tuple[tuple[float, ...], ...]
    drift_matrix: tuple[tuple[float, ...], ...]
    drift_vector: tuple[float, ...]
    process_covariance_per_second: tuple[tuple[float, ...], ...]
    observation_models: tuple[LinearObservationConfig, ...] = ()
    control_effects: dict[str, tuple[float, ...]] = Field(default_factory=dict)
    control_dose_units: dict[str, str] = Field(default_factory=dict)
    environment_effects: dict[str, tuple[float, ...]] = Field(default_factory=dict)
    environment_units: dict[str, str] = Field(default_factory=dict)
    factor_dimensions: dict[StateFactor, tuple[str, ...]] = Field(default_factory=dict)
    factor_timescales: dict[StateFactor, frozenset[Timescale]] = Field(
        default_factory=_default_factor_timescales
    )
    output_units: dict[str, str] = Field(default_factory=dict)
    supported_species_keys: tuple[str, ...] = ("homo_sapiens", "NCBITaxon:9606")

    @field_validator("state_dimensions")
    @classmethod
    def unique_dimensions(cls, dimensions: tuple[str, ...]) -> tuple[str, ...]:
        if len(dimensions) != len(set(dimensions)):
            raise ValueError("state dimensions must be unique")
        return dimensions

    @field_validator("prior_time_seconds")
    @classmethod
    def finite_prior_time(cls, value: float) -> float:
        return require_finite(value, name="reference prior time")

    @model_validator(mode="after")
    def validate_shapes_and_assignments(self) -> LinearGaussianConfig:
        size = len(self.state_dimensions)
        _require_vector("prior_mean", self.prior_mean, size)
        _require_vector("drift_vector", self.drift_vector, size)
        _require_matrix("prior_covariance", self.prior_covariance, size, size)
        _require_matrix("drift_matrix", self.drift_matrix, size, size)
        _require_matrix(
            "process_covariance_per_second", self.process_covariance_per_second, size, size
        )
        _require_covariance("prior_covariance", self.prior_covariance)
        _require_covariance("process_covariance_per_second", self.process_covariance_per_second)

        modality_keys = [model.modality_key.casefold() for model in self.observation_models]
        if len(modality_keys) != len(set(modality_keys)):
            raise ValueError("observation model modality keys must be unique")
        state_set = set(self.state_dimensions)
        for observation in self.observation_models:
            rows = len(observation.matrix)
            _require_matrix(
                f"observation matrix {observation.modality_key}", observation.matrix, rows, size
            )
            _require_matrix(
                f"observation noise {observation.modality_key}",
                observation.noise_covariance,
                rows,
                rows,
            )
            _require_covariance(
                f"observation noise {observation.modality_key}", observation.noise_covariance
            )
            if not set(observation.direct_dimensions) <= state_set:
                raise ValueError("direct observation dimensions must be declared state dimensions")
            matrix = np.asarray(observation.matrix, dtype=float)
            row_space_projection = np.linalg.pinv(matrix) @ matrix
            for dimension in observation.direct_dimensions:
                index = self.state_dimensions.index(dimension)
                basis = np.eye(size)[index]
                if not np.allclose(row_space_projection @ basis, basis, atol=1e-8):
                    raise ValueError(
                        f"direct dimension {dimension!r} is confounded in its observation matrix"
                    )

        for key, vector in (*self.control_effects.items(), *self.environment_effects.items()):
            if key != key.casefold():
                raise ValueError(
                    "control and environment effect keys must be lowercase canonical keys"
                )
            _require_vector(f"effect {key}", vector, size)
        if set(self.control_dose_units) != set(self.control_effects):
            raise ValueError("control dose units must be declared for every control effect")
        if set(self.environment_units) != set(self.environment_effects):
            raise ValueError("environment units must be declared for every environment effect")
        assigned = [dimension for values in self.factor_dimensions.values() for dimension in values]
        if len(assigned) != len(set(assigned)):
            raise ValueError("a state dimension may belong to only one structured factor")
        if not set(assigned) <= state_set:
            raise ValueError("factor mappings contain undeclared state dimensions")
        if set(self.factor_timescales) != set(StateFactor):
            raise ValueError("factor timescales must include every structured state factor")
        if any(not timescales for timescales in self.factor_timescales.values()):
            raise ValueError("factor timescale assignments must be nonempty")
        if not {key.casefold() for key in self.output_units} <= state_set:
            raise ValueError("reference supported outputs must map to declared state dimensions")
        if any(key != key.casefold() for key in self.output_units):
            raise ValueError("output-unit keys must be lowercase canonical keys")
        return self

    @property
    def observations_by_key(self) -> dict[str, LinearObservationConfig]:
        return {model.modality_key.casefold(): model for model in self.observation_models}


@dataclass(frozen=True)
class GaussianState:
    mean: FloatArray
    covariance: FloatArray


def _require_vector(name: str, values: Sequence[float], size: int) -> None:
    array = np.asarray(values, dtype=float)
    if array.shape != (size,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite vector of length {size}")


def _require_matrix(name: str, values: Sequence[Sequence[float]], rows: int, columns: int) -> None:
    array = np.asarray(values, dtype=float)
    if array.shape != (rows, columns) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite {rows}x{columns} matrix")


def _require_covariance(name: str, values: Sequence[Sequence[float]]) -> None:
    matrix = np.asarray(values, dtype=float)
    if not np.allclose(matrix, matrix.T, rtol=1e-8, atol=1e-10):
        raise ValueError(f"{name} must be symmetric")
    if np.linalg.eigvalsh(matrix).min() < -1e-10:
        raise ValueError(f"{name} must be positive semidefinite")


def _as_tuple_vector(values: FloatArray) -> tuple[float, ...]:
    return tuple(float(value) for value in values)


def _as_tuple_matrix(values: FloatArray) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(float(value) for value in row) for row in values)


def _stabilize_covariance(covariance: FloatArray) -> FloatArray:
    symmetric = (covariance + covariance.T) / 2
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    clipped = np.maximum(eigenvalues, 0)
    return cast(FloatArray, (eigenvectors * clipped) @ eigenvectors.T)


def _unsupported(reason: str) -> EvaluatedScalar:
    return EvaluatedScalar(status=SupportStatus.UNSUPPORTED, reason=reason)


class LinearGaussianReference:
    """Kalman-style reference estimator and controlled transition backend."""

    def __init__(self, config: LinearGaussianConfig) -> None:
        self.config = LinearGaussianConfig.model_validate(config.model_dump(mode="python"))

    @property
    def model_fingerprint(self) -> str:
        return canonical_fingerprint(self.config)

    @property
    def posterior_schema_id(self) -> str:
        return f"cellstate.linear_gaussian/{self.model_fingerprint}"

    @property
    def descriptor(self) -> EstimatorDescriptor:
        return EstimatorDescriptor(
            model_id=self.config.model_id,
            model_version=self.config.model_version,
            model_fingerprint=self.model_fingerprint,
            posterior_schema_id=self.posterior_schema_id,
            description=(
                "Biologically non-authoritative linear-Gaussian contract reference; "
                "not for scientific conclusions"
            ),
            biologically_validated=False,
        )

    def capabilities(self, query: StateQuery) -> CapabilityReport:
        supported_boundaries = {
            SystemBoundary.ISOLATED_CELL,
            SystemBoundary.CELL_AND_SOLUBLE_ENVIRONMENT,
        }
        unsupported_boundary = (
            query.system_boundary.value
            if query.system_boundary not in supported_boundaries
            else None
        )
        unsupported_interventions: list[str] = []
        for intervention_spec in query.intervention_space:
            key = intervention_spec.kind.key.casefold()
            effect_keys = (
                [mechanism.key.casefold() for mechanism in intervention_spec.mechanisms]
                if intervention_spec.mechanisms
                else [key]
            )
            if intervention_spec.target is not None:
                unsupported_interventions.append(
                    f"{intervention_spec.kind.key}[target={intervention_spec.target.key}]"
                )
            elif any(effect_key not in self.config.control_effects for effect_key in effect_keys):
                unsupported_interventions.append(
                    f"{intervention_spec.kind.key}[mechanism-specific]"
                )
            elif intervention_spec.dose_units is not None and any(
                intervention_spec.dose_units != self.config.control_dose_units[effect_key]
                for effect_key in effect_keys
            ):
                configured_units = sorted(
                    {self.config.control_dose_units[effect_key] for effect_key in effect_keys}
                )
                unsupported_interventions.append(
                    f"{intervention_spec.kind.key}[{intervention_spec.dose_units} != "
                    f"{configured_units}]"
                )
        supported_outputs = {key.casefold() for key in self.config.output_units}
        unsupported_environments: list[str] = []
        for environment_spec in query.environment_space:
            key = environment_spec.variable.key.casefold()
            if key not in self.config.environment_effects:
                unsupported_environments.append(environment_spec.variable.key)
            elif (
                environment_spec.units is not None
                and environment_spec.units != self.config.environment_units[key]
            ):
                unsupported_environments.append(
                    f"{environment_spec.variable.key}[{environment_spec.units} != "
                    f"{self.config.environment_units[key]}]"
                )
        unsupported_outputs = tuple(
            sorted(
                {
                    output.term.key
                    for output in query.target_outputs
                    if output.term.key.casefold() not in supported_outputs
                    or output.units != self.config.output_units.get(output.term.key.casefold())
                }
            )
        )
        unsupported_precision = tuple(
            f"{item.target.key}:{item.metric}" for item in query.precision_requirements
        )
        return CapabilityReport(
            supported=not any(
                (
                    unsupported_boundary,
                    unsupported_interventions,
                    unsupported_environments,
                    unsupported_outputs,
                    unsupported_precision,
                )
            ),
            unsupported_system_boundary=unsupported_boundary,
            unsupported_interventions=tuple(sorted(unsupported_interventions)),
            unsupported_environments=tuple(sorted(unsupported_environments)),
            unsupported_outputs=unsupported_outputs,
            unsupported_precision_requirements=unsupported_precision,
            notes=("Reference backend is not biologically validated.",),
        )

    def estimate(
        self,
        request: EstimateCellStateRequest,
        *,
        options: InferenceOptions,
    ) -> CellStateBelief:
        self._validate_history_semantics(request)
        self._validate_history_capabilities(request.history)
        posterior, start_time = self._initial_posterior(request)
        previous_event_ids = (
            set(request.previous_belief.provenance.source_event_ids)
            if request.previous_belief is not None
            else set()
        )
        evidence_observation_events = tuple(
            event
            for event in request.history.events
            if isinstance(event, ObservationEvent) and event.time_seconds <= request.as_of_seconds
        )
        observation_events = tuple(
            event
            for event in request.history.events
            if isinstance(event, ObservationEvent)
            and start_time <= event.time_seconds <= request.as_of_seconds
            and (
                request.previous_belief is None
                or (
                    event.time_seconds >= request.previous_belief.as_of_seconds
                    and event.event_id not in previous_event_ids
                )
            )
        )

        current_time = start_time
        at_start = [event for event in observation_events if event.time_seconds == current_time]
        posterior = self._update_observations(posterior, at_start)
        update_times = sorted(
            {
                event.time_seconds
                for event in observation_events
                if event.time_seconds > current_time
            }
        )
        for update_time in update_times:
            posterior = self._propagate(
                posterior, current_time, update_time, request.history.events
            )
            posterior = self._update_observations(
                posterior,
                [event for event in observation_events if event.time_seconds == update_time],
            )
            current_time = update_time
        posterior = self._propagate(
            posterior, current_time, request.as_of_seconds, request.history.events
        )

        return self._build_belief(
            request,
            posterior,
            evidence_observation_events,
            options.seed,
        )

    def evolve(
        self,
        belief: CellStateBelief,
        scenario: EvolutionScenario,
        *,
        options: InferenceOptions,
    ) -> StateForecast:
        posterior = self._decode_posterior(belief)
        events = self._scenario_events(belief, scenario)
        self._validate_events_capabilities(events)
        propagated = self._propagate(
            posterior,
            scenario.start_time_seconds,
            scenario.end_time_seconds,
            events,
        )
        factors = self._forecast_factor_beliefs(propagated, belief.factors)
        target_predictions = self._target_predictions(
            propagated, belief, scenario.end_time_seconds - scenario.start_time_seconds
        )
        dynamics = self._dynamics(propagated, scenario.end_time_seconds, events)
        uncertainty = self._uncertainty(propagated)
        scenario_hash = canonical_fingerprint(scenario)
        provenance = ProvenanceRecord(
            model_id=self.config.model_id,
            model_version=self.config.model_version,
            model_fingerprint=self.model_fingerprint,
            posterior_schema_id=self.posterior_schema_id,
            query_fingerprint=belief.query_fingerprint,
            history_fingerprint=belief.history_fingerprint,
            history_structure_fingerprint=belief.provenance.history_structure_fingerprint,
            context_fingerprint=belief.context_fingerprint,
            source_event_ids=(
                *belief.provenance.source_event_ids,
                *(event.event_id for event in (*scenario.interventions, *scenario.environments)),
            ),
            source_event_fingerprints={
                **belief.provenance.source_event_fingerprints,
                **{
                    event.event_id: canonical_fingerprint(event)
                    for event in (*scenario.interventions, *scenario.environments)
                },
            },
            seed=options.seed,
            warnings=(
                "Reference linear-Gaussian forecast; not biologically validated.",
                f"scenario_fingerprint={scenario_hash}",
            ),
        )
        return StateForecast(
            forecast_id=str(
                uuid5(
                    NAMESPACE_URL,
                    f"{belief.belief_id}:{scenario_hash}:{self.model_fingerprint}:{options.seed}",
                )
            ),
            parent_belief_id=belief.belief_id,
            scenario_id=scenario.scenario_id,
            scenario_fingerprint=scenario_hash,
            query=belief.query,
            query_fingerprint=belief.query_fingerprint,
            horizon_name=scenario.horizon_name,
            horizon_seconds=scenario.end_time_seconds - scenario.start_time_seconds,
            subject_id=belief.subject_id,
            start_time_seconds=scenario.start_time_seconds,
            end_time_seconds=scenario.end_time_seconds,
            joint_posterior=self._distribution(propagated),
            factors=factors,
            target_predictions=target_predictions,
            dynamics=dynamics,
            uncertainty=uncertainty,
            provenance=provenance,
        )

    def _initial_posterior(self, request: EstimateCellStateRequest) -> tuple[GaussianState, float]:
        if request.previous_belief is not None:
            return self._decode_posterior(
                request.previous_belief
            ), request.previous_belief.as_of_seconds
        return (
            GaussianState(
                mean=np.asarray(self.config.prior_mean, dtype=float),
                covariance=np.asarray(self.config.prior_covariance, dtype=float),
            ),
            self.config.prior_time_seconds,
        )

    def _decode_posterior(self, belief: CellStateBelief) -> GaussianState:
        provenance = belief.provenance
        if (
            provenance.model_id != self.config.model_id
            or provenance.model_version != self.config.model_version
            or provenance.model_fingerprint != self.model_fingerprint
            or provenance.posterior_schema_id != self.posterior_schema_id
        ):
            raise PosteriorCompatibilityError(
                "belief was produced by an incompatible reference model or posterior schema"
            )
        distribution = belief.joint_posterior
        if not isinstance(distribution, ParametricDistribution):
            raise PosteriorCompatibilityError("reference backend requires a parametric posterior")
        if distribution.family != "multivariate_normal":
            raise PosteriorCompatibilityError("reference backend requires a normal posterior")
        if distribution.dimensions != self.config.state_dimensions:
            raise PosteriorCompatibilityError("posterior dimensions do not match this backend")
        return GaussianState(
            mean=np.asarray(distribution.mean, dtype=float),
            covariance=np.asarray(distribution.covariance, dtype=float),
        )

    def _validate_history_capabilities(self, history: CellHistory) -> None:
        self._validate_events_capabilities(history.events)

    def _validate_history_semantics(self, request: EstimateCellStateRequest) -> None:
        if request.as_of_seconds < self.config.prior_time_seconds:
            raise CapabilityError("requested state time predates the configured reference prior")
        early_events = [
            event.event_id
            for event in request.history.events
            if event.time_seconds < self.config.prior_time_seconds
        ]
        if early_events:
            raise CapabilityError(
                "history contains events before the configured reference prior time: "
                f"{early_events}"
            )
        if request.static_context.species.key not in self.config.supported_species_keys:
            raise CapabilityError(
                f"reference prior does not support species {request.static_context.species.key!r}"
            )
        static_payload = request.static_context.model_dump(mode="python")
        unsupported_static = {
            key
            for key, value in static_payload.items()
            if key not in {"species", "attributes"}
            and value is not None
            and value != ()
            and value != {}
            and value != []
        }
        if request.static_context.attributes:
            unsupported_static.add("attributes")
        if unsupported_static:
            raise CapabilityError(
                "reference prior does not condition on supplied static context fields: "
                f"{sorted(unsupported_static)}"
            )
        if request.population_context is not None:
            raise CapabilityError("reference prior does not model population context")
        if request.history.lineage is not None:
            raise CapabilityError("reference backend does not model lineage history")
        if request.history.neighborhood is not None:
            raise CapabilityError("reference backend does not model neighborhood state")
        completeness = request.history.completeness
        if completeness.interventions is not RecordCompleteness.COMPLETE:
            raise CapabilityError(
                "reference estimation requires a complete intervention record; unknown or "
                "incomplete history cannot be interpreted as no intervention"
            )
        if completeness.environments is not RecordCompleteness.COMPLETE:
            raise CapabilityError(
                "reference estimation requires a complete environment record; unknown or "
                "incomplete history cannot be interpreted as zero environment"
            )
        if completeness.lineage is not RecordCompleteness.COMPLETE:
            raise CapabilityError(
                "reference estimation requires a complete lineage record; unknown or "
                "incomplete history cannot be interpreted as no division"
            )
        if completeness.neighborhood is not RecordCompleteness.COMPLETE:
            raise CapabilityError(
                "reference estimation requires a complete neighborhood/contact record; unknown "
                "or incomplete history cannot be interpreted as no contact"
            )
        environment_events = tuple(
            event
            for event in request.history.events
            if isinstance(event, EnvironmentEvent) and event.time_seconds <= request.as_of_seconds
        )
        recorded_environment_keys = {
            key.casefold() for event in environment_events for key in event.variables
        }
        missing_required = {
            item.variable.key
            for item in request.query.environment_space
            if item.required and item.variable.key.casefold() not in recorded_environment_keys
        }
        if missing_required:
            raise CapabilityError(
                f"required environment variables are unobserved: {sorted(missing_required)}"
            )

    def _validate_events_capabilities(self, events: Iterable[object]) -> None:
        environment_assignments: dict[tuple[float, str], str] = {}
        for event in events:
            if isinstance(event, ObservationEvent):
                if event.evidence_role is not EvidenceRole.DIRECT:
                    raise CapabilityError(
                        "reference backend does not implement lineage/population evidence updates"
                    )
                assay = event.assay
                if (
                    assay.batch is not None
                    or assay.instrument is not None
                    or assay.operator is not None
                    or assay.library_size is not None
                    or assay.capture_efficiency is not None
                    or assay.staining_panel is not None
                    or assay.segmentation_confidence is not None
                    or assay.plate_position is not None
                    or assay.processing_delay_seconds is not None
                    or assay.detection_limits
                    or assay.attributes
                    or event.quality.flags
                ):
                    raise CapabilityError(
                        "reference observation model does not condition on nondefault assay "
                        "metadata or quality flags"
                    )
                if event.missingness.status in {
                    MissingnessStatus.OBSERVED,
                    MissingnessStatus.BELOW_DETECTION,
                    MissingnessStatus.CENSORED,
                }:
                    modality_key = event.modality.key.casefold()
                    if modality_key not in self.config.observations_by_key:
                        raise UnsupportedModalityError(
                            f"reference backend has no observation model for {event.modality.key!r}"
                        )
                    observation_model = self.config.observations_by_key[modality_key]
                    if event.units != observation_model.units:
                        raise UnsupportedModalityError(
                            f"observation {event.event_id!r} units must be "
                            f"{observation_model.units!r}"
                        )
                if event.missingness.status is MissingnessStatus.OBSERVED:
                    if event.uncertainty.distribution.casefold() not in {
                        "unspecified",
                        "normal",
                        "gaussian",
                    }:
                        raise CapabilityError(
                            "reference backend only supports Gaussian measurement uncertainty"
                        )
                    if event.uncertainty.parameters:
                        raise CapabilityError(
                            "reference backend does not interpret free-form measurement "
                            "uncertainty parameters; use standard_error"
                        )
                elif event.missingness.status in {
                    MissingnessStatus.BELOW_DETECTION,
                    MissingnessStatus.CENSORED,
                }:
                    raise CapabilityError(
                        "reference backend does not implement censored observation likelihoods"
                    )
            elif isinstance(event, InterventionEvent):
                self._control_effect(event)
            elif isinstance(event, EnvironmentEvent):
                if event.spatial_region is not None:
                    raise CapabilityError(
                        "reference backend does not model spatially regional environment states"
                    )
                unsupported = {
                    key.casefold()
                    for key in event.variables
                    if key.casefold() not in self.config.environment_effects
                }
                if unsupported:
                    raise CapabilityError(
                        f"reference backend has no environment effects for {sorted(unsupported)}"
                    )
                for key, value in event.variables.items():
                    assignment_key = (event.time_seconds, key.casefold())
                    value_fingerprint = canonical_fingerprint({"value": value})
                    previous = environment_assignments.get(assignment_key)
                    if previous is not None and previous != value_fingerprint:
                        raise CapabilityError(
                            "reference backend cannot resolve conflicting same-time assignments "
                            f"for environment variable {key!r}"
                        )
                    environment_assignments[assignment_key] = value_fingerprint
            elif isinstance(event, DivisionEvent):
                raise CapabilityError("reference backend does not implement division/inheritance")
            elif isinstance(event, ContactEvent):
                raise CapabilityError("reference backend does not implement contact dynamics")

    def _scenario_events(
        self, belief: CellStateBelief, scenario: EvolutionScenario
    ) -> tuple[InterventionEvent | EnvironmentEvent, ...]:
        continued_interventions: tuple[InterventionEvent, ...] = ()
        active_interventions = belief.context.active_interventions
        if active_interventions and scenario.inherit_active_interventions is None:
            raise CapabilityError("scenario must explicitly inherit or clear active interventions")
        if scenario.inherit_active_interventions is True:
            continued: list[InterventionEvent] = []
            for event in active_interventions:
                stop_time = event.time_seconds + event.duration_seconds
                continuation_end = min(stop_time, scenario.end_time_seconds)
                if continuation_end <= scenario.start_time_seconds:
                    continue
                continued.append(
                    event.model_copy(
                        update={
                            "event_id": (
                                f"{scenario.scenario_id}:continued-intervention:{event.event_id}"
                            ),
                            "time_seconds": scenario.start_time_seconds,
                            "duration_seconds": (continuation_end - scenario.start_time_seconds),
                            "source": "continued from input belief context",
                        }
                    )
                )
            continued_interventions = tuple(continued)
        inherited: tuple[EnvironmentEvent, ...] = ()
        current_environment = belief.context.soluble_environment
        explicit_at_start = any(
            math.isclose(event.time_seconds, scenario.start_time_seconds)
            for event in scenario.environments
        )
        if (
            current_environment
            and scenario.inherit_current_environment is None
            and not explicit_at_start
        ):
            raise CapabilityError(
                "scenario must explicitly inherit, clear, or replace the current "
                "environment at start"
            )
        if scenario.inherit_current_environment is True and current_environment:
            inherited_variables = {
                key: (value if isinstance(value, Quantity) else Quantity.model_validate(value))
                for key, value in current_environment.items()
            }
            inherited = (
                EnvironmentEvent(
                    event_id=f"{scenario.scenario_id}:inherited-environment",
                    subject_id=scenario.subject_id,
                    time_seconds=scenario.start_time_seconds,
                    variables=inherited_variables,
                    source="inherited from input belief context",
                ),
            )
        return (
            *continued_interventions,
            *scenario.interventions,
            *inherited,
            *scenario.environments,
        )

    def _propagate(
        self,
        posterior: GaussianState,
        start: float,
        end: float,
        events: Iterable[object],
    ) -> GaussianState:
        if end < start:
            raise ValueError("cannot propagate backward in time")
        if end == start:
            return posterior
        event_tuple = tuple(events)
        breakpoints = {start, end}
        for event in event_tuple:
            if isinstance(event, (InterventionEvent, EnvironmentEvent)) and (
                start < event.time_seconds < end
            ):
                breakpoints.add(event.time_seconds)
            if isinstance(event, InterventionEvent) and event.duration_seconds > 0:
                stop = event.time_seconds + event.duration_seconds
                if start < stop < end:
                    breakpoints.add(stop)

        state = posterior
        points = sorted(breakpoints)
        for left, right in pairwise(points):
            elapsed = right - left
            midpoint = left + elapsed / 2
            transition, input_integral, process = _discretize_linear_system(
                np.asarray(self.config.drift_matrix, dtype=float),
                np.asarray(self.config.process_covariance_per_second, dtype=float),
                elapsed,
            )
            drive = np.asarray(self.config.drift_vector, dtype=float)
            drive = drive + self._active_control(midpoint, event_tuple)
            drive = drive + self._environment_drive(midpoint, event_tuple)
            mean = transition @ state.mean + input_integral @ drive
            covariance = transition @ state.covariance @ transition.T + process
            state = GaussianState(mean=mean, covariance=_stabilize_covariance(covariance))
        return state

    def _active_control(self, time: float, events: Iterable[object]) -> FloatArray:
        result = np.zeros(len(self.config.state_dimensions), dtype=float)
        for event in events:
            if not isinstance(event, InterventionEvent):
                continue
            ends_at = event.time_seconds + event.duration_seconds
            instantaneous = event.duration_seconds == 0 and math.isclose(time, event.time_seconds)
            active = event.time_seconds <= time < ends_at or instantaneous
            if active:
                effect, efficiency = self._control_effect(event)
                assert event.dose is not None
                result = result + effect * event.dose.value * efficiency
        return result

    def _control_effect(self, event: InterventionEvent) -> tuple[FloatArray, float]:
        if event.delivery_method is not None or event.reversible is not None:
            raise UnsupportedInterventionError(
                "reference backend does not model delivery-method or reversibility semantics"
            )
        if event.target is not None:
            raise UnsupportedInterventionError(
                "reference backend does not model target-specific intervention effects"
            )
        key = (
            event.mechanism.key.casefold()
            if event.mechanism is not None
            else event.intervention_type.key.casefold()
        )
        if key not in self.config.control_effects:
            raise UnsupportedInterventionError(
                f"reference backend has no control effect for {key!r}"
            )
        actual_efficiency = (
            event.actual_perturbation.efficiency if event.actual_perturbation is not None else None
        )
        efficiency = (
            actual_efficiency if actual_efficiency is not None else event.estimated_efficiency
        )
        if efficiency is None:
            raise UnsupportedInterventionError(
                f"intervention {event.event_id!r} has no measured or estimated efficacy"
            )
        if event.duration_seconds == 0:
            raise UnsupportedInterventionError(
                "reference backend does not implement instantaneous intervention jumps"
            )
        if event.dose is None:
            raise UnsupportedInterventionError(
                f"intervention {event.event_id!r} requires an explicit dose"
            )
        expected_units = self.config.control_dose_units[key]
        if event.dose.units != expected_units:
            raise UnsupportedInterventionError(
                f"intervention {event.event_id!r} dose units must be {expected_units!r}"
            )
        return np.asarray(self.config.control_effects[key], dtype=float), efficiency

    def _environment_drive(self, time: float, events: Iterable[object]) -> FloatArray:
        latest: dict[str, tuple[float, Quantity | object]] = {}
        for event in events:
            if isinstance(event, EnvironmentEvent) and event.time_seconds <= time:
                for key, value in event.variables.items():
                    normalized = key.casefold()
                    if normalized not in latest or event.time_seconds >= latest[normalized][0]:
                        latest[normalized] = (event.time_seconds, value)
        result = np.zeros(len(self.config.state_dimensions), dtype=float)
        for key, (_, stored_value) in latest.items():
            if key not in self.config.environment_effects:
                raise CapabilityError(f"no configured environment effect for {key!r}")
            if not isinstance(stored_value, Quantity):
                raise CapabilityError(
                    f"environment variable {key!r} requires an explicit quantity and units"
                )
            expected_units = self.config.environment_units[key]
            if stored_value.units != expected_units:
                raise CapabilityError(
                    f"environment variable {key!r} units must be {expected_units!r}"
                )
            result = result + np.asarray(self.config.environment_effects[key], dtype=float) * float(
                stored_value.value
            )
        return result

    def _update_observations(
        self, posterior: GaussianState, observations: Iterable[ObservationEvent]
    ) -> GaussianState:
        state = posterior
        identity = np.eye(len(self.config.state_dimensions))
        for observation in observations:
            if observation.missingness.status is not MissingnessStatus.OBSERVED:
                continue
            config = self.config.observations_by_key[observation.modality.key.casefold()]
            measurement = _measurement_vector(observation)
            observation_matrix = np.asarray(config.matrix, dtype=float)
            if measurement.shape != (observation_matrix.shape[0],):
                raise ValueError(
                    f"observation {observation.event_id!r} has shape {measurement.shape}; "
                    f"expected {(observation_matrix.shape[0],)}"
                )
            noise = np.asarray(config.noise_covariance, dtype=float)
            quality = max(observation.quality.score, 1e-6)
            noise = noise / quality
            if observation.uncertainty.standard_error is not None:
                noise = noise + np.eye(len(measurement)) * observation.uncertainty.standard_error**2
            innovation_covariance = (
                observation_matrix @ state.covariance @ observation_matrix.T + noise
            )
            gain = state.covariance @ observation_matrix.T @ np.linalg.pinv(innovation_covariance)
            innovation = measurement - observation_matrix @ state.mean
            mean = state.mean + gain @ innovation
            residual = identity - gain @ observation_matrix
            covariance = residual @ state.covariance @ residual.T + gain @ noise @ gain.T
            state = GaussianState(mean=mean, covariance=_stabilize_covariance(covariance))
        return state

    def _distribution(
        self,
        posterior: GaussianState,
        dimensions: tuple[str, ...] | None = None,
        indices: Sequence[int] | None = None,
    ) -> ParametricDistribution:
        if dimensions is None:
            dimensions = self.config.state_dimensions
        if indices is None:
            mean = posterior.mean
            covariance = posterior.covariance
        else:
            selected = np.asarray(indices, dtype=int)
            mean = posterior.mean[selected]
            covariance = posterior.covariance[np.ix_(selected, selected)]
        return ParametricDistribution(
            family="multivariate_normal",
            dimensions=dimensions,
            mean=_as_tuple_vector(mean),
            covariance=_as_tuple_matrix(covariance),
        )

    def _factor_beliefs(
        self,
        posterior: GaussianState,
        observations: Iterable[ObservationEvent],
        as_of_seconds: float,
    ) -> tuple[FactorBelief, ...]:
        observation_tuple = tuple(
            event
            for event in observations
            if event.missingness.status is MissingnessStatus.OBSERVED
        )
        evidence_by_dimension: dict[str, list[str]] = {
            dimension: [] for dimension in self.config.state_dimensions
        }
        modalities_by_dimension: dict[str, set[str]] = {
            dimension: set() for dimension in self.config.state_dimensions
        }
        direct_dimensions: set[str] = set()
        identifiable_dimensions = self._identifiable_dimensions(observation_tuple, as_of_seconds)
        for event in observation_tuple:
            model = self.config.observations_by_key[event.modality.key.casefold()]
            matrix = self._current_observation_matrix(event, as_of_seconds)
            for index, dimension in enumerate(self.config.state_dimensions):
                if np.any(np.abs(matrix[:, index]) > 1e-12):
                    evidence_by_dimension[dimension].append(event.event_id)
                    modalities_by_dimension[dimension].add(event.modality.key.casefold())
            if math.isclose(event.time_seconds, as_of_seconds, rel_tol=0, abs_tol=1e-12):
                direct_dimensions.update(model.direct_dimensions)

        beliefs: list[FactorBelief] = []
        for factor in StateFactor:
            dimensions = self.config.factor_dimensions.get(factor, ())
            if not dimensions:
                beliefs.append(
                    FactorBelief(
                        factor=factor,
                        timescales=self.config.factor_timescales[factor],
                        evidence_status=EvidenceStatus.UNIDENTIFIABLE,
                        posterior=UnavailableDistribution(
                            reason_code="unsupported_factor",
                            message="Reference backend has no dimensions for this factor.",
                        ),
                    )
                )
                continue
            indices = [self.config.state_dimensions.index(dimension) for dimension in dimensions]
            event_ids = tuple(
                sorted(
                    {
                        event
                        for dimension in dimensions
                        for event in evidence_by_dimension[dimension]
                    }
                )
            )
            if all(dimension in direct_dimensions for dimension in dimensions) and event_ids:
                status = EvidenceStatus.OBSERVED
            elif (
                all(dimension in identifiable_dimensions for dimension in dimensions) and event_ids
            ):
                status = EvidenceStatus.INFERRED
            else:
                status = EvidenceStatus.UNIDENTIFIABLE
            shared = tuple(
                dimension for dimension in dimensions if len(modalities_by_dimension[dimension]) > 1
            )
            private = tuple(
                dimension
                for dimension in dimensions
                if len(modalities_by_dimension[dimension]) == 1
            )
            beliefs.append(
                FactorBelief(
                    factor=factor,
                    timescales=self.config.factor_timescales[factor],
                    evidence_status=status,
                    posterior=self._distribution(posterior, dimensions, indices),
                    evidence_event_ids=event_ids,
                    shared_latent_dimensions=shared,
                    modality_private_dimensions=private,
                )
            )
        return tuple(beliefs)

    def _identifiable_dimensions(
        self,
        observations: Iterable[ObservationEvent],
        as_of_seconds: float,
    ) -> set[str]:
        matrices = [
            self._current_observation_matrix(event, as_of_seconds)
            for event in observations
            if event.missingness.status is MissingnessStatus.OBSERVED
        ]
        identifiable: set[str] = set()
        if not matrices:
            return identifiable
        combined = np.vstack(matrices)
        projection = np.linalg.pinv(combined) @ combined
        for index, dimension in enumerate(self.config.state_dimensions):
            basis = np.eye(len(self.config.state_dimensions))[index]
            if np.allclose(projection @ basis, basis, atol=1e-8):
                identifiable.add(dimension)
        return identifiable

    def _current_observation_matrix(
        self, event: ObservationEvent, as_of_seconds: float
    ) -> FloatArray:
        observation = np.asarray(
            self.config.observations_by_key[event.modality.key.casefold()].matrix,
            dtype=float,
        )
        elapsed = as_of_seconds - event.time_seconds
        if elapsed < 0:
            raise ValueError("observation cannot be later than the state being characterized")
        backward_transition = expm(-np.asarray(self.config.drift_matrix, dtype=float) * elapsed)
        return cast(FloatArray, observation @ backward_transition)

    def _forecast_factor_beliefs(
        self, posterior: GaussianState, source_factors: tuple[FactorBelief, ...]
    ) -> tuple[FactorBelief, ...]:
        source_by_factor = {factor.factor: factor for factor in source_factors}
        propagated: list[FactorBelief] = []
        for factor in StateFactor:
            source = source_by_factor[factor]
            dimensions = self.config.factor_dimensions.get(factor, ())
            if not dimensions:
                propagated.append(source)
                continue
            indices = [self.config.state_dimensions.index(dimension) for dimension in dimensions]
            status = (
                EvidenceStatus.UNIDENTIFIABLE
                if source.evidence_status is EvidenceStatus.UNIDENTIFIABLE
                else EvidenceStatus.INFERRED
            )
            propagated.append(
                FactorBelief(
                    factor=factor,
                    timescales=source.timescales,
                    evidence_status=status,
                    posterior=self._distribution(posterior, dimensions, indices),
                    evidence_event_ids=source.evidence_event_ids,
                    shared_latent_dimensions=source.shared_latent_dimensions,
                    modality_private_dimensions=source.modality_private_dimensions,
                )
            )
        return tuple(propagated)

    def _target_predictions(
        self, posterior: GaussianState, belief: CellStateBelief, horizon_seconds: float
    ) -> tuple[TargetPrediction, ...]:
        predictions: list[TargetPrediction] = []
        dimensions = self.config.state_dimensions
        for target in belief.query.target_outputs:
            key = target.term.key.casefold()
            if key not in dimensions:
                predictions.append(
                    TargetPrediction(
                        target=target,
                        units=target.units,
                        horizon_seconds=horizon_seconds,
                        status=SupportStatus.UNSUPPORTED,
                        distribution=UnavailableDistribution(
                            reason_code="missing_output_decoder",
                            message=f"No reference output decoder exists for {target.term.key!r}.",
                        ),
                    )
                )
                continue
            index = dimensions.index(key)
            predictions.append(
                TargetPrediction(
                    target=target,
                    units=target.units,
                    horizon_seconds=horizon_seconds,
                    status=SupportStatus.SUPPORTED,
                    distribution=self._distribution(posterior, (dimensions[index],), (index,)),
                    notes=(
                        "Identity readout from one reference latent dimension; not a validated "
                        "functional output model.",
                    ),
                )
            )
        return tuple(predictions)

    def _dynamics(
        self, posterior: GaussianState, time: float, events: Iterable[object]
    ) -> DynamicSummary:
        matrix = np.asarray(self.config.drift_matrix, dtype=float)
        drive = np.asarray(self.config.drift_vector, dtype=float)
        drive = drive + self._active_control(time, events) + self._environment_drive(time, events)
        velocity_mean = matrix @ posterior.mean + drive
        velocity_covariance = matrix @ posterior.covariance @ matrix.T
        velocity = ParametricDistribution(
            family="multivariate_normal",
            dimensions=tuple(f"d({name})/dt" for name in self.config.state_dimensions),
            mean=_as_tuple_vector(velocity_mean),
            covariance=_as_tuple_matrix(_stabilize_covariance(velocity_covariance)),
        )
        spectral_abscissa = float(np.max(np.real(np.linalg.eigvals(matrix))))
        return DynamicSummary(
            velocity=velocity,
            stability=EvaluatedScalar(
                status=SupportStatus.SUPPORTED,
                value=spectral_abscissa,
                units="1/s",
                reason="Maximum real eigenvalue of the reference drift matrix.",
            ),
            division_hazard=_unsupported("No discrete division model is configured."),
            death_hazard=_unsupported("No discrete death model is configured."),
            bifurcation_proximity=_unsupported(
                "Linear reference dynamics have no bifurcation model."
            ),
            recovery_timescale=_unsupported(
                "Recovery modes are not parameterized by this backend."
            ),
        )

    def _uncertainty(self, posterior: GaussianState) -> UncertaintyBreakdown:
        return UncertaintyBreakdown(
            components=(
                UncertaintyComponent(
                    kind=UncertaintyKind.MEASUREMENT,
                    status=SupportStatus.NOT_EVALUATED,
                    notes=(
                        "Posterior covariance combines prior, process, and measurement effects; "
                        "this backend does not identify a measurement-only component."
                    ),
                ),
                UncertaintyComponent(
                    kind=UncertaintyKind.BIOLOGICAL,
                    status=SupportStatus.SUPPORTED,
                    magnitude=float(
                        np.trace(np.asarray(self.config.process_covariance_per_second, dtype=float))
                    ),
                    metric="process_covariance_trace_per_second",
                ),
                UncertaintyComponent(
                    kind=UncertaintyKind.PARAMETER,
                    status=SupportStatus.UNSUPPORTED,
                    notes="Reference matrices are fixed; parameter posterior is not modeled.",
                ),
                UncertaintyComponent(
                    kind=UncertaintyKind.MODEL,
                    status=SupportStatus.UNSUPPORTED,
                    notes="No model ensemble or epistemic uncertainty estimator is configured.",
                ),
                UncertaintyComponent(
                    kind=UncertaintyKind.COUNTERFACTUAL,
                    status=SupportStatus.UNSUPPORTED,
                    notes="Counterfactual uncertainty requires experimental validation data.",
                ),
            )
        )

    def _observability(
        self, observations: Iterable[ObservationEvent], as_of_seconds: float
    ) -> ObservabilityReport:
        observation_tuple = tuple(observations)
        observed: set[str] = set()
        for event in observation_tuple:
            if event.missingness.status is not MissingnessStatus.OBSERVED:
                continue
            model = self.config.observations_by_key[event.modality.key.casefold()]
            if math.isclose(event.time_seconds, as_of_seconds, rel_tol=0, abs_tol=1e-12):
                observed.update(model.direct_dimensions)
        identifiable = self._identifiable_dimensions(observation_tuple, as_of_seconds)
        inferred = identifiable - observed
        unidentifiable = set(self.config.state_dimensions) - identifiable
        assigned_factors = set(self.config.factor_dimensions)
        unsupported = {
            f"factor:{factor.value}" for factor in StateFactor if factor not in assigned_factors
        }
        return ObservabilityReport(
            observed=tuple(sorted(observed)),
            inferred_with_support=tuple(sorted(inferred)),
            unidentifiable=tuple(sorted(unidentifiable)),
            unsupported_by_model=tuple(sorted(unsupported)),
        )

    def _recommend_measurement(
        self, _query: StateQuery, _posterior: GaussianState
    ) -> MeasurementRecommendation:
        return MeasurementRecommendation(
            status=SupportStatus.NOT_EVALUATED,
            rationale=(
                "The reference backend intentionally abstains: generic posterior covariance "
                "reduction is not decision-relevant value of information about future "
                "intervention outcomes."
            ),
        )

    def _build_belief(
        self,
        request: EstimateCellStateRequest,
        posterior: GaussianState,
        observations: tuple[ObservationEvent, ...],
        seed: int,
    ) -> CellStateBelief:
        factors = self._factor_beliefs(posterior, observations, request.as_of_seconds)
        latest_environment = _latest_environment(request.history.events, request.as_of_seconds)
        active_interventions = tuple(
            event
            for event in request.history.events
            if isinstance(event, InterventionEvent)
            and event.time_seconds
            <= request.as_of_seconds
            < event.time_seconds + event.duration_seconds
        )
        diagnostics = BeliefDiagnostics(
            ood=OODReport(
                status=SupportStatus.NOT_EVALUATED,
                notes=("No training-support or OOD detector is configured.",),
            ),
            sufficiency=SufficiencyReport(
                status=SupportStatus.NOT_EVALUATED,
                notes=(
                    "State-vs-state-plus-history sufficiency requires held-out future outcomes.",
                ),
            ),
            observability=self._observability(observations, request.as_of_seconds),
        )
        source_ids = tuple(
            event.event_id for event in request.history.through(request.as_of_seconds)
        )
        provenance = ProvenanceRecord(
            model_id=self.config.model_id,
            model_version=self.config.model_version,
            model_fingerprint=self.model_fingerprint,
            posterior_schema_id=self.posterior_schema_id,
            query_fingerprint=request.query.fingerprint,
            history_fingerprint=request.history.fingerprint,
            history_structure_fingerprint=request.history.structure_fingerprint,
            context_fingerprint=request.context_fingerprint,
            source_event_ids=source_ids,
            source_event_fingerprints={
                event.event_id: canonical_fingerprint(event)
                for event in request.history.through(request.as_of_seconds)
            },
            seed=seed,
            warnings=(
                "Reference linear-Gaussian result; not biologically validated.",
                "The synthetic prior is context-agnostic apart from a supported-species guard.",
            ),
        )
        belief_name = (
            f"{self.model_fingerprint}:{self.posterior_schema_id}:{request.history.subject_id}:"
            f"{request.as_of_seconds}:{request.query.fingerprint}:"
            f"{request.history.fingerprint}:{request.context_fingerprint}:{seed}"
        )
        return CellStateBelief(
            belief_id=uuid5(NAMESPACE_URL, belief_name),
            subject_id=request.history.subject_id,
            as_of_seconds=request.as_of_seconds,
            query=request.query,
            query_fingerprint=request.query.fingerprint,
            history_fingerprint=request.history.fingerprint,
            context_fingerprint=request.context_fingerprint,
            status=BeliefStatus.PARTIAL,
            joint_posterior=self._distribution(posterior),
            factors=factors,
            context=ContextBelief(
                active_interventions=active_interventions,
                soluble_environment=latest_environment,
                unsupported_dimensions=(
                    "static_context_conditioning",
                    "population_context",
                    "lineage_context",
                    "physical_environment",
                    "neighborhood",
                    "spatial_position",
                ),
            ),
            dynamics=self._dynamics(posterior, request.as_of_seconds, request.history.events),
            uncertainty=self._uncertainty(posterior),
            diagnostics=diagnostics,
            next_measurement=self._recommend_measurement(request.query, posterior),
            provenance=provenance,
        )


class LinearGaussianPlanner:
    """Risk-adjusted selector over explicit candidate scenarios."""

    def __init__(self, evolution_model: LinearGaussianReference) -> None:
        self.evolution_model = evolution_model

    @property
    def descriptor(self) -> EstimatorDescriptor:
        return self.evolution_model.descriptor

    def choose(
        self,
        belief: CellStateBelief,
        objective: InterventionObjective,
        candidates: Sequence[EvolutionScenario],
        *,
        options: InferenceOptions,
    ) -> InterventionPlan:
        evaluations: list[CandidateEvaluation] = []
        for candidate in candidates:
            forecast = self.evolution_model.evolve(belief, candidate, options=options)
            utility = 0.0
            uncertainty_penalty = 0.0
            notes: list[str] = []
            supported = True
            predictions = {
                prediction.target.term.key: prediction for prediction in forecast.target_predictions
            }
            for term in objective.terms:
                prediction = predictions.get(term.target.key)
                if prediction is None:
                    supported = False
                    notes.append(f"Missing objective target: {term.target.key}")
                    continue
                distribution = prediction.distribution
                if (
                    prediction.status is not SupportStatus.SUPPORTED
                    or not isinstance(distribution, ParametricDistribution)
                    or len(distribution.mean) != 1
                ):
                    supported = False
                    notes.append(
                        f"Objective target lacks a supported scalar prediction: {term.target.key}"
                    )
                    continue
                expected = float(distribution.mean[0])
                if term.direction is ObjectiveDirection.MAXIMIZE:
                    utility += term.weight * expected
                elif term.direction is ObjectiveDirection.MINIMIZE:
                    utility -= term.weight * expected
                else:
                    assert term.target_value is not None
                    utility -= term.weight * abs(expected - term.target_value.value)
                uncertainty_penalty += (
                    objective.risk_aversion
                    * term.weight
                    * math.sqrt(max(0.0, float(distribution.covariance[0][0])))
                )
            evaluations.append(
                CandidateEvaluation(
                    scenario_id=candidate.scenario_id,
                    expected_utility=utility if supported else None,
                    uncertainty_penalty=uncertainty_penalty if supported else None,
                    selection_score=(utility - uncertainty_penalty) if supported else None,
                    supported=supported,
                    notes=tuple(notes),
                )
            )
        supported_evaluations = [item for item in evaluations if item.supported]
        selected = (
            max(
                supported_evaluations,
                key=lambda item: cast(float, item.selection_score),
            ).scenario_id
            if supported_evaluations
            else None
        )
        candidate_fingerprints = tuple(canonical_fingerprint(item) for item in candidates)
        payload = (
            f"{belief.belief_id}:{canonical_fingerprint(objective)}:"
            f"{candidate_fingerprints}:{selected}:{self.evolution_model.model_fingerprint}:"
            f"{options.seed}"
        )
        objective_fingerprint = canonical_fingerprint(objective)
        candidate_references = tuple(
            ScenarioReference(
                scenario_id=candidate.scenario_id,
                fingerprint=canonical_fingerprint(candidate),
            )
            for candidate in candidates
        )
        provenance = ProvenanceRecord(
            model_id=self.evolution_model.config.model_id,
            model_version=self.evolution_model.config.model_version,
            model_fingerprint=self.evolution_model.model_fingerprint,
            posterior_schema_id=self.evolution_model.posterior_schema_id,
            query_fingerprint=belief.query_fingerprint,
            history_fingerprint=belief.history_fingerprint,
            history_structure_fingerprint=belief.provenance.history_structure_fingerprint,
            context_fingerprint=belief.context_fingerprint,
            source_event_ids=belief.provenance.source_event_ids,
            source_event_fingerprints=belief.provenance.source_event_fingerprints,
            seed=options.seed,
            warnings=("Reference linear-Gaussian intervention plan; not biologically validated.",),
        )
        return InterventionPlan(
            plan_id=sha256(payload.encode()).hexdigest()[:24],
            parent_belief_id=belief.belief_id,
            query_fingerprint=belief.query_fingerprint,
            horizon_name=objective.horizon_name,
            objective_id=objective.objective_id,
            objective_fingerprint=objective_fingerprint,
            candidates=candidate_references,
            selected_scenario_id=selected,
            evaluations=tuple(evaluations),
            rationale=(
                "Selected the highest reference-model expected utility after a marginal Gaussian "
                "risk penalty. This is not a biologically validated intervention recommendation."
            ),
            seed=options.seed,
            provenance=provenance,
        )


def sample_posterior(
    belief: CellStateBelief, count: int, *, seed: int
) -> tuple[tuple[float, ...], ...]:
    """Draw reproducible samples without mutating global random state."""

    if count <= 0:
        raise ValueError("sample count must be positive")
    distribution = belief.joint_posterior
    if not isinstance(distribution, ParametricDistribution):
        raise PosteriorCompatibilityError("sampling helper requires a parametric posterior")
    generator = np.random.default_rng(seed)
    samples = generator.multivariate_normal(
        np.asarray(distribution.mean, dtype=float),
        np.asarray(distribution.covariance, dtype=float),
        size=count,
        check_valid="raise",
    )
    return tuple(tuple(float(value) for value in row) for row in samples)


def minimal_reference_config() -> LinearGaussianConfig:
    """Return a four-dimensional demonstration configuration used by examples and tests."""

    dimensions = ("memory", "signaling", "metabolic_capacity", "functional_capacity")
    return LinearGaussianConfig(
        state_dimensions=dimensions,
        prior_mean=(0.0, 0.0, 0.0, 0.0),
        prior_covariance=(
            (1.0, 0.0, 0.0, 0.0),
            (0.0, 1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        ),
        drift_matrix=(
            (-0.0001, 0.0, 0.0, 0.0),
            (0.0, -0.02, 0.0, 0.0),
            (0.0, 0.0, -0.002, 0.0),
            (0.0, 0.01, 0.005, -0.001),
        ),
        drift_vector=(0.0, 0.0, 0.0, 0.0),
        process_covariance_per_second=(
            (0.00001, 0.0, 0.0, 0.0),
            (0.0, 0.001, 0.0, 0.0),
            (0.0, 0.0, 0.0001, 0.0),
            (0.0, 0.0, 0.0, 0.0001),
        ),
        observation_models=(
            LinearObservationConfig(
                modality_key="transcriptome",
                units="relative",
                matrix=((1.0, 0.0, 0.0, 0.0),),
                noise_covariance=((0.25,),),
            ),
            LinearObservationConfig(
                modality_key="phosphosignaling",
                units="relative",
                matrix=((0.0, 1.0, 0.0, 0.0),),
                noise_covariance=((0.1,),),
            ),
            LinearObservationConfig(
                modality_key="metabolome",
                units="relative",
                matrix=((0.0, 0.0, 1.0, 0.0),),
                noise_covariance=((0.2,),),
            ),
            LinearObservationConfig(
                modality_key="functional_readout",
                units="relative",
                matrix=((0.0, 0.0, 0.0, 1.0),),
                noise_covariance=((0.15,),),
                direct_dimensions=("functional_capacity",),
            ),
        ),
        control_effects={
            "drug": (0.0, -0.02, -0.01, 0.01),
            "cytokine": (0.0, 0.03, -0.005, 0.015),
        },
        control_dose_units={"drug": "relative", "cytokine": "relative"},
        environment_effects={"nutrient": (0.0, 0.0, 0.01, 0.005)},
        environment_units={"nutrient": "relative"},
        factor_dimensions={
            StateFactor.SLOW_MEMORY: ("memory",),
            StateFactor.SIGNALING: ("signaling",),
            StateFactor.METABOLIC: ("metabolic_capacity",),
            StateFactor.FUNCTIONAL_CAPACITY: ("functional_capacity",),
        },
        output_units={"functional_capacity": "relative"},
    )


def _measurement_vector(observation: ObservationEvent) -> FloatArray:
    value = observation.value
    if isinstance(value, bool) or value is None:
        raise ValueError("reference observations require numeric values")
    if isinstance(value, (int, float)):
        array = np.asarray([value], dtype=float)
    elif isinstance(value, list) and all(
        not isinstance(item, bool) and isinstance(item, (int, float)) for item in value
    ):
        array = np.asarray(value, dtype=float)
    else:
        raise ValueError("reference observations require a numeric scalar or vector")
    if not np.isfinite(array).all():
        raise ValueError("reference observation values must be finite")
    return cast(FloatArray, array)


def _discretize_linear_system(
    drift: FloatArray, process_covariance: FloatArray, elapsed: float
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Exactly discretize constant linear drift/input and continuous Gaussian process noise."""

    size = drift.shape[0]
    input_block = np.zeros((2 * size, 2 * size), dtype=float)
    input_block[:size, :size] = drift
    input_block[:size, size:] = np.eye(size)
    input_exponential = expm(input_block * elapsed)
    transition = input_exponential[:size, :size]
    input_integral = input_exponential[:size, size:]

    van_loan = np.zeros((2 * size, 2 * size), dtype=float)
    van_loan[:size, :size] = drift
    van_loan[:size, size:] = process_covariance
    van_loan[size:, size:] = -drift.T
    van_loan_exponential = expm(van_loan * elapsed)
    discrete_process = van_loan_exponential[:size, size:] @ transition.T
    return (
        cast(FloatArray, transition),
        cast(FloatArray, input_integral),
        _stabilize_covariance(cast(FloatArray, discrete_process)),
    )


def _latest_environment(events: Iterable[object], time: float) -> dict[str, object]:
    latest: dict[str, tuple[float, object]] = {}
    for event in events:
        if isinstance(event, EnvironmentEvent) and event.time_seconds <= time:
            for key, value in event.variables.items():
                normalized = key.casefold()
                if normalized not in latest or event.time_seconds >= latest[normalized][0]:
                    serialized = (
                        value.model_dump(mode="json") if isinstance(value, Quantity) else value
                    )
                    latest[normalized] = (event.time_seconds, serialized)
    return {key: value for key, (_, value) in sorted(latest.items())}
