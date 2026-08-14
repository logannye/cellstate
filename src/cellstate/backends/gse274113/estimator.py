"""The estimator that emits a `CellStateBelief` from real GSE274113 cells.

This is the first non-synthetic construction site for the project's central object.  Everything the
belief claims is bounded by ADR 0021 decision 5: it is a **snapshot state estimate with a
posterior**, and it is not a sufficiency result, not a faithfulness verdict, and not evidence that
the state is complete.  S1, S3 and S7 are structurally unreachable on this series -- no library
spans a timepoint -- and the belief says so in its own fields rather than only in prose.

The eight-dimensional state maps onto the belief's own structure rather than being flattened into
one opaque vector:

* the four biology dimensions become a ``REGULATORY`` factor -- the perturbations are CRISPRi
  knockdowns of transcription factors;
* the three nuisance dimensions become the ``nuisance`` block, which is what that field is for;
* the single realization coefficient becomes an ``InterventionRealizationBelief`` for the guide.

Each block is the **exact marginal** of the joint, sliced from the same covariance, so the pieces
cannot drift apart from the whole.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ...domain.belief import (
    BeliefDiagnostics,
    BeliefStatus,
    CalibrationReport,
    CausalSupportReport,
    CellStateBelief,
    ContextBelief,
    DecisionUncertaintyReport,
    DimensionIdentifiability,
    DynamicSummary,
    EvaluatedScalar,
    EvaluationStatus,
    FactorBelief,
    IdentifiabilityReport,
    InterventionRealizationBelief,
    NuisanceBelief,
    QueryReadinessReport,
    SufficiencyReport,
    SupportReport,
    UncertaintyBreakdown,
    UncertaintyComponent,
    UncertaintyKind,
)
from ...domain.common import (
    CausalStatus,
    CriterionOutcome,
    EvidenceStatus,
    OntologyTerm,
    ProvenanceRecord,
    SupportStatus,
    canonical_fingerprint,
)
from ...domain.distributions import ParametricDistribution, UnavailableDistribution
from ...domain.events import InterventionEvent, ObservationEvent
from ...domain.query import StateQuery, Timescale
from ...domain.request import EstimateCellStateRequest, InferenceOptions
from ...domain.specification import (
    CompiledStateSpecification,
    ExcludedStateFactor,
    StateFactor,
    StateFactorSpecification,
)
from ...ports.models import (
    CapabilityReport,
    EstimatorDescriptor,
    ModelArtifactKind,
    QueryCompilerDescriptor,
    estimation_capability_scope_fingerprint,
)
from .fit import FittedFold
from .likelihood import posterior

MODEL_ID = "gse274113-rna-observation-model"
MODEL_VERSION = "1.0.0"
POSTERIOR_SCHEMA_ID = "gse274113-rna-state-v1"
FAMILY = "delta_method_multinomial_gaussian"

__all__ = ["GSE274113ObservationEstimator"]


def _unavailable(reason: str) -> EvaluatedScalar:
    return EvaluatedScalar(status=SupportStatus.UNSUPPORTED, value=None, reason=reason)


@dataclass(frozen=True)
class _Blocks:
    """Index ranges of the three state blocks inside the joint."""

    biology: tuple[int, ...]
    nuisance: tuple[int, ...]
    realization: int


class GSE274113ObservationEstimator:
    """A fitted observation model for one leave-one-library-out fold."""

    def __init__(
        self,
        fold: FittedFold,
        *,
        query: StateQuery,
        slice_fingerprint: str,
        panel_fingerprint: str,
    ) -> None:
        self._fold = fold
        self._query = query
        self._slice_fingerprint = slice_fingerprint
        self._panel_fingerprint = panel_fingerprint
        biology_rank = fold.biology_basis.shape[1]
        nuisance_rank = fold.nuisance_basis.shape[1]
        self._blocks = _Blocks(
            biology=tuple(range(biology_rank)),
            nuisance=tuple(range(biology_rank, biology_rank + nuisance_rank)),
            realization=biology_rank + nuisance_rank,
        )
        # OntologyTerm.key normalizes case, so an event's target arrives as "gata1" while the fold
        # is keyed by the deposited symbol "GATA1".  Resolve through the key rather than comparing
        # raw strings, which would silently make every arm unsupported.
        self._target_by_key = {
            OntologyTerm(label=name).key: name for name in fold.target_directions
        }
        self._dimensions = tuple(
            [f"biology_{index}" for index in range(biology_rank)]
            + [f"library_nuisance_{index}" for index in range(nuisance_rank)]
            + ["guide_realization"]
        )

    # ---------------------------------------------------------------- identity

    @property
    def model_fingerprint(self) -> str:
        return canonical_fingerprint(
            {
                "model": MODEL_ID,
                "version": MODEL_VERSION,
                "fold": self._fold.held_out_library,
                "fit_libraries": sorted(self._fold.fit_library_ids),
                "slice": self._slice_fingerprint,
                "panel": self._panel_fingerprint,
            }
        )

    @property
    def descriptor(self) -> EstimatorDescriptor:
        training_id = f"gse274113-fold-{self._fold.held_out_library}"
        validation_id = f"gse274113-heldout-{self._fold.held_out_library}"
        return EstimatorDescriptor(
            model_id=MODEL_ID,
            model_version=MODEL_VERSION,
            model_fingerprint=self.model_fingerprint,
            posterior_schema_id=POSTERIOR_SCHEMA_ID,
            description=(
                "RNA observation model fitted on GSE274113 panel pseudobulk, leaving out library "
                f"{self._fold.held_out_library}. Reports a population state estimate and its "
                "posterior; claims no identified causal effect and no control capability."
            ),
            artifact_kind=ModelArtifactKind.EMPIRICAL_OBSERVATION_MODEL,
            support_envelope_id=f"gse274113-rna-obs-envelope-{self._fold.held_out_library}",
            support_envelope_fingerprint=canonical_fingerprint(
                {"envelope": MODEL_ID, "fold": self._fold.held_out_library}
            ),
            training_support_id=training_id,
            training_support_fingerprint=canonical_fingerprint(
                {"training": training_id, "libraries": sorted(self._fold.fit_library_ids)}
            ),
            validation_evidence_ids=(validation_id,),
            validation_evidence_fingerprints={
                validation_id: canonical_fingerprint(
                    {"validation": validation_id, "library": self._fold.held_out_library}
                )
            },
        )

    # ---------------------------------------------------------------- compiling

    @property
    def query_compiler(self) -> GSE274113ObservationEstimator:
        return self

    @property
    def compiler_descriptor(self) -> QueryCompilerDescriptor:
        return QueryCompilerDescriptor(
            compiler_id="gse274113-rna-observation-compiler",
            compiler_version=MODEL_VERSION,
            compiler_fingerprint=canonical_fingerprint(
                {"compiler": "gse274113-rna-observation-compiler", "model": self.model_fingerprint}
            ),
        )

    def compile(self, query: StateQuery) -> CompiledStateSpecification:
        """Compile the query verbatim, naming only the factors this model actually carries."""

        target_keys = tuple(output.term.key for output in query.target_outputs)
        descriptor = self.compiler_descriptor
        active = (
            StateFactorSpecification(
                factor=StateFactor.REGULATORY,
                dimensions=tuple(self._dimensions[index] for index in self._blocks.biology),
                timescales=frozenset({Timescale.SLOW}),
                required_for_outputs=target_keys,
                rationale=(
                    "The perturbations are CRISPRi knockdowns of transcription factors, so the "
                    "biology subspace fitted from within-library contrasts is regulatory."
                ),
            ),
        )
        return CompiledStateSpecification(
            query_fingerprint=query.fingerprint,
            subject=query.subject,
            compiler_id=descriptor.compiler_id,
            compiler_version=descriptor.compiler_version,
            compiler_fingerprint=descriptor.compiler_fingerprint,
            active_factors=active,
            # The nuisance and realization blocks are declared here, not smuggled into the factor.
            # The library axis is an observation nuisance, and the guide coefficient is the
            # realization of the declared intervention; the compiled joint is the union of all
            # three, which is what makes each belief block checkable against the whole.
            observation_nuisance_dimensions=tuple(
                self._dimensions[index] for index in self._blocks.nuisance
            ),
            intervention_realization_dimensions=(self._dimensions[self._blocks.realization],),
            excluded_factors=tuple(
                ExcludedStateFactor(
                    factor=factor,
                    rationale=(
                        "Not carried by an RNA panel observation model fitted on a single "
                        "destructive snapshot."
                    ),
                )
                for factor in StateFactor
                if factor is not StateFactor.REGULATORY
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
            target_output_keys=target_keys,
            horizon_names=tuple(horizon.name for horizon in query.prediction_horizons),
            admissible_evidence_roles=query.evidence_policy.allowed_evidence_roles,
            acceptance_thresholds=query.acceptance_thresholds,
        )

    # ------------------------------------------------------------- preflight

    def capabilities(
        self,
        request: EstimateCellStateRequest,
        state_specification: CompiledStateSpecification,
    ) -> CapabilityReport:
        """Predict every foreseeable failure here, because ``estimate`` may not raise.

        ``estimate_cell_state`` does not wrap exceptions from this method or from ``estimate``, so a
        condition not turned into a blocker here becomes an untyped traceback at the public
        boundary.
        """

        scope = estimation_capability_scope_fingerprint(request, state_specification)
        blockers: list[str] = []

        library = request.history.subject.experimental_unit_id
        # The leakage guard.  Its false branch is reachable and is tested.
        if library in self._fold.fit_library_ids:
            blockers.append(
                f"library {library} is inside this fold's fit set; a belief for it would be an "
                "in-sample estimate"
            )
        if library != self._fold.held_out_library:
            blockers.append(
                f"this fold answers only library {self._fold.held_out_library}, not {library}"
            )

        target = self._arm_target(request)
        if target is None:
            blockers.append("exactly one active CRISPRi guide is required to identify the arm")
        elif target not in self._fold.target_directions:
            blockers.append(f"target {target} has no fitted direction in this fold")

        observations = [
            event for event in request.history.events if isinstance(event, ObservationEvent)
        ]
        if len(observations) != 1:
            blockers.append("exactly one panel observation is required")

        return CapabilityReport(
            supported=not blockers,
            scope_fingerprint=scope,
            unsupported_subjects=tuple(blockers),
        )

    def _arm_target(self, request: EstimateCellStateRequest) -> str | None:
        targets = [
            self._target_by_key.get(event.target.key)
            for event in request.history.events
            if isinstance(event, InterventionEvent) and event.target is not None
        ]
        return targets[0] if len(targets) == 1 else None

    # -------------------------------------------------------------- estimating

    def estimate(
        self,
        request: EstimateCellStateRequest,
        *,
        options: InferenceOptions | None = None,
    ) -> CellStateBelief:
        resolved = options or InferenceOptions()
        target = self._arm_target(request)
        if target is None:  # pragma: no cover - capabilities blocks this first
            raise ValueError("an arm requires exactly one active guide")

        observation = next(
            event for event in request.history.events if isinstance(event, ObservationEvent)
        )
        intervention = next(
            event for event in request.history.events if isinstance(event, InterventionEvent)
        )
        value = observation.value
        if not isinstance(value, Sequence) or isinstance(value, str):
            raise TypeError("a panel observation must carry a sequence of log-composition entries")
        entries: list[float] = []
        for entry in value:
            if not isinstance(entry, int | float):
                raise TypeError("panel log-composition entries must be numeric")
            entries.append(float(entry))
        log_composition = np.asarray(entries, dtype=np.float64)
        attributes = observation.assay.attributes
        # Assay attributes are JsonValue, so the depth is narrowed explicitly rather than
        # coerced blindly: a non-numeric attribute here is a contract violation, not a fallback.
        raw_total = attributes["panel_total"]
        if not isinstance(raw_total, int | float):
            raise TypeError("panel_total assay attribute must be numeric")
        panel_total = float(raw_total)
        counts = np.asarray(
            np.rint(np.exp(log_composition) * (panel_total + log_composition.size / 2.0) - 0.5),
            dtype=np.int64,
        )

        mean, covariance = posterior(
            log_composition,
            intercept=self._fold.intercept,
            design=self._fold.design(target),
            prior_precision=self._fold.prior_precision(),
            observation_variance_diagonal=self._fold.observation_variance(panel_total),
        )
        joint = self._distribution(mean, covariance, self._dimensions)

        state_specification = self.compile(request.query)
        evidence_ids = (observation.event_id,)
        descriptor = self.descriptor

        return CellStateBelief(
            subject=request.history.subject,
            as_of_seconds=request.as_of_seconds,
            query=request.query,
            query_fingerprint=request.query.fingerprint,
            history_fingerprint=request.history.fingerprint,
            context_fingerprint=request.context_fingerprint,
            state_specification=state_specification,
            status=BeliefStatus.COMPLETE,
            joint_posterior=joint,
            factors=(
                FactorBelief(
                    factor=StateFactor.REGULATORY,
                    timescales=frozenset({Timescale.SLOW}),
                    evidence_status=EvidenceStatus.OBSERVED,
                    posterior=self._marginal(mean, covariance, self._blocks.biology),
                    evidence_event_ids=evidence_ids,
                ),
            ),
            context=ContextBelief(
                active_interventions=(intervention,),
                unsupported_dimensions=(
                    "soluble_environment: the deposit records no per-library medium, oxygen or "
                    "temperature, so no environment axis is declared or inferred",
                ),
            ),
            intervention_realizations=(
                InterventionRealizationBelief(
                    intervention_event_id=intervention.event_id,
                    evidence_status=EvidenceStatus.INFERRED,
                    posterior=self._marginal(mean, covariance, (self._blocks.realization,)),
                    evidence_event_ids=evidence_ids,
                ),
            ),
            nuisance=NuisanceBelief(
                posterior=self._marginal(mean, covariance, self._blocks.nuisance),
                evidence_event_ids=evidence_ids,
            ),
            dynamics=self._dynamics(),
            uncertainty=self._uncertainty(counts, panel_total, target, covariance),
            diagnostics=self._diagnostics(),
            readiness=self._readiness(),
            provenance=ProvenanceRecord(
                support_envelope_id=descriptor.support_envelope_id,
                support_envelope_fingerprint=descriptor.support_envelope_fingerprint,
                training_support_id=descriptor.training_support_id,
                training_support_fingerprint=descriptor.training_support_fingerprint,
                validation_evidence_ids=descriptor.validation_evidence_ids,
                validation_evidence_fingerprints=descriptor.validation_evidence_fingerprints,
                source_event_fingerprints={
                    event.event_id: canonical_fingerprint(event) for event in request.history.events
                },
                model_id=MODEL_ID,
                model_version=MODEL_VERSION,
                model_fingerprint=self.model_fingerprint,
                posterior_schema_id=POSTERIOR_SCHEMA_ID,
                query_fingerprint=request.query.fingerprint,
                history_fingerprint=request.history.fingerprint,
                context_fingerprint=request.context_fingerprint,
                seed=resolved.seed,
                source_event_ids=tuple(event.event_id for event in request.history.events),
                history_structure_fingerprint=request.history.structure_fingerprint,
                code_revision=f"fold-{self._fold.held_out_library}",
                warnings=(
                    "S1, S3 and S7 are structurally unavailable on GSE274113: no library spans a "
                    "timepoint. This belief is a snapshot state estimate and is not a sufficiency "
                    "result or a faithfulness verdict (ADR 0021 decision 5).",
                ),
            ),
        )

    # ------------------------------------------------------------------ pieces

    def _distribution(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        dimensions: tuple[str, ...],
    ) -> ParametricDistribution:
        return ParametricDistribution(
            family=FAMILY,
            dimensions=dimensions,
            mean=tuple(float(value) for value in mean),
            covariance=tuple(tuple(float(value) for value in row) for row in covariance),
        )

    def _marginal(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        indices: tuple[int, ...],
    ) -> ParametricDistribution:
        """Slice the exact marginal, so a block can never drift from the joint."""

        index = np.asarray(indices, dtype=np.int64)
        return self._distribution(
            mean[index],
            covariance[np.ix_(index, index)],
            tuple(self._dimensions[position] for position in indices),
        )

    def _dynamics(self) -> DynamicSummary:
        no_time = (
            "no library in GSE274113 spans a timepoint, so no velocity, hazard or fate is "
            "identifiable from this evidence"
        )
        return DynamicSummary(
            velocity=UnavailableDistribution(
                reason_code="no_spanning_unit",
                message=no_time,
            ),
            stability=_unavailable(no_time),
            division_hazard=_unavailable(no_time),
            death_hazard=_unavailable(no_time),
            bifurcation_proximity=_unavailable(no_time),
            recovery_timescale=_unavailable(no_time),
        )

    def _uncertainty(
        self,
        counts: np.ndarray,
        panel_total: float,
        target: str,
        covariance: np.ndarray,
    ) -> UncertaintyBreakdown:
        """Separate measurement from biological uncertainty by recomputing without ``psi^2``."""

        from .likelihood import technical_variance

        technical_only = posterior(
            np.zeros(counts.shape[0], dtype=np.float64),
            intercept=np.zeros(counts.shape[0], dtype=np.float64),
            design=self._fold.design(target),
            prior_precision=self._fold.prior_precision(),
            observation_variance_diagonal=technical_variance(self._fold.pooled_rate, panel_total),
        )[1]
        measurement = float(np.trace(technical_only))
        total = float(np.trace(covariance))
        biological = max(total - measurement, 0.0)
        unsupported = (
            "point estimates from one fold; no ensemble over fits and no counterfactual validation"
        )
        return UncertaintyBreakdown(
            components=(
                UncertaintyComponent(
                    kind=UncertaintyKind.MEASUREMENT,
                    status=SupportStatus.SUPPORTED,
                    magnitude=measurement,
                ),
                UncertaintyComponent(
                    kind=UncertaintyKind.BIOLOGICAL,
                    status=SupportStatus.SUPPORTED,
                    magnitude=biological,
                ),
                UncertaintyComponent(
                    kind=UncertaintyKind.PARAMETER,
                    status=SupportStatus.UNSUPPORTED,
                    notes=unsupported,
                ),
                UncertaintyComponent(
                    kind=UncertaintyKind.MODEL,
                    status=SupportStatus.UNSUPPORTED,
                    notes=unsupported,
                ),
                UncertaintyComponent(
                    kind=UncertaintyKind.COUNTERFACTUAL,
                    status=SupportStatus.UNSUPPORTED,
                    notes=unsupported,
                ),
            )
        )

    def _diagnostics(self) -> BeliefDiagnostics:
        # Every joint dimension is inferred through a full-rank design, never read off directly.
        dimension_status = {
            dimension: DimensionIdentifiability.INFERRED_WITH_SUPPORT
            for dimension in self._dimensions
        }
        no_span = (
            "no library in this series spans the inference cutoff, so the history block is empty "
            "and the comparison is inapplicable rather than passed"
        )
        return BeliefDiagnostics(
            support=SupportReport(
                # Not evaluated, because nothing here evaluates it.  This read EVALUATED / PASSED /
                # in_distribution_score=1.0 / ood_score=0.0 / abstention_required=False, and every
                # one of those was a literal: a uniform composition across all 100 panel genes, and
                # an inverted one, both received the identical certificate as a real held-out arm.
                # A report that returns the same verdict for data that could not exist is not a
                # measurement of support, and claiming a perfect in-distribution score for it is
                # worse than claiming nothing.
                #
                # The scores are optional precisely so a report can decline to invent them, so this
                # states the honest position rather than substituting a hastily-chosen statistic.
                # Computing a real one needs a decided estimand and a threshold predeclared before
                # the measurement -- and the threshold matters more than it looks, since this query
                # sets `maximum_ood_score=0.99`, against which a correctly computed score still
                # could not fail.  Both belong in one record; neither is repaired here.
                evaluation_status=EvaluationStatus.NOT_EVALUATED,
                outcome=CriterionOutcome.NOT_EVALUATED,
                maximum_ood_score=self._query.acceptance_thresholds.maximum_ood_score,
                abstention_required=True,
                notes=(
                    "no support envelope is computed by this observation model; the arm's position "
                    "relative to the fit libraries' distribution is not scored",
                ),
            ),
            sufficiency=SufficiencyReport(
                evaluation_status=EvaluationStatus.NOT_EVALUATED,
                outcome=CriterionOutcome.NOT_EVALUATED,
                maximum_history_information_gain=(
                    self._query.acceptance_thresholds.maximum_history_information_gain
                ),
                notes=(no_span,),
            ),
            identifiability=IdentifiabilityReport(
                evaluation_status=EvaluationStatus.NOT_EVALUATED,
                outcome=CriterionOutcome.NOT_EVALUATED,
                minimum_identifiability_score=(
                    self._query.acceptance_thresholds.minimum_identifiability
                ),
                dimension_status=dimension_status,
                notes=("identifiability is not scored by a single-arm observation model",),
            ),
            decision_uncertainty=DecisionUncertaintyReport(
                evaluation_status=EvaluationStatus.NOT_EVALUATED,
                outcome=CriterionOutcome.NOT_EVALUATED,
                maximum_decision_uncertainty=(
                    self._query.acceptance_thresholds.maximum_decision_uncertainty
                ),
                maximum_counterfactual_uncertainty=(
                    self._query.acceptance_thresholds.maximum_counterfactual_uncertainty
                ),
                notes=("no decision is requested of an observation model",),
            ),
            calibration=CalibrationReport(
                evaluation_status=EvaluationStatus.NOT_EVALUATED,
                outcome=CriterionOutcome.NOT_EVALUATED,
                minimum_coverage=self._query.acceptance_thresholds.minimum_calibration_coverage,
                maximum_calibration_error=(
                    self._query.acceptance_thresholds.maximum_calibration_error
                ),
                notes=("calibration is reported by the fold's held-out report, not per arm",),
            ),
            causal_support=CausalSupportReport(
                evaluation_status=EvaluationStatus.NOT_EVALUATED,
                outcome=CriterionOutcome.NOT_EVALUATED,
                causal_status=CausalStatus.UNSUPPORTED,
                blockers=(
                    "an empirical observation model makes no identified causal claim; "
                    "identification is gated by the admission registry (ADR 0021)",
                ),
            ),
        )

    def _readiness(self) -> QueryReadinessReport:
        return QueryReadinessReport(
            support=CriterionOutcome.NOT_EVALUATED,
            sufficiency=CriterionOutcome.NOT_EVALUATED,
            identifiability=CriterionOutcome.NOT_EVALUATED,
            decision_uncertainty=CriterionOutcome.NOT_EVALUATED,
            calibration=CriterionOutcome.NOT_EVALUATED,
            causal=CriterionOutcome.NOT_EVALUATED,
            # `measurement_model` read PASSED, and it is the one criterion `coherent_contract`
            # cannot contradict: the other six are cross-checked against a `BeliefDiagnostics`
            # report and this one has no counterpart to check against.  So it asserted itself.
            # The result was a belief reading "not valid for prediction, not valid for control,
            # abstention required -- but valid for measurement selection", from a query declaring
            # `available_assays=()`.  The reference estimator shows what earning it looks like:
            # `linear_gaussian.py` derives each criterion from `diagnostics.*.outcome` and reports
            # this one UNSUPPORTED.  Until a measurement-model report exists to be checked against,
            # NOT_EVALUATED is the only claim this backend can support.
            measurement_model=CriterionOutcome.NOT_EVALUATED,
            control_requested=True,
            valid_for_prediction=False,
            valid_for_control=False,
            valid_for_measurement_selection=False,
            # Abstention is the honest result, not a failure: the sufficiency test this project
            # defines faithfulness by is inapplicable on evidence with no spanning unit.
            abstention_required=True,
            reasons=(
                "predictive sufficiency is inapplicable: no library spans the inference cutoff",
                "this belief is a snapshot state estimate, not a faithfulness verdict",
            ),
        )
