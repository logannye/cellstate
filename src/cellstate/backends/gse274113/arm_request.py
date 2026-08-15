"""Build the typed request for one GSE274113 arm.

An *arm* is one ``(library, target)`` population at its own harvest instant, and it is the estimand
this backend answers.  The choices below are the ones that decide whether the resulting belief is
honest, so each is stated rather than left to be inferred from the code:

* **The subject is a POPULATION, and the experimental unit is the library.**  Cells here are
  destructively sampled with no cross-time identity, so no individual-cell estimand is available,
  and program rule 8 names the library as the unit a split may follow.
* **Exactly one horizon is declared.**  Declaring two would be a claim to S3, and no library in this
  series spans a timepoint.  The single horizon is named ``now`` so the query cannot be misread as a
  forecast.
* **The intervention space carries one spec per target, twenty in all.**  ``InterventionSpec``
  matches an event by comparing the target key exactly, so a single generic spec would match every
  arm and the "exactly one active intervention" check would be vacuous.
* **No environment axis is declared.**  Nothing in the deposited metadata records medium, oxygen or
  temperature per library, so declaring an axis would mean inventing values.  The absence is
  reported as an unsupported dimension rather than filled in.
* **``minimum_observed_measurements`` is 1.**  An arm whose panel total is zero was never measured;
  it fails this check and no belief exists for it, which is the zero-panel doctrine acting at the
  query rather than a special case in the estimator.
"""

from __future__ import annotations

import hashlib

from ...domain.common import OntologyTerm, Quantity
from ...domain.events import (
    ActualPerturbation,
    AssayMetadata,
    AssignmentMechanism,
    CollectionEffect,
    EvidenceLink,
    EvidenceRole,
    InterventionEvent,
    InterventionSchedule,
    MeasurementUncertainty,
    MissingnessReport,
    ObservationCollection,
    ObservationEvent,
    PerturbationStatus,
    QualityReport,
    ReversibilityStatus,
    ScheduleKind,
    StaticContext,
)
from ...domain.history import CellHistory, HistoryCompleteness, RecordCompleteness
from ...domain.query import (
    AcceptanceThresholds,
    EvidencePolicy,
    IntegerRange,
    InterventionSpec,
    LatentQuantityEndpoint,
    NumericDomain,
    OutputSpec,
    PredictionHorizon,
    QueryConstraints,
    RealizationEvidenceRequirement,
    ScalarRange,
    ScheduleDomain,
    StateQuery,
    SystemBoundary,
    TargetCensoringPolicy,
    TargetCensoringSemantics,
    TargetMissingnessPolicy,
    TargetMissingnessSemantics,
    Timescale,
    VersionedReference,
)
from ...domain.request import EstimateCellStateRequest
from ...domain.subjects import (
    AggregationStatistic,
    BeliefSubject,
    IdentityBasis,
    SubjectKind,
    SubjectSpecification,
    TargetAggregation,
)

HORIZON_NAME = "now"
HARVEST_SECONDS = 0.0
MODALITY = OntologyTerm(label="RNA-seq")
SPECIES = OntologyTerm(label="Homo sapiens")
SYSTEM = OntologyTerm(label="cultured human CD34-positive haematopoietic progenitor")

S6_NOMINAL_PROBABILITY = 0.90
"""The nominal probability S6's predictive intervals are scored at.

**It is read off the two thresholds below, not chosen.**  ``minimum_calibration_coverage=0.85`` and
``maximum_calibration_error=0.05`` were written into ``arm_query`` before any coverage number
existed.  Together they require the coverage to lie in ``[nominal - 0.05, nominal + 0.05]`` *and* to
clear 0.85, and the two constraints coincide at exactly one nominal:

    nominal - 0.05 >= 0.85   =>   nominal = 0.90

It lives here, beside the pair that forces it, rather than in the evaluation module that uses it --
partly because that is where its justification is, and partly because the other direction is a
**circular import**: ``evaluation`` measures ``backends``, so nothing under ``backends`` may import
``evaluation`` at module scope.  See ``test_every_module_imports_first``.

Fixing the nominal from an earlier declaration is the point.  This repository already carries two
thresholds a correct computation could not fail -- ``maximum_ood_score=0.99`` here, and
``maximum_calibration_error=1`` in ``examples/estimate_state.py`` -- and picking the nominal that
made the coverage pass would have quietly added a third.
"""

__all__ = [
    "HARVEST_SECONDS",
    "HORIZON_NAME",
    "S6_NOMINAL_PROBABILITY",
    "arm_history",
    "arm_query",
    "arm_request",
    "arm_subject",
]


def arm_subject(library: str, target: str) -> BeliefSubject:
    """The population of annotated cells assigned to ``target`` within ``library``."""

    return BeliefSubject(
        subject_id=f"gse274113:{library}:{target}",
        kind=SubjectKind.POPULATION,
        biological_system=SYSTEM,
        membership_semantics=(
            "the annotated cells that the deposited GSE274113 metadata assigns to this target's "
            "guide set within this 10x library; guide-calling error is absorbed into the subject "
            "definition and is not modelled"
        ),
        experimental_unit_kind="library",
        experimental_unit_id=library,
        identity_basis=IdentityBasis.DECLARED_MEMBERSHIP,
    )


def _intervention_spec(target: str) -> InterventionSpec:
    return InterventionSpec(
        spec_id=f"crispri-{target}",
        kind=OntologyTerm(label="CRISPRi guide"),
        target=OntologyTerm(label=target),
        mechanisms=(OntologyTerm(label="transcriptional repression"),),
        dose_domain=NumericDomain(minimum=0, maximum=1, units="guide_set"),
        duration_seconds=ScalarRange(minimum=0, maximum=1_209_600),
        schedule=ScheduleDomain(
            allowed_kinds=(ScheduleKind.SINGLE,),
            administration_count=IntegerRange(minimum=1, maximum=1),
            interval_seconds=None,
            washout_seconds=ScalarRange(minimum=0, maximum=0),
        ),
        delivery_methods=("lentiviral_transduction",),
        allowed_reversibility_statuses=(ReversibilityStatus.IRREVERSIBLE,),
        allowed_assignment_mechanisms=(AssignmentMechanism.RANDOMIZED,),
        assignment_unit_kind="cell",
        randomization_unit_kind="cell",
        require_randomization_unit=True,
        require_matched_control=False,
        realization_evidence=RealizationEvidenceRequirement(
            allowed_statuses=tuple(PerturbationStatus),
            allowed_modalities=(MODALITY,),
            minimum_evidence_events=0,
        ),
    )


def arm_query(targets: tuple[str, ...], *, model_fingerprint: str) -> StateQuery:
    """The frozen query every arm is answered under."""

    subject = SubjectSpecification(
        kind=SubjectKind.POPULATION,
        biological_system=SYSTEM,
        membership_semantics=arm_subject("rep1", "NT").membership_semantics,
        experimental_unit_kind="library",
        allowed_identity_bases=(IdentityBasis.DECLARED_MEMBERSHIP,),
    )
    return StateQuery(
        subject=subject,
        system_boundary=SystemBoundary.POPULATION,
        temporal_resolution_seconds=1,
        prediction_horizons=(
            PredictionHorizon(name=HORIZON_NAME, duration_seconds=1, timescale=Timescale.FAST),
        ),
        target_outputs=(
            OutputSpec(
                term=OntologyTerm(label="panel log composition"),
                units="log_fraction",
                aggregation=TargetAggregation(
                    subject_kind=SubjectKind.POPULATION,
                    statistic=AggregationStatistic.DISTRIBUTION,
                    experimental_unit="library",
                ),
                endpoint=LatentQuantityEndpoint(
                    model_reference=VersionedReference(
                        reference_id="gse274113-rna-obs",
                        version="1.0.0",
                        fingerprint=model_fingerprint,
                    )
                ),
                value_schema_reference=VersionedReference(
                    reference_id="gse274113-panel-log-composition",
                    version="1.0.0",
                    fingerprint=hashlib.sha256(b"gse274113-panel-log-composition").hexdigest(),
                ),
                missingness=TargetMissingnessSemantics(
                    policy=TargetMissingnessPolicy.NOT_APPLICABLE,
                    reportable_statuses=(),
                ),
                censoring=TargetCensoringSemantics(
                    policy=TargetCensoringPolicy.NOT_APPLICABLE,
                    allowed_directions=(),
                ),
                supported_horizon_names=(HORIZON_NAME,),
                weight=1,
                functional=False,
            ),
        ),
        intervention_space=tuple(_intervention_spec(target) for target in targets),
        available_assays=(),
        evidence_policy=EvidencePolicy(
            lookback_seconds=None,
            include_at_cutoff=True,
            allowed_modalities=(MODALITY,),
            allowed_evidence_roles=(EvidenceRole.DIRECT,),
            minimum_observed_measurements=1,
        ),
        acceptance_thresholds=AcceptanceThresholds(
            maximum_ood_score=0.99,
            maximum_history_information_gain=1,
            minimum_calibration_coverage=0.85,
            maximum_calibration_error=0.05,
            maximum_counterfactual_uncertainty=1,
            maximum_decision_uncertainty=1,
            minimum_identifiability=0.75,
        ),
        constraints=QueryConstraints(
            maximum_intervention_combination_order=1,
            require_complete_intervention_history=False,
            require_complete_environment_history=False,
            require_complete_lineage_history=False,
            require_complete_neighborhood_history=False,
            allow_transport=False,
            maximum_total_assay_cost=None,
            assay_cost_units=None,
            maximum_assay_delay_seconds=None,
        ),
    )


def arm_history(
    library: str,
    target: str,
    *,
    log_composition: tuple[float, ...],
    cells: int,
    panel_total: int,
) -> CellHistory:
    """One observation of the arm's panel, and the guide that defines it.

    Both events land at the harvest instant.  The observation is the arm's measured panel; the
    intervention is the guide set, recorded with ``UNKNOWN``
    realization.  Which guide a cell carries is deposited; how effectively the target was repressed
    is not, and this backend does not re-derive it.  ``INFERRED`` and ``MEASURED`` both require a
    realized efficiency, and supplying one we have not computed would be an invention.  The panel
    carries all nineteen target genes precisely so on-target knockdown *is* recoverable, which
    makes a measured efficiency a real follow-up rather than a permanent gap.
    """

    subject = arm_subject(library, target)
    observation = ObservationEvent(
        event_id=f"{library}:{target}:panel",
        subject=subject,
        time_seconds=HARVEST_SECONDS,
        duration_seconds=0,
        modality=MODALITY,
        evidence_link=EvidenceLink(
            source_subject=subject,
            target_subject=subject,
            role=EvidenceRole.DIRECT,
            linkage_basis=IdentityBasis.DECLARED_MEMBERSHIP,
            linkage_details="pseudobulk of the arm's own annotated cells",
            linkage_confidence=1,
            sampling_unit_id=library,
        ),
        collection=ObservationCollection(
            effect=CollectionEffect.PARTIALLY_DESTRUCTIVE_POPULATION_SAMPLING,
            # The subject IS the annotated cells, not the underlying culture, so the pseudobulk
            # observes all of it.  Declaring a fraction below one would assert a known relationship
            # to a culture size the deposit does not record; generalizing to that culture is a
            # transport claim, and the query sets allow_transport=False rather than making it.
            sampling_fraction=1.0,
        ),
        value=list(log_composition),
        units="log_fraction",
        missingness=MissingnessReport(),
        assay=AssayMetadata(
            assay_id="gse274113-rna-panel",
            attributes={"cells": cells, "panel_total": panel_total, "library": library},
        ),
        uncertainty=MeasurementUncertainty(),
        quality=QualityReport(),
    )
    intervention = InterventionEvent(
        event_id=f"{library}:{target}:guide",
        subject=subject,
        time_seconds=HARVEST_SECONDS,
        intervention_spec_id=f"crispri-{target}",
        intervention_type=OntologyTerm(label="CRISPRi guide"),
        target=OntologyTerm(label=target),
        mechanism=OntologyTerm(label="transcriptional repression"),
        dose=Quantity(value=1, units="guide_set"),
        duration_seconds=0,
        schedule=InterventionSchedule(
            kind=ScheduleKind.SINGLE, administration_count=1, washout_seconds=0
        ),
        delivery_method="lentiviral_transduction",
        reversibility_status=ReversibilityStatus.IRREVERSIBLE,
        estimated_efficiency=None,
        assignment_mechanism=AssignmentMechanism.RANDOMIZED,
        assignment_unit_kind="cell",
        assignment_unit_id=f"{library}:{target}",
        randomization_unit_kind="cell",
        randomization_unit_id=f"{library}:{target}",
        matched_control=None,
        actual_perturbation=ActualPerturbation(status=PerturbationStatus.UNKNOWN),
    )
    return CellHistory(
        subject=subject,
        events=(observation, intervention),
        completeness=HistoryCompleteness(
            observations=RecordCompleteness.COMPLETE,
            interventions=RecordCompleteness.COMPLETE,
            environments=RecordCompleteness.UNKNOWN,
            lineage=RecordCompleteness.UNKNOWN,
            neighborhood=RecordCompleteness.UNKNOWN,
        ),
    )


def arm_request(
    library: str,
    target: str,
    *,
    query: StateQuery,
    log_composition: tuple[float, ...],
    cells: int,
    panel_total: int,
) -> EstimateCellStateRequest:
    return EstimateCellStateRequest(
        query=query,
        history=arm_history(
            library,
            target,
            log_composition=log_composition,
            cells=cells,
            panel_total=panel_total,
        ),
        as_of_seconds=HARVEST_SECONDS,
        static_context=StaticContext(species=SPECIES),
    )
