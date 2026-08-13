"""The applicability judgment the sufficiency harness delegates, and its callers.

:mod:`cellstate.evaluation.sufficiency` states that a query with no admissible pre-cutoff evidence
makes ``M2`` identical to ``M1``, and that such a design is *inapplicable rather than passed* --
then delegates that judgment to "the query and its benchmark".  Until this module existed there was
no delegate: no file outside the tests called the harness at all, so nothing performed the
judgment, and the harness's own answer on such a design -- ``PASSED``, with a degenerate interval
at zero -- stood unchallenged.  This module is the delegate.  See ADR 0017.

The judgment is **computed from the request**, not declared alongside it.  A request already
carries a typed history and an as-of time, so "does this unit carry an admissible observation
before the inference cutoff" is a fact the domain can answer, and answering it here means the
population a guard applies to is derived from the evidence rather than hand-written next to it.
That is also exactly the ledger's S1 predicate, applied per unit.

An all-zero feature block is *not* the predicate.  In a sparse assay a measured zero is a
legitimate observation -- the frozen sci-Plex3 partition contains records whose entire 2,000-feature
panel total is zero -- so inferring absence from the numbers would silently reclassify measured data
as missing.  Absence is a property of the experiment and is read from the event history.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from cellstate.domain.belief import CellStateBelief, SufficiencyReport
from cellstate.domain.events import MissingnessStatus, ObservationEvent
from cellstate.domain.request import EstimateCellStateRequest
from cellstate.evaluation.bootstrap import (
    FROZEN_SCIPLEX3_CONFIDENCE_LEVEL,
    FROZEN_SCIPLEX3_RESAMPLE_COUNT,
)
from cellstate.evaluation.sufficiency import (
    PredictorCapacity,
    evaluate_predictive_sufficiency,
    inapplicable_sufficiency_report,
)


def admissible_pre_cutoff_observations(
    request: EstimateCellStateRequest,
) -> tuple[ObservationEvent, ...]:
    """Return the observations this request offers strictly as history.

    An observation qualifies when it completed at or before the inference cutoff and its
    missingness status is ``OBSERVED``.  Indirect evidence counts: ``M2`` is entitled to use any
    measurement the query admits before the cutoff, and excluding indirect evidence here would
    understate the history and bias the gain downward, toward sufficiency.
    """

    return tuple(
        event
        for event in request.history.through(request.as_of_seconds)
        if isinstance(event, ObservationEvent)
        and event.missingness.status is MissingnessStatus.OBSERVED
    )


def request_carries_pre_cutoff_evidence(request: EstimateCellStateRequest) -> bool:
    """Whether this request admits the sufficiency comparison at all -- S1, per unit."""

    return bool(admissible_pre_cutoff_observations(request))


def history_presence_for_cohort(
    requests: Sequence[EstimateCellStateRequest],
) -> tuple[bool, ...]:
    """Compute the per-unit history-presence declaration the harness requires.

    Ordered to match ``requests``, which must be ordered to match the loss vectors.  Computing this
    rather than writing it down is the point: a hand-maintained mask drifts from the evidence it
    claims to describe, and it drifts in the direction that makes the test easier to pass.
    """

    if not requests:
        raise ValueError("a cohort must contain at least one request")
    return tuple(request_carries_pre_cutoff_evidence(request) for request in requests)


def evaluate_cohort_sufficiency(
    *,
    requests: Sequence[EstimateCellStateRequest],
    state_only_losses: Sequence[float],
    state_plus_history_losses: Sequence[float],
    cluster_labels: Mapping[str, Sequence[str]],
    tolerance: float,
    metric: str,
    seed: int,
    state_only_capacity: PredictorCapacity,
    state_plus_history_capacity: PredictorCapacity,
    resample_count: int = FROZEN_SCIPLEX3_RESAMPLE_COUNT,
    confidence_level: float = FROZEN_SCIPLEX3_CONFIDENCE_LEVEL,
) -> SufficiencyReport:
    """Score a cohort of units, deriving history presence from the requests themselves.

    This is the harness's caller.  The sufficiency verdict is a population quantity -- it needs an
    interval grouped at the independent experimental unit, and one unit cannot supply one -- so the
    cohort, not the single request, is the level at which the question can be asked.
    """

    if len(requests) != len(state_only_losses):
        raise ValueError("one request per unit, ordered to match the loss vectors")
    return evaluate_predictive_sufficiency(
        state_only_losses=state_only_losses,
        state_plus_history_losses=state_plus_history_losses,
        history_present=history_presence_for_cohort(requests),
        cluster_labels=cluster_labels,
        tolerance=tolerance,
        metric=metric,
        seed=seed,
        state_only_capacity=state_only_capacity,
        state_plus_history_capacity=state_plus_history_capacity,
        resample_count=resample_count,
        confidence_level=confidence_level,
    )


@dataclass(frozen=True)
class QuerySufficiencyEvaluator:
    """A :class:`cellstate.ports.SufficiencyEvaluator` that refuses, and says which refusal it is.

    The port's signature is per-request, and a sufficiency verdict is not a per-request quantity: it
    requires a bootstrap interval grouped at the independent experimental unit, and a single unit
    supplies neither the units nor the clusters.  So this evaluator never returns ``PASSED`` or
    ``FAILED``, and that is not a stub -- it is the honest range of the signature.  What it does
    carry is the distinction between the two reasons a verdict is unavailable, which is the
    difference between a query that *cannot* be asked and one that has not *yet* been asked over a
    cohort.  Returning ``PASSED`` for either, as the harness previously did for the first, is the
    defect ADR 0017 closes.
    """

    tolerance: float

    def __post_init__(self) -> None:
        if not self.tolerance >= 0.0:
            raise ValueError("sufficiency tolerance must be nonnegative")

    def evaluate(
        self,
        belief: CellStateBelief,
        request: EstimateCellStateRequest,
    ) -> SufficiencyReport:
        del belief  # The refusal is a property of the query's evidence, not of the belief.
        if not request_carries_pre_cutoff_evidence(request):
            return inapplicable_sufficiency_report(
                reason=(
                    "The query admits no observed measurement at or before its inference cutoff, "
                    "so the state-plus-history predictor has no history to use. The comparison is "
                    "inapplicable; it is not evidence that the state is sufficient."
                ),
                tolerance=self.tolerance,
            )
        return inapplicable_sufficiency_report(
            reason=(
                "The query admits pre-cutoff evidence, but a sufficiency verdict requires a "
                "bootstrap interval grouped at the independent experimental unit and a single "
                "request supplies one unit. Score the cohort through "
                "evaluate_cohort_sufficiency."
            ),
            tolerance=self.tolerance,
        )
