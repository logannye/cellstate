"""Operational tests for approximate predictive/Markov sufficiency.

A state is predictively sufficient for a query when knowing the raw history in addition to the
state does not improve prediction of the query's declared targets.  The test is a comparison of two
predictors:

``M1``
    predicts the future from the state alone.
``M2``
    predicts the future from the state and the raw pre-cutoff history.

The history information gain is ``loss(M1) - loss(M2)`` under a negatively oriented proper score.
Approximate sufficiency is supported when the **upper end of the gain's interval** falls at or
below the declared tolerance; otherwise the state is not shown to carry what the history does.
The gate is the interval, not the point estimate: a gain of zero measured so imprecisely that the
interval reaches far past the tolerance is an underpowered comparison, and reporting it as
sufficiency would let a weak experiment certify a state it never tested.  See ADR 0016.

Two properties make the comparison mean what it claims.

**Equal capacity.**  If ``M2`` is allowed more parameters than ``M1`` it will usually win, and the
gain will measure flexibility rather than information.  Every harness call declares the capacity of
both predictors and is refused if they differ.  :func:`fit_paired_ridge_losses` achieves equality by
construction: both predictors receive design matrices of identical shape and an identical estimator,
and ``M1`` receives a *permuted* history block where ``M2`` receives the real one.  Permutation
preserves the history's marginal distribution and its contribution to capacity while destroying its
association with the target, which is also what makes it a usable null.

**A sampling distribution.**  A gain reported without an interval grouped at the independent
experimental unit is not a verdict.  The gain is bootstrapped as a *paired* per-unit difference
through :func:`cellstate.evaluation.bootstrap.multiway_clustered_bootstrap`, which is strictly more
informative than differencing two independent means, and the interval is carried on the report.

Nothing here decides whether the underlying experiment can support the comparison at all.  A query
with no admissible pre-cutoff evidence makes ``M2`` identical to ``M1`` and the test inapplicable
rather than passed; that judgment belongs to the query and its benchmark, not to this module.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from cellstate.domain.belief import EvaluationStatus, SufficiencyReport
from cellstate.domain.common import BootstrapInterval, CriterionOutcome, require_finite
from cellstate.evaluation.bootstrap import (
    FROZEN_SCIPLEX3_CONFIDENCE_LEVEL,
    FROZEN_SCIPLEX3_RESAMPLE_COUNT,
    multiway_clustered_bootstrap,
)


@dataclass(frozen=True)
class PredictorCapacity:
    """The declared capacity of one member of a paired sufficiency comparison.

    Capacity is declared rather than inferred because the harness cannot inspect an arbitrary
    caller's estimator.  Declaring it wrongly is possible; declaring it is at least visible, and
    two predictors that declare different capacity are refused outright.
    """

    family: str
    parameter_count: int
    regularization: float
    fitting_procedure: str

    def __post_init__(self) -> None:
        if not self.family:
            raise ValueError("a predictor capacity must name its family")
        if self.parameter_count < 0:
            raise ValueError("parameter count must be nonnegative")
        if not math.isfinite(self.regularization) or self.regularization < 0:
            raise ValueError("regularization must be finite and nonnegative")
        if not self.fitting_procedure:
            raise ValueError("a predictor capacity must name its fitting procedure")


def evaluate_history_information_gain(
    *,
    state_only_loss: float,
    state_plus_history_loss: float,
    tolerance: float,
    metric: str,
    interval: BootstrapInterval,
    notes: tuple[str, ...] = (),
) -> SufficiencyReport:
    """Compare future prediction from state alone against state plus raw history.

    Lower loss is assumed to be better.  This builds the report; experimental design and held-out
    predictions determine whether the comparison is scientifically meaningful.  The interval is
    required: see ADR 0015.
    """

    require_finite(state_only_loss, name="state-only loss")
    require_finite(state_plus_history_loss, name="state-plus-history loss")
    require_finite(tolerance, name="sufficiency tolerance")
    if tolerance < 0:
        raise ValueError("sufficiency tolerance must be nonnegative")
    gain = state_only_loss - state_plus_history_loss
    # The verdict gates on the *upper end of the interval*, not on the point estimate.  A point
    # estimate inside the tolerance whose interval reaches well past it is not evidence of
    # sufficiency; it is an underpowered comparison.  This is the same argument ADR 0015 made for
    # calibration, applied to the test it was written alongside.  See ADR 0016.
    supported = interval.upper <= tolerance
    verdict = (
        "Approximate sufficiency supported: the interval's upper end is within the tolerance."
        if supported
        else "Raw history may materially improve prediction; the state is not shown to be complete."
    )
    separated = (
        "The interval excludes zero."
        if interval.excludes_zero
        else "The interval does not exclude zero."
    )
    return SufficiencyReport(
        evaluation_status=EvaluationStatus.EVALUATED,
        outcome=(CriterionOutcome.PASSED if supported else CriterionOutcome.FAILED),
        state_only_loss=state_only_loss,
        state_plus_history_loss=state_plus_history_loss,
        history_information_gain=gain,
        history_information_gain_interval=interval,
        markov_sufficiency_score=math.exp(-max(gain, 0.0)),
        maximum_history_information_gain=tolerance,
        metric=metric,
        notes=(verdict, separated, *notes),
    )


def evaluate_predictive_sufficiency(
    *,
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
    maximum_degenerate_resamples: int = 0,
) -> SufficiencyReport:
    """Run the paired sufficiency comparison and report the gain with its interval.

    ``state_only_losses`` and ``state_plus_history_losses`` hold one held-out loss per independent
    experimental unit, in the same order, from predictors whose declared capacity is equal.
    ``cluster_labels`` maps each dependence dimension to that unit's cluster label, and the gain is
    bootstrapped as a paired per-unit difference grouped over those dimensions.
    """

    if state_only_capacity != state_plus_history_capacity:
        raise ValueError(
            "the paired predictors declare different capacity, so any gain would measure "
            f"flexibility rather than history information: {state_only_capacity} versus "
            f"{state_plus_history_capacity}"
        )
    state_only = np.asarray(state_only_losses, dtype=np.float64)
    state_plus_history = np.asarray(state_plus_history_losses, dtype=np.float64)
    if state_only.shape != state_plus_history.shape:
        raise ValueError("the paired loss vectors must cover the same experimental units")
    if state_only.ndim != 1 or state_only.size == 0:
        raise ValueError("losses must be one-dimensional with at least one experimental unit")
    if not np.all(np.isfinite(state_only)) or not np.all(np.isfinite(state_plus_history)):
        raise ValueError("every held-out loss must be finite")

    interval = multiway_clustered_bootstrap(
        values=state_only - state_plus_history,
        cluster_labels=cluster_labels,
        seed=seed,
        resample_count=resample_count,
        confidence_level=confidence_level,
        maximum_degenerate_resamples=maximum_degenerate_resamples,
    )
    return evaluate_history_information_gain(
        state_only_loss=float(state_only.mean()),
        state_plus_history_loss=float(state_plus_history.mean()),
        tolerance=tolerance,
        metric=metric,
        interval=interval,
        notes=(
            f"Paired over {state_only.size} experimental units with equal declared capacity "
            f"({state_only_capacity.family}, {state_only_capacity.parameter_count} parameters).",
        ),
    )


def _ridge_fit(design: NDArray[np.float64], targets: NDArray[np.float64], penalty: float) -> Any:
    """Ridge coefficients with an unpenalized intercept, solved in closed form."""

    centered_design = design - design.mean(axis=0)
    centered_targets = targets - targets.mean(axis=0)
    gram = centered_design.T @ centered_design + penalty * np.eye(design.shape[1])
    coefficients = np.linalg.solve(gram, centered_design.T @ centered_targets)
    intercept = targets.mean(axis=0) - design.mean(axis=0) @ coefficients
    return coefficients, intercept


def fit_paired_ridge_losses(
    *,
    state_features: NDArray[np.float64],
    history_features: NDArray[np.float64],
    targets: NDArray[np.float64],
    train_mask: NDArray[np.bool_],
    seed: int,
    penalty: float = 1.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64], PredictorCapacity]:
    """Fit a capacity-matched ``M1``/``M2`` pair and return their held-out per-unit losses.

    ``M2`` is fitted on ``[state, history]``.  ``M1`` is fitted on ``[state, permuted history]``,
    where the permutation is applied across units within the training split and again across the
    held-out split.  The two designs have identical shape, identical penalty, and identical solver,
    so their capacity is equal by construction rather than by assertion; only the information in the
    history block differs.

    Returns the held-out squared-error loss per unit for ``M1`` and for ``M2``, and the capacity
    both declare.  The loss is per unit so the caller can bootstrap the paired difference.
    """

    # Written as two explicit comparisons: the chained form ``a != b != c`` is ``a != b and
    # b != c``, which never compares ``a`` with ``c`` and so passes ``(10, 10, 8)`` silently.
    if (
        state_features.shape[0] != history_features.shape[0]
        or state_features.shape[0] != targets.shape[0]
    ):
        raise ValueError("state, history, and targets must cover the same units")
    if train_mask.shape[0] != targets.shape[0]:
        raise ValueError("the training mask must cover the same units")
    if not train_mask.any() or train_mask.all():
        raise ValueError("the split must hold out at least one unit and train on at least one")

    generator = np.random.Generator(np.random.PCG64DXSM(seed))
    permuted_history = np.empty_like(history_features)
    for mask in (train_mask, ~train_mask):
        indices = np.flatnonzero(mask)
        permuted_history[indices] = history_features[generator.permutation(indices)]

    informative = np.hstack([state_features, history_features])
    uninformative = np.hstack([state_features, permuted_history])
    capacity = PredictorCapacity(
        family="ridge",
        parameter_count=int(informative.shape[1]) + 1,
        regularization=float(penalty),
        fitting_procedure="closed-form ridge with unpenalized intercept",
    )

    losses: list[NDArray[np.float64]] = []
    for design in (uninformative, informative):
        coefficients, intercept = _ridge_fit(
            design[train_mask], targets[train_mask], penalty=penalty
        )
        residuals = targets[~train_mask] - (design[~train_mask] @ coefficients + intercept)
        losses.append(np.asarray((residuals**2).mean(axis=1), dtype=np.float64))
    return losses[0], losses[1], capacity
