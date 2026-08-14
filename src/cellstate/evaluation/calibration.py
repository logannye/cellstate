"""Backend-independent uncertainty calibration.

A predictive interval is calibrated when its nominal probability matches the rate at which it
actually contains the outcome.  The quantity that matters is the *absolute* deviation from nominal:
an interval that covers 99 percent of outcomes at a nominal 95 is as miscalibrated as one that
covers 91, and the first error is easier to hide because it looks conservative.

Capability S6 asks whether that deviation is within a predeclared threshold **as an upper
confidence bound**, grouped at the split unit.  So the harness reports three things, not one: the
empirical coverage, its absolute error against nominal, and a one-sided upper bound on that error
derived from the same grouped bootstrap the metrics use.  ``CalibrationReport`` gates its outcome
on the bound.  A point estimate inside the threshold with a bound outside it is not a pass.

Coverage is computed per independent experimental unit and then bootstrapped over the declared
dependence dimensions.  Pooling every outcome into one coverage fraction and bootstrapping the
outcomes would treat cells within a well as independent replicates, which they are not.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from cellstate.domain.belief import CalibrationReport, EvaluationStatus
from cellstate.domain.common import CriterionOutcome
from cellstate.evaluation.bootstrap import (
    FROZEN_SCIPLEX3_CONFIDENCE_LEVEL,
    FROZEN_SCIPLEX3_RESAMPLE_COUNT,
    multiway_clustered_bootstrap,
)


def empirical_interval_coverage(
    outcomes: Sequence[float], lower: Sequence[float], upper: Sequence[float]
) -> float:
    """The fraction of outcomes falling inside their interval, counted inclusively."""

    if not outcomes:
        raise ValueError("at least one outcome is required")
    if len(outcomes) != len(lower) or len(outcomes) != len(upper):
        raise ValueError("outcomes and interval bounds must have the same length")
    if any(low > high for low, high in zip(lower, upper, strict=True)):
        raise ValueError("lower interval bounds cannot exceed upper bounds")
    covered = sum(
        low <= value <= high for value, low, high in zip(outcomes, lower, upper, strict=True)
    )
    return covered / len(outcomes)


def evaluate_marginal_calibration(
    *,
    unit_outcomes: Sequence[Sequence[float]],
    unit_lower_bounds: Sequence[Sequence[float]],
    unit_upper_bounds: Sequence[Sequence[float]],
    cluster_labels: Mapping[str, Sequence[str]],
    nominal_probability: float,
    maximum_calibration_error: float,
    minimum_coverage: float,
    seed: int,
    metric: str,
    resample_count: int = FROZEN_SCIPLEX3_RESAMPLE_COUNT,
    confidence_level: float = FROZEN_SCIPLEX3_CONFIDENCE_LEVEL,
    maximum_degenerate_resamples: int = 0,
) -> CalibrationReport:
    """Report empirical coverage, its absolute error, and an upper bound on that error.

    ``unit_outcomes`` and the two bound sequences hold one entry per independent experimental unit;
    each entry is that unit's outcomes and per-outcome interval bounds.  Coverage is computed within
    a unit, and the unit-level coverages are bootstrapped over ``cluster_labels``.

    The upper bound on the absolute error is taken as the larger absolute deviation of the coverage
    interval's two endpoints from nominal.  That is the honest one-sided bound: it asks how far
    coverage could plausibly be from nominal in either direction, which is what an absolute error
    threshold is about.
    """

    if not 0.0 < nominal_probability < 1.0:
        raise ValueError("nominal probability must lie strictly between zero and one")
    if maximum_calibration_error < 0.0:
        raise ValueError("the calibration error threshold must be nonnegative")
    unit_count = len(unit_outcomes)
    if unit_count == 0:
        raise ValueError("at least one experimental unit is required")
    if len(unit_lower_bounds) != unit_count or len(unit_upper_bounds) != unit_count:
        raise ValueError("outcomes and interval bounds must cover the same experimental units")

    unit_coverages = [
        empirical_interval_coverage(outcomes, lower, upper)
        for outcomes, lower, upper in zip(
            unit_outcomes, unit_lower_bounds, unit_upper_bounds, strict=True
        )
    ]
    interval = multiway_clustered_bootstrap(
        values=unit_coverages,
        cluster_labels=cluster_labels,
        seed=seed,
        resample_count=resample_count,
        confidence_level=confidence_level,
        maximum_degenerate_resamples=maximum_degenerate_resamples,
    )
    coverage = float(np.clip(interval.point_estimate, 0.0, 1.0))
    error = abs(coverage - nominal_probability)
    upper_bound = max(
        abs(interval.lower - nominal_probability),
        abs(interval.upper - nominal_probability),
        error,
    )
    passed = coverage >= minimum_coverage and upper_bound <= maximum_calibration_error
    return CalibrationReport(
        evaluation_status=EvaluationStatus.EVALUATED,
        outcome=(CriterionOutcome.PASSED if passed else CriterionOutcome.FAILED),
        empirical_coverage=coverage,
        minimum_coverage=minimum_coverage,
        calibration_error=error,
        calibration_error_upper_bound=upper_bound,
        maximum_calibration_error=maximum_calibration_error,
        coverage_interval=interval,
        metric=metric,
        notes=(
            f"Coverage at nominal {nominal_probability:.2f} over {unit_count} experimental units, "
            f"grouped at {', '.join(interval.dependence_dimension_ids)}.",
            "The outcome gates on the upper confidence bound, not the point estimate.",
        ),
    )
