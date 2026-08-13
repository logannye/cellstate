"""Multiway clustered bootstrap for intervals grouped at independent experimental units.

A score reported without a sampling distribution is not a verdict.  Every metric in the frozen
sci-Plex3 suite binds this estimator, and the predictive-sufficiency and calibration harnesses
take their intervals from it rather than reimplementing one.

The resampling scheme is the multiway (pigeonhole) bootstrap for data arrays: each dependence
dimension is resampled independently with replacement over its own clusters, and an evaluation
unit's weight is the product of the draw counts of the clusters it belongs to.  Resampling the
dimensions jointly this way is what keeps two units that share a compound, or share a plate, from
being treated as independent.  Cells are never resampling units.

The frozen semantics are:

* dimensions are resampled independently; a unit's weight is the product over dimensions of its
  cluster's draw count, so a unit survives a resample only if every one of its clusters was drawn;
* each dimension's draw counts are multinomial over its own clusters with equal probability, so
  the expected weight of every unit is one;
* endpoints are the equal-tailed percentiles of the resampled statistic, taken with linear
  interpolation between order statistics, then scaled away from the point estimate by the
  small-cluster factor described below;
* a resample whose total weight is zero leaves the statistic undefined.  Such resamples are
  counted and are fatal by default; they are never silently discarded, because discarding them
  conditions the interval on the resamples that happened to be well behaved.

**Why the endpoints are scaled.**  The multinomial pigeonhole bootstrap understates variance when
a dimension has few clusters, for two separate reasons, and :func:`small_cluster_scale` corrects
both.  First, a cluster's draw count has variance ``1 - 1/K`` rather than one, so the resampled
spread is biased low by ``sqrt((K - 1) / K)`` and the bias does not vanish as the number of
resamples grows.  Second, a statistic formed from ``K`` clusters belongs against a Student ``t``
reference with ``K - 1`` degrees of freedom rather than a normal one.  Each endpoint's distance
from the point estimate is therefore scaled by
``sqrt(K / (K - 1)) * t(K - 1, 1 - alpha/2) / z(1 - alpha/2)`` for ``K = min(K_d)``.  Both factors
decay to one as cluster counts grow and inflate sharply when they are small, which is the honest
behavior: two clusters cannot support a 95 percent interval, and the interval should say so by
being wide rather than by being precise and wrong.

The rule takes the smallest cluster count across dimensions and so is deliberately conservative:
when the sparse dimension contributes little of the variance the interval is wider than it needs
to be.  That is the direction to err in, since the predicate a superiority claim rests on is
whether the interval excludes zero.  Both raw percentile endpoints are reported alongside the
scaled ones so the size of the correction is auditable and never silent.

**Measured coverage, and its limit.**  Simulated two-way random-effects designs on the *real*
incidence of the sci-Plex3 untouched-test partition — 384 wells, 4 plates, 95 compound labels, 94
of which sit on exactly two plates — at nominal 0.95, 600 replications, 400 resamples:

===================================  ==============  ====================
variance regime                      ``t/z`` only    with ``sqrt(K/(K-1))``
===================================  ==============  ====================
plate-dominated (sd 2.0 / 0.2)       0.908 +- 0.012  0.935 +- 0.010
plate-only      (sd 1.0 / 0.0)       0.912 +- 0.012  0.943 +- 0.009
balanced        (sd 1.0 / 1.0)       0.940 +- 0.010  0.960 +- 0.008
compound-dominated (sd 0.2 / 2.0)    0.998 +- 0.002  1.000
plate-free      (sd 0.0 / 1.0)       1.000           1.000
===================================  ==============  ====================

ADR 0016 added the first column's missing factor because ``t/z`` alone under-covers by a margin
that excludes nominal whenever the four-plate dimension carries the variance.  **Even corrected,
coverage at ``K = 4`` in a plate-dominated regime is about 0.94, not 0.95.**  That residual is
reported rather than tuned away: four clusters do not support a 95 percent interval, and a claim
resting on one should be read with that in mind.  It is the property Phase 2 requires a
state-bearing estimand to *not* have.

Nothing in this module decides whether a comparison is scientifically meaningful.  It reports the
sampling distribution of a statistic under the declared dependence structure and nothing else.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Final, Literal

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from cellstate.domain.common import BootstrapInterval

#: Bumped to 2.0.0 by ADR 0016: ``small_cluster_scale`` gained the ``sqrt(K/(K-1))`` variance
#: correction, which changes every reported endpoint at a fixed seed.  Endpoints produced by
#: 1.0.0 are not comparable to endpoints produced by this version.
BOOTSTRAP_IMPLEMENTATION_VERSION: Final = "2.0.0"
RNG_ALGORITHM: Final[Literal["numpy-pcg64dxsm-v1"]] = "numpy-pcg64dxsm-v1"
RESAMPLING_SCHEME: Final[Literal["multiway_clustered"]] = "multiway_clustered"
INTERVAL_METHOD: Final[Literal["equal_tailed_percentile_small_cluster_scaled"]] = (
    "equal_tailed_percentile_small_cluster_scaled"
)

#: Chunk size for the vectorized weight matrix.  Bounds peak memory at
#: ``_RESAMPLE_CHUNK * evaluation_unit_count`` float64 entries.
#:
#: This is part of the numeric contract, not a free tuning knob: the generator is consumed one
#: chunk and one dimension at a time, so changing it changes the draw sequence and therefore every
#: reported endpoint at a fixed seed.  Changing it is a version bump.
_RESAMPLE_CHUNK: Final = 256

#: The configuration the frozen sci-Plex3 metric suite declares for every one of its metrics.
FROZEN_SCIPLEX3_RESAMPLE_COUNT: Final = 2_000
FROZEN_SCIPLEX3_CONFIDENCE_LEVEL: Final = 0.95
FROZEN_SCIPLEX3_DEPENDENCE_DIMENSIONS: Final = ("compound", "plate")

Statistic = Callable[[NDArray[np.float64], NDArray[np.float64]], float]

__all__ = [
    "BOOTSTRAP_IMPLEMENTATION_VERSION",
    "FROZEN_SCIPLEX3_CONFIDENCE_LEVEL",
    "FROZEN_SCIPLEX3_DEPENDENCE_DIMENSIONS",
    "FROZEN_SCIPLEX3_RESAMPLE_COUNT",
    "INTERVAL_METHOD",
    "RESAMPLING_SCHEME",
    "RNG_ALGORITHM",
    "BootstrapInterval",
    "multiway_clustered_bootstrap",
    "small_cluster_scale",
    "weighted_mean",
]


def small_cluster_scale(*, minimum_cluster_count: int, confidence_level: float) -> float:
    """The factor by which each endpoint's distance from the point estimate is inflated.

    Two corrections, for the two distinct defects of the multinomial pigeonhole bootstrap at
    small ``K``, where ``K`` is the smallest cluster count across dependence dimensions:

    ``sqrt(K / (K - 1))``
        the *variance deficiency*.  A cluster's multinomial draw count has variance ``1 - 1/K``
        rather than one, so the resampled spread is biased low by that factor regardless of how
        many resamples are drawn.  This term removes the bias at its source.
    ``t(K - 1, 1 - alpha/2) / z(1 - alpha/2)``
        the *reference-distribution* correction, following the multiway-clustering convention of
        referring a small-``K`` statistic to Student ``t`` rather than to the normal.

    Both decay to one as ``K`` grows.  Before ADR 0016 only the second was applied, and the
    resulting interval under-covered: see the measured table in the module docstring.
    """

    if minimum_cluster_count < 2:
        raise ValueError("a bootstrap requires at least two clusters in every dimension")
    upper_tail = 0.5 + confidence_level / 2.0
    normal_quantile = float(stats.norm.ppf(upper_tail))
    student_quantile = float(stats.t.ppf(upper_tail, minimum_cluster_count - 1))
    if not math.isfinite(student_quantile) or normal_quantile <= 0.0:
        raise ValueError("the small-cluster scale is undefined at this confidence level")
    variance_deficiency = math.sqrt(
        minimum_cluster_count / (minimum_cluster_count - 1),
    )
    return variance_deficiency * student_quantile / normal_quantile


def weighted_mean(values: NDArray[np.float64], weights: NDArray[np.float64]) -> float:
    """The weight-normalized mean, the statistic behind every mean-aggregated metric."""

    total = float(weights.sum())
    if total <= 0.0:
        raise ZeroDivisionError("a weighted mean requires positive total weight")
    return float(np.dot(values, weights) / total)


def multiway_clustered_bootstrap(
    *,
    values: Sequence[float] | NDArray[np.float64],
    cluster_labels: Mapping[str, Sequence[str]],
    seed: int,
    statistic: Statistic = weighted_mean,
    resample_count: int = FROZEN_SCIPLEX3_RESAMPLE_COUNT,
    confidence_level: float = FROZEN_SCIPLEX3_CONFIDENCE_LEVEL,
    maximum_degenerate_resamples: int = 0,
) -> BootstrapInterval:
    """Bootstrap ``statistic`` over evaluation units clustered in one or more dimensions.

    ``values`` holds one contribution per evaluation unit.  ``cluster_labels`` maps each
    dependence dimension to that unit's cluster label in the same order, so
    ``cluster_labels["plate"][i]`` is the plate of unit ``i``.  Every dimension is resampled;
    passing a single dimension is a one-way cluster bootstrap and is permitted, but the frozen
    sci-Plex3 suite declares two.

    ``statistic`` receives the unweighted values and the resample weights.  The default is the
    weighted mean.  A statistic that groups internally — an equal-weight mean over compounds, for
    instance — closes over its own grouping and must honor the weights it is given, or the
    interval will describe a different estimand than the point estimate.
    """

    unit_values = np.asarray(values, dtype=np.float64)
    if unit_values.ndim != 1:
        raise ValueError("values must be one-dimensional, one entry per evaluation unit")
    unit_count = int(unit_values.shape[0])
    if unit_count == 0:
        raise ValueError("at least one evaluation unit is required")
    if not np.all(np.isfinite(unit_values)):
        raise ValueError("every evaluation unit value must be finite")
    if not cluster_labels:
        raise ValueError("at least one dependence dimension is required")
    if resample_count < 2:
        raise ValueError("at least two resamples are required to form an interval")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence level must lie strictly between zero and one")
    if maximum_degenerate_resamples < 0:
        raise ValueError("the degenerate-resample allowance must be nonnegative")

    dimension_ids = tuple(sorted(cluster_labels))
    membership: list[NDArray[np.int64]] = []
    cluster_counts: list[int] = []
    for dimension_id in dimension_ids:
        labels = tuple(cluster_labels[dimension_id])
        if len(labels) != unit_count:
            raise ValueError(
                f"dimension {dimension_id!r} has {len(labels)} labels "
                f"for {unit_count} evaluation units"
            )
        ordinals: dict[str, int] = {}
        for label in labels:
            ordinals.setdefault(label, len(ordinals))
        membership.append(np.fromiter((ordinals[label] for label in labels), dtype=np.int64))
        cluster_counts.append(len(ordinals))

    scale = small_cluster_scale(
        minimum_cluster_count=min(cluster_counts), confidence_level=confidence_level
    )
    point_estimate = float(statistic(unit_values, np.ones(unit_count, dtype=np.float64)))
    if not math.isfinite(point_estimate):
        raise ValueError("the statistic must be finite on the observed sample")

    generator = np.random.Generator(np.random.PCG64DXSM(seed))
    resampled = np.empty(resample_count, dtype=np.float64)
    degenerate = 0
    completed = 0
    while completed < resample_count:
        chunk = min(_RESAMPLE_CHUNK, resample_count - completed)
        weights = np.ones((chunk, unit_count), dtype=np.float64)
        for indices, count in zip(membership, cluster_counts, strict=True):
            draws = generator.multinomial(count, np.full(count, 1.0 / count), size=chunk)
            weights *= draws[:, indices]
        for offset in range(chunk):
            row = weights[offset]
            if row.sum() <= 0.0:
                degenerate += 1
                resampled[completed + offset] = np.nan
                continue
            resampled[completed + offset] = float(statistic(unit_values, row))
        completed += chunk

    if degenerate > maximum_degenerate_resamples:
        raise ValueError(
            f"{degenerate} of {resample_count} resamples had zero total weight, above the "
            f"allowance of {maximum_degenerate_resamples}; the interval would be conditioned "
            f"on the resamples that happened to retain a unit"
        )
    usable = resampled[np.isfinite(resampled)]
    if usable.size < 2:
        raise ValueError("fewer than two usable resamples; no interval can be formed")

    tail = (1.0 - confidence_level) / 2.0
    percentile_lower, percentile_upper = (
        float(bound) for bound in np.quantile(usable, (tail, 1.0 - tail), method="linear")
    )
    return BootstrapInterval(
        point_estimate=point_estimate,
        lower=point_estimate - scale * (point_estimate - percentile_lower),
        upper=point_estimate + scale * (percentile_upper - point_estimate),
        percentile_lower=percentile_lower,
        percentile_upper=percentile_upper,
        small_cluster_scale=scale,
        standard_error=float(usable.std(ddof=1)),
        confidence_level=confidence_level,
        resample_count=resample_count,
        evaluation_unit_count=unit_count,
        resampling_scheme=RESAMPLING_SCHEME,
        interval_method=INTERVAL_METHOD,
        dependence_dimension_ids=dimension_ids,
        cluster_counts=tuple(cluster_counts),
        degenerate_resample_count=degenerate,
        seed=seed,
        rng_algorithm=RNG_ALGORITHM,
        implementation_version=BOOTSTRAP_IMPLEMENTATION_VERSION,
    )
