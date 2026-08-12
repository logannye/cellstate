"""Executable implementations of the metrics the frozen sci-Plex3 suite declares.

These are backend-independent array computations.  They take observations and predictive samples
that have already been put on the scoring scale by the caller, and they return one number per
evaluation unit.  Aggregation across units, and the interval around that aggregate, belong to
:mod:`cellstate.evaluation.bootstrap`; nothing here aggregates across wells.

The frozen suite declares ten ``metric_id`` entries backed by six distinct computations, two of
which are parameterized by nominal probability.  :data:`METRIC_IMPLEMENTATIONS` binds every
declared identifier to the callable that computes it, and
``tests/test_q1_metric_suite_conformance.py`` fails on any identifier that does not resolve.

Numerical contracts, fixed here and not to be varied silently:

* the sample CRPS uses the unbiased pairwise term ``2 / (m (m - 1)) * sum_{i<j} |x_i - x_j|``.
  The biased ``1 / m^2`` form rewards under-dispersion, which is the opposite of what a
  posterior is asked to earn, and it is not implemented;
* the energy score uses the same unbiased pairwise term with Euclidean norms;
* the pairwise term is computed from sorted order statistics in the univariate case, which is
  exact and avoids the quadratic form;
* central predictive intervals are taken as equal-tailed sample quantiles with linear
  interpolation between order statistics;
* an evaluation unit with no observation is an error.  No metric drops, imputes, or defaults a
  missing target, in keeping with the frozen suite's missingness rule.

Every function is negatively oriented except where its name says otherwise: lower is better, so a
difference of two scores is a gain when it is positive.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Final, Literal

import numpy as np
from numpy.typing import NDArray
from scipy import stats
from scipy.spatial.distance import cdist, pdist

METRICS_IMPLEMENTATION_VERSION: Final = "1.0.0"
QUANTILE_METHOD: Final = "linear"
PAIRWISE_TERM: Final = "unbiased_m_times_m_minus_one"

#: Block size for the energy score's pairwise sum.  The energy score is irreducibly quadratic in
#: the sample count, but its *memory* need not be: summing in blocks bounds the largest allocation
#: at ``_PAIRWISE_BLOCK ** 2`` float64 entries instead of ``m ** 2 / 2``.  At ten thousand samples
#: the unblocked form asks for 400 MB and at a hundred thousand it asks for 37 GB, which is a
#: crash rather than a slow answer.  Blocking changes only the summation order.
_PAIRWISE_BLOCK: Final = 2_048

MetricDirection = Literal["minimize", "maximize"]


def _as_observation_matrix(observations: Any, *, name: str) -> NDArray[np.float64]:
    """Coerce observations to ``(observation_count, feature_count)`` and validate them."""

    matrix = np.atleast_2d(np.asarray(observations, dtype=np.float64))
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be one- or two-dimensional")
    if matrix.shape[0] == 0:
        raise ValueError(f"{name} requires at least one observation; a missing target is an error")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must be finite")
    return matrix


def _as_sample_matrix(samples: Any, *, name: str, minimum: int = 2) -> NDArray[np.float64]:
    matrix = np.atleast_2d(np.asarray(samples, dtype=np.float64))
    if matrix.shape[0] < minimum:
        raise ValueError(f"{name} requires at least {minimum} predictive samples")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must be finite")
    return matrix


def _sorted_pairwise_absolute_sums(samples: NDArray[np.float64]) -> NDArray[np.float64]:
    """Per feature, ``sum_{i<j} |x_i - x_j|`` from order statistics.

    Sorting each column and weighting the order statistics by ``2 k - m + 1`` gives the pairwise
    sum exactly in ``O(m log m)`` per feature, rather than forming the ``m by m`` differences.
    """

    ordered = np.sort(samples, axis=0)
    count = ordered.shape[0]
    coefficients = 2.0 * np.arange(count, dtype=np.float64) - count + 1.0
    return np.asarray(coefficients @ ordered, dtype=np.float64)


def _blocked_pairwise_euclidean_sum(samples: NDArray[np.float64]) -> float:
    """``sum_{i<j} ||x_i - x_j||_2``, accumulated in blocks so memory stays bounded.

    Diagonal blocks contribute their own upper triangle through ``pdist``; off-diagonal blocks
    contribute every pair exactly once through ``cdist``.  The largest allocation is one block
    against one block, never the full pairwise array.
    """

    count = samples.shape[0]
    total = 0.0
    for row_start in range(0, count, _PAIRWISE_BLOCK):
        row_stop = min(row_start + _PAIRWISE_BLOCK, count)
        rows = samples[row_start:row_stop]
        if row_stop - row_start > 1:
            total += float(pdist(rows, metric="euclidean").sum())
        for column_start in range(row_stop, count, _PAIRWISE_BLOCK):
            column_stop = min(column_start + _PAIRWISE_BLOCK, count)
            total += float(cdist(rows, samples[column_start:column_stop], metric="euclidean").sum())
    return total


def sample_crps(observations: Any, samples: Any) -> float:
    """Continuous ranked probability score, averaged over observations and features.

    ``E|X - y| - 0.5 E|X - X'|`` with ``X``, ``X'`` independent draws from the predictive
    distribution.  Negatively oriented.
    """

    observed = _as_observation_matrix(observations, name="CRPS observations")
    drawn = _as_sample_matrix(samples, name="CRPS samples")
    if observed.shape[1] != drawn.shape[1]:
        raise ValueError("observations and samples must share a feature axis")
    sample_count = drawn.shape[0]

    # Accumulate over observations rather than broadcasting to (samples, observations, features);
    # at benchmark scale that intermediate is hundreds of megabytes and this one is not.
    absolute_error = np.zeros(drawn.shape[1], dtype=np.float64)
    for observation in observed:
        absolute_error += np.abs(drawn - observation).mean(axis=0)
    absolute_error /= observed.shape[0]

    pairwise = _sorted_pairwise_absolute_sums(drawn)
    spread = 2.0 * pairwise / (sample_count * (sample_count - 1))
    return float(np.mean(absolute_error - 0.5 * spread))


def energy_score(observations: Any, samples: Any) -> float:
    """Multivariate energy score ``E||X - y||_2 - 0.5 E||X - X'||_2``, averaged over observations.

    The joint counterpart of the CRPS: it is sensitive to dependence between features, which the
    marginal score is not.  Negatively oriented.

    Cost is quadratic in the sample count and cannot be otherwise — the pairwise term is a sum
    over every pair, with no order-statistic shortcut once the norm is multivariate.  Memory is
    bounded by blocking (see :data:`_PAIRWISE_BLOCK`), so a large sample count buys a slow answer
    rather than an allocation failure, but the caller still chooses the sample count knowingly.
    """

    observed = _as_observation_matrix(observations, name="energy-score observations")
    drawn = _as_sample_matrix(samples, name="energy-score samples")
    if observed.shape[1] != drawn.shape[1]:
        raise ValueError("observations and samples must share a feature axis")
    sample_count = drawn.shape[0]

    distance_to_observations = float(cdist(drawn, observed, metric="euclidean").mean())
    pairwise_total = _blocked_pairwise_euclidean_sum(drawn)
    spread = 2.0 * pairwise_total / (sample_count * (sample_count - 1))
    return float(distance_to_observations - 0.5 * spread)


def central_interval(
    samples: Any, *, probability: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Equal-tailed central predictive interval per feature, at the given nominal probability."""

    if not 0.0 < probability < 1.0:
        raise ValueError("nominal probability must lie strictly between zero and one")
    drawn = _as_sample_matrix(samples, name="interval samples")
    tail = (1.0 - probability) / 2.0
    lower, upper = np.quantile(drawn, (tail, 1.0 - tail), axis=0, method=QUANTILE_METHOD)
    return np.asarray(lower, dtype=np.float64), np.asarray(upper, dtype=np.float64)


def marginal_interval_coverage(observations: Any, lower: Any, upper: Any) -> float:
    """The fraction of observed values falling inside their per-feature interval."""

    observed = _as_observation_matrix(observations, name="coverage observations")
    low = np.asarray(lower, dtype=np.float64)
    high = np.asarray(upper, dtype=np.float64)
    if low.shape != high.shape:
        raise ValueError("interval bounds must have the same shape")
    if low.shape[-1] != observed.shape[1]:
        raise ValueError("interval bounds and observations must share a feature axis")
    if np.any(low > high):
        raise ValueError("lower interval bounds cannot exceed upper bounds")
    inside = (observed >= low) & (observed <= high)
    return float(inside.mean())


def marginal_coverage_error(
    observations: Any, lower: Any, upper: Any, *, nominal_probability: float
) -> float:
    """Absolute distance between empirical marginal coverage and its nominal probability.

    Negatively oriented, and reported against a threshold as an upper confidence bound rather
    than as a point value: coverage that is too high is a defect as surely as coverage that is
    too low, which is why the absolute value is taken here and not by the caller.
    """

    if not 0.0 < nominal_probability < 1.0:
        raise ValueError("nominal probability must lie strictly between zero and one")
    return abs(marginal_interval_coverage(observations, lower, upper) - nominal_probability)


def marginal_interval_width(lower: Any, upper: Any) -> float:
    """Mean per-feature width of a predictive interval.

    Width alone is not a score.  It is reported beside coverage because an interval can reach
    nominal coverage by being uninformatively wide, and the pair distinguishes the two.
    """

    low = np.asarray(lower, dtype=np.float64)
    high = np.asarray(upper, dtype=np.float64)
    if low.shape != high.shape:
        raise ValueError("interval bounds must have the same shape")
    if low.size == 0:
        raise ValueError("at least one interval is required")
    if not np.all(np.isfinite(low)) or not np.all(np.isfinite(high)):
        raise ValueError("interval bounds must be finite")
    if np.any(low > high):
        raise ValueError("lower interval bounds cannot exceed upper bounds")
    return float(np.mean(high - low))


def profile_rmse(predicted_effect: Any, observed_effect: Any) -> float:
    """Root mean squared error between a predicted and an observed effect profile.

    The effect is the treated-minus-matched-control contrast the caller has already formed.
    Passing raw profiles instead of contrasts measures a different quantity, and the frozen
    suite scores the contrast.
    """

    predicted = np.asarray(predicted_effect, dtype=np.float64).ravel()
    observed = np.asarray(observed_effect, dtype=np.float64).ravel()
    if predicted.shape != observed.shape:
        raise ValueError("predicted and observed effect profiles must have the same shape")
    if predicted.size == 0:
        raise ValueError("at least one feature is required")
    if not np.all(np.isfinite(predicted)) or not np.all(np.isfinite(observed)):
        raise ValueError("effect profiles must be finite")
    return float(np.sqrt(np.mean((predicted - observed) ** 2)))


def equal_group_mean(group_labels: Sequence[str], values: Any) -> float:
    """Mean over groups of the within-group mean, so every group carries equal weight.

    The frozen four-dose diagnostic is this applied to compounds: a compound measured at four
    doses must not outweigh one measured at fewer, and unequal weighting is a property of the
    statistic rather than of the reporting.
    """

    scores = np.asarray(values, dtype=np.float64).ravel()
    if scores.shape[0] != len(group_labels):
        raise ValueError("one value is required per group label")
    if scores.size == 0:
        raise ValueError("at least one value is required")
    if not np.all(np.isfinite(scores)):
        raise ValueError("values must be finite")
    totals: dict[str, list[float]] = {}
    for label, score in zip(group_labels, scores, strict=True):
        totals.setdefault(label, []).append(float(score))
    return float(np.mean([np.mean(group) for group in totals.values()]))


def differential_expression_weighted_rmse(
    predicted_effect: Any, observed_effect: Any, *, weights: Any
) -> float:
    """Effect RMSE with per-feature weights, normalized to sum to one.

    Marginal error over all features is minimized by predicting no change, so a suite that
    reports only unweighted error cannot distinguish a model from the null.  Weighting by a
    differential-expression statistic computed on training data alone concentrates the score on
    the features where change was expected.  Negatively oriented.
    """

    predicted = np.asarray(predicted_effect, dtype=np.float64).ravel()
    observed = np.asarray(observed_effect, dtype=np.float64).ravel()
    feature_weights = np.asarray(weights, dtype=np.float64).ravel()
    if not predicted.shape == observed.shape == feature_weights.shape:
        raise ValueError("effect profiles and weights must have the same shape")
    if predicted.size == 0:
        raise ValueError("at least one feature is required")
    if not np.all(np.isfinite(predicted)) or not np.all(np.isfinite(observed)):
        raise ValueError("effect profiles must be finite")
    if not np.all(np.isfinite(feature_weights)) or np.any(feature_weights < 0.0):
        raise ValueError("feature weights must be finite and nonnegative")
    total = float(feature_weights.sum())
    if total <= 0.0:
        raise ValueError("feature weights must not sum to zero")
    normalized = feature_weights / total
    return float(np.sqrt(np.sum(normalized * (predicted - observed) ** 2)))


def effect_rank_agreement(predicted_effect: Any, observed_effect: Any) -> float:
    """One minus the Spearman rank correlation between predicted and observed effects.

    Negatively oriented, so it composes with the other scores: zero is perfect agreement, one is
    no rank information, two is perfect disagreement.  A rank-based score is insensitive to the
    overall scale of an effect, which a correlation on raw magnitudes is not.
    """

    predicted = np.asarray(predicted_effect, dtype=np.float64).ravel()
    observed = np.asarray(observed_effect, dtype=np.float64).ravel()
    if predicted.shape != observed.shape:
        raise ValueError("predicted and observed effect profiles must have the same shape")
    if predicted.size < 2:
        raise ValueError("at least two features are required to rank")
    if not np.all(np.isfinite(predicted)) or not np.all(np.isfinite(observed)):
        raise ValueError("effect profiles must be finite")
    if predicted.min() == predicted.max() or observed.min() == observed.max():
        raise ValueError(
            "the rank correlation is undefined; a constant effect profile has no ranking"
        )
    return 1.0 - float(stats.spearmanr(predicted, observed).statistic)


@dataclass(frozen=True)
class MetricImplementation:
    """The binding between a declared ``metric_id`` and the code that computes it."""

    metric_id: str
    implementation_id: str
    implementation_version: str
    direction: MetricDirection
    computation: Callable[..., float]
    parameters: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "parameters", MappingProxyType(dict(self.parameters)))


def _binding(
    metric_id: str,
    computation: Callable[..., float],
    *,
    direction: MetricDirection = "minimize",
    **parameters: float,
) -> MetricImplementation:
    return MetricImplementation(
        metric_id=metric_id,
        implementation_id=f"cellstate.metric.{metric_id}",
        implementation_version=METRICS_IMPLEMENTATION_VERSION,
        direction=direction,
        computation=computation,
        parameters=parameters,
    )


#: Every ``metric_id`` the frozen sci-Plex3 metric-suite specification declares, bound to the
#: computation that resolves it.  The conformance test reads the specification's own bytes and
#: fails on any declared identifier missing from this mapping.
#:
#: ``sciplex3.four-dose-profile-diagnostic`` binds the outer aggregation of a composition: its
#: formula is ``equal_compound_mean(rmse_dose(...))``, so a scoring run forms one
#: :func:`profile_rmse` per exact dose and passes those to :func:`equal_group_mean` keyed by
#: compound.  ``tests/test_q1_metrics.py`` exercises that composition end to end so the binding
#: names a computation that produces the declared number rather than half of one.
METRIC_IMPLEMENTATIONS: Final[Mapping[str, MetricImplementation]] = MappingProxyType(
    {
        binding.metric_id: binding
        for binding in (
            _binding("sciplex3.marginal-crps-logcp10k", sample_crps),
            _binding("sciplex3.joint-energy-train-pca", energy_score),
            _binding(
                "sciplex3.marginal-coverage-error-p50",
                marginal_coverage_error,
                nominal_probability=0.50,
            ),
            _binding(
                "sciplex3.marginal-coverage-error-p80",
                marginal_coverage_error,
                nominal_probability=0.80,
            ),
            _binding(
                "sciplex3.marginal-coverage-error-p95",
                marginal_coverage_error,
                nominal_probability=0.95,
            ),
            _binding(
                "sciplex3.marginal-interval-width-p50",
                marginal_interval_width,
                nominal_probability=0.50,
            ),
            _binding(
                "sciplex3.marginal-interval-width-p80",
                marginal_interval_width,
                nominal_probability=0.80,
            ),
            _binding(
                "sciplex3.marginal-interval-width-p95",
                marginal_interval_width,
                nominal_probability=0.95,
            ),
            _binding("sciplex3.vehicle-relative-pseudobulk-rmse", profile_rmse),
            _binding("sciplex3.four-dose-profile-diagnostic", equal_group_mean),
        )
    }
)
