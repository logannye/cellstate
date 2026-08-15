"""The path from raw-count predictions to a scored, interval-bearing metric result.

`Q1` implemented the metrics and the clustered bootstrap; `Q2` implemented the two faithfulness
harnesses.  Neither could be pointed at biology, because nothing carried a baseline's raw-count
predictions across to a metric value: the frozen panel-only scoring transform existed in `src/`
only as a content hash, and `MetricResult` had no construction site anywhere.  This module is that
missing span.

Three things happen here and nothing else.

**The frozen transform is applied.**  :func:`panel_logcp10k` is
``log1p(10000 * count_i / panel_total)`` where ``panel_total`` sums exactly the declared ordered
2,000-feature panel and no full-source-axis denominator is ever read.  The frozen specification
requires it to be applied *identically and independently* to every observed and every predicted
sample, so both go through the same function, and its validation policy is fail-closed: a wrong
length, a nonfinite, noninteger or negative coordinate, or a nonpositive panel total fails the
evaluation and is never dropped, imputed, renormalized, clipped or substituted.

**One scalar is produced per evaluation unit.**  The frozen suite weights
``equal_evaluation_unit`` and sets ``forbid_implicit_record_count_weighting``, so a well with many
recovered nuclei must not outweigh a well with few.  :func:`score_case` therefore reduces each well
to exactly one value per metric before anything is aggregated, which is the only point at which
that weighting can be enforced.

**The aggregate carries its sampling distribution.**  :func:`aggregate_metric` sends the per-well
values through the same multiway clustered bootstrap every metric in the suite binds, grouped
jointly over compound and plate, and returns the point estimate with its interval.

What this module deliberately does not do is decide whether a number it produces is admissible.
Scoring a baseline against the partition it was fitted on is an in-sample diagnostic and not an
observational floor; the frozen artifact evaluates on ``p4-untouched-test`` and reaching that
partition is a lifecycle decision (ADR 0011) this module neither makes nor implies.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

from cellstate.domain.common import BootstrapInterval
from cellstate.evaluation.bootstrap import (
    FROZEN_SCIPLEX3_CONFIDENCE_LEVEL,
    FROZEN_SCIPLEX3_RESAMPLE_COUNT,
    multiway_clustered_bootstrap,
)
from cellstate.evaluation.metrics import (
    central_interval,
    energy_score,
    marginal_coverage_error,
    marginal_interval_width,
    profile_rmse,
    sample_crps,
)

#: Identity of the frozen transform, as the scoring-transform artifact declares it.
SCORING_TRANSFORM_ID: Final = "panel-only-natural-log-cp10k"
SCORING_TRANSFORM_VERSION: Final = "1.0.0"
SCORING_TRANSFORM_SCALE: Final = 10_000.0

#: The rank the frozen benchmark artifact declares for the train-only energy projection.
FROZEN_SCIPLEX3_PROJECTION_RANK: Final = 50

#: The nominal probabilities the frozen suite declares, paired with the identifier suffix each
#: appears under.  The suffix is carried rather than derived from the probability: a metric_id is
#: an identity the frozen artifact declares, not a number formatted at runtime.
FROZEN_SCIPLEX3_COVERAGE_LEVELS: Final[tuple[tuple[str, float], ...]] = (
    ("p50", 0.50),
    ("p80", 0.80),
    ("p95", 0.95),
)

SCORING_IMPLEMENTATION_VERSION: Final = "1.0.0"

__all__ = [
    "FROZEN_SCIPLEX3_COVERAGE_LEVELS",
    "FROZEN_SCIPLEX3_PROJECTION_RANK",
    "SCORING_IMPLEMENTATION_VERSION",
    "SCORING_TRANSFORM_ID",
    "SCORING_TRANSFORM_SCALE",
    "SCORING_TRANSFORM_VERSION",
    "EvaluationCaseInputs",
    "ScoredCase",
    "ScoringTransformError",
    "TrainProjection",
    "aggregate_metric",
    "equal_group_weighted_mean",
    "panel_logcp10k",
    "score_case",
]


class ScoringTransformError(ValueError):
    """A fatal scoring-transform condition.

    The frozen validation policy is that any of these fails the metric evaluation and therefore
    blocks admission.  It is deliberately not recoverable: dropping, excluding, imputing,
    renormalizing, clipping or substituting the offending sample would change the estimand while
    leaving the reported number looking the same.
    """


def panel_logcp10k(counts: object, *, feature_count: int) -> NDArray[np.float64]:
    """Apply the frozen panel-only logCP10k transform to raw integer UMI counts.

    ``counts`` is one ``(n, feature_count)`` block of raw counts in exact declared panel order, or
    a single vector of that length.  The denominator is the sum of exactly those coordinates; no
    external or full-source-axis denominator is read, which is what makes the transform computable
    from a predicted sample that has no source library behind it at all.
    """

    matrix = np.atleast_2d(np.asarray(counts))
    if matrix.ndim != 2:
        raise ScoringTransformError("counts must form a two-dimensional block of panel vectors")
    if matrix.shape[1] != feature_count:
        raise ScoringTransformError(
            f"vector_length_not_exactly_{feature_count}: got {matrix.shape[1]}"
        )
    if matrix.shape[0] == 0:
        raise ScoringTransformError("at least one count vector is required")

    if not np.issubdtype(matrix.dtype, np.integer):
        as_float = np.asarray(matrix, dtype=np.float64)
        if not np.all(np.isfinite(as_float)):
            raise ScoringTransformError("nonfinite_coordinate")
        if not np.all(as_float == np.rint(as_float)):
            raise ScoringTransformError("noninteger_coordinate")
        matrix = np.asarray(np.rint(as_float), dtype=np.int64)

    values = np.asarray(matrix, dtype=np.int64)
    if bool(np.any(values < 0)):
        raise ScoringTransformError("negative_coordinate")
    panel_total = values.sum(axis=1, dtype=np.int64)
    if bool(np.any(panel_total <= 0)):
        raise ScoringTransformError("panel_total_less_than_or_equal_to_zero")

    scaled = SCORING_TRANSFORM_SCALE * (
        values.astype(np.float64) / panel_total.astype(np.float64)[:, None]
    )
    transformed = np.log1p(scaled)
    if not np.all(np.isfinite(transformed)):  # pragma: no cover - unreachable given the guards
        raise ScoringTransformError("nonfinite_coordinate")
    return np.asarray(transformed, dtype=np.float64)


@dataclass(frozen=True)
class TrainProjection:
    """A rank-truncated linear projection fitted on the training partition only.

    The frozen suite scores its joint energy in ``a projection fit on train only after the
    panel-only scoring transform``, at the rank the benchmark artifact declares.  Fitting it
    anywhere but train would let held-out structure into the metric's own definition, so the
    partition this was fitted on travels with it.
    """

    rank: int
    center: NDArray[np.float64]
    components: NDArray[np.float64]
    fit_partition_id: str

    @classmethod
    def fit(
        cls,
        transformed_train: NDArray[np.float64],
        *,
        rank: int = FROZEN_SCIPLEX3_PROJECTION_RANK,
        fit_partition_id: str,
    ) -> TrainProjection:
        values = np.asarray(transformed_train, dtype=np.float64)
        if values.ndim != 2 or values.shape[0] < 2:
            raise ValueError("a projection requires at least two transformed training rows")
        if rank < 1:
            raise ValueError("projection rank must be positive")
        if not np.all(np.isfinite(values)):
            raise ValueError("transformed training values must be finite")
        effective = min(rank, values.shape[0] - 1, values.shape[1])
        if effective < 1:
            raise ValueError("the training block is too small to support a projection")
        center = values.mean(axis=0)
        _, _, right = np.linalg.svd(values - center, full_matrices=False)
        return cls(
            rank=effective,
            center=np.asarray(center, dtype=np.float64),
            components=np.asarray(right[:effective], dtype=np.float64),
            fit_partition_id=fit_partition_id,
        )

    def project(self, transformed: NDArray[np.float64]) -> NDArray[np.float64]:
        values = np.atleast_2d(np.asarray(transformed, dtype=np.float64))
        if values.shape[1] != self.components.shape[1]:
            raise ValueError("projected values must lie on the panel the projection was fitted on")
        return np.asarray((values - self.center) @ self.components.T, dtype=np.float64)


@dataclass(frozen=True)
class EvaluationCaseInputs:
    """Everything one evaluation unit contributes, with its dependence labels.

    ``vehicle_counts`` are the matched same-plate no-action control rows.  The frozen effect
    metrics score the treated-minus-matched-control contrast, so the control is an input to the
    case rather than something the scorer goes and finds.
    """

    case_id: str
    evaluation_unit_id: str
    compound_id: str
    plate_id: str
    observed_counts: NDArray[np.int64]
    predicted_counts: NDArray[np.int64]
    vehicle_counts: NDArray[np.int64]


@dataclass(frozen=True)
class ScoredCase:
    """One evaluation unit reduced to one value per metric, with its dependence labels."""

    case_id: str
    evaluation_unit_id: str
    compound_id: str
    plate_id: str
    values: Mapping[str, float]


def score_case(
    case: EvaluationCaseInputs,
    *,
    projection: TrainProjection,
    feature_count: int,
    coverage_levels: Sequence[tuple[str, float]] = FROZEN_SCIPLEX3_COVERAGE_LEVELS,
) -> ScoredCase:
    """Reduce one evaluation unit to one value per declared ``metric_id``.

    Every metric is computed from the same transformed observed block and the same transformed
    predictive block, so a difference between two metrics is a difference of statistic and never
    of preprocessing.
    """

    observed = panel_logcp10k(case.observed_counts, feature_count=feature_count)
    predicted = panel_logcp10k(case.predicted_counts, feature_count=feature_count)
    vehicle = panel_logcp10k(case.vehicle_counts, feature_count=feature_count)

    values: dict[str, float] = {
        "sciplex3.marginal-crps-logcp10k": sample_crps(observed, predicted),
        "sciplex3.joint-energy-train-pca": energy_score(
            projection.project(observed), projection.project(predicted)
        ),
    }

    for tag, probability in coverage_levels:
        lower, upper = central_interval(predicted, probability=probability)
        values[f"sciplex3.marginal-coverage-error-{tag}"] = marginal_coverage_error(
            observed, lower, upper, nominal_probability=probability
        )
        values[f"sciplex3.marginal-interval-width-{tag}"] = marginal_interval_width(lower, upper)

    # The vehicle-relative contrast, formed once and scored twice: the pseudobulk metric averages
    # it equally over wells, the four-dose diagnostic equally over compounds.  They are the same
    # per-unit quantity under two weightings, which is why only the aggregation differs.
    vehicle_profile = vehicle.mean(axis=0)
    effect_rmse = profile_rmse(
        predicted.mean(axis=0) - vehicle_profile, observed.mean(axis=0) - vehicle_profile
    )
    values["sciplex3.vehicle-relative-pseudobulk-rmse"] = effect_rmse
    values["sciplex3.four-dose-profile-diagnostic"] = effect_rmse

    for metric_id, value in values.items():
        if not math.isfinite(value):
            raise ScoringTransformError(
                f"metric {metric_id!r} produced a nonfinite value on case {case.case_id!r}; "
                "a nonfinite score is a failed evaluation, never a zero"
            )
    return ScoredCase(
        case_id=case.case_id,
        evaluation_unit_id=case.evaluation_unit_id,
        compound_id=case.compound_id,
        plate_id=case.plate_id,
        values=dict(values),
    )


def equal_group_weighted_mean(
    group_labels: Sequence[str],
) -> Callable[[NDArray[np.float64], NDArray[np.float64]], float]:
    """A bootstrap statistic giving every group equal weight, honoring resample weights.

    The four-dose diagnostic weights compounds equally rather than wells, so the grouping is a
    property of the statistic.  A statistic that groups internally must consume the weights the
    bootstrap hands it, or the interval would describe a different estimand than the point
    estimate; a group whose every member was dropped from a resample contributes nothing rather
    than contributing a zero.
    """

    labels = tuple(group_labels)

    def statistic(values: NDArray[np.float64], weights: NDArray[np.float64]) -> float:
        if values.shape[0] != len(labels):
            raise ValueError("one value is required per group label")
        weighted: dict[str, list[float]] = {}
        totals: dict[str, float] = {}
        for label, value, weight in zip(labels, values, weights, strict=True):
            weighted.setdefault(label, []).append(float(value) * float(weight))
            totals[label] = totals.get(label, 0.0) + float(weight)
        means = [sum(weighted[label]) / totals[label] for label in weighted if totals[label] > 0.0]
        if not means:
            raise ZeroDivisionError("no group retained positive weight")
        return float(np.mean(means))

    return statistic


def aggregate_metric(
    scored: Sequence[ScoredCase],
    metric_id: str,
    *,
    seed: int,
    equal_weight_group: str | None = None,
    resample_count: int = FROZEN_SCIPLEX3_RESAMPLE_COUNT,
    confidence_level: float = FROZEN_SCIPLEX3_CONFIDENCE_LEVEL,
) -> BootstrapInterval:
    """Aggregate one metric over evaluation units and return its sampling distribution.

    ``equal_weight_group`` selects the four-dose diagnostic's compound weighting; the default of
    ``None`` is the suite's ``equal_evaluation_unit`` scheme, one well one vote.
    """

    if not scored:
        raise ValueError("at least one scored evaluation unit is required")
    missing = [case.case_id for case in scored if metric_id not in case.values]
    if missing:
        raise ValueError(
            f"metric {metric_id!r} is absent from {len(missing)} scored case(s); the frozen suite "
            "sets error_on_missing and an absent target is never silently dropped"
        )
    unit_ids = [case.evaluation_unit_id for case in scored]
    if len(set(unit_ids)) != len(unit_ids):
        raise ValueError("evaluation units must be unique; a repeated unit would be double counted")

    values = [case.values[metric_id] for case in scored]
    cluster_labels = {
        "compound": [case.compound_id for case in scored],
        "plate": [case.plate_id for case in scored],
    }
    statistic = None
    if equal_weight_group is not None:
        if equal_weight_group not in cluster_labels:
            raise ValueError(f"unknown equal-weight group {equal_weight_group!r}")
        statistic = equal_group_weighted_mean(cluster_labels[equal_weight_group])

    kwargs: dict[str, object] = {
        "values": values,
        "cluster_labels": cluster_labels,
        "seed": seed,
        "resample_count": resample_count,
        "confidence_level": confidence_level,
    }
    if statistic is not None:
        kwargs["statistic"] = statistic
    return multiway_clustered_bootstrap(**kwargs)  # type: ignore[arg-type]
