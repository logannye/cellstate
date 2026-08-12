"""Numerical references for the multiway clustered bootstrap.

The bootstrap is the first component here whose correctness is a statistical property rather than
an arithmetic one, so its references are of three kinds: exact arithmetic where the answer is
determined, published Student ``t`` table values for the small-cluster scale, and a measured
coverage study for the property that actually matters — whether a nominal 0.95 interval covers the
truth 95 percent of the time.

The coverage study is recorded here as a reference and not as a smoke test.  At the four-plate
shape of the sci-Plex3 untouched-test partition the unscaled percentile interval covers about
0.82 to 0.86 across seeds, and the scaled interval covers about 0.96.  Removing the scale makes
this file fail, which is the point: an interval that says 0.95 and delivers 0.83 is a gate that
fires when it should not.
"""

from __future__ import annotations

import math
from itertools import pairwise
from typing import Any

import numpy as np
import pytest

from cellstate.evaluation import bootstrap
from cellstate.evaluation.bootstrap import (
    FROZEN_SCIPLEX3_CONFIDENCE_LEVEL,
    FROZEN_SCIPLEX3_DEPENDENCE_DIMENSIONS,
    FROZEN_SCIPLEX3_RESAMPLE_COUNT,
    BootstrapInterval,
    multiway_clustered_bootstrap,
    small_cluster_scale,
    weighted_mean,
)

#: Shape of the frozen benchmark's untouched-test partition: 384 wells, 95 compounds, 4 plates.
UNTOUCHED_TEST_PLATES = 4
UNTOUCHED_TEST_COMPOUNDS = 95
UNTOUCHED_TEST_WELLS = 384


def _two_way_design(
    *,
    plates: int,
    compounds: int,
    wells: int,
    plate_sd: float,
    compound_sd: float,
    noise_sd: float,
    generator: np.random.Generator,
) -> tuple[np.ndarray, dict[str, list[str]]]:
    """A balanced two-way random-effects design with a true mean of zero."""

    plate_labels = [f"plate{index % plates}" for index in range(wells)]
    compound_labels = [f"compound{index % compounds}" for index in range(wells)]
    plate_index = np.array([index % plates for index in range(wells)])
    compound_index = np.array([index % compounds for index in range(wells)])
    plate_effects = generator.normal(0.0, plate_sd, plates)
    compound_effects = generator.normal(0.0, compound_sd, compounds)
    values = (
        plate_effects[plate_index]
        + compound_effects[compound_index]
        + generator.normal(0.0, noise_sd, wells)
    )
    return values, {"compound": compound_labels, "plate": plate_labels}


def _coverage(*, resample_count: int, replications: int, seed: int) -> tuple[float, float]:
    """Coverage of zero by the scaled and unscaled intervals, at the untouched-test shape."""

    generator = np.random.default_rng(seed)
    scaled = 0
    unscaled = 0
    for replication in range(replications):
        values, labels = _two_way_design(
            plates=UNTOUCHED_TEST_PLATES,
            compounds=UNTOUCHED_TEST_COMPOUNDS,
            wells=UNTOUCHED_TEST_WELLS,
            plate_sd=0.6,
            compound_sd=1.0,
            noise_sd=0.5,
            generator=generator,
        )
        interval = multiway_clustered_bootstrap(
            values=values,
            cluster_labels=labels,
            seed=seed * 1000 + replication,
            resample_count=resample_count,
        )
        scaled += interval.lower <= 0.0 <= interval.upper
        unscaled += interval.percentile_lower <= 0.0 <= interval.percentile_upper
    return scaled / replications, unscaled / replications


class TestSmallClusterScale:
    """The scale is ``t(K - 1, 0.975) / z(0.975)`` and is checked against published quantiles."""

    @pytest.mark.parametrize(
        ("clusters", "student_quantile"),
        [(2, 12.7062), (4, 3.1824), (11, 2.2281), (12, 2.2010), (95, 1.9855)],
    )
    def test_matches_published_student_quantiles(
        self, clusters: int, student_quantile: float
    ) -> None:
        normal_quantile = 1.959964
        expected = student_quantile / normal_quantile
        assert small_cluster_scale(
            minimum_cluster_count=clusters, confidence_level=0.95
        ) == pytest.approx(expected, rel=1e-4)

    def test_decreases_monotonically_toward_one(self) -> None:
        scales = [
            small_cluster_scale(minimum_cluster_count=count, confidence_level=0.95)
            for count in (2, 4, 8, 16, 64, 256, 4096)
        ]
        assert all(earlier > later for earlier, later in pairwise(scales))
        assert scales[-1] == pytest.approx(1.0, abs=5e-4)
        assert all(scale >= 1.0 for scale in scales)

    def test_a_single_cluster_cannot_be_bootstrapped(self) -> None:
        with pytest.raises(ValueError, match="at least two clusters"):
            small_cluster_scale(minimum_cluster_count=1, confidence_level=0.95)


class TestCoverage:
    """The property the estimator exists to deliver, measured rather than assumed."""

    @pytest.mark.parametrize("seed", [3, 23])
    def test_scaled_interval_covers_and_unscaled_interval_does_not(self, seed: int) -> None:
        scaled, unscaled = _coverage(resample_count=400, replications=300, seed=seed)
        assert scaled >= 0.93, f"scaled coverage {scaled} fell below the nominal 0.95"
        assert unscaled <= 0.90, (
            f"unscaled coverage {unscaled} was not anticonservative; if the percentile interval "
            "now covers, the recorded justification for the small-cluster scale no longer holds "
            "and must be revisited rather than the threshold relaxed"
        )

    def test_coverage_holds_at_the_frozen_resample_count(self) -> None:
        scaled, unscaled = _coverage(
            resample_count=FROZEN_SCIPLEX3_RESAMPLE_COUNT, replications=100, seed=3
        )
        assert scaled >= 0.93
        assert unscaled <= 0.90


class TestStandardError:
    def test_tracks_the_analytic_two_way_standard_error(self) -> None:
        plates, compounds, wells = 12, 30, 360
        plate_sd, compound_sd, noise_sd = 1.0, 0.7, 0.5
        generator = np.random.default_rng(101)
        values, labels = _two_way_design(
            plates=plates,
            compounds=compounds,
            wells=wells,
            plate_sd=plate_sd,
            compound_sd=compound_sd,
            noise_sd=noise_sd,
            generator=generator,
        )
        analytic = math.sqrt(
            plate_sd**2 / plates + compound_sd**2 / compounds + noise_sd**2 / wells
        )
        interval = multiway_clustered_bootstrap(
            values=values, cluster_labels=labels, seed=5, resample_count=2_000
        )
        assert interval.standard_error == pytest.approx(analytic, rel=0.35)

    def test_a_one_way_bootstrap_of_independent_units_matches_the_plain_standard_error(
        self,
    ) -> None:
        generator = np.random.default_rng(202)
        values = generator.normal(0.0, 1.0, 500)
        labels = {"unit": [f"unit{index}" for index in range(500)]}
        interval = multiway_clustered_bootstrap(
            values=values, cluster_labels=labels, seed=6, resample_count=2_000
        )
        plain = float(values.std(ddof=1) / math.sqrt(values.size))
        assert interval.standard_error == pytest.approx(plain, rel=0.15)


class TestDeterminismAndReporting:
    def _interval(self, seed: int = 9) -> BootstrapInterval:
        generator = np.random.default_rng(303)
        values, labels = _two_way_design(
            plates=6,
            compounds=20,
            wells=120,
            plate_sd=0.5,
            compound_sd=0.8,
            noise_sd=0.4,
            generator=generator,
        )
        return multiway_clustered_bootstrap(
            values=values, cluster_labels=labels, seed=seed, resample_count=200
        )

    def test_the_same_seed_reproduces_the_interval_exactly(self) -> None:
        assert self._interval().model_dump() == self._interval().model_dump()

    def test_a_different_seed_changes_the_interval(self) -> None:
        assert self._interval(seed=9).lower != self._interval(seed=10).lower

    def test_reports_the_dependence_structure_it_used(self) -> None:
        interval = self._interval()
        assert interval.dependence_dimension_ids == ("compound", "plate")
        assert interval.cluster_counts == (20, 6)
        assert interval.minimum_cluster_count == 6
        assert interval.evaluation_unit_count == 120
        assert interval.resampling_scheme == "multiway_clustered"
        assert interval.degenerate_resample_count == 0

    def test_the_scaled_interval_contains_the_percentile_interval(self) -> None:
        interval = self._interval()
        assert interval.lower <= interval.percentile_lower
        assert interval.upper >= interval.percentile_upper
        assert interval.small_cluster_scale > 1.0
        assert interval.width == pytest.approx(interval.upper - interval.lower)

    def test_excludes_zero_is_recomputed_from_the_endpoints(self) -> None:
        generator = np.random.default_rng(404)
        values, labels = _two_way_design(
            plates=8,
            compounds=40,
            wells=160,
            plate_sd=0.2,
            compound_sd=0.2,
            noise_sd=0.2,
            generator=generator,
        )
        shifted = multiway_clustered_bootstrap(
            values=values + 50.0, cluster_labels=labels, seed=11, resample_count=400
        )
        assert shifted.excludes_zero is True
        assert shifted.lower > 0.0
        centered = multiway_clustered_bootstrap(
            values=values - float(np.mean(values)),
            cluster_labels=labels,
            seed=11,
            resample_count=400,
        )
        assert centered.excludes_zero is False


class TestStatisticContract:
    def test_the_default_statistic_is_the_weighted_mean(self) -> None:
        values = np.array([1.0, 2.0, 3.0, 4.0])
        weights = np.array([0.0, 2.0, 1.0, 1.0])
        assert weighted_mean(values, weights) == pytest.approx((4.0 + 3.0 + 4.0) / 4.0)

    def test_a_weighted_mean_of_zero_total_weight_is_undefined(self) -> None:
        with pytest.raises(ZeroDivisionError, match="positive total weight"):
            weighted_mean(np.array([1.0]), np.array([0.0]))

    def test_the_statistic_receives_the_resample_weights(self) -> None:
        seen: list[float] = []

        def recording_statistic(values: np.ndarray, weights: np.ndarray) -> float:
            seen.append(float(weights.sum()))
            return weighted_mean(values, weights)

        generator = np.random.default_rng(505)
        values, labels = _two_way_design(
            plates=4,
            compounds=8,
            wells=32,
            plate_sd=0.5,
            compound_sd=0.5,
            noise_sd=0.5,
            generator=generator,
        )
        multiway_clustered_bootstrap(
            values=values,
            cluster_labels=labels,
            seed=12,
            statistic=recording_statistic,
            resample_count=50,
        )
        assert seen[0] == pytest.approx(32.0), "the point estimate must use unit weights"
        assert any(total != pytest.approx(32.0) for total in seen[1:])


class TestFailsClosed:
    def _small_design(self) -> tuple[np.ndarray, dict[str, list[str]]]:
        values = np.array([1.0, 2.0, 3.0, 4.0])
        return values, {
            "compound": ["a", "a", "b", "b"],
            "plate": ["p", "q", "p", "q"],
        }

    def test_degenerate_resamples_are_fatal_by_default(self) -> None:
        values = np.array([1.0, 2.0])
        labels = {"compound": ["a", "b"], "plate": ["p", "q"]}
        with pytest.raises(ValueError, match="zero total weight"):
            multiway_clustered_bootstrap(
                values=values, cluster_labels=labels, seed=13, resample_count=2_000
            )

    def test_degenerate_resamples_must_be_allowed_explicitly(self) -> None:
        values = np.array([1.0, 2.0])
        labels = {"compound": ["a", "b"], "plate": ["p", "q"]}
        interval = multiway_clustered_bootstrap(
            values=values,
            cluster_labels=labels,
            seed=13,
            resample_count=2_000,
            maximum_degenerate_resamples=2_000,
        )
        assert interval.degenerate_resample_count > 0

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"resample_count": 1}, "at least two resamples"),
            ({"confidence_level": 1.0}, "strictly between zero and one"),
            ({"confidence_level": 0.0}, "strictly between zero and one"),
            ({"maximum_degenerate_resamples": -1}, "nonnegative"),
        ],
    )
    def test_rejects_impossible_configurations(self, kwargs: dict[str, Any], message: str) -> None:
        values, labels = self._small_design()
        with pytest.raises(ValueError, match=message):
            multiway_clustered_bootstrap(values=values, cluster_labels=labels, seed=14, **kwargs)

    def test_rejects_a_label_vector_of_the_wrong_length(self) -> None:
        values, labels = self._small_design()
        labels["plate"] = ["p", "q"]
        with pytest.raises(ValueError, match="labels"):
            multiway_clustered_bootstrap(values=values, cluster_labels=labels, seed=15)

    def test_rejects_missing_dependence_structure(self) -> None:
        with pytest.raises(ValueError, match="at least one dependence dimension"):
            multiway_clustered_bootstrap(values=np.array([1.0, 2.0]), cluster_labels={}, seed=16)

    def test_rejects_nonfinite_and_empty_values(self) -> None:
        _, labels = self._small_design()
        with pytest.raises(ValueError, match="finite"):
            multiway_clustered_bootstrap(
                values=np.array([1.0, 2.0, 3.0, np.inf]), cluster_labels=labels, seed=17
            )
        with pytest.raises(ValueError, match="at least one evaluation unit"):
            multiway_clustered_bootstrap(values=np.array([]), cluster_labels={"plate": []}, seed=18)

    def test_rejects_multidimensional_values(self) -> None:
        with pytest.raises(ValueError, match="one-dimensional"):
            multiway_clustered_bootstrap(
                values=np.zeros((2, 2)), cluster_labels={"plate": ["p", "q"]}, seed=19
            )


def test_the_resample_chunk_size_is_pinned() -> None:
    """The chunk size is part of the numeric contract, not a tuning knob.

    The generator is consumed one chunk and one dimension at a time, so a different chunk size
    produces a different draw sequence and different endpoints at the same seed.  Changing it is
    a version bump, and this test is what forces that to be a decision.
    """

    assert bootstrap._RESAMPLE_CHUNK == 256
    assert bootstrap.BOOTSTRAP_IMPLEMENTATION_VERSION == "1.0.0"


def test_frozen_configuration_constants_are_what_the_suite_declares() -> None:
    assert FROZEN_SCIPLEX3_RESAMPLE_COUNT == 2_000
    assert FROZEN_SCIPLEX3_CONFIDENCE_LEVEL == 0.95
    assert FROZEN_SCIPLEX3_DEPENDENCE_DIMENSIONS == ("compound", "plate")
