"""The span from a raw-count prediction to an interval-bearing metric value.

Every reference here is derived independently of the implementation: closed forms, hand
arithmetic, or an identity that must hold for reasons outside the code under test.  The frozen
scoring transform is short enough to write out longhand, and that is what the first class does.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from cellstate.evaluation.metrics import equal_group_mean, profile_rmse
from cellstate.evaluation.scoring import (
    FROZEN_SCIPLEX3_PROJECTION_RANK,
    EvaluationCaseInputs,
    ScoringTransformError,
    TrainProjection,
    aggregate_metric,
    equal_group_weighted_mean,
    panel_logcp10k,
    score_case,
)

#: Read from the frozen specification's own bytes rather than written out here.  A hand-kept list
#: would drift silently: the scorer would keep producing the ten identifiers someone once typed
#: while the suite it claims to satisfy declared a different set.
_SPECIFICATION = (
    Path(__file__).resolve().parents[1]
    / "benchmarks/vertical-a/sciplex3-k562-24h-v1/support/metric-suite-spec.json"
)
DECLARED_METRIC_IDS = tuple(
    entry["metric_id"] for entry in json.loads(_SPECIFICATION.read_text())["metrics"]
)


class TestTheFrozenTransform:
    def test_matches_the_formula_written_out_longhand(self) -> None:
        counts = np.array([[1, 3, 6]])
        expected = np.array(
            [
                math.log1p(10000 * 1 / 10),
                math.log1p(10000 * 3 / 10),
                math.log1p(10000 * 6 / 10),
            ]
        )
        result = panel_logcp10k(counts, feature_count=3)
        assert result[0] == pytest.approx(expected, rel=1e-15, abs=1e-15)

    def test_the_denominator_is_the_panel_and_only_the_panel(self) -> None:
        """The transform is compositional, which is what makes it computable on a prediction.

        A predicted sample has no source library behind it, so a transform that needed a
        full-source-axis denominator could not be applied to one at all.  Doubling every panel
        coordinate must therefore leave the result unchanged.
        """

        counts = np.array([[1, 3, 6]])
        assert panel_logcp10k(counts * 2, feature_count=3) == pytest.approx(
            panel_logcp10k(counts, feature_count=3)
        )

    def test_a_zero_row_is_never_silently_rescued(self) -> None:
        with pytest.raises(ScoringTransformError, match="panel_total_less_than_or_equal_to_zero"):
            panel_logcp10k(np.array([[0, 0, 0]]), feature_count=3)

    @pytest.mark.parametrize(
        ("counts", "condition"),
        [
            (np.array([[1, 2]]), "vector_length_not_exactly_3"),
            (np.array([[1.5, 2.0, 3.0]]), "noninteger_coordinate"),
            (np.array([[-1, 2, 3]]), "negative_coordinate"),
            (np.array([[np.nan, 2.0, 3.0]]), "nonfinite_coordinate"),
            (np.array([[np.inf, 2.0, 3.0]]), "nonfinite_coordinate"),
        ],
    )
    def test_every_declared_fatal_condition_fires(self, counts: np.ndarray, condition: str) -> None:
        with pytest.raises(ScoringTransformError, match=condition):
            panel_logcp10k(counts, feature_count=3)

    def test_an_integer_valued_float_is_accepted(self) -> None:
        """The policy forbids noninteger values, not the float dtype they may arrive in."""

        assert panel_logcp10k(np.array([[1.0, 3.0, 6.0]]), feature_count=3) == pytest.approx(
            panel_logcp10k(np.array([[1, 3, 6]]), feature_count=3)
        )


class TestTheTrainProjection:
    def test_it_truncates_to_the_declared_rank(self) -> None:
        rng = np.random.default_rng(0)
        train = rng.normal(size=(80, 30))
        projection = TrainProjection.fit(train, rank=5, fit_partition_id="p1-train")
        assert projection.rank == 5
        assert projection.project(train).shape == (80, 5)

    def test_it_cannot_exceed_what_the_training_block_supports(self) -> None:
        rng = np.random.default_rng(1)
        train = rng.normal(size=(6, 30))
        projection = TrainProjection.fit(
            train, rank=FROZEN_SCIPLEX3_PROJECTION_RANK, fit_partition_id="p1-train"
        )
        assert projection.rank == 5

    def test_it_recovers_an_exactly_low_rank_subspace(self) -> None:
        """An independent check: data built in a 3-dimensional subspace must project losslessly."""

        rng = np.random.default_rng(2)
        basis = rng.normal(size=(3, 20))
        train = rng.normal(size=(60, 3)) @ basis
        projection = TrainProjection.fit(train, rank=3, fit_partition_id="p1-train")
        reconstructed = projection.project(train) @ projection.components + projection.center
        assert np.max(np.abs(reconstructed - train)) < 1e-8

    def test_the_partition_it_was_fitted_on_travels_with_it(self) -> None:
        projection = TrainProjection.fit(
            np.random.default_rng(3).normal(size=(40, 10)), rank=2, fit_partition_id="p1-train"
        )
        assert projection.fit_partition_id == "p1-train"


def _case(
    *,
    case_id: str = "case-1",
    unit: str = "plate1_A1",
    compound: str = "compoundA",
    plate: str = "plate1",
    seed: int = 0,
) -> EvaluationCaseInputs:
    rng = np.random.default_rng(seed)
    return EvaluationCaseInputs(
        case_id=case_id,
        evaluation_unit_id=unit,
        compound_id=compound,
        plate_id=plate,
        observed_counts=rng.poisson(6.0, size=(25, 12)).astype(np.int64) + 1,
        predicted_counts=rng.poisson(6.0, size=(40, 12)).astype(np.int64) + 1,
        vehicle_counts=rng.poisson(5.0, size=(20, 12)).astype(np.int64) + 1,
    )


class TestScoringOneEvaluationUnit:
    @pytest.fixture
    def projection(self) -> TrainProjection:
        rng = np.random.default_rng(7)
        train = panel_logcp10k(
            rng.poisson(6.0, size=(60, 12)).astype(np.int64) + 1, feature_count=12
        )
        return TrainProjection.fit(train, rank=4, fit_partition_id="p1-train")

    def test_it_produces_exactly_the_metric_ids_the_frozen_suite_declares(
        self, projection: TrainProjection
    ) -> None:
        """Both directions, against the specification's own bytes.

        `Q1`'s conformance test proved every declared identifier resolves to *code*.  This proves
        the scoring path actually *emits* a value for each one and invents none, which is the
        difference between a registry that type-checks and a run that scores.
        """

        assert len(DECLARED_METRIC_IDS) == 10
        scored = score_case(_case(), projection=projection, feature_count=12)
        assert set(scored.values) == set(DECLARED_METRIC_IDS)
        assert all(math.isfinite(value) for value in scored.values.values())

    def test_it_reduces_a_unit_to_one_value_however_many_nuclei_it_has(
        self, projection: TrainProjection
    ) -> None:
        """``forbid_implicit_record_count_weighting`` can only be enforced at this reduction."""

        scored = score_case(_case(), projection=projection, feature_count=12)
        for value in scored.values.values():
            assert np.ndim(value) == 0

    def test_interval_width_increases_with_the_nominal_probability(
        self, projection: TrainProjection
    ) -> None:
        scored = score_case(_case(), projection=projection, feature_count=12)
        widths = [scored.values[f"sciplex3.marginal-interval-width-p{tag}"] for tag in (50, 80, 95)]
        assert widths[0] < widths[1] < widths[2]

    def test_the_effect_metrics_score_the_vehicle_relative_contrast(
        self, projection: TrainProjection
    ) -> None:
        """Recompute the contrast independently and check the scorer formed the same one."""

        case = _case()
        observed = panel_logcp10k(case.observed_counts, feature_count=12)
        predicted = panel_logcp10k(case.predicted_counts, feature_count=12)
        vehicle = panel_logcp10k(case.vehicle_counts, feature_count=12).mean(axis=0)
        expected = profile_rmse(predicted.mean(axis=0) - vehicle, observed.mean(axis=0) - vehicle)
        scored = score_case(case, projection=projection, feature_count=12)
        assert scored.values["sciplex3.vehicle-relative-pseudobulk-rmse"] == pytest.approx(expected)
        assert scored.values["sciplex3.four-dose-profile-diagnostic"] == pytest.approx(expected)

    def test_a_zero_total_observed_row_fails_the_evaluation(
        self, projection: TrainProjection
    ) -> None:
        case = _case()
        broken = np.array(case.observed_counts, copy=True)
        broken[0, :] = 0
        with pytest.raises(ScoringTransformError):
            score_case(
                EvaluationCaseInputs(
                    case_id=case.case_id,
                    evaluation_unit_id=case.evaluation_unit_id,
                    compound_id=case.compound_id,
                    plate_id=case.plate_id,
                    observed_counts=broken,
                    predicted_counts=case.predicted_counts,
                    vehicle_counts=case.vehicle_counts,
                ),
                projection=projection,
                feature_count=12,
            )


class TestTheEqualCompoundStatistic:
    def test_at_unit_weights_it_equals_the_frozen_equal_group_mean(self) -> None:
        """The point estimate must be the statistic the suite declares, not an approximation."""

        labels = ["a", "a", "a", "b"]
        values = np.array([1.0, 2.0, 3.0, 10.0])
        statistic = equal_group_weighted_mean(labels)
        assert statistic(values, np.ones(4)) == pytest.approx(equal_group_mean(labels, values))

    def test_it_differs_from_the_plain_mean_when_groups_are_unbalanced(self) -> None:
        labels = ["a", "a", "a", "b"]
        values = np.array([1.0, 2.0, 3.0, 10.0])
        assert equal_group_weighted_mean(labels)(values, np.ones(4)) == pytest.approx(6.0)
        assert float(values.mean()) == pytest.approx(4.0)

    def test_it_honors_the_weights_it_is_given(self) -> None:
        """A statistic that ignored its weights would report an interval for another estimand."""

        labels = ["a", "a", "b"]
        values = np.array([1.0, 3.0, 10.0])
        statistic = equal_group_weighted_mean(labels)
        # Dropping the first member of group 'a' must move the answer to mean(3, 10).
        assert statistic(values, np.array([0.0, 1.0, 1.0])) == pytest.approx(6.5)

    def test_a_group_that_lost_every_member_contributes_nothing_rather_than_zero(self) -> None:
        labels = ["a", "a", "b"]
        values = np.array([1.0, 3.0, 10.0])
        statistic = equal_group_weighted_mean(labels)
        assert statistic(values, np.array([1.0, 1.0, 0.0])) == pytest.approx(2.0)


class TestAggregation:
    def _scored(self, *, plates: int = 4, compounds: int = 12) -> list:
        """A fully crossed plate-by-compound design, as the frozen partition approximately is.

        Full crossing matters: under the pigeonhole bootstrap a unit survives only if *both* its
        clusters were drawn, so a sparsely crossed design produces resamples with zero total
        weight, which the estimator refuses rather than silently discards.  That refusal is
        exercised separately below.
        """

        projection = TrainProjection.fit(
            panel_logcp10k(
                np.random.default_rng(11).poisson(6.0, size=(60, 12)).astype(np.int64) + 1,
                feature_count=12,
            ),
            rank=4,
            fit_partition_id="p1-train",
        )
        return [
            score_case(
                _case(
                    case_id=f"case-{plate}-{compound}",
                    unit=f"plate{plate}_W{compound}",
                    compound=f"compound{compound}",
                    plate=f"plate{plate}",
                    seed=plate * compounds + compound,
                ),
                projection=projection,
                feature_count=12,
            )
            for plate in range(plates)
            for compound in range(compounds)
        ]

    def test_it_returns_an_interval_grouped_at_compound_and_plate(self) -> None:
        interval = aggregate_metric(
            self._scored(), "sciplex3.marginal-crps-logcp10k", seed=5, resample_count=200
        )
        assert interval.dependence_dimension_ids == ("compound", "plate")
        assert interval.cluster_counts == (12, 4)
        assert interval.lower <= interval.point_estimate <= interval.upper
        assert interval.evaluation_unit_count == 48

    def test_a_resample_that_retains_no_unit_is_refused_not_discarded(self) -> None:
        """A sparsely crossed design must fail loudly; conditioning on the well-behaved
        resamples would narrow the interval for a reason that has nothing to do with the data."""

        sparse = [case for case in self._scored() if case.plate_id[-1] == case.compound_id[-1]]
        with pytest.raises(ValueError, match="zero total weight"):
            aggregate_metric(sparse, "sciplex3.marginal-crps-logcp10k", seed=5, resample_count=200)

    def test_the_point_estimate_is_the_equal_unit_mean(self) -> None:
        scored = self._scored()
        interval = aggregate_metric(
            scored, "sciplex3.marginal-crps-logcp10k", seed=5, resample_count=200
        )
        expected = float(
            np.mean([case.values["sciplex3.marginal-crps-logcp10k"] for case in scored])
        )
        assert interval.point_estimate == pytest.approx(expected)

    def test_the_four_dose_diagnostic_weights_compounds_not_wells(self) -> None:
        scored = self._scored()
        grouped = aggregate_metric(
            scored,
            "sciplex3.four-dose-profile-diagnostic",
            seed=5,
            equal_weight_group="compound",
            resample_count=200,
        )
        expected = equal_group_mean(
            [case.compound_id for case in scored],
            [case.values["sciplex3.four-dose-profile-diagnostic"] for case in scored],
        )
        assert grouped.point_estimate == pytest.approx(expected)

    def test_a_missing_metric_is_an_error_and_never_a_silent_drop(self) -> None:
        scored = self._scored(plates=2, compounds=3)
        with pytest.raises(ValueError, match="error_on_missing"):
            aggregate_metric(scored, "sciplex3.not-a-metric", seed=5, resample_count=50)

    def test_a_repeated_evaluation_unit_is_refused(self) -> None:
        scored = self._scored(plates=2, compounds=3)
        with pytest.raises(ValueError, match="double counted"):
            aggregate_metric(
                [*scored, scored[0]],
                "sciplex3.marginal-crps-logcp10k",
                seed=5,
                resample_count=50,
            )
