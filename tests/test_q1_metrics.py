"""Numerical references for the frozen metric suite's computations.

Every reference here is derived independently of the implementation: a closed form, an identity
that holds by construction, or arithmetic small enough to check by hand.  No expected value in
this file was produced by running the code it tests.

The closed forms used are:

* the CRPS of a normal predictive distribution, ``sigma * (w (2 Phi(w) - 1) + 2 phi(w) -
  1 / sqrt(pi))`` for ``w = (y - mu) / sigma`` (Gneiting and Raftery, 2007);
* the energy score reduces exactly to the CRPS in one dimension, since the Euclidean norm is the
  absolute value there;
* for a degenerate predictive distribution concentrated at ``c``, the pairwise term vanishes and
  the energy score is ``||c - y||_2``;
* for ``X ~ N(0, 1)`` and ``y = 0``, ``E|X - y| = sqrt(2 / pi)`` and ``E|X - X'| = 2 / sqrt(pi)``,
  so the score is ``sqrt(2 / pi) - 1 / sqrt(pi)``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats

from cellstate.evaluation.metrics import (
    METRIC_IMPLEMENTATIONS,
    central_interval,
    differential_expression_weighted_rmse,
    effect_rank_agreement,
    energy_score,
    equal_group_mean,
    marginal_coverage_error,
    marginal_interval_coverage,
    marginal_interval_width,
    profile_rmse,
    sample_crps,
)


def normal_crps(observation: float, mean: float, standard_deviation: float) -> float:
    """The closed-form CRPS of ``N(mean, standard_deviation)`` at ``observation``."""

    standardized = (observation - mean) / standard_deviation
    return standard_deviation * (
        standardized * (2.0 * float(stats.norm.cdf(standardized)) - 1.0)
        + 2.0 * float(stats.norm.pdf(standardized))
        - 1.0 / math.sqrt(math.pi)
    )


class TestSampleCrps:
    @pytest.mark.parametrize(
        ("mean", "standard_deviation", "observation"),
        [(0.0, 1.0, 0.0), (0.0, 1.0, 1.5), (2.0, 0.5, 1.0), (-1.0, 2.0, 3.0)],
    )
    def test_converges_to_the_closed_form(
        self, mean: float, standard_deviation: float, observation: float
    ) -> None:
        generator = np.random.default_rng(0)
        samples = generator.normal(mean, standard_deviation, (200_000, 1))
        assert sample_crps([[observation]], samples) == pytest.approx(
            normal_crps(observation, mean, standard_deviation), rel=0.01
        )

    def test_the_pairwise_term_is_unbiased_on_a_hand_computable_case(self) -> None:
        # samples 0, 1, 3; observation 1.
        # E|X - y| = (1 + 0 + 2) / 3 = 1.
        # sum_{i<j} |x_i - x_j| = 1 + 3 + 2 = 6; unbiased E|X - X'| = 2 * 6 / (3 * 2) = 2.
        # CRPS = 1 - 0.5 * 2 = 0.
        assert sample_crps([[1.0]], [[0.0], [1.0], [3.0]]) == pytest.approx(0.0)

    def test_a_point_prediction_reduces_to_absolute_error(self) -> None:
        samples = np.full((7, 1), 2.0)
        assert sample_crps([[5.0]], samples) == pytest.approx(3.0)

    def test_averages_over_features_and_observations(self) -> None:
        samples = np.array([[0.0, 10.0], [0.0, 10.0]])
        # Both features are point predictions: |1 - 0| = 1 and |13 - 10| = 3, averaged to 2.
        assert sample_crps([[1.0, 13.0]], samples) == pytest.approx(2.0)
        # A second observation at the prediction halves each feature's error.
        assert sample_crps([[1.0, 13.0], [0.0, 10.0]], samples) == pytest.approx(1.0)

    def test_prefers_a_calibrated_predictive_to_an_underdispersed_one(self) -> None:
        """The property that motivates the unbiased pairwise term.

        The biased ``1 / m^2`` estimator shrinks the spread term, rewarding a predictive
        distribution that is too narrow.  Averaged over draws from the truth, the calibrated
        predictive must score better.
        """

        generator = np.random.default_rng(7)
        truth = generator.normal(0.0, 1.0, (400, 1))
        calibrated = generator.normal(0.0, 1.0, (512, 1))
        underdispersed = generator.normal(0.0, 0.25, (512, 1))
        assert sample_crps(truth, calibrated) < sample_crps(truth, underdispersed)

    def test_rejects_a_missing_observation(self) -> None:
        with pytest.raises(ValueError, match="a missing target is an error"):
            sample_crps(np.empty((0, 3)), np.zeros((4, 3)))

    def test_rejects_too_few_samples_and_mismatched_features(self) -> None:
        with pytest.raises(ValueError, match="at least 2 predictive samples"):
            sample_crps([[1.0]], [[1.0]])
        with pytest.raises(ValueError, match="share a feature axis"):
            sample_crps([[1.0, 2.0]], np.zeros((4, 3)))

    def test_rejects_nonfinite_input(self) -> None:
        with pytest.raises(ValueError, match="finite"):
            sample_crps([[np.nan]], np.zeros((4, 1)))
        with pytest.raises(ValueError, match="finite"):
            sample_crps([[1.0]], np.array([[0.0], [np.inf]]))


class TestEnergyScore:
    def test_equals_the_crps_in_one_dimension(self) -> None:
        generator = np.random.default_rng(1)
        samples = generator.normal(0.0, 1.0, (4_000, 1))
        assert energy_score([[0.0]], samples) == pytest.approx(
            sample_crps([[0.0]], samples), rel=1e-12
        )

    def test_a_degenerate_predictive_reduces_to_the_euclidean_distance(self) -> None:
        samples = np.tile([[1.0, 2.0, 3.0]], (5, 1))
        assert energy_score([[0.0, 0.0, 0.0]], samples) == pytest.approx(math.sqrt(14.0))

    def test_matches_the_standard_normal_closed_form(self) -> None:
        generator = np.random.default_rng(2)
        samples = generator.normal(0.0, 1.0, (100_000, 1))
        expected = math.sqrt(2.0 / math.pi) - 1.0 / math.sqrt(math.pi)
        assert energy_score([[0.0]], samples) == pytest.approx(expected, rel=0.01)

    def test_is_sensitive_to_dependence_the_marginal_score_cannot_see(self) -> None:
        """Two predictives whose marginals are identical by construction.

        Reversing one column is a permutation, so each column's empirical distribution is
        unchanged and the marginal CRPS is identical up to summation order.  Only the dependence
        differs, and only the energy score can see it.
        """

        generator = np.random.default_rng(3)
        ordered = np.sort(generator.normal(0.0, 1.0, 2_000))
        correlated = np.column_stack([ordered, ordered])
        anticorrelated = np.column_stack([ordered, ordered[::-1]])
        observation = [[1.0, 1.0]]

        assert sample_crps(observation, correlated) == pytest.approx(
            sample_crps(observation, anticorrelated), rel=1e-12
        )
        assert energy_score(observation, correlated) < energy_score(observation, anticorrelated)


class TestIntervals:
    def test_central_interval_is_the_equal_tailed_sample_quantile(self) -> None:
        samples = np.arange(101, dtype=float).reshape(-1, 1)
        lower, upper = central_interval(samples, probability=0.8)
        assert lower == pytest.approx([10.0])
        assert upper == pytest.approx([90.0])

    def test_coverage_counts_inclusively(self) -> None:
        lower = np.array([0.0, 0.0])
        upper = np.array([1.0, 1.0])
        assert marginal_interval_coverage([[0.0, 1.0]], lower, upper) == pytest.approx(1.0)
        assert marginal_interval_coverage([[0.5, 2.0]], lower, upper) == pytest.approx(0.5)
        assert marginal_interval_coverage([[-1.0, 2.0]], lower, upper) == pytest.approx(0.0)

    def test_coverage_error_is_absolute_in_both_directions(self) -> None:
        lower = np.array([0.0])
        upper = np.array([1.0])
        over = marginal_coverage_error([[0.5]], lower, upper, nominal_probability=0.5)
        under = marginal_coverage_error([[9.0]], lower, upper, nominal_probability=0.5)
        assert over == pytest.approx(0.5)
        assert under == pytest.approx(0.5)

    def test_width_is_the_mean_per_feature_width(self) -> None:
        assert marginal_interval_width([0.0, 1.0], [2.0, 5.0]) == pytest.approx(3.0)

    def test_an_interval_can_reach_coverage_by_being_uninformative(self) -> None:
        """Why width is reported beside coverage rather than instead of it."""

        observations = [[0.4, 0.6]]
        tight = (np.array([0.0, 0.0]), np.array([1.0, 1.0]))
        loose = (np.array([-100.0, -100.0]), np.array([100.0, 100.0]))
        assert marginal_interval_coverage(observations, *tight) == pytest.approx(1.0)
        assert marginal_interval_coverage(observations, *loose) == pytest.approx(1.0)
        assert marginal_interval_width(*loose) > marginal_interval_width(*tight)

    @pytest.mark.parametrize(
        ("lower", "upper", "message"),
        [
            ([1.0], [0.0], "cannot exceed"),
            ([0.0, 0.0], [1.0], "same shape"),
        ],
    )
    def test_rejects_impossible_bounds(
        self, lower: list[float], upper: list[float], message: str
    ) -> None:
        with pytest.raises(ValueError, match=message):
            marginal_interval_width(lower, upper)

    def test_rejects_an_out_of_range_probability(self) -> None:
        with pytest.raises(ValueError, match="strictly between zero and one"):
            central_interval(np.zeros((4, 1)), probability=1.0)
        with pytest.raises(ValueError, match="strictly between zero and one"):
            marginal_coverage_error([[0.0]], [0.0], [1.0], nominal_probability=0.0)

    def test_rejects_bounds_that_do_not_match_the_observations(self) -> None:
        with pytest.raises(ValueError, match="share a feature axis"):
            marginal_interval_coverage([[0.0, 0.0]], [0.0], [1.0])

    def test_rejects_nonfinite_or_empty_bounds(self) -> None:
        with pytest.raises(ValueError, match="finite"):
            marginal_interval_width([0.0], [np.inf])
        with pytest.raises(ValueError, match="at least one interval"):
            marginal_interval_width([], [])


class TestEffectMetrics:
    def test_profile_rmse_is_hand_computable(self) -> None:
        assert profile_rmse([1.0, 2.0, 3.0], [1.0, 2.0, 5.0]) == pytest.approx(math.sqrt(4.0 / 3.0))

    def test_profile_rmse_is_zero_only_on_an_exact_match(self) -> None:
        assert profile_rmse([1.0, -2.0], [1.0, -2.0]) == pytest.approx(0.0)

    def test_equal_group_mean_gives_every_group_the_same_weight(self) -> None:
        # Group "a" contributes (1 + 3) / 2 = 2 despite having twice the members of "b".
        assert equal_group_mean(["a", "a", "b"], [1.0, 3.0, 10.0]) == pytest.approx(6.0)
        assert equal_group_mean(["a", "b"], [2.0, 10.0]) == pytest.approx(6.0)

    def test_differential_expression_weighting_concentrates_the_score(self) -> None:
        # Error of 1 on the first feature only, weights 3 and 1: sqrt(3/4).
        assert differential_expression_weighted_rmse(
            [0.0, 0.0], [1.0, 0.0], weights=[3.0, 1.0]
        ) == pytest.approx(math.sqrt(0.75))
        # Uniform weights reproduce the unweighted RMSE.
        assert differential_expression_weighted_rmse(
            [0.0, 0.0], [1.0, 0.0], weights=[1.0, 1.0]
        ) == pytest.approx(profile_rmse([0.0, 0.0], [1.0, 0.0]))

    def test_differential_expression_weighting_punishes_the_no_change_prediction(self) -> None:
        """Unweighted error is minimized by predicting nothing when most features do not move."""

        observed = np.zeros(100)
        observed[:5] = 3.0
        no_change = np.zeros(100)
        informed = observed.copy()
        weights = np.zeros(100)
        weights[:5] = 1.0
        assert differential_expression_weighted_rmse(
            no_change, observed, weights=weights
        ) == pytest.approx(3.0)
        assert differential_expression_weighted_rmse(
            informed, observed, weights=weights
        ) == pytest.approx(0.0)

    def test_the_four_dose_diagnostic_composes_rmse_then_equal_compound_weighting(self) -> None:
        """``equal_compound_mean(rmse_dose(...))`` end to end on a hand-computable case.

        Compound A is measured at two doses and compound B at one.  Equal-compound weighting must
        give B the same say as A despite A contributing twice the doses.
        """

        observed = {
            ("A", 1.0): [0.0, 0.0],
            ("A", 10.0): [0.0, 0.0],
            ("B", 1.0): [0.0, 0.0],
        }
        predicted = {
            ("A", 1.0): [1.0, 1.0],  # RMSE 1
            ("A", 10.0): [3.0, 3.0],  # RMSE 3
            ("B", 1.0): [10.0, 10.0],  # RMSE 10
        }
        keys = list(observed)
        per_dose = [profile_rmse(predicted[key], observed[key]) for key in keys]
        compounds = [compound for compound, _ in keys]

        assert per_dose == pytest.approx([1.0, 3.0, 10.0])
        # Equal-compound mean: ((1 + 3) / 2 + 10) / 2 = 6.  A per-dose mean would give 14 / 3.
        assert equal_group_mean(compounds, per_dose) == pytest.approx(6.0)
        assert equal_group_mean(compounds, per_dose) != pytest.approx(sum(per_dose) / len(per_dose))

    def test_rank_agreement_spans_perfect_agreement_to_perfect_disagreement(self) -> None:
        assert effect_rank_agreement([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(0.0)
        assert effect_rank_agreement([1.0, 2.0, 3.0], [3.0, 2.0, 1.0]) == pytest.approx(2.0)

    def test_rank_agreement_ignores_monotone_rescaling(self) -> None:
        predicted = [1.0, 2.0, 3.0, 4.0]
        assert effect_rank_agreement(predicted, [10.0, 200.0, 3_000.0, 40_000.0]) == (
            pytest.approx(effect_rank_agreement(predicted, predicted))
        )

    def test_rank_agreement_is_undefined_on_a_constant_profile(self) -> None:
        with pytest.raises(ValueError, match="constant effect profile"):
            effect_rank_agreement([1.0, 2.0, 3.0], [5.0, 5.0, 5.0])

    @pytest.mark.parametrize(
        ("call", "message"),
        [
            (lambda: profile_rmse([1.0], [1.0, 2.0]), "same shape"),
            (lambda: profile_rmse([], []), "at least one feature"),
            (lambda: profile_rmse([np.nan], [1.0]), "finite"),
            (lambda: equal_group_mean(["a"], [1.0, 2.0]), "one value is required"),
            (lambda: equal_group_mean(["a"], [np.inf]), "finite"),
            (
                lambda: differential_expression_weighted_rmse([1.0], [1.0], weights=[-1.0]),
                "nonnegative",
            ),
            (
                lambda: differential_expression_weighted_rmse([1.0], [1.0], weights=[0.0]),
                "sum to zero",
            ),
            (
                lambda: differential_expression_weighted_rmse([1.0], [1.0], weights=[1.0, 1.0]),
                "same shape",
            ),
            (lambda: effect_rank_agreement([1.0], [1.0]), "at least two features"),
            (lambda: effect_rank_agreement([1.0, np.nan], [1.0, 2.0]), "finite"),
        ],
    )
    def test_rejects_invalid_input(self, call: object, message: str) -> None:
        with pytest.raises(ValueError, match=message):
            call()  # type: ignore[operator]


class TestRegistry:
    def test_every_binding_names_itself_consistently(self) -> None:
        for metric_id, binding in METRIC_IMPLEMENTATIONS.items():
            assert binding.metric_id == metric_id
            assert binding.implementation_id == f"cellstate.metric.{metric_id}"
            assert binding.implementation_version == "1.0.0"
            assert binding.direction == "minimize"
            assert callable(binding.computation)

    def test_parameterized_metrics_carry_their_nominal_probability(self) -> None:
        for level in (0.50, 0.80, 0.95):
            suffix = f"p{int(level * 100)}"
            for family in ("marginal-coverage-error", "marginal-interval-width"):
                binding = METRIC_IMPLEMENTATIONS[f"sciplex3.{family}-{suffix}"]
                assert binding.parameters["nominal_probability"] == pytest.approx(level)

    def test_binding_parameters_cannot_be_mutated(self) -> None:
        binding = METRIC_IMPLEMENTATIONS["sciplex3.marginal-coverage-error-p50"]
        with pytest.raises(TypeError):
            binding.parameters["nominal_probability"] = 0.99  # type: ignore[index]
