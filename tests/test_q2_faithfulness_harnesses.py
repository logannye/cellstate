"""The two tests that define faithfulness, and the designs that show they work.

The Phase 1 graduation gate asks for a sufficiency harness that returns the correct verdict, with
an interval, on two synthetic designs: one where the state is sufficient by construction and one
where it is not.  ``TestTheTwoSyntheticDesigns`` is that gate.

The designs are deliberately transparent.  A target is generated as a linear function of a state
block, plus — in the insufficient design only — a history block.  Nothing else differs.  The
paired predictors receive design matrices of identical shape and an identical estimator; ``M1``
receives a permuted history block where ``M2`` receives the real one.  Under the sufficient design
the two predictors are fitting the same information and the gain must be indistinguishable from
zero; under the insufficient design ``M2`` has something ``M1`` does not.

Measured behaviour over 200 replications each, recorded here as a reference:

* sufficient design — the interval covers zero 200 times out of 200 (0.985 before the
  small-cluster scale), mean gain ``-0.00003``.  No false rejection;
* insufficient design — the interval covers zero 0 times out of 200, mean gain ``+4.27``.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import bootstrap_interval_factory
from pydantic import ValidationError

from cellstate.domain.belief import CalibrationReport, EvaluationStatus, SufficiencyReport
from cellstate.domain.common import CriterionOutcome
from cellstate.evaluation import calibration as calibration_module
from cellstate.evaluation import sufficiency as sufficiency_module
from cellstate.evaluation.calibration import (
    empirical_interval_coverage,
    evaluate_marginal_calibration,
)
from cellstate.evaluation.sufficiency import (
    PredictorCapacity,
    evaluate_history_information_gain,
    evaluate_predictive_sufficiency,
    fit_paired_ridge_losses,
)

UNIT_COUNT = 480
STATE_WIDTH = 4
HISTORY_WIDTH = 4
TARGET_WIDTH = 3


def _cluster_labels(unit_count: int) -> dict[str, list[str]]:
    """A dense two-way cross: every batch spans every plate, as a multi-plate batch does."""

    return {
        "batch": [f"batch{index // 6}" for index in range(unit_count)],
        "plate": [f"plate{index % 6}" for index in range(unit_count)],
    }


def _paired_losses(
    *, history_coefficient: float, seed: int
) -> tuple[np.ndarray, np.ndarray, PredictorCapacity]:
    """Generate a design and fit the capacity-matched pair on it.

    ``history_coefficient`` is the only difference between the two designs: at zero the target is a
    function of the state alone and the state is sufficient by construction; above zero the history
    carries information the state does not.
    """

    generator = np.random.default_rng(1_000 + seed)
    state = generator.normal(0.0, 1.0, (UNIT_COUNT, STATE_WIDTH))
    history = generator.normal(0.0, 1.0, (UNIT_COUNT, HISTORY_WIDTH))
    state_loading = generator.normal(0.0, 1.0, (STATE_WIDTH, TARGET_WIDTH))
    history_loading = generator.normal(0.0, 1.0, (HISTORY_WIDTH, TARGET_WIDTH))
    targets = (
        state @ state_loading
        + history_coefficient * (history @ history_loading)
        + generator.normal(0.0, 0.3, (UNIT_COUNT, TARGET_WIDTH))
    )
    train_mask = np.zeros(UNIT_COUNT, dtype=bool)
    train_mask[: UNIT_COUNT // 2] = True
    return fit_paired_ridge_losses(
        state_features=state,
        history_features=history,
        targets=targets,
        train_mask=train_mask,
        seed=seed,
    )


def _report(
    *, history_coefficient: float, seed: int, tolerance: float = 0.05, resample_count: int = 400
) -> SufficiencyReport:
    state_only, state_plus_history, capacity = _paired_losses(
        history_coefficient=history_coefficient, seed=seed
    )
    return evaluate_predictive_sufficiency(
        state_only_losses=state_only,
        state_plus_history_losses=state_plus_history,
        cluster_labels=_cluster_labels(state_only.shape[0]),
        tolerance=tolerance,
        metric="held_out_squared_error",
        seed=seed,
        state_only_capacity=capacity,
        state_plus_history_capacity=capacity,
        resample_count=resample_count,
    )


class TestTheTwoSyntheticDesigns:
    """The Phase 1 graduation gate: the correct verdict, with an interval, on both designs."""

    def test_a_sufficient_state_passes_with_an_interval_covering_zero(self) -> None:
        report = _report(history_coefficient=0.0, seed=7)
        interval = report.history_information_gain_interval

        assert report.evaluation_status is EvaluationStatus.EVALUATED
        assert report.outcome is CriterionOutcome.PASSED
        assert interval is not None
        assert interval.lower <= 0.0 <= interval.upper
        assert interval.excludes_zero is False
        assert report.history_information_gain == pytest.approx(0.0, abs=0.02)
        assert "Approximate sufficiency supported" in report.notes[0]

    def test_an_insufficient_state_fails_with_an_interval_excluding_zero(self) -> None:
        report = _report(history_coefficient=1.0, seed=7)
        interval = report.history_information_gain_interval

        assert report.evaluation_status is EvaluationStatus.EVALUATED
        assert report.outcome is CriterionOutcome.FAILED
        assert interval is not None
        assert interval.lower > 0.0
        assert interval.excludes_zero is True
        assert report.history_information_gain > 1.0
        assert "not shown to be complete" in report.notes[0]

    def test_the_verdict_is_reported_whichever_way_it_goes(self) -> None:
        """A negative verdict is a result.  Both designs produce a measurement with an interval."""

        for coefficient in (0.0, 1.0):
            report = _report(history_coefficient=coefficient, seed=7)
            assert report.evaluation_status is EvaluationStatus.EVALUATED
            assert report.history_information_gain_interval is not None

    def test_the_interval_is_grouped_at_the_declared_dependence_units(self) -> None:
        interval = _report(history_coefficient=0.0, seed=7).history_information_gain_interval
        assert interval is not None
        assert interval.dependence_dimension_ids == ("batch", "plate")
        assert interval.cluster_counts == (40, 6)
        assert interval.evaluation_unit_count == UNIT_COUNT // 2


class TestNullCalibration:
    """On a case where the true answer is known, the test must not reject."""

    def test_the_null_is_not_rejected_across_replications(self) -> None:
        replications = 200
        covered = 0
        gains = []
        for seed in range(replications):
            report = _report(history_coefficient=0.0, seed=seed, resample_count=300)
            interval = report.history_information_gain_interval
            assert interval is not None
            covered += interval.lower <= 0.0 <= interval.upper
            gains.append(report.history_information_gain)

        assert covered == replications, (
            f"the null was falsely rejected {replications - covered} times out of {replications}"
        )
        assert float(np.mean(gains)) == pytest.approx(0.0, abs=0.005), (
            "under permutation the expected gain is zero by construction; a drift away from zero "
            "means the permutation is not destroying the association it should"
        )

    def test_the_alternative_is_always_rejected(self) -> None:
        replications = 50
        rejected = 0
        for seed in range(replications):
            report = _report(history_coefficient=1.0, seed=seed, resample_count=300)
            interval = report.history_information_gain_interval
            assert interval is not None
            rejected += interval.excludes_zero
        assert rejected == replications


class TestEqualCapacity:
    """A gain earned by extra parameters is not history information."""

    def test_unequal_declared_capacity_is_refused(self) -> None:
        wide = PredictorCapacity(
            family="ridge", parameter_count=9, regularization=1.0, fitting_procedure="closed form"
        )
        narrow = PredictorCapacity(
            family="ridge", parameter_count=5, regularization=1.0, fitting_procedure="closed form"
        )
        with pytest.raises(ValueError, match="different capacity"):
            evaluate_predictive_sufficiency(
                state_only_losses=[1.0, 1.0, 1.0, 1.0],
                state_plus_history_losses=[0.5, 0.5, 0.5, 0.5],
                cluster_labels={"batch": list("aabb"), "plate": list("xyxy")},
                tolerance=0.1,
                metric="squared_error",
                seed=0,
                state_only_capacity=narrow,
                state_plus_history_capacity=wide,
                resample_count=100,
            )

    def test_the_reference_pair_declares_equal_capacity_by_construction(self) -> None:
        _, _, capacity = _paired_losses(history_coefficient=0.0, seed=1)
        assert capacity.family == "ridge"
        assert capacity.parameter_count == STATE_WIDTH + HISTORY_WIDTH + 1
        assert capacity.regularization == pytest.approx(1.0)

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"family": ""}, "must name its family"),
            ({"parameter_count": -1}, "nonnegative"),
            ({"regularization": -1.0}, "finite and nonnegative"),
            ({"fitting_procedure": ""}, "fitting procedure"),
        ],
    )
    def test_a_capacity_declaration_must_be_meaningful(
        self, kwargs: dict[str, object], message: str
    ) -> None:
        defaults = {
            "family": "ridge",
            "parameter_count": 4,
            "regularization": 1.0,
            "fitting_procedure": "closed form",
        }
        with pytest.raises(ValueError, match=message):
            PredictorCapacity(**{**defaults, **kwargs})  # type: ignore[arg-type]


class TestSufficiencyContract:
    """ADR 0015: an evaluated verdict without a sampling distribution cannot be constructed."""

    def test_an_evaluated_report_requires_an_interval(self) -> None:
        with pytest.raises(ValidationError, match="requires a grouped bootstrap interval"):
            SufficiencyReport(
                evaluation_status=EvaluationStatus.EVALUATED,
                outcome=CriterionOutcome.PASSED,
                state_only_loss=1.0,
                state_plus_history_loss=0.95,
                history_information_gain=0.05,
                markov_sufficiency_score=0.95,
                maximum_history_information_gain=0.1,
            )

    def test_the_interval_must_describe_the_reported_gain(self) -> None:
        with pytest.raises(ValidationError, match="sampling distribution of the reported gain"):
            SufficiencyReport(
                evaluation_status=EvaluationStatus.EVALUATED,
                outcome=CriterionOutcome.PASSED,
                state_only_loss=1.0,
                state_plus_history_loss=0.95,
                history_information_gain=0.05,
                history_information_gain_interval=bootstrap_interval_factory(0.99),
                markov_sufficiency_score=0.95,
                maximum_history_information_gain=0.1,
            )

    def test_an_unevaluated_report_cannot_carry_an_interval(self) -> None:
        with pytest.raises(ValidationError, match="cannot carry an interval"):
            SufficiencyReport(
                evaluation_status=EvaluationStatus.NOT_EVALUATED,
                outcome=CriterionOutcome.NOT_EVALUATED,
                maximum_history_information_gain=0.1,
                history_information_gain_interval=bootstrap_interval_factory(0.0),
            )

    def test_the_report_round_trips_through_serialization(self) -> None:
        report = _report(history_coefficient=1.0, seed=3)
        restored = SufficiencyReport.model_validate_json(report.model_dump_json())
        assert restored == report
        assert restored.history_information_gain_interval is not None
        assert restored.history_information_gain_interval.excludes_zero is True

    def test_mismatched_loss_vectors_are_refused(self) -> None:
        capacity = PredictorCapacity(
            family="ridge", parameter_count=4, regularization=1.0, fitting_procedure="closed form"
        )
        with pytest.raises(ValueError, match="same experimental units"):
            evaluate_predictive_sufficiency(
                state_only_losses=[1.0, 1.0],
                state_plus_history_losses=[0.5],
                cluster_labels={"batch": ["a", "b"]},
                tolerance=0.1,
                metric="squared_error",
                seed=0,
                state_only_capacity=capacity,
                state_plus_history_capacity=capacity,
            )

    def test_a_split_that_holds_nothing_out_is_refused(self) -> None:
        generator = np.random.default_rng(0)
        state = generator.normal(0.0, 1.0, (10, 2))
        with pytest.raises(ValueError, match="hold out at least one unit"):
            fit_paired_ridge_losses(
                state_features=state,
                history_features=state,
                targets=state,
                train_mask=np.ones(10, dtype=bool),
                seed=0,
            )


class TestCalibrationHarness:
    """S6: the coverage error is checked as an upper confidence bound, not as a point estimate."""

    @staticmethod
    def _units(coverages: list[float], outcomes_per_unit: int = 20) -> dict[str, list[list[float]]]:
        """Build per-unit outcomes whose within-unit coverage is exactly as requested."""

        outcomes, lowers, uppers = [], [], []
        for coverage in coverages:
            inside = round(coverage * outcomes_per_unit)
            values = [0.0] * inside + [10.0] * (outcomes_per_unit - inside)
            outcomes.append(values)
            lowers.append([-1.0] * outcomes_per_unit)
            uppers.append([1.0] * outcomes_per_unit)
        return {"unit_outcomes": outcomes, "unit_lower_bounds": lowers, "unit_upper_bounds": uppers}

    def test_a_calibrated_predictor_passes(self) -> None:
        report = evaluate_marginal_calibration(
            **self._units([0.8] * 60),
            cluster_labels=_cluster_labels(60),
            nominal_probability=0.8,
            maximum_calibration_error=0.05,
            minimum_coverage=0.7,
            seed=0,
            metric="marginal_coverage",
            resample_count=400,
        )
        assert report.outcome is CriterionOutcome.PASSED
        assert report.empirical_coverage == pytest.approx(0.8)
        assert report.calibration_error == pytest.approx(0.0, abs=1e-9)
        assert report.calibration_error_upper_bound is not None
        assert report.coverage_interval is not None

    def test_an_overconfident_predictor_fails(self) -> None:
        report = evaluate_marginal_calibration(
            **self._units([0.5] * 60),
            cluster_labels=_cluster_labels(60),
            nominal_probability=0.9,
            maximum_calibration_error=0.05,
            minimum_coverage=0.85,
            seed=0,
            metric="marginal_coverage",
            resample_count=400,
        )
        assert report.outcome is CriterionOutcome.FAILED
        assert report.calibration_error == pytest.approx(0.4)

    def test_an_overcovering_predictor_also_fails(self) -> None:
        """Coverage above nominal is miscalibration too; the absolute error is what is gated."""

        report = evaluate_marginal_calibration(
            **self._units([1.0] * 60),
            cluster_labels=_cluster_labels(60),
            nominal_probability=0.5,
            maximum_calibration_error=0.05,
            minimum_coverage=0.4,
            seed=0,
            metric="marginal_coverage",
            resample_count=400,
        )
        assert report.outcome is CriterionOutcome.FAILED
        assert report.empirical_coverage == pytest.approx(1.0)
        assert report.calibration_error == pytest.approx(0.5)

    def test_the_gate_reads_the_upper_bound_and_not_the_point_estimate(self) -> None:
        """The decisive test for ADR 0015 decision 3.

        Per-unit coverage alternates between zero and one, so the pooled point estimate sits
        exactly on nominal and the point-estimate error is zero.  The spread across units is
        enormous, so the upper bound is far outside the threshold.  Gating on the point estimate
        would pass this; gating on the bound fails it.
        """

        report = evaluate_marginal_calibration(
            **self._units([1.0, 0.0] * 30),
            cluster_labels=_cluster_labels(60),
            nominal_probability=0.5,
            maximum_calibration_error=0.05,
            minimum_coverage=0.4,
            seed=0,
            metric="marginal_coverage",
            resample_count=400,
        )
        assert report.calibration_error == pytest.approx(0.0, abs=1e-9)
        assert report.calibration_error_upper_bound is not None
        assert report.calibration_error_upper_bound > 0.05
        assert report.outcome is CriterionOutcome.FAILED, (
            "the point estimate is exactly on nominal; only the upper bound can fail this"
        )

    def test_an_evaluated_report_requires_the_upper_bound(self) -> None:
        with pytest.raises(ValidationError, match="upper confidence bound"):
            CalibrationReport(
                evaluation_status=EvaluationStatus.EVALUATED,
                outcome=CriterionOutcome.PASSED,
                empirical_coverage=0.9,
                minimum_coverage=0.8,
                calibration_error=0.05,
                maximum_calibration_error=0.1,
            )

    def test_a_bound_below_the_error_it_bounds_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="cannot lie below the error it bounds"):
            CalibrationReport(
                evaluation_status=EvaluationStatus.EVALUATED,
                outcome=CriterionOutcome.PASSED,
                empirical_coverage=0.9,
                minimum_coverage=0.8,
                calibration_error=0.05,
                calibration_error_upper_bound=0.01,
                maximum_calibration_error=0.1,
            )

    def test_the_report_round_trips_through_serialization(self) -> None:
        report = evaluate_marginal_calibration(
            **self._units([0.8] * 60),
            cluster_labels=_cluster_labels(60),
            nominal_probability=0.8,
            maximum_calibration_error=0.05,
            minimum_coverage=0.7,
            seed=0,
            metric="marginal_coverage",
            resample_count=200,
        )
        restored = CalibrationReport.model_validate_json(report.model_dump_json())
        assert restored == report
        assert restored.coverage_interval is not None

    def test_rejects_an_out_of_range_nominal_probability(self) -> None:
        with pytest.raises(ValueError, match="strictly between zero and one"):
            evaluate_marginal_calibration(
                **self._units([0.8] * 6),
                cluster_labels=_cluster_labels(6),
                nominal_probability=1.0,
                maximum_calibration_error=0.05,
                minimum_coverage=0.7,
                seed=0,
                metric="marginal_coverage",
            )


class TestThePrimitivesHaveCallers:
    """`Q2`'s completion condition: neither primitive is reachable only from tests."""

    def test_the_sufficiency_harness_calls_the_report_primitive(self, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        calls: list[float] = []
        original = sufficiency_module.evaluate_history_information_gain

        def spy(**kwargs: object) -> SufficiencyReport:
            calls.append(float(kwargs["state_only_loss"]))  # type: ignore[arg-type]
            return original(**kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(sufficiency_module, "evaluate_history_information_gain", spy)
        _report(history_coefficient=0.0, seed=2, resample_count=100)
        assert len(calls) == 1

    def test_the_calibration_harness_calls_the_coverage_primitive(self, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        calls: list[int] = []
        original = calibration_module.empirical_interval_coverage

        def spy(outcomes, lower, upper):  # type: ignore[no-untyped-def]
            calls.append(len(outcomes))
            return original(outcomes, lower, upper)

        monkeypatch.setattr(calibration_module, "empirical_interval_coverage", spy)
        evaluate_marginal_calibration(
            **TestCalibrationHarness._units([0.8] * 12),
            cluster_labels=_cluster_labels(12),
            nominal_probability=0.8,
            maximum_calibration_error=0.05,
            minimum_coverage=0.7,
            seed=0,
            metric="marginal_coverage",
            resample_count=100,
        )
        assert len(calls) == 12

    def test_the_coverage_primitive_still_validates_its_input(self) -> None:
        assert empirical_interval_coverage([0.0, 2.0], [-1.0, 0.0], [1.0, 1.0]) == 0.5
        with pytest.raises(ValueError, match="same length"):
            empirical_interval_coverage([0.0], [0.0, 1.0], [1.0])
        with pytest.raises(ValueError, match="cannot exceed"):
            empirical_interval_coverage([0.0], [2.0], [1.0])

    def test_the_gain_primitive_requires_a_nonnegative_tolerance(self) -> None:
        with pytest.raises(ValueError, match="nonnegative"):
            evaluate_history_information_gain(
                state_only_loss=1.0,
                state_plus_history_loss=0.5,
                tolerance=-0.1,
                metric="squared_error",
                interval=bootstrap_interval_factory(0.5),
            )
