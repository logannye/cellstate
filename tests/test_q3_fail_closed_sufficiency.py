"""`Q3` -- the sufficiency verdict must fail closed on a design it cannot evaluate.

The defect these tests pin is not that the harness computed a wrong number. It computed the right
number: with no history, ``M2`` *is* ``M1``, so the gain really is zero and its interval really is
degenerate at zero. The defect is that the contract read that as sufficiency, and gating on the
interval's upper end -- correct in every other case -- made the reading maximally confident exactly
where there was no evidence at all.

Authorized by ADR 0017.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from conftest import (
    SYNTHETIC_TEST_OPTIONS,
    bootstrap_interval_factory,
    minimal_reference_config,
    observation_factory,
    request_factory,
    subject_factory,
)
from pydantic import ValidationError

from cellstate.domain.belief import CriterionOutcome, EvaluationStatus, SufficiencyReport
from cellstate.domain.events import MissingnessReport, MissingnessStatus
from cellstate.domain.history import CellHistory
from cellstate.evaluation.query_sufficiency import (
    QuerySufficiencyEvaluator,
    admissible_pre_cutoff_observations,
    evaluate_cohort_sufficiency,
    history_presence_for_cohort,
    request_carries_pre_cutoff_evidence,
)
from cellstate.evaluation.sufficiency import (
    PredictorCapacity,
    evaluate_predictive_sufficiency,
    fit_paired_ridge_losses,
    inapplicable_sufficiency_report,
)
from cellstate.ports import SufficiencyEvaluator
from cellstate.reference import LinearGaussianReference

UNIT_COUNT = 240
TOLERANCE = 0.1


def _design(*, seed: int, history_scale: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A design in which history drives the target whenever ``history_scale`` is nonzero."""

    generator = np.random.default_rng(seed)
    state = generator.normal(size=(UNIT_COUNT, 8))
    history = generator.normal(size=(UNIT_COUNT, 4))
    targets = (
        state @ generator.normal(size=(8, 6))
        + history_scale * (history @ generator.normal(size=(4, 6)))
        + 0.1 * generator.normal(size=(UNIT_COUNT, 6))
    )
    return state, history, targets


def _paired_losses(
    *, seed: int, history_scale: float, absent: slice | None = None
) -> tuple[np.ndarray, np.ndarray, PredictorCapacity]:
    state, history, targets = _design(seed=seed, history_scale=history_scale)
    if absent is not None:
        history = history.copy()
        history[absent] = 0.0
    train = np.zeros(UNIT_COUNT, dtype=bool)
    train[: UNIT_COUNT // 2] = True
    return fit_paired_ridge_losses(
        state_features=state,
        history_features=history,
        targets=targets,
        train_mask=train,
        seed=seed,
    )


def _belief(request):
    """A real belief from the reference model; the evaluator ignores it, the port requires it."""

    return LinearGaussianReference(minimal_reference_config()).estimate(
        request, options=SYNTHETIC_TEST_OPTIONS
    )


def _clusters(count: int) -> dict[str, list[str]]:
    return {
        "compound": [f"c{index % 12}" for index in range(count)],
        "plate": [f"p{index % 8}" for index in range(count)],
    }


class TestTheDefect:
    """The exact failure recorded in ADR 0017, pinned so it cannot return."""

    def test_a_design_with_no_history_is_refused_not_passed(self) -> None:
        state_only, state_plus_history, capacity = _paired_losses(seed=3, history_scale=2.0)
        held_out = state_only.size

        report = evaluate_predictive_sufficiency(
            state_only_losses=state_only,
            state_plus_history_losses=state_plus_history,
            history_present=[False] * held_out,
            cluster_labels=_clusters(held_out),
            tolerance=TOLERANCE,
            metric="held_out_squared_error",
            seed=5,
            state_only_capacity=capacity,
            state_plus_history_capacity=capacity,
            resample_count=200,
        )

        assert report.evaluation_status is EvaluationStatus.NOT_EVALUATED
        assert report.outcome is CriterionOutcome.NOT_EVALUATED
        assert report.outcome is not CriterionOutcome.PASSED
        assert report.history_information_gain is None
        assert report.history_information_gain_interval is None
        assert report.retained_unit_fraction is None
        assert any("refused" in note for note in report.notes)
        assert any("inapplicable, not sufficient" in note for note in report.notes)

    def test_the_same_design_with_history_still_reports_insufficiency(self) -> None:
        """The refusal must not be achieved by making the harness refuse everything."""

        state_only, state_plus_history, capacity = _paired_losses(seed=3, history_scale=2.0)
        held_out = state_only.size

        report = evaluate_predictive_sufficiency(
            state_only_losses=state_only,
            state_plus_history_losses=state_plus_history,
            history_present=[True] * held_out,
            cluster_labels=_clusters(held_out),
            tolerance=TOLERANCE,
            metric="held_out_squared_error",
            seed=5,
            state_only_capacity=capacity,
            state_plus_history_capacity=capacity,
            resample_count=200,
        )

        assert report.evaluation_status is EvaluationStatus.EVALUATED
        assert report.outcome is CriterionOutcome.FAILED
        assert report.history_information_gain is not None
        assert report.history_information_gain > TOLERANCE
        assert report.retained_unit_fraction == pytest.approx(1.0)


class TestDilution:
    """Units without a pre-cutoff observation are excluded, not averaged in as zero-gain units."""

    def test_excluded_units_do_not_move_the_gain(self) -> None:
        state_only, state_plus_history, capacity = _paired_losses(seed=7, history_scale=2.0)
        held_out = state_only.size
        clusters = _clusters(held_out)
        present = [index % 3 != 0 for index in range(held_out)]

        excluded = evaluate_predictive_sufficiency(
            state_only_losses=state_only,
            state_plus_history_losses=state_plus_history,
            history_present=present,
            cluster_labels=clusters,
            tolerance=TOLERANCE,
            metric="held_out_squared_error",
            seed=11,
            state_only_capacity=capacity,
            state_plus_history_capacity=capacity,
            resample_count=200,
        )
        kept = np.flatnonzero(np.asarray(present))
        restricted = evaluate_predictive_sufficiency(
            state_only_losses=state_only[kept],
            state_plus_history_losses=state_plus_history[kept],
            history_present=[True] * kept.size,
            cluster_labels={
                dimension: [labels[int(index)] for index in kept]
                for dimension, labels in clusters.items()
            },
            tolerance=TOLERANCE,
            metric="held_out_squared_error",
            seed=11,
            state_only_capacity=capacity,
            state_plus_history_capacity=capacity,
            resample_count=200,
        )

        # Excluding a unit must be identical to never having offered it -- including in the
        # interval, which is why the cluster labels are filtered alongside the losses. Leaving a
        # dropped unit's label in place would let it keep supplying a cluster to resample.
        assert excluded.history_information_gain == pytest.approx(
            restricted.history_information_gain
        )
        assert excluded.history_information_gain_interval is not None
        assert restricted.history_information_gain_interval is not None
        assert excluded.history_information_gain_interval.lower == pytest.approx(
            restricted.history_information_gain_interval.lower
        )
        assert excluded.history_information_gain_interval.upper == pytest.approx(
            restricted.history_information_gain_interval.upper
        )

    def test_the_retained_fraction_is_reported(self) -> None:
        state_only, state_plus_history, capacity = _paired_losses(seed=7, history_scale=2.0)
        held_out = state_only.size
        present = [index % 4 != 0 for index in range(held_out)]

        report = evaluate_predictive_sufficiency(
            state_only_losses=state_only,
            state_plus_history_losses=state_plus_history,
            history_present=present,
            cluster_labels=_clusters(held_out),
            tolerance=TOLERANCE,
            metric="held_out_squared_error",
            seed=11,
            state_only_capacity=capacity,
            state_plus_history_capacity=capacity,
            resample_count=200,
        )

        assert report.retained_unit_fraction == pytest.approx(sum(present) / held_out)
        assert any("were excluded rather than averaged in" in note for note in report.notes)

    def test_presence_must_be_declared_for_every_unit(self) -> None:
        state_only, state_plus_history, capacity = _paired_losses(seed=7, history_scale=2.0)
        with pytest.raises(ValueError, match="exactly the units the losses cover"):
            evaluate_predictive_sufficiency(
                state_only_losses=state_only,
                state_plus_history_losses=state_plus_history,
                history_present=[True],
                cluster_labels=_clusters(state_only.size),
                tolerance=TOLERANCE,
                metric="held_out_squared_error",
                seed=11,
                state_only_capacity=capacity,
                state_plus_history_capacity=capacity,
            )


class TestTheContract:
    """A report that omits the retained fraction, or claims none, must be unrepresentable."""

    def test_an_evaluated_report_requires_the_retained_fraction(self) -> None:
        with pytest.raises(ValidationError, match="admissible pre-cutoff observation"):
            SufficiencyReport(
                evaluation_status=EvaluationStatus.EVALUATED,
                outcome=CriterionOutcome.PASSED,
                state_only_loss=1.0,
                state_plus_history_loss=0.95,
                history_information_gain=0.05,
                markov_sufficiency_score=0.95,
                maximum_history_information_gain=0.1,
                history_information_gain_interval=bootstrap_interval_factory(0.05),
                metric="squared_error",
            )

    def test_a_retained_fraction_of_zero_is_unrepresentable(self) -> None:
        with pytest.raises(ValidationError):
            SufficiencyReport(
                evaluation_status=EvaluationStatus.EVALUATED,
                outcome=CriterionOutcome.PASSED,
                state_only_loss=1.0,
                state_plus_history_loss=0.95,
                history_information_gain=0.05,
                markov_sufficiency_score=0.95,
                maximum_history_information_gain=0.1,
                history_information_gain_interval=bootstrap_interval_factory(0.05),
                retained_unit_fraction=0.0,
                metric="squared_error",
            )

    def test_a_refused_report_carries_neither_interval_nor_fraction(self) -> None:
        report = inapplicable_sufficiency_report(reason="no pre-cutoff evidence", tolerance=0.1)
        assert report.retained_unit_fraction is None
        assert report.history_information_gain_interval is None
        with pytest.raises(ValidationError, match="interval or a retained fraction"):
            report.model_copy(update={"retained_unit_fraction": 0.5}).model_validate(
                report.model_dump(mode="python") | {"retained_unit_fraction": 0.5}
            )

    def test_a_refusal_must_name_its_reason(self) -> None:
        with pytest.raises(ValueError, match="must name its reason"):
            inapplicable_sufficiency_report(reason="", tolerance=0.1)

    def test_the_retained_fraction_round_trips(self) -> None:
        report = SufficiencyReport(
            evaluation_status=EvaluationStatus.EVALUATED,
            outcome=CriterionOutcome.PASSED,
            state_only_loss=1.0,
            state_plus_history_loss=0.95,
            history_information_gain=0.05,
            markov_sufficiency_score=0.95,
            maximum_history_information_gain=0.1,
            history_information_gain_interval=bootstrap_interval_factory(0.05, half_width=0.04),
            retained_unit_fraction=0.625,
            metric="squared_error",
        )
        restored = SufficiencyReport.model_validate(json.loads(report.model_dump_json()))
        assert restored.retained_unit_fraction == pytest.approx(0.625)
        assert restored == report


class TestTheApplicabilityPredicate:
    """S1, per unit, computed from the request rather than declared beside it."""

    def _request_with(self, *, time_seconds: float, status: MissingnessStatus):
        subject = subject_factory()
        observation = observation_factory(
            subject=subject,
            time_seconds=time_seconds,
            missingness=MissingnessReport(status=status),
            value=None if status is not MissingnessStatus.OBSERVED else 0.5,
            units=None if status is not MissingnessStatus.OBSERVED else "relative",
        )
        return request_factory(
            history=CellHistory(subject=subject, events=(observation,)),
            as_of_seconds=10,
        )

    def test_an_observed_pre_cutoff_measurement_admits_the_comparison(self) -> None:
        request = self._request_with(time_seconds=2, status=MissingnessStatus.OBSERVED)
        assert request_carries_pre_cutoff_evidence(request)
        assert len(admissible_pre_cutoff_observations(request)) == 1

    def test_the_request_contract_already_forbids_a_post_cutoff_observation(self) -> None:
        """So every event a valid request carries is pre-cutoff by construction.

        This is worth pinning rather than assuming: the predicate's ``through(as_of_seconds)``
        filter looks redundant against a valid request, and a later relaxation of this contract
        would silently make it load-bearing again. If this test ever fails, the filter is the
        only thing standing between a post-cutoff readout and the history block.
        """

        subject = subject_factory()
        with pytest.raises(ValidationError, match="events after as_of_seconds"):
            request_factory(
                history=CellHistory(
                    subject=subject,
                    events=(observation_factory(subject=subject, time_seconds=50),),
                ),
                as_of_seconds=10,
            )

    @pytest.mark.parametrize(
        "status",
        [
            MissingnessStatus.NOT_MEASURED,
            MissingnessStatus.MISSING,
            MissingnessStatus.ASSAY_FAILURE,
        ],
    )
    def test_an_unobserved_pre_cutoff_measurement_does_not(self, status: MissingnessStatus) -> None:
        # A slot in the timeline is not an observation. Counting one would let a unit that was
        # never measured supply history, which biases the gain toward zero -- toward sufficiency.
        assert not request_carries_pre_cutoff_evidence(
            self._request_with(time_seconds=2, status=status)
        )

    def test_an_empty_history_does_not(self) -> None:
        subject = subject_factory()
        request = request_factory(history=CellHistory(subject=subject), as_of_seconds=10)
        assert not request_carries_pre_cutoff_evidence(request)

    def test_the_cohort_mask_is_ordered_with_the_requests(self) -> None:
        present = self._request_with(time_seconds=2, status=MissingnessStatus.OBSERVED)
        absent = request_factory(history=CellHistory(subject=subject_factory()), as_of_seconds=10)
        assert history_presence_for_cohort([present, absent, present]) == (True, False, True)

    def test_an_empty_cohort_is_refused(self) -> None:
        with pytest.raises(ValueError, match="at least one request"):
            history_presence_for_cohort([])


class TestTheCaller:
    """`Q2`'s unmet done-when: the harness has a caller outside the tests."""

    def test_the_evaluator_satisfies_the_declared_port(self) -> None:
        evaluator: SufficiencyEvaluator = QuerySufficiencyEvaluator(tolerance=0.1)
        assert evaluator is not None

    def test_a_query_without_pre_cutoff_evidence_is_refused_by_evidence(self) -> None:
        subject = subject_factory()
        request = request_factory(history=CellHistory(subject=subject), as_of_seconds=10)
        report = QuerySufficiencyEvaluator(tolerance=0.1).evaluate(_belief(request), request)
        assert report.evaluation_status is EvaluationStatus.NOT_EVALUATED
        assert any("no observed measurement at or before" in note for note in report.notes)

    def test_a_query_with_pre_cutoff_evidence_is_refused_for_a_different_reason(self) -> None:
        estimate_request = request_factory()
        report = QuerySufficiencyEvaluator(tolerance=0.1).evaluate(
            _belief(estimate_request), estimate_request
        )
        assert report.evaluation_status is EvaluationStatus.NOT_EVALUATED
        # The distinction is the point: "cannot be asked" and "not yet asked over a cohort" are
        # different facts, and collapsing them is how a refusal becomes decorative.
        assert any("a single request supplies one unit" in note for note in report.notes)
        assert not any("no observed measurement at or before" in note for note in report.notes)

    def test_the_cohort_caller_derives_presence_from_the_requests(self) -> None:
        state_only, state_plus_history, capacity = _paired_losses(seed=13, history_scale=2.0)
        held_out = state_only.size
        subject = subject_factory()
        observed = request_factory(
            history=CellHistory(
                subject=subject,
                events=(observation_factory(subject=subject, time_seconds=2),),
            ),
            as_of_seconds=10,
        )
        unobserved = request_factory(history=CellHistory(subject=subject), as_of_seconds=10)
        # Drop on a stride coprime to both cluster dimensions (12 compounds, 8 plates). Dropping
        # on a stride that divides one of them empties whole clusters, and the bootstrap then
        # correctly refuses the design -- a fixture artifact, not the behaviour under test.
        requests = [unobserved if index % 7 == 0 else observed for index in range(held_out)]

        report = evaluate_cohort_sufficiency(
            requests=requests,
            state_only_losses=state_only,
            state_plus_history_losses=state_plus_history,
            cluster_labels=_clusters(held_out),
            tolerance=TOLERANCE,
            metric="held_out_squared_error",
            seed=17,
            state_only_capacity=capacity,
            state_plus_history_capacity=capacity,
            resample_count=200,
        )

        assert report.evaluation_status is EvaluationStatus.EVALUATED
        expected = sum(1 for index in range(held_out) if index % 7) / held_out
        assert report.retained_unit_fraction == pytest.approx(expected)

    def test_a_cohort_of_units_with_no_evidence_is_refused(self) -> None:
        state_only, state_plus_history, capacity = _paired_losses(seed=13, history_scale=2.0)
        held_out = state_only.size
        subject = subject_factory()
        unobserved = request_factory(history=CellHistory(subject=subject), as_of_seconds=10)

        report = evaluate_cohort_sufficiency(
            requests=[unobserved] * held_out,
            state_only_losses=state_only,
            state_plus_history_losses=state_plus_history,
            cluster_labels=_clusters(held_out),
            tolerance=TOLERANCE,
            metric="held_out_squared_error",
            seed=17,
            state_only_capacity=capacity,
            state_plus_history_capacity=capacity,
            resample_count=200,
        )

        assert report.evaluation_status is EvaluationStatus.NOT_EVALUATED
        assert report.outcome is not CriterionOutcome.PASSED

    def test_the_cohort_must_supply_one_request_per_unit(self) -> None:
        state_only, state_plus_history, capacity = _paired_losses(seed=13, history_scale=2.0)
        with pytest.raises(ValueError, match="one request per unit"):
            evaluate_cohort_sufficiency(
                requests=[request_factory()],
                state_only_losses=state_only,
                state_plus_history_losses=state_plus_history,
                cluster_labels=_clusters(state_only.size),
                tolerance=TOLERANCE,
                metric="held_out_squared_error",
                seed=17,
                state_only_capacity=capacity,
                state_plus_history_capacity=capacity,
            )
