from __future__ import annotations

import pytest
from conftest import bootstrap_interval_factory

from cellstate.domain import CriterionOutcome, EvaluationStatus
from cellstate.evaluation import empirical_interval_coverage, evaluate_history_information_gain
from cellstate.training import LossKind, default_training_objective


def test_history_information_gain_reports_incomplete_state() -> None:
    report = evaluate_history_information_gain(
        state_only_loss=1.0,
        state_plus_history_loss=0.5,
        tolerance=0.1,
        metric="negative_log_likelihood",
        interval=bootstrap_interval_factory(0.5),
    )
    assert report.evaluation_status is EvaluationStatus.EVALUATED
    assert report.outcome is CriterionOutcome.FAILED
    assert report.history_information_gain == pytest.approx(0.5)
    assert "not shown to be complete" in report.notes[0]


def test_interval_coverage() -> None:
    assert empirical_interval_coverage([0, 2], [-1, 0], [1, 1]) == 0.5
    with pytest.raises(ValueError, match="same length"):
        empirical_interval_coverage([0], [0, 1], [1])
    with pytest.raises(ValueError, match="cannot exceed"):
        empirical_interval_coverage([0], [2], [1])


def test_training_objective_requires_all_components() -> None:
    objective = default_training_objective()
    values = {loss.kind: 1.0 for loss in objective.losses}
    assert objective.combine(values) == pytest.approx(sum(loss.weight for loss in objective.losses))
    values.pop(LossKind.FUNCTIONAL_OUTCOME)
    with pytest.raises(ValueError, match="missing"):
        objective.combine(values)
