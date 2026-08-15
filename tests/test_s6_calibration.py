"""S6: the first readiness criterion this backend actually evaluates.

Every number the model card and ADR 0024 quote for S6 is pinned here.  The point of the suite is
not that the coverage is 0.8836 -- it is that the belief now carries a criterion that **came out
FAILED**, on thresholds written before the number existed, and that the fields downstream of it
move when it moves.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from cellstate.backends.gse274113.arm_request import S6_NOMINAL_PROBABILITY, arm_query
from cellstate.backends.gse274113.estimator import GSE274113ObservationEstimator
from cellstate.backends.gse274113.fit import ArmSlice, fit_fold
from cellstate.backends.gse274113.usage import estimate_arm
from cellstate.domain.common import CriterionOutcome
from cellstate.evaluation.gse274113_reports import (
    calibration_shape_diagnostics,
    measure_calibration_coverage,
    replicate_standard_scores,
)

ROOT = Path(__file__).resolve().parents[1]
SLICE_PATH = ROOT / "backends/vertical-a/gse274113-rna-obs-v1/arms.json"

MINIMUM_COVERAGE = 0.85
MAXIMUM_CALIBRATION_ERROR = 0.05


@pytest.fixture(scope="module")
def arm_slice() -> ArmSlice:
    return ArmSlice.from_payload(json.loads(SLICE_PATH.read_text(encoding="utf-8")))


def _report(arm_slice: ArmSlice, **overrides: float):
    settings = {
        "minimum_coverage": MINIMUM_COVERAGE,
        "maximum_calibration_error": MAXIMUM_CALIBRATION_ERROR,
    }
    settings.update(overrides)
    return measure_calibration_coverage(arm_slice, **settings)  # type: ignore[arg-type]


# ------------------------------------------------------------------ the measurement


def test_s6_fails_against_thresholds_written_before_the_number_existed(
    arm_slice: ArmSlice,
) -> None:
    """The published S6 figures, and the verdict they support."""

    report = _report(arm_slice)
    assert report.empirical_coverage == pytest.approx(0.8836, abs=5e-4)
    assert report.calibration_error == pytest.approx(0.0164, abs=5e-4)
    assert report.calibration_error_upper_bound == pytest.approx(0.0548, abs=5e-4)
    assert report.coverage_interval.lower == pytest.approx(0.8452, abs=5e-4)
    assert report.coverage_interval.upper == pytest.approx(0.9220, abs=5e-4)
    assert report.outcome is CriterionOutcome.FAILED


def test_the_point_estimate_passes_and_the_bound_is_what_fails(arm_slice: ArmSlice) -> None:
    """S6 fails on the bound alone, which is the whole reason the harness gates on the bound.

    Coverage 0.8836 clears the 0.85 floor and its error 0.0164 is comfortably inside 0.05.  Read as
    a point estimate this is a pass.  The interval reaches 0.8452, so the error could plausibly be
    0.0548, and a criterion that reports a point estimate would have called this calibrated.
    """

    report = _report(arm_slice)
    assert report.empirical_coverage >= MINIMUM_COVERAGE
    assert report.calibration_error <= MAXIMUM_CALIBRATION_ERROR
    assert report.calibration_error_upper_bound > MAXIMUM_CALIBRATION_ERROR
    assert report.outcome is CriterionOutcome.FAILED


def test_the_verdict_does_not_depend_on_the_bootstrap_seed(arm_slice: ArmSlice) -> None:
    """0.0548 against 0.05 is a thin margin.  A verdict that thin must be shown to be stable."""

    bounds = []
    for seed in range(20260813, 20260813 + 8):
        report = measure_calibration_coverage(
            arm_slice,
            minimum_coverage=MINIMUM_COVERAGE,
            maximum_calibration_error=MAXIMUM_CALIBRATION_ERROR,
            seed=seed,
        )
        assert report.outcome is CriterionOutcome.FAILED
        bounds.append(report.calibration_error_upper_bound)
    assert min(bounds) > MAXIMUM_CALIBRATION_ERROR
    assert max(bounds) - min(bounds) < 0.005


def test_the_nominal_is_forced_by_the_predeclared_pair() -> None:
    """0.90 is the only nominal at which the two shipped thresholds are mutually consistent.

    This is what keeps S6 from being a bound supplied by the query -- the failure mode already
    present in `maximum_ood_score=0.99`.  Had the nominal been free, 0.95 would have made the
    measured 0.8836 fail the floor outright and 0.85 would have made it pass, and either could have
    been justified after the fact.
    """

    thresholds = arm_query(("GATA1",), model_fingerprint="0" * 64).acceptance_thresholds
    assert thresholds.minimum_calibration_coverage == MINIMUM_COVERAGE
    assert thresholds.maximum_calibration_error == MAXIMUM_CALIBRATION_ERROR
    consistent = [
        candidate / 100
        for candidate in range(1, 100)
        if candidate / 100 - thresholds.maximum_calibration_error
        >= thresholds.minimum_calibration_coverage - 1e-12
        and candidate / 100 + thresholds.maximum_calibration_error <= 1.0 + 1e-12
    ]
    assert min(consistent) == pytest.approx(S6_NOMINAL_PROBABILITY)
    assert S6_NOMINAL_PROBABILITY == 0.90


# ------------------------------------------------------------------ S6 is not S2 restated


def test_two_percent_of_the_outcomes_carry_the_whole_s2_failure(arm_slice: ArmSlice) -> None:
    """The decomposition that changes what the repair should be.

    S2's 0.8415 reads as "the interval is uniformly about 16 percent too narrow", and the repair
    that reading implies is a larger ``psi^2``.  Trimming says otherwise: 28 of 1,400 gene-library
    outcomes carry the entire shortfall, and with them removed the spread is exactly earned.
    Inflating ``psi^2`` would push the other 98 percent into over-coverage while still falling
    nine standard deviations short of the outliers.
    """

    shape = calibration_shape_diagnostics(arm_slice)
    assert shape.standard_deviation == pytest.approx(1.2848, abs=5e-4)
    assert shape.trimmed_standard_deviation == pytest.approx(1.0045, abs=5e-4)
    assert shape.largest_absolute_score == pytest.approx(9.47, abs=0.01)
    # An RMS ratio is 1/sd when the residuals are standardized, so the trimmed sd IS the trimmed
    # S2. Untrimmed it fails; trimmed by 2% it is 1.00 to within half a percent.
    assert 1.0 / shape.standard_deviation == pytest.approx(0.7784, abs=5e-4)
    assert 1.0 / shape.trimmed_standard_deviation == pytest.approx(0.9955, abs=5e-4)


def test_the_measured_coverage_is_not_what_a_uniform_rescaling_would_give(
    arm_slice: ArmSlice,
) -> None:
    """If the shortfall were a uniform scale error, coverage at 0.90 would be 0.7996.  It is 0.8836.

    The gap is the whole reason to count coverage on evidence a ratio has already scored: the two
    statistics disagree about the bulk of the panel, and the ratio is the one that is misleading.

    The reference here is ``sd(z)``, the spread of the standardized residuals, which is the scale a
    "uniform rescaling" would mean on this axis.  Aggregating the way S2 does instead -- RMS of
    claimed spread over RMS of realized error, 0.8415 -- implies 0.8337, because pooling numerator
    and denominator separately is not the same as standardizing per gene when the per-gene spreads
    vary.  The two references bracket 0.80-0.83 and the measured 0.8836 is above both, so the
    conclusion does not turn on the choice.
    """

    from statistics import NormalDist

    critical = NormalDist().inv_cdf(0.5 + S6_NOMINAL_PROBABILITY / 2.0)
    shape = calibration_shape_diagnostics(arm_slice)
    uniform = 2 * NormalDist().cdf(critical / shape.standard_deviation) - 1
    s2_aggregated = 2 * NormalDist().cdf(critical * 0.8415) - 1
    assert uniform == pytest.approx(0.7996, abs=1e-3)
    assert s2_aggregated == pytest.approx(0.8337, abs=1e-3)
    measured = _report(arm_slice).empirical_coverage
    assert measured == pytest.approx(0.8836, abs=5e-4)
    assert measured > max(uniform, s2_aggregated) + 0.04


def test_coverage_falls_as_the_library_gets_deeper(arm_slice: ArmSlice) -> None:
    """The gradient S6's single pooled number hides, and it points at ``psi^2``.

    ``likelihood.py`` names ``psi^2`` as the defence against "more sequencing depth mistaken for
    more knowledge about the biology".  Coverage runs from 0.94 in the shallowest library to 0.76
    in the deepest, so on this evidence the defence does not hold.

    ⚠️ Depth and differentiation day are collinear here and the design cannot separate them: depth
    rises with day, and within a day the depth range is too narrow to resolve anything.
    """

    shape = calibration_shape_diagnostics(arm_slice)
    assert shape.depth_coverage_correlation == pytest.approx(-0.8573, abs=5e-3)
    coverage = dict(shape.coverage_by_library)
    assert coverage["rep1"] > coverage["rep13"]
    assert max(coverage.values()) - min(coverage.values()) > 0.15


def test_the_scores_are_the_same_evidence_s2_is_measured_on(arm_slice: ArmSlice) -> None:
    """S6 must be a second reading of ADR 0023's estimand, not a second estimand."""

    scores = replicate_standard_scores(arm_slice)
    assert [entry.library for entry in scores] == list(arm_slice.libraries)
    assert all(entry.scores.shape == (100,) for entry in scores)
    assert sum(entry.scores.size for entry in scores) == 1400


# ------------------------------------------------------------------ what reaches the belief


def test_the_belief_carries_an_evaluated_calibration() -> None:
    """The criterion that read NOT_EVALUATED from the day this backend shipped."""

    belief = estimate_arm("rep1", "GATA1")
    report = belief.diagnostics.calibration
    assert report.evaluation_status.value == "evaluated"
    assert report.outcome is CriterionOutcome.FAILED
    assert report.empirical_coverage == pytest.approx(0.8836, abs=5e-4)
    assert belief.readiness.calibration is CriterionOutcome.FAILED


def test_the_abstention_names_the_calibration_failure_with_its_numbers() -> None:
    """Abstention that says why.  It used to say only that it was required."""

    reasons = estimate_arm("rep1", "GATA1").readiness.reasons
    s6 = [reason for reason in reasons if reason.startswith("S6 calibration FAILED")]
    assert len(s6) == 1
    assert "0.8836" in s6[0] and "0.0548" in s6[0] and "0.05" in s6[0]
    assert any(reason.startswith("criteria not met:") for reason in reasons)


def test_every_arm_reports_the_same_deposit_level_calibration() -> None:
    """S6 is a property of the deposit, not of an arm: only ``NT`` carries a replicate."""

    first = estimate_arm("rep1", "GATA1").diagnostics.calibration
    second = estimate_arm("rep9", "SPI1").diagnostics.calibration
    assert first.empirical_coverage == second.empirical_coverage
    assert first.outcome is second.outcome


def test_the_readiness_derivation_responds_when_the_calibration_passes(
    arm_slice: ArmSlice,
) -> None:
    """The half of the derivation the shipped configuration never reaches.

    `abstention_required` is now computed from the criteria rather than asserted, and this is the
    proof that the computation is live: loosen the error threshold to 0.10, S6 passes, the reason
    naming it disappears, and `calibration` drops out of the unmet list.

    ⚠️ **Abstention itself is still `True`, and that is not a bug.**  Six criteria remain
    NOT_EVALUATED, so no configuration of S6 alone can release it.  S6 is the first of the seven to
    become a measurement, not the last.
    """

    relaxed = measure_calibration_coverage(
        arm_slice, minimum_coverage=MINIMUM_COVERAGE, maximum_calibration_error=0.10
    )
    assert relaxed.outcome is CriterionOutcome.PASSED

    thresholds = arm_query(arm_slice.targets, model_fingerprint="0" * 64).acceptance_thresholds
    query = arm_query(arm_slice.targets, model_fingerprint="0" * 64).model_copy(
        update={
            "acceptance_thresholds": thresholds.model_copy(
                update={"maximum_calibration_error": 0.10}
            )
        }
    )
    estimator = GSE274113ObservationEstimator(
        fit_fold(arm_slice, "rep1"),
        query=query,
        slice_fingerprint="0" * 64,
        panel_fingerprint="0" * 64,
        calibration=relaxed,
    )
    readiness = estimator._readiness(estimator._diagnostics())
    assert readiness.calibration is CriterionOutcome.PASSED
    assert not [r for r in readiness.reasons if r.startswith("S6 calibration FAILED")]
    unmet = next(r for r in readiness.reasons if r.startswith("criteria not met:"))
    assert "calibration" not in unmet
    assert readiness.abstention_required is True


# ------------------------------------------------------------------ the import order


@pytest.mark.parametrize(
    "module",
    [
        "cellstate",
        "cellstate.evaluation",
        "cellstate.evaluation.gse274113_reports",
        "cellstate.backends.gse274113",
        "cellstate.backends.gse274113.arm_request",
        "cellstate.backends.gse274113.estimator",
        "cellstate.backends.gse274113.usage",
        "cellstate.ui.server",
    ],
)
def test_every_module_imports_first(module: str) -> None:
    """Import each module into a fresh interpreter as that program's FIRST import.

    Wiring S6 into the belief made ``backends.gse274113.usage`` import
    ``evaluation.gse274113_reports``, which imports back into ``backends.gse274113`` -- a cycle that
    raised ImportError for anyone whose first import happened to be the evaluation module.  **The
    entire suite stayed green**, because every existing test reaches these modules through an order
    that happens to work.  A cycle is invisible from inside a process that already resolved it, so
    this test spends a subprocess per module to get an unpolluted one.
    """

    if module == "cellstate.ui.server":
        pytest.importorskip("fastapi", reason="the web UI needs the 'ui' extra")
    # The child inherits this process's import path but none of its already-imported modules.  Both
    # halves matter: a fresh `sys.modules` is what makes the cycle visible, and the inherited path
    # is what makes the child resolve the same `cellstate` pytest did.  Without the path it picks
    # up whatever `site-packages` offers, which on a machine that has ever built a wheel here is a
    # partial namespace package -- and the test then fails for a reason that has nothing to do with
    # import cycles.
    environment = {**os.environ, "PYTHONPATH": os.pathsep.join(p for p in sys.path if p)}
    completed = subprocess.run(
        [sys.executable, "-c", f"import {module}"],
        capture_output=True,
        text=True,
        check=False,
        env=environment,
    )
    assert completed.returncode == 0, completed.stderr
