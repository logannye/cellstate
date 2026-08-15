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

from cellstate.backends.gse274113.arm_request import (
    S6_NOMINAL_GRID_STEP,
    S6_NOMINAL_INTERVAL,
    S6_NOMINAL_PROBABILITIES,
    S6_REFERENCE_NOMINAL,
    arm_query,
)
from cellstate.backends.gse274113.estimator import GSE274113ObservationEstimator
from cellstate.backends.gse274113.fit import ArmSlice, fit_fold
from cellstate.backends.gse274113.usage import estimate_arm
from cellstate.domain.common import CriterionOutcome
from cellstate.evaluation.gse274113_reports import (
    calibration_shape_diagnostics,
    measure_calibration_coverage,
    measure_calibration_level_set,
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


def test_the_predeclared_pair_is_coherent_on_an_interval_not_a_point() -> None:
    """The correction to ADR 0024 decision 3, recomputed from the shipped thresholds.

    The superseded claim was that the pair coincides at *exactly one* nominal.  It does not: it is
    coherent on a closed interval, and 0.90 is that interval's minimum.  What is unique about 0.90
    is narrower -- it is the single point where the floor and the error bound's lower edge coincide,
    which is the fact the superseded claim had hold of.

    The previous version of this test asserted ``min(consistent) == 0.90`` under the name
    ``test_the_nominal_is_forced_by_the_predeclared_pair``.  **A test named for uniqueness that
    asserted a minimum**: it computed the six-element set and then checked only its first element,
    so the overclaim was in the name and in the ADR while the code beneath was quietly weaker.
    Asserting set equality is what makes the two agree.
    """

    thresholds = arm_query(("GATA1",), model_fingerprint="0" * 64).acceptance_thresholds
    assert thresholds.minimum_calibration_coverage == MINIMUM_COVERAGE
    assert thresholds.maximum_calibration_error == MAXIMUM_CALIBRATION_ERROR

    # Below the interval the error bound admits a coverage the floor rejects; above it, the bound's
    # upper half asks for a coverage above one.  Both are recomputed, not asserted.
    coherent = [
        candidate / 100
        for candidate in range(1, 101)
        if candidate / 100 - thresholds.maximum_calibration_error
        >= thresholds.minimum_calibration_coverage - 1e-12
        and candidate / 100 + thresholds.maximum_calibration_error <= 1.0 + 1e-12
    ]
    assert min(coherent) == pytest.approx(S6_NOMINAL_INTERVAL[0])
    assert max(coherent) == pytest.approx(S6_NOMINAL_INTERVAL[1])
    assert S6_NOMINAL_INTERVAL == (0.90, 0.95)

    # 0.90's actual distinction: floor and error-bound lower edge coincide there, and only there.
    coinciding = [
        candidate / 100
        for candidate in range(1, 101)
        if abs(
            (candidate / 100 - thresholds.maximum_calibration_error)
            - thresholds.minimum_calibration_coverage
        )
        < 1e-12
    ]
    assert coinciding == [pytest.approx(S6_REFERENCE_NOMINAL)]


def test_the_declared_levels_are_the_grid_on_the_derived_interval() -> None:
    """The gated levels are the interval on the declared step -- set equality, not a minimum.

    The interval is derived; the 0.01 step is a choice.  Recomputing the tuple from both keeps the
    declaration from drifting away from what it claims to be.
    """

    low, high = S6_NOMINAL_INTERVAL
    steps = round((high - low) / S6_NOMINAL_GRID_STEP)
    expected = tuple(round(low + index * S6_NOMINAL_GRID_STEP, 10) for index in range(steps + 1))
    assert set(S6_NOMINAL_PROBABILITIES) == set(expected)
    assert S6_NOMINAL_PROBABILITIES == (0.90, 0.91, 0.92, 0.93, 0.94, 0.95)
    assert S6_REFERENCE_NOMINAL in S6_NOMINAL_PROBABILITIES


def test_the_gate_is_the_conjunction_and_every_level_fails(arm_slice: ArmSlice) -> None:
    """All six, with the reference level's published figures unchanged by the widening."""

    levels = measure_calibration_level_set(
        arm_slice,
        minimum_coverage=MINIMUM_COVERAGE,
        maximum_calibration_error=MAXIMUM_CALIBRATION_ERROR,
    )
    assert levels.outcome is CriterionOutcome.FAILED
    assert levels.failing_nominals == S6_NOMINAL_PROBABILITIES
    assert levels.reference.empirical_coverage == pytest.approx(0.8836, abs=5e-4)
    assert levels.reference.calibration_error_upper_bound == pytest.approx(0.0548, abs=5e-4)

    # The bound rises monotonically across the interval, so the reference level is the LOOSEST of
    # the six -- which is why gating there alone was the easiest reading of the predeclaration.
    bounds = [report.calibration_error_upper_bound for report in levels.reports]
    assert bounds == sorted(bounds)
    assert bounds[0] == pytest.approx(0.0548, abs=5e-4)
    assert bounds[-1] == pytest.approx(0.0767, abs=5e-4)


# ------------------------------------------------------------------ a gate a WRONG answer passes


def _rescaled(monkeypatch: pytest.MonkeyPatch, arm_slice: ArmSlice, factor: float) -> None:
    """Multiply every predictive standard deviation by ``factor`` -- and change nothing else.

    Dividing the standardized residuals is exactly equivalent to inflating the predictive sd, and
    it is the crudest possible "repair": no mechanism, no fitted quantity, no claim about the
    biology. If a gate can be cleared this way then passing it is not evidence the model improved.
    """

    from cellstate.evaluation import gse274113_reports as reports

    original = reports.replicate_standard_scores(arm_slice)
    scaled = tuple(
        reports.ReplicateStandardScores(
            library=entry.library,
            replicate_depth=entry.replicate_depth,
            scores=entry.scores / factor,
        )
        for entry in original
    )
    monkeypatch.setattr(reports, "replicate_standard_scores", lambda _slice: scaled)


def test_a_constant_rescaling_clears_the_single_level_gate(
    monkeypatch: pytest.MonkeyPatch, arm_slice: ArmSlice
) -> None:
    """**The reason the gate is six levels and not one.**

    This repository keeps finding checks a correct computation cannot fail. This is the mirror: a
    check a *wrong* computation passes. Multiplying every predictive standard deviation by 1.11 --
    no modelling of any kind -- takes the reference level from FAILED to PASSED with a bound of
    0.0368, better than the shipped 0.0548 and better than any mechanism-based repair measured so
    far. A one-level gate tests the residuals' scale, and scale is free.
    """

    _rescaled(monkeypatch, arm_slice, 1.11)
    cheated = measure_calibration_coverage(
        arm_slice,
        minimum_coverage=MINIMUM_COVERAGE,
        maximum_calibration_error=MAXIMUM_CALIBRATION_ERROR,
        nominal_probability=S6_REFERENCE_NOMINAL,
    )
    assert cheated.outcome is CriterionOutcome.PASSED
    assert cheated.empirical_coverage == pytest.approx(0.9000, abs=1e-3)
    assert cheated.calibration_error_upper_bound == pytest.approx(0.0368, abs=5e-3)
    assert cheated.calibration_error_upper_bound < 0.0548


def test_the_same_constant_does_not_clear_the_six_level_gate(
    monkeypatch: pytest.MonkeyPatch, arm_slice: ArmSlice
) -> None:
    """And the reason six levels is a real gate: the same free trick fails it.

    At 1.11 the upper half of the interval still fails, so the conjunction fails. Across the whole
    interval only a narrow band of scalars clears every level, which is what makes the widened gate
    a statement about the residuals' *shape* rather than their scale.
    """

    _rescaled(monkeypatch, arm_slice, 1.11)
    with pytest.raises(ValueError, match="reference level"):
        measure_calibration_level_set(
            arm_slice,
            minimum_coverage=MINIMUM_COVERAGE,
            maximum_calibration_error=MAXIMUM_CALIBRATION_ERROR,
        )


def test_the_six_level_gate_can_be_passed(
    monkeypatch: pytest.MonkeyPatch, arm_slice: ArmSlice
) -> None:
    """The conjunction's PASSED branch, taken.

    A gate that has only ever been observed failing is not yet known to be a gate. Exactly one
    scalar in ``[1.00, 1.80]`` clears all six levels: 1.20 leaves 0.95 failing, 1.22 puts 0.90 back
    over, and 1.21 threads them. That the passing window is one step wide is the substance of ADR
    0025 -- across six levels a rescaling has nowhere to hide, where at one level eighteen scalars
    work.
    """

    _rescaled(monkeypatch, arm_slice, 1.21)
    levels = measure_calibration_level_set(
        arm_slice,
        minimum_coverage=MINIMUM_COVERAGE,
        maximum_calibration_error=MAXIMUM_CALIBRATION_ERROR,
    )
    assert levels.outcome is CriterionOutcome.PASSED
    assert levels.failing_nominals == ()
    assert all(report.outcome is CriterionOutcome.PASSED for report in levels.reports)


def test_the_passing_window_is_one_grid_step_wide(
    monkeypatch: pytest.MonkeyPatch, arm_slice: ArmSlice
) -> None:
    """Either side of 1.21 the six-level gate closes again, from opposite ends.

    Below it the upper levels still fail; above it the reference level fails. Neither neighbour
    reaches the gate, so the constant that clears it is not a range a repair could stumble into.
    """

    for factor, expected_failure in ((1.20, 0.95), (1.22, 0.90)):
        with monkeypatch.context() as patched:
            _rescaled(patched, arm_slice, factor)
            try:
                levels = measure_calibration_level_set(
                    arm_slice,
                    minimum_coverage=MINIMUM_COVERAGE,
                    maximum_calibration_error=MAXIMUM_CALIBRATION_ERROR,
                )
            except ValueError as raised:
                # 1.20 clears the reference level but not the top of the interval, so the
                # disagreement guard fires before a level set is returned. That IS the gate closing.
                assert "reference level" in str(raised)
                assert expected_failure == 0.95
                continue
            assert levels.outcome is CriterionOutcome.FAILED
            assert expected_failure in levels.failing_nominals


def test_the_reference_level_may_not_disagree_with_the_conjunction(
    monkeypatch: pytest.MonkeyPatch, arm_slice: ArmSlice
) -> None:
    """The disagreement guard, exercised from the side that fires.

    It cannot fire on the committed slice -- all six levels fail together -- so without a
    constructed case it would be a branch nobody has ever seen taken. Rescaling by 1.11 builds one:
    the reference level PASSES while the conjunction FAILS, and the code refuses to publish either
    verdict rather than picking the convenient one.
    """

    _rescaled(monkeypatch, arm_slice, 1.11)

    reference = measure_calibration_coverage(
        arm_slice,
        minimum_coverage=MINIMUM_COVERAGE,
        maximum_calibration_error=MAXIMUM_CALIBRATION_ERROR,
        nominal_probability=S6_REFERENCE_NOMINAL,
    )
    upper = measure_calibration_coverage(
        arm_slice,
        minimum_coverage=MINIMUM_COVERAGE,
        maximum_calibration_error=MAXIMUM_CALIBRATION_ERROR,
        nominal_probability=S6_NOMINAL_PROBABILITIES[-1],
    )
    assert reference.outcome is CriterionOutcome.PASSED
    assert upper.outcome is CriterionOutcome.FAILED

    with pytest.raises(ValueError) as raised:
        measure_calibration_level_set(
            arm_slice,
            minimum_coverage=MINIMUM_COVERAGE,
            maximum_calibration_error=MAXIMUM_CALIBRATION_ERROR,
        )
    message = str(raised.value)
    assert "reference level 0.9" in message
    assert "passed" in message and "failed" in message
    assert "ADR" in message


def test_a_reference_level_outside_the_gated_set_is_refused(arm_slice: ArmSlice) -> None:
    """A belief may not publish a level the gate never evaluated."""

    with pytest.raises(ValueError, match="not one of the gated levels"):
        measure_calibration_level_set(
            arm_slice,
            minimum_coverage=MINIMUM_COVERAGE,
            maximum_calibration_error=MAXIMUM_CALIBRATION_ERROR,
            reference_nominal=0.80,
        )


def test_the_result_reports_the_levels_it_measured_not_the_declared_ones(
    arm_slice: ArmSlice,
) -> None:
    """``nominals`` describes the measurement, never the module constant.

    It was briefly a property returning ``S6_NOMINAL_PROBABILITIES``, which would have labelled a
    result measured at any other set with the default levels -- a field describing the declaration
    rather than what was computed.
    """

    custom = (0.92, 0.93)
    levels = measure_calibration_level_set(
        arm_slice,
        minimum_coverage=MINIMUM_COVERAGE,
        maximum_calibration_error=MAXIMUM_CALIBRATION_ERROR,
        nominals=custom,
        reference_nominal=0.92,
    )
    assert levels.nominals == custom
    assert levels.nominals != S6_NOMINAL_PROBABILITIES
    assert len(levels.reports) == 2
    assert levels.reference.empirical_coverage == pytest.approx(0.8943, abs=5e-4)


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

    critical = NormalDist().inv_cdf(0.5 + S6_REFERENCE_NOMINAL / 2.0)
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
