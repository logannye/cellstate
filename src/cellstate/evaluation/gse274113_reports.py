"""Held-out capability measurements for the GSE274113 observation model.

Every number here is computed on **leave-one-library-out** state estimates: the state for an arm in
library *L* comes only from the fold that excluded *L*.  Every interval is a cluster bootstrap
grouped at the **library**, which is the independent experimental unit under program rule 8.

Three capabilities are measured, and each can fail:

**S2 -- is the posterior's spread earned?**  The state is inferred from one half of the ``NT`` arm
and the predictive distribution it implies is scored against the other half, so the claimed spread
meets an error the model genuinely has not seen.

This is the test an earlier revision of this docstring described but no code performed.  That
revision warned that the shipped comparison was **not like-for-like** -- the posterior conditioned
on the arm's own observation while the point predictor did not, so a ratio below one was partly
definitional -- and observed that a clean test needs a held-out *replicate of the same arm*, which
this design supplies for ``NT``.  [ADR 0023] makes that the estimand.  The warning is retired rather
than restated, and the number it qualified moved from 0.28 to 0.84 in the process.

⚠️ What remains, and it is a real limit: ``NT`` is the only arm with a replicate, so S2 is measured
on **null biology only**, across 14 libraries rather than 280 arms.

**S4 -- does the ``do`` operator move when, and only when, the intervention moves?**  Both halves.
The null half uses ``NT``, whose direction is estimated from a placebo split rather than fixed at
zero, so it is a measurement that can come out wrong rather than an identity.

**S5 -- is the nuisance axis separated from biology?**  The across-library spread at a fixed target,
measured in the *inferred state* rather than in the observation, as a fraction of the across-target
spread.  Separation means library variation lands in the nuisance block and leaves the biology block
alone.

Nothing here is a sufficiency result.  S1, S3 and S7 remain structurally unreachable on this
evidence, and no combination of the measurements below substitutes for them.

[ADR 0023]: ../../../docs/adr/0023-the-s2-estimand-is-a-split-half-replicate.md
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import NormalDist

import numpy as np
from numpy.typing import NDArray

from ..backends.gse274113.arm_request import S6_NOMINAL_PROBABILITY
from ..backends.gse274113.fit import (
    NULL_TARGET,
    PLACEBO_TARGETS,
    ArmSlice,
    FittedFold,
    fit_fold,
)
from ..backends.gse274113.likelihood import posterior
from ..domain.belief import CalibrationReport
from .bootstrap import BootstrapInterval, multiway_clustered_bootstrap
from .calibration import evaluate_marginal_calibration

FloatArray = NDArray[np.float64]

DEFAULT_SEED = 20260813

__all__ = [
    "S6_NOMINAL_PROBABILITY",
    "ArmState",
    "CalibrationShape",
    "CapabilityMeasurement",
    "ReplicateStandardScores",
    "calibration_shape_diagnostics",
    "held_out_states",
    "measure_calibration_coverage",
    "measure_earned_spread",
    "measure_intervention_response",
    "measure_nuisance_separation",
    "measure_point_predictor_spread",
    "replicate_standard_scores",
]


@dataclass(frozen=True)
class ArmState:
    """One arm's held-out state estimate and the residual of a point prediction for it."""

    library: str
    target: str
    biology: FloatArray
    realization: float
    realization_sd: float
    predictive_sd: FloatArray
    point_residual: FloatArray


def _biology(fold: FittedFold, slice_data: ArmSlice, library: str, target: str) -> FloatArray:
    """Biology-block coefficients for one arm, in the fold's shared W coordinates.

    ``W`` is shared across every target, so biology coefficients ARE comparable between arms.  The
    realization direction is not: it is target-specific and orthogonal to ``W``.
    """

    composition, depth = slice_data.log_composition(library, target)
    design = fold.design(NULL_TARGET if target.startswith("NT_") else target)
    mean, _ = posterior(
        composition,
        intercept=fold.intercept,
        design=design,
        prior_precision=fold.prior_precision(),
        observation_variance_diagonal=fold.observation_variance(depth),
    )
    return np.asarray(mean[: fold.biology_basis.shape[1]], dtype=np.float64)


@dataclass(frozen=True)
class CapabilityMeasurement:
    """A measured quantity with its interval, and the verdict that interval supports."""

    name: str
    value: float
    interval: BootstrapInterval
    unit_count: int
    passed: bool
    statement: str


def held_out_states(slice_data: ArmSlice, *, seed: int = DEFAULT_SEED) -> tuple[ArmState, ...]:
    """Estimate every arm from the fold that never saw its library."""

    del seed
    folds: dict[str, FittedFold] = {
        library: fit_fold(slice_data, library) for library in slice_data.libraries
    }

    # A point predictor for the same arms: the target's mean coefficients over the fit libraries.
    # It is what the posterior has to beat on spread, so it must be built from the same folds and
    # never from the held-out library.
    states: list[ArmState] = []
    for library in slice_data.libraries:
        fold = folds[library]
        for target in slice_data.targets:
            composition, depth = slice_data.log_composition(library, target)
            design = fold.design(target)
            variance = fold.observation_variance(depth)
            mean, covariance = posterior(
                composition,
                intercept=fold.intercept,
                design=design,
                prior_precision=fold.prior_precision(),
                observation_variance_diagonal=variance,
            )
            predictive_variance = np.einsum("gi,ij,gj->g", design, covariance, design) + variance

            typical = np.mean(
                [
                    np.linalg.lstsq(
                        design,
                        slice_data.log_composition(other, target)[0] - fold.intercept,
                        rcond=None,
                    )[0]
                    for other in fold.fit_library_ids
                ],
                axis=0,
            )
            residual = composition - fold.intercept - design @ typical

            rank = fold.biology_basis.shape[1]
            states.append(
                ArmState(
                    library=library,
                    target=target,
                    biology=np.asarray(mean[:rank], dtype=np.float64),
                    realization=float(mean[-1]),
                    realization_sd=float(np.sqrt(covariance[-1, -1])),
                    predictive_sd=np.asarray(np.sqrt(predictive_variance), dtype=np.float64),
                    point_residual=np.asarray(residual, dtype=np.float64),
                )
            )
    return tuple(states)


def _interval(values: list[float], libraries: list[str], *, seed: int) -> BootstrapInterval:
    return multiway_clustered_bootstrap(
        values=values,
        cluster_labels={"library": libraries},
        seed=seed,
    )


def measure_earned_spread(
    slice_data: ArmSlice, *, seed: int = DEFAULT_SEED
) -> CapabilityMeasurement:
    """S2: the posterior's claimed spread against the error it makes on a held-out replicate.

    Per [ADR 0023] the estimand is a **split-half replicate on the declared-null arm**.  The state
    is inferred from ``NT_A``, the predictive distribution is formed for ``NT_B``, and the claimed
    spread is compared against the error actually realized on ``NT_B``.  The ratio must exceed one
    for the spread to be earned, and the *lower* interval end decides it -- a point estimate above
    one with an interval straddling it has not shown anything.

    **Both sides condition on the same information**, which is what the four point-predictor
    constructions could not supply: each of those scored a posterior that had seen the arm against a
    predictor that had not, so the ratio was as much a statement about the handicap as about
    calibration.  The superseded construction is kept below as a labelled diagnostic.

    The fold discipline is intact and was checked rather than assumed.  ``NT``'s direction is
    estimated from the placebo split, so an estimand built on that same split invites the question
    of whether the held-out library's halves informed the design they are scored under.  They do
    not: ``fit.py`` draws its placebo libraries from the fit libraries alone, so for a fold that
    excludes *L* the intercept, both subspaces, ``psi^2``, the pooled rate and the ``NT`` direction
    all come from the other thirteen libraries, and **both** halves of *L*'s ``NT`` arm are held
    out.

    ⚠️ **Scope: null biology only, across 14 libraries rather than 280 arms.**  ``NT`` is the only
    arm in this design carrying a replicate.  This does not establish calibration on a perturbed
    arm, and the limit is a property of the estimand rather than a caveat on one run.

    ⚠️ **The halves are shallower than a full arm, and it does not explain the result.**  ``NT`` is
    the bitwise sum of ``NT_A`` and ``NT_B`` in all fourteen libraries, so each half is **exactly**
    half the arm: 684,370 panel counts on average against 1,368,741, a **2.00x** shortfall.  Lower
    depth inflates both the claimed spread and the realized error.  Recomputing with the technical
    term at the full-arm depth on both sides moves the ratio from 0.8415 to **0.7727** -- about
    8.2%, and in the direction that makes the failure *worse*.

    An earlier revision of this paragraph put the shortfall at "2.06x" and the recomputation at
    "0.8322 -- about 1.1%".  Neither reproduced.  The 2.06 was the depth of ``NT_B`` alone (665,763)
    quoted as though it were the mean over both halves, and 0.8322 is what a **1.1x** depth
    multiplier gives, not the 2.0x the sentence names.  The conclusion the caveat was written to
    support is unchanged and in fact stronger -- depth does not excuse the failure, it deepens it --
    but the numbers were a recorded claim that no code checked.  They are pinned now; see
    ``test_the_s2_depth_caveat_reproduces``.

    Both sides aggregate across genes as a root-mean-square.  The superseded form paired a *mean* of
    per-gene standard deviations against an RMS residual, which is not like-for-like and understates
    the numerator by 8.0% on this evidence.

    [ADR 0023]: ../../../docs/adr/0023-the-s2-estimand-is-a-split-half-replicate.md
    """

    ratios: list[float] = []
    libraries: list[str] = []
    for library in slice_data.libraries:
        fold = fit_fold(slice_data, library)
        design = fold.design(NULL_TARGET)
        inferred_from, inferred_depth = slice_data.log_composition(library, PLACEBO_TARGETS[0])
        predicted_for, predicted_depth = slice_data.log_composition(library, PLACEBO_TARGETS[1])
        mean, covariance = posterior(
            inferred_from,
            intercept=fold.intercept,
            design=design,
            prior_precision=fold.prior_precision(),
            observation_variance_diagonal=fold.observation_variance(inferred_depth),
        )
        predictive_variance = np.einsum(
            "gi,ij,gj->g", design, covariance, design
        ) + fold.observation_variance(predicted_depth)
        residual = predicted_for - fold.intercept - design @ mean
        ratios.append(float(np.sqrt(np.mean(predictive_variance)) / np.sqrt(np.mean(residual**2))))
        libraries.append(library)

    interval = _interval(ratios, libraries, seed=seed)
    passed = interval.lower > 1.0
    return CapabilityMeasurement(
        name="S2 earned posterior spread (split-half replicate on NT)",
        value=float(np.mean(ratios)),
        interval=interval,
        unit_count=len(set(libraries)),
        passed=passed,
        statement=(
            "the posterior's claimed spread covers the error it makes on a held-out replicate"
            if passed
            else "the posterior claims a spread NARROWER than the error it makes on a held-out "
            "replicate; on this evidence the spread is not earned"
        ),
    )


@dataclass(frozen=True)
class ReplicateStandardScores:
    """One library's standardized predictive residuals on the ADR 0023 split-half replicate.

    ``scores[j]`` is ``(observed - predicted) / predictive_sd`` for panel gene ``j``, where the
    prediction is formed from ``NT_A`` and the observation is ``NT_B``.  Under the observation
    model these are standard normal.  Whether they are is the whole of S6.
    """

    library: str
    replicate_depth: float
    scores: FloatArray


def replicate_standard_scores(slice_data: ArmSlice) -> tuple[ReplicateStandardScores, ...]:
    """Standardize the S2 replicate residuals, one vector of panel genes per library.

    This is deliberately the *same* construction [ADR 0023] decided for S2 -- state inferred from
    ``NT_A``, predictive distribution formed for ``NT_B`` at ``NT_B``'s own depth -- so that S2 and
    S6 are two readings of one evidence set rather than two estimands whose disagreement could be
    blamed on the setup.  S2 aggregates these into a root-mean-square ratio; S6 counts how many
    land inside their interval.  Those are different questions, and on this evidence they give
    materially different answers.

    [ADR 0023]: ../../../docs/adr/0023-the-s2-estimand-is-a-split-half-replicate.md
    """

    scores: list[ReplicateStandardScores] = []
    for library in slice_data.libraries:
        fold = fit_fold(slice_data, library)
        design = fold.design(NULL_TARGET)
        inferred_from, inferred_depth = slice_data.log_composition(library, PLACEBO_TARGETS[0])
        predicted_for, predicted_depth = slice_data.log_composition(library, PLACEBO_TARGETS[1])
        mean, covariance = posterior(
            inferred_from,
            intercept=fold.intercept,
            design=design,
            prior_precision=fold.prior_precision(),
            observation_variance_diagonal=fold.observation_variance(inferred_depth),
        )
        predictive_variance = np.einsum(
            "gi,ij,gj->g", design, covariance, design
        ) + fold.observation_variance(predicted_depth)
        residual = predicted_for - fold.intercept - design @ mean
        scores.append(
            ReplicateStandardScores(
                library=library,
                replicate_depth=predicted_depth,
                scores=np.asarray(residual / np.sqrt(predictive_variance), dtype=np.float64),
            )
        )
    return tuple(scores)


def measure_calibration_coverage(
    slice_data: ArmSlice,
    *,
    minimum_coverage: float,
    maximum_calibration_error: float,
    nominal_probability: float = S6_NOMINAL_PROBABILITY,
    seed: int = DEFAULT_SEED,
) -> CalibrationReport:
    """S6: does the predictive interval contain the replicate at the rate it claims?

    The independent experimental unit is the **library**, as it is everywhere else here.  Coverage
    is counted over the 100 panel genes *within* a library and only libraries are resampled, which
    is what keeps the compositional dependence between genes out of the interval: genes in one
    panel are emphatically not 100 independent trials, and pooling them into one fraction and
    bootstrapping *that* would claim a precision this design cannot buy.

    **S6 is not a restatement of S2, and on this evidence it disagrees with it.**  S2 reports a
    root-mean-square ratio of claimed spread to realized error, which reads as "the interval is
    uniformly ~16 percent too narrow".  Coverage says otherwise -- see
    ``calibration_shape_diagnostics``, whose trimmed spreads show that removing **2 percent** of
    the 1,400 gene-library outcomes moves the S2-style ratio from 0.7784 to 0.9955.  The bulk of
    the panel is *better* than the interval claims; a small number of genes are far outside it.
    A ratio cannot tell those two situations apart and a coverage count can, which is the reason
    to have both rather than either.

    ⚠️ **Scope, inherited from [ADR 0023] unchanged: null biology only, 14 libraries.**  ``NT`` is
    the only arm in this design carrying a replicate.  S6 does not establish calibration on a
    perturbed arm, and the reason is a property of the deposit rather than of this code.

    [ADR 0023]: ../../../docs/adr/0023-the-s2-estimand-is-a-split-half-replicate.md
    """

    critical = NormalDist().inv_cdf(0.5 + nominal_probability / 2.0)
    per_library = replicate_standard_scores(slice_data)
    outcomes = [entry.scores.tolist() for entry in per_library]
    lower = [(-critical * np.ones_like(entry.scores)).tolist() for entry in per_library]
    upper = [(critical * np.ones_like(entry.scores)).tolist() for entry in per_library]
    return evaluate_marginal_calibration(
        unit_outcomes=outcomes,
        unit_lower_bounds=lower,
        unit_upper_bounds=upper,
        cluster_labels={"library": [entry.library for entry in per_library]},
        nominal_probability=nominal_probability,
        maximum_calibration_error=maximum_calibration_error,
        minimum_coverage=minimum_coverage,
        seed=seed,
        metric="panel log-composition, split-half replicate (ADR 0023 estimand)",
    )


@dataclass(frozen=True)
class CalibrationShape:
    """Why S6 and S2 disagree: the tail, and a coverage gradient S6 alone would hide."""

    standard_deviation: float
    trimmed_standard_deviation: float
    trimmed_fraction: float
    largest_absolute_score: float
    coverage_by_library: tuple[tuple[str, float], ...]
    depth_coverage_correlation: float


def calibration_shape_diagnostics(
    slice_data: ArmSlice,
    *,
    nominal_probability: float = S6_NOMINAL_PROBABILITY,
    trimmed_fraction: float = 0.02,
) -> CalibrationShape:
    """Decompose S6's shortfall into a tail term and a depth gradient.

    **A failing ratio names two suspects.**  S2's 0.8415 is consistent with an interval that is
    uniformly too narrow *and* with an interval that is right for most genes and hopeless for a
    few, and the two diagnoses call for opposite repairs: the first says inflate ``psi^2``, the
    second says that inflating it would push an already-conservative bulk into over-coverage while
    still not reaching the outliers.  Trimming separates them.

    The second diagnostic is a gradient S6's single pooled number cannot show.  Coverage falls
    monotonically with the library's panel depth, which is the exact failure ``likelihood.py``
    names ``psi^2`` as the defence against -- "more sequencing depth is mistaken for more knowledge
    about the biology".  ADR 0022 stopped ``psi^2`` from being clamped; it did not stop this.

    ⚠️ **Depth and differentiation day are collinear in this deposit and cannot be separated.**
    Panel depth rises with day by construction, and within a day the depth range is too narrow to
    resolve anything.  The gradient is real; which of the two drives it is not identified here.
    """

    critical = NormalDist().inv_cdf(0.5 + nominal_probability / 2.0)
    per_library = replicate_standard_scores(slice_data)
    stacked = np.concatenate([entry.scores for entry in per_library])
    keep = round(trimmed_fraction * stacked.size)
    trimmed = np.delete(stacked, np.argsort(-np.abs(stacked))[:keep])
    coverage = [
        (entry.library, float(np.mean(np.abs(entry.scores) <= critical))) for entry in per_library
    ]
    depths = np.log([entry.replicate_depth for entry in per_library])
    return CalibrationShape(
        standard_deviation=float(stacked.std(ddof=1)),
        trimmed_standard_deviation=float(trimmed.std(ddof=1)),
        trimmed_fraction=trimmed_fraction,
        largest_absolute_score=float(np.abs(stacked).max()),
        coverage_by_library=tuple(coverage),
        depth_coverage_correlation=float(
            np.corrcoef(depths, [value for _, value in coverage])[0, 1]
        ),
    )


def measure_point_predictor_spread(
    states: tuple[ArmState, ...], *, seed: int = DEFAULT_SEED
) -> CapabilityMeasurement:
    """The superseded S2 construction, retained as a diagnostic. **This is not the S2 verdict.**

    [ADR 0023] decision 3 keeps this reported rather than deleted, so that the change of estimand
    stays visible in the record instead of appearing as a number that moved on its own.

    It compares the posterior against a point predictor taking every coefficient from the target's
    mean over the fold's fit libraries.  That predictor is handicapped twice: it never sees the arm,
    while the posterior it is scored against does, and averaging coefficients across libraries
    washes out the nuisance block, so it is then charged for library variation it was structurally
    forbidden to know.  Both handicaps push the ratio down, which is why this construction sat at
    the extreme pessimistic corner of the swept grid.

    The reported value here is **not** the published 0.27806.  That figure was computed under the
    mean-of-standard-deviations aggregation ADR 0023 decision 2 supersedes; under the decided
    root-mean-square aggregation the same construction measures 0.30206.  One convention is kept in
    code and the history is kept in the model card.

    [ADR 0023]: ../../../docs/adr/0023-the-s2-estimand-is-a-split-half-replicate.md
    """

    ratios = [
        float(np.sqrt(np.mean(state.predictive_sd**2)) / np.sqrt(np.mean(state.point_residual**2)))
        for state in states
    ]
    libraries = [state.library for state in states]
    interval = _interval(ratios, libraries, seed=seed)
    return CapabilityMeasurement(
        name="S2 diagnostic: posterior against a library-blind point predictor (NOT the verdict)",
        value=float(np.mean(ratios)),
        interval=interval,
        unit_count=len(set(libraries)),
        passed=interval.lower > 1.0,
        statement=(
            "diagnostic only, superseded as the S2 estimand by ADR 0023: the point predictor is "
            "denied both the arm and its own library, so this ratio measures the handicap as much "
            "as the calibration"
        ),
    )


def measure_intervention_response(
    slice_data: ArmSlice, *, seed: int = DEFAULT_SEED
) -> tuple[CapabilityMeasurement, CapabilityMeasurement]:
    """S4, both halves, measured as a CONTRAST in shared biology coordinates.

    An earlier version of this measured the realization coefficient instead, and it could not have
    worked: ``u_g`` is orthogonalized against ``W``, and ``W`` already absorbs the perturbation
    signal, so the residual direction carries almost none of it.  Both halves came out
    indistinguishable for a modelling reason rather than a biological one.  Reporting that as an S4
    verdict would have been a measurement artefact dressed as a result.

    The contrast below is like-for-like.  Both halves are a difference between two arms of the same
    library, in the same shared coordinates:

    * **null**: ``NT_B - NT_A``, two halves of the same non-targeting population.  Same biology, so
      the expected contrast is zero -- but it is *estimated*, so it can come out otherwise.
    * **non-null**: ``target - NT`` within the same library.

    ⚠️ **The null half is biased upward, the bias is not corrected here, and it is too small to
    explain the result.**  Both contrasts' sampling noise goes as ``sqrt(1/n1 + 1/n2)``; the placebo
    compares two half-sized NT populations while the perturbed contrast compares an arm against the
    *full* NT arm.  Computed over all 266 (library, target) pairs, the placebo contrast carries
    **1.16x** the multinomial noise sd of the perturbed one -- not the ~1.8x a naive per-arm cell
    count suggests, since placebo halves average 409 cells against 474 for a perturbed arm.
    Deflating the null by that factor leaves it at 1.74, still inside the perturbed interval.  **So
    the failure is not explained by the imbalance**, and the caveat is recorded as a caveat rather
    than as an excuse.  Equalising the cell counts is still the clean fix and is not applied yet.

    Grouped at the library, which is also what makes the two halves paired rather than pooled.
    """

    folds = {library: fit_fold(slice_data, library) for library in slice_data.libraries}
    perturbed_targets = [target for target in slice_data.targets if target != NULL_TARGET]

    null_values: list[float] = []
    null_libraries: list[str] = []
    perturbed_values: list[float] = []
    perturbed_libraries: list[str] = []

    for library in slice_data.libraries:
        fold = folds[library]
        half_a = _biology(fold, slice_data, library, "NT_A")
        half_b = _biology(fold, slice_data, library, "NT_B")
        null_values.append(float(np.linalg.norm(half_b - half_a)))
        null_libraries.append(library)

        baseline = _biology(fold, slice_data, library, NULL_TARGET)
        for target in perturbed_targets:
            state = _biology(fold, slice_data, library, target)
            perturbed_values.append(float(np.linalg.norm(state - baseline)))
            perturbed_libraries.append(library)

    null_interval = _interval(null_values, null_libraries, seed=seed)
    perturbed_interval = _interval(perturbed_values, perturbed_libraries, seed=seed + 1)

    # Judged on non-overlapping intervals, never on point estimates.
    separated = null_interval.upper < perturbed_interval.lower
    return (
        CapabilityMeasurement(
            name="S4 null half (placebo NT_A vs NT_B)",
            value=float(np.mean(null_values)),
            interval=null_interval,
            unit_count=len(set(null_libraries)),
            passed=separated,
            statement=(
                "a declared-null contrast moves the state strictly less than a perturbation does"
                if separated
                else "the declared-null contrast is NOT separated from the perturbed contrast"
            ),
        ),
        CapabilityMeasurement(
            name="S4 non-null half (target vs NT, 19 targets)",
            value=float(np.mean(perturbed_values)),
            interval=perturbed_interval,
            unit_count=len(set(perturbed_libraries)),
            passed=separated,
            statement=(
                "perturbed contrasts exceed the declared-null band"
                if separated
                else "perturbed contrasts do NOT exceed the declared-null band"
            ),
        ),
    )


def measure_nuisance_separation(
    states: tuple[ArmState, ...],
    *,
    bound: float,
    seed: int = DEFAULT_SEED,
) -> CapabilityMeasurement:
    """S5: across-library spread at a fixed target, as a fraction of across-target spread.

    Measured in the inferred *state*, not in the observation.  The observation is expected to move
    with the library -- that is what a nuisance axis is.  What S5 asks is that the inferred state
    does not.

    ``bound`` is predeclared by the caller and must be fixed before this runs.
    """

    by_target: dict[str, list[FloatArray]] = {}
    for state in states:
        by_target.setdefault(state.target, []).append(state.biology)

    target_means = {target: np.mean(values, axis=0) for target, values in by_target.items()}
    across_target = float(np.mean(np.var(np.stack(list(target_means.values())), axis=0)))
    if across_target <= 0.0:
        raise ValueError("across-target spread vanished; the ratio would be undefined")

    ratios = []
    libraries = []
    for state in states:
        deviation = state.biology - target_means[state.target]
        ratios.append(float(np.mean(deviation**2) / across_target))
        libraries.append(state.library)

    interval = _interval(ratios, libraries, seed=seed)
    value = float(np.mean(ratios))
    passed = interval.upper <= bound
    return CapabilityMeasurement(
        name="S5 nuisance separation in the inferred state",
        value=value,
        interval=interval,
        unit_count=len(set(libraries)),
        passed=passed,
        statement=(
            f"across-library spread at a fixed target is within the predeclared bound {bound}"
            if passed
            else f"across-library spread at a fixed target EXCEEDS the predeclared bound {bound}; "
            "library variation is reaching the inferred state"
        ),
    )
