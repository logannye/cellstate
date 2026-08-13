# ADR 0016: The verdict gates on the interval, and the interval is corrected

- **Status:** Accepted
- **Date:** 2026-08-12

## Context

[ADR 0015](0015-faithfulness-reports-carry-their-sampling-distribution.md) gave both faithfulness
reports a sampling distribution and argued, at length, that calibration must gate its outcome on the
upper confidence bound rather than on the point estimate: *"a point estimate inside the threshold
with a bound outside it is not a pass."*

A review of the delivered `Q1` and `Q2` work found that the argument was applied to one of the two
tests, that the calibration certificate it produced was never checked against the interval it claims
to summarize, and that the interval estimator both tests depend on under-covers on the partition
this repository actually froze. Three defects, one theme: a quantity is reported with a sampling
distribution attached, and the distribution does not bind anything.

### 1. The published definition of sufficiency was not the shipped one

`docs/concepts/predictive-sufficiency.md`, which declares itself canonical for this test, `README.md`,
and `docs/architecture/full-buildout.md` all state that a belief is sufficient for a query when **the
upper end of the interval** on the gain falls below the declared tolerance.

`evaluation/sufficiency.py` computed `supported = gain <= tolerance` and `domain/belief.py` enforced
`PASSED if gain <= threshold` — the point estimate. The contract did not merely fail to implement the
documented rule; it *rejected* a report constructed under it. Measured on the repository's own
synthetic design, `PASSED` was returned alongside an interval whose upper end was 3.3x the tolerance,
and across 200 replications at history coefficients 0.05 through 0.12 between 1 and 81 reports per
200 passed while their own interval excluded zero on the positive side.

The direction of the error is the one that matters. Gating sufficiency on the point estimate makes it
*easier* to certify a state as sufficient, and the easiest way to obtain a point estimate near zero is
to run an underpowered comparison. The gate rewarded weak evidence.

### 2. The calibration bound was a certificate filed and never presented

`CalibrationReport.calibration_error_upper_bound` is the quantity the outcome gates on. It was a bare
scalar: `coverage_interval` was optional, was never dereferenced by the validator, and the bound was
never checked against it. An `EVALUATED`/`PASSED` report was accepted with `coverage_interval=None`
and a hand-typed bound, and also with an interval describing an unrelated quantity — point estimate
0.50, `[0.10, 0.90]`, against a reported `empirical_coverage` of 0.95.

A gate whose input can be typed by hand is not a gate. `SufficiencyReport` already cross-checks its
interval's point estimate against the reported gain; calibration had no equivalent.

### 3. The interval estimator under-covers where it is actually used

`small_cluster_scale` documented two distinct defects of the multinomial pigeonhole bootstrap at
small `K` and corrected only one. Its own docstring named the variance deficiency — a cluster's draw
count has variance `1 - 1/K` rather than one, so the resampled spread is biased low by
`sqrt((K - 1) / K)` and the bias does not vanish as resamples grow — and then applied only the
Student-`t` reference correction.

Measured on the **real** incidence of the frozen untouched-test partition (384 wells, 4 plates, 95
compound labels, 94 of which sit on exactly two plates), at nominal 0.95, 600 replications, 400
resamples:

| variance regime | `t/z` only | with `sqrt(K/(K-1))` |
| --- | --- | --- |
| plate-dominated (sd 2.0 / 0.2) | **0.908 +- 0.012** | 0.935 +- 0.010 |
| plate-only (sd 1.0 / 0.0) | **0.912 +- 0.012** | 0.943 +- 0.009 |
| balanced (sd 1.0 / 1.0) | 0.940 +- 0.010 | 0.960 +- 0.008 |
| compound-dominated (sd 0.2 / 2.0) | 0.998 +- 0.002 | 1.000 |
| plate-free (sd 0.0 / 1.0) | 1.000 | 1.000 |

Nominal 0.95 lies outside the measured interval whenever the four-plate dimension carries the
variance. The previously recorded figure of "about 0.96" was obtained at the single variance
configuration hard-wired into the test generator, not at the regime a plate-based drug screen is
expected to occupy. Capability S9 is decided by whether an interval excludes zero, so an interval
that is too narrow converts directly into superiority claims that are not earned.

## Decision

1. **The sufficiency verdict gates on the upper end of the gain interval.**
   `evaluate_history_information_gain` computes `supported = interval.upper <= tolerance`, and
   `SufficiencyReport` validates `PASSED` against the same predicate. This makes the shipped contract
   equal to the published definition, and applies ADR 0015's calibration argument to the test it was
   written alongside.

2. **An evaluated `CalibrationReport` requires its interval, and the bound is recomputed from it.**
   The validator requires `coverage_interval` when `EVALUATED`, requires its point estimate to equal
   the reported `empirical_coverage`, and requires `calibration_error_upper_bound` to equal
   `max(|lower - nominal|, |upper - nominal|, error)` — the derivation
   `evaluate_marginal_calibration` performs. `nominal_probability` is not carried on the report but is
   recoverable from coverage and error up to a sign, so the bound is checked against both candidates
   and must match one.

3. **`small_cluster_scale` applies both corrections**, becoming
   `sqrt(K/(K-1)) * t(K-1, 1-alpha/2) / z(1-alpha/2)`, and
   `BOOTSTRAP_IMPLEMENTATION_VERSION` is bumped to `2.0.0`. Endpoints produced by `1.0.0` are not
   comparable to endpoints produced by `2.0.0`.

## Schema version

Unchanged, at `"2.0"`, for the reasons ADR 0015 records and which still hold. Decisions 1 and 2 add no
fields; they tighten validators. Decision 3 changes numbers a `BootstrapInterval` carries, not its
shape. No stored artifact in this repository contains a `BeliefDiagnostics`, a `SufficiencyReport`, a
`CalibrationReport`, or a `BootstrapInterval`, and no biological backend has emitted one, so no stored
bytes change meaning. `state-query.json` remains hash-pinned at `d0fa67f3...` and continues to load.

## Consequences

**A `PASSED` sufficiency verdict now requires a measurement precise enough to support it.** This is
strictly harder to obtain and is meant to be: on the four-plate proving ground, where the corrected
interval is wide for good reason, most comparisons will not clear it. That is the correct report of
what four clusters can establish, and it is the property Phase 2 requires a state-bearing estimand to
*not* have.

**Even corrected, coverage at `K = 4` in a plate-dominated regime is about 0.94, not 0.95.** The
residual is recorded in `evaluation/bootstrap.py` rather than tuned away. Four clusters do not support
a 95 percent interval; a scaling factor chosen to make them appear to would be the defect this ADR
exists to remove.

**One limitation carries over from ADR 0015.** Decisions 1 and 2 live in validators, so they are
invisible in the emitted JSON Schema. A consumer reading the schema sees that `coverage_interval`
exists and is optional; it cannot see that an evaluated report is now refused without it.

**Test fallout was 50 tests**, all of which were assertions about the old semantics rather than
independent checks that broke. One shared belief fixture constructed a `PASSED` sufficiency report
whose interval was `+-1.0` against a tolerance of `0.1`; under this ADR that is correctly no longer a
pass, and the fixture was narrowed. `tests/conftest.py` gained `calibration_error_bound`, which mirrors
the harness derivation so no test hand-types a bound.
