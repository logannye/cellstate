# ADR 0025: S6 is gated on the whole coherent nominal interval, not at one level

- **Status:** Accepted. Supersedes [ADR 0024](0024-s6-is-measured-and-readiness-is-derived.md)
  decision 3; that record's other decisions stand.
- **Date:** 2026-08-15

## Context

ADR 0024 gated S6 at a single nominal probability, 0.90, on the reasoning that the predeclared pair
`minimum_calibration_coverage=0.85` and `maximum_calibration_error=0.05` forced it. **That reasoning
was wrong**, and its correction is annotated on that record. The pair is coherent on the closed
interval **[0.90, 0.95]**; 0.90 is its minimum, and the only thing unique about it is that the floor
and the error bound's lower edge coincide there.

A wording correction would not have been worth a record. This is worth a record because of what the
error let through.

**The ledger had already asked for this.** `docs/roadmap.md` defines S6 as

> Absolute marginal coverage error is within a predeclared threshold **at every declared level**, as
> an upper confidence bound grouped at the split unit.

That sentence predates ADR 0024. Gating at one level satisfied it only because that ADR had declared
one level, on reasoning that turned out to be wrong — the letter was met by shrinking what was
declared. This record does not add a requirement; it stops discarding five levels the predeclared
thresholds already implied.

**A gate at one level is clearable by a constant.** Multiply every predictive standard deviation by
a scalar — no mechanism, no fitted quantity, no claim about the biology — and the shipped gate
opens:

| sd × | 0.90 | 0.91 | 0.92 | 0.93 | 0.94 | 0.95 |
| --- | --- | --- | --- | --- | --- | --- |
| 1.00 | 0.0548 | 0.0590 | 0.0641 | 0.0660 | 0.0725 | 0.0767 |
| 1.05 | **0.0455** | **0.0475** | 0.0545 | 0.0625 | 0.0667 | 0.0673 |
| 1.11 | **0.0368** | **0.0425** | **0.0473** | 0.0518 | 0.0566 | 0.0615 |
| 1.20 | **0.0444** | **0.0416** | **0.0358** | **0.0415** | **0.0486** | 0.0504 |
| 1.21 | **0.0480** | **0.0414** | **0.0358** | **0.0407** | **0.0463** | **0.0475** |
| 1.25 | 0.0514 | **0.0428** | **0.0371** | **0.0363** | **0.0404** | **0.0475** |

(Bold clears the 0.05 threshold. Upper confidence bound on the calibration error.)

**Eighteen scalars in `[1.04, 1.21]` clear the level ADR 0024 gated at.** At 1.11 the coverage lands
on exactly 0.9000 with a bound of 0.0368 — *better than the shipped 0.0548, and better than any
mechanism-based repair measured on this slice.* Across the whole interval, exactly **one** scalar in
`[1.00, 1.80]` clears every level.

That razor-thin window is the point. **A one-level gate tests the residuals' scale, and scale is
free. Six levels test their shape.** Since several proposed repairs to this model — a `psi^2`
correction, a heavier-tailed observation model — are approximately uniform rescalings, a one-level
gate would have handed this project its first ledger-adjacent PASS for doing nothing.

This repository has twice found the opposite defect: a threshold a *correct* computation cannot fail
(`maximum_ood_score=0.99`; `maximum_calibration_error=1` in `examples/estimate_state.py`). This is
its mirror — a threshold a *wrong* computation passes — and it is the harder one to notice, because
it looks like a gate that works right up until something clears it.

## Decision

**1. S6's gate is the conjunction over every level in `S6_NOMINAL_PROBABILITIES`.**
`measure_calibration_level_set` is the gate. `measure_calibration_coverage` reads a single level and
its docstring now says it is not the gate.

**2. The interval is derived; the grid spacing is a declared choice, and the two are named
separately.** `S6_NOMINAL_INTERVAL = (0.90, 0.95)` follows from thresholds written before any
coverage number existed. `S6_NOMINAL_GRID_STEP = 0.01` follows from nothing and is recorded as a
decision. A future reader must be able to tell which half was derived; the superseded record's error
was exactly the loss of that distinction.

**3. The belief publishes the reference level, and the reference level is never the verdict.** A
`CalibrationReport` carries one coverage, so a six-level gate still has to choose which level's
numbers travel with a belief. That is `S6_REFERENCE_NOMINAL = 0.90` — the interval's one
distinguished point, and the choice that keeps the published figures stable across this change.

**4. A reference level that disagrees with the conjunction is refused, not resolved.**
`measure_calibration_level_set` raises if the reference report's outcome differs from the
conjunction. Which level a belief should publish once the levels disagree is an ADR's decision, not
a branch's. The guard cannot fire on the committed slice — all six fail together — so it is
exercised from the side that fires by `test_the_reference_level_may_not_disagree_with_the_conjunction`,
which constructs the case with the same 1.11 rescaling.

**5. No estimator changes and no published number is re-pinned.** `fit.py` and `likelihood.py` are
untouched. The reference level's coverage (0.8836), interval ([0.8452, 0.9220]) and bound (0.0548)
are what they were.

## What it measures

| nominal | coverage | error | bound | outcome |
| --- | --- | --- | --- | --- |
| **0.90** (reference) | **0.8836** | 0.0164 | **0.0548** | fails |
| 0.91 | 0.8886 | 0.0214 | 0.0590 | fails |
| 0.92 | 0.8943 | 0.0257 | 0.0641 | fails |
| 0.93 | 0.9000 | 0.0300 | 0.0660 | fails |
| 0.94 | 0.9043 | 0.0357 | 0.0725 | fails |
| 0.95 | 0.9093 | 0.0407 | 0.0767 | fails |

**Conjunction: FAILED, at all six.** The verdict is unchanged from ADR 0024; the evidence behind it
is six times as much.

## Consequences

**The gate is now harder in a way that is not arbitrary.** Every level follows from the same
predeclared pair. Widening from one level to six does not introduce a new threshold — it stops
discarding five that were already implied.

**Three of the queued repairs must now clear a shape test.** The `psi^2` degrees-of-freedom
correction and a Student-t observation model both act on this evidence as approximately uniform
rescalings; under ADR 0024's gate either could have flipped S6 to PASSED. Whether they still can is
now a question about the residuals' shape, which is the question S6 exists to ask.

**The `1.11` scalar is retained as a test, not as a repair.** It is not a proposal; it is the
cheapest possible wrong answer, kept so the gate is exercised by something that should not pass it.

## What this does not do

- It does not change any capability verdict. The ledger stays 0 of 10 and S6 stays FAILED.
- It does not repair the tail, the depth gradient, or `psi^2` — all still recorded and unrepaired
  under ADR 0024.
- It does not make the other six readiness criteria evaluable. S6 remains the only one measured, and
  `abstention_required` remains pinned `True` by the other six.
