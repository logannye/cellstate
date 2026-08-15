# ADR 0024: S6 is measured, and readiness is derived from the criteria rather than declared

- **Status:** Accepted.
- **Date:** 2026-08-15

## Context

`GSE274113ObservationEstimator` emitted a belief whose seven readiness criteria all read
`NOT_EVALUATED`, and whose `abstention_required` was the literal `True`. Twenty occurrences of
`NOT_EVALUATED` in one file.

The calibration slot carried a forward reference:

> `notes=("calibration is reported by the fold's held-out report, not per arm",)`

**No such report existed anywhere in the repository.** Grepping the phrase across `src` and `docs`
returned one hit: the note itself.

Meanwhile `evaluation/calibration.py` was finished work — grouped cluster bootstrap, a one-sided
upper confidence bound, and an explicit refusal to pass on a point estimate whose bound sits outside
threshold. It had exactly one caller: `tests/test_q2_faithfulness_harnesses.py`, on synthetic
arrays. A capability harness that had never been pointed at the ship.

So the belief this project exists to emit had never been scored. Everything measured to date scores
the **substrate** (`knockdown`, `spectrum`, `day`) or the **decomposition** (S2, S4, S5). Nothing
scored the output object.

There was a second problem with `abstention_required = True`, and it is the more serious one.
It was the correct answer, and it was not a measurement. Nothing computed it, so nothing could have
computed it differently, and from outside there was no way to distinguish a belief that abstains
from a belief that cannot do anything else. A field that cannot come out the other way is the defect
this repository keeps finding in other people's code, sitting in the field the whole contract turns
on.

## Decision

**1. S6's estimand is gene-space posterior-predictive coverage on the ADR 0023 split-half
replicate.**

For library *L*, in the fold that excluded *L*: infer the state from `NT_A`, form the predictive
distribution for `NT_B` at `NT_B`'s own depth, and count the panel genes whose observed
log-composition lands inside the interval.

    mean = intercept + design @ u_hat
    var  = diag(design @ Sigma @ design.T) + observation_variance(depth_B)

This is deliberately the *same construction* ADR 0023 decided for S2, so S2 and S6 are two readings
of one evidence set rather than two estimands whose disagreement could be blamed on the setup.

**2. The independent experimental unit is the library, and only libraries are resampled.**

Coverage is counted over the 100 panel genes *within* a library; the cluster bootstrap resamples the
14 libraries. Genes in one panel are not 100 independent trials — they are compositionally
constrained and co-expressed — so pooling them into one fraction and bootstrapping that would claim
a precision this design cannot buy.

**3. The nominal probability is 0.90, and it is forced rather than chosen.**

`arm_request.py` predeclared `minimum_calibration_coverage=0.85` and
`maximum_calibration_error=0.05` before any coverage number existed. Together they require the
coverage to lie within 0.05 of nominal *and* to clear 0.85, which coincide at exactly one nominal.

This matters more than usual here. This repository already carries two thresholds a correct
computation could not fail — `maximum_ood_score=0.99`, and `maximum_calibration_error=1` in
`examples/estimate_state.py`. Leaving the nominal free would have added a third: at 0.95 the
measured coverage fails the floor outright, at 0.85 it passes comfortably, and either could have
been justified after the fact.

**4. The criteria are read from the diagnostics report, and `abstention_required` is computed from
them.**

`_readiness` now takes the `BeliefDiagnostics` that `coherent_contract` checks it against, reads
each criterion off it, and derives abstention as "not every criterion PASSED". `measurement_model`
remains the stated exception, for the reason already recorded: it is the one criterion with no
diagnostics counterpart, so it is the one that can assert itself.

**5. S6 is computed, never stored.**

Fitting all fourteen folds and bootstrapping takes about 0.2 s on the committed slice. Committing
the number as an artifact would recreate the failure mode this project keeps finding — a recorded
claim and the code that would check it drifting apart with nothing going red. There is nothing here
to drift.

## What it measures

| | value |
| --- | --- |
| empirical coverage at nominal 0.90 | **0.8836** |
| coverage interval (cluster bootstrap, 14 libraries) | [0.8452, 0.9220] |
| calibration error | 0.0164 |
| **upper confidence bound on the error** | **0.0548** |
| predeclared maximum | 0.05 |
| **outcome** | **FAILED** |

**S6 fails on the bound alone.** The point estimate clears the 0.85 floor and its error of 0.0164 is
comfortably inside 0.05. A criterion that reported a point estimate would have called this
calibrated. The interval reaches 0.8452, so the error could plausibly be 0.0548, and ADR 0015's rule
— gate on the bound — is what turns that into a verdict.

The margin is thin, so it was checked: across eight bootstrap seeds the bound ranges 0.0532–0.0565
and the outcome is FAILED in all of them.

## The finding that changes what the repair should be

S2 on this same evidence is 0.8415 — an RMS ratio of claimed spread to realized error. That reads as
*"the interval is uniformly about 16 percent too narrow"*, and the repair it implies is a larger
`psi^2`.

**That reading is wrong, and coverage is what shows it.** Standardizing the residuals and trimming:

| outcomes removed | sd of z | implied S2-style ratio |
| --- | --- | --- |
| none (1,400) | 1.2848 | 0.7784 |
| worst 1% (14) | 1.1121 | 0.8992 |
| **worst 2% (28)** | **1.0045** | **0.9955** |
| worst 5% (70) | 0.8178 | 1.2228 |

**Twenty-eight of 1,400 gene-library outcomes carry the entire shortfall.** With them removed the
posterior's spread is exactly earned. The bulk of the panel is *better* than the interval claims —
coverage at nominal 0.50 is 0.66 — and a handful of genes are catastrophically outside it, with a
maximum standardized residual of 9.47.

The two diagnoses call for opposite repairs. Inflating `psi^2` by the ~1.19x the ratio suggests would
push the already-conservative 98 percent into over-coverage while still falling far short of a
9.5-sigma outlier. What the shape actually indicates is a heavier-tailed observation model, or an
identification of which genes blow up. **Neither is decided here**; this ADR records the
decomposition, not the repair.

## The gradient S6's pooled number hides

Per-library coverage is not homogeneous. It runs monotonically against panel depth:

| library | replicate depth | coverage @ 0.90 |
| --- | --- | --- |
| rep3 | 225,374 | 0.930 |
| rep1 | 274,375 | 0.940 |
| rep5 | 493,919 | 0.980 |
| rep9 | 761,048 | 0.850 |
| rep13 | 1,277,246 | 0.760 |
| rep14 | 1,423,613 | 0.780 |

`corr(log depth, coverage) = -0.857` over 14 libraries.

This is the exact failure `likelihood.py` names `psi^2` as the defence against — *"the failure mode
where more sequencing depth is mistaken for more knowledge about the biology"*. ADR 0022 stopped
`psi^2` being clamped in all fourteen folds; it did not stop this. The technical share of the
observation variance falls from 0.55 to 0.35 across the depth range while `psi^2` stays near 0.055,
so the interval narrows with depth and the biology does not.

⚠️ **Depth and differentiation day are collinear in this deposit and the design cannot separate
them.** Panel depth rises with day by construction, and within a day the depth range is too narrow
to resolve anything — the three day-11 libraries span 720k–768k and their coverages span 0.83–0.85.
The gradient is real. Which of the two drives it is **not identified here**, and no amount of
re-analysis of this deposit will identify it.

## Consequences

**S6 is the first of the seven criteria to become a measurement.** It is not the last, and the
belief now says so: `reasons` carries an explicit `criteria not met:` line naming the other six.

**Abstention remains `True` and remains structurally pinned.** Six criteria are still
`NOT_EVALUATED`, so no configuration of S6 alone can release it. What changed is that abstention is
now a function of the criteria rather than an assertion beside them, and
`test_the_readiness_derivation_responds_when_the_calibration_passes` exercises the branch by
loosening the error threshold until S6 passes and watching the reason and the unmet list both move.

**A circular import was introduced and fixed while doing this.** Wiring S6 into the belief made
`backends.gse274113.usage` import `evaluation.gse274113_reports`, which imports back into
`backends.gse274113` through its `__init__`. The whole suite stayed green — 34 backend tests passed
— while `import cellstate.evaluation.gse274113_reports` as a program's first import raised
`ImportError`, because every existing test reaches these modules through an order that happens to
resolve. `evaluation` measures `backends`; nothing under `backends` may import `evaluation` at
module scope. `test_every_module_imports_first` spends a subprocess per module to check it, and was
run once with the cycle reinstated to confirm it fires.

**A silent basis truncation was found by the previous change and refused by this one's neighbour.**
Unrelated to S6, recorded for the same reason: `fit_fold` returned a 13-column basis for
`nuisance_rank=20` without saying so.

## What this does not do

- It does not establish calibration on a **perturbed** arm. `NT` is the only arm in this design
  carrying a replicate, so S6 inherits ADR 0023's scope unchanged: null biology, 14 libraries, not
  280 arms. A cross-library construction would reach the perturbed arms, and it would be charged for
  nuisance variation it is structurally forbidden to know — the exact handicap ADR 0023 decision 3
  superseded `measure_point_predictor_spread` for. It is not attempted here.
- It does not repair the tail, the depth gradient, or `psi^2`.
- It does not make the belief usable. The ledger stands at 0/10 and this changes it to 0/10 with one
  criterion measured instead of zero.
