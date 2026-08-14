# ADR 0023: the S2 estimand is a split-half replicate on the declared-null arm

- **Status:** Accepted. **Decision 4's two arithmetic figures are corrected below; the decision
  itself stands.** The body is left as written, per the rule that an accepted ADR's body is a
  historical record.
- **Date:** 2026-08-14

> **Correction (decision 4, the depth asymmetry).** Neither figure in decision 4 reproduces from
> the committed slice, and both are now computed by
> `test_the_s2_depth_caveat_reproduces`:
>
> - The shortfall is **exactly 2.00×**, not 2.06×. `NT` is the bitwise sum of `NT_A` and `NT_B` in
>   all fourteen libraries, so each half is exactly half the arm — 684,370 panel counts on average
>   against 1,368,741. The 665,763 quoted in the body is the depth of `NT_B` *alone*, and quoting
>   one half as "the halves" is what produced 2.06.
> - The recomputation at full-arm depth gives **0.7727, about 8.2%** — not 0.8322, about 1.1%.
>   0.8322 is what a 1.1× depth multiplier returns, not the 2.0× the sentence names.
>
> **The decision is unaffected and its conclusion is strengthened.** The move is still in the
> direction that makes the failure *worse*, and by seven times the stated margin: the depth
> asymmetry does not explain S2's failure, it deepens it. What failed was the ADR's own standard —
> "a caveat that is only ever stated is a caveat nobody has checked" — applied to the caveat that
> sentence appears in.

## Context

S2 requires the posterior's spread to be **earned** — strictly wider than a point predictor's
residual spread on held-out units. The ledger fixes the threshold and the direction. It does not
fix *which point predictor*, and neither did anything else in this repository.

`measure_earned_spread` shipped one. It takes every coefficient from the target's mean over the fold's
fit libraries, and that choice was written rather than decided. [ADR 0022](0022-the-technical-variance-is-evaluated-at-a-pooled-rate.md)
decision 6 recorded it as open and deferred it here.

**Two free parameters were being set silently, not one.** The first is what the point predictor is
allowed to condition on. The second is how the per-gene spreads are aggregated into a scalar — the
shipped statistic pairs a *mean* of per-gene standard deviations against a *root-mean-square*
residual, which is not like-for-like: by Jensen's inequality the mean of the square roots is the
smaller of the two, so the numerator is systematically understated relative to its denominator.

Both were swept on the committed slice, post-ADR-0022, with intervals grouped at the library:

| | point predictor conditions on | mean-of-sd (as shipped) | RMS-of-sd |
| --- | --- | --- | --- |
| **A** | the target's mean over fit libraries | 0.2781 [0.2072, 0.3467] | 0.3021 [0.2253, 0.3758] |
| **B** | A, plus the arm's own nuisance block | 0.8013 [0.6942, 0.9008] | 0.8708 [0.7499, 0.9834] |
| **C** | A, plus its own nuisance and realization | 0.8221 [0.7131, 0.9215] | 0.8934 [0.7715, 1.0065] |
| **D** | its own least-squares fit, all three blocks | 0.9908 [0.8541, 1.1282] | 1.0773 [0.9221, 1.2326] |
| **E** | one half of the arm, scored on the other half | 0.7718 [0.6579, 0.8923] | 0.8415 [0.7144, 0.9759] |

**Every construction fails**, so the verdict was never in question. The reported *number* moves by
3.9× across the grid, and the shipped combination sits at the extreme pessimistic corner of it.

A is handicapped twice over. Its predictor never sees the arm — while the posterior it is compared
against does — and averaging coefficients across fit libraries washes out the library nuisance block,
so the predictor is then charged for library variation it was structurally forbidden to know. D
removes both handicaps and is the ceiling: a least-squares fit to the arm's own composition leaves the
smallest residual any predictor on this design can leave. That the ceiling lands at 1.08 with an
interval reaching down to 0.92 is the substantive finding — **the ratio's entire admissible range sits
at or below the threshold.**

None of A–D is a clean test, because none of them makes the two sides condition on the same
information. `evaluation/gse274113_reports.py` has said so since it was written:

> A clean S2 test needs a held-out *replicate of the same arm* — split the arm's cells, infer from
> one half, predict the other — which this design supplies only for NT.

The design supplies exactly that, on all fourteen libraries, and no code did it.

## Decision

1. **The S2 estimand is the split-half replicate on the declared-null arm — construction E.** The
   state is inferred from `NT_A`, the posterior predictive is formed for `NT_B`, and the claimed
   spread is compared against the error actually realized on `NT_B`. Both sides see one half and are
   scored on the other, so the comparison is like-for-like and the standing "the posterior conditions
   on the arm and the predictor does not" caveat is retired rather than restated.

   **The fold discipline survives this, and that was checked rather than assumed.** `NT`'s direction
   is estimated from the placebo split, so an estimand built on that same split invites the question
   of whether the held-out library's halves informed the design they are then scored under. They do
   not: `fit.py` draws `placebo_libraries` from `fit_libraries` alone. For a fold excluding library
   *L*, the intercept, both subspaces, `psi^2`, the pooled rate and the `NT` direction all come from
   the other thirteen libraries, and **both** halves of *L*'s `NT` arm are genuinely held out. Had
   that gone the other way the estimand would have been inadmissible, and a split-half test is
   precisely the shape where such a leak would be easy to miss and would flatter the result.

2. **Per-gene spreads are aggregated as a root-mean-square on both sides.** Comparing a mean of
   standard deviations against an RMS residual mixes two aggregations and understates the numerator;
   on construction E it costs 8.0% (0.7718 against 0.8415). Whichever convention is chosen, it must be
   the same one on both sides of the ratio, and RMS is the one that keeps the statistic a comparison
   of variances.

3. **Construction A is retained and reported as a labelled diagnostic, not deleted.** It is the
   published number and the record must stay legible; a silent replacement would make the change
   invisible in exactly the way [ADR 0021](0021-an-admissible-kind-for-a-fitted-observation-model.md)
   warns about. It is reported as what it is — a point predictor denied the arm's library — and it is
   no longer the S2 verdict.

4. **The depth asymmetry is quantified here rather than carried as a qualifier.** The `NT` halves
   average 665,763 panel counts against 1,368,741 for the full arm, a 2.06× shortfall, and lower depth
   inflates both the claimed spread and the realized error. Recomputing the ratio with the technical
   term evaluated at the full-arm depth on both sides moves it from **0.8415 to 0.8322** — about 1.1%,
   and in the direction that makes the failure *worse*, not better. **The depth asymmetry does not
   explain the result.** This follows the treatment S4's 1.16× contrast-noise asymmetry received: a
   caveat that is only ever stated is a caveat nobody has checked.

5. **The scope limit is declared.** `NT` is the only arm in this design with a replicate, so E measures
   the posterior's calibration on **null biology only**, across 14 units rather than 280. It does not
   establish calibration on a perturbed arm. On this substrate that costs less than it appears to —
   mean on-target knockdown is −0.043 log2 fold-change — but the limitation is a property of the
   estimand and belongs in the model card, not in a footnote to one measurement.

6. **The threshold is not set by this record and is not available to be set by it.** S2's requirement
   that the ratio exceed one, decided on the *lower* interval end, comes from the ledger and predates
   all of this. What this ADR fixes is the estimand.

   **The ordering is stated plainly: the sweep above was run before this decision, so the estimand is
   chosen with the numbers visible.** That is the opposite of the discipline ADR 0022 decision 5
   applied, and it is admissible only because the choice cannot be number-shopping — the constructions
   all fail, and **E is not the most favourable of the five.** D is, under both aggregations, and D is
   the one being declined. The full grid is recorded above so that a reader can check that rather than
   take it on trust.

7. **What this ADR does not decide.** It does not decide whether the misfit E measures is repairable,
   or by what. It does not construct a perturbed-arm replicate — this design has none, and
   manufacturing one by splitting a perturbed arm's cells would measure the same population twice and
   test nothing about the perturbation. It does not revisit S4 or S5, whose estimands are unaffected.

## What this measures, and the diagnosis it supports

Under the decided estimand, decomposed across the fourteen folds:

| term | value | share of claimed |
| --- | --- | --- |
| parameter uncertainty, `A Sigma A'` | 0.00897 | 7.4% |
| `psi^2`, biological | 0.05785 | 47.7% |
| technical, at the half's depth | 0.05444 | 44.9% |
| **total claimed variance** | **0.12126** | |
| realized residual² on the held-out half | 0.19711 | |
| of which technical | 0.05444 | 27.6% |
| **implied systematic misfit²** | **0.14267** | |

**The misfit alone exceeds everything the posterior claims.** With all sampling noise removed, the
model's systematic error on a genuine replicate is 0.1427 against a total claimed variance of 0.1213.
That is the diagnosis S2 was supposed to deliver and, under construction A, could not: A's failure was
consistent with a merely unfair comparison, and this one is not.

## What this costs

- **The published S2 number changes, 0.28 → 0.84.** The verdict is unchanged — the capability is not
  advanced, and the interval's lower end is 0.71 — but the claim it supports is different. "The
  posterior is 3.6× narrower than a predictor's residual spread" becomes "the posterior claims a
  spread about 16% narrower than the error it actually makes." The second is a statement about
  calibration; the first was substantially a statement about how the predictor was handicapped.
- **The unit count drops from 280 arms to 14 libraries**, and the interval widens accordingly. This is
  not a real loss of information: the 280 arms were never 280 independent units, program rule 8 binds
  the bootstrap at the library, and both estimands resolve to K = 14 clusters. What is lost is the
  within-library averaging, and the wider interval is the honest price of the cleaner comparison.
- **Calibration is established on null biology only**, per decision 5.

## Consequences

- `measure_earned_spread` is re-implemented against the decided estimand, and the pinned test values
  change. The pins move deliberately and visibly, which is what
  [PR #28](https://github.com/logannye/cellstate/pull/28)'s pinning was for.
- The model card's S2 qualification is rewritten. Its current text quotes 0.65, 0.67 and 0.79 for
  B, C and D; those are pre-ADR-0022 and are now 0.87, 0.89 and 1.08 under the decided aggregation.
  **Stale alternatives are worse than no alternatives**, because they read as a sensitivity analysis
  that has been maintained.
- `Q7`'s roadmap entry records both numbers and the reason the estimand changed.
- The docstring caveat at `gse274113_reports.py:9-19` is rewritten: it currently explains why the
  comparison is not like-for-like, and under this decision it is.
