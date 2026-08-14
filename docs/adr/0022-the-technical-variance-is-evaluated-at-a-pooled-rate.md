# ADR 0022: the technical variance is evaluated at a pooled rate, not at the arm's own count

- **Status:** Accepted
- **Date:** 2026-08-13

## Context

`likelihood.py` states that the observation variance carries two separable sources, and that the
second one is real:

> `psi^2` is what stops that, and it is fitted, not assumed.

**That sentence is false as shipped.** Measured on the committed slice, the fitted extra-multinomial
dispersion is **negative before clamping in all fourteen folds**, so
`max(mean(residual^2) - mean(technical), 1e-6)` returns the floor every time. `psi^2` is `1e-6` in
every fold this repository has ever fitted, which means every posterior width it has reported is
the technical term alone — precisely the failure the docstring says `psi^2` exists to prevent.

The cause is not the dispersion estimator. It is the technical term, and specifically the decision
to evaluate it at the arm's **own observed count**:

    omega_technical_j = 1/(y_j + 1/2) - 1/(n + G/2)

`1/(y + 1/2)` is a plug-in for `1/lambda`, and it is a good one only when `lambda` is large. At
`y = 0` it returns **2.0** regardless of how small the true rate is, and a gene whose expected count
is near zero is in fact nearly deterministic — it is observed as zero almost every time. So the
estimator claims its largest variance exactly where the data carry their least.

Measured against the observed residual mean-square, per count bucket, on fold `rep1`:

| count | observed residual² | claimed technical | ratio |
| --- | --- | --- | --- |
| `y = 0` | 0.3113 | **2.0000** | **6.42** |
| 1–4 | 0.3581 | 0.4708 | 1.31 |
| 5–19 | 0.2401 | 0.1074 | 0.45 |
| 20–99 | 0.1245 | 0.0239 | 0.19 |
| ≥ 100 | 0.0243 | 0.0017 | 0.07 |

The structure this table shows is the one the model assumes: above a few counts the data carry
**more** spread than sampling explains, and that excess is what `psi^2` is for. The zero bucket is
11.1% of panel entries and **79.8% of the claimed technical mass**, and it drowns the signal — the
global difference comes out at −0.164 and clamps.

This is the same defect the current docstring already documents one level up. It records that the
*naive* `1/(n p) - 1/n` form was replaced because it overstated sampling noise and drove `psi^2` to
its floor. The replacement fixed the Haldane correction and left the plug-in, so the same failure
survived in a smaller form and was not re-measured.

## Decision

1. **The technical variance is evaluated at a pooled expected rate, not at the arm's own count.**

       omega_technical_j = (n * p_j) / (n * p_j + 1/2)^2  -  1/(n + G/2)

   where `p_j` is the gene's mean composition over the **fold's fit libraries only**. This is the
   delta-method variance of `log(y + 1/2)` evaluated at the expected rate, which is what the naive
   form was always approximating. It is clipped below at zero, since the multinomial correction can
   exceed the first term for a gene with a vanishing rate.

2. **The rate is pooled globally across arms, not per target.** Both were measured; they agree to
   four decimal places on the fitted dispersion (+0.0638 either way), so the per-target variant buys
   nothing and costs a leakage surface.

3. **The rate comes from the fold, so the fold discipline is unchanged.** A held-out library
   contributes nothing to the rate used to weight its own arms. This is stated as a decision because
   a pooled quantity is exactly where leakage enters an otherwise clean leave-one-out design.

4. **`psi^2` keeps its clamp, and the clamp becomes reachable-but-not-reached rather than load
   bearing.** A fold whose dispersion still comes out non-positive is a real signal about that fold
   and must not be silently floored, so the fitted value is recorded per fold and the model card
   reports how many folds sit at the clamp. Under this decision that count is measured as **0 of
   14**; if a future change returns it to 14, the card says so.

5. **The bounds for the re-measurement are fixed here, before the re-measurement is run.** This is
   the debt `Q5` owes: its previous bound was introduced in the same commit that reported the result
   against it. These are the ledger's own thresholds, restated so that a commit can be pointed at:

   - **S5** — across-library spread at a fixed target, in the inferred state, as a fraction of
     across-target spread: **≤ 0.35**, judged on the upper end of a bootstrap interval grouped at
     the library.
   - **S2** — posterior predictive spread against the point predictor's held-out residual spread:
     **> 1**, judged on the lower end of the same interval.
   - **S4** — the declared-null contrast's interval must lie strictly below the perturbed
     contrast's.

   No result under this ADR may be reported against a threshold not written above.

6. **What this ADR does not decide.** It does not settle which point-predictor construction is the
   S2 estimand. That question is separate, it is live — the measured ratio moves from 0.22 to 0.79
   across four defensible constructions, all failing — and it gets its own record. This ADR changes
   only the variance the posterior is formed under.

## What this costs

The four measurements this repository has published all move, because every one of them reads a
posterior formed under this variance. They are pinned in
`tests/test_gse274113_observation_model.py`, so the change cannot land quietly; the pins are updated
in the implementing commit and the old values stay in the git record.

**The direction is not predicted here, and that is deliberate.** A larger `psi^2` widens every
posterior, which mechanically helps S2's numerator — and S2 is a test this model currently fails. An
ADR that both authorizes a change and forecasts the change's own benefit invites the forecast to
become the acceptance criterion. Decision 5 fixes the thresholds instead, and the result is reported
against them whatever it says. Rule 10 governs: producing the measurement is what passes the gate.

The honest residual risk: a scalar `psi^2` cannot calibrate a variance that varies with count. After
this change the claimed total still understates the low-count buckets and overstates the highest one,
where the ratio runs to roughly 2.7. That is a real limitation of the `technical + scalar` form, it
is not repaired here, and it belongs in the model card rather than in a later rediscovery.

## Consequences

- `technical_variance` changes signature: it needs the pooled rate, so it can no longer be computed
  from `(counts, depth)` alone. `FittedFold` carries the rate, which keeps the fold the single
  owner of everything a held-out arm is scored against.
- The fitted dispersion becomes a genuine per-fold quantity, so the belief's uncertainty breakdown
  separates measurement from biology as the contract has always claimed it does.
- `likelihood.py`'s "it is fitted, not assumed" becomes true, and the model card's item 6 — which
  currently records the clamp as a known defect — is rewritten to record the fitted values instead.
- `Q5`'s predeclaration clause, retracted as unmet in the roadmap, is satisfied for the re-run by
  decision 5. The original run stays recorded as measured against a post-hoc threshold; this does
  not retroactively repair it.
