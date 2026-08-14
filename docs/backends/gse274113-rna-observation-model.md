# GSE274113 RNA observation model — model card

The first model in this repository to compute a representation of cell state from real cells.
Program rule 5 requires every model to declare its scope; this is that declaration, and the parts
that say what the model **cannot** do are the load-bearing ones.

- **Model id:** `gse274113-rna-observation-model`, version `1.0.0`
- **Artifact kind:** `empirical_observation_model` ([ADR 0021](../adr/0021-an-admissible-kind-for-a-fitted-observation-model.md))
- **Authorized by:** [ADR 0019](../adr/0019-build-the-representation-on-held-evidence.md) (build on
  held evidence), [ADR 0020](../adr/0020-rna-first-nuisance-separation.md) (RNA first),
  [ADR 0021](../adr/0021-an-admissible-kind-for-a-fitted-observation-model.md) (the artifact kind)

## What it claims

A **snapshot state estimate with a posterior**, for one `(library, target)` population of
CRISPRi-perturbed human CD34+ haematopoietic progenitors at its own harvest instant.

It claims nothing else. In particular it is **not** a sufficiency result, **not** a faithfulness
verdict, and **not** evidence that the state is complete. ADR 0021 decision 5 governs, and the
belief enforces it in its own fields rather than in prose: `causal_support` is `UNSUPPORTED`,
`sufficiency` is not evaluated, `dynamics.velocity` is an `UnavailableDistribution`, and
`readiness.abstention_required` is true with stated reasons.

## Data card

| | |
| --- | --- |
| Source | `GSE274113` Perturb-Multiome, RNA half only |
| Provenance | 15 artifacts re-fetched from GEO and matched on byte count and SHA-256, 15 of 15 |
| Licence | **Unresolved at source, recorded**: no terms asserted by submitter or journal, owner-authorized for internal method development, **not cleared for any published biological claim** (ADR 0020 decision 4) |
| Cells | 137,604 annotated, 0 dropped for a zero panel total |
| Arms | 280 real `(library, target)`, all populated; plus 28 NT placebo arms |
| Panel | 100 genes: 19 CRISPRi target TFs, haematopoietic lineage markers, housekeeping depth anchors |
| Depth per arm | 192k to 3.0M panel counts, median 584k |
| Experimental unit | the **library** (program rule 8) |

**The ATAC half is deliberately unused.** Its feature space is 14 distinct peak sets whose size
tracks differentiation time at `r` = −0.973, so the observation space would be a function of the
biological variable under study. See [ADR 0020](../adr/0020-rna-first-nuisance-separation.md).

## The model

Conjugate Gaussian on the Haldane-corrected log-composition, so the posterior is closed-form —
exact, deterministic, and with no sampler that can return a mode dressed as a posterior.

```
c   = log((y + 1/2) / (n + G/2))
c   = alpha + [W | V | u_g] u + eps,   eps ~ N(0, Omega)
Omega = diag( lam/(lam + 1/2)^2 - 1/(n + G/2) )  +  psi^2,   lam = n * p_pooled
```

The eight-dimensional state is `[biology(4) | library nuisance(3) | guide realization(1)]`.

**Where the split is attempted, and why it is not where S5 is decided.** `V` is the leading subspace
of the NT arms across libraries — NT is the same biology everywhere, so what moves there is the
library. `W` is the leading subspace of *within-library* contrasts — differencing inside a library
cancels the library, so what moves there is the perturbation. An earlier revision of this card called
that "where S5 is won structurally." It is not: a construction cannot win a capability whose
definition is a held-out measurement, and when the measurement was run it failed at 10.36 against a
bound of 0.35. The construction does what it says — the nuisance block absorbs 134× more library
variance than leaks past it — and S5 still fails, because `W` is fitted from contrasts that sit near
the sampling floor. See the ledger section below.

`p_pooled` is the panel composition averaged over the fold's fit libraries, never the arm's own
counts ([ADR 0022](../adr/0022-the-technical-variance-is-evaluated-at-a-pooled-rate.md)).

**The declared-null intervention is estimated, not assumed.** `u_NT` comes from a deterministic
within-library placebo split of the NT cells. A structurally-zero NT column would make the S4 null
half unfailable, which is the inert-`do` defect ADR 0019 names explicitly.

## Declared biases and limitations

1. **`W` is orthogonalized against `V`.** Biology genuinely aligned with the library axis is charged
   to nuisance. This biases *against* finding biology, but it is a choice, not a neutrality.
2. **The subject is the annotated cells, not the underlying culture.** Sampling fraction is
   therefore 1.0 by definition. Generalizing to the culture is a transport claim, and the query sets
   `allow_transport=False` rather than making it.
3. **Guide-calling error is absorbed into the subject definition** and is not modelled.
4. **Realized knockdown efficiency is `UNKNOWN`.** Which guide a cell carries is deposited; how
   effectively the target was repressed is not, and this model does not re-derive it. The panel
   carries all 19 target genes, so a measured efficiency is a real follow-up.
5. **The delta-method Gaussian is poor at very low counts.** Accepted deliberately; the realized
   depth per arm is recorded above so a reader can judge it.
6. **`psi^2` is fitted in all 14 folds, and the count of clamped folds is reported rather than
   assumed.** It ranges 0.0517 to 0.0620 across the folds and reaches its `1e-6` floor in **0 of
   14**. This was not true until [ADR 0022](../adr/0022-the-technical-variance-is-evaluated-at-a-pooled-rate.md):
   evaluating the technical term at the arm's own count made `1/(y + 1/2)` return 2.0 at every zero
   entry, so 10.6% of panel entries carried 79.6% of the claimed technical mass, the fitted
   dispersion came out **negative in every fold**, and it clamped. Every posterior width this model
   reported before that decision was technical-only — the exact failure `likelihood.py` says
   `psi^2` exists to prevent. `FittedFold.dispersion_is_clamped` exposes the pre-clamp value so a
   return to that state is visible.
7. **A scalar `psi^2` cannot calibrate a variance that varies with count**, and this is not
   repaired. After ADR 0022 the claimed total still understates the low-count buckets and
   overstates the highest, where the ratio runs to roughly 2.7. It is a limitation of the
   `technical + scalar` form rather than of the fit.
8. **Day-14 selection.** Annotation rate is 98–99% at days 7, 9 and 11 and **62%** at day 14, and
   the excluded day-14 barcodes are real cells rather than empty droplets. Composition also shifts
   with perturbation over time. Beliefs about day-14 libraries rest on a differently-selected
   population, and this is not modelled.

## Ledger capabilities

**Advanced, on held evidence: none.**

S5, S2 and S4 were measured on held-out libraries after this card was first written, with intervals
grouped at the library, and **all three failed**. The measurements are computed by
`evaluation/gse274113_reports.py` and reproduce from the committed slice.

| | Measured | Interval | Required | |
| --- | --- | --- | --- | --- |
| S5 nuisance separation | 10.36 | [6.27, 16.66] | ≤ 0.35 | fails |
| S2 earned spread | 0.84 | [0.71, 0.98] | > 1 | fails |
| S4 null half (placebo) | 2.03 | [1.44, 2.67] | below the perturbed band | fails |
| S4 non-null half | 2.09 | [1.62, 2.56] | above the null band | fails |

These are the values **after** [ADR 0022](../adr/0022-the-technical-variance-is-evaluated-at-a-pooled-rate.md),
measured against bounds that decision fixed before the run. The pre-ADR values were 19.22, 0.22,
2.90 and 3.08, measured against a bound introduced in the same commit as its own result.

S2's row also reflects a change of **estimand** rather than only of variance, per
[ADR 0023](../adr/0023-the-s2-estimand-is-a-split-half-replicate.md): it was 0.28 under the
superseded point-predictor construction and is 0.84 under the split-half replicate. The superseded
construction is still computed and reported, as `measure_point_predictor_spread`, at 0.30 [0.23,
0.38] under the decided aggregation. See the first qualification below.

An earlier revision of this section declared **S5 and S2 advanced**, on the structural argument that
the nuisance axis is a declared subspace and that the posterior's components are computed rather
than declared. That argument was wrong in the way [ADR 0021](../adr/0021-an-admissible-kind-for-a-fitted-observation-model.md)
warned it would be: a *construction* is not a *measurement*, and the measurement went the other way.
The claim is withdrawn. ADR 0021 decision 5 requires the advanced entries to be named here, and the
honest list is empty.

Two qualifications, so that these failures are not read as more than they are:

- **S2's estimand is now fixed, and the reported number changed when it was.**
  [ADR 0023](../adr/0023-the-s2-estimand-is-a-split-half-replicate.md) makes S2 a **split-half
  replicate on the declared-null arm**: the state is inferred from `NT_A`, the predictive is formed
  for `NT_B`, and the claimed spread is scored against the error actually realized. Both sides
  condition on the same information, so the "the posterior has seen the arm and the predictor has
  not" caveat this card used to carry is retired rather than restated.

  The previously published **0.28** came from a point predictor denied both the arm and its own
  library's nuisance coefficients, and it sat at the extreme pessimistic corner of a swept grid:
  across five constructions and two gene-aggregation conventions the ratio runs **0.30 to 1.08**,
  post-ADR-0022, with intervals grouped at the library. **Every construction fails**, so the verdict
  never depended on the choice — but the shortfall's *size* did, and the honest figure is 0.84 rather
  than 0.28. The full grid is in ADR 0023. That record also fixes the aggregation: both sides are a
  root-mean-square, where the superseded form paired a *mean* of per-gene standard deviations against
  an RMS residual and understated the numerator by 8.0%.

  Two limits belong with the new number. It measures calibration on **null biology only** — `NT` is
  the sole arm carrying a replicate — across 14 libraries rather than 280 arms. And the halves are
  shallower than a full arm, 665,763 panel counts against 1,368,741; evaluating the technical term at
  the full-arm depth moves the ratio from 0.8415 to **0.8322**, about 1.1% and in the direction that
  makes the failure worse, so **the depth asymmetry does not explain it.**

  What does: with sampling noise removed, the systematic misfit is **0.1427** against a total claimed
  variance of **0.1213**. The misfit alone exceeds everything the posterior claims. Under the old
  construction that diagnosis was unavailable, because its failure was equally consistent with a
  predictor that had simply been handicapped.
- **The substrate carries almost no perturbation signal**, so S2, S4 and S5 are verdicts on
  `GSE274113`'s CRISPRi arm before they are verdicts on this model. Mean on-target knockdown across
  the 19 targets is **−0.043** log2 fold-change and 6 of 19 move the **wrong way**; restricted to the
  15 targets detected above 200 panel-CPM it is **−0.094**. SNAI2 (0 CPM) and PRDM16 (3 CPM) are not
  expressed at all, so two targets are unmeasurable in this readout. A working CRISPRi knockdown is
  roughly −1 to −2. All three capabilities divide by or compare against a between-target biology
  variance of **0.257**. ⚠️ These knockdown figures are measured from the committed slice but **no
  committed runner computes them**; they are a recorded claim, not a checked one, and carry the same
  standing as the census in the representability ledger.

S5's failure is the one that does bear on the model, and its diagnosis is not the obvious one.
Decomposed by block, across-library variance is **81.30** in the nuisance block against **0.609** in
biology, so the nuisance basis absorbs roughly 134× more than leaks past it. What fails is the
denominator: between-target variance is 0.109, smaller than the residual library variation the
biology block carries. Raising the nuisance rank does not repair this and was measured not to.

ADR 0022 moved both terms and the ratio hid it: library variation leaking into the biology block
fell **5×** (3.07 → 0.609), a real improvement in exactly the separation S5 names, while the
between-target signal fell **2.4×** with it (0.257 → 0.109). S5 improved from 19.22 to 10.36 and
still fails by a factor of thirty.

**Reachable but not yet reported with intervals:** S6, S8.

**Structurally unreachable on this evidence, and not claimed:**

- **S1** — 0 of 14 libraries span a timepoint.
- **S3** — exactly one horizon is declared; declaring two would be a claim.
- **S7** — follows from S1 and S3. **Nothing produced by this model is a sufficiency verdict.**
- **S9, S10** — no baseline suite is fitted here, and no external study is used.

## Fold discipline

Leave-one-library-out, 14 folds. A belief about an arm in library *L* is emitted only by the fold
that excluded *L*. The estimator refuses an in-fold library through `capabilities()`, and that
refusal is tested from the side that fails.
