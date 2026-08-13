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
Omega = diag( 1/(y + 1/2) - 1/(n + G/2) )  +  psi^2
```

The eight-dimensional state is `[biology(4) | library nuisance(3) | guide realization(1)]`.

**Where S5 is won structurally.** `V` is the leading subspace of the NT arms across libraries — NT
is the same biology everywhere, so what moves there is the library. `W` is the leading subspace of
*within-library* contrasts — differencing inside a library cancels the library, so what moves there
is the perturbation.

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
6. **The residual sits at the sampling floor**, so `psi^2` is small and the posterior is tight.
   Whether that tightness is *earned* is S2, and this model may not assert it about itself —
   calibration adjudicates, and can fail.
7. **Day-14 selection.** Annotation rate is 98–99% at days 7, 9 and 11 and **62%** at day 14, and
   the excluded day-14 barcodes are real cells rather than empty droplets. Composition also shifts
   with perturbation over time. Beliefs about day-14 libraries rest on a differently-selected
   population, and this is not modelled.

## Ledger capabilities

**Advanced, on held evidence:**

- **S5** — the nuisance axis is a declared subspace, separated from biology by construction.
- **S2** — the belief carries a posterior whose spread responds to depth, and whose measurement and
  biological components are computed rather than declared. Whether the spread is *earned* against
  the point predictor's residual is not asserted here.

**Reachable but not yet reported with intervals:** S4 (both halves), S6, S8.

**Structurally unreachable on this evidence, and not claimed:**

- **S1** — 0 of 14 libraries span a timepoint.
- **S3** — exactly one horizon is declared; declaring two would be a claim.
- **S7** — follows from S1 and S3. **Nothing produced by this model is a sufficiency verdict.**
- **S9, S10** — no baseline suite is fitted here, and no external study is used.

## Fold discipline

Leave-one-library-out, 14 folds. A belief about an arm in library *L* is emitted only by the fold
that excluded *L*. The estimator refuses an in-fold library through `capabilities()`, and that
refusal is tested from the side that fails.
