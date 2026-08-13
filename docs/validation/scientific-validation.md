# Scientific validation contract

Software correctness is necessary and insufficient. This document states the standard a biological
backend must meet, aligned to the exact `StateQuery` it claims. [`../roadmap.md`](../roadmap.md) is
the sole authority for the order in which that work happens and for what has been met so far. The
answer today: no biological backend is registered, no benchmark is scientifically admitted, and the
repository has produced no scientific numbers.

## The rules that decide whether a number counts

**1. Splits follow independent experimental units.** Well, plate, library, biological replicate,
donor, animal, clone, cell line, or study — whichever unit carried the assignment or the outcome.
Cells from a shared unit are pseudoreplicates. **Random cell-level splits are prohibited for
scientific claims**, and split memberships are frozen before model selection.

**2. Every reported quantity carries an interval, grouped at that unit.** A proper score, an
intervention-effect error, a sufficiency gain, or a coverage error reported as a bare point is not a
verdict. Resampling is at the split unit, never at the cell.

**3. Baselines are mandatory and must be beaten individually.** Persistence or no-change,
matched-control resampling, condition or perturbation mean, nearest known condition, pseudobulk
generalized linear models, and simple hierarchical random-effects and low-rank or linear
state-space models, each in the version appropriate to the query. The mandatory set is ledger entry
`S9` in [`../roadmap.md`](../roadmap.md); that entry, not this sentence, is authoritative. A backend
graduates only by beating every applicable baseline **each individually**, with intervals that
exclude zero — not on average, and not against one weak comparator. A baseline that is inapplicable
to a query must be declared inapplicable on the face of the scoreboard rather than quietly omitted;
applicability is a checkable property (`BaselineApplicabilityRule.applies_to`).

**4. Marginal error and all-gene correlation never stand alone.** Both are maximized by predicting no
change. Every metric suite frozen from Phase 1 onward carries at least one
differential-expression-weighted metric and at least one rank-based metric alongside them. The
sci-Plex3 suite frozen by [ADR 0008](../adr/0008-sciplex3-k562-component-benchmark.md) predates this
rule and is deliberately not retrofitted.

**5. Coverage is reported as an upper confidence bound.** "Nominal 90% intervals came out near 90%"
is not a result. The absolute coverage error is reported at every declared level with an upper
confidence bound, grouped at the split unit, against the query's declared threshold. This is
enforced rather than asked for: `CalibrationReport` gates its outcome on the bound and refuses to
construct an evaluated report that has none, so a point estimate inside the threshold with a bound
outside it fails
([ADR 0015](../adr/0015-faithfulness-reports-carry-their-sampling-distribution.md)).

**6. A negative result is a result.** A failing sufficiency verdict or a failing coverage report,
published with its interval, satisfies its gate. Only a suppressed measurement fails it.

## Required evidence for a biological backend

- multi-horizon future molecular and functional prediction, each horizon scored separately;
- interventions, environments, donors or genotypes, and combinations outside the training support;
- randomized perturbations, matched controls, distributional comparisons, or lineage and sister
  designs wherever an intervention effect is claimed;
- division timing, offspring distributions, inheritance, and sibling divergence when lineage is in
  scope;
- missing-modality robustness, with unavailable evidence typed rather than imputed as zero;
- interval calibration reported as absolute coverage error with an upper confidence bound;
- effective out-of-support behavior: on a deliberately out-of-support partition the abstention rate
  exceeds the in-support rate, and the risk-coverage curve is monotone — discarding the
  lowest-confidence decile does not increase held-out risk;
- mechanistic residuals for every constraint the backend claims to represent, ablatable without
  changing contract semantics;
- the `M1` versus `M2` [predictive-sufficiency](../concepts/predictive-sufficiency.md) comparison,
  with its bootstrap interval;
- external replication on an untouched study, with no test-time refitting.

## What is not validation

Random cell-level train/test splits. Cluster coherence, cluster purity, or annotation concordance.
Reconstruction of the current assay. Low-dimensional projection appearance. Interpolation among
perturbations the model has already seen. Passing contract tests, `make check`, or reproducing a
golden fixture — these establish that the software does what it says, which is a precondition for a
scientific claim and never one itself.

## Support is earned, per query and per version

A backend may claim only the species, systems, subjects, interventions, doses, environments,
horizons, and outputs its validation evidence covers, and must abstain elsewhere. Eligibility is
claim-specific: it is a property of the tuple *(exact data slice, exact claim, exact loss, exact
split unit, exact use policy)*, not of a dataset name. Biological training, calibration, and
validation evidence comes only from publicly downloadable real experiments carrying accession,
version, license, checksum, and provenance. Synthetic data tests software and is never biological
evidence.

Graduation proceeds in this order, per query and per version:

1. contract and provenance correctness;
2. deterministic ingestion and leakage-safe splits;
3. calibrated assay likelihoods;
4. future and intervention prediction beyond every applicable mandatory baseline;
5. calibrated uncertainty and effective out-of-support abstention;
6. state versus state-plus-history sufficiency;
7. replication on an untouched external study;
8. and only then, pseudo-prospective intervention or assay planning.

[`../roadmap.md`](../roadmap.md) governs scheduling and records which gate is current. Prospective
laboratory validation remains required before any operational biological claim.
