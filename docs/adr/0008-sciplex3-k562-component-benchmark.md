# ADR 0008: freeze the first Vertical A component benchmark

- **Status:** Accepted; the frozen benchmark definition stands as written. Its component was
  reclassified off the state path by [ADR 0013](0013-state-first-roadmap-reordering.md), which
  retains the partitions, split discipline, leakage review, and golden fixtures as infrastructure
  and as the Phase 1 metric proving ground. The remaining admission work described under
  "Admission boundary" below is no longer scheduled
- **Date:** 2026-08-09
- **Decision owners:** cellstate maintainers
- **Scope:** Vertical A query and benchmark semantics

## Context

The first Vertical A draft asked for a pre-intervention molecular belief followed by a calibrated
future population response. The reviewed Replogle K562 artifact cannot validate that task: it is a
single destructive endpoint from one culture-level experiment and has no independent biological
performance split. It remains valuable as a population-sampling and provenance fixture, but a
large cell count does not manufacture replication.

The public sci-Plex3 K562 screen provides a stronger, narrower task. It contains small-molecule
assignments, matched vehicle controls, two screen replicates, independent wells, four exact doses,
and a 24-hour destructive sci-RNA-seq endpoint. It still contains no pre-treatment molecular
measurement, tracked cell, second horizon, functional endpoint, or external validation study.

## Decision

Freeze `vertical-a.sciplex3-k562-24h-replicate-transfer.v1` as a **component benchmark**, not as the
complete Vertical A benchmark and not as evidence that a biological backend is ready.

The exact query is deliberately context-only at its cutoff:

- the belief and prediction subject is the K562 population assigned to one independently treated
  well at `t = 0`; future recovery does not define subject membership;
- `t = 0 s` is immediately before compound or vehicle addition;
- admissible inference evidence is static K562 and source-declared experimental context only;
- no 24-hour RNA, QC, cell-recovery, assay-batch, or matched-control outcome may enter inference;
- one source-identified administered agent is applied continuously at exactly 10, 100, 1,000, or
  10,000 nM, with no combination or washout. Its source label is not treated as a resolved chemical
  ontology identity or biological target, and reversibility remains explicitly unknown;
- the only horizon is 86,400 seconds; and
- the target is the population distribution of integer sci-RNA-seq UMI count vectors on the exact
  ordered 2,000-feature panel derived from TRAIN only, among recovered K562 nuclei. The panel is a
  content-addressed assay-output schema, not universal state. The target is not latent abundance, a
  living-cell population, viability, total population size, a functional outcome, or an
  individual-cell trajectory.

The terminal endpoint assay is target-only. It has no invented price or turnaround time and is not
eligible for `recommend_next_measurement`; query assay budgets are absent because the benchmark has
no measurement-selection candidate.

The corrected scPerturb v1.4 artifact is the executable data source:

```text
SrivatsanTrapnell2020_sciplex3.h5ad
bytes: 2,526,631,614
MD5: c9e70629505d98c7ca1a837f62b14e89
SHA-256: 603ed16c5e25401c8a7f5bb0b2b045179701017d65dcfc6aeea71722a66cd10a
```

The prior unversioned local H5AD did not match any reviewed release. It was quarantined and cannot
be cited. The official v1.4 file is necessary because that release explicitly corrected sci-Plex3
processing errors.

## Split and estimand

Partitions are fixed by whole culture plates. Cells never cross the partition of their composite
`(plate, well)` parent.

| Role | Source subset | Wells | Cells | Treated conditions | Vehicle wells |
| --- | --- | ---: | ---: | ---: | ---: |
| Train | replicate 1, plates 1--8 | 768 | 94,785 | 188 compounds x 4 doses | 16 |
| Calibration | replicate 2, plates 25--26 | 192 | 18,001 | 47 compounds x 4 doses | 4 |
| Model-selection validation | replicate 2, plates 27--28 | 192 | 20,481 | 47 compounds x 4 doses | 4 |
| Untouched test | replicate 2, plates 29--32 | 384 | 40,385 | 94 compounds x 4 doses | 8 |

The replicate-2 compound sets are mutually disjoint across calibration, validation, and test. Every
evaluation compound-dose condition has its exact replicate-1 training counterpart. The benchmark
therefore measures in-support replicate transfer; it does not measure unseen-compound, new-cell-
line, new-protocol, or cross-study generalization.

The reference experimental contrast is the assigned compound-dose well versus vehicle wells from
the same culture plate on the captured-nucleus endpoint. The common source-matched vehicle
background is part of the frozen experimental context; a no-action case means no active compound
beyond that background. A model forecast remains a predictive
association unless a separate content-addressed causal or transport report passes. Assignment does
not prove target engagement, per-protocol response, survival response, or external transport.

## Metrics and baselines

All primary results give equal weight to independent wells or predeclared compound-dose groups;
cell count is never treated as replicate count. Transformations, feature order, grouping, random
seeds, and uncertainty blocks are content addressed and fit only on the allowed partition.
The full source-axis library-size transform is used only to select the feature panel from TRAIN.
Scoring is a separate, content-addressed transform that normalizes each observed or predicted
sample by the sum of its exact ordered 2,000 panel counts, so the declared target contains every
required input. A wrong length or order, a nonfinite, noninteger, or negative count, or a zero panel
total fails evaluation; it is never imputed or excluded.

The frozen metric set comprises:

1. marginal CRPS on the training-derived feature panel and log-count scale;
2. a joint energy score in a training-only low-dimensional projection;
3. vehicle-relative pseudobulk effect error;
4. marginal predictive coverage and sharpness; and
5. a secondary four-dose profile diagnostic.

The screen has one held-out well per exact condition. It therefore cannot validate latent
between-well uncertainty for each condition, and that claim remains unassessed.

Mandatory applicable baselines are matched-vehicle resampling, exact-condition replicate-1
empirical resampling, an exact-condition negative-binomial model, a hierarchical well-level
negative-binomial model, and a low-rank compound-dose response model. Nearest-dose interpolation is
secondary. Persistence and temporal state-space baselines are machine-verifiably inapplicable
because no `t = 0` assay and no second future horizon exist. A missing, crashed, or unevaluated
applicable baseline blocks scientific admission; it is never reclassified as inapplicable.

## Admission boundary

The benchmark definition is frozen while its admission remains component-only. Its exact manifest,
scientific-assessment, and benchmark-permission resolutions pass. Full scientific admission
additionally requires executable metric and uncertainty implementations with golden cases, a
complete passing leakage audit, executed mandatory baselines, complete metric reporting, paired
compound-block comparisons, and passing thresholds frozen before protected untouched-test endpoint
and scoring access. Planned
specifications are typed separately from executable implementations and cannot satisfy those
gates.

Even after those gates pass, this benchmark validates only a context-to-24-hour assay-response
component. It cannot graduate the full belief-state system, prove predictive sufficiency from a
molecular baseline, validate multiple horizons, or authorize intervention planning.

## Consequences

- Replogle and Live-seq retain their separate representability roles; neither is substituted into
  this benchmark.
- Data adapters may target this exact source and split only after the checked-in benchmark artifact
  resolves. This consequence was written against the superseded phase numbering; see
  [ADR 0013](0013-state-first-roadmap-reordering.md).
- The first biological backend must expose the lack of pre-cutoff molecular evidence and must not
  label condition lookup as a learned cellular belief.
- A later complete Vertical A benchmark needs a replicated source with admissible pre-cutoff state,
  future endpoints, and preferably an external study.
