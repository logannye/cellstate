# GSE274113 Perturb-Multiome structural census (queue item `Q4`, evidence half)

- **Census date:** 2026-08-12
- **Status:** structural census complete; **licence, use terms, and provenance review outstanding**,
  so this is not yet the reviewed manifest `Q4` requires
- **Reviewed bytes:** the fourteen author-deposited `filtered_feature_bc_matrix.h5` libraries and
  the annotated metadata table, held locally
- **Decision it informs:** whether `Q5` may freeze this source as the Phase 2 state-bearing estimand

## Why this census exists

[`../../roadmap.md`](../../roadmap.md) names `GSE274113` as the primary Phase 2 candidate and states
plainly that the intended design — day 7 as the pre-cutoff population observation, days 9, 11 and 14
as horizons, the library as split and bootstrap unit — is **a hypothesis `Q4` must confirm**, that
none of the recorded figures were verified against source bytes, and that if no unit spans the
inference cutoff the source fails S1 and must be redesigned or rejected.

Every number below is computed from the bytes. Nothing is taken from the publication.

## What the census found

### The intended design is not realizable: timepoint is perfectly aliased with library

| Timepoint | Libraries | Cells |
| --- | --- | --- |
| day 7 | `rep1`, `rep2`, `rep3`, `rep4` | 43,723 |
| day 9 | `rep5`, `rep6`, `rep7`, `rep8` | 50,116 |
| day 11 | `rep9`, `rep10`, `rep12` | 20,791 |
| day 14 | `rep13`, `rep14`, `rep16` | 22,974 |

137,604 annotated cells across **fourteen** libraries. `rep11` and `rep15` do not exist in the
series. **No library appears at more than one timepoint: libraries spanning >1 timepoint = 0.**

The consequence is exactly the one the roadmap anticipated. The library cannot be the unit that
spans the inference cutoff, because no library is observed both before and after it. Any day-7
versus day-14 difference is a difference between two disjoint sets of libraries, so a timepoint
effect is not separable from a library effect at the library level.

### The intervention half is clean, and exclusion 2 does not apply

| Property | Measured |
| --- | --- |
| Distinct targets | 20 — 19 transcription factors plus `NT` control |
| `(library, target)` combinations populated | **280 of 280 possible** |
| Smallest `(library, target)` cell count | 105, at `(rep13, RUNX1)` |
| Targets confined to a single library | **0** |
| Cells per `(timepoint, target)` | min 383 (day 14) to max 4,291 (day 9) |

Every target is present in every library. Intervention is fully crossed with library and therefore
cleanly separable from it — the roadmap's exclusion 2 ("approximately one library per treatment
arm") does not apply. This is the source's real strength and it is why the source is not simply
rejected.

### Local byte identity

Recorded so that a later reviewed manifest binds the same bytes this census measured. These are the
**local** artifacts; they have not yet been checked against the GEO record, which is item 2 of what
`Q4` still owes below.

| Artifact | Bytes | SHA-256 |
| --- | --- | --- |
| `GSE274113_annotated_metadata.csv.gz` | 3,307,117 | `6330eac958790fdde5a0d5fd96419125fed4a621d3adfb100cf96675478aa261` |
| `GSE274113_rep1_filtered_feature_bc_matrix.h5` | 289,235,875 | `6e11423c900caf37b8eb063de185dacc038d83ebca01bb7c98296e25cdd65ab0` |
| `GSE274113_rep2_filtered_feature_bc_matrix.h5` | 228,241,935 | `6698ae04b5b4102c285393fd79ff84cc7894f032c5de46718da9b828afa65995` |
| `GSE274113_rep3_filtered_feature_bc_matrix.h5` | 184,666,689 | `109b50baf15e782bc4c2178fe54b83f6c637502f82ff10f2b402807aec788af6` |
| `GSE274113_rep4_filtered_feature_bc_matrix.h5` | 316,962,263 | `460b1db3b101ffbadf0d5c6298a31a1ff7125693cda20195af57aafa4f92d1ea` |
| `GSE274113_rep5_filtered_feature_bc_matrix.h5` | 235,083,990 | `38f2553bc255bfc04519e7ac59193b33ca6f94236aa9424692c85bf3f67a4b80` |
| `GSE274113_rep6_filtered_feature_bc_matrix.h5` | 257,300,982 | `691d253e518672085b2fcef7f7709248e0639b0d0fb16724cc32204071f50e8e` |
| `GSE274113_rep7_filtered_feature_bc_matrix.h5` | 243,529,044 | `d0ee24735f5dee1857726e45cc16099a4d243f2f446b8828eccc45ba489201b3` |
| `GSE274113_rep8_filtered_feature_bc_matrix.h5` | 320,391,753 | `3c669d57eb40f0fd1961f65f54859639467dd8d0c490dd20912ca80b79b82c7b` |
| `GSE274113_rep9_filtered_feature_bc_matrix.h5` | 175,044,644 | `1958eaa086537c2f8cd89f4dbc103d4bfa4f152be228ae195544d0cd3806cc60` |
| `GSE274113_rep10_filtered_feature_bc_matrix.h5` | 185,593,406 | `3dfe66517cbfafe9b93360bad984a4b5d3e1278d1ae707ebccebfe14aa9f8df5` |
| `GSE274113_rep12_filtered_feature_bc_matrix.h5` | 172,401,672 | `3b000feb92d8433a01e7369d04f54fed73c66d5ded5139693119c823ca425cb2` |
| `GSE274113_rep13_filtered_feature_bc_matrix.h5` | 167,753,914 | `384c330d2d63837355c5532d15b07d5b1172f3ec90d46a5dcc5fd2a3684d6c03` |
| `GSE274113_rep14_filtered_feature_bc_matrix.h5` | 167,889,976 | `4dce4979cc61718d45f36f82dbf41a71b78815ccad352128a6798f973a2c1e60` |
| `GSE274113_rep16_filtered_feature_bc_matrix.h5` | 198,289,278 | `e41e31ec86a265909687f32103a9e273216ce113731616071aaf2c6b39d7395a` |

## The measurement the roadmap actually needs: are the libraries independent?

Requirement `roadmap.md` places on a state-bearing query: *"a split unit that is a genuine
independent replicate, present in sufficient number to bootstrap."* The publication describes the
libraries as technical replicates of one continuous culture. **That is an annotation, and this
project's rule is that the unit of independence is measured, not annotated.**

Readout: the composition of each `(library, target)` population over the ten annotated cell types.
This is a genuine population-state observation — the differentiation-state distribution of the
population — and it requires no expression matrix. The effect of a target is its composition minus
the composition of `NT` **in the same library**, which differences out that library's batch.

### Effects replicate across libraries — strongly early, weakly late

Pearson `r` between two libraries' effect vectors (19 targets x 10 cell types = 190 values):

| Timepoint | Library pairs | Mean `r` | Spearman-Brown reliability of the k-library mean |
| --- | --- | --- | --- |
| day 7 | 6 | **0.833** | 0.952 |
| day 9 | 6 | **0.854** | 0.959 |
| day 11 | 3 | 0.451 | 0.712 |
| day 14 | 3 | 0.449 | 0.710 |

Days 7 and 9 are measured well. Days 11 and 14 are not: three libraries and roughly half the cells
give a library-averaged effect with reliability near 0.71.

### Library variation is *not* negligible

Ratio of the standard deviation across libraries at a fixed target to the standard deviation across
targets: **0.53** (day 7), **0.51** (day 9), **0.89** (day 11), **0.99** (day 14).

This cuts against reading the fourteen libraries as interchangeable technical splits — a pure
technical split would contribute variation small relative to the biology. It does **not**
establish biological independence either; these bytes cannot separate independent-culture variation
from chip, capture and depth variation. What it does establish is that by day 11 the
library-to-library spread is as large as the entire between-target effect, so a bootstrap over three
such libraries has very little to work with whatever their provenance.

## The finding that decides the horizon set: the effect does not persist

Correlation between the library-averaged effect vectors at two timepoints, raw and disattenuated by
the reliabilities above (`rho = r / sqrt(rel_a * rel_b)`). A target-label permutation null over
1,000 draws puts the 95th percentile of `|r|` at **0.025**.

| Pair | Lag | `r` observed | `rho` disattenuated | Verdict |
| --- | --- | --- | --- | --- |
| day 7 -> day 9 | 2 days | 0.552 | **0.577** | persists |
| day 7 -> day 11 | 4 days | 0.147 | **0.179** | persists, weakly |
| day 7 -> day 14 | 7 days | -0.022 | **-0.026** | **indistinguishable from the null** |
| day 9 -> day 11 | 2 days | 0.448 | 0.542 | persists |
| day 9 -> day 14 | 5 days | -0.175 | -0.213 | exceeds null, sign reversed |
| day 11 -> day 14 | 3 days | 0.011 | 0.015 | indistinguishable from the null |

The disattenuation matters: it is what the correlation would be if both timepoints were measured
without noise. A near-zero `rho` therefore cannot be explained away as attenuation. **On this
readout the day-7 population state carries essentially no information about the day-14 population
state.** The effect decays monotonically — 0.577 at two days, 0.179 at four, zero at seven.

### What this does and does not license

It **does** show that the intended four-timepoint design is empty at its longest horizon. A
sufficiency test with day 7 as cutoff and day 14 as a horizon would be asking a state to predict
something the intervention no longer affects, and both `M1` and `M2` would be predicting noise.

It **does not** reject `GSE274113`. Composition over ten cell types is a coarse summary. A
transcriptional or chromatin state may persist after the composition of the population has
reconverged, and that is a plausible biology rather than a special pleading. What the census
establishes is that the persistence question is now **an empirical one with a measured answer on at
least one readout**, and that the burden has moved: a day-14 horizon must be justified against this
null, not assumed.

## What `Q4` still owes

This is the evidence half only. Before the reviewed manifest can be written:

1. **Licence and use terms**, and whether they permit the intended claim.
2. **Exact byte identity** against the GEO record, not only against the local copies.
3. **The number of independent parent cultures**, which is not in GEO — every sample carries only
   `tissue: Hematopoietic cells` — and must come from the publication's methods, recorded as a
   citation rather than as a measurement.
4. **Whether the ATAC feature space is shared.** RNA is a fixed 36,601 features in every library.
   The peak sets are not obviously shared and this has not been verified here; if the source
   survives, a fixed-bin or peak-union aggregation becomes a named `Q5` deliverable.
5. **A committed runner.** Every number above was computed from the bytes, but by a script held
   outside this repository, so no reader can reproduce them from a checkout. Until a runner is
   committed alongside the reviewed manifest, this census is a recorded claim rather than a checked
   one, which is the same defect this project's rules exist to prevent. The tables stand as
   measurements; they do not yet stand as reproducible measurements.

## The decision this hands to the ADR

`Q5` must not begin until an ADR records the census and decides, on this evidence, among:

- **Reject** the source for the Phase 2 estimand, which makes the roadmap's deferral of Vertical B
  the binding constraint, since that deferral is conditioned on a Vertical A sufficiency verdict
  that rejection makes unreachable;
- **Redesign** to a short-horizon estimand — day 7 cutoff, days 9 and 11 as the two horizons S3
  requires, day 14 dropped — with the target as the unit that spans the cutoff and the library
  retained as a nested batch, accepting n=19 non-control units for the bootstrap;
- **Redesign** onto a readout other than composition, and carry the burden this census sets.

The census does not make that choice. It removes the option of making it by assumption.
