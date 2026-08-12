# GSE141064 Live-seq functional-recorder representability review

**Review date:** 2026-08-09  
**Decision:** conditionally representable as an associational, same-cell functional-outcome
estimand; not admitted for training, model selection, calibration, causal inference, or benchmark
testing.

## Exact estimand

For one image-tracked RAW264.7-G9 cell, estimate the distribution of its future LPS-induced
Tnf-promoter mCherry response slope from 3 to 7.5 hours after LPS, conditional on the partial
cytoplasmic Live-seq transcriptome collected from that same viable cell 0.5 to 1.5 hours before
LPS.

The inference cutoff is the bounded Live-seq observation window `[-5400, -1800]` seconds relative
to LPS addition. The functional target is the derived `mCherry.log.slope` over
`[10800, 27000]` seconds relative to LPS. The system boundary is the individual cell plus its
declared soluble culture environment. The evidence supports predictive association only; it does
not identify the effect of LPS, a same-cell counterfactual, or transport to another cell system.

This review binds the [GEO series GSE141064](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE141064),
the [primary publication](https://www.nature.com/articles/s41586-022-05046-9), and the authors'
[archived v1.0.0 analysis](https://doi.org/10.5281/zenodo.6611232).

## Why the reviewed cohort has 17 cells

Several superficially similar counts refer to different experiments or different stages of the
functional-recorder cohort:

| Count | Meaning | Why it is not the reviewed cohort |
| --- | --- | --- |
| 24 RAW cells | Cells first sampled in the separate repeated-transcriptome experiment | Only four yielded two QC-passing transcriptomes; this is not the RNA-to-reporter recorder analysis. |
| 12 paired cells | Four RAW plus eight ASPC cells with two QC-passing transcriptomes | These combine different cell systems, interventions, clocks, and identity methods. Public tabular metadata expose 13 apparently complete pair IDs, so the publication's exact 12 cannot be reconstructed without guessing. |
| 40 cells | Cells reported to have both Live-seq and tracked Tnf-mCherry | The public analysis reports that only 17 passed RNA quality control; the 40 source memberships and raw image traces are not published as an exact downloadable table. |
| 24 metadata rows | Rows in `meta.final.csv` with non-null `mCherry.log.intercept` and `mCherry.log.slope` | Five rows are already `LPS_treated`; only 19 are marked `not_treated`, and the authors' recorded analysis applies an additional batch restriction. |
| **17 cells** | Exact public analysis slice selected by `mCherry.log.intercept > 0`, `treatment == "not_treated"`, and `Batch == "8_8"` | This is the reproducible same-cell transcriptome-to-future-reporter cohort reviewed here. |

The exact record IDs, in canonical lexicographic order, are:

```text
sample372
sample378
sample386
sample404
sample405
sample416
sample421
sample441
sample452
sample454
sample459
sample466
sample468
sample486
sample498
sample499
sample505
```

Their canonical compact JSON UTF-8 string-array digest is
`2e26c9f32124bc5b92cc1dbc281189f44968d95a86d6e202e2475c000bddf8ff`. The digest excludes a
trailing newline. The tagged author script that contains the selection predicate has SHA-256
`1196d1fc8478623bc1405701062b335df12c205777b3bafa7a2a8360dbb0c1a3`.

The checked-in representability artifact is a machine-checked reviewed attestation ledger. Its
structural verifier binds these declared digests and the reviewed selector locator, but does not
resolve either source, replay the selector, or independently recompute the 17-cell membership.

## Public artifacts and checksums

| Artifact | Release and role | Bytes | SHA-256 |
| --- | --- | ---: | --- |
| [`GSE141064_count.final.csv.gz`](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE141nnn/GSE141064/suppl/GSE141064_count.final.csv.gz) | GSE141064, last update 2022-11-29; processed gene-by-sample counts | 10,935,830 | `0e8a07d36a4b06bc8ced881bf97114adc2b55681b5ccf00d29e4f80e05107b22` |
| [`GSE141064_Meta_data.xls.gz`](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE141nnn/GSE141064/suppl/GSE141064_Meta_data.xls.gz) | GSE141064, last update 2022-11-29; GEO sample and raw-file metadata | 145,125 | `28385f24ea851f8c193c6a8bcba14c770eacaf4591ef5344fba5794ee481ad44` |
| [`DeplanckeLab/Live-seq-1.0.0.zip`](https://zenodo.org/api/records/6611232/files/DeplanckeLab/Live-seq-1.0.0.zip/content) | Zenodo record 6611232, v1.0.0, published 2022-06-03; archived metadata and analysis | 3,839,925 | `c7c4c18f3a157b9e74443f3b091d1f9cee538e00589f672bf9ae7b90fdeab8b7` |
| [`data/meta.final.csv`](https://raw.githubusercontent.com/DeplanckeLab/Live-seq/fd47d7cc294e8da86b4f04e2f3f3545c4b76367b/data/meta.final.csv) | Git commit `fd47d7cc294e8da86b4f04e2f3f3545c4b76367b`; exact record IDs, QC, batch, treatment, and derived reporter values | 361,855 | `dcce9bea9444857a6a042fc64be3a5c7b31a00095a21eabc1488e516abd689c8` |
| [`5_Liveseq_with_LiveCell_imaging.R`](https://raw.githubusercontent.com/DeplanckeLab/Live-seq/fd47d7cc294e8da86b4f04e2f3f3545c4b76367b/05_Liveseq_with_LiveCell_imaging/5_Liveseq_with_LiveCell_imaging.R) | Same commit; exact selector and downstream model code | 30,677 | `1196d1fc8478623bc1405701062b335df12c205777b3bafa7a2a8360dbb0c1a3` |
| [Primary article PDF](https://www.nature.com/articles/s41586-022-05046-9.pdf) | Nature 608, published 2022-08-17; protocol, cohort flow, timing, viability, and derivation documentation | 8,223,214 | `cfaa5c013874dfc435158b77f557bbc07fcf9673d265c8e33b31fb6750e0e94a` |
| [`LICENSE`](https://raw.githubusercontent.com/DeplanckeLab/Live-seq/fd47d7cc294e8da86b4f04e2f3f3545c4b76367b/LICENSE) | GPL-3.0 terms for the analysis repository | 35,149 | `3972dc9744f6499f0f9b2dbf76696f2ae7ad8af9b23dde66d6af86c9dfb36986` |

The Zenodo record also publishes MD5 `c3febc67b0910510b368ef82e79fe4a1` for the v1.0.0 ZIP. Raw
sequencing is available under SRA study `SRP233365` and BioProject `PRJNA591913`; the GEO metadata
include raw filenames and per-file MD5 values. Raw sequencing is not required to establish this
representability proof and no large biological artifact is committed to Git.

## Experimental structure and same-cell evidence

Live-seq uses FluidFM to remove a partial cytoplasmic biopsy followed by enhanced Smart-seq2. The
primary paper reports general post-extraction viability of 85--89%, a mean extracted volume of about
1.1 pl, recovery of RAW-cell pre-extraction volume within roughly 100--320 minutes, and continued
cell-cycle progression. Those results support a viability-preserving collection classification but
do not prove an absence of collection effects.

For the functional recorder, the cells remained in one microscope field and were directly tracked
from the Live-seq biopsy through LPS exposure and later reporter imaging. This is direct same-cell
identity evidence, not an expression-similarity match. The alignment key is `sample_ID`; the
record-level subject is an individual cell. `orig.ident` partitions the selected records across
plate1 (7), plate3 (4), and plate4 (6). It is a technical grouping and must not be relabeled as an
independent biological replicate.

The paper reports LPS addition 0.5--1.5 hours after extraction and a final concentration of
100 ng/ml in Methods. The GEO overview displays a conflicting concentration, so the manifest uses
the publication Methods value only as documented protocol context and records the discrepancy. All
selected cells receive LPS; there is no randomized treatment contrast inside this slice.

The reporter target is derived rather than a direct downloadable time series. For each cell, the
authors fit a linear model between time after LPS and the natural logarithm of Tnf-mCherry
fluorescence over the 3--7.5-hour response window. The downloadable value field is
`mCherry.log.slope`, in log normalized fluorescence arbitrary units per hour. The raw per-frame
fluorescence traces are not present in GEO or the tagged repository.

## Attrition and missingness

The publication's functional-recorder flow is 40 cells with both assays to 17 cells passing the
Live-seq quality filters. The filters reported for Live-seq include more than 1,000 genes, less than
30% mitochondrial reads, and more than 30% uniquely mapped reads. Image analysis also excluded
cells that moved out of view or focus, died, or overlapped. The public sources do not provide an
exact reason-coded row for every excluded cell.

Selection into the 17-cell cohort is therefore outcome-observation and RNA-QC dependent. The proof
does not assume missing-at-random attrition, and the dataset is too small for a credible random
cell-level performance split.

## Claim and objective ledger

The one positive scientific assessment is:

- `FUNCTIONAL_OUTCOME`: **conditionally eligible**, associational, for the exact same-cell
  Live-seq-to-`mCherry.log.slope` scope and temporal windows above.

Conditions include the partial transcriptome being an informative proxy for pre-LPS state, bounded
observation staleness, a correctly derived reporter target, direct image-tracking linkage,
selection/attrition, and narrow transport to this RAW264.7-G9 culture protocol.

The manifest makes the following negative boundaries explicit:

- `INDIVIDUAL_LONGITUDINAL_DYNAMICS`: not assessed; the selected slice contains one RNA biopsy and
  a later functional reporter summary, not repeated state measurements.
- `INTERVENTION_EFFECT`: ineligible; no randomized or otherwise identified LPS contrast exists in
  the selected cohort.
- `COUNTERFACTUAL_GENERALIZATION`: ineligible; one cell does not reveal both potential outcomes,
  and no transport design is present.
- `RETROSPECTIVE_INTERVENTION_SELECTION`: ineligible; there is only one assigned intervention and
  no candidate-set outcome graph.
- `POPULATION_DYNAMICS`, `LINEAGE_FATE`, `SAME_CELL_MULTIMODAL_FUSION`, and spatial claims: not
  assessed or ineligible as appropriate; this slice does not supply the required evidence.

Every loss and metric role remains `NOT_ASSESSED`. In particular, the conditional scientific claim
does not authorize a functional-outcome training loss, cell-level validation split, calibration
claim, or held-out test result. A frozen benchmark contract must establish benchmark membership, a defensible split,
target formula, and metric semantics first; this 17-cell cohort alone is not a credible performance
benchmark.

## Use-policy boundary

The primary article is CC BY 4.0. The tagged analysis repository contains GPL-3.0; the Zenodo record
labels its archive `other-open`, which does not erase the included GPL terms. GEO exposes the
dataset publicly but does not display a dataset-specific reuse license. Public downloadability is
not permission for every downstream use.

Accordingly, GEO count and metadata sources remain `UNKNOWN` for research training, commercial
training, benchmark evaluation, redistribution, derived-model distribution, and publication until
legal review resolves the applicable terms. Repository code and article documentation have
separate, source-scoped policy layers. The effective workflow permission for any assessment using
the GEO evidence is therefore `UNKNOWN`, even when its scientific status is conditionally eligible.

## Representability conclusion

The reviewed 17-cell slice proves that the manifest can encode a real public individual-cell
state-to-future functional estimand without inventing cell identity or collapsing a response window
to a point. It does not admit data for model fitting or testing, validate a biological backend,
freeze a query, support a causal effect, or rescue the unreconstructible 12-pair repeated-RNA
cohort. This conclusion is a reviewed scientific attestation whose ledger passes structural checks,
not a claim that the current verifier fetched biological bytes or executed the selector.
