# Replogle 2022 K562 essential-scale representability review

- **Review date:** 2026-08-09
- **Decision:** accepted by the machine-checked reviewed destructive-population representability
  ledger; not
  admitted for a future, causal, longitudinal, training-loss, or benchmark-metric role
- **Reviewed bytes:** scPerturb 1.4 secondary packaging of the authors' day-6 raw-count
  single-cell object

## Review result

The Replogle et al. K562 essential-scale Perturb-seq experiment is a valid test of whether the
dataset contracts preserve a population subject under destructive single-cell sampling. It is not
evidence that the endpoint cells were observed before perturbation, that any cell was followed over
time, or that the 48 Chromium GEM groups are independent biological cultures.

The one positive assessment is therefore a **conditionally eligible descriptive snapshot-state
prior** for the exact post-selection K562 culture population. The contract records explicit
negative assessments for population dynamics, individual longitudinal dynamics, intervention
effects, counterfactual generalization, lineage fate, and every other unsupported scientific role.
No loss or metric assessment is admitted.

## Exact source identity

| Property | Reviewed value |
| --- | --- |
| Biological study | Replogle et al., *Cell* 2022, DOI `10.1016/j.cell.2022.05.013` |
| Author processed-data release | Figshare+ `10.25452/figshare.plus.20029387.v1`, version 1 |
| Reviewed harmonized release | scPerturb Zenodo `10.5281/zenodo.13350497`, version 1.4 |
| File | `ReplogleWeissman2022_K562_essential.h5ad` |
| Public download | `https://zenodo.org/api/records/13350497/files/ReplogleWeissman2022_K562_essential.h5ad/content` |
| Byte count | `1,546,729,675` |
| SHA-256 | `412fd0df8c4ccea9f4db91cd88033c49200838b29d40945e48574be588b48789` |
| Repository MD5 | `d8cba17576d1a8afc0f7d71b79cad0f7` |
| Matrix | 310,385 cells by 8,563 genes; integer UMI counts |
| License | CC BY 4.0 |

The local artifact's byte count and MD5 match the Zenodo 1.4 record, and its SHA-256 is recorded in
the reviewed manifest. The scPerturb processing code reads the author
`K562_essential_raw_singlecell_01.h5ad`, adds harmonized annotations and QC summaries, renames gene
indices, and writes a gzip-compressed H5AD. Consequently, the reviewed source is classified as
**processed**, even though its expression matrix retains raw UMI values.

The authors' primary Figshare raw-count H5AD is 10,661,879,995 bytes with repository MD5
`4f1122ce1c7f13299a68df6459a266d3`. It is not bound as a source artifact because it was not
downloaded and the manifest requires SHA-256 rather than accepting the repository MD5 as a
substitute.

The Figshare version-1 API response is also bound as an exact metadata source: 11,402 bytes with
SHA-256 `4d7abba9bdedf8b8484aab0aacdce9fbbb21b32fb10b6ef62c633385449f8968`.
Its description identifies the essential-scale K562 experiment as sampled six days after
transduction, distinguishes it from the day-8 genome-wide experiment, identifies the four AnnData
representations, and reports CC BY 4.0.

Primary references:

- [author processed-data release](https://plus.figshare.com/articles/dataset/_Mapping_information-rich_genotype-phenotype_landscapes_with_genome-scale_Perturb-seq_Replogle_et_al_2022_processed_Perturb-seq_datasets/20029387)
- [harmonized scPerturb release](https://zenodo.org/records/13350497)
- [primary methods and study report](https://pmc.ncbi.nlm.nih.gov/articles/PMC9380471/)
- [raw-sequencing BioProject PRJNA831566](https://www.ncbi.nlm.nih.gov/bioproject/831566)
- [scPerturb source transformation](https://github.com/sanderlab/scPerturb/blob/master/dataset_processing/scripts/ReplogleWeissman2022.py)

## Machine-checked reviewed proof binding

The reviewed manifest is
`data_manifests/reviewed/replogle-2022-k562-essential.json`; its fingerprint is
`63257e3c13652e6052a71481a66c6b1ee95da360a9cfa67555215c5bf2c82881`. The proof is
`data_manifests/proofs/replogle-2022-k562-destructive-population.json`; its fingerprint is
`8e5fbd11b4705f3fedb92ef58b88bc92165f38b5848546e8cf2980f9304dfa7a`.

The proof binds the exact manifest, slice, positive assessment, five required negative
assessments, and both declared source checksums. All nine reviewed destructive-population criteria
pass structurally:

- declared exact slice and source-byte digests;
- population subject, destructive collection, and endpoint linkage boundary; and
- rejection of individual-cell, clone, causal, and transported casts.

`verify_representability()` returns `accepted=true` with no failed or structurally failed
criterion. It checks the reviewed ledger's content bindings and typed attestations; it does not
resolve source bytes, recompute whole-axis membership, or replay any selector. Its result reports
`selector_execution_replayed=false`, `source_bytes_resolved=false`,
`use_permission_evaluated=false`, and `use_authorized=false`: representability is deliberately
neither an execution receipt nor a workflow authorization.

## Content-addressed population slice

The proof covers the whole reviewed H5AD, not a favorable post hoc subset. Its record axis is the
unique AnnData observation index named `cell_barcode`.

To bind membership without committing 310,385 identifiers, the proof sorts every cell-barcode
string lexicographically, serializes the result as a compact UTF-8 JSON array with no insignificant
whitespace, and hashes those exact bytes. This is the contract encoding
`canonical_json_utf8_string_array_v1`.

| Membership property | Value |
| --- | --- |
| Slice kind | `whole_artifact` |
| Selected record count | `310385` |
| Unique record count | `310385` |
| Selected population-subject count | `1` |
| Encoded membership byte count | `6771445` |
| Selected-record-ID SHA-256 | `27c981a89a625aa9c3245a9eda97db44df17f7c22d4a91c7ecd906880b6d3546` |

The ordered source IDs are already lexicographically sorted, but the proof does not rely on that
incidental ordering.

## Experimental-unit interpretation

| Concept | Reviewed interpretation |
| --- | --- |
| Belief subject | The one pooled, transduced K562 culture population |
| Sampling subject | That population, observed once by destructive endpoint sampling |
| Observation unit | One captured cell |
| Technical sample/batch | One of 48 GEM-group/lane aliquots (`batch`) |
| Intervention label | One dual-sgRNA construct targeting one gene, or a non-targeting construct |
| Biological replicate | Not documented for the day-6 Perturb-seq population |
| Randomization unit | Not established by the released design metadata |
| Safe structural split unit | The culture, which has only one observed value in this artifact |

`study_population_id` is the manifest's constant identity for that one reviewed pooled culture;
it is not presented as a per-cell source column. The source `batch` and `cell_barcode` fields define
the nested technical-sample and observation units, respectively.

The authors targeted 20Q1 DepMap common essential genes and non-targeting controls. A targeting
construct contains two distinct guides against the same gene, so this is not a combinatorial
two-gene intervention. The reviewed object contains 299,694 targeted cells and 10,691 control
cells, 2,057 observed target-gene identifiers, and controls in every GEM group.

Low-rate lentiviral transduction may be stochastic in practice, but the release does not record an
allocation probability or a randomized experimental-unit protocol. FACS selection, viability, and
depletion of essential-gene perturbations also occur between assignment and the measured endpoint.
The manifest consequently records assigned-nonrandom intervention evidence and measured guide
realization, not an identified randomized intervention effect.

## Scope of the conditional positive assessment

The positive claim is limited to the distribution of day-6 transcript-count observations in the
captured, post-selection K562 population, conditioned descriptively on the measured CRISPRi label.
It assumes that:

1. capture, QC, survival, and guide-call selection are modeled for this exact target population;
2. GEM-group, library-size, and assay variation remain nuisance variables rather than biological
   replication; and
3. transcript counts are an assay proxy for the scoped hidden state, not noiseless state truth.

These assumptions prohibit promoting the assessment to a causal `do(CRISPRi)` claim or
transporting it to another cell line, protocol, time, or assay.

## Explicit non-support

The reviewed slice does not support:

- population dynamics, because there is one destructively sampled molecular endpoint;
- individual-cell dynamics, because cells are destroyed and have no cross-time identity;
- intervention effects, because population replication and a reviewed randomization design are
  absent;
- counterfactual generalization or retrospective intervention selection;
- same-cell or sample-level multimodal fusion;
- lineage, spatial-neighborhood, environmental-history, or functional-outcome claims; or
- an assay measurement model from raw-read and calibration evidence.

The separate day-8 genome-wide K562 screen can later provide external cross-screen evidence for
overlapping targets. It differs in library and endpoint time and is not silently treated as a
biological replicate of this day-6 artifact.

## Why no loss or metric is admitted

A multi-horizon loss requires linked future population observations, and an intervention-effect
loss requires an eligible intervention-effect assessment. Neither is present. The current
experimental manifest also has no empirical snapshot-density loss kind. Population-distribution
metrics cannot be admitted through a dynamics or causal claim that this slice does not support.

Actual benchmark membership belongs to a future split manifest. Splitting cells or GEM groups from
this single culture would measure within-culture or technical-batch generalization, not an
independent-population test. The proof therefore establishes representability only; it does not
authorize training, calibration, validation, or testing.

## Data-use boundary

The exact Figshare+ v1 metadata and scPerturb 1.4 H5AD are both released under CC BY 4.0. The
manifest records all six enumerated use cases as legally permitted, with attribution requirements
to credit the primary study, identify both releases, link the license, and indicate changes. That
legal result does not strengthen the conditional scientific assessment and does not create a loss,
metric, split, or admission decision.
