# Evidence inventory

**Audit date:** 2026-08-09
**Status:** sanitized local-presence audit; no dataset is admitted for training or validation until a
reviewed machine-readable manifest verifies exact source bytes, checksums, use terms, and
experimental structure.

This inventory preserves the evidence behind the project's local-data conclusion without making
machine-specific storage paths part of the scientific contract.

## Strongest locally available components

| Dataset | Locally verified content | Candidate role | Decisive limitation |
| --- | --- | --- | --- |
| `GSE274113` Perturb-Multiome | 137,604 cells; paired RNA+ATAC; 20 CRISPR targets including control; days 7, 9, 11, and 14; replicate libraries | Same-cell RNA/ATAC observation modeling and perturbation-conditioned population dynamics | Destructive cells, no cross-time identity, donor/genotype, lineage, spatial context, or independent function |
| GWCD4i / `GSE314342` | Local pseudobulk and differential-effect assets; four donors; rest, 8-hour stimulation, and 48-hour stimulation; genome-scale CRISPRi | Donor/context-conditioned intervention effects | Cell-level source was not downloaded locally; no washout, lineage, spatial context, or independent function |
| RASA2 repeated stimulation / `GSE204862` | Primary human T-cell repeated antigen exposure, control versus RASA2 KO, molecular samples plus killing, cytokine, metabolic, and persistence figures | Exposure-history-to-function validation at donor/condition level | Bulk molecular data; outcome linkage remains donor/condition-level and must not be assigned to cells |
| HCA Tonsil Atlas | scRNA, scATAC, true multiome subset, CITE-seq, Visium, and 17-donor context | Multimodal likelihoods, donor priors, and spatial context | No controlled perturbation, time course, or later functional target |
| Anti-PD-1 NSCLC / `GSE243013` | Approximately 1.25 million cells from 234 post-treatment surgical patients; scRNA, TCR clonotypes, regimen and response metadata | Patient/context priors and population-level outcome association | Predominantly post-treatment and cross-sectional; TCR clonotype is not parent-child lineage; causal treatment effects are not identified |
| Immune Dictionary / `GSE202186` | 386,703 mouse lymph-node cells; 86 cytokines; PBS controls; three mice per cytokine; four-hour endpoint | Environment/cytokine response and OOD tests | One mouse RNA endpoint; no same-cell future function or lineage |

## Additional locally available evidence

| Dataset family | Candidate use | Limitation that must remain explicit |
| --- | --- | --- |
| TOX multiome `GSE255042`/`GSE255043` | TOX intervention, antigen rechallenge, paired RNA+ATAC | Essentially one library per treatment arm, making treatment and library difficult to separate |
| Belk exhaustion `GSE203591`/`GSE203592`/`GSE203593` | Perturbation, chromatin time course, persistence screens | RNA and ATAC are not same-cell paired; no clean withdrawal/recovery arm |
| Weber CAR-T `GSE164949` | Antigen withdrawal and rest across donors | Bulk RNA, few nonzero rest horizons, no same-cell function |
| Drug-seq `GSE306429` | Acute compound effects across donors/libraries | Local artifact is a CD8 pseudobulk derivative with one dose and one 24-hour endpoint |
| Norman Perturb-seq `GSE133344` | K562 single and double CRISPRa response | One destructive transcriptomic endpoint |
| Local scPerturb collection | Replogle, Norman, and Frangieh harmonized perturbation matrices | Harmonization cannot replace source timing, controls, use terms, or experimental-unit review |
| NSCLC `GSE207422` | Treatment/response population association | Pre-biopsy and post-surgery single-cell samples are not a paired patient time course |
| LuCA NSCLC atlas | Large donor-, tissue-, disease-, and genotype-aware population prior | Observational atlas without controlled interventions or future outcomes |
| LCMV exhaustion trajectory | Spliced/unspliced RNA over four population times | No matched intervention/recovery arm; sample and time are confounded |
| Washout `GSE150369` | Memory/exhausted/recovered endpoint geometry | Three destructive populations, not a longitudinal recovery trajectory |
| Tabula Sapiens, CPTAC/TCGA, DepMap, LINCS and related bulk/atlas assets | Static or population priors, stable context, mechanistic pretraining | Not individual-cell event histories; bulk or observational evidence cannot identify cell-level controlled dynamics |

## Excluded from the public-real biological program

The locally present `ultimate_cell_v1_io_nsclc_d0` resource is not an all-real empirical dataset. Its
own condition tables and build/source reports describe generated conditions and semi-synthetic
anchors. It may be useful for software exercises but must not enter a public-real biological
training, calibration, validation, or benchmark manifest.

## Audit conclusion

No locally examined dataset contains a complete, coherent same-cell record with multimodal state,
controlled intervention and environment history, parent-child lineage, spatial neighborhood, and
later function. No locally examined resource establishes explicit parent-child lineage; TCR
clonotypes provide clonal grouping, not an observed lineage tree.

The local collection is nevertheless a strong complementary portfolio. `GSE274113` is closest to a
multimodal population-dynamics substrate, `GSE204862` is closest to history-to-function evidence,
HCA Tonsil is closest to multimodal/spatial observation, and GWCD4i is strongest for broad
donor/context-conditioned perturbation effects. Their measurements remain separate experimental
evidence and may not be joined as fictitious same-cell rows.
