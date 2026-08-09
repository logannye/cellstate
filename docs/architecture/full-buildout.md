# Full-build architecture: public-real-data cell-state system

## Status and purpose

This document is the durable blueprint for expanding `cellstate` from a backend-neutral contract
scaffold into a scientifically validated family of biological models. It records the agreed system
boundary, data policy, modeling strategy, validation doctrine, and initial vertical slices.

The project goal is not to learn a universal embedding called “cell state.” It is to estimate a
**query-conditioned belief** that meets predeclared predictive-sufficiency and complexity criteria
for named future outcomes under named interventions, environments, and horizons.

The existing linear-Gaussian backend remains a contract and software reference. Nothing in that
backend is biological validation.

## Governing scientific conclusion

No single known local or publicly downloadable real-cell dataset jointly supplies all of:

- multimodal measurements of the same cell,
- causal perturbation assignment and realized engagement,
- dense temporal observations of that same cell,
- lineage and division history,
- local spatial/environmental context,
- independently measured future function,
- diverse donors, genotypes, interventions, doses, and controls.

The system must therefore use a **composite evidence portfolio**. Each dataset is eligible only for
the likelihoods, transitions, targets, causal effects, and metrics justified by its experimental
design. Overlap between cell systems, conditions, or perturbations can support hierarchical
transport; it does not create cell-level pairing between studies.

This means a public-real-data-only program can support rigorous retrospective and
pseudo-prospective models for explicit query families. It cannot yet validate an unrestricted
universal virtual cell or arbitrary counterfactual planning.

## Formal estimand

For query `Q` at experimental time `t`, estimate:

```text
B_t^Q =
P(X_t^Q, \Theta, R_{\leq t}, \Xi \mid H_{\leq t}, C, Q)
```

where:

- `X_t^Q` is the query-relevant dynamic state;
- `Theta` contains stable or slowly varying subject, donor, genotype, and lineage parameters;
- `R_{<=t}` is realized perturbation or target engagement, distinct from intended assignment;
- `Xi` contains assay, batch, capture, segmentation, and other measurement nuisance variables;
- `H_{<=t}` is the causally ordered evidence history;
- `C` is declared population, lineage, spatial, and environmental context.

Future prediction is a separate operation:

```text
P(Z_{t+h} \mid B_t^Q, \operatorname{do}(U_{t:t+h}), E_{t:t+h}, Q)
```

where `Z` contains query targets rather than backend-private latent coordinates. The use of
`do(U)` in notation is not itself a causal guarantee. Every output must label its
evidential status:

- predictive association;
- identified population intervention effect;
- transported effect under enumerated assumptions;
- mechanistic extrapolation;
- unsupported.

## Belief subject and evidence semantics

The first schema evolution must prevent a common scientific category error. Destructive assays
sample different cells at different times; those observations do not constitute a longitudinal
history of one cell.

The domain should distinguish at least:

| Belief subject | Identity evidence | Permitted interpretation |
| --- | --- | --- |
| Individual cell | Direct tracking or viability-preserving repeated sampling | Conditional state and future of that cell |
| Clone/lineage | Heritable barcode, phylogeny, or parent–child tracking | Shared ancestry, inheritance, and fate probabilities |
| Population | Experimental sample, condition, well, or distribution | Population transition and response distribution |
| Spatial niche | Region plus cells/neighborhood graph | Contextual state and neighborhood-dependent response |

Evidence also needs a role: direct, ancestor, descendant, sibling, clone aggregate, population,
neighborhood, or external prior. Backends must reject evidence roles they do not model.

No ingestion or training pipeline may infer subject identity from similar expression, nearest
neighbors, optimal transport, shared cluster membership, or adjacent sampling times. Those methods
may estimate population couplings, but the coupling must remain labeled as inferred.

## System architecture

```text
StateQuery
    │
    ▼
Query compiler ──────────────► support/eligibility gate
    │                                   │
    ▼                                   ▼
Canonical causal history       explicit abstention/reason
    │
    ├── assay likelihoods and nuisance variables
    ├── donor/genotype/lineage/population priors
    └── environment and neighborhood context
    │
    ▼
Posterior belief B(t) ─────────► uncertainty and identifiability report
    │
    ▼
Controlled stochastic dynamics + event hazards/jumps
    │
    ▼
Typed target × horizon predictive distributions
    ├── calibration and predictive-sufficiency evaluation
    ├── OOD/support decision
    └── measurement/intervention planning, only after validation
```

### Intended package growth

The existing stable packages remain the contract kernel. Biological and data infrastructure should
enter through adapters and replaceable backends:

```text
src/cellstate/
  domain/                 # Existing public boundary contracts
  engine/                 # Existing deterministic mechanics
  ports/                  # Existing backend interfaces
  reference/              # Existing non-biological reference backend
  schema/migrations/      # Explicit semantic schema migrations
  ontology/               # Gene/protein/chemical/cell/assay/unit mappings
  data/
    catalog/              # Dataset and capability manifests
    acquisition/          # Download, checksums, licensing, immutable provenance
    normalization/        # Source-specific canonicalization
    event_builder/        # Experimental timelines and subject graphs
    eligibility/          # Claim/loss/metric eligibility
    leakage/              # Split and target leakage audits
  adapters/               # GEO/SRA, AnnData, Zarr, Parquet, FCS, OME-Zarr
  artifacts/              # Content-addressed array/image/model references
  models/
    observation/          # Modality-specific likelihoods
    priors/                # Stable, population, donor, and lineage priors
    state/                 # Structured shared/private latent factors
    dynamics/              # Controlled flows, diffusion, hazards, and jumps
    decoders/              # Future molecular and functional target distributions
    mechanisms/            # Soft biological constraints
    uncertainty/           # Uncertainty decomposition and transport risk
  inference/               # Encoders, filters, smoothers, particles, fusion
  backends/
    population_response/
    longitudinal_cell/
    lineage_fate/
    multimodal/
    spatial/
  training/                # Cutoff samplers, curriculum, losses, reproducibility
  evaluation/              # Frozen splits, metrics, calibration, OOD, sufficiency
  planning/                # Value of information and intervention selection
  registry/                # Versioned model/data cards and support envelopes
  workflows/               # Manifest-driven acquisition, training, and evaluation
configs/
data_manifests/
benchmark_manifests/
```

Framework-specific tensors and trainers must remain behind ports. PyTorch/JAX, AnnData, FCS, Zarr,
and image representations must not become public-domain schema requirements.

## Query compiler and support gate

A query compiler translates a public `StateQuery` into a backend task specification:

- belief subject and evidence roles;
- active state factors;
- target decoders and units;
- horizons and temporal resolution;
- intervention/environment semantics;
- required observation likelihoods;
- minimum identifiability and precision;
- dataset/model support requirements;
- OOD and abstention policy.

This is the control plane for avoiding a monolithic latent. A query about six-hour phosphosignaling
does not require the same dynamic blocks as a seven-day lineage-fate query.

The support gate runs before inference and again before forecasting/planning. It compares the task
to a model artifact's declared:

- species and biological system;
- subject level;
- cell types, donors, genotypes, and culture states;
- interventions, mechanisms, doses, and combinations;
- environments and context;
- observed modalities and target decoders;
- time range and horizons;
- causal evidence class;
- calibration and OOD domain.

A syntactically valid request outside that envelope returns a structured unsupported or
unidentifiable result—not a plausible-looking number.

## Public-real-data foundation

### Storage layers

Use three immutable logical layers:

1. **Bronze:** exact downloaded source bytes, accession/release, retrieval URL/date, license, and
   checksum. Never silently update an accession in place.
2. **Silver:** normalized event and observation tables that preserve raw counts, experimental
   design, source-row provenance, units, missingness, censoring, and nuisance metadata.
3. **Gold:** query-specific examples and features derived by a versioned transformation and assigned
   to a frozen benchmark split.

Recommended physical formats are Parquet/Arrow for event and metadata tables, AnnData or Zarr for
sparse matrices, OME-Zarr for images, and content-addressed artifacts for posterior samples and
weights. Raw/minimally processed values remain recoverable; global batch-corrected embeddings are
never the only retained representation.

### Dataset manifest

The evidence manifest graph has separate responsibilities:

- a **dataset manifest** records scientific identity, experimental structure, use terms,
  capabilities, scoped claim assessments, and acquired source-artifact records with retrieval time,
  resolved bytes, and checksums;
- a **normalization manifest** records transformations and source-row provenance; and
- a **split manifest** records immutable train, calibration, and test membership.

Together those records should cover:

- canonical name, accession, DOI, release/version, URLs, license, and checksums;
- species, tissue/system, cell line/type, donor/animal, genotype, disease, and culture;
- experimental unit, replicate hierarchy, plate/well/sample/batch;
- subject level and whether cell, clone, or spatial identities persist across times;
- modalities and whether they are measured in the same cells;
- intended intervention, mechanism, target, dose, route, start, duration, washout, and controls;
- realized perturbation or target-engagement evidence;
- environment, co-culture, stimulation, and neighborhood/spatial information;
- observation times, future outcomes, lineage/fate, and censoring;
- known processing, exclusions, selection bias, and missingness;
- allowed training, validation, causal, transport, and commercial-use claims;
- immutable dataset-manifest identity that downstream normalization and split manifests reference
  by fingerprint. Dataset manifests never gain mutable downstream back-references.

### Claim-capability ledger

Eligibility is not one Boolean. A dataset can be eligible for some roles and forbidden for others:

| Capability | Minimum evidence |
| --- | --- |
| Assay likelihood | Raw/minimally processed measurements, controls or replicates, nuisance metadata |
| Population transition | Comparable populations at declared times/conditions with experimental units retained |
| Individual transition | Direct repeated measurement/tracking of the same viable cell |
| Lineage transition | Heritable barcode or observed parent–child relationship |
| Intervention effect | Assigned exposure, matched controls, timing, replicate structure, and relevant confounding assumptions |
| Same-cell multimodality | Modalities demonstrably measured in the same cell, not merely the same sample |
| Spatial-context effect | Spatial coordinates/neighborhood plus adequate samples or interventions for the claim |
| Future functional target | Outcome measured after the inference cutoff at the appropriate subject/population level |

Loss functions and metrics consume only examples whose capability mask makes them eligible.

## Candidate evidence portfolio

**Review status:** landscape review completed 2026-08-09. Every entry remains a candidate until a
reviewed machine-readable manifest is admitted. The table records a plausible role, not an approved
eligibility decision.

No row in this portfolio is the whole system. The goal is complementary constraint.

| Role | Candidate public datasets | Candidate contribution | Main limitation |
| --- | --- | --- | --- |
| Broad chemical response | Tahoe-100M; sci-Plex | Drug/cell-line/dose response distributions | Destructive endpoint RNA |
| Genetic response | Norman; Replogle Perturb-seq | CRISPR single/pair effects and interactions | Primarily endpoint RNA in transformed lines |
| Drug response and viability | MIX-Seq; PRISM | Molecular population response linked to condition-level viability | Not same-cell molecular-to-function pairing |
| Fast signaling | DREAM signaling; Bodenmiller mass cytometry | Stimulus/drug/dose phosphosignaling dynamics | Narrow panels and systems |
| Perturbed multimodality | Multiome Perturb-seq; Perturb-CITE-seq | Same-cell RNA+ATAC or RNA+protein under perturbation | Usually one destructive endpoint |
| Individual state to future | Live-seq (`GSE141064`) | Viability-preserving state measurement followed by future phenotype | Small and biologically narrow |
| Lineage/fate | LARRY; CellTagging; `GSE284197` | Clone inheritance, differentiation, and perturbation/fate | Clone/sister linkage, not repeated whole-cell measurement |
| Spatial perturbation | Perturb-FISH; Perturb-map; Perturb-DBiT | Neighborhood-dependent response and spatial intervention | Targeted panels or terminal spatial assays |
| Physical dynamics | JUMP Cell Painting; MitoCheck | Morphology, batch robustness, division/death/event dynamics | Molecular state sparse or in separate studies |
| Metabolism/proteomics | SpaceM; SCoPE2; CCLE metabolomics; decryptM | Metabolic/protein observation and target-engagement anchors | Often baseline or population level |
| Population priors | CELLxGENE Census; DepMap | Cell-type/context priors and stable cell-line features | Observational/bulk; no dynamic cell belief |

The initial high-value acquisition order is maintained in the
[`roadmap`](../roadmap.md#phase-1-reproducible-public-real-data-foundation). Full Tahoe and JUMP
image downloads are intentionally deferred; stream or download bounded subsets aligned to the first
query family.

### Local-data conclusion

Previously downloaded real datasets contain valuable multimodal snapshots, perturbation screens,
clinical observational atlases, and bulk functional studies. They do not, individually or jointly,
provide a single coherent complete-cell trajectory with every required causal and contextual field.
The sanitized, accession-level evidence for that conclusion is preserved in the
[local evidence inventory](../data/evidence-inventory.md).

They may enter only for roles justified in their manifests—for example:

- same-cell RNA/ATAC observation modeling and population timepoints;
- donor-aware or clinical population priors;
- CRISPR endpoint response;
- condition-level functional outcomes;
- spatial or atlas priors.

Pseudobulk data cannot support cell-level likelihood validation, TCR clonotype does not generally
establish parent–child lineage, and post-treatment observational cohorts do not identify arbitrary
intervention effects. Semi-synthetic local resources are excluded from public-real biological
training and evaluation even if they remain useful for software tests.

## Biological model family

### Structured state

The query compiler activates a subset of structured factors:

- stable identity and genotype;
- slow epigenetic/developmental memory;
- regulatory execution;
- fast signaling;
- metabolism and energetic capacity;
- physical organization and cell-cycle state;
- stress, damage, senescence, and death propensity;
- functional capacities such as proliferation, secretion, killing, differentiation, or resistance.

Each factor can contain shared and modality-private variables. Context—environment, extracellular
transport, neighborhood, and contacts—remains distinct from intracellular state even when the two
are jointly inferred.

### Observation model

Each modality has its own likelihood and nuisance structure:

```text
p_{\eta_m}(Y_t^{(m)} \mid X_t, \Theta, \Xi_m, C_t)
```

Examples include overdispersed count likelihoods for RNA, sparse binary/count likelihoods for ATAC,
panel-aware continuous/censored likelihoods for protein and phosphosignaling, image-feature or
image-generative likelihoods for morphology, and task-appropriate likelihoods for viability,
secretion, killing, or fate.

The likelihood must model explicit missingness and limits of detection where available. Global batch
removal is not an acceptable replacement for nuisance-variable inference.

### Priors and hierarchy

Priors should be hierarchical over species, cell system, donor, genotype, clone, culture,
environment, and study. Population atlases and bulk resources may inform priors or stable parameters
without pretending to be single-cell trajectories.

Study-specific observation heads and random effects protect biological state from being defined by
assay identity. Transport across studies is conditional and uncertainty-increasing.

### Dynamics and events

Continuous evolution can be represented by a controlled stochastic differential equation:

```text
dX_t = f_\psi(X_t, R_t, E_t, C_t, \Theta, Q)dt
      + g_\psi(X_t, Q)dW_t
```

Discrete biological events require explicit hazards and jump kernels:

```text
\lambda_k(t \mid X_t, R_t, E_t, C_t), \qquad
X_{t^+} \sim J_k(X_{t^-}, \Theta)
```

Events include division, death, differentiation, commitment, migration/contact change, and
environmental switches. Division creates branching beliefs with inheritance parameters. Particle or
mixture representations are preferred near bifurcations, rare fates, and multimodal response.

Population snapshot data constrain evolution of distributions. Directly tracked cells and lineage
barcodes constrain individual or branching dynamics. The model must keep those supervision modes
separate.

### Mechanistic knowledge

Regulatory networks, signaling topology, ligand–receptor relations, metabolic stoichiometry, and
mass-balance constraints may enter as:

- hierarchical priors;
- energy functions;
- residual penalties;
- sparsity/topology regularizers;
- constrained decoder components.

Every constraint is versioned, weighted, auditable, and ablatable. External predictive improvement
and calibration—not biological familiarity—justify retaining it.

### Inference

- Use variational smoothing during training only when all included observations precede the
  permitted training cutoff.
- Use amortized filtering or particle updates for deployable real-time beliefs.
- Carry posterior samples/particles and structured uncertainty through evolution and decoding.
- Decompose aleatoric/process, measurement, parameter, model, and transport/OOD uncertainty.
- Preserve observability and identifiability labels when posterior contraction is not justified.

## Training curriculum

The system should not be trained end-to-end from arbitrary pooled datasets on day one. Use a staged
curriculum with dataset capability masks:

1. **Observation calibration:** learn modality likelihoods and nuisance effects from raw values,
   controls, and replicates.
2. **Cross-modal state:** learn shared/private factors using genuinely same-cell multimodal data and
   modality masking.
3. **Endpoint interventional response:** learn matched treatment/control population distributions,
   doses, combinations, and target engagement.
4. **Temporal state-space dynamics:** learn population time evolution, fast signaling, and
   transcriptional dynamics with causal cutoffs.
5. **Lineage and event hazards:** add clone inheritance, fate branching, division, and death.
6. **Environment and spatial context:** add neighborhood, transport, co-culture, and spatial
   intervention effects.
7. **Query-specific refinement:** fine-tune only the state factors and target decoders required for
   a declared query family.
8. **Calibration and support fitting:** freeze predictive calibration and OOD thresholds on separate
   validation data.

Potential losses, applied only to eligible examples, include:

- future target negative log likelihood and proper scores;
- treatment-minus-control effect and population-distribution losses;
- held-out modality likelihood;
- lineage/fate/hazard likelihood;
- functional outcome likelihood;
- calibrated interval and survival objectives;
- mechanistic residual penalties;
- complexity/information-bottleneck penalties conditioned on predictive sufficiency.

The latent-state dimension is not selected by reconstruction quality alone. The state should be no
larger than necessary to make raw past evidence redundant for the declared future query, within a
predeclared tolerance.

## Validation doctrine

### Independent split units

The primary split unit follows how the experiment was conducted: well, plate, biological replicate,
donor, animal, clone, perturbation, or study. Cells within one experimental unit cannot be scattered
randomly across train and test for a claim whose intervention or outcome was assigned at that unit.

Maintain frozen benchmark families for:

- held-out wells/plates and biological replicates;
- future times and longer horizons;
- doses and interpolation/extrapolation ranges;
- interventions, mechanisms, and drug–gene combinations;
- donors, cell lines, genotypes, and environments;
- entire clones or lineages;
- missing modalities and different assay panels;
- entire external datasets/laboratories;
- deliberate OOD species, cell systems, interventions, and horizons.

### Required metrics

Use metrics aligned to returned distributions and query targets:

- negative log predictive density;
- CRPS and energy score;
- interval coverage, sharpness, PIT or reliability diagnostics;
- MMD/Wasserstein or suitable distributional distances for cell populations;
- treatment-minus-control effect size, sign, rank, and dose-response error;
- AUROC/Brier or survival/hazard metrics for events and fate;
- lineage-transition calibration;
- OOD AUROC plus risk–coverage curves;
- belief-only versus belief-plus-history performance difference;
- planner top-k recovery and regret when planning is enabled.

Reconstruction loss, cluster purity, UMAP appearance, and annotation concordance are diagnostic—not
proof of a sufficient causal state.

### Mandatory baselines

Every benchmark includes query-appropriate versions of:

- no-change/persistence;
- matched-control resampling;
- condition or perturbation mean;
- nearest known perturbation or condition;
- pseudobulk generalized linear models;
- simple dose/time and hierarchical random-effects models;
- low-rank or linear state-space models;
- a simple conditional generative or transport baseline when appropriate.

A complex biological backend graduates only if it beats these baselines on frozen external targets
and calibration—not because it is more expressive.

### Predictive sufficiency test

For a held-out future target, compare equally capable predictors:

```text
M_1: \hat Z_{t+h} = f(B_t^Q, U, E, Q)

M_2: \hat Z_{t+h} = f(B_t^Q, H_{\le t}, U, E, Q)
```

If `M_2` materially improves a predeclared proper score, the belief is not sufficient for that
query. Expand its factors or narrow the query/support claim. This diagnostic requires genuinely
held-out future evidence; reconstruction cannot substitute for it.

### Counterfactual validation limit

One cell cannot reveal both potential outcomes. Public data therefore validate counterfactual
claims through randomized or well-controlled population distributions, matched controls,
replicates, sister/lineage designs, and external replication. The system must not describe these as
observed same-cell counterfactual truth.

## Model registry and reproducibility

Every registered biological model artifact must bind:

- code commit and environment lock;
- exact training/validation/test dataset manifests and content hashes;
- preprocessing/normalization versions;
- query compiler and posterior schema versions;
- model configuration and weights hashes;
- frozen split memberships;
- scientific metrics with confidence intervals;
- support envelope, OOD thresholds, abstention policy, and known failure modes;
- causal-status and permitted-use statement;
- license compatibility across all constituent data.

Recursive inference and forecasting already require model/configuration/posterior compatibility.
Production artifacts extend that requirement to biological training support and calibration
identity.

## First two vertical slices

### A. K562/A549 population response

**Scope:** cultured human cancer cell lines; CRISPR and small molecules; roughly 3–72 hours;
transcript distribution first, then signaling, morphology, proliferation, and viability where
supported.

**Direct-response candidates:** Norman/Replogle K562 genetic perturbations, K562/A549 arms verified
within sci-Plex or Tahoe, and compatible MIX-Seq/PRISM conditions.

**Transported-prior candidates:** breast-cancer DREAM signaling, PBMC Bodenmiller signaling,
mostly-U2OS JUMP morphology, SHARE-seq baselines, and DepMap stable context. These may inform priors
only under declared transport assumptions; they cannot serve as direct K562/A549 validation unless
an exact overlapping system and protocol are independently verified.

**First releasable query:** given cell line, baseline context, compound or genetic intervention,
dose, and horizon, estimate a calibrated distribution over supported future transcriptional targets
and abstain outside observed systems/interventions/horizons.

**Why first:** it has the densest intervention coverage, tractable culture environments, replicated
controls, and overlapping public systems. It exercises the full data and validation spine before
adding individual-cell claims.

### B. Primary human T-cell stimulation and recovery

**Scope:** activation, repeated antigen exposure, exhaustion, withdrawal/rechallenge, and recovery;
hours-to-days horizons; donor-conditioned molecular and functional targets.

**Evidence bridge:** donor-aware perturbation screens, repeated-stimulation datasets, T-cell
multimodal atlases, Perturb-CITE-seq, and available killing/cytokine/metabolic outcome studies.

**First releasable query:** estimate calibrated donor/population-level distributions of activation
or recovery markers and supported future functions under a declared stimulation/intervention
history.

**Constraint:** many molecular and functional endpoints occur in separate assays or studies. The
backend must surface transport uncertainty and may not imply they were measured in the same cells.

## Planning is a gated downstream capability

Measurement value-of-information and intervention choice must wait until calibrated target decoders
exist. A planner integrates over:

- the full current posterior;
- candidate intervention/environment uncertainty;
- query target and horizon;
- predictive and transport uncertainty;
- cost, dose, timing, toxicity, and feasibility constraints.

Retrospective hidden-intervention experiments can evaluate pseudo-prospective ranking and regret.
They do not replace real prospective laboratory validation. No planner may propose arbitrary latent
coordinates or unsupported interventions simply because the transition model returns numbers.

## Scientific non-goals and fail-closed boundaries

- No universal or minimal state claim across all organisms, tissues, assays, and tasks.
- No fabricated same-cell trajectories from destructive snapshots.
- No missing-as-zero semantics or silent global batch correction.
- No causal claim from a perturbation label alone.
- No assumption that current RNA alone is Markovian or functionally sufficient.
- No clinical-use claim from retrospective public datasets.
- No random-cell test split presented as intervention or donor generalization.
- No validation based mainly on clusters, embeddings, or reconstruction.
- No numeric output when a target decoder, unit, horizon, factor, or support condition is absent.

The mature system should be judged by a narrower and more defensible statement:

> For this declared query and support envelope, the returned belief retains the available
> information needed to predict these future molecular or functional outcomes under these
> interventions and environments, with calibrated uncertainty and explicit failure modes.
