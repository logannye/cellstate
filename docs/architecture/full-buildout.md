# Full-build architecture: public-real-data cell-state system

## Status and purpose

This project computes a faithful and accurate representation of hidden cellular state. Everything
else — contracts, schemas, manifests, admission machinery, runtime infrastructure — exists to make
that representation trustworthy.

This document is the durable design target for the biological system that will produce it: the
system boundary, the formal estimand, belief-subject and evidence semantics, the data policy, the
modeling strategy, and the validation doctrine. It is written in the prescriptive present tense and
describes what the system must be, not what is built. It confers no schedule and no status.
[`../roadmap.md`](../roadmap.md) is the sole authority for implementation order, for the
state-capability ledger S1–S10, and for graduation status.

The representation is not a universal embedding called “cell state.” It is a **query-conditioned
belief** that meets predeclared predictive-sufficiency and calibration criteria for named future
outcomes under named interventions, environments, and horizons — the two tests in
[validation doctrine](#validation-doctrine).

No biological backend is registered, no belief has been emitted by a biological model, and no
benchmark is scientifically admitted. The existing linear-Gaussian backend remains a contract and
software reference. Nothing in that backend is biological validation.

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
B_t^Q = P(X_t^Q, Theta, R_{<=t}, Xi | H_{<=t}, C, Q)
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
P(Z_{t+h} | B_t^Q, do(U_{t:t+h}), E_{t:t+h}, Q)
```

where `Z` contains query targets rather than backend-private latent coordinates. The use of
`do(U)` in notation is not itself a causal guarantee. Every output must label its
evidential status:

- predictive association;
- identified population intervention effect;
- transported effect under enumerated assumptions;
- mechanistic extrapolation;
- unsupported.

The public boundary keeps four computations separate:

1. `estimate_cell_state` constructs the query-conditioned belief;
2. `evolve_cell_state` propagates that belief through one declared scenario;
3. `choose_intervention` compares an objective over a bounded candidate set; and
4. `recommend_next_measurement` evaluates whether buying one candidate assay improves that same
   downstream decision enough to justify its cost, delay, and collection effect.

The fourth operation consumes a standalone `MeasurementDecisionRequest` and returns a standalone
`MeasurementRecommendation`. It never mutates the belief or treats an estimator's internal
uncertainty ranking as a decision result.

## Belief subject and evidence semantics

Schema v2 prevents a common scientific category error: destructive assays sample different cells
at different times, and those observations do not constitute a longitudinal history of one cell.

The active domain distinguishes:

| Belief subject | Identity evidence | Permitted interpretation |
| --- | --- | --- |
| Individual cell | Direct tracking or viability-preserving repeated sampling | Conditional state and future of that cell |
| Clone/lineage | Heritable barcode, phylogeny, or parent–child tracking | Shared ancestry, inheritance, and fate probabilities |
| Population | Experimental sample, condition, well, or distribution | Population transition and response distribution |
| Spatial niche | Region plus cells/neighborhood graph | Contextual state and neighborhood-dependent response |

The four subjects are not interchangeable, and the available identity evidence — not convenience —
determines which one a representation may claim. A population belief is a complete instance of the
object, not a fallback: it is hidden, it is inferred from evidence, it evolves under intervention,
and it is subject to both faithfulness tests. Individual-cell claims require individual-cell
evidence.

Evidence also needs a role. The active domain enumerates nine: direct, ancestor, descendant,
sibling, clone aggregate, matched population, general population, spatial neighbor, and external
reference. Backends must reject evidence roles they do not model.

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
    ├── intervention planning, only after validation
    └── standalone measurement decision, only with validated EVSI components
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
  capabilities, independently scoped claim/loss/metric assessments, and acquired source-artifact
  records with retrieval time, resolved bytes, and checksums;
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

Loss functions and metrics consume only examples authorized by an exact assessment reference:
`(dataset-manifest fingerprint, assessment ID, assessment fingerprint)`. A loss assessment never
authorizes model selection or testing, a diagnostic metric never graduates a scientific claim, and
a declared split unit remains only a requirement until a content-addressed split manifest proves
membership.

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
| Individual state to future | Live-seq (`GSE141064`) | Viability-preserving state measurement followed by future phenotype | Reviewed cohort is on the order of tens of cells — too few independent units to bootstrap a sufficiency verdict |
| Lineage/fate | LARRY; CellTagging; `GSE284197` | Clone inheritance, differentiation, and perturbation/fate | Clone/sister linkage, not repeated whole-cell measurement |
| Spatial perturbation | Perturb-FISH; Perturb-map; Perturb-DBiT | Neighborhood-dependent response and spatial intervention | Targeted panels or terminal spatial assays |
| Physical dynamics | JUMP Cell Painting; MitoCheck | Morphology, batch robustness, division/death/event dynamics | Molecular state sparse or in separate studies |
| Metabolism/proteomics | SpaceM; SCoPE2; CCLE metabolomics; decryptM | Metabolic/protein observation and target-engagement anchors | Often baseline or population level |
| Population priors | CELLxGENE Census; DepMap | Cell-type/context priors and stable cell-line features | Observational/bulk; no dynamic cell belief |

Acquisition order is maintained in the [`roadmap`](../roadmap.md#implementation-queue) and is driven
by the requirements of the next state-bearing estimand, not by dataset size or availability. A source
carries a state estimand only when it supplies an admissible observation in the target modality
before the inference cutoff **on a unit that is also observed after it**, at least two horizons after
the cutoff, an identified intervention with matched controls, and enough independent experimental
units to bootstrap a verdict. No row in the table above has been shown to meet all four; each row
records a plausible role, and the source under review for the first state-bearing estimand is named
in the roadmap, not here. Full Tahoe and JUMP image downloads are intentionally deferred; stream or
download bounded subsets aligned to the current query family.

### Local-data conclusion

Previously downloaded real datasets contain valuable multimodal snapshots, perturbation screens,
clinical observational atlases, and bulk functional studies. They do not, individually or jointly,
provide a single coherent complete-cell trajectory with every required causal and contextual field.
The sanitized, accession-level evidence for that conclusion is preserved in the
[local evidence inventory](../data/evidence-inventory.md).

The weaker question that actually gates the state path is whether any of them carries a state
estimand at all: an admissible pre-cutoff observation on a unit that spans the inference cutoff, two
or more horizons after it, an identified intervention with matched controls, and enough independent
units to bootstrap. Three failure modes recur and are recorded here so they are not rediscovered — a
same-cell longitudinal cohort of tens of cells has too few independent units; a design with
approximately one library per treatment arm cannot separate intervention from library; and a
single-timepoint destructive screen has no pre-cutoff observation and one horizon. Which local
source, if any, survives this test is settled by the reviewed manifest and unit census scheduled in
the roadmap, not by this document.

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

A belief is **faithful** for a query when two things hold: it is predictively sufficient for that
query's declared targets, and its predictive distributions are calibrated. Both are numeric
verdicts, and **neither is a verdict without a sampling distribution.** A sufficiency gain or a
coverage error reported as a bare point number is a diagnostic, not a result: each is reported with
an interval obtained by resampling the declared independent experimental unit — a multiway grouped
bootstrap where units are crossed — and a sufficiency verdict additionally names the tolerance the
interval is compared against. Resampling cells inside a shared unit measures sampling noise, not
replication.

A test that fails, reported with its interval, is a result. Suppressing it is not.

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

Every metric suite frozen from [ADR 0013](../adr/0013-state-first-roadmap-reordering.md) forward
carries at least one differential-expression-weighted metric and one rank-based metric. Marginal
error and all-gene correlation are maximized by predicting no change and never stand alone. The
sci-Plex3 suite frozen by [ADR 0008](../adr/0008-sciplex3-k562-component-benchmark.md) predates
that requirement and is not retrofitted.

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

For a held-out future target, compare two predictors of declared equal capacity:

```text
M1: Z_{t+h} = f(B_t^Q, U, E, Q)
M2: Z_{t+h} = f(B_t^Q, H_{<=t}, U, E, Q)
gain = score(M1) - score(M2)
```

`score` is negatively oriented — a loss such as CRPS, lower is better — so `gain >= 0`. The belief
is sufficient for `Q` when the upper end of the interval on `gain`, bootstrapped at the declared
independent experimental unit, falls below the predeclared tolerance. If raw history materially
improves prediction, the belief is not sufficient: expand its factors, or narrow the query and the
support claim.

The test is meaningful only where a history exists. Under a query with no admissible pre-cutoff
observation, `M2` and `M1` receive the same inputs, the gain is identically zero, and the test
passes trivially — it is inapplicable, not satisfied. A query must therefore supply an admissible
pre-cutoff observation on a unit that is also observed after the cutoff, and at least two horizons
after it. This diagnostic requires genuinely held-out future evidence; reconstruction cannot
substitute for it.

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

## Verticals

Two verticals are planned, in order. Vertical B is deferred until Vertical A has produced a
sufficiency verdict; the roadmap is the authority for when either begins.

### A. Cultured-cell population state under genetic and chemical intervention

**Scope:** cultured human cell lines; CRISPR and small molecules; horizons set by the frozen
state-bearing estimand rather than assumed in advance; transcript and chromatin distributions first,
then signaling, morphology, proliferation, and viability where supported.

**Candidate direct evidence:** a perturbation source that observes the same experimental units both
before an inference cutoff and at two or more later times. Single-endpoint destructive screens —
Norman and Replogle K562 genetic perturbations, sci-Plex, Tahoe, MIX-Seq, and PRISM conditions —
supply population-response comparisons, assay likelihoods, and baselines, but none of them supplies
a pre-cutoff observation, so none can carry this vertical's estimand on its own.

**Transported-prior candidates:** breast-cancer DREAM signaling, PBMC Bodenmiller signaling,
mostly-U2OS JUMP morphology, SHARE-seq baselines, and DepMap stable context. These may inform priors
only under declared transport assumptions; they cannot serve as direct K562/A549 validation unless
an exact overlapping system and protocol are independently verified.

**First releasable query:** the complete, deliberately unfrozen estimand is maintained in the
[Vertical A scientific estimand](../verticals/vertical-a-estimand.md). In brief: given a typed
cultured-cell population, cutoff-safe baseline evidence and causal history, a bounded compound or
genetic intervention and environment, and two or more named horizons after the cutoff, estimate
calibrated distributions over supported future targets with explicit realization uncertainty, causal
status, transport assumptions, and abstention. The query is frozen only after its contract and
real-data representability gates pass, at the roadmap phase that freezes a state-bearing estimand.

**Why first:** it has the densest intervention coverage, tractable culture environments, replicated
controls, and overlapping public systems. It exercises the full data and validation spine before
adding individual-cell claims.

### B. Primary human T-cell state and recovery (deferred)

**Scope:** activation, repeated antigen exposure, exhaustion, withdrawal/rechallenge, and recovery;
hours-to-days horizons; donor-conditioned molecular and functional targets.

**Evidence bridge:** donor-aware perturbation screens, repeated-stimulation datasets, T-cell
multimodal atlases, Perturb-CITE-seq, and available killing/cytokine/metabolic outcome studies.

**First releasable query:** estimate calibrated donor/population-level distributions of activation
or recovery markers and supported future functions under a declared stimulation/intervention
history.

**Constraint:** many molecular and functional endpoints occur in separate assays or studies. The
backend must surface transport uncertainty and may not imply they were measured in the same cells.

**Deferred.** Richer biology, sparser paired evidence, and stricter transport assumptions mean this
vertical begins only after Vertical A has produced a sufficiency verdict with its interval.

## Planning is a gated downstream capability

Measurement value-of-information and intervention choice must wait until calibrated target decoders
exist. An intervention planner integrates over:

- the full current posterior;
- candidate intervention/environment uncertainty;
- query target and horizon;
- predictive and transport uncertainty;
- cost, dose, timing, toxicity, and feasibility constraints.

Measurement selection is a separate nested decision calculation. For candidate assay `a`, supported
gross EVSI has the form

```text
E_{Y_a | B}[max_u E[utility(u, Z) | B, Y_a]]
    - max_u E[utility(u, Z) | B]
```

and net value subtracts the explicitly declared assay, delay, and collection-effect penalties. A
backend may report that quantity only if it has:

- a calibrated assay-outcome model for `Y_a` under the current belief;
- a valid hypothetical posterior update for every integrated assay outcome;
- counterfactual replanning over the same objective and ordered intervention candidates; and
- a declared decision utility over supported query targets and horizons.

Entropy, marginal variance, or posterior covariance reduction does not meet this definition.
`NOT_EVALUATED` means the calculation was not performed; `UNSUPPORTED` means a required component
failed or lies outside support; `ABSTAINED` means supported numeric values do not clear the declared
threshold. None uses numeric sentinels. The linear-Gaussian contract reference intentionally
returns `NOT_EVALUATED`; it is not a measurement-value model.

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
- No assay recommendation based only on entropy, variance, or covariance shrinkage.

The mature system should be judged by a narrower and more defensible statement:

> For this declared query and support envelope, the returned belief retains the available
> information needed to predict these future molecular or functional outcomes under these
> interventions and environments, with calibrated uncertainty and explicit failure modes.
