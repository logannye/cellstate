# Cell-state full-build roadmap

This roadmap turns the contract scaffold into a scientifically defensible system for estimating a
query-conditioned probability distribution over hidden, causally relevant cellular state. The
system will be built as a family of support-bounded models. It will not claim that one embedding is
a universal cellular state.

The detailed model, data, and evaluation design is in
[`architecture/full-buildout.md`](architecture/full-buildout.md). This file defines implementation
order and graduation gates.

## Program rules

The following rules apply to every phase:

1. Biological training, validation, and test evidence comes only from publicly downloadable real
   cell-biology datasets with accession/version, license, checksum, and provenance.
2. Synthetic examples are permitted for unit and property tests, but never as biological evidence
   or as a substitute for a real benchmark.
3. Dataset eligibility is claim-specific. Destructive snapshots support population transitions,
   not reconstructed individual-cell trajectories.
4. Splits follow independent experimental units—not random cells—and are frozen before model
   selection.
5. Every model declares its query family, system boundary, assays, interventions, environments,
   horizons, and OOD/abstention behavior.
6. Each phase graduates on scientific acceptance evidence. Running code and low reconstruction
   error are not graduation criteria.

## Target verticals

### Vertical A: cultured-cell population response

The first engineering target is K562 and A549 response to CRISPR perturbations and small molecules
over hours to days. Initial targets are future transcript distributions and, where cross-study
support is adequate, phosphosignaling, morphology, proliferation, and viability.

This vertical is first because public perturbation data are broad, controls are relatively clean,
and overlapping cell systems make external replication possible. Cross-study overlap is a
population-level transport bridge, never a claim that cells were paired across studies.

### Vertical B: primary human T-cell state and recovery

The first translational target is primary human T-cell activation, repeated stimulation,
exhaustion, withdrawal/rechallenge, and recovery. Targets include persistence, killing, cytokine
production, metabolism, and molecular response. This vertical is biologically richer but has less
uniformly paired public evidence, so it requires stricter transport assumptions and abstention.

## Phase 0 — freeze scientific scope and benchmark semantics

**Objective:** make invalid claims and leakage difficult before introducing a biological model.

Deliverables:

- A frozen query-family specification for Vertical A, including target, horizon, intervention,
  environment, and precision requirements. Vertical B remains deferred until its own evidence and
  query-family review is complete.
- Distinct individual-cell, clone, population, and spatial-niche belief-subject semantics, with ADR
  0005 deciding whether they use separate public contracts or a typed subject descriptor and
  whether that decision requires a v1-to-v2 schema change.
- Dataset capability and claim ledger: observational, interventional, longitudinal, lineage,
  same-cell, multimodal, spatial, and functional evidence are separate fields.
- Dataset, normalization, benchmark, and split manifest schemas.
- Frozen validation protocols and mandatory baselines for the first benchmark.
- Written causal-status vocabulary: associative, identified population effect, transported,
  mechanistic extrapolation, and unsupported.

Graduation gate:

- A schema review can represent each selected public dataset without fabricating same-cell links.
- Leakage checks reject random-cell and shared-well/plate/donor/clone leakage.
- A query-support check can explain why a dataset is or is not eligible for every planned loss and
  metric.
- No training code is required to demonstrate these properties.

## Phase 1 — reproducible public-real-data foundation

**Objective:** acquire and normalize public datasets without losing experimental design or raw
provenance.

Deliverables:

- Immutable bronze storage for exact downloaded bytes and content hashes.
- Silver event/observation storage preserving raw counts, missingness, censoring, batch, controls,
  donor, well/plate, intervention assignment, realized engagement, environment, time, clone, and
  spatial relationships.
- Gold query-specific examples generated from versioned transformations and frozen splits.
- Ontology and unit mappings for genes, proteins, chemicals, cell types, assays, species, doses,
  and time.
- Source adapters for AnnData/HDF5, matrix-market/GEO, Parquet/Zarr, FCS, and OME-Zarr as demanded by
  selected datasets.
- Checksum-pinned public-real “golden slices” small enough for integration and CI tests.

Initial acquisition order follows the first population benchmark, then expands the subject and
timescale coverage:

1. One replicated K562 perturbation source for Vertical A's first population benchmark.
2. Live-seq (`GSE141064`) for a separate same-cell state-to-future contract test.
3. LARRY (`GSE140802`) and cortical lineage Perturb-seq (`GSE284197`) for clone/fate structure.
4. MIX-Seq and sci-Plex for cell-line/drug population response.
5. DREAM/Bodenmiller signaling for fast response and measurement likelihoods.
6. Multiome Perturb-seq and Perturb-CITE-seq for perturbed same-cell multimodality.
7. Perturb-FISH/Perturb-map for neighborhood and spatial intervention effects.
8. JUMP `cpg0000` profiles plus a bounded image subset for physical-state modeling.
9. Targeted streamed subsets of Tahoe-100M; avoid an indiscriminate full download initially.

Graduation gate:

- Two independent machines can reproduce the same golden slices and split memberships from
  manifests.
- Every normalized value traces to source bytes and a versioned transformation.
- License enforcement rejects an artifact whose declared use is incompatible with its dataset.
- At least one complete K562/A549 benchmark and one longitudinal or lineage benchmark pass data
  integrity and leakage audits.

## Phase 2 — assay observation models and posterior infrastructure

**Objective:** make the belief mean “uncertainty about biology after accounting for measurement,”
not a compressed assay vector.

Deliverables:

- Count-aware RNA and ATAC likelihoods; calibrated protein/phosphoprotein, image, metabolic, and
  functional likelihoods as the selected data permit.
- Explicit nuisance variables for library/capture effects, staining panels, segmentation quality,
  plate, batch, and detection/censoring.
- Shared and modality-private latent factors, with held-out-modality inference tests.
- Posterior artifact format for samples/particles, weights, structured state factors, and uncertainty
  decomposition.
- Amortized inference interfaces supporting filtering for deployment and smoothing only when the
  evidence cutoff allows it.

Graduation gate:

- Likelihoods are calibrated on held-out technical and biological replicates.
- Posterior predictive checks reproduce supported assay statistics without erasing biological
  treatment effects.
- Missing-modality tests distinguish unavailable evidence from observed zero.
- A frozen belief cannot gain information from observations after its `as_of` time.

## Phase 3 — first population-response biological backend

**Objective:** ship the first useful, narrow biological backend for Vertical A.

Deliverables:

- Hierarchical K562/A549 priors conditioned on supported genotype, culture, and environment.
- Endpoint and multi-horizon controlled transition models for CRISPR and small molecules.
- Query-target decoders for future RNA distributions and supported functional targets.
- Dose/time representation, intended-versus-realized perturbation handling, and matched controls.
- Explicit inter-study transport variables rather than globally corrected pooled cells.
- Registered model/data cards and an API-level support envelope with abstention.

Graduation gate:

- Beats persistence, control resampling, perturbation mean, nearest known condition, pseudobulk GLM,
  and simple hierarchical/low-rank baselines on frozen proper-score metrics.
- Treatment-minus-control effects and dose-response order are accurate on held-out wells,
  replicates, doses, and perturbations.
- Predictive intervals attain declared coverage and OOD risk decreases monotonically with
  abstention.
- Performance replicates on a completely held-out public study where compatible evidence exists.
- Results are described as population-level effects unless individual-cell linkage was actually
  observed.

## Phase 4 — temporal dynamics, events, and branching

**Objective:** represent evolution rather than only condition-to-endpoint mappings.

Deliverables:

- Controlled neural SDE/ODE dynamics with explicit continuous process uncertainty.
- Hazards and jump kernels for division, death, differentiation, contact change, and commitment.
- Particle or mixture inference around bifurcations and rare transitions.
- Individual-cell filtering validated only on truly tracked/sampled cells such as Live-seq and
  time-lapse imaging.
- Clone-level inheritance and fate models using LARRY, CellTagging, lineage Perturb-seq, and
  appropriate imaging tracks.
- Population distribution dynamics for destructive time courses; no invented identity matching.

Graduation gate:

- Multi-horizon future scores beat endpoint-only and no-change baselines at frozen temporal cutoffs.
- Event hazards and lineage/fate probabilities are calibrated on entirely held-out cells or clones,
  as the design requires.
- Belief-only prediction is non-inferior to belief-plus-history within a predeclared sufficiency
  margin, or the state/query is revised.
- The backend correctly distinguishes individual, lineage, and population forecasts in serialized
  results and model cards.

## Phase 5 — multimodal, environmental, and spatial context

**Objective:** model how context changes the state and its future consequences.

Deliverables:

- RNA/ATAC/protein/signaling/morphology/metabolism fusion through assay likelihoods.
- Neighborhood graphs, regional environments, transport fields, contacts, and cell–cell interaction
  terms where observations support them.
- Perturb-FISH/Perturb-map spatial benchmarks and multimodal perturbation benchmarks.
- Mechanistic GRN, signaling, ligand–receptor, and metabolic constraints as auditable soft priors.
- Counterfactual transport diagnostics separating interpolation, transport, and unsupported
  extrapolation.

Graduation gate:

- Multimodal/contextual models improve future query-target proper scores—not merely reconstruction—
  on held-out experiments.
- Context effects replicate across independent samples/studies where possible.
- A modality or neighborhood has predeclared incremental value where claimed; removing an
  uninformative input does not worsen calibration, while removing an informative input changes
  uncertainty and accuracy in a measured, biologically interpretable way.
- Mechanistic constraints improve external prediction or calibration and can be ablated without
  changing contract semantics.

## Phase 6 — translational T-cell vertical

**Objective:** apply the proven infrastructure to primary human immune-cell state and function.

Deliverables:

- Donor-aware priors and transport diagnostics for primary T-cell experiments.
- Dynamics for activation, repeated antigen exposure, exhaustion, withdrawal, and recovery.
- Molecular and supported functional decoders for persistence, killing, cytokines, and metabolism.
- Explicit reconciliation of local/public T-cell datasets by condition and population—never
  fabricated same-cell pairing across studies.
- Donor-, intervention-, context-, and whole-study-held-out evaluations.

Graduation gate:

- Functional distributions are calibrated on held-out donors or studies, not cells from familiar
  donors.
- The model improves over molecular-only and condition-mean baselines.
- Transport limitations and clinically unsupported uses are explicit and mechanically enforced.
- No patient-level or clinical decision claim is made without a separately designed validation
  program.

## Phase 7 — measurement value and intervention planning

**Objective:** choose useful next measurements and interventions only inside validated support.

Deliverables:

- Decision-relevant value-of-information over query targets, candidate interventions,
  environments, and horizons.
- Risk-aware intervention simulation using full predictive distributions.
- Constraint handling for dose, timing, cost, toxicity, and experimental feasibility.
- Retrospective blinded/pseudo-prospective ranking benchmarks on hidden interventions.
- A fail-closed planner that rejects incompatible model versions, targets, units, horizons, and OOD
  candidates.

Graduation gate:

- Measurement selection reduces calibrated uncertainty or decision regret on held-out real
  experiments versus cost-matched policies.
- Intervention ranking improves top-k recovery and regret over simple effect-size and nearest-
  neighbor policies.
- Recommendations are limited to supported systems/candidates and communicate uncertainty and
  causal status.
- Real prospective experimental validation remains a requirement before operational biological or
  clinical claims.

## Immediate implementation queue

The next development work should proceed in this order:

1. **Completed foundation:** harden the experimental dataset capability/claim manifest scaffold;
   normalization and split manifests remain separate follow-on contracts.
2. Freeze Vertical A's first query specification, outputs, horizons, intervention/dose scope,
   causal status, metrics, and mandatory baselines.
3. Resolve the individual/clone/population/spatial-niche belief-subject contract in ADR 0005 and
   decide whether it requires schema v2.
4. Define the frozen benchmark and split manifests plus leakage validators before acquiring data.
5. Build immutable acquisition plus provenance/checksum and use-policy verification.
6. Admit and ingest a small real-data golden slice from one K562 perturbation dataset, then Live-seq
   for a separate longitudinal contract test.
7. Implement the RNA observation baseline and simple population-response baselines before a deep
   biological model.

That sequence creates a reproducible scientific spine early, exercises both individual and
population semantics, and gives every later model a trustworthy comparison target.
