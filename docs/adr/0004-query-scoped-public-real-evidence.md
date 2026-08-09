# ADR 0004: Query-scoped backends and public-real biological evidence

- **Status:** Accepted
- **Date:** 2026-08-09

## Context

No universal cellular state is sufficient for every intervention, environment, horizon, and target.
Likewise, no public dataset jointly measures at useful scale the same cell's complete perturbation
and environment history, multimodal molecular state, lineage and neighborhood, and later functional
outcomes.

Most public single-cell assays are destructive. Cells sampled at different times from the same
condition are therefore exchangeable population samples, not repeated observations of one cell.
Combining them under a single cell identifier would invent longitudinal evidence. Cross-study
integration also cannot manufacture causal or same-cell linkage that the experimental designs did
not record.

The repository's synthetic linear-Gaussian backend is valuable for executable contract tests, but
it is not biological evidence.

## Decision

1. Biological implementations will be a family of **query-scoped, support-bounded backends**. A
   model version supports only the species, systems, subject kinds, assays, interventions, doses,
   environments, horizons, and outputs declared in its support manifest.
2. Biological training, calibration, validation, and benchmark claims will use traceable **real,
   publicly downloadable experimental data**, either directly or through registration that does
   not require access approval. Synthetic examples may test software and mathematics, but they
   never contribute evidence for biological validity.
3. Dataset eligibility is claim-specific. Each dataset receives a machine-readable manifest that
   records experimental units, sampling design, temporal linkage, modalities, interventions,
   controls, outcomes, lineage/spatial evidence, access terms, and the claims it may support.
4. Subject semantics are explicit. Individual-cell, clone, population, and spatial-niche beliefs
   are distinct contracts. Destructive snapshot data may train population-distribution dynamics;
   they may not be represented as an observed individual trajectory.
5. Dataset-specific observation models and explicit missingness are retained. Studies are joined
   only through measured overlaps or declared transport assumptions, never through fictitious
   paired rows or missing-as-zero values.
6. Benchmark splits are made at the randomized or shared experimental unit--for example donor,
   animal, well, plate, clone, cell line, perturbation, or study--rather than by randomly sampled
   cells.
7. A state is adequate only when future prediction from the belief is not materially improved by
   adding raw history, within the exact query scope. Calibration, OOD abstention, intervention
   effects, and external-study replication are separate graduation gates.
8. Intervention and assay planning will be implemented only after query-target predictive
   distributions are calibrated. Planning never optimizes private latent coordinates or actions
   outside empirical support.

## Initial vertical slices

The first engineering vertical is a cultured human cancer-cell **population-response** backend for
drug and genetic perturbations, with explicit dose and time and transcriptomic, signaling,
morphology, proliferation, or viability targets where the experiments support them.

The first translational vertical is a primary human T-cell **stimulation, exhaustion, and recovery**
backend using repeated antigen, genetic perturbation, cytokine, withdrawal, and functional-outcome
studies. The two verticals share contracts and evaluation infrastructure but do not claim a shared
universal latent state.

Lineage/fate, longitudinal-cell, multimodal-observation, and spatial/neighborhood backends are
separate evidence-qualified verticals. Their implementation order is intentionally non-normative in
this ADR; `docs/roadmap.md` is the sole sequencing authority.

## Consequences

- The next implementation milestone is data governance and a frozen benchmark, not a large neural
  model.
- Schema v1 remains the stable contract kernel. Breaking subject, sampling, causal-support, and
  support-envelope semantics require an explicit schema v2 migration.
- A backend may graduate for one named query while remaining unsupported for others.
- Public data can support rigorous retrospective and pseudo-prospective validation, but cannot by
  itself validate a universal virtual cell or arbitrary counterfactual planning.
- Licenses and data-use restrictions are executable inputs to eligibility. Public availability is
  not assumed to permit unrestricted commercial model training.
