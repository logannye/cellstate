# Persistent project context

## Mission

`cellstate` estimates a query-conditioned probability distribution over hidden, causally relevant
cellular state. It is a belief-state and forecasting system, not a cell-labeler, embedding service,
or claim that one universal minimal cellular state exists.

The repository currently contains backend-neutral contracts and a deliberately narrow
linear-Gaussian reference backend. That backend is for software and contract validation only. It is
not a biologically or clinically validated cell model.

## Architectural invariants

- Keep `estimate_cell_state`, `evolve_cell_state`, and `choose_intervention` separate. Estimation,
  forecasting, and decision-making have different scientific contracts.
- Every state is scoped to a declared query, targets, horizons, system boundary, and support
  envelope. Unsupported or unidentifiable requests must abstain explicitly.
- Return posterior and predictive distributions, not point-only latent vectors. Preserve
  measurement, biological, parameter, model, and transport/counterfactual uncertainty.
- Preserve causal time, intended assignment versus realized perturbation, environment, lineage,
  neighborhood, provenance, missingness, censoring, and assay nuisance variables.
- Never turn destructive cross-sectional measurements into a same-cell trajectory. Distinguish
  individual-cell, clone, population, and spatial-niche beliefs and evidence.
- Use modality-specific observation likelihoods. Do not treat missing as zero or make a globally
  batch-corrected matrix the source of truth.
- Treat GRNs, signaling maps, metabolic constraints, and other mechanisms as weighted, auditable,
  overridable priors or penalties—not unquestionable ground truth.
- Planners may score calibrated, query-target predictive distributions only. They must not optimize
  backend-private latent coordinates.
- Breaking semantic contract changes require a schema-version and migration decision. Run
  `make schemas` after contract changes.

## Public-real-data policy

- Biological training, validation, benchmark, and scientific claims must use publicly downloadable
  real cell-biology data with recorded accession, version, license, retrieval date, and checksum.
- Tiny synthetic arrays and the reference backend are allowed for unit/property tests of algebra,
  serialization, and software behavior. They do not count as biological evidence.
- Large biological artifacts, images, genomic data, and model weights stay outside Git and are
  addressed through immutable manifests and content hashes.
- Dataset eligibility is claim-specific. A dataset may train an observation model without being
  eligible for causal, temporal, lineage, same-cell, spatial, or functional validation.
- Public availability is not equivalent to permission for every use. Encode license and usage
  restrictions in dataset manifests and enforce them in workflows.
- No single known public dataset satisfies the complete system contract. Build a composite evidence
  portfolio and never imply that measurements from different studies are paired at the cell level.

## Validation rules

- Split by the true independent experimental unit: well, plate, biological replicate, donor,
  animal, clone, intervention, or study as appropriate—not by randomly sampled cells.
- Maintain frozen tests for held-out future times, doses, interventions/mechanisms, combinations,
  donors/cell lines, environments, lineages, modalities, and entire external studies.
- Evaluate future-target proper scores, calibration, intervention effects, predictive sufficiency,
  OOD risk/coverage, and decision regret. UMAP or cluster appearance is not core validation.
- Compare against persistence, matched-control, perturbation-mean, nearest-condition, simple linear
  or hierarchical models, and other query-appropriate baselines.
- Test predictive sufficiency by comparing equally capable predictors using belief-only versus
  belief-plus-raw-history inputs. Narrow the query or expand the state when history still helps.
- Report causal status explicitly: associative, identified population effect, transported under
  stated assumptions, mechanistic extrapolation, or unsupported.

## Current milestone

Phase 0 is active. The experimental public-data/claim manifest scaffold is hardened; next freeze
Vertical A's first query and benchmark specification and resolve belief-subject semantics in
proposed ADR 0005. Do not begin biological model training or treat candidate datasets as admitted
before those gates are complete.

`docs/roadmap.md` is the sole authority for sequence and status. The full target architecture lives
in `docs/architecture/full-buildout.md`; accepted rationale lives in ADRs. The sanitized local audit
is in `docs/data/evidence-inventory.md`. Avoid copying detailed phase ordering into other files.

Every phase has a scientific graduation gate. Code completion alone does not graduate a biological
model.

## Working practices

- Read `README.md`, `docs/architecture/overview.md`, `docs/architecture/data-contracts.md`,
  `docs/validation/scientific-validation.md`, `docs/architecture/full-buildout.md`, and
  `docs/roadmap.md` before changing biological semantics or adding a backend. Read proposed ADR 0005
  before changing subject or sampling semantics.
- Preserve user changes in a dirty worktree. Keep changes focused and add acceptance tests in
  proportion to scientific and software risk.
- Prefer manifest-driven, reproducible workflows. Record assumptions and support limits in model
  and dataset cards rather than leaving them implicit in notebooks.
