# Persistent project context

## Mission

This project builds one thing: a system that computes a faithful and accurate representation of
hidden cellular state. `cellstate` estimates a query-conditioned probability distribution over
hidden, causally relevant cellular state and forecasts its future under declared interventions. It
is not a cell-labeler, an embedding service, or a claim that one universal minimal cellular state
exists.

Everything else here — contracts, schemas, manifests, admission machinery, benchmark tooling,
containment, runtime infrastructure — is instrumental. None of it is an end, and work that advances
no state capability is not scheduled. A change that presents instrumental work as the project's
substance is misaligned regardless of how well it is engineered.

The repository currently contains backend-neutral contracts, dataset and benchmark admission
machinery, and a deliberately narrow linear-Gaussian reference backend. That backend is for
software and contract validation only. It is not a biologically or clinically validated cell
model.

## Architectural invariants

- Keep `estimate_cell_state`, `evolve_cell_state`, `choose_intervention`, and
  `recommend_next_measurement` separate. Estimation, forecasting, control, and buying information
  have different scientific contracts; a measurement decision must bind the exact downstream
  candidate set and utility.
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
  out-of-support risk-coverage, and decision regret. UMAP or cluster appearance is not core
  validation.
- Beat the observational floor: persistence, matched control, condition mean, nearest condition,
  pseudobulk GLM, and simple hierarchical and low-rank baselines, each individually, with an
  interval that excludes zero. The mandatory set is ledger entry `S9` in `docs/roadmap.md`; do not
  restate a shorter list here.
- Test predictive sufficiency by comparing equally capable predictors using belief-only versus
  belief-plus-raw-history inputs. The gain must carry a bootstrap interval grouped at the declared
  independent experimental unit; a gain reported without an interval is not a verdict. Narrow the
  query or expand the state when history still helps.
- Report causal status explicitly: associative, identified population effect, transported under
  stated assumptions, mechanistic extrapolation, or unsupported.

## Current milestone

Phase 1 is active: make the faithfulness tests executable and measure the observational floor.

Phase 0 is complete. Schema v2 enforces typed subjects, destructive evidence, bounded query support,
query-compiled state, perturbation realization, scientific readiness and abstention, causal status,
and standalone decision-oriented measurement selection. Manifest `0.3-experimental` carries
claim-specific eligibility, content-addressed slices, interval-valued evidence clocks, and
machine-checked reviewed representability ledgers, which validate bound declarations and attestations
without resolving source bytes or replaying selectors.

No biological backend is registered, no belief has been emitted by a biological model, and no
benchmark is scientifically admitted. The repository has produced no scientific number: no metric
implementation exists in any frozen suite, the sufficiency and calibration functions have no caller
outside tests, and the observational floor is unmeasured.

The sci-Plex3 K562 24-hour component remains a non-executable `SCAFFOLD` and its benchmark remains
`COMPONENT_BENCHMARK`, not admitted. Its estimand has no admissible pre-cutoff observation and one
horizon, so it cannot carry a sufficiency verdict. It is retained as data and split infrastructure and
as the Phase 1 metric proving ground, not as a step toward the purpose. Its `p2`, `p3`, and `p4`
endpoint values, outcomes, and scoring authority remain hard sealed; public split and design metadata
are not an access grant.

The Item 12.3 protected-execution control plane is suspended by the roadmap and fails closed on
dispatch. No proposal is to be approved and no protected execution is to be dispatched. The rank-16
continuous admixture candidate family is retired for the state path. Historical Item 11 and 12.x bytes
under `audits/`, `benchmarks/`, and `containers/`, and ADRs 0011 and 0012, are frozen evidence, not
executable work. See ADR 0013 for the reordering and its rationale.

Next work is the implementation queue in `docs/roadmap.md`, in order.

`docs/roadmap.md` is the sole authority for sequence and status. The full target architecture lives
in `docs/architecture/full-buildout.md`; accepted rationale lives in ADRs. The sanitized local audit
is in `docs/data/evidence-inventory.md`. Avoid copying detailed phase ordering into other files.

Every phase has a scientific graduation gate. Code completion alone does not graduate a biological
model.

## Working practices

- Read `README.md`, `docs/architecture/overview.md`, `docs/architecture/data-contracts.md`,
  `docs/validation/scientific-validation.md`, `docs/architecture/full-buildout.md`, and
  `docs/roadmap.md` before changing biological semantics or adding a backend. Read accepted ADR 0005
  before changing subject or sampling semantics.
- Preserve user changes in a dirty worktree. Keep changes focused and add acceptance tests in
  proportion to scientific and software risk.
- Name the ledger capabilities a change advances. A change that advances none is not scheduled.
  A commit that amends the roadmap may not also implement the work it authorizes. A change to phase
  order, to the ledger, or to a graduation gate requires a contemporaneous ADR.
- Cite artifacts and ADRs, never roadmap queue IDs, in any document other than `docs/roadmap.md`.
  Queue IDs are ordinals in the current queue and are not stable identifiers; the historical
  Item 1-12 numbering is bound into content-addressed manifests and must never be reused.
- Treat `audits/`, `benchmarks/`, `data_manifests/`, and `containers/` as frozen evidence, and
  accepted ADRs and `CHANGELOG.md` entries as historical records. Amend an ADR's Status line;
  never rewrite its body or its decision.
- Prefer manifest-driven, reproducible workflows. Record assumptions and support limits in model
  and dataset cards rather than leaving them implicit in notebooks.
