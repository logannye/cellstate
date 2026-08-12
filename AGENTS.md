# Persistent project context

## Mission

`cellstate` estimates a query-conditioned probability distribution over hidden, causally relevant
cellular state. It is a belief-state and forecasting system, not a cell-labeler, embedding service,
or claim that one universal minimal cellular state exists.

The repository currently contains backend-neutral contracts and a deliberately narrow
linear-Gaussian reference backend. That backend is for software and contract validation only. It is
not a biologically or clinically validated cell model.

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
  OOD risk/coverage, and decision regret. UMAP or cluster appearance is not core validation.
- Compare against persistence, matched-control, perturbation-mean, nearest-condition, simple linear
  or hierarchical models, and other query-appropriate baselines.
- Test predictive sufficiency by comparing equally capable predictors using belief-only versus
  belief-plus-raw-history inputs. Narrow the query or expand the state when history still helps.
- Report causal status explicitly: associative, identified population effect, transported under
  stated assumptions, mechanistic extrapolation, or unsupported.

## Current milestone

Phase 0 is active. The experimental public-data/claim manifest `0.3-experimental` supports
canonical repeated claim assessments, exact loss/metric eligibility, layered permission
resolution, content-addressed dataset slices, interval-valued evidence clocks, and machine-checked
reviewed representability ledgers. Those ledgers validate bound declarations and attestations; the
current verifier does not resolve source bytes or replay selectors. Schema v2 enforces typed
subjects, destructive evidence, bounded query
support, query-compiled state, perturbation realization, scientific readiness/abstention, causal
status, and standalone decision-oriented measurement selection. The Replogle K562 destructive-
population proof and the GSE141064 Live-seq individual functional-recorder proof passed on
2026-08-09 without authorizing data use or admitting either source for model training. The first
Vertical A component benchmark is now frozen around the corrected sci-Plex3 K562 24-hour endpoint,
with exact well/plate partitions and an explicit `COMPONENT_BENCHMARK` result. Its assessment and
permission gates pass, but its executable metric, leakage, baseline, and performance gates have not
run, so it is not scientifically admitted. The experimental biological-bundle contract and first
population assay-response scaffold are now in place. The scaffold exhaustively maps all original
model stages, exposes no public cell-state operation, contains no trained weights, and rejects
execution. Its direct context-and-assignment to 24-hour assay response is not a hidden-state belief.
Contract v0.1 now implements the trusted admission boundary around those declarations. It streams
exact-byte verification, accepts real-data execution sources only from an authenticated typed-
workflow selection, authenticates isolated loader and semantic-evaluator observations against
external nonserialized HMAC trust roots, checks loaded objects against an application-owned
interface registry, distinguishes verified validation semantics from passed results, recompiles
query-dependent prerequisites, and returns only just-in-time reverified runtime handles. Persisted
receipts never authorize execution by themselves; one immutable code snapshot is both hashed and
loaded by the registry-owned trusted JIT loader, so independently supplied objects cannot borrow
admitted bytes. The sci-Plex3 artifact remains a non-executable `SCAFFOLD`. Item 11 completed its
immutable `p1` loader, close-reauthenticated count scan, and six mandatory non-admissible baselines.
Item 12 candidate revisions v1–v4 all failed closed without issuing a model, prediction, metric,
calibration, or lifecycle result. Item 12.1a's exact harness, tests, parent driver, and historical
reports are canonical under `audits/item12_1a`. A source-code audit subsequently confirmed that v4's
dose objective omits the full ELBO's `94785/768` equal-well multiplier, so v4 is retired and the
planned real-`p1` Item 12.1b replay must not run. Item 12.2 has now completed its source-free v5
software scope: one exact equal-well objective and compatible all-well action/context M-step;
independently checked derivatives and nondecrease; exact-positive, request-bounded sampling;
immutable-generation atomic publication; exact executable/input/staged-output closure; reproducible
Linux `amd64` OCI identities whose clean-final archive is byte-identical across three independent
native and emulated builder environments under frozen inputs and toolchain; and parent-owned whole-
container wall/memory containment with host-effective UID/GID filesystem isolation on native Linux.
The final image excludes build-host caches, and its exact Buildx/BuildKit authority is frozen by the
runtime lock and workflow. It opened no protected source, ran no real-`p1` fit, and issued no
candidate artifact, plan, observation, evidence, or lifecycle result. The component therefore
remains `SCAFFOLD`. Item 12.3's source-free authorization, runtime-preparation, containment, and
terminal-evidence control plane is now assembled for a separately authorized, version-bound,
nonissuing real-`p1` v5 execution. It grants no authority: the canonical pending proposal is absent,
no exact proposal digest has been approved or consumed, and no Item 12.3 source run has occurred.
Durable runtime distribution is complete through the separate immutable, release-and-asset-attested
`sciplex3-v5-runtime-20260811-locked` GitHub Release. A future approved attempt must dispatch the
trusted workflow from `refs/heads/main` at exact execution commit `C`, supplying both the exact
proposal SHA-256 and its proposal-only child commit `D`. The workflow treats `D` only as inert object
data and the asset-free attempt-release target, never checks it out or executes it, remains on `C`,
and reauthenticates the runtime before source acquisition. The monolithic source
must be transferred and snapshotted whole and its full selector axes decoded to resolve `p1`; only
`p1-train` expression/raw-count values may be decoded. Held-out expression/count values, endpoints,
outcomes, scoring, and lifecycle authority remain sealed; public split and design metadata are not
an access grant. Only bounded sanitized terminal evidence may leave the ephemeral runner, and it
grants no lifecycle or scientific authority. No biological runtime or validated belief may be
registered before its separate operation-specific admission and scientific gates pass.

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
- Prefer manifest-driven, reproducible workflows. Record assumptions and support limits in model
  and dataset cards rather than leaving them implicit in notebooks.
