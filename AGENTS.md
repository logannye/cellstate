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

A biological backend **is** registered and beliefs **have** been emitted from real cells. The
`gse274113` RNA observation model fits on Cas9-knockout human CD34+ progenitors — 14 libraries,
137,604 cells, a 100-gene panel declared a priori — and `estimate_arm("rep1", "GATA1")` returns a
typed `CellStateBelief` from a bare checkout in under a second. Its capability measurements are
computed, reproduce from committed bytes, and are pinned by tests. The three sentences that stood
here previously ("no biological backend is registered", "no belief has been emitted by a biological
model", "the repository has produced no scientific number") were true when written and have been
false since PR #27; the README retracted them in PR #34 and this file did not.

What remains true: **no benchmark is scientifically admitted, and the eligibility ledger is 0 of 10.**
No metric implementation exists in any frozen suite, `evaluate_marginal_calibration` and the
sufficiency functions still have no caller outside tests, and the observational floor is unmeasured.

**The "measured null" verdict on this deposit is WITHDRAWN.** It rested on a mean on-target
log₂ fold-change of about −0.06, read against a CRISPRi expectation of −1 to −2 — a threshold that
belongs to dCas9-KRAB, where repression forces the transcript down. GSE274113 is **Cas9
nuclease knockout** ([Science 10.1126/science.ads7951](https://doi.org/10.1126/science.ads7951)):
cutting destroys the protein, the transcript falls only through nonsense-mediated decay, and edits
that escape NMD give zero or positive fold change. On-target mRNA is therefore not a validity
control for this source, and the number carries no information about whether the perturbation
worked.

**The deposit is not empty, and that is now measured rather than assumed either way.**
`scripts/explore.py consistency` runs a within-library target-label permutation screen on the
committed slice — no fold, no fitted basis, no observation-variance model, no bound. Arms sharing a
target carry **0.1897** of the within-library sum of squares against a permutation null of
**0.0714 [0.0615, 0.0821]**, a ratio of **2.66×**, with **0 of 2000** draws reaching the observed
value (p = 5.0e-4, the floor that many draws can resolve). The null sits at 1/K for K = 14
libraries, exactly where label exchangeability puts it, and a test asserts that.

So "the substrate is empty" is no longer available as a complete explanation for the failing
ledger. **What that does not establish is the converse.** The signal is real and it is small, which
is consistent with the negative capability measurements being verdicts on the estimator — but
consistency is not evidence, and nothing here measures the estimator. The honest state is: the
deposit carries target-consistent structure; whether the model can see it is untested.

Do not cite the withdrawn verdict, and do not select a new corpus on the strength of it. The
modality-appropriate controls — guide-level replication, expression-dependence of effect size, and
the cutting-versus-non-cutting contrast against the AAVS1 arm — remain unrun.

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
