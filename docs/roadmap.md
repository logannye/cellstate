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

- A standalone `recommend_next_measurement` operation whose request binds a parent belief,
  intervention objective, ordered candidate scenarios, candidate assays, timing, utility units, and
  assay, delay, and collection-effect penalties.
- Decision-relevant EVSI over query targets, candidate interventions, environments, and horizons,
  backed by a calibrated assay-outcome model, hypothetical posterior updates, counterfactual
  replanning, and declared decision utility.
- Risk-aware intervention simulation using full predictive distributions.
- Constraint handling for dose, timing, cost, toxicity, and experimental feasibility.
- Retrospective blinded/pseudo-prospective ranking benchmarks on hidden interventions.
- A fail-closed planner that rejects incompatible model versions, targets, units, horizons, and OOD
  candidates.
- A fail-closed measurement policy that distinguishes `NOT_EVALUATED` calculations from
  `UNSUPPORTED` components and threshold-based `ABSTAINED` decisions, all without numeric
  sentinels; covariance shrinkage alone never counts as EVSI.

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

**Active gate -- executable first population component:** the schema-v2 semantic spine and manifest
`0.3-experimental` eligibility ledger pass their contract and adversarial reviews. On 2026-08-09,
content-addressed proofs demonstrated that Replogle K562 destructive single-cell data can represent
a population snapshot and that a GSE141064 Live-seq slice can represent an individual same-cell
future-function relationship without fabricated linkage. Step 8 then froze the corrected
sci-Plex3 K562 24-hour component benchmark: 173,652 real nuclei, 1,536 independent well subjects,
16 protected plates, 752 exact compound-dose actions, four materialized partitions, authoritative
well-level cases, a training-derived 2,000-feature output schema, and prespecified metrics,
baselines, and paired acceptance rules. Its exact assessment and benchmark-use permission gates
pass. Its executable scoring, leakage, baseline, and performance gates remain deliberately
incomplete, so it is `COMPONENT_BENCHMARK`, not scientifically admitted. Step 9 approved an
exhaustive biological support-port map and bound a non-runnable direct population assay-response
scaffold to the exact artifacts. The component is intentionally separate from hidden-state
estimation and exposes none of the four public operations. Step 10 completed the trusted admission
boundary around contract v0.1 declarations: exact bytes, loaded interfaces, semantic results, and
query-derived prerequisites can now be authenticated and rebound without turning a serialized
receipt into runtime authority. That infrastructure does not supply a trained model, run a metric,
or pass a scientific gate. Item 11 has now completed the first frozen-data software path. Its
immutable, single-purpose `p1-train` loader close-reauthenticated the exact 2,526,631,614-byte
source, scanned all 94,785 training records across 768 wells, and froze content-addressed fitted-
state identities for all six probabilistic baselines. The loader does not parse held-out outcomes
and refuses `p2`, `p3`, and `p4` before protected raw-endpoint access unless future
lifecycle-bound grants exist.
These are software provenance artifacts, not benchmark results: no prediction campaign, metric,
comparison, or performance gate has run. Item 12 now defines the separate p1-only training and
verification boundary for a first candidate distribution model. Its original rank-16
Gamma--Poisson family with a separate `q`/capture term and 16 free factor shapes failed closed on
the exact real `p1` fit before artifact emission because variance reallocation drove a factor shape
to the rejected boundary. V2 removed `q`/capture, but its free pooled shape drifted toward the lower
guard and its outer fit remained unconverged. A fixed-shape v3 counterfactual still failed the
relative-ELBO and terminal factor-order gates. A bounded characterization cleared its activation
rank but rejected its tail-dominated independent lognormal plate nuisance. The audited v4
software-only design fixes `r_theta=0.1`, adds row-local inner equilibration, and proposes an
empirical whole-`rho`-row unseen-plate context. Its valid exact-reference, p1-only, nonissuing run
failed closed during initial inner equilibration, before an outer update or ELBO trace, and its
provisional `rho` reached near single-plate concentration. A subsequent source-code audit also found
that the v4 dose Newton objective omits the exact `94785/768` equal-well multiplier carried by the
corresponding terms in the tracked full ELBO, so its dose penalty is on the wrong relative scale.
V4 is retired and remains `NO-GO` and unissued. The sci-Plex3 component therefore remains a
non-executable `SCAFFOLD` and must not present a candidate design, provisional initialization
state, or unrun performance as a validated biological belief.

The next development work proceeds in this order:

1. **Completed 2026-08-09:** record this pause and keep the current query and candidate sources
   explicitly unfrozen.
2. **Completed 2026-08-09:** draft the complete Vertical A estimand, including belief subject and aggregation, inference
   cutoff, admissible pre-cutoff evidence, intervention timing and realization evidence, target
   timing and units, causal class, and transport assumptions.
3. **Completed 2026-08-09:** accept and expand ADR 0005, including an explicit schema-v2 and
   compatibility decision.
4. **Completed 2026-08-09:** implement v2 contracts for typed subjects and destructive evidence,
   bounded action/environment spaces, a compiled active-state specification,
   perturbation-realization belief, and query-scoped support, validity, causal-status, and
   abstention semantics. Identified effects additionally require typed query/scenario estimands,
   eligible designs, and content-addressed support and validation evidence.
5. **Completed 2026-08-09:** restore decision-oriented next-measurement selection as a separately
   parameterized fourth public operation. Its public contract binds at least two semantically
   distinct ordered candidate regimes, exact decision-set and causal evidence, timing, transport,
   and every collection cost. A supported numeric EVSI requires calibrated assay outcomes,
   hypothetical updating, counterfactual replanning, and decision utility. The non-biological
   reference returns `NOT_EVALUATED`, never a covariance proxy. This completes the software
   contract only; biological EVSI remains gated on real-data validation in Phase 7.
6. **Completed 2026-08-09:** allow repeated, independently scoped dataset claim assessments and
   separate metric/loss eligibility. Manifest `0.3-experimental` now binds canonical scopes, exact
   readouts, claim references, evidence sources, split safety, and layered legal permission; metric
   families whose benchmark semantics are not yet typed remain ineligible.
7. **Completed 2026-08-09:** add content-addressed dataset slices, interval-valued evidence clocks,
   and machine-checked reviewed representability ledgers. They validate bound declarations and
   reviewer attestations without resolving source bytes or replaying selectors. Replogle K562
   establishes destructive population-snapshot semantics; GSE141064 Live-seq establishes
   viability-preserving same-cell future-function semantics. Both keep legal use unauthorized and
   all unsupported scientific casts explicit.
8. **Completed 2026-08-09:** freeze Vertical A's first component query, benchmark, split semantics,
   metrics, mandatory baselines, and acceptance policy on corrected sci-Plex3 K562 data. The
   content-addressed artifact passes exact claim, loss, metric, split, source, and permission
   resolution while remaining non-admitted until executable performance gates run.
9. **Completed 2026-08-09:** approve an exhaustive biological model-bundle/support-port contract and
   bind the first exact population assay-response scaffold behind it. The contract binds
   content-addressed training, calibration, model-selection, validation, implementation, and
   benchmark declarations, but v0.1 deliberately cannot convert declarations into execution
   receipts. The sci-Plex3 scaffold remains at `SCAFFOLD`, exposes no estimator, evolution,
   planning, or measurement operation, and cannot emit either a response distribution or
   `CellStateBelief` until trusted verification and the component's scientific gates pass.
10. **Completed 2026-08-10:** implement the trusted, scope-bound admission boundary. Artifact
    verification incrementally hashes byte streams and requires closed-world one-to-one coverage.
    Real-data execution inputs come only from a capability-scoped, HMAC-authenticated workflow
    selection whose typed resolution artifacts are themselves byte-covered. Application-owned
    isolated loaders authenticate bounded observations of exact code and loaded objects; a separate
    verifier checks those objects against an application-owned interface registry. Typed result
    manifests bind exact evidence roles, partitions, cases, model and implementation scope, and
    supporting bytes, while readiness records semantic verification separately from scientific
    pass/fail. The deterministic prerequisite compiler derives and fingerprints conditional ports
    from the exact query, envelope, and target surface. External secrets and live interface objects
    remain nonserialized. Execution guards seal one reacquired code stream, verify and load that
    same immutable snapshot through the registry-owned trusted loader, repeat interface checks,
    and return operation-scoped nonserialized handles for only those checked objects. Missing trust, stale
    scope, forged attestations, failed results, or omitted/extra receipts remain blockers. The
    sci-Plex3 component remains at `SCAFFOLD` because it still has no trained artifact, executable
    evaluation, mandatory baseline results, or admitted benchmark.
11. **Completed 2026-08-10:** implement the first frozen-data software path without opening a
    held-out lifecycle stage. The single-purpose loader authenticates the exact sci-Plex3 H5AD and
    checked-in `p1` closure, then yields immutable sparse raw-count batches on the ordered 2,000-
    feature panel. Public frozen design metadata precommits exact split-membership arrays,
    record/well/plate identities, and outcome-free prediction cases, but protected `p2` calibration
    endpoint values, `p3` selection endpoint values, and `p4` source outcomes and scoring authority
    remain hard sealed pending future grants bound respectively to
    `TRAINED_CANDIDATE`,
    `CALIBRATED_CANDIDATE`, and `MODEL_SELECTED_FROZEN` lifecycle states. Six mandatory
    probabilistic algorithms fit only from `p1`: matched-vehicle resampling, exact-condition
    empirical resampling, exact-condition negative binomial, hierarchical well negative binomial,
    low-rank compound-dose response, and nearest-supported-dose resampling. Every algorithm has an
    explicit no-action path; matched vehicles come only from `p1` source plates, and nearest dose
    excludes the exact requested dose. Frozen prediction semantics are 512 samples per case and
    seed, seeds 0 through 4, and NumPy `PCG64DXSM`. Fitted-state manifests and prediction shards are
    designed for content addressing and streaming rather than a dense campaign-wide materialization.
    The exact source was authenticated before and after use; all 94,785 `p1` records across 768
    wells were scanned; seven genuine zero-panel rows were retained; and six fitted-state identities
    were materialized with no held-out access or lifecycle authority. None of this is baseline
    performance, scientific admission, or a public cell-state runtime.
12. **In progress 2026-08-10:** establish the first honest p1-trained candidate boundary. The
    immutable pre-fit plan binds the exact typed `p1` role, count stream, scan, assembly, design,
    feature, action, target, candidate specification, trainer/factory code, output schema, and
    complete single-thread Linux `x86_64` runtime: CPython 3.11.15, NumPy 2.4.6, SciPy 1.17.1, and
    `scipy-openblas` 0.3.31.188.0. Runtime-only HMAC source-selection and fit-semantic receipts,
    exact byte resolution, query-prerequisite verification, and an application-owned loaded-
    interface registry are all required before `TRAINED_CANDIDATE` can be derived; deterministic
    files remain evidence rather than authority. The first rank-16 Gamma--Poisson design with a
    separate `q`/capture term and 16 free shapes failed closed at the Gamma-shape boundary on the
    exact real `p1` data and emitted no model. V2 removed `q`/capture, but its free pooled shape
    crossed below `0.1`, reached approximately `0.073524`, and the outer trajectory remained
    unconverged at pass 50. V3 fixed `r_theta=0.1`, but its nonissuing counterfactual still failed
    relative-ELBO convergence and terminal factor-order stability. Its follow-up characterization showed a raw
    activation-rank ratio `8.472584996122713e-7`, about 56.85 times the strict gate, and isolated the
    original rank rejection to a 12-decimal quantization half-boundary. It also showed that the
    independent mean-one lognormal unseen-plate model was tail-dominated at
    `sigma_plate=5.66126548675`.

    The audited incompatible v4 software design retains 16 continuous Gamma--Poisson factors,
    fixes `r_theta=0.1`, adds deterministic row-local inner equilibration, and proposes replacing
    the lognormal plate draw with uniform selection of one complete observed `p1` `rho` row.
    Its calibration declaration precommits to `tau_j=exp(j/20)` for integer `j=-20,...,6`, with
    shape `0.1/tau_j^2` and factorwise-renormalized `rho^tau_j`; it does not read `p2` or choose a
    value.

    The first v4 launch produced report
    `4677fc8ef1a458bf3616abc507250572c2da7a8d53c1c8a7a03d4b097f3d4877`, but its audit hook
    rejected h5py's nonpersistent `/dev/null` probe before fitting. That attempt is infrastructure-
    invalid, contains no fit, and must not be interpreted as science. The replacement execution was
    an exact-reference, p1-only, nonissuing run. Its report,
    `66e9debc1a402e7aa68cbc934f7c5f641529eea3187ec15606364c912af8faa8`, passed integrity and
    post-hoc resource acceptance checks and kept all provisional tensors finite. It failed initial,
    untraced inner
    equilibration at sweep 50 with passing streak 0, `Rshape=0.24714465227035654`, and
    `Relog=3.750385840630546`, before any outer update or ELBO trace. The two residuals are
    respectively 24,714,465 and 375,038,584 times the `1e-8` tolerance. The provisional loading-
    rank ratio `0.101239623839`, activation-rank ratio `0.001249342162`, and minimum contribution
    share `0.001237124212` describe initialization only and are not evidence that rescues the failed
    fit. The provisional `rho` maximum was `7.999995552807402`; because each factor column sums to
    eight, at least one factor places `99.9999444101%` of its context mass on one plate and has an
    effective context count of approximately `1.0000011118`. This is a candidate-family risk, not a
    fitted-model pathology. The empirical whole-row context must be reexamined and is not a
    validated unseen-plate model.

    V4 is `NO-GO`. It issued no model, plan, observation, training evidence, materialization, or
    lifecycle result. Item 12 remains in progress and the component remains `SCAFFOLD`. Throughout
    this work protected `p2`, `p3`, and `p4` raw H5AD/UMI endpoint values and lifecycle scoring
    authority remain sealed, and calibration, selection, performance, admission, component
    execution, and every public cell-state runtime remain false.

**Item 12.1a — local-map and plate-context characterization software freeze** is complete. Its
historical evidence is now canonical under
[`audits/item12_1a`](https://github.com/logannye/cellstate/tree/main/audits/item12_1a). The frozen
harness SHA-256 is
`f4e6b76847bd926952995d66233389768f091135699fb60a38d7d9762bb03ff1`; its test SHA-256 is
`8989618e259fb4aed0e0798bc010e40092c45e6bd30234bb3a7b534cdc562903`.
The exact parent driver (`795c5929...`) and both historical reports (`4677fc8e...` and
`66e9debc...`) are stored beside them. Twenty-six focused tests, Ruff check/format, compilation, and
an independent SHA-bound math/containment audit reported no P0/P1 finding within the frozen
local-map and containment scope. That audit did not validate the outer dose objective. The contract
fixes 50 production maps
`A0 -> ... -> A50`, exactly one diagnostic-only `A51` lookahead, per-sweep `Rshape` and `Relog`,
synchronized local objectives, one- and two-step distances and update cosine, deterministic worst
row/factor/count summaries, shape/rate/mass invariants, per-sweep analytic 16-by-16 Jacobian
spectral radii, exact replay equality, state/allocation digests, and all 16 bounded `rho`
effective-context counts and maximum shares. An objective decrease routes to an implementation
fix. An intact objective plus a two-cycle or noncontractive map routes to a versioned v5
safeguarded local solver. Near-zero prior/context behavior or effective context near one routes to
v5 plate regularization or neutralization. The contract does not prefreeze damping or shrinkage.

**Item 12.1b — retired before execution.** The planned exact-reference real-`p1` replay of the v4
harness is no longer the next step and must not run. Item 12.1a opened no source. The later
source-code audit established that v4's dose-block objective minimizes an unweighted per-well Gamma
term plus the global dose penalty, while the tracked full ELBO multiplies that Gamma term by
`94785/768`. Because this changes the intended objective and can invalidate outer-step
nondecrease, another real-source characterization of v4 would spend authorization on a candidate
already known to be mathematically inconsistent. No Item 12.1b report, artifact, or lifecycle
evidence exists.

**Item 12.2 — source-free v5 objective, M-step, and execution-containment redesign** is the next
milestone. Before any new real-`p1` authorization, v5 must define one objective whose equal-well
normalization, dose penalty, and action/context M-step agree exactly; pass finite-difference
full-ELBO gradient tests and fixed-`q` M-step nondecrease tests; and run behind hard process or
container wall-clock and memory limits rather than relying on post-hoc elapsed/RSS acceptance
checks. The source-free redesign must optimize `alpha`, constrained `log-rho`, and dose effects
against an independently computed fixed-`q` full ELBO. Feasible-coordinate gradient tests under the
arithmetic-mean-one `rho` gauge, treated-well adversarial fixtures, accepted-substep and complete-
block nondecrease, and Hessian finite differences if Newton remains are required. Any surrogate or
block schedule must nondecrease the same canonical full ELBO and share its stationary points.

Sampling must either be exactly conditioned on a positive panel, or the API must freeze a maximum
request size and request-level failure budget over every admitted action, context, and calibration
`tau`; a target-only `supports()` result is insufficient because the 33-attempt failure compounds
with sample count. Poisson-rate/RNG overflow must be included in the same support decision. The
builder must publish each candidate in an immutable generation and switch one atomic pointer only
after the complete generation verifies. A forced process termination mid-publication must recover
automatically, leave no stale lock, and let readers observe only the complete old or complete new
generation.

Parent-enforced wall and memory limits must cover the whole source-touching process tree from before
source open through close and verification. The exact limit policy and runtime-image digest must be
frozen; timeout, cgroup/container OOM, descendant cleanup, no-canonical-publication, and next-run
recovery tests are mandatory. This milestone does not open `p1`, change a fitted artifact, or
authorize `p2`. Only after it passes review may a separately authorized, version-bound nonissuing v5
real-`p1` execution be proposed.

Only a future successful trusted verification may move the component to `TRAINED_CANDIDATE`. A
future one-use grant may open `p2` for calibration only after that exact state; a separate grant may
open `p3` for model selection and freezing only
after calibration; and `p4` source outcomes and locked scoring remain untouched until a locked
evaluator receives an exact `MODEL_SELECTED_FROZEN` candidate. Public frozen p4 design metadata is
not evaluation authority. Passing that component lifecycle would authorize only the exact
assay-response API; a hidden-state backend still requires its own observation/prior, inference,
dynamics, sufficiency, identifiability, and operation-specific evidence.

This sequence corrects the semantic spine before adapters or models depend on it. It preserves the
existing manifest work while ensuring the first implementation cannot satisfy software contracts by
violating the scientific estimand.
