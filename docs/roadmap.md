# Cell-state roadmap

## Purpose

This project builds one thing: a system that computes a faithful and accurate representation of
hidden cellular state.

Everything else in this repository — contracts, schemas, manifests, admission machinery, benchmark
tooling, runtime infrastructure — exists only to make that representation trustworthy. None of it is
an end. An artifact that does not advance the representation, or the ability to recognize a faithful
one, is not on this roadmap.

[`architecture/full-buildout.md`](architecture/full-buildout.md) holds the model, data, and
evaluation design, including the formal estimand and the belief-subject semantics. This file defines
implementation order and graduation gates, and is the sole authority for both.

**If you are starting work:** the active phase is Phase 1 and the next action is item `Q5` of the
[implementation queue](#implementation-queue) — fit the observation model on `GSE274113`. `Q1` is
delivered; `Q2` is delivered in substance but incomplete against its own done-when; `Q3` repairs the
sufficiency verdict; `Q4` is decided, with a negative result, and owes a reviewed manifest. `Q7`, the
estimand freeze, is blocked on source selection and is deliberately not next. Everything before the
phase list is the standard that work is held to, not work.

**The state of the object:** the state-capability ledger stands at 0 of 10, and `CellStateBelief` is
constructed at exactly one site in `src/` — a synthetic reference. No belief has been emitted from
real cells. Every item delivered so far has built the apparatus that judges a representation; `Q5`
and `Q6` are the first that build the representation itself. See
[ADR 0019](adr/0019-build-the-representation-on-held-evidence.md).

## What "faithful" means here

The definition lives in [`concepts/predictive-sufficiency.md`](concepts/predictive-sufficiency.md)
and [`architecture/full-buildout.md`](architecture/full-buildout.md#validation-doctrine): a belief is
faithful for a query when it is predictively sufficient for that query's declared targets and its
predictive distributions are calibrated.

Two commitments are specific to this roadmap. First, **both tests must return a numeric verdict with
a sampling distribution.** A sufficiency gain or a coverage error reported without an interval,
grouped at the declared independent experimental unit, is not a verdict. Second, **selecting evidence
that can carry those tests is a first-class engineering task**, scheduled ahead of modeling, because
most public single-cell data cannot carry them at all.

## The state-capability ledger

Every scheduled item names the capabilities it advances. Progress is measured against this list and
nothing else. Each entry is worded so that a reader can determine objectively whether it is
satisfied.

| ID | Capability | Satisfied when |
| --- | --- | --- |
| S1 | Admissible pre-cutoff evidence | The frozen query admits a non-empty evidence set in the target modality before the inference cutoff, and at least one independent unit was observed both before and after it. Without this there is no hidden state to infer and no history for a belief to be sufficient against. |
| S2 | A posterior, not a point | The belief carries samples or particles whose spread is earned: on held-out units the posterior predictive is strictly wider than the point predictor's residual spread, and a posterior-predictive check on a supported assay statistic is not rejected. |
| S3 | At least two future horizons | The frozen query declares two or more horizons after the cutoff, and each is separately scored. One horizon cannot distinguish a state from a condition label. |
| S4 | Controlled evolution under `do(U)` | The forecast moves when and only when the intervention moves: a declared-null intervention leaves the predictive distribution unchanged to numerical tolerance, and a non-null one changes it by more than between-seed variation. |
| S5 | Assay-appropriate likelihoods | Varying a nuisance variable — library, capture, depth — at fixed biology changes the predicted observation and not the inferred state, within a predeclared bound on held-out units. |
| S6 | Calibration | Absolute marginal coverage error is within a predeclared threshold at every declared level, as an upper confidence bound grouped at the split unit. |
| S7 | An executable sufficiency verdict | The harness returns a history-information gain with a bootstrap interval grouped at the split unit, on genuinely held-out future evidence. This is the definition of faithful; without it the project cannot recognize its own success or failure. |
| S8 | Identifiability and abstention | On a deliberately out-of-support partition the abstention rate exceeds the in-support rate, and the risk-coverage curve is monotone: discarding the lowest-confidence decile does not increase held-out risk. |
| S9 | Superiority over the observational floor | Beats persistence, matched control, condition mean, nearest condition, pseudobulk GLM, and simple hierarchical and low-rank baselines, each individually, on the frozen proper scores, with a bootstrap interval grouped at the split unit that excludes zero. |
| S10 | External replication | Performance holds on an untouched external study with no test-time refitting. Faithfulness that does not transport is overfitting to one experiment. |

A query for which S1 or S3 is structurally unavailable cannot test S7. Such a query may be a useful
engineering exercise, but it is not a step toward the purpose and must not be scheduled as one.

## Program rules

These apply to every phase.

1. **The purpose test.** Every queue item states which ledger capabilities it advances. An item that
   advances none is not scheduled, however well engineered. Enforced by
   `tests/test_roadmap_queue_contract.py`.
2. **Authorization precedes implementation.** A commit that amends this roadmap may not also
   implement the work it authorizes. Authorization lands first, as its own merged change.
3. **Numbered items only.** The implementation queue is a single ordered list. Appended prose confers
   no authorization and creates no queue item. Queue IDs are ordinals within the current queue, not
   stable identifiers; documents outside this file cite artifacts and ADRs, never queue IDs. Enforced
   by `tests/test_roadmap_queue_contract.py`.
4. **Order changes require an ADR.** Any change to phase order, to the ledger, or to a graduation
   gate is a decision record, contemporaneous with the change. Rule 4 is satisfied for this
   reordering by [ADR 0013](adr/0013-state-first-roadmap-reordering.md), and for the Phase 1
   completion condition by [ADR 0014](adr/0014-phase-1-completion-condition.md). The serialized
   contracts the faithfulness tests report through are decided by
   [ADR 0015](adr/0015-faithfulness-reports-carry-their-sampling-distribution.md) and
   [ADR 0016](adr/0016-the-verdict-gates-on-the-interval.md). Retiring the proving-ground floor item,
   restating Phase 1's third graduation-gate bullet, and the queue renumbering that follows are
   decided by [ADR 0017](adr/0017-the-sufficiency-verdict-must-fail-closed.md), which also records
   the old-to-new queue mapping. Reordering the queue so the representation is built on evidence
   already held, ahead of freezing an estimand, is decided by
   [ADR 0019](adr/0019-build-the-representation-on-held-evidence.md), which carries its own mapping
   table.
5. **Every model declares its scope.** Query family, system boundary, assays, interventions,
   environments, horizons, and out-of-support and abstention behavior, in a registered model card
   alongside its data card.
6. **Public real evidence.** Biological training, validation, and test evidence comes only from
   publicly downloadable real cell-biology datasets with accession, version, license, checksum, and
   provenance. Synthetic data tests software and is never biological evidence.
7. **Claim-specific eligibility.** A dataset may support one estimand and not another. Destructive
   snapshots support population transitions, never reconstructed individual-cell trajectories.
8. **Splits follow independent experimental units** — well, plate, library, donor, clone, or study —
   never random cells, and are frozen before model selection.
9. **Graduation is scientific.** Running code, passing contracts, and low reconstruction error are
   not graduation criteria. Each phase graduates on evidence about biology.
10. **A negative verdict graduates.** A phase whose gate is a measurement passes that gate by
    producing the measurement with its interval. A sufficiency test that fails, reported honestly, is
    a result and satisfies the gate. The phase gate reads `evaluation_status` and the presence of an
    interval; it never reads `outcome`. A `FAILED` outcome blocks a runtime claim — correctly — but
    does not block the phase. A gate that can only be passed by success creates pressure to suppress
    failure.

## Subjects

The four belief subjects are defined in
[`architecture/full-buildout.md`](architecture/full-buildout.md#belief-subject-and-evidence-semantics)
and are not interchangeable; the available evidence determines which one a representation may
claim.

The first faithful representation this project produces will be a **population** belief. That is a
complete instance of the purpose, not a compromise: a population state is hidden, is inferred from
evidence, evolves under intervention, and is subject to both faithfulness tests. Individual-cell
claims require individual-cell evidence and follow later.

## Verticals

**Vertical A — cultured-cell population state under genetic and chemical intervention.** First
because public perturbation evidence is broad, controls are clean, and overlapping cell systems make
external replication possible.

**Vertical B — primary human T-cell state and recovery.** Activation, repeated stimulation,
exhaustion, withdrawal, rechallenge, and recovery, with molecular and functional targets. Richer
biology, sparser paired evidence, stricter transport assumptions. Deferred until Vertical A has
produced a sufficiency verdict.

---

## Phase 0 — scope, contracts, and evidence semantics

**Advances:** none directly; it is the precondition for measuring S1–S10 at all.

**Objective:** make invalid claims and leakage difficult before any biological model exists.

**Status: complete.**

Delivered: typed belief subjects with explicit destructive-evidence semantics; bounded query support
and a compiled, fingerprinted state specification that travels with every belief; perturbation
realization distinct from intended assignment; causal-status vocabulary; scientific-readiness and
typed abstention; a standalone measurement-decision contract; the dataset capability and claim
ledger, where eligibility is a property of the tuple (exact slice, exact claim, exact loss, exact
split unit, exact use policy) rather than of a dataset name; and machine-checked representability
ledgers that record what a dataset cannot support as durably as what it can.

**Gate passed:** a schema review can represent each candidate dataset without fabricating same-cell
links; leakage checks reject random-cell and shared-unit leakage; a query-support check explains why
a dataset is or is not eligible for every planned loss and metric.

## Phase 1 — make the faithfulness tests executable and measure the floor

**Advances:** S6, S7, S9.

**Objective:** the project currently cannot recognize a faithful representation, because the tests
that define faithfulness have no executable implementation and no callers outside tests. Nothing
downstream is meaningful until they do. This phase produces the repository's first scientific
numbers; the only prior measurements are failed-fit diagnostics from a retired candidate.

Deliverables:

- Executable metric implementations resolving every `metric_id` declared by the frozen sci-Plex3
  metric-suite specification: sample CRPS, energy score, marginal coverage error, marginal interval
  width, vehicle-relative pseudobulk effect error, and the equal-compound four-dose profile
  diagnostic. Each with a golden fixture and an independently derived numerical reference.
- The multiway clustered bootstrap those metrics declare, at the configuration the frozen suite
  specifies: resampling over the compound and plate dependence units jointly, 2,000 resamples, a
  0.95 interval. Every metric in that suite binds this estimator, and the sufficiency and calibration
  harnesses take their intervals from it rather than reimplementing it.
- At least one differential-expression-weighted metric and one rank-based metric in every metric
  suite frozen from this phase onward. Marginal error and all-gene correlation are maximized by
  predicting no change and must not stand alone. The sci-Plex3 suite frozen by
  [ADR 0008](adr/0008-sciplex3-k562-component-benchmark.md) is not retrofitted: its ten metrics
  across three families stay as frozen, and it serves only as the implementation proving ground and
  as the specification the implementations are conformance-tested against.
- A predictive-sufficiency harness: paired `M1`/`M2` predictors with declared equal capacity, the
  multiway bootstrap above grouped at the declared split unit, a confidence interval on the history
  information gain, and null calibration on a case where the true answer is known.
- A calibration harness reporting absolute coverage error with an upper confidence bound.
- The serialized-contract changes these require: a grouped-bootstrap interval on `SufficiencyReport`
  and a coverage-error upper bound on the calibration report, each with a schema-version decision,
  regenerated JSON Schemas, and round-trip tests.

Graduation gate:

- Every `metric_id` declared by the frozen sci-Plex3 metric-suite specification, and the uncertainty
  method they bind, resolves to an executable implementation with a golden fixture. A conformance
  test reads that specification and fails on any entry that does not resolve. Separately, no metric
  suite frozen under this roadmap contains a specification-only entry — a constraint that first binds
  at `Q5`, since no such suite exists yet.
- The sufficiency harness returns the correct verdict, with an interval, on two synthetic designs:
  one where the state is sufficient by construction and one where it is not.
- The sufficiency harness **refuses** a design on which the question is not being asked, rather than
  returning a verdict on it. A comparison whose history blocks are absent resolves to a not-evaluated
  report; it never resolves to `PASSED`. This replaces the proving-ground scoreboard that this bullet
  previously required. That scoreboard is unreachable by design, not by delay — the floor moved onto
  the Phase 2 estimand and the proving-ground query cannot satisfy S1, S3, or S7 — and a gate that
  cannot be passed, carried without a record of why, is exactly what rule 4 exists to prevent. See
  [ADR 0017](adr/0017-the-sufficiency-verdict-must-fail-closed.md). The floor itself is measured at
  `Q5`, with the same interval discipline.

## Phase 2 — freeze a state-bearing estimand

**Advances:** S1, S3, S9.

**Objective:** acquire an estimand in which a hidden state exists to be inferred and can be tested.

A query with no admissible pre-cutoff observation and a single horizon cannot satisfy S1 or S3. Under
such a query the sufficiency test is not merely unrun; it is inapplicable, because the raw history is
empty and `M2` is identical to `M1`. The persistence and temporal state-space baselines are likewise
inapplicable, which removes the two comparisons that would demonstrate the state carries information.
No amount of contract rigor repairs this. It is a property of the experiment.

Requirements for a state-bearing query:

- at least one admissible pre-cutoff observation in the target modality, observed on a unit that is
  also observed after the cutoff;
- at least two future horizons after the inference cutoff;
- a split unit that is a genuine independent replicate, present in sufficient number to bootstrap;
- enough distinct interventions that a held-out-intervention fold still retains a bootstrappable
  number of independent units; the fold design and its per-fold unit count are stated before a model
  exists;
- a randomized or otherwise identified intervention with matched controls;
- persistence and temporal state-space baselines both **applicable and mandatory**.

Primary candidate, pending the unit census in queue item `Q4`: `GSE274113` Perturb-Multiome. A
local-presence audit ([`data/evidence-inventory.md`](data/evidence-inventory.md)) records paired
same-cell RNA and ATAC, about 137,600 cells, twenty CRISPR targets including control, and days 7, 9,
11, and 14 with replicate libraries. **None of these figures is verified against source bytes.** The
number of libraries, the number of libraries per treatment arm, and whether any population unit was
sampled both at day 7 and at a later day are all unknown.

The intended design — day 7 as the pre-cutoff population observation; days 9, 11, and 14 as horizons;
the library as split and bootstrap unit; the subject declared a population because cells are
destructive with no cross-time identity — is a hypothesis that `Q4` must confirm before `Q5` freezes
it. If no unit spans the inference cutoff, the day-7 observation is condition-level and this source
fails S1. If the design is approximately one library per treatment arm, it falls under exclusion 2
below. In either case the estimand is redesigned or the source is rejected.

Explicitly not candidates for this estimand, with reasons recorded rather than rediscovered:

1. Same-cell longitudinal sources whose reviewed cohort is on the order of tens of cells: too few
   independent units to bootstrap a sufficiency gap.
2. Designs with approximately one library per treatment arm: intervention and library are not
   separable.
3. Single-timepoint destructive cross-sectional screens: no pre-cutoff observation and one horizon.

Deliverables:

- The reproducible data foundation, which is a prerequisite for admitting any source: immutable
  bronze storage for exact downloaded bytes and content hashes; silver event storage preserving raw
  counts, missingness, censoring, batch, controls, donor, library, well and plate, intervention
  assignment, realized engagement, environment, time, clone, and spatial relationships; gold
  query-specific examples generated from versioned transformations and frozen splits; ontology and
  unit mappings for genes, proteins, chemicals, cell types, assays, species, doses, and time; source
  adapters for AnnData/HDF5, matrix-market/GEO, Parquet/Zarr, FCS, and OME-Zarr as the selected
  sources demand; and checksum-pinned public-real golden slices small enough for CI.
- A frozen state-bearing query with declared subject, inference cutoff, admissible pre-cutoff
  evidence, intervention timing and realization evidence, target timing and units, horizons, causal
  class, and transport assumptions.
- Library-level partitions with a measured leakage audit whose overlap counts are computed from the
  membership arrays, not declared.
- A frozen metric suite drawn from Phase 1, including the sufficiency and coverage gates with numeric
  acceptance thresholds set before any model exists.
- A support envelope and bundle contract for the corresponding backend.

Graduation gate:

- Two independent machines reproduce the same golden slices and split memberships from manifests.
- License enforcement rejects an artifact whose declared use is incompatible with its dataset.
- The frozen query admits a non-empty pre-cutoff evidence set on units that span the cutoff, and at
  least two horizons.
- Persistence and temporal state-space appear in the mandatory baseline set as **applicable**. This
  is checkable today through `BaselineApplicabilityRule.applies_to`.
- The frozen query's support envelope declares the runtime operation that derives
  `sufficiency_evaluator` into `required_ports`; `derive_query_prerequisite_report` returns it with
  no entry in `invalid_dispositions`; and the bundle binds it `provided`. The disposition is derived
  and then satisfied, never typed into the bundle by hand. No bundle in this repository derives it
  today, because the only registered bundle is a component scaffold whose prerequisites come from the
  fixed component-prerequisite map. A runtime bundle whose support envelope declares
  `estimate_cell_state` does derive the port: `OPERATION_REQUIRED_PORTS[ESTIMATE_CELL_STATE]` contains
  it, and `_derive_runtime_target_prerequisites` emits every entry of that map as an
  operation-contract floor. Satisfying the gate therefore requires registering a runtime support
  envelope and bundle, not amending the derivation.
- The leakage audit reports computed overlap counts at the library level.

## Phase 3 — observation models and posterior inference

**Advances:** S2, S5.

**Objective:** make the belief mean "uncertainty about biology after accounting for measurement," not
a compressed assay vector. This is the first phase that computes a belief from biology.

Deliverables:

- Count-aware RNA and ATAC likelihoods, fitted on the frozen estimand's paired modalities.
- Explicit nuisance variables for library, capture, depth, and detection or censoring, inferred
  rather than corrected away.
- Shared and modality-private latent factors, with a held-out-modality test: predict ATAC from a
  belief conditioned on RNA alone, and the reverse.
- A posterior artifact format carrying samples or particles, weights, structured state factors, and a
  decomposition of measurement, biological, parameter, model, and transport uncertainty.
- Filtering for deployment and smoothing only where the evidence cutoff allows it.
- Biological runtime execution enabled through the public API, gated on resolved implementation,
  model, training, and validation bytes and on query-specific operation prerequisites.

Graduation gate:

- Likelihoods are calibrated on held-out technical and biological replicates.
- Posterior predictive checks reproduce supported assay statistics without erasing treatment effects.
- Missing-modality tests distinguish unavailable evidence from observed zero.
- A frozen belief cannot gain information from observations after its `as_of` time.
- A biological backend emits a `CellStateBelief` through the public API, with its support envelope,
  registered model and data cards, and abstention behavior enforced.

## Phase 4 — first state backend and the sufficiency verdict

**Advances:** S4, S6, S7, S8, S9. This is the phase the project exists to reach.

**Objective:** produce the first belief that is tested for faithfulness, and report the verdict.

Model guidance, derived from what has already been learned here. The exact supporting evidence is in
[ADR 0012](adr/0012-sciplex3-p1-trained-candidate.md) and `audits/item12_1a`.

- Begin from the simplest model that can carry a state: a hierarchical count model extended with a
  per-timepoint latent and a random effect at the split unit. Complexity is added only after a
  simpler model has been scored.
- Actions enter through features, not through a free parameter per observed action. A per-action free
  parameter cannot generalize to an unseen action in principle, and forecloses S10.
- Equal weighting of experimental units is a property of the objective, not of the reporting. Every
  term of the objective and of every derivative carries the same unit normalization.
- A context parameter must be accompanied by a measured effective-context diagnostic. A context
  dimension that collapses to an effective count near one is unidentified, and the diagnostic belongs
  in the model's own gates, not in a retired harness.

Deliverables:

- Hierarchical priors conditioned on supported genotype, culture, and environment.
- Controlled multi-horizon transition under `do(U)`, with intended versus realized perturbation
  handled distinctly and matched controls.
- Query-target decoders with calibrated abstention outside the support envelope.
- The complete mandatory baseline suite fitted and scored, including persistence and temporal
  state-space.
- An executed sufficiency verdict and an executed coverage report.

Graduation gate:

- Beats every mandatory baseline individually on the frozen proper scores, with intervals.
- Absolute marginal coverage error within the predeclared threshold at every declared level, as an
  upper confidence bound.
- Out-of-support risk decreases monotonically with abstention.
- Effect direction and dose or time ordering are accurate on held-out units.
- **A sufficiency verdict is reported with its interval.** A verdict of insufficiency graduates the
  phase and routes to a declared response: expand the state factors, or narrow the query. Either
  response requires an ADR.
- Results are described at the subject level the evidence supports, and nowhere else.

## Phase 5 — dynamics, events, and branching

**Advances:** S3, S4, S8.

**Objective:** represent evolution rather than horizon-to-horizon mappings.

Deliverables: controlled continuous dynamics with explicit process uncertainty; hazards and jump
kernels for division, death, differentiation, and commitment; particle or mixture inference around
bifurcations; population distribution dynamics for destructive time courses without invented identity
matching; clone-level inheritance and fate models where lineage was actually observed; individual-cell
filtering only where the same cell was actually measured twice.

Graduation gate: multi-horizon scores beat horizon-specific and no-change baselines at frozen temporal
cutoffs; event hazards and fate probabilities are calibrated on entirely held-out units; the
sufficiency margin is maintained or the state and query are revised; serialized results distinguish
individual, lineage, and population forecasts.

## Phase 6 — multimodal, environmental, and spatial context

**Advances:** S5, S8, S10.

**Objective:** model how context changes the state and its future consequences.

Deliverables: fusion across RNA, ATAC, protein, signaling, morphology, and metabolism through assay
likelihoods rather than concatenation; neighborhood graphs, environments, contacts, and interaction
terms where measured; explicit inter-study transport variables rather than globally corrected pooled
cells; mechanistic regulatory, signaling, and metabolic constraints as auditable, overridable soft
priors; transport diagnostics separating interpolation, transport, and unsupported extrapolation.

Graduation gate: contextual inputs improve future target proper scores, not reconstruction, on
held-out experiments; context effects replicate across independent studies; a claimed modality has
predeclared incremental value — removing an uninformative input does not worsen calibration, while
removing an informative one changes accuracy and uncertainty measurably; mechanistic constraints
improve external prediction or calibration and can be ablated without changing contract semantics.

## Phase 7 — translational T-cell vertical

**Advances:** S8, S10.

**Objective:** apply a validated state representation to primary human immune-cell state and function.

Deliverables: donor-aware priors and transport diagnostics; dynamics for activation, repeated
exposure, exhaustion, withdrawal, and recovery; molecular and supported functional decoders for
persistence, killing, cytokines, and metabolism; reconciliation of public T-cell datasets at the
condition and population level, never by fabricated same-cell pairing; donor-, intervention-,
context-, and whole-study-held-out evaluation.

Graduation gate: functional distributions are calibrated on held-out donors or studies; the model
improves over molecular-only and condition-mean baselines; transport limits and clinically
unsupported uses are mechanically enforced; no patient-level or clinical claim without a separately
designed validation program.

## Phase 8 — measurement value and intervention planning

**Advances:** S6, S8.

**Objective:** choose useful next measurements and interventions strictly inside validated support.
Planning is last because it consumes a calibrated representation and cannot substitute for one.

Deliverables: decision-relevant expected value of sample information over query targets, candidate
interventions, environments, and horizons, backed by a calibrated assay-outcome model, hypothetical
posterior updates, counterfactual replanning, and a declared decision utility; risk-aware intervention
simulation over full predictive distributions; constraint handling for dose, timing, cost, toxicity,
and feasibility; blinded or pseudo-prospective ranking benchmarks; fail-closed planner and measurement
policies that distinguish an unevaluated calculation from an unsupported component and from a
threshold-based abstention, with no numeric sentinels. Posterior covariance shrinkage is never
reported as decision value.

Graduation gate: measurement selection reduces calibrated uncertainty or decision regret on held-out
real experiments against cost-matched policies; intervention ranking improves top-k recovery and
regret over effect-size and nearest-neighbor policies; recommendations are limited to supported
systems and communicate uncertainty and causal status; prospective experimental validation remains
required before any operational biological claim.

---

## Current status

- **Phase 0:** complete.
- **Phase 1:** active. `Q1` is delivered: every `metric_id` the frozen sci-Plex3 suite declares
  resolves to an implementation, and the multiway clustered bootstrap those metrics bind is
  implemented and its coverage measured. `Q2` is **incomplete against its own done-when**. Both
  faithfulness tests execute, return an interval, and are enforced by their serialized contracts, but
  the clause requiring non-test callers is unmet: `evaluate_predictive_sufficiency` is called from no
  file in `src/`, `scripts/`, or `examples/`. An earlier revision of this section recorded `Q2`
  delivered and claimed that **the project can recognize a faithful representation**. That claim is
  withdrawn. The project can compute a history-information gain with an interval. It cannot yet
  recognize a faithful representation, because a design carrying no admissible pre-cutoff evidence
  returns `PASSED` with a degenerate interval — the strongest certificate the contract can express,
  earned by the absence of evidence — and the applicability judgment that would refuse it is
  delegated to a caller that does not exist. `Q3` repairs this; see
  [ADR 0017](adr/0017-the-sufficiency-verdict-must-fail-closed.md).

  What the project has not done at all is apply the tests to biology: no baseline has been scored
  against any other, no observational floor is measured, and no belief has been emitted by a
  biological model.

  Two results from `Q1` bear on later items. First, the frozen benchmark's untouched-test partition
  contains **four plates** and 95 compounds across 384 wells, computed from its membership arrays.
  Second, four plates is at the edge of what a bootstrap can support, and the measurement says so.
  Simulated on the *real* incidence of that partition at nominal 0.95, the plain percentile
  multiway bootstrap covers 0.908 when the plate dimension carries the variance; the scaling the
  implementation applies after [ADR 0016](adr/0016-the-verdict-gates-on-the-interval.md) brings it
  to 0.935, and **not to 0.95**. The earlier figures recorded here — 0.82 to 0.86 unscaled and
  about 0.96 scaled — were measured on a balanced generator at one variance configuration and did
  not survive contact with the partition's own compound-by-plate incidence. `Q3` and `Q5` should
  read the residual under-coverage as the property Phase 2 requires a state-bearing estimand *not*
  to have: enough independent units that its interval does not need rescuing.
- **Phase 2:** not started, and **has no candidate source.** `Q4` is complete with a negative
  result: `GSE274113`, this roadmap's only named Phase 2 candidate, is rejected for the
  state-bearing estimand by [ADR 0018](adr/0018-gse274113-rejected-for-the-state-estimand.md). It
  is retained under rule 7 for `Q6`. **Registered sources satisfying S1 — a unit observed both
  before and after an inference cutoff — remain zero**, and S1 is the binding constraint on the
  whole program: a query failing it cannot test S7, and S7 is the definition of faithful. Unlike
  every capability advanced so far, S1 cannot be advanced by writing code. It is a property of an
  experiment that already happened, so the next milestone is source selection against criteria
  fixed before a source is named.
- **Phases 3–8:** not started.
- **Biological backends registered:** none. No belief has been emitted by a biological model.
- **Benchmarks scientifically admitted:** none.

The cultured-cell population-response component frozen against a single-timepoint destructive drug
screen remains at `SCAFFOLD`, and its benchmark remains `COMPONENT_BENCHMARK`, not admitted. Its
estimand has no pre-cutoff observation and one horizon, so it cannot satisfy S1, S3, or S7, and it is
not a step toward the purpose. It is reclassified accordingly: a completed engineering exercise that
proved the data and split machinery, and the proving ground for the Phase 1 metric work.

Its partitions, manifests, split discipline, and definition-time leakage review — computed overlap
counts, all zero — are retained as infrastructure. That review records its own unassessed residue,
source-duplicate detection beyond exact row identifiers, and the admission-time leakage audit required
by ADR 0008 has not run and is not required, since the benchmark is not on the admission path.

In scope for continued use: the frozen partitions, membership arrays, leakage review, golden fixtures,
and the six fitted `p1` baseline states. Out of scope: the candidate lifecycle and the execution
control plane, which are suspended below.

## Suspended and off-path work

Work in this section is frozen, not deleted. It may be resumed only by an ADR that states which ledger
capability it advances.

1. **Protected-execution authorization control plane — suspended.** It governs the execution of a
   candidate for the component described above, which cannot emit a cell-state belief. It advances no
   ledger capability. No proposal is to be approved and no protected execution is to be dispatched;
   the workflow fails closed on dispatch.
2. **Container containment and reproducible runtime distribution — frozen as delivered.** The software
   is retained and no further work is scheduled. Resource limits on a training run are a dependency of
   a run, not a milestone.
3. **The rank-16 continuous admixture candidate family — retired for the state path.** It is a
   condition-level mean model with a free parameter per observed action, indexed at the well and not
   the cell. Its corrected objective mathematics, its equal-unit normalization constant, and its
   effective-context diagnostic are salvaged into the Phase 4 guidance above. The lifecycle scaffolding
   around it is not.

## Implementation queue

Each item names the ledger capabilities it advances. Items are worked in order. Queue IDs are prefixed
`Q` and do not continue the historical Item 1–12 sequence, whose numbering is bound into
content-addressed manifests and must not be reused. The queue extends through Phase 4; Phases 5–8 are
not yet decomposed into items and must not be started from prose.

1. **`Q1` — implement the metric suite and its interval estimator.** [S6, S9] Sample CRPS, energy
   score, marginal coverage error, marginal interval width, vehicle-relative pseudobulk effect error,
   and the equal-compound four-dose profile diagnostic — the six distinct computations behind the ten
   `metric_id` entries the frozen sci-Plex3 suite declares — plus the multiway clustered bootstrap
   every one of them binds. Each with a golden fixture and an independently derived numerical
   reference; the bootstrap's reference is a design whose interval is known analytically, never a
   recorded output of the implementation under test. Add a differential-expression-weighted metric and
   a rank-based metric for suites frozen from here on.
   *Files:* new `src/cellstate/evaluation/metrics.py` and
   `src/cellstate/evaluation/bootstrap.py`; port kinds at `src/cellstate/backends/contracts.py`;
   suite gate at `src/cellstate/data/benchmarks.py`.
   *Done when:* a conformance test reads
   `benchmarks/vertical-a/sciplex3-k562-24h-v1/support/metric-suite-spec.json` and resolves every
   `metric_id` it declares, and the uncertainty method they bind, to an executable implementation —
   failing on any entry that does not resolve — and `make check` is green. The frozen benchmark
   artifact's own bindings stay `specification_only`; re-versioning it is `Q3`'s decision, per
   [ADR 0014](adr/0014-phase-1-completion-condition.md).
2. **`Q2` — implement the sufficiency and calibration harnesses.** [S7, S6] Paired `M1`/`M2`
   predictors with declared equal capacity, `Q1`'s multiway bootstrap grouped at the split unit, a
   confidence interval on the history information gain, and null calibration. This requires adding a
   grouped-bootstrap interval to `SufficiencyReport` and a coverage-error upper bound to the
   calibration report; both are serialized-contract changes needing a schema-version decision,
   regenerated JSON Schemas, and round-trip tests before any harness is written.
   *Files:* `src/cellstate/evaluation/sufficiency.py`, `src/cellstate/evaluation/calibration.py`,
   `src/cellstate/domain/belief.py`, `schemas/`.
   *Done when:* both functions have non-test callers, both reports carry intervals, and the harness
   returns the correct verdict on a sufficient and an insufficient synthetic design.
3. **`Q3` — make the sufficiency verdict fail closed.** [S7] The harness returns `PASSED` with a
   degenerate interval `[0.0000, 0.0000]` when every unit's history block is absent, on a design
   where history demonstrably drives the target — so a query with no admissible pre-cutoff evidence
   receives the strongest certificate of sufficiency the contract can express. `sufficiency.py`
   delegates that applicability judgment to the query and its benchmark; no caller exists to make it,
   which is also why `Q2`'s "non-test callers" clause is unmet. Authorized by
   [ADR 0017](adr/0017-the-sufficiency-verdict-must-fail-closed.md): the module refuses an
   inapplicable design rather than passing it; units lacking a pre-cutoff observation are excluded
   from the paired comparison; the retained fraction becomes a required report field and a retained
   fraction of zero is a refusal. This item also supplies the missing caller — the evaluator adapter
   from a belief and a query to the paired loss vectors — which is `Q2`'s own unmet clause and not
   new scope. Deliberately out of scope, and scheduled as evidence rather than decided: whether the
   gain requires a noise-share decomposition, which the split-half reliability of the state and
   history blocks on real bytes settles.
   *Files:* `src/cellstate/evaluation/sufficiency.py`, `src/cellstate/domain/belief.py`, `schemas/`.
   *Done when:* a design whose history blocks are all absent returns a not-evaluated report rather
   than `PASSED`; the retained fraction is carried on `SufficiencyReport` and round-trips; and
   `evaluate_predictive_sufficiency` has a non-test caller in `src/`.
4. **`Q4` — review and manifest the state-bearing source.** [S1, S3] **Decided, not yet complete.**
   The review is finished and its result is negative; the reviewed manifest this item's done-when
   requires is not yet written, and until it is this item stays open — recording it as complete
   against an unmet done-when is the error [ADR 0017](adr/0017-the-sufficiency-verdict-must-fail-closed.md)
   corrected for `Q2`. The structural census and the provenance review are in
   [the representability ledger](data/representability/gse274113-perturb-multiome.md); the decision
   is [ADR 0018](adr/0018-gse274113-rejected-for-the-state-estimand.md). `GSE274113` is **rejected
   for the state-bearing estimand**: 0 of 14 libraries span a timepoint, the day 7 → day 14 effect
   is ρ = −0.026 against a permutation null of 0.025, and above the library the series supplies one
   independent parent culture, which neither the publication nor the deposited metadata can raise.
   The target-as-spanning-unit redesign is inadmissible under rule 8 and would itself need a rule 4
   amendment. The source is retained under rule 7 for `Q6`, where a measured library nuisance axis
   is S5's object rather than a defect, subject to resolved licence terms and established byte
   identity. A rejection recorded for cause is a legitimate outcome for this item — rule 10's
   principle, that a gate which is a measurement is passed by producing the measurement, applies to
   source review as much as to a verdict — but the outcome still has to be recorded in the artifact
   the item names. What remains is a reviewed manifest carrying the claim-specific split ADR 0018
   decides: ineligible for the state-bearing estimand, retained for `Q6`. That split is currently
   prose, and `ClaimAssessment` is the mechanism that makes it machine-checkable.
   *Done when:* unchanged — the census is in the reviewed manifest.
5. **`Q5` — fit the observation model and separate the nuisance axis.** [S5] On `GSE274113`, the
   source [ADR 0018](adr/0018-gse274113-rejected-for-the-state-estimand.md) rejected for the state
   estimand and retained under rule 7 for exactly this. Nuisance separation with a held-out-modality
   test in both directions across the paired RNA and ATAC readouts. The library is the nuisance axis
   and it is already measured: across-library spread at a fixed target reaches 0.53, 0.51, 0.89 and
   0.99 of across-target spread at days 7, 9, 11 and 14. S5 asks that varying a nuisance variable at
   fixed biology change the predicted observation and **not** the inferred state, within a
   predeclared bound on held-out units — so a nuisance axis of measured size is the material this
   test needs, not an obstacle to it. Requires resolved licence terms and established byte identity
   first; both are preconditions recorded by ADR 0018, not follow-ups.
   *Done when:* the held-out-modality test runs in both directions with an interval grouped at the
   library, and the predeclared bound is stated before the test is run.
6. **`Q6` — emit the first biological belief, and test the `do` operator's null half.** [S2, S4]
   Posterior inference through the public API, with support envelope, model and data cards, and
   abstention enforced. This is the item that makes the project's object exist: `CellStateBelief` is
   currently constructed at exactly one site in `src/`, a synthetic reference, so no belief has ever
   been emitted from real cells.
   S2 requires the posterior's spread to be **earned** — strictly wider than the point predictor's
   residual spread on held-out units, with a posterior-predictive check on a supported assay
   statistic. S4 is newly reachable here and was not reachable before: `GSE274113` carries `NT` as a
   **declared-null** intervention against 19 perturbed transcription factors, all fully crossed with
   library, which supplies both halves of S4's contrast — the null arm must leave the predictive
   distribution unchanged to numerical tolerance, and a non-null arm must change it by more than
   between-seed variation. The null half is currently asserted nowhere in this repository, so an
   inert or spuriously-moving operator would pass every test it has.
   *Done when:* a `CellStateBelief` is emitted from real cells through the public API; the S2 spread
   comparison and both halves of the S4 contrast are reported with intervals grouped at the library.
7. **`Q7` — freeze the state-bearing estimand and benchmark, and measure its floor.** [S1, S3, S9]
   **Blocked on source selection, not on `Q4`.** Registered sources satisfying S1 are zero, and no
   source may be named before the numeric selection criteria are fixed — setting them afterward
   invites fitting them to the candidate. Pre-cutoff observation, multiple horizons, library-level
   partitions, computed leakage audit, mandatory baselines including applicable persistence and
   temporal state-space, held-out-intervention fold design with per-fold unit counts, and numeric
   acceptance thresholds set before a model exists. The bundle contract derives
   `sufficiency_evaluator` into `required_ports` and binds it `provided`.
   Carries the observational floor absorbed from the retired proving-ground item, per
   [ADR 0017](adr/0017-the-sufficiency-verdict-must-fail-closed.md), with the same interval
   discipline; and inherits from that record the decision on `benchmark_version` `1.1.0` re-homed
   from [ADR 0014](adr/0014-phase-1-completion-condition.md), the **zero-panel doctrine** — a sample
   whose panel total is zero is a missing observation under an observation model, not an evaluable
   target, and no frozen policy may make a benchmark unevaluable on its own data — and the
   **reachable-threshold constraint**: no metric is frozen with an acceptance threshold not
   demonstrated reachable on real bytes.
8. **`Q8` — fit the first state backend and report the sufficiency verdict.** [S4, S6, S7, S8, S9]
   Simplest model that can carry a state; full baseline suite; coverage report; risk-coverage
   monotonicity; sufficiency verdict with its interval, whatever it says.

Items `Q1` through `Q3` require no new data, no data-access decision, and no model fit. They are the
shortest path to knowing whether the faithfulness tests work at all — and `Q3` exists because the
answer turned out to be *not yet*: a test that cannot refuse a design carrying no evidence has not
been shown to work, it has only been shown to run. `Q4` is the first item that needs bytes this
project does not already hold, and `Q5` the first that needs an access decision.
