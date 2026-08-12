# Vertical A scientific estimand

- **Status:** Draft; deliberately unfrozen
- **Draft version:** `0.1-draft`
- **Last reviewed:** 2026-08-12
- **Owning roadmap gate:** Phase 2, freeze a state-bearing estimand

This document defines the scientific question that the first cultured-cell population backend is
intended to answer. It is complete enough to drive schema design and dataset representability
reviews, but it is **not** a benchmark, support claim, or permission to train a biological model.
The query, systems, horizons, targets, and thresholds become frozen only through the Phase 2
requirements in [`../roadmap.md`](../roadmap.md), which supersede the freeze conditions this
document was written under. The K562 and Live-seq representability proofs are complete and are no
longer the gating condition.

ADR 0008 freezes a deliberately narrower sci-Plex3 K562 component benchmark: static experimental
context plus an assigned compound-dose predicts a 24-hour captured-nucleus assay distribution on a
training-derived feature panel. That component has no pretreatment molecular observation and does
not satisfy the complete estimand below. It exists to exercise the first population adapter and
response-model boundary without redefining context lookup as a molecular cell-state belief.

## Intended question

For a declared cultured-cell population at an inference cutoff, given only admissible evidence at or
before that cutoff, estimate the query-relevant population belief and use it to predict calibrated
future molecular or functional distributions under a bounded intervention and environment.

The first candidate slice was a K562 population-response task, which has no admissible pre-cutoff
evidence and one horizon and therefore cannot carry the sufficiency test.
[ADR 0013](../adr/0013-state-first-roadmap-reordering.md) supersedes it with a state-bearing
estimand; the cell system, source, and admission of that estimand are settled by the roadmap's
source review, not here. A549 and additional modalities are transport or expansion tasks until
separately admitted. The system must not reduce the task to a
lookup of `endpoint ~ intervention + cell line`: the inferred belief must condition on a declared
pre-cutoff evidence history, propagate uncertainty, and pass the state-only versus
state-plus-history sufficiency test.

## Belief subject and estimand unit

The primary subject is a **population distribution**, not an individually tracked cell:

- The estimand represents cells belonging to one declared culture population or experimental unit
  under a defined protocol.
- Destructively assayed cells are samples from that population. They do not acquire longitudinal
  identity merely because their barcodes, expression profiles, treatment labels, or collection
  times resemble one another.
- A well, aliquot, sibling culture, matched control, or condition aggregate may provide evidence
  only through an explicit observed relationship to the target population.
- The sampling unit, randomization unit, biological replicate, belief subject, prediction target,
  and split unit are separate typed concepts. Their identifiers and relationships must be retained.
- Individual-cell predictions are outside this first slice unless direct tracking or
  viability-preserving repeated measurement establishes identity.

The target aggregation for molecular outputs is the distribution of cells within the declared
population, with the experimental unit retained for replication and uncertainty. Pseudobulk
summaries may be auxiliary targets or baselines, but may not replace the cell-distribution target
when cell-level source measurements exist.

## Time origin, cutoff, and evidence window

The query uses experiment-relative time and declares a cutoff `t`.

For the initial pre-intervention task:

- `t = 0` is immediately before the candidate intervention begins.
- Every observation used to estimate the belief must have an evidence interval ending at or before
  `t`.
- Static context, culture history, prior interventions, media/environment changes, and washouts
  before `t` are part of the conditioning history when recorded.
- Baseline destructive measurements from a matched population are population evidence, not direct
  repeated measurements of future endpoint cells.
- Events, outcomes, assay metadata, or normalization statistics derived after `t` are forbidden
  inference inputs.

Later early-response variants may place `t > 0` after an intervention and assimilate early
measurements. Each such variant is a distinct query with its own cutoff, eligible evidence, support,
and leakage audit; it is not an implicit extension of the pre-intervention task.

The frozen benchmark will declare at least two named horizons after the cutoff, will include only
horizons directly supported by admitted source records, and will never infer temporal coverage from
filenames or adjacent conditions. No horizon range is asserted in advance of source admission; the
earlier 3--72 hour range was written for a superseded single-endpoint candidate.

## Formal belief and forecast

For target population `p`, query `Q`, and cutoff `t`, estimate

```text
B^Q_{p,t} =
P(X^Q_{p,t}, Theta_p, R_{p,<=t}, Xi_p
  | H^{eligible}_{p,<=t}, C_p, Q)
```

where:

- `X^Q` contains only the dynamic factors compiled for the query;
- `Theta` contains stable cell-system, genotype, culture, and replicate effects;
- `R` is uncertain realized exposure or target engagement, distinct from assignment;
- `Xi` contains assay and other measurement nuisance variables;
- `H^eligible` is the causally ordered, cutoff-safe evidence graph; and
- `C` contains the declared population, experimental, and environmental context.

For each named horizon `h` and admissible scenario, return

```text
P(Z_{p,t+h} | B^Q_{p,t}, do(U_{p,t:t+h}), E_{p,t:t+h}, Q)
```

with an explicit causal-evidence label. The notation `do(U)` expresses the requested intervention;
it does not promote an associative model into an identified causal model.

## Admissible conditioning evidence

Every frozen query must enumerate its evidence channels. Candidate channels for the first slice are:

- pre-cutoff raw or minimally processed RNA counts with assay/library nuisance metadata;
- a declared baseline or matched-control population and its linkage to the target population;
- cell line, genotype, construct, clone or culture provenance when actually observed;
- replicate, plate, well, sample, batch, and randomization identifiers;
- intended past interventions plus measured/inferred realization evidence;
- culture medium, temperature, oxygen, serum, density, stimulation, and other recorded environment;
- explicit completeness and missingness for unrecorded histories.

No query may silently substitute:

- endpoint observations for baseline evidence;
- condition means computed using test examples;
- destructive cells at another time for the same living cell;
- inferred nearest-neighbor or optimal-transport couplings for observed identity;
- a globally batch-corrected embedding for raw/minimally processed evidence; or
- absent metadata for a confirmed biological zero or no-event history.

## Intervention domain

The first frozen slice will choose one bounded intervention family after source admission. Candidate
families are CRISPR perturbations and small molecules; they remain separate capability claims.

Each supported action must declare:

- intervention kind, canonical identity, target, and mechanism when known;
- assignment/randomization unit and matched control;
- dose value and units or genetic construct multiplicity;
- route/delivery method;
- start, stop or duration, schedule, washout, and reversibility semantics;
- whether combinations are permitted and, if so, their maximum order and interaction support;
- realization-evidence requirements and uncertainty; and
- interpolation versus extrapolation status.

Matching only an intervention name or dose unit is insufficient. Values outside the declared range,
unsupported combinations, unknown delivery semantics, or a future action carrying retrospectively
measured realization evidence must fail support checks.

## Environment domain

The query must bind the culture environment over the entire conditioning and forecast interval.
Each controllable or conditioning variable declares units, allowed values or ranges, temporal
persistence, and missing-history behavior. A recorded fixed protocol may define a singleton domain;
unrecorded culture conditions do not.

The initial slice does not claim support for spatial gradients, co-culture, mechanical inputs, or
neighborhood interventions unless an admitted source and model support them explicitly. An isolated
intracellular system boundary may still condition on exogenous environment; the boundary defines
what is jointly represented in state, not whether external conditions exist.

## Intervention realization

Assignment and realization are separate random variables. Realization may include exposure,
delivery, edit status, guide activity, target engagement, toxicity, construct expression, and
off-target effects.

The query declares which realization evidence is required. If no suitable evidence exists, the
belief must retain realization uncertainty, downgrade causal status as appropriate, and abstain when
that uncertainty exceeds the query's decision threshold. A nominal dose or guide label is not full
engagement. Every realization-evidence observation, including evidence labeled inferred or unknown,
must begin at or after the intervention; pre-action measurements may inform a prior but cannot count
as evidence of the realized perturbation.

## Targets, units, and aggregation

The primary candidate target is a calibrated future RNA distribution for a predeclared gene set at
each named horizon. Before freezing, the benchmark must decide whether each output denotes:

- future assay counts conditional on a declared assay model;
- a latent abundance on a declared measurement scale; or
- a population summary derived by a versioned transformation.

Each target declares an ontology identifier, units, cell/population aggregation, experimental unit,
measurement protocol, censoring/missingness semantics, and horizon. A target is unavailable rather
than numerically fabricated when its decoder or unit bridge is absent.

Phosphosignaling, morphology, proliferation, and viability are secondary target families. They may
be added only through independently scoped evidence and transport assumptions. Measurements from
different studies or subjects must never be presented as same-cell paired molecular and functional
outcomes.

## Causal and transport status

Every prediction is labeled as exactly one of:

- predictive association;
- identified population intervention effect;
- transported effect under enumerated assumptions;
- mechanistic extrapolation; or
- unsupported.

An identified or transported population effect is not a free-text annotation. It must cover the
exact requested target and horizon, preserve the target aggregation and experimental unit, name the
declared intervention/environment contrast and comparator, use a typed randomized or
quasi-experimental design, and cite a content-addressed validation claim within the model's exact
support envelope. A perturbation label or arbitrary history event is never identification evidence.

An identified population effect requires an eligible assignment/control design, retained
experimental units, cutoff-safe outcomes, and an explicit identification basis. Cross-study or
cross-cell-line use requires a source and target domain, overlap variables, exchangeability and
measurement-bridge assumptions, and additional transport uncertainty. A549 evidence cannot silently
validate a K562 claim, and a bulk or pseudobulk target cannot silently validate a cell-distribution
claim.

## Query-scoped support and abstention

The frozen query will set numerical or categorical acceptance gates for:

- supported subject and aggregation;
- cell system, genotype, culture, and assay domain;
- intervention identity, mechanism, dose, schedule, combination, and realization evidence;
- environment and history completeness;
- target, unit, and horizon coverage;
- predictive sufficiency;
- calibration and required precision;
- OOD score and coverage;
- counterfactual/transport uncertainty; and
- decision-critical identifiability.

Producing a distribution is not evidence that these gates passed. Estimation may return a belief
with an explicit invalid or unsupported result; forecasting must label unsupported targets; planning
must abstain unless its readiness policy passes.

## Validation estimand

Validation is grouped by the true independent experimental unit. The benchmark will predeclare
train, calibration, and untouched test partitions before model selection. Candidate generalization
axes include held-out wells/replicates, perturbations, doses, horizons, and an external accession
where comparable evidence exists.

Primary metrics operate on returned distributions and include proper predictive scores,
treatment-minus-control effect error, population-distribution distance, calibration coverage and
sharpness, OOD risk--coverage, and the belief-only versus belief-plus-history sufficiency gap.
Every metric suite frozen against this document carries at least one differential-expression-weighted
and one rank-based metric; marginal error and all-gene correlation are maximized by predicting no
change and never stand alone. Every verdict is reported with a bootstrap interval grouped at the
declared independent experimental unit; a gain or a coverage error without an interval is not a
verdict.

Mandatory baselines are ledger entry `S9` in [`../roadmap.md`](../roadmap.md), plus a temporal
state-space or low-rank model. Persistence and temporal state-space are not conditional here: a
state-bearing estimand has the pre-cutoff observation and the second horizon that make them
applicable, and they are the two comparisons that would show that the state carries information.

No random-cell split, reconstruction score, cluster coherence, or embedding visualization can
graduate this query.

## Freeze blockers

This draft must remain unfrozen until all of the following are complete:

1. **Satisfied 2026-08-09:** [ADR 0005](../adr/0005-belief-subject-semantics.md) accepts the
   belief-subject and schema-v2 decision.
2. **Satisfied in Phase 0:** the v2 query encodes the bounded domains and acceptance gates above
   through `NumericDomain`, `CategoricalDomain`, `ScalarRange`, `IntegerRange`, `ScheduleDomain`,
   and `AcceptanceThresholds` in `src/cellstate/domain/query.py`.
3. **Satisfied in Phase 0:** `CompiledStateSpecification` carries `active_factors`,
   `excluded_factors`, `intervention_realization_dimensions`, `observation_nuisance_dimensions`,
   and `context_modulator_dimensions`.
4. **Satisfied in Phase 0:** forecasts carry target aggregation, causal status through
   `CausalSupportReport`, scientific support, and typed abstention.
5. **Satisfied 2026-08-09:** dataset manifest `0.3-experimental` allows independently scoped claim,
   loss, and metric eligibility. Concrete benchmark metric definitions and split membership remain
   blocker 8 rather than being inferred from this structural ledger.
6. **Satisfied 2026-08-09:** a reviewed Replogle K562 destructive-sampling manifest and
   machine-checked reviewed proof ledger establish population-snapshot representability without
   inventing same-cell linkage. The structural verifier does not resolve source bytes or replay a
   selector.
7. **Satisfied 2026-08-09:** a reviewed GSE141064 Live-seq 17-cell functional-recorder slice and
   machine-checked reviewed proof ledger establish viability-preserving same-cell future-function
   linkage without claiming causal intervention identification, repeated transcriptomic state, or
   runtime selector replay.
8. Exact source-backed horizons, targets, intervention ranges, thresholds, splits, and baselines are
   approved in a versioned frozen benchmark manifest.
9. The metrics this estimand will be scored on have executable implementations, and the sufficiency
   and calibration harnesses return a verdict with an interval grouped at the independent
   experimental unit. Today no metric implementation exists in any frozen suite and neither harness
   has a caller outside tests, so a benchmark frozen against this document could not yet be run.

Until then, this document is a design input—not a scientific support claim.
