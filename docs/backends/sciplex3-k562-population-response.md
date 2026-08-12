# sci-Plex3 K562 24-hour component

## What this component is

Infrastructure, not a step toward a cell-state representation.
[ADR 0013](../adr/0013-state-first-roadmap-reordering.md) reclassified it off the state path: a
completed engineering exercise that proved the repository's data, split, and admission machinery,
and the proving ground for the Phase 1 metric implementations. Its frozen partitions, membership
arrays, manifests, split discipline, definition-time leakage review, golden fixtures, and six fitted
`p1` baseline states remain in scope. Its candidate lifecycle does not.

Nothing in that reclassification asserts a defect in the work.
[ADR 0008](../adr/0008-sciplex3-k562-component-benchmark.md) and
[ADR 0009](../adr/0009-population-response-component-boundary.md) stand as written.

`SciPlex3K562PopulationAssayResponseScaffold` (`src/cellstate/backends/sciplex3_k562.py`) is a
non-runnable `SCAFFOLD`. It implements the separate typed `PopulationAssayResponseModel` component
port — never one of the four public cell-state operations — only to resolve exact artifacts,
preflight one frozen task, and reject execution. It has no accepted model weights, training run,
calibration evidence, model-selection freeze, locked evaluation, or admitted benchmark performance.
`sample_response` is typed `Never` and always raises. The component exposes no `estimate`, `evolve`,
planning, measurement, or condition-response lookup entry point.

## The frozen task

Its only eventual computation is:

```text
P(raw nonnegative integer UMI vector on the exact ordered 2,000-feature panel
  in recovered nuclei at 24 hours
  | static K562 source-well/plate context, intended compound-dose assignment or no-action control)
```

Static context is not a prior over hidden state. The destructive 24-hour RNA assay and matched
vehicle wells are future target and comparator observations, never pre-cutoff evidence. The forecast
label is `predictive_association`, intervention realization remains `unknown`, and the component
cannot claim viability, survival, target engagement, individual-cell dynamics, transport, or a
current-state belief.

## Why it cannot carry a state verdict

The frozen query declares no admissible pre-cutoff observation and one horizon. It therefore fails
capabilities S1 and S3 of the [roadmap](../roadmap.md) ledger and cannot test S7. Under an empty
pre-cutoff history the predictive-sufficiency comparison is not merely unrun; it is inapplicable,
because `M1` and `M2` receive the same inputs.

Three checked-in artifacts record this, rather than leaving it to prose:

- the bundle contract binds `sufficiency_evaluator` as `unsupported`, with the rationale "One
  context-to-endpoint experiment cannot establish hidden-state sufficiency";
- the frozen baseline suite declares `persistence` inapplicable — "requires a pre-cutoff
  target-modality observation; frozen query has none" — and `temporal-state-space` inapplicable —
  "also requires at least two future horizons; frozen query has one". Those are exactly the two
  comparisons that would show that a state carries information;
- the benchmark artifact's admission status is `component_benchmark`, and one recorded reason is
  that this endpoint-only component "has no pre-intervention molecular measurement, same-cell
  linkage, viability target, second horizon, or external-study transport evidence."

## Frozen artifacts and identities

Five artifacts are re-read and rehashed on every preflight:

| Artifact | Path | SHA-256 |
| --- | --- | --- |
| Reviewed dataset manifest | `data_manifests/reviewed/sciplex3-k562-24h.json` | `6248e63237a4c0c7ae53538666a1294cf1108569792eb54702ec15f439d9cb31` |
| StateQuery | `benchmarks/vertical-a/sciplex3-k562-24h-v1/state-query.json` | `d0fa67f31a8ea1d7b2e8839dfe7629fd6f359ea7eed4f6d336e2cd1d8813971e` |
| Benchmark artifact | `benchmarks/vertical-a/sciplex3-k562-24h-v1/benchmark-artifact.json` | `97bfb8f00f9efd93ad19635ce1a843a126c3c1b23ae6002102353c5e3bded76e` |
| Component support envelope | `backends/vertical-a/sciplex3-k562-24h-v1/support-envelope.json` | `17aa440c14b40981f97358119085f44b2ffeb9bed75ba322114ffe2c1c53dd9f` |
| Component bundle contract | `backends/vertical-a/sciplex3-k562-24h-v1/bundle-contract.json` | `69eddd15eb87b167ea0ef484d54234f6af5ebd6b353ad4aa72a96c2dca3f6343` |

The `p1` execution surface adds three more:

| Artifact | Path | SHA-256 |
| --- | --- | --- |
| Item 11 `p1` loader contract | `benchmarks/vertical-a/sciplex3-k562-24h-v1/support/p1-loader-contract.json` | `3de5be54b60ba1403995ba79d122ee8232218be5c027da1bf530cb610ae80f90` |
| Baseline golden fixture | `benchmarks/vertical-a/sciplex3-k562-24h-v1/support/baseline-golden-fixtures.json` | `59fd7410df297ce8a63e37068fc7d5727ebd12268526a5f43be6bded553dde49` |
| Real `p1` baseline materialization | `benchmarks/vertical-a/sciplex3-k562-24h-v1/item11-p1/materialization-manifest.json` | `7dd28d3ddca5d09d81779bfc3e02ec15d09428be354f6972e1ceda20ee1dd0e6` |

## Frozen partitions

The split is a whole-plate replicate transfer. The assignment unit is the plate, the metric
evaluation unit is the well identified by composite `(plate, well)`, and the partition rule uses only
pre-outcome design metadata. Sixteen plates: eight in `p1`, eight held out.

| Partition | Plates | Wells | Records | Condition groups | Compounds | Control wells |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `p1-train` | 8 | 768 | 94,785 | 753 | 188 | 16 |
| `p2-calibration` | 2 | 192 | 18,001 | 189 | 47 | 4 |
| `p3-model-selection-validation` | 2 | 192 | 20,481 | 189 | 47 | 4 |
| `p4-untouched-test` | 4 | 384 | 40,385 | 377 | 94 | 8 |

Four exact doses — 10, 100, 1,000, and 10,000 nM — are supported.

The definition-time leakage review computed its overlap counts rather than declaring them. Record,
well, and protected-plate overlaps are all zero; the three held-out partitions share no compound with
one another, over a held-out compound union of 188; and every evaluation condition has a
replicate-1 counterpart in `p1`. The outcome-free evaluation schedule holds 1,536 well-level cases —
1,504 treated and 32 no-action control — over 16 content-addressed static plate contexts, each
treated case naming its matched-control evaluation units and matching stratum.

That review records its own unassessed residue: source-duplicate detection beyond exact source
record identifiers has not run. The admission-time leakage audit required by ADR 0008 has not run
and is not required, because the benchmark is not on the admission path.

## Partition access and the lifecycle seal

Partition identity and access purpose are one-to-one:

| Purpose | Exact partition | Permitted lifecycle use |
| --- | --- | --- |
| `TRAIN_PARAMETERS` | `p1-train` | Fit candidate model parameters |
| `FIT_CALIBRATION` | `p2-calibration` | Fit/freeze uncertainty calibration only |
| `MODEL_SELECTION` | `p3-model-selection-validation` | Select and freeze a candidate |
| `UNTOUCHED_EVALUATION` | `p4-untouched-test` | Evaluate the already frozen component only |

Requesting `p2`, `p3`, or `p4` semantic value access for parameter training fails. In particular,
`p4` cannot tune weights, calibration, thresholds, feature order, baselines, or hyperparameters.

The loader is narrower still: its session type exposes only the `p1-train` role. `p2` requires a
future trusted grant bound to an exact `TRAINED_CANDIDATE`; `p3` requires one bound to an exact
`CALIBRATED_CANDIDATE`; and `p4` raw endpoint outcomes and scoring require a locked-evaluation grant
bound to an exact `MODEL_SELECTED_FROZEN` candidate. Those grants are issued by
[ADR 0011](../adr/0011-sciplex3-p1-loader-and-baselines.md)'s lifecycle machinery, whose issuing
control plane is suspended; see below. The `p1` session does not parse held-out source outcomes or
the public held-out membership ledgers.

That partition seal is semantic, not a claim that the physical H5AD can be split before access.
Resolving exact `p1` rows requires transfer, snapshot, and full-axis selector-metadata decode of the
entire opaque asset, including held-out selectors. Only `p1-train` expression and raw-count values
may be read or decoded; held-out expression and count values, endpoints, outcomes, scoring, and
lifecycle authority remain unavailable.

The checked-in benchmark intentionally exposes public frozen design metadata: exact split-membership
arrays, record/well/plate identities, well-level cases, action assignments, matched-control
identities, and an outcome-free prediction schedule. Reading that metadata does not authorize raw
endpoint access, scoring, or lifecycle evidence.

## The `p1` count surface

The loader (`src/cellstate/backends/sciplex3_loader.py`) exposes only a `p1-train` semantic session,
authenticates the exact source and `p1` closure, and yields sparse raw-count batches on the ordered
2,000-feature panel. It refuses `p2`, `p3`, and `p4` before protected raw-value access.

Its exact, close-reauthenticated scan of source SHA-256
`603ed16c5e25401c8a7f5bb0b2b045179701017d65dcfc6aeea71722a66cd10a` covered 94,785 `p1` records
across 768 wells — 752 treated and 16 control — retained seven zero-panel records, and emitted six
fitted-state identities without reading or decoding held-out expression/count values or endpoints.
The receipt records `heldout_memberships_parsed: false`, `heldout_outcome_values_parsed: false`,
`lifecycle_evidence_issued: false`, and `scientifically_admissible: false`.

## Mandatory baseline software

Six implementations are in frozen scope. Five are mandatory probabilistic baselines:

1. matched-vehicle resampling;
2. exact-condition replicate-1 empirical resampling;
3. exact-condition negative binomial;
4. hierarchical well negative binomial; and
5. low-rank compound-dose response.

Nearest-supported-dose resampling is the sixth, recorded as a secondary applicable baseline.

All six implement the no-action target. Matched-vehicle pools use only same-plate vehicle wells from
`p1`; held-out controls are scoring comparators, not fit data. Nearest-supported-dose resampling
excludes the exact requested dose, minimizes absolute log10-dose distance, and resolves a tie toward
the lower dose. Every exact `p1` record is retained in fit statistics — a zero-panel training row is
never excluded or imputed — while predictive draws remain strictly positive.

The fixed prediction campaign requests 512 raw-count samples for each case and each seed, uses seeds
`0`, `1`, `2`, `3`, and `4`, and instantiates NumPy `PCG64DXSM` explicitly. Fitted-state manifests
bind implementation, feature order, random-number contract, statistical semantics, and the exact
arrays or empirical pools; the six identities are listed in the materialization manifest above.
Runner scaffolding content-addresses fitted state before held-out access and writes prediction shards
incrementally, so a complete campaign need not be materialized as one dense array.

The suite's recorded execution status is `software_golden_passed_real_benchmark_not_run`. These are
reproducibility mechanisms, not evidence that a prediction campaign, metric, acceptance comparison,
or benchmark performance gate passed. Scoring these baselines against one another is the
observational-floor measurement the roadmap schedules in Phase 1, and it is blocked: it needs
protected-partition predictions, and `p2`/`p3`/`p4` are sealed behind the ADR 0011 grants named
above. The roadmap holds the unblocking condition. Persistence and temporal state-space are
inapplicable to this query and unimplemented; any scoreboard produced here must say so on its face.

## Fail-closed preflight

The gate re-reads and hashes all five frozen artifacts on every preflight. It rejects:

- query, bundle, support-envelope, manifest, or benchmark byte drift;
- unknown or modified well cases, plate contexts, action/dose assignments, targets, or horizons;
- partition/access-purpose mismatches and untouched-test mutation;
- future endpoint RNA supplied before the inference cutoff;
- environment or transport requests;
- point-only output in place of a predictive distribution;
- identified/transported causal labels, known realization, viability/survival, or hidden-state
  interpretations; and
- missing weights, calibration/model-selection/locked-test evidence, executable metrics and
  uncertainty, completed and passed baseline runs, source-duplicate audit, or scientific admission.

Even a request with exact scope receives readiness blockers, and `sample_response` raises before
reading or returning biological values.

Regenerate and verify the two canonical component artifacts with:

```bash
PYTHONPATH=src uv run --no-sync python scripts/build_sciplex3_k562_component_scaffold.py --check
```

## There is no admitted component response contract

There is deliberately no constructible or exported admitted component response contract. A future
one must resolve and inspect its payload bytes — not trust `ArtifactRef` metadata — and verify shape
`(sample_count, 2000)`, integer dtype, nonnegative raw-UMI values, nonzero totals, ordered-feature
identity `8b9e0e71d9bfc79a5e1db29f73bc30006bf117a29db8abf63b1607403629401f`, and target-value-schema
identity `b2463271246eca932824ad4d0089aaf3c924afcedec865dec8e04c4bbf7b23e2`. Its provenance must bind
the exact bundle, support envelope, query, benchmark, training run, model artifact, validation
evidence, evaluation case, task fingerprint, and seed. Until that byte-verifying boundary exists,
`sample_response` is typed `Never` and always raises.

Even a complete component lifecycle would authorize only this direct component surface. It would not
authorize a hidden-state estimator.

## Retired and suspended work

This work is frozen, not deleted. It may be resumed only by an ADR stating which state-capability
the resumption advances. [ADR 0012](../adr/0012-sciplex3-p1-trained-candidate.md) is the historical
record of the candidate lifecycle and why it ended.

**The rank-16 continuous admixture candidate family is retired for the state path.** It is a
condition-level mean model whose latent is indexed at the well, with a free parameter per observed
action that cannot generalize to an unseen action in principle. Versions v1 through v4 each failed
their real or nonissuing `p1` fits; a later source-code audit found that v4's dose-block objective,
gradient, and Hessian omit the equal-well factor `N/W = 94785/768` applied to the corresponding
action likelihood in its tracked ELBO, so the dose update did not optimize that ELBO. No candidate
artifact, plan, observation, training evidence, materialization, or lifecycle result was ever
issued, and `TRAINED_CANDIDATE` was never derived. What survives is carried into the roadmap's
Phase 4 model guidance: the corrected objective mathematics, the equal-unit normalization constant,
and the effective-context diagnostic. The lifecycle scaffolding is not.

The Item 12.1a audit lineage is canonical and frozen under
[`audits/item12_1a`](https://github.com/logannye/cellstate/tree/main/audits/item12_1a), which holds
its own README:

| Frozen file | SHA-256 |
| --- | --- |
| Characterization harness | `f4e6b76847bd926952995d66233389768f091135699fb60a38d7d9762bb03ff1` |
| Characterization tests | `8989618e259fb4aed0e0798bc010e40092c45e6bd30234bb3a7b534cdc562903` |
| Parent driver | `795c59296f5cefb1b6dd78a021ea0eb8e795217eda5226becf6c5bf909f6623a` |
| Infrastructure-invalid v4 no-fit report | `4677fc8ef1a458bf3616abc507250572c2da7a8d53c1c8a7a03d4b097f3d4877` |
| Valid exact-reference v4 nonissuing report | `66e9debc1a402e7aa68cbc934f7c5f641529eea3187ec15606364c912af8faa8` |

Both reports are unsuccessful execution records. Neither is a model, training evidence, or a trusted
lifecycle result, and neither must be interpreted as science.

An earlier v2 diagnostic trajectory was rejected and its software is no longer in the working tree.
Its identities are retained here so the lineage is not lost: implementation source
`87e08b4d65596b9a1e2234d2db234293fdf1392ad443d2611283a6911cbcb3c0`, runner source
`0c963dd035577567f28f11cd62727141a1c7bc627dc1326a3278591633953344`, specification
`7bd027ee95a238c039d35f1aa5547d48158b2515b19da3372ebede41a24ee670`, output-model schema
`8ce0511161df45ea434aeda3292a534cae4668d80d2ba82a84dd527db76911ff`, synthetic golden model
`2ab05dc29bcad67aaa60640b8c6b3090127023fdcea745f2bba31f84c44ad64f`, and synthetic golden sample
`26b601ce6779cb5bdca9337ed1f6eaeb41bd3e10c728ae1d111c15ba1bca8e01`. They identify a rejected
diagnostic trajectory, not a fitted model, a training plan, or a current candidate family.

**The protected-execution authorization control plane is suspended.** It governs execution of a
candidate for a component that cannot emit a cell-state belief, so it advances no state capability.
No proposal is to be approved and no protected execution is to be dispatched. The suspension is
enforced by a fail-closed first step in `.github/workflows/item12-3-sciplex3-v5.yml`, not by prose:
the job refuses dispatch before any other step runs. No canonical pending proposal exists, no
proposal digest has been approved or consumed, and no protected run has occurred.

**Container containment and reproducible runtime distribution are frozen as delivered.** The
Dockerfile, requirements lock, image lock, and distribution lock live under
[`containers/sciplex3-v5-runtime`](https://github.com/logannye/cellstate/tree/main/containers/sciplex3-v5-runtime),
whose README is the authority for the build, containment, and verification detail and is itself
marked suspended. The reproducible OCI archive has SHA-256
`37c2fa5846acfbd8357476859bd7f8f0ac6591261d79c2f6f46f0aa22fb76454` and is distributed as the sole
asset of the immutable `sciplex3-v5-runtime-20260811-locked` GitHub Release; the index, child
manifest, config, and builder identities are recorded in `runtime-image-lock.json` and
`runtime-distribution-lock.json`. Distribution is not source access, and no further work on this
software is scheduled.

The trained-candidate builder
(`scripts/build_sciplex3_k562_trained_candidate.py`) remains checked in and fail-closed: no candidate
plan, model, observation, scan, assembly, or materialization artifact exists at its canonical paths.
