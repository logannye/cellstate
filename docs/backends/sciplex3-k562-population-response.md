# sci-Plex3 K562 population assay-response scaffold

## Status

This is a non-runnable `SCAFFOLD`, not a biological model and not one of the four public cell-state
operations. `SciPlex3K562PopulationAssayResponseScaffold` implements the separate typed
`PopulationAssayResponseModel` component port only to resolve exact artifacts, preflight one frozen
task, and reject execution. It has no model weights, training run, calibration evidence, model-
selection freeze, locked evaluation, or admitted benchmark performance.

Item 11 adds a separate, non-public software path for this frozen scope. Its immutable H5AD loader
can open only `p1-train`, authenticates the exact source and `p1` closure, and yields sparse raw-count
batches on the ordered 2,000-feature panel. The loader refuses `p2`, `p3`, and `p4` before source
access; future access requires grants bound to the preceding lifecycle state. Six probabilistic
baseline algorithms fit from `p1` only. The exact source scan and all six fitted-state identities
are recorded, but no predictions, metrics, baseline comparisons, or performance results exist; the
software path does not make this scaffold executable.

Its only eventual computation is:

```text
P(raw nonnegative integer UMI vector on the exact ordered 2,000-feature panel
  in recovered nuclei at 24 hours
  | static K562 source-well/plate context, intended compound-dose assignment or no-action control)
```

Static context is not a prior over hidden state. The destructive 24-hour RNA assay and matched
vehicle wells are future target/comparator observations, never pre-cutoff evidence. The forecast
label is `predictive_association`, intervention realization remains `unknown`, and the component
cannot claim viability, survival, target engagement, individual-cell dynamics, transport, or a
current-state belief.

## Frozen identities

| Artifact | SHA-256 |
| --- | --- |
| Reviewed dataset manifest | `6248e63237a4c0c7ae53538666a1294cf1108569792eb54702ec15f439d9cb31` |
| StateQuery | `d0fa67f31a8ea1d7b2e8839dfe7629fd6f359ea7eed4f6d336e2cd1d8813971e` |
| Benchmark artifact | `97bfb8f00f9efd93ad19635ce1a843a126c3c1b23ae6002102353c5e3bded76e` |
| Component support envelope | `17aa440c14b40981f97358119085f44b2ffeb9bed75ba322114ffe2c1c53dd9f` |
| Component bundle contract | `69eddd15eb87b167ea0ef484d54234f6af5ebd6b353ad4aa72a96c2dca3f6343` |
| Item 11 p1 loader contract | `3de5be54b60ba1403995ba79d122ee8232218be5c027da1bf530cb610ae80f90` |
| Item 11 baseline golden fixture | `59fd7410df297ce8a63e37068fc7d5727ebd12268526a5f43be6bded553dde49` |
| Item 11 real p1 materialization | `7dd28d3ddca5d09d81779bfc3e02ec15d09428be354f6972e1ceda20ee1dd0e6` |

There is deliberately no constructible or exported response/provenance model yet. A future admitted
response contract must resolve and inspect its payload bytes—not trust `ArtifactRef` metadata—and
verify shape `(sample_count, 2000)`, integer dtype, nonnegative raw-UMI values, nonzero totals,
ordered-feature identity
`8b9e0e71d9bfc79a5e1db29f73bc30006bf117a29db8abf63b1607403629401f`, and target-value-schema
identity `b2463271246eca932824ad4d0089aaf3c924afcedec865dec8e04c4bbf7b23e2`. Their provenance must
bind the exact bundle, support envelope, query, benchmark, training run, model artifact, validation
evidence, evaluation case, task fingerprint, and seed. Until that byte-verifying boundary exists,
`sample_response` is typed `Never` and always raises.

## Partition access

Partition identity and access purpose are one-to-one:

| Purpose | Exact partition | Permitted lifecycle use |
| --- | --- | --- |
| `TRAIN_PARAMETERS` | `p1-train` | Fit candidate model parameters |
| `FIT_CALIBRATION` | `p2-calibration` | Fit/freeze uncertainty calibration only |
| `MODEL_SELECTION` | `p3-model-selection-validation` | Select and freeze a candidate |
| `UNTOUCHED_EVALUATION` | `p4-untouched-test` | Evaluate the already frozen component only |

Opening `p2`, `p3`, or `p4` for parameter training fails. In particular, `p4` cannot tune weights,
calibration, thresholds, feature order, baselines, or hyperparameters.

The Item 11 loader is narrower still: its current session type opens `p1-train` only. `p2` requires
a future trusted grant bound to an exact `TRAINED_CANDIDATE`; `p3` requires one bound to an exact
`CALIBRATED_CANDIDATE`; and `p4` requires a locked-evaluation grant bound to an exact
`MODEL_SELECTED_FROZEN` candidate. The `p1` session does not parse held-out outcome or membership
ledgers.

## Mandatory baseline software

The six frozen-scope implementations are:

1. matched-vehicle resampling;
2. exact-condition replicate-1 empirical resampling;
3. exact-condition negative binomial;
4. hierarchical well negative binomial;
5. low-rank compound-dose response; and
6. nearest-supported-dose resampling.

All six implement the no-action target. Matched-vehicle pools use only same-plate vehicle wells from
`p1`; held-out controls are scoring comparators, not fit data. Nearest-supported-dose resampling
excludes the exact requested dose, minimizes absolute log10-dose distance, and resolves a tie toward
the lower dose. The fixed prediction campaign requests 512 raw-count samples for each case and each
seed, uses seeds `0`, `1`, `2`, `3`, and `4`, and instantiates NumPy `PCG64DXSM` explicitly.

Fitted-state manifests bind implementation, feature order, random-number contract, statistical
semantics, and the exact arrays or empirical pools. Runner scaffolding content-addresses fitted
state before held-out access and writes prediction shards incrementally so a complete campaign need
not be materialized as one dense array. Item 11 recorded the exact close-reauthenticated `p1` scan
and six software-only fitted-state identities. These are reproducibility mechanisms, not evidence
that a prediction campaign, metric, acceptance comparison, or benchmark performance gate passed.

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
  uncertainty, completed and passed baseline runs, source-duplicate audit, or scientific
  admission.

Even a request with exact scope receives readiness blockers and `sample_response` raises before
reading or returning biological values. The component exposes no `estimate`, `evolve`, planning,
measurement, or condition-response lookup entry point.

Regenerate and verify the two canonical component artifacts with:

```bash
PYTHONPATH=src uv run --no-sync python scripts/build_sciplex3_k562_component_scaffold.py --check
```

Bundle contract v0.1 now has its trusted artifact, loaded-interface, result-semantic, and
query-prerequisite verifier boundary. Item 11 supplies the `p1`-only loader and mandatory baseline
algorithms, plus content-addressed streaming-run scaffolding. Its exact scan of source SHA-256
`603ed16c5e25401c8a7f5bb0b2b045179701017d65dcfc6aeea71722a66cd10a` covered 94,785 `p1` records
across 768 wells, retained seven zero-panel records, and emitted six fitted-state identities without
opening a held-out partition. The later sequence remains a candidate distribution model, p2
calibration, p3 freeze, and one locked p4 evaluation.
Passing those gates could authorize only this direct component surface; it would not authorize a
hidden-state estimator.
