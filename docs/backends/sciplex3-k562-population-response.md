# sci-Plex3 K562 population assay-response scaffold

## Status

This is a non-runnable `SCAFFOLD`, not a biological model and not one of the four public cell-state
operations. `SciPlex3K562PopulationAssayResponseScaffold` implements the separate typed
`PopulationAssayResponseModel` component port only to resolve exact artifacts, preflight one frozen
task, and reject execution. It has no model weights, training run, calibration evidence, model-
selection freeze, locked evaluation, or admitted benchmark performance.

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
| Benchmark artifact | `b6feb9f74d07f513211202139df0607a1e897b864937d64d6614dd90cf8f75a1` |
| Component support envelope | `987674307b27a5740d5e026546ab9a5543bc001727dc45af7f574f60e0044400` |
| Component bundle contract | `1bd400f5a03c1e16dff4790ef7b8a180c2cfa7d26d2f0ccef2e91f19ec599264` |

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
  uncertainty, passed baselines, source-duplicate audit, or scientific admission.

Even a request with exact scope receives readiness blockers and `sample_response` raises before
reading or returning biological values. The component exposes no `estimate`, `evolve`, planning,
measurement, or condition-response lookup entry point.

Regenerate and verify the two canonical component artifacts with:

```bash
PYTHONPATH=src uv run --no-sync python scripts/build_sciplex3_k562_component_scaffold.py --check
```

The next implementation step is the trusted artifact, loaded-interface, result-semantic, and
query-prerequisite verifier boundary required by bundle contract v0.1. After that comes the immutable
partition-aware loader and mandatory baseline suite, followed by a distribution model, p2
calibration, p3 freeze, and one locked p4 evaluation. Passing those gates could authorize only this
direct component surface; it would not authorize a hidden-state estimator.
