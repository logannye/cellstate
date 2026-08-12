# ADR 0009: Keep endpoint assay response separate from hidden-state belief

- **Status:** Accepted
- **Date:** 2026-08-09

## Context

The first frozen Vertical A component benchmark asks for the distribution of a destructive
single-nucleus RNA endpoint 24 hours after an assigned sci-Plex3 compound and dose. Its input is
static K562 well context plus the intended assignment. The source has no pre-treatment molecular
observation of the same population, no individual-cell temporal linkage, no measured target
engagement, and no survival or viability target.

It would therefore be scientifically incorrect to make this task satisfy `estimate_cell_state` by
calling its design context a prior, calling its 24-hour vehicle observations a current-state
reference, or returning its endpoint response as a `CellStateBelief`. It would also be incorrect to
register evolution, intervention-planning, or measurement-selection surfaces that the evidence
cannot validate.

At the same time, this endpoint task is a useful first engineering vertical. It can exercise exact
data, query, split, support, training, calibration, prediction, and benchmark boundaries before a
full state-space model exists.

## Decision

Introduce an experimental biological-bundle contract and a separate
`PopulationAssayResponseModel` component boundary.

The sci-Plex3 component has the exact meaning:

```text
P(raw ordered 2,000-panel UMI counts in recovered nuclei at 24 hours
  | static K562 well context, intended compound-dose assignment)
```

It is not a posterior over hidden state at the inference cutoff. Its output, once admitted, must be
predictive count samples with the exact feature order and target schema. Its causal label remains
`predictive_association`; intervention realization remains unknown. It may not claim viability,
survival, source-population size, mechanism, transport, or individual-cell dynamics.

Every biological bundle must classify the complete proposed stage-port surface. The narrow
component additionally declares the infrastructure needed for exact artifact resolution, query and
scope checking, partitioned data access, context/action encoding, distribution prediction, support
gating, calibration, and provenance. A port is `required`, `provided`, `planned`,
`not_applicable`, or `unsupported`; missing implementations cannot be represented as successful
no-ops.

Public runtime operations have fixed prerequisite port sets and exact high-level implementation
bindings. Merely implementing a Python protocol does not authorize an operation. A component
scaffold exposes none of `estimate_cell_state`, `evolve_cell_state`, `choose_intervention`, or
`recommend_next_measurement`, does not mint a biological `EstimatorDescriptor`, and cannot be
inserted into `CellStateModelBundle`.

The contract reserves this evidence-derived lifecycle and never accepts caller-set readiness flags:

1. `SCAFFOLD`
2. `TRAINED_CANDIDATE` -- parameters fit only from the training partition
3. `CALIBRATED_CANDIDATE` -- exact calibration evidence from the calibration partition
4. `MODEL_SELECTED_FROZEN` -- selection evidence and a frozen model before protected untouched-test
   endpoint and scoring access
5. `COMPONENT_EVALUATED` -- complete executable benchmark results and leakage evidence
6. `COMPONENT_GATES_PASSED` -- the component benchmark is scientifically admitted

Contract version 0.1 cannot advance this lifecycle at all: it has no trusted byte resolver,
loaded-interface verifier, validation-result verifier, or query-derived conditional prerequisite
engine. Every bundle therefore derives as `SCAFFOLD` even if its declarations look complete. A
future version may advance the lifecycle only from verifier receipts. Even then, it can authorize
only the exact component API. Hidden-state estimation, evolution, control, and measurement selection
require separate future evidence and remain unreachable from component admission.

The initial checked-in sci-Plex3 bundle is deliberately a non-runnable `SCAFFOLD`. Its support
envelope, query, manifest, benchmark, and complete port dispositions are content-addressed. Model
weights, training and calibration runs, executable metrics, passed baselines, validation evidence,
and performance admission are absent. The execution boundary must recompute readiness from the
source artifacts and reject the call.

## Consequences

- The project can implement and test its first real-data adapter without misrepresenting an
  endpoint regression task as hidden-state inference.
- Partition-role violations, target leakage, hash drift, unsupported scope, causal overclaims, and
  point-only outputs fail before model execution.
- Later endpoint response models may share data and evaluation infrastructure with full biological
  backends, but they cannot silently acquire belief-state semantics.
- The next implementation step is a trusted artifact/interface/result verification boundary,
  followed by an executable provenance-bound training and evaluation path for the frozen component,
  beginning with mandatory baselines. It is not registration of a public cell-state runtime.
  [Historical. The verification boundary landed as [ADR 0010](0010-trusted-admission-verification.md)
  and the p1 loader and baselines as [ADR 0011](0011-sciplex3-p1-loader-and-baselines.md). The
  component's training and evaluation path is no longer scheduled; see
  [ADR 0013](0013-state-first-roadmap-reordering.md).]
- A later state-space backend needs pre-cutoff biological observations or an independently
  validated prior, posterior inference, dynamics, sufficiency, identifiability, calibration, and
  operation-specific admission evidence.

## Rejected alternatives

- **Return a `CellStateBelief` containing the endpoint distribution.** This confuses a future assay
  target with current hidden state.
- **Use 24-hour vehicle wells as a time-zero prior.** They are destructive endpoint comparators.
- **Expose stub implementations for all four public operations.** Interface completeness is not
  scientific capability.
- **Treat `COMPONENT_BENCHMARK` or `verified=True` as admission.** Scope classification and
  structural verification do not establish performance.
- **Allow a model card or descriptor to assert readiness.** Readiness must be recomputed from exact
  training, calibration, validation, benchmark, and implementation artifacts.
