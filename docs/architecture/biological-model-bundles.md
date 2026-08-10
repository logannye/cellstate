# Biological model bundles and support ports

Biological execution is authorized by an exact collection of artifacts, never by a model name or
a caller-set `validated` flag. The experimental contract in `cellstate.backends` binds a query,
benchmark, support envelope, training run, model artifact, validation evidence, every internal
stage port, and every public-operation implementation by content hash.

An incomplete artifact is valid to serialize. It remains non-runnable. This is how the first
sci-Plex3 implementation can exist honestly as a scaffold without creating a biological
`EstimatorDescriptor` or returning a `CellStateBelief`.

## Port map

Every `BiologicalModelBundleContract` classifies all stages in the original proposed model bundle:

- query compiler and reference prior;
- observation and evidence-transfer models;
- intervention-realization, transition, division/inheritance, interaction, and extracellular-
  transport models;
- functional decoders and soft mechanistic constraints;
- posterior inference and model ensemble;
- uncertainty calibration, OOD detection, sufficiency, and identifiability;
- value of information.

It also classifies the infrastructure needed by a narrow direct population-response component:
exact artifact resolution, query/scope validation, train/calibration loading, action/context
encoding, the population assay-response distribution, a strict support gate, and artifact and
provenance writing. The shared uncertainty-calibration stage is reused rather than duplicated.

Each port has exactly one disposition:

| Disposition | Meaning |
| --- | --- |
| `REQUIRED` | Mandatory inside the exact envelope, but not implemented yet |
| `PROVIDED` | Declares a versioned implementation; this is not proof that its bytes execute |
| `PLANNED` | Future scope, not required by the current envelope |
| `NOT_APPLICABLE` | Scientifically irrelevant to the exact computation |
| `UNSUPPORTED` | Relevant family that this artifact explicitly refuses |

A required port can execute only after it is `PROVIDED`, its content-addressed bytes have been
resolved, its loaded entry point has passed interface verification, and exact validation results
cover it. A `PYTHON_ENTRY_POINT` string is only a declaration. It never makes a port executable,
and a specification-only implementation cannot pass the gate.

## Public-operation floor

Declaring a public operation adds a fixed minimum port set. For example, estimation requires query
compilation, a prior, posterior inference, calibration, OOD, sufficiency, and identifiability;
evolution requires transition, target decoding, calibration, and OOD. Planning includes the
evolution stack. Measurement selection additionally requires observation outcomes, hypothetical
posterior inference, counterfactual evolution/replanning, and value-of-information support.

The bundle must also declare one exact high-level implementation for every operation in the
support envelope. A future admission verifier must resolve its code bytes and validate its loaded
interface. Internal ports alone cannot silently register `estimate_cell_state`,
`evolve_cell_state`, `choose_intervention`, or `recommend_next_measurement`.

The fixed minimum port sets are only a conservative floor. Query-sensitive conditional
prerequisites are not implemented in contract version 0.1, so any declared public operation also
receives the typed `QUERY_DERIVED_OPERATION_PREREQUISITES_UNVERIFIED` blocker.

## Derived admission

`assess_biological_model_bundle` revalidates and derives:

1. exact query, benchmark, and support-envelope binding;
2. exact model/training binding to the benchmark's train and calibration partitions;
3. typed validation roles, exact partitions and evaluation cases, and port/operation coverage;
4. a noncircular implementation-scope fingerprint over the bundle kind, posterior schema, model
   artifact, every port disposition/code hash/interface/entry point, and every high-level
   operation implementation;
5. required-port provision and declarative implementation coverage;
6. benchmark admission via `verify_benchmark_artifact`; and
7. eligibility to register a public runtime surface.

Validation evidence names carry no authority. The support envelope assigns every required result a
typed kind and benchmark partition role. Each evidence binding must name exactly those partition
IDs and all corresponding frozen evaluation-case IDs. Its implementation-scope fingerprint makes
any bundle-kind, posterior-schema, model-weight, port code, interface, entry-point, or
operation-code change invalidate the old evidence without hashing the evidence back into itself.

Contract version 0.1 deliberately has no trusted resolver/receipt boundary, loaded-interface
verifier, or validation-result semantics verifier. Assessment therefore derives
`artifact_bytes_resolved=false`, `implementation_interfaces_verified=false`, and
`validation_results_verified=false` for every bundle. These yield typed blocker codes and keep
both the narrow component surface and all public runtime operations non-executable even when every
declaration looks complete. A future version must implement those verifiers; callers cannot assert
their results.

The execution guards repeat that assessment from source artifacts rather than trusting a
caller-created readiness object. `require_biological_component_execution` can authorize an admitted
`COMPONENT_MODEL` only on its separate direct component surface. `require_biological_execution`
also requires the exact requested public operation to be declared, implemented, and evidenced in
an admitted full bundle. In v0.1 that guard always fails before execution, so
`build_admitted_estimator_descriptor` cannot produce a usable biological descriptor. A future
bridge may mint one only for a bundle admitted specifically for `estimate_cell_state`; evolution
or planning declarations can never mint an estimator descriptor.

The contract names the future ordered component lifecycle:
`SCAFFOLD`, `TRAINED_CANDIDATE`, `CALIBRATED_CANDIDATE`, `MODEL_SELECTED_FROZEN`,
`COMPONENT_EVALUATED`, and `COMPONENT_GATES_PASSED`. No declaration can advance it. Because v0.1
lacks byte, interface, and result verifiers, its derived stage is always `SCAFFOLD`, including for
a complete-looking synthetic bundle. Future advancement will require verifier receipts plus exact
training/model bindings, an evidenced calibrator, frozen typed validation results, complete
executable benchmark evaluation, and benchmark admission.

## First component boundary

The sci-Plex3 K562 24-hour task is a direct population assay-response component:

```text
(static context, assigned compound and dose) -> population distribution of future assay counts
```

Its 24-hour recovered-nucleus RNA observations are future targets. Matched 24-hour vehicle wells
are endpoint comparators. Neither is a pre-cutoff observation or a prior over hidden current state.
The component therefore registers none of the four public cell-state operations, has no biological
`EstimatorDescriptor`, and cannot emit a current-state belief. Its checked-in scaffold remains
non-runnable while training artifacts, validation evidence, executable benchmark performance, and
scientific admission are absent.
