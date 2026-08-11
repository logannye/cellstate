# Biological model bundles and support ports

Biological execution is authorized by an exact collection of artifacts and runtime observations,
never by a model name, a caller-set `validated` flag, or a serialized receipt alone. The
experimental contracts in `cellstate.backends` bind a query, benchmark, support envelope, training
run, model artifact, validation evidence, every internal stage port, and every public-operation
implementation by content hash. The trusted admission boundary then rebinds those declarations to
the bytes and objects actually observed.

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
- uncertainty calibration, OOD detection, sufficiency, and identifiability; and
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
resolved, its loaded entry point has passed interface verification, and exact typed validation
results cover and pass it. A `PYTHON_ENTRY_POINT` string is only a declaration. It never makes a
port executable, and a specification-only implementation cannot pass the gate.

## Trusted admission boundary

The admission context contains persisted audit records plus runtime-only trust material. Its
`TrustedAdmissionVerifier` entries bind a versioned verifier identity and capability to an external
HMAC-SHA256 key. Its `TrustedRuntimeInterface` entries bind interface declarations to
application-owned live interface classes. Secrets and live classes are dataclass fields excluded
from every Pydantic model, JSON payload, fingerprint, and receipt. A report submitter therefore
cannot create authority by copying or editing a serialized receipt.

### Execution sources and exact bytes

Real-data sources are not inferred from every source mentioned in a review manifest. An
application-owned science-and-permission resolver derives the exact execution sources from typed
dataset-assessment resolutions and emits an authenticated `ExecutionSourceSelectionReceipt`. The
selection binds the bundle, sources, resolver identity, and the workflow-resolution artifacts that
justify the selection. Those workflow artifacts and selected sources are themselves part of exact
byte coverage.

Artifact resolution hashes concrete byte streams incrementally, so large sources need not be
materialized in memory. Each receipt binds the complete declaration, observed URI, SHA-256, byte
count, verifier identity, audit evidence, and keyed attestation. Coverage is closed-world and
one-to-one across contract artifacts, selected dataset sources, nested benchmark and run artifacts,
validation artifacts, implementation code, and trusted interface definitions. Missing, extra,
duplicate, truncated, or digest-mismatched inputs fail.

### Isolated loaded interfaces

Loading untrusted implementation code beside a verifier secret would collapse the trust boundary.
An application-owned isolated loader therefore acquires the exact code, loads the declared entry
point outside the receipt issuer, and authenticates a bounded observation of the loaded object and
its interface members. A separate verifier authenticates that observation, compares it with the
exact implementation requirement and application-owned interface registry, and issues a
`LoadedInterfaceReceipt`.

The receipt binds the port or public operation, implementation scope, code artifact, entry point,
loaded-object identity, interface artifact, required member signatures, conformance observation,
loader identity, and verifier identity. Specification-only declarations, import failures, stale
code, substituted registries, inherited protocol stubs, abstract implementations, or mismatched
signatures fail.

### Typed validation results

A validation evidence filename or generic `passed` Boolean has no authority. A canonical
`ValidationResultManifest` binds the evidence role, exact query, benchmark and support envelope,
training/model and implementation scope, partition roles, authoritative case IDs, covered ports and
operations, required semantic criteria, and supporting result artifacts. Semantic evaluator
receipts reuse the authoritative exact-byte receipts for the manifest and every support artifact,
and are authenticated against a capability-scoped external trust root.

Admission records two separate states: `validation_results_verified` means the complete typed
semantics were authenticated and rebound; `validation_results_passed` means every required
criterion actually passed. A well-formed failed evaluation is therefore verified but still receives
`VALIDATION_RESULTS_FAILED` and cannot authorize execution. This distinction also lets the derived
lifecycle record that evaluation occurred without relabeling failure as scientific admission.

### Query-derived prerequisites

Fixed operation-port maps remain conservative floors. A deterministic compiler additionally
derives conditional ports from the exact query, support envelope, bundle, and requested component
or public-operation surface. Observation channels, evidence-transfer roles, intervention
realization, environments and transport, spatial context, uncertainty, planning, and
measurement-value requirements therefore cannot be hidden behind a smaller static declaration.

The compiled report binds compiler identity and fingerprint, query, envelope, bundle, per-target
reasons, the required-port union, missing or extra envelope ports, unavailable dispositions, and
scope issues. Admission recompiles the report from source objects and requires structural
satisfaction; a stale or caller-edited report fails.

### P1-only trained-candidate boundary

Training uses a smaller trust surface than full admission so fitting cannot resolve protected
held-out raw H5AD/UMI endpoint values or lifecycle authority, or create a circular dependency on
its own model output. A canonical
`CandidateTrainingPlan` exists before fitting and binds opaque query, benchmark, and support
identities; exact p1 loader, count-stream, scan, assembly, design, feature, action, and target
identities; candidate specification and model schema; deterministic runtime and seed policy;
contained-execution policy and runtime-image lock; the complete Python executable closure; the
exact mounted code plus declared public-control input closure; one pre-render immutable-generation
seed; and distinct trainer and candidate-factory code. It contains no final model or bundle
fingerprint.

An HMAC-authenticated `TrainingSourceSelectionReceipt` binds that plan to the exact p1 source and
the workflow resolution that selected it. `P1TrainingEvidence` then records the close-
reauthenticated source and complete count-stream closure with access restricted to the training
role. Its p2, p3, and p4 membership, case, and outcome-read flags are literal false because that
training session does not parse even the public held-out design files. After model close, reread,
and rehash, a separate semantic verifier authenticates a `CandidateFitReceipt` that binds
the plan, source selection, p1 evidence, deterministic training observation, exact model bytes,
behavior-state manifest, convergence and finiteness checks, and a successful exact reload.

The signed source and fit receipts live only in the runtime `TrainingVerificationContext`. They are
not content artifacts in `TrainingRunBinding`, so deterministic model contracts do not depend on a
secret or issuance timestamp and persisted receipt bytes never become access authority. Stage byte
coverage is mechanically derived and includes only top-level query/benchmark/support identities,
the deterministic training closure, the model and run, the separately selected p1 source, and the
candidate-factory interface. `TrainingCodeClosureManifest` inventories the canonical Python
executable closure; `ExecutionInputClosureManifest` adds every exact mounted public JSON/runtime
input without treating the protected source as a public control file. The worker and parent share a
typed, no-follow `StagedTrainingInventory`, excluding only their observation metadata, and a
`ContainedTrainingObservation` joins source pre/post authentication and that stage inventory to the
parent's image, policy, process-tree, and cleanup evidence. Stage coverage does not recursively
expand public benchmark p2/p3/p4 membership and cases, protected outcomes, or later-stage
implementations.

The candidate port must provide a nonvacuous exact class interface containing `load_exact`,
`model_artifact_sha256`, `supports`, `sample`, `model_bytes`, and `behavior_manifest`; the other
support-envelope ports may honestly remain `REQUIRED`. The active v5 sampling specification makes
`supports` request-level: only an exact `CandidateSampleRequest`, including its seed and bounded
sample count, can pass; a target object alone cannot. `BundleReadiness` records training artifact,
interface, and semantic verification separately. `TRAINED_CANDIDATE` is derived only when all
three, the structural training binding, and exact query prerequisites verify. Calibration and
model-selection lifecycle stages additionally require their own future semantic verifications:
merely attaching artifacts cannot advance them. This boundary grants no protected p2 endpoint
access or calibration authority, component execution, scientific admission, public operation, or
runnable backend.

### Just-in-time execution authority

`assess_biological_model_bundle` accepts an optional runtime-only `AdmissionVerificationContext`.
Without it, all four trust-bearing readiness gates remain false. With it, assessment authenticates
the source selection and every receipt, re-derives exact coverage and prerequisites, and separates
semantic verification from scientific pass/fail. It never accepts a caller-created readiness
object as authority.

The execution guards repeat assessment from source artifacts. After admission passes, a code-only
provider reacquires the admitted code inside the isolated execution worker. The guard consumes that
stream once into an immutable snapshot, checks its exact byte count and hash, and passes that same
snapshot—not a caller-supplied object—to the registry-owned `TrustedJITLoader` bound to the
authenticated admission loader identity and key. It then repeats object-identity and interface
checks immediately before invocation. The guard returns a nonserialized
`BiologicalExecutionAuthorization` containing
`VerifiedRuntimeHandle` objects for the exact checked objects. Persisted receipts, stale loaded
objects, and unverified replacements are not invocable authority.

## Public-operation floor

Declaring a public operation adds a fixed minimum port set. Estimation requires query compilation,
a prior, posterior inference, calibration, OOD, sufficiency, and identifiability. Evolution
requires transition, target decoding, calibration, and OOD. Planning includes the evolution stack.
Measurement selection additionally requires observation outcomes, hypothetical posterior
inference, counterfactual evolution/replanning, and value-of-information support.

The bundle must also declare one exact high-level implementation for every operation in the
support envelope. Internal ports alone cannot silently register `estimate_cell_state`,
`evolve_cell_state`, `choose_intervention`, or `recommend_next_measurement`. Admission and the
just-in-time guard are operation-specific: a component authorization cannot invoke a public belief
operation, and evolution or planning evidence cannot mint an estimator descriptor.

## Derived admission and lifecycle

`assess_biological_model_bundle` revalidates exact query, benchmark, support, training, model,
implementation, validation, and benchmark-admission bindings. Its implementation-scope fingerprint
is noncircular: it covers the model artifact, posterior schema, every port disposition and code
declaration, and every high-level operation without hashing the validation evidence back into
itself. Any model, schema, code, interface, entry-point, query, split, case, receipt, trust-root, or
compiler drift invalidates the corresponding gate.

The ordered lifecycle is derived rather than declared: `SCAFFOLD`, `TRAINED_CANDIDATE`,
`CALIBRATED_CANDIDATE`, `MODEL_SELECTED_FROZEN`, `COMPONENT_EVALUATED`, and
`COMPONENT_GATES_PASSED`. Trusted admission infrastructure alone advances nothing. Progress also
requires exact stage-scoped byte, interface, and semantic verification in addition to structural
training/model, calibration, and selection bindings; complete typed evaluation; executable
benchmark results; mandatory baselines; acceptance-policy pass; and benchmark admission
appropriate to each stage. Artifact presence alone cannot advance calibration or model selection.
A verified failed evaluation can reach the evaluated stage but never the gates-passed stage.

`require_biological_component_execution` can authorize only an admitted `COMPONENT_MODEL` on its
separate direct component surface. `require_biological_execution` additionally requires the exact
requested public operation to be declared, implemented, evidenced, and admitted in a full bundle.
`build_admitted_estimator_descriptor` can bridge only a full bundle admitted specifically for
`estimate_cell_state`.

## First component boundary

The sci-Plex3 K562 24-hour task is a direct population assay-response component:

```text
(static context, assigned compound and dose) -> population distribution of future assay counts
```

Its 24-hour recovered-nucleus RNA observations are future targets. Matched 24-hour vehicle wells
are endpoint comparators. Neither is a pre-cutoff observation or a prior over hidden current state.
The component therefore registers none of the four public cell-state operations, has no biological
`EstimatorDescriptor`, and cannot emit a current-state belief.

Item 10 supplies the trusted admission machinery. Item 11 separately supplies the immutable p1-only
loader, exact training scan, and six mandatory fitted baseline-state identities; it does not supply
the trained-candidate factory/interface, a candidate model, or lifecycle evidence. Neither item
supplies prediction runs, metrics, performance evidence, or an admitted benchmark. Candidate
revisions v1 through v4 all failed closed without issuing a model or lifecycle result.

Item 12.2 has now completed only the source-free v5 software boundary: one exact equal-well
objective and compatible all-well M-step; independent gradient, Hessian, and nondecrease checks;
exact-positive request-level sampling through 512 draws under a global `2^-64` conditional signed-
`int64` tail budget; immutable-generation atomic publication and recovery; exact code, mounted-
input, and staged-output closure; reproducible Linux `amd64` OCI identities; and parent-owned
whole-container wall/memory and host-effective UID/GID containment. It opened no protected source,
ran no real-`p1` fit, and issued no candidate artifact, plan, observation, evidence, or lifecycle
result. The clean-final OCI archive is byte-identical across three independent native and emulated
builder environments under frozen inputs and toolchain and excludes build-host caches; its builder
identity is frozen in the workflow and runtime lock. The checked-in sci-Plex3 component therefore
remains `SCAFFOLD` and non-executable. Item 12.3 is only a proposed, separately authorized, version-
bound, nonissuing real-`p1` v5 execution; it has not been authorized or run. Durable distribution of
the exact locally loaded archive is a prerequisite, and no remote image publication is claimed. See
the [roadmap](../roadmap.md), the sole sequence and status authority.
