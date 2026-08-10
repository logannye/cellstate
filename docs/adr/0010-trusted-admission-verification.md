# ADR 0010: Trusted, scope-bound biological admission verification

- **Status:** Accepted
- **Date:** 2026-08-10
- **Extends:** ADR 0009 population-response component boundary

## Context

The experimental biological bundle contract can bind a query, benchmark, support envelope,
training run, model artifact, validation evidence, stage-port implementations, and public-operation
implementations by content hash. Contract version `0.1-experimental` deliberately treats those
objects as declarations. A matching checksum string, Python entry-point string, result filename, or
caller-set Boolean does not prove that bytes were read, code was loaded, an interface conformed, or
scientific results passed their frozen protocol.

The distinction matters because otherwise a complete-looking bundle could select an unauthorized
data source, reuse stale validation after code drift, rename an arbitrary file as locked-test
evidence, omit a query-dependent stage, or replay a receipt for a substituted runtime object.
Loading unresolved implementation code in the same process as a signing secret would also allow the
code being inspected to compromise its own verifier. The first sci-Plex3 component must remain
non-runnable until exact evidence is checked without collapsing those trust boundaries.

## Decision

Biological admission uses an external, content-addressed verification boundary. Serialized
receipts are auditable records; they are not bearer tokens and are never sufficient on their own.
Readiness and execution guards rebind them to the exact source bundle, external trust roots, typed
workflow decisions, and live runtime observations before deriving state.

### Authenticated execution-source selection

Real-data execution inputs come only from an application-owned workflow resolver. The resolver
derives exact source artifacts from typed scientific-assessment and permission resolutions, then
authenticates a selection bound to the bundle. The canonical resolution artifacts that justify the
selection are mandatory byte-covered inputs. Admission does not infer execution sources from every
accession or review artifact mentioned by a manifest.

### Streaming exact artifact-byte resolution

Every consumed artifact is acquired and read. The verifier computes SHA-256 and byte count
incrementally from bytes, a binary stream, or bounded byte chunks, so multi-gigabyte artifacts need
not be held in memory. A resolution receipt binds:

- the complete typed artifact reference, including URI, media type, byte count, and SHA-256;
- the observed URI, byte count, and SHA-256 computed from the acquired stream;
- a versioned, content-addressed verifier identity and capability;
- external audit-evidence references; and
- a keyed attestation over the exact canonical payload.

Coverage is closed-world and one-to-one. Query, benchmark, envelope, model, run, implementation,
validation, selected dataset-source, workflow-resolution, nested evaluation, and interface-
definition artifacts required by admission must occur exactly once. Missing, extra, duplicate,
truncated, or digest-mismatched bytes fail. Catalog metadata and transport headers do not count as
byte verification.

### External, nonserialized trust roots

Verifier authority is held in runtime-only `TrustedAdmissionVerifier` objects. Each combines an
exact verifier identity, a capability scope, a key ID, and an external secret of at least 32 bytes.
Receipt payloads use the versioned `hmac-sha256-v1` attestation. The secret is excluded from
Pydantic models, serialization, fingerprints, and representations. Application-owned live runtime
interfaces are likewise held outside serialized contracts.

Admission authenticates every receipt against the exact external identity/key pair and required
capability. A valid payload fingerprint without the keyed authentication tag is insufficient. This
boundary prevents accidental or stale authorization inside the supported local runtime model; a
hostile process that can steal secrets or rewrite the guards remains outside that model.

### Isolated loaded-interface conformance

A Python entry point remains a declaration until its code artifact is byte-verified and an exact
object has been loaded. The application loads code outside the receipt-issuer process and emits a
bounded, authenticated `IsolatedLoadedInterfaceObservation`. The receipt issuer authenticates that
loader observation and compares it with an application-owned `TrustedRuntimeInterface` registry
entry.

The resulting receipt binds the exact port or public operation, implementation-scope fingerprint,
code artifact, entry point, interface artifact, loaded-object identity, required member signatures,
conformance observation, isolated-loader identity, and verification identity. Specification-only
bindings, import failures, wrong objects, stale code, substituted interface registries, inherited
protocol stubs, abstract members, and incompatible signatures cannot pass. The isolated loader and
interface verifier may use separate capability-scoped trust roots.

### Typed validation-result semantics

Validation evidence is checked by a versioned semantic evaluator whose observation is authenticated
against an external trust root. A canonical result manifest binds the exact evidence kind, query,
benchmark, support envelope, training run, model, implementation scope, partition roles,
authoritative evaluation cases, covered ports and operations, required semantic criteria, and every
supporting artifact actually read. Semantic receipts must reuse the authoritative byte receipts for
the manifest and support artifacts. A filename containing `test` or a generic `passed` Boolean has
no authority.

Readiness separates `validation_results_verified` from `validation_results_passed`. Verification
means the complete typed observation is authentic and scope-correct. Passing means every required
semantic criterion passed. A well-formed failed result remains verified, receives the typed
`VALIDATION_RESULTS_FAILED` blocker, and cannot authorize execution. Calibration, model selection,
locked evaluation, support/OOD, port, and operation evidence remain distinct roles and cannot be
replayed after scope drift.

### Query-derived prerequisites

The fixed operation-port maps are conservative floors. A deterministic prerequisite compiler also
derives conditional stages from the exact query, support envelope, bundle, and target computation
surface. Observation channels, evidence-transfer roles, intervention realization, environments and
transport, neighborhood or spatial context, uncertainty, planning, and measurement-value
requirements therefore cannot be hidden behind a smaller fixed operation declaration.

The report binds compiler identity and fingerprint, source-scope fingerprints, per-target reasons,
the required-port union, missing or extra envelope ports, unavailable dispositions, and scope
issues. Admission recompiles it from the sources. A stale report or any required stage classified
as planned, unsupported, not applicable, required-but-unimplemented, or absent blocks the target
surface.

### Just-in-time execution authority

Admission is operation-specific. A component receipt cannot authorize a public belief operation;
evolution or planning evidence cannot mint an estimator descriptor. The execution guards rederive
readiness from source artifacts and the runtime-only admission context rather than accepting a
readiness object.

After all gates pass, a code-only provider reacquires the exact admitted code inside the isolated
execution worker. The guard consumes the stream once into an immutable snapshot, rechecks its exact
count and digest, and passes that same snapshot to a registry-owned `TrustedJITLoader` whose
identity and key match the authenticated admission loader. Callers cannot pair one admitted byte
stream with a separately supplied object. The guard then repeats loaded-object identity and
interface conformance checks and returns nonserialized `VerifiedRuntimeHandle` objects scoped to
the requested operation or direct-component prerequisites. Invocation authority is the exact
object carried by a just-in-time handle, not the persisted receipt or a previously imported
replacement.

## Consequences

- Declarations and receipts remain inspectable and reproducible without storing verifier secrets,
  live interface classes, local paths, or large biological bytes in Git.
- Missing trust, forged authentication, source-selection drift, incomplete byte coverage, stale
  code, semantic-result drift, failed criteria, query drift, or object substitution fails closed.
- A verified failed evaluation can be represented honestly without being promoted to a passing
  scientific result.
- Implementing the boundary does not graduate a biological model or expose a new public operation.
- The first sci-Plex3 population-response artifact remains at `SCAFFOLD`: it has no trained model,
  immutable executable loader, completed baseline runs, locked performance evidence, or admitted
  benchmark.
- The next engineering milestone is the immutable sci-Plex3 loader and mandatory baseline suite
  recorded in the [roadmap](../roadmap.md).

## Acceptance criteria

The boundary is complete only when tests prove that it:

- hashes large inputs incrementally and rejects missing, extra, duplicate, truncated, and digest-
  mismatched bytes;
- rejects execution sources not selected by the authenticated typed workflow;
- rejects missing, wrong-capability, wrong-key, stale-identity, and forged HMAC trust;
- rejects specification-only, stale, unloadable, abstract, and interface-incompatible
  implementations, including authenticated observations from the wrong isolated loader;
- rejects validation evidence with the wrong kind, scope, partition, case membership, model,
  implementation, evaluator, criteria, or result bytes while distinguishing verified failure from
  unverified evidence;
- changes prerequisite fingerprints when scientifically relevant query scope changes and rejects
  every conditional prerequisite not provided by the exact bundle;
- rederives all readiness and execution decisions without caller-supplied admission Booleans;
- reloads and reverifies the exact object immediately before execution and exposes only its
  nonserialized handle; and
- leaves the checked-in sci-Plex3 scaffold non-runnable and unable to emit a biological result.
