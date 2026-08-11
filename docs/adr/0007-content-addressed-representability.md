# ADR 0007: content-addressed real-data representability proofs

- **Status:** accepted
- **Date:** 2026-08-09

## Context

A valid dataset manifest is not evidence that a dataset can express a particular cell-state
estimand. Public studies often mix cohorts, collection protocols, destructive and nondestructive
measurements, uncertain clocks, and endpoints in the same deposited artifact. Treating the whole
accession as one homogeneous scientific unit would fabricate linkage. Conversely, calling a source
"representable" must not imply that its license permits training or that it satisfies a benchmark
loss, metric, split, or causal-identification requirement.

## Decision

The experimental manifest advances to `0.3-experimental` and requires an exact dataset slice.
Content-selected slices bind a versioned selector, selector digest, canonical selected-record
membership digest, record and subject counts, selection stages, and every evidence source used to
derive membership. Whole-artifact slices bind their complete record axis. Interval-valued
observation and outcome clocks remain intervals; implementations may not replace them with an
invented midpoint.

`RepresentabilityProof` is a separate `0.1-experimental` artifact. It binds one manifest
fingerprint, slice fingerprint, exact positive and negative assessment fingerprints, declared
source-byte digests, and a complete typed criterion ledger. `verify_representability` derives the
result from those bindings and structural facts. A caller cannot assert a free-form passing boolean.

The current verifier is a machine check of a reviewed attestation ledger. It does not fetch or
resolve source bytes, execute a selector, or recompute selected membership. In particular, the wire
value `selector_execution` names the evidence method attested by a reviewer; it is not a runtime
execution receipt. Every resolution reports `selector_execution_replayed=false` and
`source_bytes_resolved=false` so structural acceptance cannot be mistaken for an execution claim.

Representability, scientific eligibility, and legal permission remain three distinct decisions:

1. a proof may establish that an exact slice can express a subject/evidence relationship;
2. claim, loss, and metric assessments determine whether that slice is scientifically eligible for
   an exact use; and
3. layered use-policy resolution determines whether the use is authorized.

A representability resolution always reports that it did not evaluate or authorize legal use.

## First accepted reviewed ledgers

- Replogle 2022 K562 essential-scale Perturb-seq establishes in the reviewed ledger that destructive
  single-cell observations
  can be nested honestly under one population snapshot. It does not establish individual dynamics,
  independent biological replication, or an identified perturbation effect.
- GSE141064 Live-seq establishes in the reviewed ledger that a content-addressed 17-cell slice can
  link a viability-preserving
  pre-LPS transcriptomic biopsy to a later same-cell Tnf-mCherry response window. It is an
  associational future-function relationship, not a transported or identified causal effect.

Both proofs retain explicit negative assessments and `use_authorized=false`.

## Consequences

- Large biological bytes remain outside Git; committed artifacts contain immutable hashes,
  content-bound selector declarations and reviewed execution attestations, or whole-axis membership
  bindings. Structural verification does not replay those attestations.
- Heterogeneous accessions require exact slices rather than prose-only cohort descriptions.
- Dataset timing uncertainty stays visible in assessment identity and support checks.
- A passing proof cannot be used as shorthand for training admission, benchmark readiness, causal
  support, or legal permission.
- Vertical A can proceed to benchmark design, but model implementation remains blocked until the
  exact query, eligible source, split membership, metrics, baselines, and permissions are frozen.
