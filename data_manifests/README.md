# Dataset manifests

**Status:** the `0.3-experimental` contract is under Phase 0 review. Two reviewed representability
proofs and one reviewed sci-Plex3 K562 component-benchmark manifest are checked in. The sci-Plex3
assessment and benchmark-evaluation permission resolve for its exact CC-BY source bytes, but the
benchmark is not scientifically admitted because executable metrics, leakage, baselines, and
performance gates have not passed.

This directory will contain reviewed, machine-readable manifests for the public real-cell datasets
used by `cellstate` biological backends and benchmarks. Large source artifacts do not belong in Git.

A manifest is admitted only after verifying:

- the exact accession, release, source URL, media type, byte count, and SHA-256 digest;
- the license, attribution, redistribution, derivative-use, and commercial-use conditions;
- species, biological system, experimental-unit hierarchy, controls, replication, and batches;
- whether sampling is destructive, population-linked, clone/lineage-linked, or genuinely
  longitudinal in the same cell;
- modalities and whether they are paired in the same cell, sample, clone, or population;
- intervention assignment, target/dose/timing, matched controls, and realization evidence;
- time, lineage, spatial, environment, and functional-outcome coverage; and
- independently scoped eligibility or ineligibility for every scientific claim, training loss, and
  metric the project intends to use.

A reviewed manifest never makes its entire dataset eligible. Every downstream use names the exact
manifest fingerprint, assessment ID, and assessment fingerprint. The same claim may be supported at
one horizon or endpoint and unsupported at another. Repeated ontology outputs use stable functional
readout IDs so one endpoint cannot silently substitute for another.

Do not commit aspirational manifests, placeholder hashes, or inferred licensing. Candidate datasets
remain in the evidence portfolio in `docs/architecture/full-buildout.md` until review is complete.

A representability proof answers only whether an exact real-data slice can express a declared
subject/evidence relationship. It does not resolve use permission, admit a loss or metric, or make
the source a benchmark. The first two reviewed proofs are:

1. Replogle 2022 K562 essential-scale Perturb-seq: a destructive day-6 single-cell assay nested in
   one population snapshot. It proves population/destructive semantics and explicitly does not
   claim same-cell dynamics or an identified intervention effect.
2. GSE141064 Live-seq RAW functional recorder: an exact 17-cell, content-addressed slice linking a
   viability-preserving pre-LPS transcriptomic biopsy to a later same-cell Tnf-mCherry response
   window. It proves individual future-function linkage and remains associational.

The first benchmark-oriented reviewed manifest is the corrected scPerturb v1.4 sci-Plex3 K562
slice. It binds 173,652 nuclei to 1,536 composite `(plate, well)` population subjects, preserves
plates as split units, and supports only the randomized assignment-to-24-hour captured-nucleus
endpoint estimand. It does not support a pretreatment molecular belief, same-cell dynamics,
viability, unseen-compound transport, or a complete Vertical A benchmark.

The remaining manifest queue is:

1. a replicated source with an admissible pretreatment molecular state and independent future
   endpoint for the complete Vertical A benchmark;
2. LARRY (`GSE140802`) -- clone and fate evidence;
3. MIX-Seq -- time-resolved population drug response and condition-level viability; and
4. one fast-signaling source -- short-horizon phosphosignaling dynamics.

The manifest graph keeps lifecycle concerns separate:

- dataset manifests describe scientific identity, study design, layered use policy, scoped claim,
  loss, and metric assessments, and acquired source-artifact records binding retrieval time,
  resolved bytes, checksums, and storage;
- normalization manifests bind transformations and source-row provenance; and
- benchmark split manifests bind immutable train, calibration, and test membership.

Training-run, model-card, and validation-claim records will reference that graph by manifest and
assessment fingerprints. A declared metric split unit is a structural requirement, not proof of
actual split membership; the future split manifest supplies that proof.

The generated JSON Schemas are structural. Scientific invariants that depend on several nested
fields require validation through the Python `DatasetManifest` and `RepresentabilityProof` models
and `verify_representability`; JSON-Schema validation alone is not an eligibility decision. The
contract remains experimental until Vertical A's query, subject, and benchmark semantics are
frozen.
