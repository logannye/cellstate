# Dataset manifests

**Status:** the `0.1-experimental` contract is under Phase 0 review. No manifest has been admitted.

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
- eligibility or ineligibility for every scientific claim the project intends to make.

Do not commit aspirational manifests, placeholder hashes, or inferred licensing. Candidate datasets
remain in the evidence portfolio in `docs/architecture/full-buildout.md` until review is complete.

The first manifest queue is:

1. one K562 perturbation source with replicated controls -- first population benchmark;
2. Live-seq (`GSE141064`) -- separate same-cell state-to-future contract evidence;
3. LARRY (`GSE140802`) -- clone and fate evidence;
4. MIX-Seq -- time-resolved population drug response and condition-level viability; and
5. one fast-signaling source -- short-horizon phosphosignaling dynamics.

The manifest graph keeps lifecycle concerns separate:

- dataset manifests describe scientific identity, study design, use policy, scoped capability
  assessments, and acquired source-artifact records binding retrieval time, resolved bytes,
  checksums, and storage;
- normalization manifests bind transformations and source-row provenance; and
- benchmark split manifests bind immutable train, calibration, and test membership.

Training-run, model-card, and validation-claim records will reference that graph by fingerprint.

The generated JSON Schema is structural. Scientific invariants that depend on several nested fields
require validation through the Python `DatasetManifest` model; JSON-Schema validation alone is not
an eligibility decision. The contract remains experimental until Vertical A's query, subject, and
benchmark semantics are frozen.
