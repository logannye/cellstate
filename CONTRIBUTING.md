# Contributing

This project computes a faithful and accurate representation of hidden cellular state. Everything
else is instrumental to that, and work that advances no state capability is not scheduled however
well engineered.

Before proposing work, read [`docs/roadmap.md`](docs/roadmap.md). It is the sole authority for
implementation order, for the state-capability ledger `S1`-`S10`, and for graduation status, and it
binds four rules on contributions:

1. **The purpose test.** Every queue item names the ledger capabilities it advances. An item that
   advances none is not scheduled. Enforced by `tests/test_roadmap_queue_contract.py`.
2. **Authorization precedes implementation.** A change that amends the roadmap may not also
   implement the work it authorizes; authorization lands first, as its own merged change.
3. **Numbered items only.** The implementation queue is one ordered list. Appended prose confers no
   authorization and creates no queue item.
4. **Order changes require an ADR.** Any change to phase order, to the ledger, or to a graduation
   gate is a contemporaneous decision record under `docs/adr/`.

Install the development environment with `uv sync --all-extras --no-editable`, then run
`make check`. The non-editable install keeps the documented workflow reliable on Python runtimes
that skip filesystem-hidden `.pth` files. Run
`uv sync --all-extras --no-editable --reinstall-package cellstate` after changing package source so
the local wheel cannot remain stale.

Changes to serialized domain models require a schema-version decision, regenerated JSON schemas,
and round-trip tests. New scientific backends must document training support, uncertainty semantics,
out-of-distribution behavior, and validation evidence. Passing software tests is not evidence that a
model is biologically valid.

Biological data enter through `cellstate.data.DatasetManifest`. A manifest must use verified source
URLs, exact checksums, and reviewed license/use terms; placeholder hashes and guessed eligibility
are not acceptable. Dataset, donor, well, clone, intervention, or other shared experimental units
must remain together in benchmark splits as required by the study design. Synthetic fixtures may
test software behavior but cannot support a biological validation claim. The manifest contract is
currently `0.3-experimental`; its generated JSON Schema is structural, and use requires Python model
validation of the cross-field scientific invariants plus exact manifest, slice, and assessment
fingerprint references. Representability proofs require the Python verifier and do not authorize
data use. Never treat a reviewed manifest or accession as dataset-wide eligibility.

`make check` runs lint, types, tests, schema export, and the package build; CI additionally runs
`mkdocs build --strict`, so a documentation link or anchor can break the build without failing
`make check`.

Some parts of the repository are records rather than working files. Accepted ADRs under `docs/adr/`
are historical decision records: amend a Status line or append a bracketed pointer, but never
rewrite a decision or its rationale. `CHANGELOG.md` records what happened and is append-only —
entries are never rewritten or removed. Artifacts under `data_manifests/`, `benchmarks/`,
`backends/`, `audits/`, and `containers/` are frozen evidence, several of them bound by
content-addressed fingerprints; do not edit them to make a document read better.
