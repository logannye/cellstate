# Contributing

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
currently `0.1-experimental`; its generated JSON Schema is structural, and admission requires Python
model validation of the cross-field scientific invariants.
