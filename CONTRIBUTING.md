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
