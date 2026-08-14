.PHONY: install format lint lock-check type test schemas example example-reference explore build check

install:
	uv sync --all-extras --no-editable --reinstall-package cellstate

format:
	uv run --no-editable ruff format .
	uv run --no-editable ruff check --fix .

lint:
	uv run --no-editable ruff format --check .
	uv run --no-editable ruff check .

# The committed lock must satisfy the committed manifest. Nothing enforced this, so a
# dependency bump that raised a floor in pyproject.toml and left uv.lock behind stayed
# green: every job runs `uv sync` without `--frozen`, silently re-resolving and passing
# on a lock it had just rewritten in place.
lock-check:
	uv lock --check

type:
	uv run --no-editable mypy src

test:
	uv run --no-editable pytest

schemas:
	uv run --no-editable --reinstall-package cellstate python scripts/export_schemas.py

# The real-cell path, run against the INSTALLED package. This is the lane that was missing:
# `make example` used to drive the synthetic reference backend, so nothing ever exercised the
# biological backend the way a consumer gets it, and the wheel shipped without its slice for a
# release without anything noticing.
example:
	uv run --no-editable --reinstall-package cellstate python examples/estimate_real_cell_state.py

example-reference:
	uv run --no-editable --reinstall-package cellstate python examples/estimate_state.py

explore:
	uv run python scripts/explore.py knockdown
	uv run python scripts/explore.py spectrum
	uv run python scripts/explore.py day

build:
	uv build

check: lint lock-check type test schemas example build
