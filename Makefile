.PHONY: install format lint type test schemas example build check

install:
	uv sync --all-extras --no-editable --reinstall-package cellstate

format:
	uv run --no-editable ruff format .
	uv run --no-editable ruff check --fix .

lint:
	uv run --no-editable ruff format --check .
	uv run --no-editable ruff check .

type:
	uv run --no-editable mypy src

test:
	uv run --no-editable pytest

schemas:
	uv run --no-editable --reinstall-package cellstate python scripts/export_schemas.py

example:
	uv run --no-editable --reinstall-package cellstate python examples/estimate_state.py

build:
	uv build

check: lint type test schemas build
