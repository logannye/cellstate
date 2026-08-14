"""The completion condition for queue item `Q1`, as a check that can fail.

[ADR 0014](../docs/adr/0014-phase-1-completion-condition.md) binds `Q1` to the frozen sci-Plex3
metric-suite specification rather than to the set of suites frozen under the current roadmap,
which is empty and would remain empty until `Q5`.  This module is that binding.

It reads the specification's own bytes, verifies its content hash and length against the values
the frozen benchmark artifact declares, and fails on any declared `metric_id` — or the uncertainty
method they all bind — that does not resolve to an implementation.  Verifying the hash first is
what stops the condition from being satisfied by editing the specification instead of writing the
code: deleting a metric from the file changes its hash and fails here before any identifier is
read.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from cellstate.evaluation.bootstrap import (
    FROZEN_SCIPLEX3_CONFIDENCE_LEVEL,
    FROZEN_SCIPLEX3_DEPENDENCE_DIMENSIONS,
    FROZEN_SCIPLEX3_RESAMPLE_COUNT,
    RESAMPLING_SCHEME,
    multiway_clustered_bootstrap,
)
from cellstate.evaluation.metrics import METRIC_IMPLEMENTATIONS

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_ROOT = REPOSITORY_ROOT / "benchmarks" / "vertical-a" / "sciplex3-k562-24h-v1"
METRIC_SUITE_SPECIFICATION = BENCHMARK_ROOT / "support" / "metric-suite-spec.json"
BENCHMARK_ARTIFACT = BENCHMARK_ROOT / "benchmark-artifact.json"

#: Declared by the frozen benchmark artifact for the `sciplex3-frozen-metric-suite` artifact.
#: These are the bytes ADR 0014 binds `Q1` to; they are not a description of the file on disk.
DECLARED_SPECIFICATION_SHA256 = "6f94fe0102f6e987cd5c5a1c6d31e58d5ad7c449c83e6d2b8e64196b38cf5634"
DECLARED_SPECIFICATION_BYTE_COUNT = 4_664


@pytest.fixture(scope="module")
def specification_bytes() -> bytes:
    return METRIC_SUITE_SPECIFICATION.read_bytes()


@pytest.fixture(scope="module")
def specification(specification_bytes: bytes) -> dict[str, Any]:
    return json.loads(specification_bytes)


@pytest.fixture(scope="module")
def declared_metric_ids(specification: dict[str, Any]) -> tuple[str, ...]:
    return tuple(entry["metric_id"] for entry in specification["metrics"])


class TestTheSpecificationIsTheFrozenOne:
    def test_content_hash_matches_the_declaration(self, specification_bytes: bytes) -> None:
        digest = hashlib.sha256(specification_bytes).hexdigest()
        assert digest == DECLARED_SPECIFICATION_SHA256, (
            "the metric-suite specification no longer hashes to the value the frozen benchmark "
            "artifact declares; the conformance target has drifted from the frozen bytes"
        )

    def test_byte_count_matches_the_declaration(self, specification_bytes: bytes) -> None:
        assert len(specification_bytes) == DECLARED_SPECIFICATION_BYTE_COUNT

    def test_the_declaration_matches_the_benchmark_artifact(self) -> None:
        artifact = json.loads(BENCHMARK_ARTIFACT.read_text())
        declarations = {
            (
                metric["implementation_binding"]["specification_artifact"]["sha256"],
                metric["implementation_binding"]["specification_artifact"]["byte_count"],
            )
            for metric in artifact["definition"]["metrics"]
        }
        assert declarations == {(DECLARED_SPECIFICATION_SHA256, DECLARED_SPECIFICATION_BYTE_COUNT)}


class TestEveryDeclaredMetricResolves:
    def test_the_specification_declares_the_expected_shape(
        self, declared_metric_ids: tuple[str, ...]
    ) -> None:
        """A guard on the guard: if the suite shrinks, the resolution test gets easier."""

        assert len(declared_metric_ids) == 10
        assert len(set(declared_metric_ids)) == 10

    def test_every_declared_metric_id_resolves_to_an_implementation(
        self, declared_metric_ids: tuple[str, ...]
    ) -> None:
        unresolved = [
            metric_id
            for metric_id in declared_metric_ids
            if metric_id not in METRIC_IMPLEMENTATIONS
        ]
        assert not unresolved, (
            f"{len(unresolved)} declared metric(s) have no implementation: "
            f"{', '.join(sorted(unresolved))}"
        )

    def test_no_implementation_claims_a_metric_the_suite_does_not_declare(
        self, declared_metric_ids: tuple[str, ...]
    ) -> None:
        undeclared = set(METRIC_IMPLEMENTATIONS) - set(declared_metric_ids)
        assert not undeclared, (
            f"implementations claim identifiers absent from the frozen suite: "
            f"{', '.join(sorted(undeclared))}"
        )

    def test_every_resolved_implementation_is_callable(
        self, declared_metric_ids: tuple[str, ...]
    ) -> None:
        for metric_id in declared_metric_ids:
            assert callable(METRIC_IMPLEMENTATIONS[metric_id].computation)


class TestTheUncertaintyMethodResolves:
    def test_the_declared_scheme_and_configuration_are_implemented(self) -> None:
        artifact = json.loads(BENCHMARK_ARTIFACT.read_text())
        configurations = {
            (
                metric["uncertainty"]["resampling_scheme"],
                metric["uncertainty"]["resample_count"],
                metric["uncertainty"]["confidence_level"],
                tuple(
                    sorted(
                        unit["dependence_id"] for unit in metric["uncertainty"]["dependence_units"]
                    )
                ),
            )
            for metric in artifact["definition"]["metrics"]
        }
        assert configurations == {
            (
                RESAMPLING_SCHEME,
                FROZEN_SCIPLEX3_RESAMPLE_COUNT,
                FROZEN_SCIPLEX3_CONFIDENCE_LEVEL,
                tuple(sorted(FROZEN_SCIPLEX3_DEPENDENCE_DIMENSIONS)),
            )
        }

    def test_the_estimator_runs_at_the_declared_configuration(self) -> None:
        """The uncertainty method resolves to code that executes, not to a named constant."""

        wells = 384
        values = [float((index % 7) - 3) for index in range(wells)]
        labels = {
            "compound": [f"compound{index % 95}" for index in range(wells)],
            "plate": [f"plate{index % 4}" for index in range(wells)],
        }
        interval = multiway_clustered_bootstrap(
            values=values,
            cluster_labels=labels,
            seed=0,
            resample_count=FROZEN_SCIPLEX3_RESAMPLE_COUNT,
            confidence_level=FROZEN_SCIPLEX3_CONFIDENCE_LEVEL,
        )
        assert interval.resampling_scheme == RESAMPLING_SCHEME
        assert interval.resample_count == FROZEN_SCIPLEX3_RESAMPLE_COUNT
        assert interval.confidence_level == FROZEN_SCIPLEX3_CONFIDENCE_LEVEL
        assert interval.dependence_dimension_ids == tuple(
            sorted(FROZEN_SCIPLEX3_DEPENDENCE_DIMENSIONS)
        )
        assert interval.cluster_counts == (95, 4)
        assert interval.lower < interval.point_estimate < interval.upper


def test_the_frozen_artifact_still_binds_its_metrics_as_specification_only() -> None:
    """ADR 0014 decision 5: implementing a metric does not re-freeze the benchmark.

    The artifact's bindings describe a benchmark nothing has been scored against.  Flipping them
    to `executable` is `Q3`'s decision, taken when the implementations have actually run against
    the frozen partitions.  This test fails if that happens without the ADR.
    """

    artifact = json.loads(BENCHMARK_ARTIFACT.read_text())
    kinds = {
        metric["implementation_binding"]["kind"] for metric in artifact["definition"]["metrics"]
    }
    assert kinds == {"specification_only"}
