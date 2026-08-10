from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from cellstate.backends import BiologicalModelBundleContract, BundleContractKind
from cellstate.data import (
    BaselineRunStatus,
    BenchmarkAdmissionStatus,
    BenchmarkArtifact,
    ExecutableImplementationBinding,
    SpecificationOnlyImplementationBinding,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PREPARATION_DIRECTORY = REPOSITORY_ROOT / "benchmarks/artifacts/sciplex3-k562-24h-v1"
BENCHMARK_DIRECTORY = REPOSITORY_ROOT / "benchmarks/vertical-a/sciplex3-k562-24h-v1"
P1_CONTRACT_PATH = BENCHMARK_DIRECTORY / "support/p1-loader-contract.json"
BASELINE_GOLDEN_PATH = BENCHMARK_DIRECTORY / "support/baseline-golden-fixtures.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_p1_loader_contract_is_an_exact_train_only_closure() -> None:
    contract = json.loads(P1_CONTRACT_PATH.read_bytes())
    assert contract["artifact_schema"] == "sciplex3-k562-p1-loader-contract"
    assert contract["artifact_schema_version"] == "1.0.0"
    assert contract["heldout_memberships_referenced"] is False
    assert contract["loader_outputs_can_mint_lifecycle_evidence"] is False
    assert contract["scientifically_admissible_without_trusted_workflow_receipt"] is False
    assert contract["partition"] == {
        "access_purpose": "train_parameters",
        "artifact_role": "train",
        "partition_id": "p1-train",
        "record_count": 94_785,
        "selector": {
            "cell_line": "K562",
            "plates": [f"plate{index}" for index in range(1, 9)],
            "replicate": "rep1",
        },
        "well_count": 768,
    }
    assert set(contract["artifacts"]) == {
        "feature_panel",
        "plate_ids",
        "record_ids",
        "record_to_well",
        "source_verification",
        "well_ids",
        "well_to_condition",
    }
    assert not any(
        heldout in reference["relative_path"]
        for reference in contract["artifacts"].values()
        for heldout in ("calibration", "model_selection", "untouched_test")
    )

    for name, reference in contract["artifacts"].items():
        relative_path = reference["relative_path"]
        path = (
            REPOSITORY_ROOT / relative_path
            if relative_path.startswith("benchmarks/")
            else PREPARATION_DIRECTORY / relative_path
        )
        assert path.is_file(), name
        assert path.stat().st_size == reference["byte_count"], name
        assert _sha256(path) == reference["sha256"], name


def test_p1_loader_contract_binds_current_scientific_artifacts() -> None:
    contract = json.loads(P1_CONTRACT_PATH.read_bytes())
    bindings = contract["bindings"]
    assert bindings == {
        "benchmark_sha256": _sha256(BENCHMARK_DIRECTORY / "benchmark-artifact.json"),
        "dataset_manifest_sha256": _sha256(
            REPOSITORY_ROOT / "data_manifests/reviewed/sciplex3-k562-24h.json"
        ),
        "query_sha256": _sha256(BENCHMARK_DIRECTORY / "state-query.json"),
        "scoring_transform_sha256": _sha256(BENCHMARK_DIRECTORY / "support/scoring-transform.json"),
        "target_value_schema_sha256": _sha256(
            BENCHMARK_DIRECTORY / "support/target-value-schema.json"
        ),
    }


def test_baseline_golden_fixture_covers_every_executable_algorithm_without_admission() -> None:
    fixture = json.loads(BASELINE_GOLDEN_PATH.read_bytes())
    assert fixture["artifact_schema"] == "sciplex3-k562-baseline-golden-fixtures"
    assert fixture["artifact_schema_version"] == "1.0.0"
    assert fixture["biological_performance_evidence"] is False
    assert fixture["scientific_admission_authorized"] is False
    assert fixture["production_sampling_contract"] == {
        "rng_algorithm": "numpy-pcg64dxsm-v1",
        "samples_per_case_per_seed": 512,
        "seeds": [0, 1, 2, 3, 4],
    }
    assert {result["baseline_id"] for result in fixture["results"]} == {
        "exact-condition-negative-binomial",
        "exact-condition-rep1-empirical-resampling",
        "hierarchical-well-negative-binomial",
        "low-rank-compound-dose-response",
        "matched-vehicle-resampling",
        "nearest-supported-dose",
    }
    for result in fixture["results"]:
        assert len(result["fitted_state_manifest_sha256"]) == 64
        assert result["fitted_state_manifest_byte_count"] > 0
        assert [sample["seed"] for sample in result["samples"]] == [0, 1, 2, 3, 4] * 2
        assert [sample["condition"]["kind"] for sample in result["samples"]] == [
            *("compound_dose" for _ in range(5)),
            *("no_action" for _ in range(5)),
        ]
        for sample in result["samples"]:
            assert sample["shape"] == [512, 2_000]
            assert sample["dtype"] == "little-endian-int64"
            assert len(sample["samples_sha256"]) == 64
            assert len(sample["panel_totals_sha256"]) == 64


@pytest.mark.skipif(
    sys.version_info[:2] != (3, 11),
    reason="exact stochastic golden currentness is bound to the Python 3.11 CI leg",
)
def test_item11_generated_artifacts_are_current() -> None:
    assert np.__version__ == "2.4.6"
    environment = {**os.environ, "PYTHONPATH": str(REPOSITORY_ROOT / "src")}
    subprocess.run(
        [
            sys.executable,
            str(REPOSITORY_ROOT / "scripts/build_sciplex3_k562_item11_artifacts.py"),
            "--check",
        ],
        cwd=REPOSITORY_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )


def test_item11_software_does_not_advance_benchmark_or_bundle_lifecycle() -> None:
    benchmark = BenchmarkArtifact.model_validate_json(
        (BENCHMARK_DIRECTORY / "benchmark-artifact.json").read_bytes()
    )
    bundle = BiologicalModelBundleContract.model_validate_json(
        (
            REPOSITORY_ROOT / "backends/vertical-a/sciplex3-k562-24h-v1/bundle-contract.json"
        ).read_bytes()
    )
    assert benchmark.admission.status is BenchmarkAdmissionStatus.COMPONENT_BENCHMARK
    baseline_by_id = {baseline.baseline_id: baseline for baseline in benchmark.definition.baselines}
    executable_ids = {
        "exact-condition-negative-binomial",
        "exact-condition-rep1-empirical-resampling",
        "hierarchical-well-negative-binomial",
        "low-rank-compound-dose-response",
        "matched-vehicle-resampling",
        "nearest-supported-dose",
    }
    assert all(
        isinstance(
            baseline_by_id[baseline_id].implementation_binding, ExecutableImplementationBinding
        )
        for baseline_id in executable_ids
    )
    assert all(
        isinstance(
            baseline_by_id[baseline_id].implementation_binding,
            SpecificationOnlyImplementationBinding,
        )
        for baseline_id in {"persistence", "temporal-state-space"}
    )
    assert all(
        run.status in {BaselineRunStatus.NOT_RUN, BaselineRunStatus.NOT_APPLICABLE}
        for run in benchmark.admission.baseline_runs
    )
    assert bundle.bundle_kind is BundleContractKind.COMPONENT_SCAFFOLD
    assert bundle.model_artifact is None
    assert bundle.training_run is None
    assert bundle.validation_evidence == ()
    assert bundle.operation_implementations == ()
