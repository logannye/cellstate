from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

from cellstate.backends.sciplex3_k562 import SCIPLEX3_K562_BENCHMARK_SHA256
from cellstate.domain.common import canonical_json_bytes

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIRECTORY = REPOSITORY_ROOT / "benchmarks/vertical-a/sciplex3-k562-24h-v1/item11-p1"
BENCHMARK_DIRECTORY = "benchmarks/vertical-a/sciplex3-k562-24h-v1"
MANIFEST_PATH = OUTPUT_DIRECTORY / "materialization-manifest.json"
EXPECTED_BASELINE_IDS = {
    "exact-condition-negative-binomial",
    "exact-condition-rep1-empirical-resampling",
    "hierarchical-well-negative-binomial",
    "low-rank-compound-dose-response",
    "matched-vehicle-resampling",
    "nearest-supported-dose",
}
SOURCE_SHA256 = "603ed16c5e25401c8a7f5bb0b2b045179701017d65dcfc6aeea71722a66cd10a"


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _load_canonical(path: Path) -> tuple[dict[str, Any], bytes]:
    payload = path.read_bytes()
    value = json.loads(payload)
    assert isinstance(value, dict)
    assert canonical_json_bytes(value) == payload
    return cast(dict[str, Any], value), payload


def _resolve(reference: Mapping[str, Any]) -> tuple[Path, bytes]:
    relative = Path(reference["relative_path"])
    assert not relative.is_absolute()
    assert ".." not in relative.parts
    path = OUTPUT_DIRECTORY / relative
    payload = path.read_bytes()
    assert len(payload) == reference["byte_count"]
    assert _sha256(payload) == reference["sha256"]
    assert reference["media_type"] == "application/json"
    return path, payload


def _assert_no_authority(value: Mapping[str, Any], keys: tuple[str, ...]) -> None:
    for key in keys:
        assert value[key] is False


def test_real_p1_materialization_is_canonical_content_addressed_and_current() -> None:
    manifest, _ = _load_canonical(MANIFEST_PATH)
    assert manifest["artifact_schema"] == "sciplex3-k562-p1-baseline-materialization"
    assert manifest["artifact_schema_version"] == "1.0.0"
    assert manifest["scope"] == {
        "access_purpose": "train_parameters",
        "baseline_ids": sorted(EXPECTED_BASELINE_IDS),
        "batch_size": 512,
        "feature_count": 2_000,
        "partition_id": "p1-train",
    }
    assert manifest["source"] == {
        "byte_count": 2_526_631_614,
        "filename": "SrivatsanTrapnell2020_sciplex3.h5ad",
        "md5": "c9e70629505d98c7ca1a837f62b14e89",
        "sha256": SOURCE_SHA256,
    }
    assert manifest["exact_bindings"]["benchmark_sha256"] == SCIPLEX3_K562_BENCHMARK_SHA256

    expected_bindings = {
        "action_domain": f"{BENCHMARK_DIRECTORY}/support/action-domain-mapping.json",
        "baseline_code": "src/cellstate/evaluation/sciplex3_baselines.py",
        "baseline_golden_fixture": f"{BENCHMARK_DIRECTORY}/support/baseline-golden-fixtures.json",
        "baseline_suite_specification": (f"{BENCHMARK_DIRECTORY}/support/baseline-suite-spec.json"),
        "dataset_manifest": "data_manifests/reviewed/sciplex3-k562-24h.json",
        "feature_panel": "benchmarks/artifacts/sciplex3-k562-24h-v1/feature-panel.json",
        "loader_code": "src/cellstate/backends/sciplex3_loader.py",
        "loader_contract": f"{BENCHMARK_DIRECTORY}/support/p1-loader-contract.json",
        "materializer_code": "scripts/materialize_sciplex3_k562_p1_baselines.py",
        "query": f"{BENCHMARK_DIRECTORY}/state-query.json",
        "runner_code": "src/cellstate/evaluation/sciplex3_runner.py",
        "scoring_transform": f"{BENCHMARK_DIRECTORY}/support/scoring-transform.json",
        "target_value_schema": f"{BENCHMARK_DIRECTORY}/support/target-value-schema.json",
    }
    assert set(manifest["repository_bindings"]) == set(expected_bindings)
    for name, relative_path in expected_bindings.items():
        reference = manifest["repository_bindings"][name]
        assert reference["relative_path"] == relative_path
        payload = (REPOSITORY_ROOT / relative_path).read_bytes()
        assert reference["byte_count"] == len(payload)
        assert reference["sha256"] == _sha256(payload)


def test_real_p1_scan_proves_exact_close_reverified_counts_without_heldout_access() -> None:
    manifest, _ = _load_canonical(MANIFEST_PATH)
    artifacts = manifest["artifacts"]
    scan_path, scan_payload = _resolve(artifacts["finalized_count_scan_receipt"])
    assembly_path, assembly_payload = _resolve(artifacts["assembly_receipt"])
    scan, _ = _load_canonical(scan_path)
    assembly, _ = _load_canonical(assembly_path)

    assert scan["artifact_schema"] == "sciplex3-k562-p1-finalized-count-scan-receipt"
    assert scan["artifact_schema_version"] == "1.0.0"
    assert scan["partition_id"] == assembly["partition_id"] == "p1-train"
    assert scan["access_purpose"] == assembly["access_purpose"] == "train_parameters"
    assert scan["accessed_partition_roles"] == ["p1-train"]
    assert scan["accessed_count_datasets"] == ["X.data", "X.indices", "X.indptr", "obs.ncounts"]
    assert scan["source_sha256"] == assembly["source_sha256"] == SOURCE_SHA256
    assert scan["source_descriptor_identity_before"] == scan["source_descriptor_identity_after"]
    assert scan["record_count"] == assembly["record_count"] == 94_785
    assert scan["well_count"] == assembly["well_count"] == 768
    assert scan["treated_well_count"] == assembly["treated_well_count"] == 752
    assert scan["control_well_count"] == assembly["control_well_count"] == 16
    assert scan["batch_count"] == assembly["batch_count"] > 0
    assert scan["panel_nonzero_count"] == assembly["panel_nonzero_count"] >= 0
    assert scan["zero_panel_record_count"] == assembly["zero_panel_record_count"]
    assert scan["zero_panel_record_count"] == 7
    assert manifest["p1_scan"]["zero_panel_well_count"] == 0
    assert scan["panel_umi_total"] == assembly["panel_umi_total"] > 0
    assert scan["full_source_umi_total"] == assembly["full_source_umi_total"]
    assert scan["full_source_umi_total"] >= scan["panel_umi_total"]
    assert scan["panel_count_stream_sha256"] == assembly["runner_panel_count_stream_sha256"]
    assert scan["panel_count_stream_sha256"] == assembly["loader_panel_count_stream_sha256"]
    assert (
        scan["emitted_source_row_indices_sha256"] == assembly["emitted_source_row_indices_sha256"]
    )
    assert scan["count_scan_complete"] is True
    assert scan["close_reverification_completed"] is True
    assert scan["source_descriptor_reverified"] is True
    assert scan["exact_record_coverage"] is True
    assert scan["finalized"] is True
    assert assembly["exact_record_coverage"] is True
    assert assembly["count_scan_complete"] is True
    assert assembly["close_reverification_completed"] is True
    assert assembly["finalized_count_scan_fingerprint"] == _sha256(scan_payload)
    assert manifest["p1_scan"]["finalized_count_scan_fingerprint"] == _sha256(scan_payload)
    assert manifest["p1_scan"]["assembly_fingerprint"] == _sha256(assembly_payload)
    _assert_no_authority(
        scan,
        (
            "heldout_memberships_parsed",
            "heldout_outcome_values_parsed",
            "trusted_workflow_receipt_present",
            "lifecycle_evidence_issued",
            "scientifically_admissible",
        ),
    )
    _assert_no_authority(
        assembly,
        (
            "heldout_memberships_read",
            "heldout_outcomes_read",
            "can_mint_lifecycle_evidence",
            "scientifically_admissible",
        ),
    )


def test_real_p1_materialization_contains_all_six_exact_fits_and_no_authority() -> None:
    manifest, _ = _load_canonical(MANIFEST_PATH)
    artifacts = manifest["artifacts"]
    scan_path, _ = _resolve(artifacts["finalized_count_scan_receipt"])
    assembly_path, assembly_payload = _resolve(artifacts["assembly_receipt"])
    scan, _ = _load_canonical(scan_path)
    assembly, _ = _load_canonical(assembly_path)
    fitted_entries = artifacts["fitted_baselines"]
    assert len(fitted_entries) == 6
    assert {entry["baseline_id"] for entry in fitted_entries} == EXPECTED_BASELINE_IDS

    fit_runtimes: list[dict[str, Any]] = []
    exact = manifest["exact_bindings"]
    for entry in fitted_entries:
        fit_path, fit_payload = _resolve(entry)
        fit, canonical_payload = _load_canonical(fit_path)
        assert fit_payload == canonical_payload
        baseline_id = entry["baseline_id"]
        assert fit["artifact_schema"] == "sciplex3-k562-p1-baseline-fitted-state"
        assert fit["artifact_schema_version"] == "1.0.0"
        assert fit["baseline"]["baseline_id"] == baseline_id
        assert fit["fit_partition"] == {
            "access_purpose": "train_parameters",
            "control_well_count": 16,
            "partition_id": "p1-train",
            "record_count": 94_785,
            "treated_well_count": 752,
            "well_count": 768,
        }
        assert fit["preparation_fingerprint"] == _sha256(assembly_payload)
        assert fit["input_bindings"] == assembly
        assert fit["finalized_count_scan"] == scan
        assert fit["fitted_state"]["baseline_id"] == baseline_id
        assert _sha256(canonical_json_bytes(fit["fitted_state"])) == fit["fitted_state_sha256"]
        assert fit["code"]["baseline"]["sha256"] == exact["baseline_code_sha256"]
        assert fit["code"]["runner"]["sha256"] == exact["runner_code_sha256"]
        assert (
            fit["executable_binding"]["golden_fixture"]["sha256"]
            == exact["baseline_golden_fixture_sha256"]
        )
        assert (
            fit["executable_binding"]["baseline_suite_specification"]["sha256"]
            == exact["baseline_suite_specification_sha256"]
        )
        assert fit["executable_binding"]["runner_code"]["sha256"] == exact["runner_code_sha256"]
        _assert_no_authority(
            fit["safety_boundary"],
            (
                "baseline_run_status_issued",
                "can_mint_lifecycle_evidence",
                "heldout_memberships_read",
                "heldout_outcomes_read",
                "metric_results_issued",
                "scientifically_admissible",
                "trusted_workflow_receipt_issued",
            ),
        )
        fit_runtimes.append(fit["runtime"])
    assert all(runtime == fit_runtimes[0] for runtime in fit_runtimes)
    assert manifest["runtime"]["baseline_fit"] == fit_runtimes[0]

    assert manifest["safety_boundary"] == {
        "accessed_partition_roles": ["p1-train"],
        "baseline_run_status_issued": False,
        "can_mint_lifecycle_evidence": False,
        "heldout_memberships_read": False,
        "heldout_outcomes_read": False,
        "lifecycle_evidence_issued": False,
        "metric_results_issued": False,
        "p2_calibration_accessed": False,
        "p3_model_selection_accessed": False,
        "p4_untouched_test_accessed": False,
        "public_runtime_registered": False,
        "scientifically_admissible": False,
        "trusted_workflow_receipt_issued": False,
    }


def test_real_p1_materialization_cheap_check_does_not_require_the_source() -> None:
    environment = {**os.environ, "PYTHONPATH": str(REPOSITORY_ROOT / "src")}
    result = subprocess.run(
        [
            sys.executable,
            str(REPOSITORY_ROOT / "scripts/materialize_sciplex3_k562_p1_baselines.py"),
            "--check",
        ],
        cwd=REPOSITORY_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.startswith("item11_p1_materialization_sha256 ")
