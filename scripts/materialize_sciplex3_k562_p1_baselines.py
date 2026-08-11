#!/usr/bin/env python3
"""Materialize exact sci-Plex3 K562 p1 scan and baseline-fit identities.

This command deliberately stops before held-out design or outcome access.  It authenticates and
fully consumes the immutable p1 count stream, requires the loader's post-close source receipt,
fits the six frozen p1-only baselines, and writes canonical content-addressed identity artifacts.
The outputs are software/data provenance only: they cannot issue a ``BaselineRun`` result,
lifecycle evidence, scientific admission, or a public runtime.

Materialization is exclusive and directory-atomic.  ``--check`` is intentionally cheap: it
reauthenticates only the checked-in JSON and current repository bytes and never opens the H5AD.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import asdict, fields
from pathlib import Path
from typing import cast

from cellstate.backends.sciplex3_k562 import (
    SCIPLEX3_K562_BENCHMARK_SHA256,
    PopulationComponentAccessPurpose,
)
from cellstate.backends.sciplex3_loader import (
    SCIPLEX3_SOURCE_BYTE_COUNT,
    SCIPLEX3_SOURCE_FILENAME,
    SCIPLEX3_SOURCE_MD5,
    SCIPLEX3_SOURCE_SHA256,
    SciPlex3K562H5ADLoader,
    SciPlex3P1FinalizedCountScanReceipt,
)
from cellstate.domain.common import canonical_json_bytes
from cellstate.evaluation.sciplex3_baselines import SCIPLEX3_BASELINE_IMPLEMENTATIONS
from cellstate.evaluation.sciplex3_runner import (
    SciPlex3BaselinePreparation,
    SciPlex3P1AssemblyReceipt,
    assemble_sciplex3_p1_training_data,
    fit_and_write_sciplex3_baseline,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIRECTORY = REPOSITORY_ROOT / "benchmarks/vertical-a/sciplex3-k562-24h-v1/item11-p1"
MATERIALIZATION_MANIFEST = "materialization-manifest.json"
FINALIZED_SCAN_RECEIPT = "p1-finalized-count-scan-receipt.json"
ASSEMBLY_RECEIPT = "p1-assembly-receipt.json"
MATERIALIZATION_BATCH_SIZE = 512
EXPECTED_ZERO_PANEL_RECORD_COUNT = 7
BASELINE_IDS = tuple(sorted(SCIPLEX3_BASELINE_IMPLEMENTATIONS))

_EXPECTED_BASELINE_IDS = (
    "exact-condition-negative-binomial",
    "exact-condition-rep1-empirical-resampling",
    "hierarchical-well-negative-binomial",
    "low-rank-compound-dose-response",
    "matched-vehicle-resampling",
    "nearest-supported-dose",
)
_REPOSITORY_BINDING_PATHS: Mapping[str, Path] = {
    "action_domain": Path(
        "benchmarks/vertical-a/sciplex3-k562-24h-v1/support/action-domain-mapping.json"
    ),
    "baseline_code": Path("src/cellstate/evaluation/sciplex3_baselines.py"),
    "baseline_golden_fixture": Path(
        "benchmarks/vertical-a/sciplex3-k562-24h-v1/support/baseline-golden-fixtures.json"
    ),
    "baseline_suite_specification": Path(
        "benchmarks/vertical-a/sciplex3-k562-24h-v1/support/baseline-suite-spec.json"
    ),
    "dataset_manifest": Path("data_manifests/reviewed/sciplex3-k562-24h.json"),
    "feature_panel": Path("benchmarks/artifacts/sciplex3-k562-24h-v1/feature-panel.json"),
    "loader_code": Path("src/cellstate/backends/sciplex3_loader.py"),
    "loader_contract": Path(
        "benchmarks/vertical-a/sciplex3-k562-24h-v1/support/p1-loader-contract.json"
    ),
    "materializer_code": Path("scripts/materialize_sciplex3_k562_p1_baselines.py"),
    "query": Path("benchmarks/vertical-a/sciplex3-k562-24h-v1/state-query.json"),
    "runner_code": Path("src/cellstate/evaluation/sciplex3_runner.py"),
    "scoring_transform": Path(
        "benchmarks/vertical-a/sciplex3-k562-24h-v1/support/scoring-transform.json"
    ),
    "target_value_schema": Path(
        "benchmarks/vertical-a/sciplex3-k562-24h-v1/support/target-value-schema.json"
    ),
}
_SAFETY_BOUNDARY: Mapping[str, object] = {
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


class MaterializationError(RuntimeError):
    """Raised when Item 11 outputs cannot be materialized or reauthenticated exactly."""


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _read_bytes(path: Path, *, name: str) -> bytes:
    try:
        return path.read_bytes()
    except OSError as error:
        raise MaterializationError(f"cannot read {name}: {path}") from error


def _load_canonical_object(path: Path, *, name: str) -> tuple[dict[str, object], bytes]:
    payload = _read_bytes(path, name=name)
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise MaterializationError(f"{name} is not valid JSON: {path}") from error
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise MaterializationError(f"{name} must be a JSON object: {path}")
    result = cast(dict[str, object], value)
    if canonical_json_bytes(result) != payload:
        raise MaterializationError(f"{name} is not exact canonical JSON: {path}")
    return result, payload


def _as_mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise MaterializationError(f"{name} must be an object")
    return cast(Mapping[str, object], value)


def _mutable_json_value(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _mutable_json_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_mutable_json_value(item) for item in value]
    return value


def _as_list(value: object, *, name: str) -> list[object]:
    if not isinstance(value, list):
        raise MaterializationError(f"{name} must be an array")
    return cast(list[object], value)


def _repository_reference(relative_path: Path) -> dict[str, object]:
    payload = _read_bytes(REPOSITORY_ROOT / relative_path, name=str(relative_path))
    return {
        "byte_count": len(payload),
        "relative_path": relative_path.as_posix(),
        "sha256": _sha256(payload),
    }


def _repository_bindings() -> dict[str, object]:
    return {
        name: _repository_reference(relative_path)
        for name, relative_path in sorted(_REPOSITORY_BINDING_PATHS.items())
    }


def _artifact_reference(path: Path, output_directory: Path) -> dict[str, object]:
    payload = _read_bytes(path, name="materialized artifact")
    try:
        relative_path = path.relative_to(output_directory).as_posix()
    except ValueError as error:
        raise MaterializationError("materialized artifact escaped its output directory") from error
    return {
        "byte_count": len(payload),
        "media_type": "application/json",
        "relative_path": relative_path,
        "sha256": _sha256(payload),
    }


def _write_exclusive(path: Path, value: object) -> None:
    payload = canonical_json_bytes(value)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            written = stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except OSError as error:
        raise MaterializationError(f"cannot write materialized artifact: {path}") from error
    if written != len(payload):
        raise MaterializationError(f"short write for materialized artifact: {path}")
    if _read_bytes(path, name="new materialized artifact") != payload:
        raise MaterializationError(f"materialized artifact differs on immediate re-read: {path}")


def _binding_digest(bindings: Mapping[str, object], name: str) -> str:
    reference = _as_mapping(bindings.get(name), name=f"repository binding {name}")
    digest = reference.get("sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        raise MaterializationError(f"repository binding {name} has no exact SHA-256")
    return digest


def _build_manifest(
    preparation: SciPlex3BaselinePreparation,
    output_directory: Path,
    fitted_entries: list[dict[str, object]],
    fit_runtime: Mapping[str, object],
    repository_bindings: Mapping[str, object],
) -> dict[str, object]:
    receipt = preparation.receipt
    scan = preparation.finalized_count_scan_receipt
    assembled_zero_panel_record_count = sum(
        int(well.counts.indptr[row]) == int(well.counts.indptr[row + 1])
        for well in preparation.training_data.wells
        for row in range(well.counts.row_count)
    )
    zero_panel_well_count = sum(
        int(well.counts.indptr[0]) == int(well.counts.indptr[-1])
        for well in preparation.training_data.wells
    )
    if (
        assembled_zero_panel_record_count != receipt.zero_panel_record_count
        or receipt.zero_panel_record_count != EXPECTED_ZERO_PANEL_RECORD_COUNT
        or zero_panel_well_count != 0
    ):
        raise MaterializationError(
            "assembled p1 zero-panel rows/wells differ from the exact real-source closure"
        )
    scan_reference = _artifact_reference(
        output_directory / FINALIZED_SCAN_RECEIPT, output_directory
    )
    assembly_reference = _artifact_reference(output_directory / ASSEMBLY_RECEIPT, output_directory)
    expected_bindings = {
        "action_domain": receipt.action_domain_sha256,
        "dataset_manifest": scan.dataset_manifest_sha256,
        "feature_panel": receipt.feature_panel_artifact_sha256,
        "loader_code": receipt.loader_implementation_sha256,
        "loader_contract": receipt.loader_contract_sha256,
        "query": receipt.query_sha256,
        "scoring_transform": receipt.scoring_transform_sha256,
        "target_value_schema": receipt.target_value_schema_sha256,
    }
    for name, expected_digest in expected_bindings.items():
        if _binding_digest(repository_bindings, name) != expected_digest:
            raise MaterializationError(
                f"p1 receipt differs from current repository binding: {name}"
            )
    if receipt.benchmark_sha256 != SCIPLEX3_K562_BENCHMARK_SHA256:
        raise MaterializationError("p1 receipt benchmark binding drifted")
    if scan_reference["sha256"] != scan.fingerprint:
        raise MaterializationError("persisted finalized scan differs from its receipt fingerprint")
    if assembly_reference["sha256"] != receipt.fingerprint:
        raise MaterializationError("persisted assembly differs from its receipt fingerprint")
    return {
        "artifact_schema": "sciplex3-k562-p1-baseline-materialization",
        "artifact_schema_version": "1.0.0",
        "artifacts": {
            "assembly_receipt": assembly_reference,
            "finalized_count_scan_receipt": scan_reference,
            "fitted_baselines": fitted_entries,
        },
        "exact_bindings": {
            "action_domain_sha256": receipt.action_domain_sha256,
            "baseline_code_sha256": _binding_digest(repository_bindings, "baseline_code"),
            "baseline_golden_fixture_sha256": _binding_digest(
                repository_bindings, "baseline_golden_fixture"
            ),
            "baseline_suite_specification_sha256": _binding_digest(
                repository_bindings, "baseline_suite_specification"
            ),
            "benchmark_sha256": receipt.benchmark_sha256,
            "dataset_manifest_sha256": scan.dataset_manifest_sha256,
            "feature_panel_artifact_sha256": receipt.feature_panel_artifact_sha256,
            "loader_contract_sha256": receipt.loader_contract_sha256,
            "loader_implementation_sha256": receipt.loader_implementation_sha256,
            "materializer_code_sha256": _binding_digest(repository_bindings, "materializer_code"),
            "ordered_feature_keys_sha256": receipt.ordered_feature_keys_sha256,
            "query_sha256": receipt.query_sha256,
            "runner_code_sha256": _binding_digest(repository_bindings, "runner_code"),
            "scoring_transform_sha256": receipt.scoring_transform_sha256,
            "target_value_schema_sha256": receipt.target_value_schema_sha256,
        },
        "p1_scan": {
            "assembly_fingerprint": receipt.fingerprint,
            "batch_count": receipt.batch_count,
            "control_well_count": receipt.control_well_count,
            "count_scan_complete": receipt.count_scan_complete,
            "finalized_count_scan_fingerprint": scan.fingerprint,
            "full_source_umi_total": receipt.full_source_umi_total,
            "panel_count_stream_sha256": receipt.runner_panel_count_stream_sha256,
            "panel_nonzero_count": receipt.panel_nonzero_count,
            "panel_umi_total": receipt.panel_umi_total,
            "record_count": receipt.record_count,
            "treated_well_count": receipt.treated_well_count,
            "well_count": receipt.well_count,
            "zero_panel_record_count": receipt.zero_panel_record_count,
            "zero_panel_well_count": zero_panel_well_count,
        },
        "repository_bindings": dict(repository_bindings),
        "runtime": {
            "baseline_fit": dict(fit_runtime),
            "loader": {
                "h5py_version": scan.h5py_version,
                "hdf5_version": scan.hdf5_version,
                "numpy_version": scan.numpy_version,
                "python_implementation": scan.python_implementation,
                "python_version": scan.python_version,
            },
        },
        "safety_boundary": dict(_SAFETY_BOUNDARY),
        "scope": {
            "access_purpose": "train_parameters",
            "baseline_ids": list(BASELINE_IDS),
            "batch_size": MATERIALIZATION_BATCH_SIZE,
            "feature_count": len(preparation.training_data.ordered_feature_keys),
            "partition_id": "p1-train",
        },
        "source": {
            "byte_count": scan.source_byte_count,
            "filename": SCIPLEX3_SOURCE_FILENAME,
            "md5": scan.source_md5,
            "sha256": scan.source_sha256,
        },
    }


def _check_reference(
    reference: Mapping[str, object],
    *,
    root: Path,
    name: str,
) -> tuple[Path, bytes]:
    relative = reference.get("relative_path")
    digest = reference.get("sha256")
    byte_count = reference.get("byte_count")
    if (
        not isinstance(relative, str)
        or not relative
        or not isinstance(digest, str)
        or len(digest) != 64
        or isinstance(byte_count, bool)
        or not isinstance(byte_count, int)
        or byte_count < 0
        or reference.get("media_type") != "application/json"
    ):
        raise MaterializationError(f"malformed {name} artifact reference")
    relative_path = Path(relative)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise MaterializationError(f"{name} artifact reference escapes its root")
    path = root / relative_path
    payload = _read_bytes(path, name=name)
    if len(payload) != byte_count or _sha256(payload) != digest:
        raise MaterializationError(f"content-addressed {name} artifact drifted")
    return path, payload


def _require_false(mapping: Mapping[str, object], keys: tuple[str, ...], *, name: str) -> None:
    if any(mapping.get(key) is not False for key in keys):
        raise MaterializationError(f"{name} crosses the non-admissible Item 11 boundary")


def check_materialization(output_directory: Path = DEFAULT_OUTPUT_DIRECTORY) -> str:
    """Reauthenticate checked-in Item 11 outputs without opening or hashing the source H5AD."""

    output = Path(output_directory).resolve()
    manifest, manifest_payload = _load_canonical_object(
        output / MATERIALIZATION_MANIFEST,
        name="Item 11 materialization manifest",
    )
    if (
        manifest.get("artifact_schema") != "sciplex3-k562-p1-baseline-materialization"
        or manifest.get("artifact_schema_version") != "1.0.0"
        or BASELINE_IDS != _EXPECTED_BASELINE_IDS
    ):
        raise MaterializationError("Item 11 materialization header or baseline registry drifted")

    scope = _as_mapping(manifest.get("scope"), name="materialization scope")
    if dict(scope) != {
        "access_purpose": "train_parameters",
        "baseline_ids": list(_EXPECTED_BASELINE_IDS),
        "batch_size": MATERIALIZATION_BATCH_SIZE,
        "feature_count": 2_000,
        "partition_id": "p1-train",
    }:
        raise MaterializationError("Item 11 materialization is not the exact p1-only scope")
    source = _as_mapping(manifest.get("source"), name="materialization source")
    if dict(source) != {
        "byte_count": SCIPLEX3_SOURCE_BYTE_COUNT,
        "filename": SCIPLEX3_SOURCE_FILENAME,
        "md5": SCIPLEX3_SOURCE_MD5,
        "sha256": SCIPLEX3_SOURCE_SHA256,
    }:
        raise MaterializationError("Item 11 materialization source identity drifted")
    safety = _as_mapping(manifest.get("safety_boundary"), name="materialization safety boundary")
    if dict(safety) != dict(_SAFETY_BOUNDARY):
        raise MaterializationError("Item 11 materialization safety boundary drifted")

    repository_bindings = _as_mapping(
        manifest.get("repository_bindings"), name="materialization repository bindings"
    )
    if set(repository_bindings) != set(_REPOSITORY_BINDING_PATHS):
        raise MaterializationError("Item 11 repository-binding closure is not exact")
    current_bindings = _repository_bindings()
    if dict(repository_bindings) != current_bindings:
        raise MaterializationError("Item 11 materialization is stale against repository bytes")
    exact_bindings = _as_mapping(manifest.get("exact_bindings"), name="exact bindings")
    expected_exact = {
        "action_domain_sha256": _binding_digest(current_bindings, "action_domain"),
        "baseline_code_sha256": _binding_digest(current_bindings, "baseline_code"),
        "baseline_golden_fixture_sha256": _binding_digest(
            current_bindings, "baseline_golden_fixture"
        ),
        "baseline_suite_specification_sha256": _binding_digest(
            current_bindings, "baseline_suite_specification"
        ),
        "benchmark_sha256": SCIPLEX3_K562_BENCHMARK_SHA256,
        "dataset_manifest_sha256": _binding_digest(current_bindings, "dataset_manifest"),
        "feature_panel_artifact_sha256": _binding_digest(current_bindings, "feature_panel"),
        "loader_contract_sha256": _binding_digest(current_bindings, "loader_contract"),
        "loader_implementation_sha256": _binding_digest(current_bindings, "loader_code"),
        "materializer_code_sha256": _binding_digest(current_bindings, "materializer_code"),
        "ordered_feature_keys_sha256": (
            "8b9e0e71d9bfc79a5e1db29f73bc30006bf117a29db8abf63b1607403629401f"
        ),
        "query_sha256": _binding_digest(current_bindings, "query"),
        "runner_code_sha256": _binding_digest(current_bindings, "runner_code"),
        "scoring_transform_sha256": _binding_digest(current_bindings, "scoring_transform"),
        "target_value_schema_sha256": _binding_digest(current_bindings, "target_value_schema"),
    }
    if dict(exact_bindings) != expected_exact:
        raise MaterializationError("Item 11 exact semantic/code bindings drifted")

    artifacts = _as_mapping(manifest.get("artifacts"), name="materialization artifacts")
    if set(artifacts) != {
        "assembly_receipt",
        "finalized_count_scan_receipt",
        "fitted_baselines",
    }:
        raise MaterializationError("Item 11 materialized-artifact closure is not exact")
    scan_path, scan_payload = _check_reference(
        _as_mapping(artifacts.get("finalized_count_scan_receipt"), name="finalized scan reference"),
        root=output,
        name="finalized p1 count-scan receipt",
    )
    assembly_path, assembly_payload = _check_reference(
        _as_mapping(artifacts.get("assembly_receipt"), name="assembly reference"),
        root=output,
        name="p1 assembly receipt",
    )
    scan, canonical_scan = _load_canonical_object(scan_path, name="finalized p1 count-scan receipt")
    assembly, canonical_assembly = _load_canonical_object(assembly_path, name="p1 assembly receipt")
    if scan_payload != canonical_scan or assembly_payload != canonical_assembly:
        raise MaterializationError("Item 11 receipt changed between authentication reads")
    if set(scan) != {field.name for field in fields(SciPlex3P1FinalizedCountScanReceipt)}:
        raise MaterializationError("finalized p1 count-scan receipt field closure drifted")
    if set(assembly) != {field.name for field in fields(SciPlex3P1AssemblyReceipt)}:
        raise MaterializationError("p1 assembly receipt field closure drifted")
    if (
        scan.get("artifact_schema") != "sciplex3-k562-p1-finalized-count-scan-receipt"
        or scan.get("partition_id") != "p1-train"
        or scan.get("access_purpose") != "train_parameters"
        or scan.get("accessed_partition_roles") != ["p1-train"]
        or scan.get("source_sha256") != SCIPLEX3_SOURCE_SHA256
        or scan.get("source_byte_count") != SCIPLEX3_SOURCE_BYTE_COUNT
        or scan.get("source_md5") != SCIPLEX3_SOURCE_MD5
        or scan.get("source_descriptor_identity_before")
        != scan.get("source_descriptor_identity_after")
        or scan.get("record_count") != 94_785
        or scan.get("well_count") != 768
        or scan.get("treated_well_count") != 752
        or scan.get("control_well_count") != 16
        or scan.get("accessed_count_datasets") != ["X.data", "X.indices", "X.indptr", "obs.ncounts"]
        or scan.get("exact_record_coverage") is not True
        or scan.get("count_scan_complete") is not True
        or scan.get("source_descriptor_reverified") is not True
        or scan.get("close_reverification_completed") is not True
        or scan.get("finalized") is not True
    ):
        raise MaterializationError("finalized p1 count-scan receipt scope drifted")
    _require_false(
        scan,
        (
            "heldout_memberships_parsed",
            "heldout_outcome_values_parsed",
            "trusted_workflow_receipt_present",
            "lifecycle_evidence_issued",
            "scientifically_admissible",
        ),
        name="finalized count-scan receipt",
    )
    if (
        assembly.get("partition_id") != "p1-train"
        or assembly.get("access_purpose") != "train_parameters"
        or assembly.get("source_sha256") != SCIPLEX3_SOURCE_SHA256
        or assembly.get("record_count") != 94_785
        or assembly.get("well_count") != 768
        or assembly.get("treated_well_count") != 752
        or assembly.get("control_well_count") != 16
        or assembly.get("batch_count") != scan.get("batch_count")
        or assembly.get("panel_nonzero_count") != scan.get("panel_nonzero_count")
        or assembly.get("zero_panel_record_count") != scan.get("zero_panel_record_count")
        or assembly.get("emitted_source_row_indices_sha256")
        != scan.get("emitted_source_row_indices_sha256")
        or not isinstance(scan.get("zero_panel_record_count"), int)
        or isinstance(scan.get("zero_panel_record_count"), bool)
        or cast(int, scan.get("zero_panel_record_count")) != EXPECTED_ZERO_PANEL_RECORD_COUNT
        or assembly.get("panel_umi_total") != scan.get("panel_umi_total")
        or assembly.get("full_source_umi_total") != scan.get("full_source_umi_total")
        or assembly.get("runner_panel_count_stream_sha256")
        != assembly.get("loader_panel_count_stream_sha256")
    ):
        raise MaterializationError("p1 assembly receipt closure drifted")
    _require_false(
        assembly,
        (
            "heldout_memberships_read",
            "heldout_outcomes_read",
            "can_mint_lifecycle_evidence",
            "scientifically_admissible",
        ),
        name="p1 assembly receipt",
    )

    exact_receipt_bindings = {
        "action_domain_sha256": expected_exact["action_domain_sha256"],
        "benchmark_sha256": expected_exact["benchmark_sha256"],
        "feature_panel_artifact_sha256": expected_exact["feature_panel_artifact_sha256"],
        "loader_contract_sha256": expected_exact["loader_contract_sha256"],
        "loader_implementation_sha256": expected_exact["loader_implementation_sha256"],
        "ordered_feature_keys_sha256": expected_exact["ordered_feature_keys_sha256"],
        "query_sha256": expected_exact["query_sha256"],
        "scoring_transform_sha256": expected_exact["scoring_transform_sha256"],
        "target_value_schema_sha256": expected_exact["target_value_schema_sha256"],
    }
    for field_name, expected_digest in exact_receipt_bindings.items():
        if assembly.get(field_name) != expected_digest:
            raise MaterializationError(f"p1 assembly exact binding drifted: {field_name}")
    for field_name in (
        "benchmark_sha256",
        "feature_panel_artifact_sha256",
        "loader_implementation_sha256",
        "ordered_feature_keys_sha256",
        "query_sha256",
        "scoring_transform_sha256",
        "target_value_schema_sha256",
    ):
        if scan.get(field_name) != exact_receipt_bindings[field_name]:
            raise MaterializationError(f"finalized p1 scan exact binding drifted: {field_name}")
    if (
        scan.get("p1_loader_contract_sha256") != expected_exact["loader_contract_sha256"]
        or scan.get("dataset_manifest_sha256") != expected_exact["dataset_manifest_sha256"]
    ):
        raise MaterializationError("finalized p1 scan artifact binding drifted")

    p1_scan = _as_mapping(manifest.get("p1_scan"), name="materialization p1 scan")
    expected_p1_scan = {
        "assembly_fingerprint": _sha256(assembly_payload),
        "batch_count": assembly.get("batch_count"),
        "control_well_count": assembly.get("control_well_count"),
        "count_scan_complete": True,
        "finalized_count_scan_fingerprint": _sha256(scan_payload),
        "full_source_umi_total": assembly.get("full_source_umi_total"),
        "panel_count_stream_sha256": assembly.get("runner_panel_count_stream_sha256"),
        "panel_nonzero_count": assembly.get("panel_nonzero_count"),
        "panel_umi_total": assembly.get("panel_umi_total"),
        "record_count": assembly.get("record_count"),
        "treated_well_count": assembly.get("treated_well_count"),
        "well_count": assembly.get("well_count"),
        "zero_panel_record_count": assembly.get("zero_panel_record_count"),
        "zero_panel_well_count": 0,
    }
    if (
        dict(p1_scan) != expected_p1_scan
        or assembly.get("finalized_count_scan_fingerprint") != _sha256(scan_payload)
        or p1_scan.get("panel_count_stream_sha256") != scan.get("panel_count_stream_sha256")
    ):
        raise MaterializationError("materialization index differs from exact p1 receipts")

    raw_fitted = _as_list(artifacts.get("fitted_baselines"), name="fitted baselines")
    fitted_by_id: dict[str, Mapping[str, object]] = {}
    for raw_entry in raw_fitted:
        entry = _as_mapping(raw_entry, name="fitted baseline entry")
        baseline_id = entry.get("baseline_id")
        if not isinstance(baseline_id, str) or baseline_id in fitted_by_id:
            raise MaterializationError("fitted baseline identifiers are invalid or duplicated")
        fitted_by_id[baseline_id] = entry
    if tuple(sorted(fitted_by_id)) != _EXPECTED_BASELINE_IDS:
        raise MaterializationError("Item 11 does not contain exactly six fitted baselines")

    fit_runtimes: list[Mapping[str, object]] = []
    for baseline_id in _EXPECTED_BASELINE_IDS:
        entry = fitted_by_id[baseline_id]
        expected_relative_path = f"fitted-baselines/{baseline_id}/fitted-state-manifest.json"
        if (
            set(entry)
            != {
                "baseline_id",
                "byte_count",
                "media_type",
                "relative_path",
                "sha256",
            }
            or entry.get("relative_path") != expected_relative_path
        ):
            raise MaterializationError(f"fitted baseline reference is not exact: {baseline_id}")
        fit_path, fit_payload = _check_reference(
            entry, root=output, name=f"fitted baseline {baseline_id}"
        )
        fit, canonical_fit = _load_canonical_object(fit_path, name=f"fitted baseline {baseline_id}")
        if fit_payload != canonical_fit:
            raise MaterializationError("fitted artifact changed between authentication reads")
        if set(fit) != {
            "artifact_schema",
            "artifact_schema_version",
            "baseline",
            "code",
            "executable_binding",
            "finalized_count_scan",
            "fit_partition",
            "fitted_state",
            "fitted_state_sha256",
            "input_bindings",
            "preparation_fingerprint",
            "runtime",
            "safety_boundary",
        }:
            raise MaterializationError(f"fitted baseline field closure drifted: {baseline_id}")
        baseline = _as_mapping(fit.get("baseline"), name=f"{baseline_id} identity")
        fitted_state = _as_mapping(fit.get("fitted_state"), name=f"{baseline_id} fitted state")
        if (
            entry.get("baseline_id") != baseline_id
            or baseline.get("baseline_id") != baseline_id
            or fit.get("artifact_schema") != "sciplex3-k562-p1-baseline-fitted-state"
            or fit.get("preparation_fingerprint") != _sha256(assembly_payload)
            or fit.get("input_bindings") != assembly
            or fit.get("finalized_count_scan") != scan
            or fitted_state.get("baseline_id") != baseline_id
            or fit.get("fitted_state_sha256") != _sha256(canonical_json_bytes(fitted_state))
        ):
            raise MaterializationError(f"fitted baseline binding drifted: {baseline_id}")
        safety_boundary = _as_mapping(
            fit.get("safety_boundary"), name=f"{baseline_id} safety boundary"
        )
        if set(safety_boundary) != {
            "baseline_run_status_issued",
            "can_mint_lifecycle_evidence",
            "heldout_memberships_read",
            "heldout_outcomes_read",
            "metric_results_issued",
            "scientifically_admissible",
            "trusted_workflow_receipt_issued",
        }:
            raise MaterializationError(f"fitted safety-boundary fields drifted: {baseline_id}")
        _require_false(
            safety_boundary,
            (
                "baseline_run_status_issued",
                "can_mint_lifecycle_evidence",
                "heldout_memberships_read",
                "heldout_outcomes_read",
                "metric_results_issued",
                "scientifically_admissible",
                "trusted_workflow_receipt_issued",
            ),
            name=f"fitted baseline {baseline_id}",
        )
        code = _as_mapping(fit.get("code"), name=f"{baseline_id} code")
        baseline_code = _as_mapping(code.get("baseline"), name=f"{baseline_id} baseline code")
        runner_code = _as_mapping(code.get("runner"), name=f"{baseline_id} runner code")
        if (
            baseline_code.get("sha256") != expected_exact["baseline_code_sha256"]
            or runner_code.get("sha256") != expected_exact["runner_code_sha256"]
        ):
            raise MaterializationError(f"fitted baseline code binding drifted: {baseline_id}")
        executable = _as_mapping(
            fit.get("executable_binding"), name=f"{baseline_id} executable binding"
        )
        if set(executable) != {
            "baseline_suite_specification",
            "golden_fixture",
            "implementation_code",
            "runner_code",
        }:
            raise MaterializationError(f"fitted executable-binding closure drifted: {baseline_id}")
        for key, expected_digest in (
            ("baseline_suite_specification", expected_exact["baseline_suite_specification_sha256"]),
            ("golden_fixture", expected_exact["baseline_golden_fixture_sha256"]),
            ("implementation_code", expected_exact["baseline_code_sha256"]),
            ("runner_code", expected_exact["runner_code_sha256"]),
        ):
            if (
                _as_mapping(executable.get(key), name=f"{baseline_id} {key}").get("sha256")
                != expected_digest
            ):
                raise MaterializationError(f"fitted executable binding drifted: {baseline_id}")
        fit_runtimes.append(_as_mapping(fit.get("runtime"), name=f"{baseline_id} runtime"))

    if not fit_runtimes or any(runtime != fit_runtimes[0] for runtime in fit_runtimes[1:]):
        raise MaterializationError("fitted baselines do not share one exact runtime identity")
    runtime = _as_mapping(manifest.get("runtime"), name="materialization runtime")
    if runtime.get("baseline_fit") != fit_runtimes[0]:
        raise MaterializationError("materialization runtime differs from fitted-state artifacts")
    loader_runtime = _as_mapping(runtime.get("loader"), name="materialization loader runtime")
    if dict(loader_runtime) != {
        "h5py_version": scan.get("h5py_version"),
        "hdf5_version": scan.get("hdf5_version"),
        "numpy_version": scan.get("numpy_version"),
        "python_implementation": scan.get("python_implementation"),
        "python_version": scan.get("python_version"),
    }:
        raise MaterializationError("materialization loader runtime differs from finalized scan")
    return _sha256(manifest_payload)


def materialize(
    source_h5ad: Path,
    output_directory: Path = DEFAULT_OUTPUT_DIRECTORY,
) -> str:
    """Execute one exact real p1 scan and write the six frozen fitted-state identities."""

    if BASELINE_IDS != _EXPECTED_BASELINE_IDS:
        raise MaterializationError("frozen sci-Plex3 baseline registry is not the exact six")
    source = Path(source_h5ad).resolve()
    output = Path(output_directory).resolve()
    if output.exists():
        raise MaterializationError(f"refusing to overwrite existing output directory: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output.parent / f".{output.name}.materialization.lock"
    try:
        lock_descriptor = os.open(lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except OSError as error:
        raise MaterializationError(
            f"cannot acquire exclusive materialization lock: {lock_path}"
        ) from error

    temporary: Path | None = None
    try:
        repository_bindings_before = _repository_bindings()
        if output.exists():
            raise MaterializationError(f"refusing to overwrite existing output directory: {output}")
        temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
        loader = SciPlex3K562H5ADLoader.open_for_purpose(
            source,
            REPOSITORY_ROOT,
            access_purpose=PopulationComponentAccessPurpose.TRAIN_PARAMETERS,
            partition_id="p1-train",
        )
        try:
            preparation = assemble_sciplex3_p1_training_data(
                loader,
                REPOSITORY_ROOT,
                batch_size=MATERIALIZATION_BATCH_SIZE,
            )
        finally:
            loader.close()

        _write_exclusive(
            temporary / FINALIZED_SCAN_RECEIPT,
            preparation.finalized_count_scan_manifest(),
        )
        _write_exclusive(temporary / ASSEMBLY_RECEIPT, asdict(preparation.receipt))

        fitted_entries: list[dict[str, object]] = []
        fit_runtime: Mapping[str, object] | None = None
        for baseline_id in BASELINE_IDS:
            fitted = fit_and_write_sciplex3_baseline(
                preparation,
                baseline_id,
                temporary / "fitted-baselines" / baseline_id,
            )
            artifact_reference = _artifact_reference(fitted.artifact.path, temporary)
            fitted_entry = {"baseline_id": baseline_id, **artifact_reference}
            fitted_entries.append(fitted_entry)
            current_runtime = _as_mapping(
                fitted.artifact_manifest.get("runtime"), name=f"{baseline_id} runtime"
            )
            current_runtime = cast(Mapping[str, object], _mutable_json_value(current_runtime))
            if fit_runtime is None:
                fit_runtime = current_runtime
            elif current_runtime != fit_runtime:
                raise MaterializationError("baseline fits did not share one exact runtime identity")
        if fit_runtime is None:
            raise MaterializationError("no frozen sci-Plex3 baselines were fitted")

        repository_bindings_after = _repository_bindings()
        if repository_bindings_after != repository_bindings_before:
            raise MaterializationError("repository bytes changed during p1 materialization")
        manifest = _build_manifest(
            preparation,
            temporary,
            fitted_entries,
            fit_runtime,
            repository_bindings_after,
        )
        _write_exclusive(temporary / MATERIALIZATION_MANIFEST, manifest)
        fingerprint = check_materialization(temporary)
        if output.exists():
            raise MaterializationError(
                f"refusing to overwrite output created concurrently: {output}"
            )
        temporary.rename(output)
        temporary = None
        return fingerprint
    finally:
        if temporary is not None:
            shutil.rmtree(temporary, ignore_errors=True)
        os.close(lock_descriptor)
        with suppress(FileNotFoundError):
            lock_path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--source-h5ad", type=Path)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    args = parser.parse_args()
    if args.check:
        if args.source_h5ad is not None:
            parser.error("--check never accepts or opens --source-h5ad")
        fingerprint = check_materialization(args.output_directory)
        print(f"item11_p1_materialization_sha256 {fingerprint}")
        return
    if args.source_h5ad is None:
        parser.error("--source-h5ad is required unless --check is used")
    fingerprint = materialize(
        args.source_h5ad,
        args.output_directory,
    )
    print(f"item11_p1_materialization_sha256 {fingerprint}")


if __name__ == "__main__":
    main()
