"""Software-only synthetic P0 tests for the purpose-bound sci-Plex3 H5AD loader.

These tests do not constitute a real-source smoke test or scientific execution evidence.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import cellstate.backends.sciplex3_loader as loader_module
from cellstate.backends.sciplex3_k562 import PopulationComponentAccessPurpose
from cellstate.backends.sciplex3_loader import (
    SciPlex3FeaturePanel,
    SciPlex3K562H5ADLoader,
    SciPlex3SparseCountBatch,
    SciPlex3TrainingDataLoader,
)
from cellstate.data.benchmarks import BenchmarkPartitionRole
from cellstate.errors import ContractViolationError

ROOT = Path(__file__).resolve().parents[1]


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def test_feature_selection_row_membership_uses_frozen_lexicographic_encoding() -> None:
    rows = np.asarray([2, 10, 100], dtype=np.int64)
    payload = loader_module._decimal_string_membership_payload(rows)
    assert payload == b'["10","100","2"]'
    assert payload != _canonical(["2", "10", "100"])


class _FakeDataset:
    def __init__(self, values: np.ndarray[Any, Any], *, attrs: dict[str, object] | None = None):
        self.values = values
        self.attrs = attrs or {}
        self.reads = 0
        self.read_keys: list[object] = []

    @property
    def dtype(self) -> np.dtype[Any]:
        return self.values.dtype

    def __len__(self) -> int:
        return len(self.values)

    def __getitem__(self, key: object) -> np.ndarray[Any, Any]:
        self.reads += 1
        self.read_keys.append(key)
        return self.values[key]


class _FakeGroup:
    def __init__(
        self,
        children: dict[str, object],
        *,
        attrs: dict[str, object] | None = None,
    ) -> None:
        self.children = children
        self.attrs = attrs or {}

    def keys(self) -> Any:
        return self.children.keys()

    def __getitem__(self, key: str) -> Any:
        return self.children[key]


class _FakeH5AD(_FakeGroup):
    def __init__(self, children: dict[str, object]) -> None:
        super().__init__(children)
        self.closed = False

    def close(self) -> None:
        self.closed = True


class _FakeH5Py:
    __version__ = "3.16.0-synthetic"

    class version:
        hdf5_version = "2.0.0-synthetic"

    def __init__(self, h5ad: _FakeH5AD) -> None:
        self.h5ad = h5ad
        self.stream: object | None = None
        self.mode: str | None = None
        self.open_count = 0

    def File(self, stream: object, mode: str) -> _FakeH5AD:
        self.stream = stream
        self.mode = mode
        self.open_count += 1
        return self.h5ad


@dataclass(frozen=True)
class _SyntheticLoaderEnvironment:
    source: Path
    repository_root: Path
    h5py: _FakeH5Py
    data: _FakeDataset
    indices: _FakeDataset
    indptr: _FakeDataset
    ncounts: _FakeDataset
    feature_panel_path: Path


def test_checked_in_preparation_registry_and_ordered_panel_are_exact() -> None:
    artifact_root = ROOT / "benchmarks/artifacts/sciplex3-k562-24h-v1"
    contract_path = (
        ROOT / "benchmarks/vertical-a/sciplex3-k562-24h-v1/support/p1-loader-contract.json"
    )
    assert _sha256(contract_path.read_bytes()) == loader_module.SCIPLEX3_P1_LOADER_CONTRACT_SHA256
    source = json.loads((artifact_root / "source-verification.json").read_bytes())
    assert source["source"]["byte_count"] == loader_module.SCIPLEX3_SOURCE_BYTE_COUNT
    assert source["source"]["md5"] == loader_module.SCIPLEX3_SOURCE_MD5
    assert source["source"]["sha256"] == loader_module.SCIPLEX3_SOURCE_SHA256

    panel = json.loads((artifact_root / "feature-panel.json").read_bytes())
    ordered_indices = tuple(item["source_feature_index"] for item in panel["features"])
    ordered_keys = [f"{item['ensembl_id']}|{item['gene_symbol']}" for item in panel["features"]]
    assert len(ordered_indices) == loader_module.SCIPLEX3_FEATURE_COUNT
    assert len(set(ordered_indices)) == loader_module.SCIPLEX3_FEATURE_COUNT
    assert _sha256(_canonical(ordered_keys)) == (loader_module.SCIPLEX3_ORDERED_FEATURE_KEYS_SHA256)

    descriptor = SciPlex3K562H5ADLoader.training_partition_descriptor(ROOT)
    assert descriptor.partition_id == "p1-train"
    assert descriptor.record_count == 94_785
    assert descriptor.well_count == 768
    assert descriptor.loader_contract_sha256 == loader_module.SCIPLEX3_P1_LOADER_CONTRACT_SHA256


def _write_json(path: Path, value: object, *, canonical: bool = False) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        _canonical(value) if canonical else json.dumps(value, indent=2, sort_keys=True).encode()
    )
    path.write_bytes(payload)
    return payload


@pytest.fixture
def synthetic_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> _SyntheticLoaderEnvironment:
    source_payload = b"exact-synthetic-h5ad"
    source_sha256 = _sha256(source_payload)
    source_md5 = hashlib.md5(source_payload, usedforsecurity=False).hexdigest()
    source = tmp_path / "SrivatsanTrapnell2020_sciplex3.h5ad"
    source.write_bytes(source_payload)

    record_ids = ["r0", "r1", "r2", "r3", "r4"]
    plates = ["plate1", "plate1", "plate25", "plate27", "plate29"]
    wells = ["A1", "A2", "B1", "C1", "D1"]
    replicates = ["rep1", "rep1", "rep2", "rep2", "rep2"]
    perturbations = ["drug-a", "control", "drug-b", "drug-c", "drug-d"]
    doses = [10, 0, 100, 1_000, 10_000]
    roles = [
        "train",
        "train",
        "calibration",
        "model_selection_validation",
        "untouched_test",
    ]
    partition_ids = {
        "train": "p1-train",
        "calibration": "p2-calibration",
        "model_selection_validation": "p3-model-selection-validation",
        "untouched_test": "p4-untouched-test",
    }
    conditions = [
        "source-label:drug-a@10nM",
        "source-control@0nM",
        "source-label:drug-b@100nM",
        "source-label:drug-c@1000nM",
        "source-label:drug-d@10000nM",
    ]
    composite_wells = [
        _canonical([plate, well]).decode() for plate, well in zip(plates, wells, strict=True)
    ]

    # Five raw rows by three source features.  The ordered panel is [source 2, source 0].
    dense = np.asarray(
        [
            [1, 2, 0],
            [0, 1, 3],
            [2, 0, 1],
            [1, 1, 1],
            [4, 0, 2],
        ],
        dtype=np.int64,
    )
    csr_data: list[int] = []
    csr_indices: list[int] = []
    indptr = [0]
    for row in dense:
        for index in np.flatnonzero(row):
            csr_indices.append(int(index))
            csr_data.append(int(row[index]))
        indptr.append(len(csr_data))
    data_node = _FakeDataset(np.asarray(csr_data, dtype=np.int64))
    index_node = _FakeDataset(np.asarray(csr_indices, dtype=np.int32))
    indptr_node = _FakeDataset(np.asarray(indptr, dtype=np.int64))
    matrix = _FakeGroup(
        {"data": data_node, "indices": index_node, "indptr": indptr_node},
        attrs={"shape": [5, 3], "encoding-type": "csr_matrix"},
    )

    def strings(values: list[str]) -> _FakeDataset:
        return _FakeDataset(np.asarray([value.encode() for value in values], dtype=object))

    ncounts_node = _FakeDataset(dense.sum(axis=1).astype(np.float64))
    obs = _FakeGroup(
        {
            "_index": strings(record_ids),
            "cell_line": strings(["K562"] * 5),
            "dose_unit": strings(["nM"] * 5),
            "dose_value": _FakeDataset(np.asarray(doses, dtype=np.float64)),
            "ncounts": ncounts_node,
            "perturbation": strings(perturbations),
            "plate": strings(plates),
            "replicate": strings(replicates),
            "time": _FakeDataset(np.full(5, 24.0, dtype=np.float64)),
            "well": strings(wells),
        }
    )
    ensembl_ids = ["ENSG0", "ENSG1", "ENSG2"]
    gene_symbols = ["G0", "G1", "G2"]
    var = _FakeGroup(
        {
            "ensembl_id": strings(ensembl_ids),
            "gene_symbol": strings(gene_symbols),
        }
    )
    fake_h5ad = _FakeH5AD({"X": matrix, "obs": obs, "var": var})
    fake_h5py = _FakeH5Py(fake_h5ad)

    artifact_root = tmp_path / "benchmarks" / "artifacts" / "sciplex3-k562-24h-v1"
    artifacts: dict[str, bytes] = {}

    feature_axis = [
        [index, ensembl, symbol]
        for index, (ensembl, symbol) in enumerate(zip(ensembl_ids, gene_symbols, strict=True))
    ]
    feature_axis_sha256 = _sha256(_canonical(feature_axis))
    feature_keys = ["ENSG2|G2", "ENSG0|G0"]
    feature_keys_sha256 = _sha256(_canonical(feature_keys))
    p1_source_rows_payload = _canonical(["0", "1"])
    source_verification = {
        "artifact_schema": "sciplex3-k562-source-verification",
        "source": {
            "filename": source.name,
            "byte_count": len(source_payload),
            "md5": source_md5,
            "sha256": source_sha256,
        },
        "h5ad_structure": {
            "matrix_encoding": "csr_matrix",
            "matrix_shape": [5, 3],
            "matrix_nonzero_count": len(csr_data),
            "matrix_value_dtype": "int64",
            "required_observation_fields": sorted(loader_module._REQUIRED_OBSERVATION_FIELDS),
            "source_feature_axis_sha256": feature_axis_sha256,
        },
    }
    artifacts["source-verification.json"] = _write_json(
        artifact_root / "source-verification.json", source_verification
    )

    role_rows: dict[str, list[int]] = {
        role: [index for index, value in enumerate(roles) if value == role]
        for role in partition_ids
    }
    universe_record_to_well = sorted(zip(record_ids, composite_wells, strict=True))
    universe_well_to_condition = sorted(zip(composite_wells, conditions, strict=True))
    universe_memberships: dict[str, object] = {
        "memberships/universe-record-ids.json": sorted(record_ids),
        "memberships/universe-well-ids.json": sorted(composite_wells),
        "memberships/universe-record-to-well.json": universe_record_to_well,
        "memberships/universe-well-to-condition.json": universe_well_to_condition,
    }
    for relative, value in universe_memberships.items():
        artifacts[relative] = _write_json(artifact_root / relative, value, canonical=True)

    universe_record_payload = artifacts["memberships/universe-record-ids.json"]
    universe = {
        "artifact_schema": "sciplex3-k562-record-universe",
        "source_sha256": source_sha256,
        "selector": {"cell_line": "K562"},
        "record_count": 5,
        "composite_well_count": 5,
        "record_ids_sha256": _sha256(universe_record_payload),
        "record_ids_encoded_byte_count": len(universe_record_payload),
    }
    artifacts["k562-universe.json"] = _write_json(artifact_root / "k562-universe.json", universe)

    partition_summaries: list[dict[str, object]] = []
    expected_partition_counts: dict[str, tuple[int, int]] = {}
    for role, rows in role_rows.items():
        role_record_ids = sorted(record_ids[index] for index in rows)
        role_wells = sorted(composite_wells[index] for index in rows)
        role_plates = sorted({plates[index] for index in rows})
        role_record_to_well = sorted((record_ids[index], composite_wells[index]) for index in rows)
        role_well_to_condition = sorted(
            (composite_wells[index], conditions[index]) for index in rows
        )
        payloads = {
            "record-ids": role_record_ids,
            "well-ids": role_wells,
            "plate-ids": role_plates,
            "record-to-well": role_record_to_well,
            "well-to-condition": role_well_to_condition,
        }
        for suffix, value in payloads.items():
            relative = f"memberships/{role}-{suffix}.json"
            artifacts[relative] = _write_json(artifact_root / relative, value, canonical=True)
        record_payload = artifacts[f"memberships/{role}-record-ids.json"]
        well_payload = artifacts[f"memberships/{role}-well-ids.json"]
        mapping_payload = artifacts[f"memberships/{role}-record-to-well.json"]
        partition_summaries.append(
            {
                "partition_role": role,
                "partition_id": partition_ids[role],
                "record_count": len(rows),
                "well_count": len(rows),
                "record_ids_sha256": _sha256(record_payload),
                "record_ids_encoded_byte_count": len(record_payload),
                "composite_well_ids_sha256": _sha256(well_payload),
                "composite_well_ids_encoded_byte_count": len(well_payload),
                "record_to_well_sha256": _sha256(mapping_payload),
                "record_to_well_encoded_byte_count": len(mapping_payload),
            }
        )
        expected_partition_counts[role] = (len(rows), len(rows))
    partitions = {
        "artifact_schema": "sciplex3-k562-frozen-partitions",
        "source_sha256": source_sha256,
        "assignment_unit": "plate",
        "partition_rule_uses_only_preoutcome_design_metadata": True,
        "partitions": partition_summaries,
    }
    artifacts["partitions.json"] = _write_json(artifact_root / "partitions.json", partitions)

    groups: list[dict[str, object]] = []
    for index, record_id in enumerate(record_ids):
        member_payload = _canonical([record_id])
        groups.append(
            {
                "composite_well_id": composite_wells[index],
                "plate": plates[index],
                "well": wells[index],
                "replicate": replicates[index],
                "partition_role": roles[index],
                "source_perturbation_label": perturbations[index],
                "normalized_perturbation_label": perturbations[index].strip(),
                "dose_value_nm": doses[index],
                "is_vehicle_control": perturbations[index] == "control",
                "source_scoped_condition_id": conditions[index],
                "record_count": 1,
                "record_ids_sha256": _sha256(member_payload),
                "record_ids_encoded_byte_count": len(member_payload),
            }
        )
    well_groups = {
        "artifact_schema": "sciplex3-k562-well-descendant-groups",
        "source_sha256": source_sha256,
        "group_count": 5,
        "groups": sorted(groups, key=lambda item: str(item["composite_well_id"])),
    }
    artifacts["well-groups.json"] = _write_json(artifact_root / "well-groups.json", well_groups)

    feature_panel = {
        "artifact_schema": "sciplex3-k562-train-feature-panel",
        "source_sha256": source_sha256,
        "feature_count": 2,
        "ordered_feature_keys_sha256": feature_keys_sha256,
        "ordered_feature_keys_encoded_byte_count": len(_canonical(feature_keys)),
        "feature_selection": {
            "selection_partition_role": "train",
            "count_accessed_partition_roles": ["train"],
            "heldout_count_rows_accessed": 0,
            "accessed_source_row_count": 2,
            "accessed_source_row_indices_sha256": _sha256(p1_source_rows_payload),
            "accessed_source_row_indices_encoded_byte_count": len(p1_source_rows_payload),
            "train_record_ids_sha256": _sha256(artifacts["memberships/train-record-ids.json"]),
        },
        "features": [
            {
                "rank": 1,
                "source_feature_index": 2,
                "ensembl_id": "ENSG2",
                "gene_symbol": "G2",
            },
            {
                "rank": 2,
                "source_feature_index": 0,
                "ensembl_id": "ENSG0",
                "gene_symbol": "G0",
            },
        ],
    }
    feature_panel_path = artifact_root / "feature-panel.json"
    artifacts["feature-panel.json"] = _write_json(feature_panel_path, feature_panel)

    entries = [
        {
            "relative_path": relative,
            "media_type": "application/json",
            "byte_count": len(payload),
            "sha256": _sha256(payload),
        }
        for relative, payload in sorted(artifacts.items())
    ]
    _write_json(
        artifact_root / "artifact-index.json",
        {
            "artifact_schema": "sciplex3-k562-preparation-index",
            "source_sha256": source_sha256,
            "artifacts": entries,
        },
    )

    def reference(relative_path: str, payload: bytes, encoding: str) -> dict[str, object]:
        return {
            "relative_path": relative_path,
            "byte_count": len(payload),
            "sha256": _sha256(payload),
            "encoding": encoding,
        }

    p1_contract = {
        "artifact_schema": "sciplex3-k562-p1-loader-contract",
        "artifact_schema_version": "1.0.0",
        "source": {
            "filename": source.name,
            "byte_count": len(source_payload),
            "md5": source_md5,
            "sha256": source_sha256,
        },
        "h5ad_structure": {
            "matrix_encoding": "csr_matrix",
            "matrix_shape": [5, 3],
            "matrix_nonzero_count": len(csr_data),
            "matrix_value_dtype": "int64",
            "required_observation_fields": sorted(loader_module._REQUIRED_OBSERVATION_FIELDS),
            "source_feature_axis_sha256": feature_axis_sha256,
        },
        "feature_panel": {
            "feature_count": 2,
            "ordered_feature_keys_sha256": feature_keys_sha256,
            "source_feature_axis_sha256": feature_axis_sha256,
        },
        "partition": {
            "access_purpose": "train_parameters",
            "artifact_role": "train",
            "partition_id": "p1-train",
            "record_count": 2,
            "selector": {
                "cell_line": "K562",
                "plates": ["plate1"],
                "replicate": "rep1",
            },
            "well_count": 2,
        },
        "bindings": {
            "benchmark_sha256": loader_module.SCIPLEX3_K562_BENCHMARK_SHA256,
            "dataset_manifest_sha256": loader_module.SCIPLEX3_K562_MANIFEST_SHA256,
            "query_sha256": loader_module.SCIPLEX3_K562_QUERY_SHA256,
            "scoring_transform_sha256": loader_module.SCIPLEX3_SCORING_TRANSFORM_SHA256,
            "target_value_schema_sha256": (loader_module.SCIPLEX3_K562_TARGET_VALUE_SCHEMA_SHA256),
        },
        "artifacts": {
            "source_verification": reference(
                "benchmarks/artifacts/sciplex3-k562-24h-v1/source-verification.json",
                artifacts["source-verification.json"],
                "canonical_json_utf8_object_v1",
            ),
            "feature_panel": reference(
                "benchmarks/artifacts/sciplex3-k562-24h-v1/feature-panel.json",
                artifacts["feature-panel.json"],
                "canonical_json_utf8_object_v1",
            ),
            "record_ids": reference(
                "memberships/train-record-ids.json",
                artifacts["memberships/train-record-ids.json"],
                "canonical_json_utf8_string_array_v1",
            ),
            "well_ids": reference(
                "memberships/train-well-ids.json",
                artifacts["memberships/train-well-ids.json"],
                "canonical_json_utf8_string_array_v1",
            ),
            "plate_ids": reference(
                "memberships/train-plate-ids.json",
                artifacts["memberships/train-plate-ids.json"],
                "canonical_json_utf8_string_array_v1",
            ),
            "record_to_well": reference(
                "memberships/train-record-to-well.json",
                artifacts["memberships/train-record-to-well.json"],
                "canonical_json_utf8_string_pair_array_v1",
            ),
            "well_to_condition": reference(
                "memberships/train-well-to-condition.json",
                artifacts["memberships/train-well-to-condition.json"],
                "canonical_json_utf8_string_pair_array_v1",
            ),
        },
        "heldout_memberships_referenced": False,
        "loader_outputs_can_mint_lifecycle_evidence": False,
        "scientifically_admissible_without_trusted_workflow_receipt": False,
    }
    contract_payload = _write_json(
        tmp_path / "benchmarks/vertical-a/sciplex3-k562-24h-v1/support/p1-loader-contract.json",
        p1_contract,
        canonical=True,
    )

    patches: dict[str, object] = {
        "SCIPLEX3_SOURCE_BYTE_COUNT": len(source_payload),
        "SCIPLEX3_SOURCE_MD5": source_md5,
        "SCIPLEX3_SOURCE_SHA256": source_sha256,
        "SCIPLEX3_H5AD_SHAPE": (5, 3),
        "SCIPLEX3_H5AD_NNZ": len(csr_data),
        "SCIPLEX3_FEATURE_COUNT": 2,
        "SCIPLEX3_ORDERED_FEATURE_KEYS_SHA256": feature_keys_sha256,
        "SCIPLEX3_SOURCE_FEATURE_AXIS_SHA256": feature_axis_sha256,
        "SCIPLEX3_FEATURE_PANEL_ARTIFACT_SHA256": _sha256(artifacts["feature-panel.json"]),
        "SCIPLEX3_P1_LOADER_CONTRACT_SHA256": _sha256(contract_payload),
        "_EXPECTED_P1_RECORD_COUNT": 2,
        "_EXPECTED_P1_WELL_COUNT": 2,
        "_EXPECTED_P1_TREATED_WELL_COUNT": 1,
        "_EXPECTED_P1_CONTROL_WELL_COUNT": 1,
        "_P1_PLATES": ("plate1",),
    }
    for name, value in patches.items():
        monkeypatch.setattr(loader_module, name, value)
    monkeypatch.setattr(
        SciPlex3K562H5ADLoader,
        "_import_h5py",
        staticmethod(lambda: fake_h5py),
    )
    return _SyntheticLoaderEnvironment(
        source=source,
        repository_root=tmp_path,
        h5py=fake_h5py,
        data=data_node,
        indices=index_node,
        indptr=indptr_node,
        ncounts=ncounts_node,
        feature_panel_path=feature_panel_path,
    )


def test_exact_source_object_is_given_to_h5py_and_reverified_at_close(
    synthetic_environment: _SyntheticLoaderEnvironment,
) -> None:
    loader = SciPlex3K562H5ADLoader.open(
        synthetic_environment.source,
        synthetic_environment.repository_root,
        access_purpose=PopulationComponentAccessPurpose.TRAIN_PARAMETERS,
    )
    source_stream = loader._source_stream
    assert source_stream is not None
    assert synthetic_environment.h5py.stream is source_stream
    assert synthetic_environment.h5py.mode == "r"
    assert synthetic_environment.h5py.open_count == 1
    loader.close()
    assert source_stream.closed
    assert synthetic_environment.h5py.h5ad.closed
    with pytest.raises(RuntimeError, match="closed"):
        loader.describe_partition()


def test_partition_purposes_are_one_to_one_before_any_count_read(
    synthetic_environment: _SyntheticLoaderEnvironment,
) -> None:
    descriptor = SciPlex3K562H5ADLoader.training_partition_descriptor(
        synthetic_environment.repository_root
    )
    assert descriptor.partition_id == "p1-train"
    assert descriptor.benchmark_role is BenchmarkPartitionRole.TRAIN
    assert descriptor.access_purpose is PopulationComponentAccessPurpose.TRAIN_PARAMETERS
    assert not descriptor.count_access_sealed
    assert not descriptor.can_mint_lifecycle_evidence
    assert not descriptor.scientifically_admissible

    with SciPlex3K562H5ADLoader(
        synthetic_environment.source,
        synthetic_environment.repository_root,
        access_purpose=PopulationComponentAccessPurpose.TRAIN_PARAMETERS,
    ) as loader:
        reads_before = synthetic_environment.data.reads
        with pytest.raises(ContractViolationError, match="bound only to p1-train"):
            loader.iter_batches(partition_id="p4-untouched-test")
        with pytest.raises(ContractViolationError, match="bound only to p1-train"):
            loader.iter_parameter_training_batches(partition_id="p2-calibration")
        assert synthetic_environment.data.reads == reads_before

    open_count = synthetic_environment.h5py.open_count
    reads_before = synthetic_environment.data.reads
    for purpose in (
        PopulationComponentAccessPurpose.FIT_CALIBRATION,
        PopulationComponentAccessPurpose.MODEL_SELECTION,
        PopulationComponentAccessPurpose.UNTOUCHED_EVALUATION,
    ):
        with pytest.raises(ContractViolationError, match="hard sealed"):
            SciPlex3K562H5ADLoader.open_for_purpose(
                synthetic_environment.source,
                synthetic_environment.repository_root,
                access_purpose=purpose,
            )
    assert synthetic_environment.h5py.open_count == open_count
    assert synthetic_environment.data.reads == reads_before


def test_p1_session_does_not_parse_global_or_heldout_ledgers(
    synthetic_environment: _SyntheticLoaderEnvironment,
) -> None:
    artifact_root = (
        synthetic_environment.repository_root / "benchmarks/artifacts/sciplex3-k562-24h-v1"
    )
    for relative in (
        "artifact-index.json",
        "partitions.json",
        "k562-universe.json",
        "well-groups.json",
        "memberships/calibration-record-ids.json",
        "memberships/model_selection_validation-record-ids.json",
        "memberships/untouched_test-record-ids.json",
    ):
        (artifact_root / relative).write_bytes(b"heldout-sentinel-not-json")

    with SciPlex3K562H5ADLoader.open_for_purpose(
        synthetic_environment.source,
        synthetic_environment.repository_root,
        access_purpose=PopulationComponentAccessPurpose.TRAIN_PARAMETERS,
    ) as loader:
        assert loader.describe_partition().partition_id == "p1-train"
        assert not loader.source_scan_receipt.heldout_memberships_parsed
        assert not hasattr(loader, "frozen_partition_descriptors")
        assert synthetic_environment.data.reads == 0
        assert synthetic_environment.indices.reads == 0
        assert synthetic_environment.indptr.reads == 0
        assert synthetic_environment.ncounts.reads == 0


def test_batches_are_deterministic_raw_sparse_immutable_and_provenanced(
    synthetic_environment: _SyntheticLoaderEnvironment,
) -> None:
    with SciPlex3K562H5ADLoader.open(
        synthetic_environment.source,
        synthetic_environment.repository_root,
        access_purpose=PopulationComponentAccessPurpose.TRAIN_PARAMETERS,
    ) as loader:
        assert isinstance(loader, SciPlex3TrainingDataLoader)
        assert synthetic_environment.data.reads == 0
        assert synthetic_environment.indices.reads == 0
        assert synthetic_environment.indptr.reads == 0
        assert synthetic_environment.ncounts.reads == 0
        receipt = loader.source_scan_receipt
        assert len(receipt.fingerprint) == 64
        assert receipt.partition_id == "p1-train"
        assert not receipt.heldout_memberships_parsed
        assert not receipt.heldout_outcome_values_parsed
        assert not receipt.lifecycle_evidence_issued
        assert not receipt.scientifically_admissible
        assert not receipt.trusted_workflow_receipt_present
        assert receipt.close_reverification_required
        assert not receipt.count_scan_complete
        assert not receipt.close_reverification_completed
        assert receipt.count_records_consumed == 0
        assert receipt.count_batches_consumed == 0
        with pytest.raises(ContractViolationError, match="unavailable before source close"):
            _ = loader.finalized_count_scan_receipt
        assert loader.feature_panel.source_feature_indices == (2, 0)
        assert loader.feature_panel.ordered_feature_keys == ("ENSG2|G2", "ENSG0|G0")
        batches = list(loader.iter_parameter_training_batches(batch_size=2))
        finalized = loader.finalize_parameter_training_count_scan()

    assert len(batches) == 1
    batch = batches[0]
    assert isinstance(batch, SciPlex3SparseCountBatch)
    assert batch.record_ids == ("r0", "r1")
    assert batch.composite_well_ids == ('["plate1","A1"]', '["plate1","A2"]')
    assert batch.condition_ids == (
        "source-label:drug-a@10nM",
        "source-control@0nM",
    )
    assert batch.source_row_indices.tolist() == [0, 1]
    assert batch.partition.partition_id == "p1-train"
    assert batch.partition.access_purpose is PopulationComponentAccessPurpose.TRAIN_PARAMETERS
    assert batch.shape == (2, 2)
    assert batch.indptr.tolist() == [0, 1, 2]
    assert batch.feature_indices.tolist() == [1, 0]
    assert batch.counts.tolist() == [1, 3]
    assert batch.panel_totals.tolist() == [1, 3]
    assert synthetic_environment.data.read_keys == [slice(0, 2), slice(2, 4)]
    assert synthetic_environment.indices.read_keys == [slice(0, 2), slice(2, 4)]
    assert synthetic_environment.indptr.read_keys == [0, 1, 1, 2]
    assert synthetic_environment.ncounts.read_keys == [0, 1]
    expected_count_stream = [
        [
            "r0",
            0,
            '["plate1","A1"]',
            "source-label:drug-a@10nM",
            [[1, 1]],
            1,
        ],
        [
            "r1",
            1,
            '["plate1","A2"]',
            "source-control@0nM",
            [[0, 3]],
            3,
        ],
    ]
    assert finalized.finalized
    assert finalized.count_scan_complete
    assert finalized.close_reverification_completed
    assert finalized.source_descriptor_reverified
    assert finalized.exact_record_coverage
    assert finalized.record_count == 2
    assert finalized.well_count == 2
    assert finalized.treated_well_count == 1
    assert finalized.control_well_count == 1
    assert finalized.batch_count == 1
    assert finalized.panel_nonzero_count == 2
    assert finalized.zero_panel_record_count == 0
    assert finalized.panel_umi_total == 4
    assert finalized.full_source_umi_total == 7
    assert finalized.emitted_source_row_indices_sha256 == _sha256(_canonical([0, 1]))
    assert finalized.panel_count_stream_sha256 == _sha256(_canonical(expected_count_stream))
    assert finalized.accessed_partition_roles == ("p1-train",)
    assert finalized.accessed_count_datasets == (
        "X.data",
        "X.indices",
        "X.indptr",
        "obs.ncounts",
    )
    assert not finalized.heldout_memberships_parsed
    assert not finalized.heldout_outcome_values_parsed
    assert not finalized.lifecycle_evidence_issued
    assert not finalized.scientifically_admissible
    assert len(finalized.loader_implementation_sha256) == 64
    assert len(finalized.fingerprint) == 64
    assert loader.finalized_count_scan_receipt is finalized
    for array in (
        batch.source_row_indices,
        batch.indptr,
        batch.feature_indices,
        batch.counts,
        batch.panel_totals,
    ):
        assert not array.flags.writeable
        with pytest.raises(ValueError):
            array.flags.writeable = True


def test_public_panel_and_batch_constructors_snapshot_caller_lists(
    synthetic_environment: _SyntheticLoaderEnvironment,
) -> None:
    with SciPlex3K562H5ADLoader.open(
        synthetic_environment.source,
        synthetic_environment.repository_root,
        access_purpose=PopulationComponentAccessPurpose.TRAIN_PARAMETERS,
    ) as loader:
        original_panel = loader.feature_panel
        original_batch = next(loader.iter_parameter_training_batches(batch_size=2))
        source_indices = list(original_panel.source_feature_indices)
        feature_keys = list(original_panel.ordered_feature_keys)
        panel = SciPlex3FeaturePanel(
            source_feature_indices=source_indices,  # type: ignore[arg-type]
            ordered_feature_keys=feature_keys,  # type: ignore[arg-type]
            ordered_feature_keys_sha256=original_panel.ordered_feature_keys_sha256,
            source_feature_axis_sha256=original_panel.source_feature_axis_sha256,
        )
        record_ids = list(original_batch.record_ids)
        well_ids = list(original_batch.composite_well_ids)
        condition_ids = list(original_batch.condition_ids)
        batch = SciPlex3SparseCountBatch(
            partition=original_batch.partition,
            batch_index=original_batch.batch_index,
            record_ids=record_ids,  # type: ignore[arg-type]
            composite_well_ids=well_ids,  # type: ignore[arg-type]
            condition_ids=condition_ids,  # type: ignore[arg-type]
            source_row_indices=original_batch.source_row_indices,
            indptr=original_batch.indptr,
            feature_indices=original_batch.feature_indices,
            counts=original_batch.counts,
            panel_totals=original_batch.panel_totals,
        )

    source_indices[0] = 999
    feature_keys[0] = "mutated"
    record_ids[0] = "mutated"
    well_ids[0] = "mutated"
    condition_ids[0] = "mutated"
    assert panel.source_feature_indices == original_panel.source_feature_indices
    assert panel.ordered_feature_keys == original_panel.ordered_feature_keys
    assert batch.record_ids == original_batch.record_ids
    assert batch.composite_well_ids == original_batch.composite_well_ids
    assert batch.condition_ids == original_batch.condition_ids


def test_finalized_count_scan_requires_one_fully_exhausted_iterator(
    synthetic_environment: _SyntheticLoaderEnvironment,
) -> None:
    loader = SciPlex3K562H5ADLoader.open(
        synthetic_environment.source,
        synthetic_environment.repository_root,
        access_purpose=PopulationComponentAccessPurpose.TRAIN_PARAMETERS,
    )
    iterator = loader.iter_parameter_training_batches(batch_size=1)
    first = next(iterator)
    assert first.record_ids == ("r0",)
    with pytest.raises(ContractViolationError, match="fully exhausted"):
        loader.finalize_parameter_training_count_scan()
    loader.close()
    with pytest.raises(ContractViolationError, match="no finalized"):
        _ = loader.finalized_count_scan_receipt


def test_batch_size_validation_reads_no_counts(
    synthetic_environment: _SyntheticLoaderEnvironment,
) -> None:
    with SciPlex3K562H5ADLoader.open(
        synthetic_environment.source,
        synthetic_environment.repository_root,
        access_purpose=PopulationComponentAccessPurpose.TRAIN_PARAMETERS,
    ) as loader:
        before = synthetic_environment.data.reads
        for value in (0, -1, True, 1.5):
            with pytest.raises(ValueError, match="positive integer"):
                loader.iter_batches(batch_size=value)  # type: ignore[arg-type]
        assert synthetic_environment.data.reads == before


def test_negative_counts_fail_closed(
    synthetic_environment: _SyntheticLoaderEnvironment,
) -> None:
    synthetic_environment.data.values[0] = -1
    with (
        SciPlex3K562H5ADLoader.open(
            synthetic_environment.source,
            synthetic_environment.repository_root,
            access_purpose=PopulationComponentAccessPurpose.TRAIN_PARAMETERS,
        ) as loader,
        pytest.raises(ContractViolationError, match="negative raw UMI"),
    ):
        next(loader.iter_parameter_training_batches(batch_size=1))


def test_source_row_sum_must_equal_obs_ncounts(
    synthetic_environment: _SyntheticLoaderEnvironment,
) -> None:
    synthetic_environment.data.values[0] = 2
    with (
        SciPlex3K562H5ADLoader.open(
            synthetic_environment.source,
            synthetic_environment.repository_root,
            access_purpose=PopulationComponentAccessPurpose.TRAIN_PARAMETERS,
        ) as loader,
        pytest.raises(ContractViolationError, match="differs from obs ncounts"),
    ):
        next(loader.iter_parameter_training_batches(batch_size=1))


def test_zero_panel_training_rows_are_retained_and_counted_without_imputation(
    synthetic_environment: _SyntheticLoaderEnvironment,
) -> None:
    # First raw row is [1, 2, 0].  Move its panel count into non-panel source feature 1.
    synthetic_environment.data.values[0] = 0
    synthetic_environment.data.values[1] = 3
    with SciPlex3K562H5ADLoader.open(
        synthetic_environment.source,
        synthetic_environment.repository_root,
        access_purpose=PopulationComponentAccessPurpose.TRAIN_PARAMETERS,
    ) as loader:
        batches = list(loader.iter_parameter_training_batches(batch_size=1))
        finalized = loader.finalize_parameter_training_count_scan()
    assert batches[0].indptr.tolist() == [0, 0]
    assert batches[0].counts.tolist() == []
    assert batches[0].panel_totals.tolist() == [0]
    assert finalized.zero_panel_record_count == 1
    assert finalized.record_count == 2


def test_source_content_drift_while_open_is_detected_on_close(
    synthetic_environment: _SyntheticLoaderEnvironment,
) -> None:
    loader = SciPlex3K562H5ADLoader.open(
        synthetic_environment.source,
        synthetic_environment.repository_root,
        access_purpose=PopulationComponentAccessPurpose.TRAIN_PARAMETERS,
    )
    assert len(list(loader.iter_parameter_training_batches(batch_size=2))) == 1
    original = synthetic_environment.source.read_bytes()
    synthetic_environment.source.write_bytes(bytes([original[0] ^ 1]) + original[1:])
    with pytest.raises(ContractViolationError, match="drifted during H5AD access"):
        loader.finalize_parameter_training_count_scan()
    with pytest.raises(ContractViolationError, match="no finalized"):
        _ = loader.finalized_count_scan_receipt


def test_artifact_drift_fails_before_h5py_or_counts(
    synthetic_environment: _SyntheticLoaderEnvironment,
) -> None:
    synthetic_environment.feature_panel_path.write_bytes(b"{}")
    with pytest.raises(ContractViolationError, match="content-addressed p1 artifact drift"):
        SciPlex3K562H5ADLoader.open(
            synthetic_environment.source,
            synthetic_environment.repository_root,
            access_purpose=PopulationComponentAccessPurpose.TRAIN_PARAMETERS,
        )
    assert synthetic_environment.h5py.open_count == 0
    assert synthetic_environment.data.reads == 0


def test_optional_h5py_failure_is_explicit_and_does_not_leak_a_handle(
    synthetic_environment: _SyntheticLoaderEnvironment,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unavailable() -> object:
        raise RuntimeError("reading the sci-Plex3 H5AD requires optional h5py")

    monkeypatch.setattr(
        SciPlex3K562H5ADLoader,
        "_import_h5py",
        staticmethod(unavailable),
    )
    with pytest.raises(RuntimeError, match="optional h5py"):
        SciPlex3K562H5ADLoader.open(
            synthetic_environment.source,
            synthetic_environment.repository_root,
            access_purpose=PopulationComponentAccessPurpose.TRAIN_PARAMETERS,
        )
    assert synthetic_environment.h5py.open_count == 0


def test_loader_has_no_public_biological_result_surface() -> None:
    public_names = {name for name in dir(loader_module) if not name.startswith("_")}
    assert "CellStateBelief" not in public_names
    assert "PopulationAssayResponse" not in public_names
    assert not hasattr(SciPlex3K562H5ADLoader, "estimate")
    assert not hasattr(SciPlex3K562H5ADLoader, "sample_response")
    assert not hasattr(SciPlex3K562H5ADLoader, "issue_training_receipt")
    assert not hasattr(SciPlex3K562H5ADLoader, "issue_lifecycle_evidence")
