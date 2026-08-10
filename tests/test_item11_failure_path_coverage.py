"""Compact failure-path coverage for the Item 11 loader and runner boundaries."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import cellstate.backends.sciplex3_loader as loader
import cellstate.evaluation.sciplex3_runner as runner
from cellstate.backends.sciplex3_k562 import PopulationComponentAccessPurpose
from cellstate.data.benchmarks import BenchmarkPartitionRole
from cellstate.errors import ContractViolationError
from cellstate.evaluation.sciplex3_baselines import NO_ACTION, PredictionTarget

_DIGEST = "0" * 64


def _descriptor_values() -> dict[str, Any]:
    return {
        "partition_id": "p1-train",
        "artifact_role": "train",
        "benchmark_role": BenchmarkPartitionRole.TRAIN,
        "access_purpose": PopulationComponentAccessPurpose.TRAIN_PARAMETERS,
        "record_count": 1,
        "well_count": 1,
        "record_ids_sha256": _DIGEST,
        "record_to_well_sha256": _DIGEST,
        "source_sha256": _DIGEST,
        "loader_contract_sha256": _DIGEST,
        "dataset_manifest_sha256": _DIGEST,
        "query_sha256": _DIGEST,
        "benchmark_sha256": _DIGEST,
        "target_value_schema_sha256": _DIGEST,
        "scoring_transform_sha256": _DIGEST,
        "feature_panel_artifact_sha256": _DIGEST,
        "ordered_feature_keys_sha256": _DIGEST,
        "count_access_sealed": False,
    }


def _descriptor(**overrides: object) -> loader.SciPlex3PartitionDescriptor:
    values = _descriptor_values()
    values.update(overrides)
    return loader.SciPlex3PartitionDescriptor(**values)


def _batch_values() -> dict[str, Any]:
    return {
        "partition": _descriptor(),
        "batch_index": 0,
        "record_ids": ("record",),
        "composite_well_ids": ("well",),
        "condition_ids": ("condition",),
        "source_row_indices": np.asarray([0], dtype=np.int64),
        "indptr": np.asarray([0, 1], dtype=np.int64),
        "feature_indices": np.asarray([0], dtype=np.int64),
        "counts": np.asarray([1], dtype=np.int64),
        "panel_totals": np.asarray([1], dtype=np.int64),
    }


def _invalid_batch(**overrides: object) -> None:
    values = _batch_values()
    values.update(overrides)
    with pytest.raises(ContractViolationError):
        loader.SciPlex3SparseCountBatch(**values)


def test_loader_scalar_tuple_and_array_failure_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ContractViolationError, match="exact tuple"):
        loader._require_exact_tuple([], name="value", item_type=int)
    with pytest.raises(ContractViolationError, match="wrong tuple length"):
        loader._require_exact_tuple((1,), name="value", item_type=int, length=2)
    with pytest.raises(ContractViolationError, match="non-exact scalar"):
        loader._require_exact_tuple((True,), name="value", item_type=int)
    with pytest.raises(ContractViolationError, match="non-UTF-8"):
        loader._decode_scalar(b"\xff")
    with pytest.raises(ContractViolationError, match="exact integer"):
        loader._readonly_int64(np.asarray([1.0]))
    with pytest.raises(ContractViolationError, match="signed 64-bit"):
        loader._readonly_int64(np.asarray([np.iinfo(np.uint64).max], dtype=np.uint64))

    def fail_read(path: Path) -> bytes:
        del path
        raise OSError("unreadable")

    with monkeypatch.context() as patch:
        patch.setattr(Path, "read_bytes", fail_read)
        with pytest.raises(ContractViolationError, match="cannot authenticate"):
            loader._read_loader_implementation_sha256()


def test_feature_panel_rejects_each_invalid_axis_property() -> None:
    indices = tuple(range(loader.SCIPLEX3_FEATURE_COUNT))
    keys = tuple(f"feature-{index}" for index in indices)

    invalid_axes = (
        {"source_feature_indices": indices[:-1], "ordered_feature_keys": keys},
        {"source_feature_indices": indices, "ordered_feature_keys": keys[:-1]},
        {"source_feature_indices": (0,) * len(indices), "ordered_feature_keys": keys},
        {"source_feature_indices": indices, "ordered_feature_keys": ("same",) * len(keys)},
        {
            "source_feature_indices": (-1, *indices[1:]),
            "ordered_feature_keys": keys,
        },
        {
            "source_feature_indices": indices,
            "ordered_feature_keys": ("", *keys[1:]),
        },
        {
            "source_feature_indices": indices,
            "ordered_feature_keys": (" padded ", *keys[1:]),
        },
    )
    for axes in invalid_axes:
        with pytest.raises(ContractViolationError):
            loader.SciPlex3FeaturePanel(
                **axes,
                ordered_feature_keys_sha256=_DIGEST,
                source_feature_axis_sha256=_DIGEST,
            )


def test_partition_descriptor_rejects_nonexact_enums() -> None:
    with pytest.raises(ContractViolationError, match="benchmark_role"):
        _descriptor(benchmark_role="train")
    with pytest.raises(ContractViolationError, match="access_purpose"):
        _descriptor(access_purpose="train_parameters")


def test_sparse_batch_rejects_malformed_public_inputs() -> None:
    _invalid_batch(partition=object())
    _invalid_batch(batch_index=True)
    _invalid_batch(record_ids=("",))
    _invalid_batch(
        record_ids=(),
        composite_well_ids=(),
        condition_ids=(),
        source_row_indices=np.asarray([], dtype=np.int64),
        indptr=np.asarray([0], dtype=np.int64),
        feature_indices=np.asarray([], dtype=np.int64),
        counts=np.asarray([], dtype=np.int64),
        panel_totals=np.asarray([], dtype=np.int64),
    )
    _invalid_batch(composite_well_ids=())
    _invalid_batch(indptr=np.asarray([0], dtype=np.int64))
    _invalid_batch(indptr=np.asarray([1, 1], dtype=np.int64))
    _invalid_batch(indptr=np.asarray([0, 2], dtype=np.int64))
    _invalid_batch(source_row_indices=np.asarray([-1], dtype=np.int64))
    _invalid_batch(counts=np.asarray([-1], dtype=np.int64), panel_totals=np.asarray([0]))
    _invalid_batch(panel_totals=np.asarray([-1], dtype=np.int64))
    _invalid_batch(feature_indices=np.asarray([-1], dtype=np.int64))
    _invalid_batch(feature_indices=np.asarray([loader.SCIPLEX3_FEATURE_COUNT], dtype=np.int64))
    _invalid_batch(
        indptr=np.asarray([0, 2], dtype=np.int64),
        feature_indices=np.asarray([1, 1], dtype=np.int64),
        counts=np.asarray([1, 1], dtype=np.int64),
        panel_totals=np.asarray([2], dtype=np.int64),
    )
    _invalid_batch(panel_totals=np.asarray([2], dtype=np.int64))


def test_count_scan_rejects_invalid_totals() -> None:
    state = loader._CountScanState()
    arguments = {
        "record_id": "record",
        "source_row": 0,
        "well_id": "well",
        "condition_id": "condition",
        "panel_pairs": [],
    }
    with pytest.raises(ContractViolationError, match="raw-count domain"):
        state.append(**arguments, panel_total=-1, full_source_total=1)
    with pytest.raises(ContractViolationError, match="raw-count domain"):
        state.append(**arguments, panel_total=0, full_source_total=0)


def test_runner_parsing_and_file_failure_paths(tmp_path: Path) -> None:
    with pytest.raises(runner.SciPlex3RunnerError, match="canonical-JSON"):
        runner._canonical_json({"bad": object()})
    with pytest.raises(runner.SciPlex3RunnerError, match="missing absent"):
        runner._read_bytes(tmp_path / "absent", name="absent")

    payload_path = tmp_path / "payload.json"
    payload_path.write_bytes(b"{}")
    with pytest.raises(runner.SciPlex3RunnerError, match="SHA-256 drift"):
        runner._read_exact(payload_path, _DIGEST, name="payload")
    with pytest.raises(runner.SciPlex3RunnerError, match="invalid JSON"):
        runner._json_value(b"{", name="value")
    with pytest.raises(runner.SciPlex3RunnerError, match="JSON object"):
        runner._json_object(b"[]", name="value")
    with pytest.raises(runner.SciPlex3RunnerError, match="must be an object"):
        runner._as_mapping([], name="value")
    with pytest.raises(runner.SciPlex3RunnerError, match="must be an array"):
        runner._as_list({}, name="value")
    with pytest.raises(runner.SciPlex3RunnerError, match="not canonical JSON"):
        runner._parse_composite_well("not-json")
    with pytest.raises(runner.SciPlex3RunnerError, match=r"exact.*pair"):
        runner._parse_composite_well('["plate", "well"]')
    with pytest.raises(runner.SciPlex3RunnerError, match="exact nonblank"):
        runner._exact_text(" padded ", name="value")
    with pytest.raises(runner.SciPlex3RunnerError, match="lowercase SHA"):
        runner._exact_sha256("A" * 64, name="digest")


def test_runner_contract_artifact_reference_failures(tmp_path: Path) -> None:
    with pytest.raises(runner.SciPlex3RunnerError, match="malformed"):
        runner._resolve_contract_artifact(tmp_path, {}, name="artifact")
    with pytest.raises(runner.SciPlex3RunnerError, match="escapes"):
        runner._resolve_contract_artifact(
            tmp_path,
            {"relative_path": "../x", "sha256": _DIGEST, "byte_count": 0},
            name="artifact",
        )

    artifact = tmp_path / runner._PREPARATION_DIRECTORY / "small.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"{}")
    with pytest.raises(runner.SciPlex3RunnerError, match="byte count drifted"):
        runner._resolve_contract_artifact(
            tmp_path,
            {
                "relative_path": "small.json",
                "sha256": hashlib.sha256(b"{}").hexdigest(),
                "byte_count": 3,
            },
            name="artifact",
        )


def _action(index: int, *, compound: str | None = None) -> runner.SciPlex3ActionBinding:
    name = compound or f"compound-{index}"
    return runner.SciPlex3ActionBinding(
        source_condition_id=f"source-label:{name}@10nM",
        query_spec_id=f"query-{index}",
        compound=name,
        dose_nm=10,
        intervention_kind_key="kind",
    )


@pytest.fixture(scope="module")
def action_maps() -> tuple[
    dict[str, runner.SciPlex3ActionBinding],
    dict[str, runner.SciPlex3ActionBinding],
]:
    actions = tuple(_action(index) for index in range(runner.SCIPLEX3_ACTION_ENTRY_COUNT))
    return (
        {action.source_condition_id: action for action in actions},
        {action.query_spec_id: action for action in actions},
    )


def _p1_design_values() -> dict[str, Any]:
    return {
        "query_sha256": _DIGEST,
        "query_fingerprint": _DIGEST,
        "benchmark_sha256": _DIGEST,
        "action_domain_sha256": _DIGEST,
        "scoring_transform_sha256": _DIGEST,
        "target_value_schema_sha256": _DIGEST,
        "ordered_feature_keys_sha256": _DIGEST,
        "actions_by_source_condition": {},
        "actions_by_query_spec": {},
    }


def _invalid_p1_design(**overrides: object) -> None:
    values = _p1_design_values()
    values.update(overrides)
    with pytest.raises(runner.SciPlex3RunnerError):
        runner.SciPlex3P1DesignBindings(**values)


def test_action_and_p1_design_constructor_failures(
    action_maps: tuple[
        dict[str, runner.SciPlex3ActionBinding],
        dict[str, runner.SciPlex3ActionBinding],
    ],
) -> None:
    with pytest.raises(runner.SciPlex3RunnerError, match="supported integer"):
        runner.SciPlex3ActionBinding("source-label:x@11nM", "q", "x", 11, "kind")
    with pytest.raises(runner.SciPlex3RunnerError, match="differs"):
        runner.SciPlex3ActionBinding("source-label:y@10nM", "q", "x", 10, "kind")
    with pytest.raises(runner.SciPlex3RunnerError, match="case-folded"):
        runner.SciPlex3ActionBinding("source-label:x@10nM", "q", "x", 10, "Kind")

    _invalid_p1_design(can_mint_lifecycle_evidence=True)
    _invalid_p1_design(actions_by_source_condition=object())
    binding = _action(0)
    _invalid_p1_design(
        actions_by_source_condition={binding.source_condition_id: object()},
        actions_by_query_spec={},
    )
    _invalid_p1_design(
        actions_by_source_condition={binding.source_condition_id: binding},
        actions_by_query_spec={"wrong-query": binding},
    )
    _invalid_p1_design()

    source_actions, query_actions = action_maps
    _invalid_p1_design(
        actions_by_source_condition=source_actions,
        actions_by_query_spec={},
    )
    mismatched_query_actions = dict(query_actions)
    mismatched_query_actions["query-0"] = _action(0, compound="different")
    _invalid_p1_design(
        actions_by_source_condition=source_actions,
        actions_by_query_spec=mismatched_query_actions,
    )


def _target(index: int) -> PredictionTarget:
    return PredictionTarget(
        case_id=f"case-{index:03d}",
        target_well_id=f"well-{index:03d}",
        plate_id="plate",
        partition_id="p4-untouched-test",
        condition=NO_ACTION,
    )


def _p4_design_values() -> dict[str, Any]:
    return {
        "query_sha256": _DIGEST,
        "benchmark_sha256": _DIGEST,
        "action_domain_sha256": _DIGEST,
        "evaluation_cases_sha256": _DIGEST,
        "scoring_transform_sha256": _DIGEST,
        "target_value_schema_sha256": _DIGEST,
        "ordered_feature_keys_sha256": _DIGEST,
        "fitted_state_artifact_sha256": _DIGEST,
        "baseline_suite_specification_sha256": _DIGEST,
        "preparation_fingerprint": _DIGEST,
        "prediction_targets_sha256": runner.SCIPLEX3_P4_PREDICTION_TARGETS_SHA256,
        "p4_targets": (),
    }


def _invalid_p4_design(**overrides: object) -> None:
    values = _p4_design_values()
    values.update(overrides)
    with pytest.raises(runner.SciPlex3RunnerError):
        runner.SciPlex3P4PredictionDesign(**values)


def test_p4_design_constructor_failure_paths() -> None:
    _invalid_p4_design(p4_targets=(object(),))
    _invalid_p4_design(heldout_outcomes_read=True)
    _invalid_p4_design()

    targets = tuple(_target(index) for index in range(runner.SCIPLEX3_P4_CASE_COUNT))
    duplicate_targets = (targets[0], targets[0], *targets[2:])
    _invalid_p4_design(p4_targets=duplicate_targets)
    _invalid_p4_design(p4_targets=targets)


def _assembly_values() -> dict[str, Any]:
    hash_names = (
        "loader_source_scan_fingerprint",
        "finalized_count_scan_fingerprint",
        "loader_implementation_sha256",
        "loader_contract_sha256",
        "source_sha256",
        "record_ids_sha256",
        "record_to_well_sha256",
        "well_ids_sha256",
        "well_to_condition_sha256",
        "source_row_indices_sha256",
        "emitted_source_row_indices_sha256",
        "ordered_record_source_well_condition_sha256",
        "runner_panel_count_stream_sha256",
        "loader_panel_count_stream_sha256",
        "ordered_feature_keys_sha256",
        "feature_panel_artifact_sha256",
        "action_domain_sha256",
        "query_sha256",
        "benchmark_sha256",
        "scoring_transform_sha256",
        "target_value_schema_sha256",
    )
    values: dict[str, Any] = {name: _DIGEST for name in hash_names}
    values.update(
        record_count=1,
        well_count=1,
        treated_well_count=1,
        control_well_count=1,
        panel_nonzero_count=1,
        zero_panel_record_count=0,
        panel_umi_total=1,
        full_source_umi_total=1,
        batch_count=1,
    )
    return values


def _invalid_assembly(**overrides: object) -> None:
    values = _assembly_values()
    values.update(overrides)
    with pytest.raises(runner.SciPlex3RunnerError):
        runner.SciPlex3P1AssemblyReceipt(**values)


def test_runner_receipt_and_artifact_constructor_failures(tmp_path: Path) -> None:
    _invalid_assembly(record_count=True)
    _invalid_assembly(exact_record_coverage=False)
    _invalid_assembly(heldout_outcomes_read=True)

    with pytest.raises(runner.SciPlex3RunnerError, match="path-like"):
        runner.LocalContentAddressedArtifact(object(), _DIGEST, 1, "application/json")
    with pytest.raises(runner.SciPlex3RunnerError, match="positive exact integer"):
        runner.LocalContentAddressedArtifact(tmp_path / "x", _DIGEST, 0, "application/json")
    with pytest.raises(runner.SciPlex3RunnerError, match="authority flags"):
        runner.LocalContentAddressedArtifact(
            tmp_path / "x",
            _DIGEST,
            1,
            "application/json",
            can_mint_lifecycle_evidence=True,
        )

    with pytest.raises(runner.SciPlex3RunnerError, match="exact local artifact"):
        runner.FittedSciPlex3Baseline(object(), object(), {}, _DIGEST)
    artifact = runner.LocalContentAddressedArtifact(tmp_path / "x", _DIGEST, 1, "application/json")
    with pytest.raises(runner.SciPlex3RunnerError, match="authority flags"):
        runner.FittedSciPlex3Baseline(
            object(),
            artifact,
            {},
            _DIGEST,
            can_mint_lifecycle_evidence=True,
        )


@dataclass
class _JsonReadyFixture:
    path: Path
    values: tuple[int, ...]


def test_runner_internal_container_failure_paths(tmp_path: Path) -> None:
    with pytest.raises(runner.SciPlex3RunnerError, match="not a dataclass"):
        runner._json_ready_dataclass(object())
    assert runner._json_ready_dataclass(_JsonReadyFixture(tmp_path, (1, 2))) == {
        "path": str(tmp_path),
        "values": [1, 2],
    }
    with pytest.raises(runner.SciPlex3RunnerError, match="contains no rows"):
        runner._WellBuilder().freeze()
