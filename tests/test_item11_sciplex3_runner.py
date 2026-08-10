from __future__ import annotations

import hashlib
import json
import sys
from collections.abc import Iterator, Mapping
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pytest

import cellstate.evaluation.sciplex3_runner as runner_module
from cellstate.backends.sciplex3_k562 import PopulationComponentAccessPurpose
from cellstate.backends.sciplex3_loader import (
    SCIPLEX3_FEATURE_COUNT,
    SCIPLEX3_H5AD_NNZ,
    SCIPLEX3_H5AD_SHAPE,
    SCIPLEX3_ORDERED_FEATURE_KEYS_SHA256,
    SCIPLEX3_SOURCE_BYTE_COUNT,
    SCIPLEX3_SOURCE_FEATURE_AXIS_SHA256,
    SCIPLEX3_SOURCE_MD5,
    SCIPLEX3_SOURCE_SHA256,
    SciPlex3FeaturePanel,
    SciPlex3K562H5ADLoader,
    SciPlex3P1FinalizedCountScanReceipt,
    SciPlex3P1SourceScanReceipt,
    SciPlex3PartitionDescriptor,
    SciPlex3SparseCountBatch,
)
from cellstate.errors import ContractViolationError
from cellstate.evaluation.sciplex3_baselines import (
    SCIPLEX3_BASELINE_SAMPLES_PER_CASE_PER_SEED,
    SCIPLEX3_BASELINE_SEEDS,
    BaselineSampleRequest,
    ExactConditionRep1EmpiricalResampling,
    ImmutableCSRCounts,
    NoAction,
    P1TrainingData,
    PredictionTarget,
    PredictiveRawCountSamples,
    SciPlex3BaselineError,
    SciPlex3RawCountBaseline,
)
from cellstate.evaluation.sciplex3_runner import (
    SCIPLEX3_ACTION_ENTRY_COUNT,
    SCIPLEX3_BASELINE_CODE_SHA256,
    SCIPLEX3_BASELINE_GOLDEN_FIXTURE_SHA256,
    SCIPLEX3_LOADER_CODE_SHA256,
    SCIPLEX3_P1_CONTROL_WELL_COUNT,
    SCIPLEX3_P1_RECORD_COUNT,
    SCIPLEX3_P1_TREATED_WELL_COUNT,
    SCIPLEX3_P1_WELL_COUNT,
    SCIPLEX3_P4_CASE_COUNT,
    SCIPLEX3_P4_CONTROL_CASE_COUNT,
    SCIPLEX3_P4_PREDICTION_TARGETS_SHA256,
    SCIPLEX3_P4_TREATED_CASE_COUNT,
    SCIPLEX3_PREDICTION_SHARD_DRAW_COUNT,
    FittedSciPlex3Baseline,
    PredictionShardEntry,
    SciPlex3ActionBinding,
    SciPlex3BaselinePreparation,
    SciPlex3P4PredictionDesign,
    SciPlex3PredictionArtifactWriter,
    SciPlex3RunnerError,
    assemble_sciplex3_p1_training_data,
    fit_and_write_sciplex3_baseline,
    open_sciplex3_p4_prediction_design,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PREPARATION_ROOT = REPOSITORY_ROOT / "benchmarks/artifacts/sciplex3-k562-24h-v1"
FEATURE_PANEL_PATH = PREPARATION_ROOT / "feature-panel.json"
MEMBERSHIP_ROOT = PREPARATION_ROOT / "memberships"
BENCHMARK_PATH = (
    REPOSITORY_ROOT / "benchmarks/vertical-a/sciplex3-k562-24h-v1/benchmark-artifact.json"
)
CASES_PATH = (
    REPOSITORY_ROOT / "benchmarks/vertical-a/sciplex3-k562-24h-v1/support/evaluation-cases.json"
)


def _golden_runtime_or_skip() -> None:
    if sys.version_info[:2] != (3, 11) or np.__version__ != "2.4.6":
        pytest.skip("fitted artifacts require the exact golden Python/NumPy runtime")


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


class _SyntheticExactP1Loader:
    """Exact metadata closure with sparse synthetic rows; no H5AD is opened."""

    def __init__(
        self,
        *,
        corrupt_final_stream: bool = False,
        zero_panel_record_count: int = 7,
    ) -> None:
        if not 0 <= zero_panel_record_count <= SCIPLEX3_P1_RECORD_COUNT:
            raise ValueError("invalid synthetic zero-panel record count")
        self._descriptor = SciPlex3K562H5ADLoader.training_partition_descriptor(REPOSITORY_ROOT)
        feature_document = json.loads(FEATURE_PANEL_PATH.read_bytes())
        features = feature_document["features"]
        self._panel = SciPlex3FeaturePanel(
            source_feature_indices=tuple(row["source_feature_index"] for row in features),
            ordered_feature_keys=tuple(
                f"{row['ensembl_id']}|{row['gene_symbol']}" for row in features
            ),
            ordered_feature_keys_sha256=feature_document["ordered_feature_keys_sha256"],
            source_feature_axis_sha256=SCIPLEX3_SOURCE_FEATURE_AXIS_SHA256,
        )
        self._record_ids = tuple(
            json.loads((MEMBERSHIP_ROOT / "train-record-ids.json").read_bytes())
        )
        self._record_to_well = tuple(
            tuple(value)
            for value in json.loads((MEMBERSHIP_ROOT / "train-record-to-well.json").read_bytes())
        )
        self._well_to_condition = dict(
            json.loads((MEMBERSHIP_ROOT / "train-well-to-condition.json").read_bytes())
        )
        self._well_by_record = dict(self._record_to_well)
        self._source_rows = list(range(SCIPLEX3_P1_RECORD_COUNT))
        binding_rows = [
            [
                record_id,
                source_row,
                self._well_by_record[record_id],
                self._well_to_condition[self._well_by_record[record_id]],
            ]
            for source_row, record_id in enumerate(self._record_ids)
        ]
        source_rows_sha256 = _sha256([str(row) for row in sorted(self._source_rows)])
        self._initial_receipt = SciPlex3P1SourceScanReceipt(
            source_sha256=SCIPLEX3_SOURCE_SHA256,
            source_md5=SCIPLEX3_SOURCE_MD5,
            source_byte_count=SCIPLEX3_SOURCE_BYTE_COUNT,
            p1_loader_contract_sha256=self._descriptor.loader_contract_sha256,
            feature_panel_artifact_sha256=self._descriptor.feature_panel_artifact_sha256,
            ordered_feature_keys_sha256=SCIPLEX3_ORDERED_FEATURE_KEYS_SHA256,
            record_ids_sha256=self._descriptor.record_ids_sha256,
            record_to_well_sha256=self._descriptor.record_to_well_sha256,
            source_row_indices_sha256=source_rows_sha256,
            ordered_record_source_well_condition_sha256=_sha256(binding_rows),
            dataset_manifest_sha256=self._descriptor.dataset_manifest_sha256,
            query_sha256=self._descriptor.query_sha256,
            benchmark_sha256=self._descriptor.benchmark_sha256,
            target_value_schema_sha256=self._descriptor.target_value_schema_sha256,
            scoring_transform_sha256=self._descriptor.scoring_transform_sha256,
            matrix_shape=SCIPLEX3_H5AD_SHAPE,
            matrix_nonzero_count=SCIPLEX3_H5AD_NNZ,
            matrix_value_dtype="int64",
        )
        self._corrupt_final_stream = corrupt_final_stream
        self._started = False
        self._exhausted = False
        self._closed = False
        self._batch_count = 0
        self._panel_nonzero_count = 0
        self._zero_panel_record_count = 0
        self._panel_umi_total = 0
        self._zero_panel_positions = frozenset(range(zero_panel_record_count))
        self._panel_stream = hashlib.sha256(b"[")

    @property
    def feature_panel(self) -> SciPlex3FeaturePanel:
        return self._panel

    @property
    def access_purpose(self) -> PopulationComponentAccessPurpose:
        return PopulationComponentAccessPurpose.TRAIN_PARAMETERS

    @property
    def source_scan_receipt(self) -> SciPlex3P1SourceScanReceipt:
        return self._initial_receipt

    def describe_partition(self) -> SciPlex3PartitionDescriptor:
        return self._descriptor

    def iter_parameter_training_batches(
        self,
        *,
        batch_size: int = 512,
        partition_id: str = "p1-train",
    ) -> Iterator[SciPlex3SparseCountBatch]:
        if self._started or self._closed or partition_id != "p1-train":
            raise SciPlex3RunnerError("synthetic loader is single-use and p1-only")
        self._started = True
        for batch_index, start in enumerate(range(0, len(self._record_ids), batch_size)):
            stop = min(start + batch_size, len(self._record_ids))
            record_ids = self._record_ids[start:stop]
            wells = tuple(self._well_by_record[record_id] for record_id in record_ids)
            conditions = tuple(self._well_to_condition[well] for well in wells)
            source_rows = np.arange(start, stop, dtype=np.int64)
            feature_indices: list[int] = []
            counts: list[int] = []
            panel_totals: list[int] = []
            indptr = [0]
            for offset, (record_id, well_id, condition_id) in enumerate(
                zip(record_ids, wells, conditions, strict=True)
            ):
                position = start + offset
                is_zero_panel = position in self._zero_panel_positions
                count = 0 if is_zero_panel else (position % 3) + 1
                feature_index = position % SCIPLEX3_FEATURE_COUNT
                pairs: list[list[int]] = []
                if count:
                    feature_indices.append(feature_index)
                    counts.append(count)
                    pairs.append([feature_index, count])
                panel_totals.append(count)
                indptr.append(len(counts))
                if position:
                    self._panel_stream.update(b",")
                self._panel_stream.update(
                    _canonical_bytes(
                        [
                            record_id,
                            position,
                            well_id,
                            condition_id,
                            pairs,
                            count,
                        ]
                    )
                )
            self._batch_count += 1
            self._panel_nonzero_count += len(counts)
            self._zero_panel_record_count += sum(total == 0 for total in panel_totals)
            self._panel_umi_total += sum(counts)
            yield SciPlex3SparseCountBatch(
                partition=self._descriptor,
                batch_index=batch_index,
                record_ids=record_ids,
                composite_well_ids=wells,
                condition_ids=conditions,
                source_row_indices=source_rows,
                indptr=np.asarray(indptr, dtype=np.int64),
                feature_indices=np.asarray(feature_indices, dtype=np.int64),
                counts=np.asarray(counts, dtype=np.int64),
                panel_totals=np.asarray(panel_totals, dtype=np.int64),
            )
        self._panel_stream.update(b"]")
        self._exhausted = True

    def finalize_parameter_training_count_scan(
        self,
    ) -> SciPlex3P1FinalizedCountScanReceipt:
        if not self._exhausted or self._closed:
            raise SciPlex3RunnerError("synthetic count scan was not fully exhausted")
        self._closed = True
        binding_rows = [
            [
                record_id,
                source_row,
                self._well_by_record[record_id],
                self._well_to_condition[self._well_by_record[record_id]],
            ]
            for source_row, record_id in enumerate(self._record_ids)
        ]
        panel_stream_sha256 = self._panel_stream.hexdigest()
        if self._corrupt_final_stream:
            panel_stream_sha256 = "0" * 64
        descriptor_identity = (1, 2, SCIPLEX3_SOURCE_BYTE_COUNT, 3, 4)
        return SciPlex3P1FinalizedCountScanReceipt(
            initial_source_authentication_fingerprint=self._initial_receipt.fingerprint,
            source_sha256=SCIPLEX3_SOURCE_SHA256,
            source_md5=SCIPLEX3_SOURCE_MD5,
            source_byte_count=SCIPLEX3_SOURCE_BYTE_COUNT,
            source_descriptor_identity_before=descriptor_identity,
            source_descriptor_identity_after=descriptor_identity,
            loader_implementation_sha256=SCIPLEX3_LOADER_CODE_SHA256,
            p1_loader_contract_sha256=self._descriptor.loader_contract_sha256,
            dataset_manifest_sha256=self._descriptor.dataset_manifest_sha256,
            query_sha256=self._descriptor.query_sha256,
            benchmark_sha256=self._descriptor.benchmark_sha256,
            target_value_schema_sha256=self._descriptor.target_value_schema_sha256,
            scoring_transform_sha256=self._descriptor.scoring_transform_sha256,
            feature_panel_artifact_sha256=self._descriptor.feature_panel_artifact_sha256,
            ordered_feature_keys_sha256=self._descriptor.ordered_feature_keys_sha256,
            record_ids_sha256=self._descriptor.record_ids_sha256,
            record_to_well_sha256=self._descriptor.record_to_well_sha256,
            emitted_source_row_indices_sha256=_sha256(self._source_rows),
            ordered_record_source_well_condition_sha256=_sha256(binding_rows),
            count_stream_encoding=(
                "canonical_json_utf8_array_of_[record_id,source_row_index,"
                "composite_well_id,condition_id,[[panel_feature_index,count],...],"
                "panel_total]_v1"
            ),
            panel_count_stream_sha256=panel_stream_sha256,
            record_count=SCIPLEX3_P1_RECORD_COUNT,
            well_count=SCIPLEX3_P1_WELL_COUNT,
            treated_well_count=SCIPLEX3_P1_TREATED_WELL_COUNT,
            control_well_count=SCIPLEX3_P1_CONTROL_WELL_COUNT,
            batch_count=self._batch_count,
            panel_nonzero_count=self._panel_nonzero_count,
            zero_panel_record_count=self._zero_panel_record_count,
            panel_umi_total=self._panel_umi_total,
            full_source_umi_total=self._panel_umi_total + self._zero_panel_record_count,
            python_version="3.11.test",
            python_implementation="CPython",
            numpy_version=np.__version__,
            h5py_version="3.11.test",
            hdf5_version="1.14.test",
        )

    def close(self) -> None:
        self._closed = True


@pytest.fixture(scope="module")
def exact_preparation() -> tuple[SciPlex3BaselinePreparation, tuple[Path, ...]]:
    observed_reads: list[Path] = []
    original = runner_module._read_bytes

    def read_spy(path: Path, *, name: str) -> bytes:
        observed_reads.append(Path(path).resolve())
        return original(path, name=name)

    runner_module._read_bytes = read_spy
    try:
        preparation = assemble_sciplex3_p1_training_data(
            _SyntheticExactP1Loader(), REPOSITORY_ROOT, batch_size=4_096
        )
    finally:
        runner_module._read_bytes = original
    return preparation, tuple(observed_reads)


@pytest.fixture(scope="module")
def fitted_and_design(
    exact_preparation: tuple[SciPlex3BaselinePreparation, tuple[Path, ...]],
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[FittedSciPlex3Baseline, SciPlex3P4PredictionDesign, tuple[Path, ...]]:
    _golden_runtime_or_skip()
    preparation, _ = exact_preparation
    fitted = fit_and_write_sciplex3_baseline(
        preparation,
        "exact-condition-rep1-empirical-resampling",
        tmp_path_factory.mktemp("fit") / "empirical",
    )
    observed_reads: list[Path] = []
    original = runner_module._read_bytes

    def read_spy(path: Path, *, name: str) -> bytes:
        observed_reads.append(Path(path).resolve())
        return original(path, name=name)

    runner_module._read_bytes = read_spy
    try:
        design = open_sciplex3_p4_prediction_design(preparation, fitted)
    finally:
        runner_module._read_bytes = original
    return fitted, design, tuple(observed_reads)


def test_assembler_requires_exact_finalized_p1_and_does_not_read_cases_before_fit(
    exact_preparation: tuple[SciPlex3BaselinePreparation, tuple[Path, ...]],
) -> None:
    preparation, observed_reads = exact_preparation
    assert preparation.receipt.record_count == SCIPLEX3_P1_RECORD_COUNT
    assert preparation.receipt.well_count == SCIPLEX3_P1_WELL_COUNT
    assert preparation.receipt.treated_well_count == SCIPLEX3_P1_TREATED_WELL_COUNT
    assert preparation.receipt.control_well_count == SCIPLEX3_P1_CONTROL_WELL_COUNT
    assert preparation.receipt.count_scan_complete is True
    assert preparation.receipt.close_reverification_completed is True
    assert preparation.receipt.zero_panel_record_count == 7
    assert preparation.finalized_count_scan_receipt.zero_panel_record_count == 7
    assert preparation.receipt.runner_panel_count_stream_sha256 == (
        preparation.receipt.loader_panel_count_stream_sha256
    )
    assert preparation.finalized_count_scan_receipt.fingerprint == (
        preparation.receipt.finalized_count_scan_fingerprint
    )
    assert preparation.finalized_count_scan_receipt.loader_implementation_sha256 == (
        SCIPLEX3_LOADER_CODE_SHA256
    )
    assert len(preparation.training_data.wells) == SCIPLEX3_P1_WELL_COUNT
    assert sum(well.counts.row_count for well in preparation.training_data.wells) == (
        SCIPLEX3_P1_RECORD_COUNT
    )
    assert len(preparation.design.actions_by_query_spec) == SCIPLEX3_ACTION_ENTRY_COUNT
    assert not hasattr(preparation.design, "p4_targets")
    assert BENCHMARK_PATH.resolve() not in observed_reads
    assert CASES_PATH.resolve() not in observed_reads
    assert preparation.can_mint_lifecycle_evidence is False
    assert preparation.scientifically_admissible is False


def test_assembler_rejects_a_fabricated_final_count_stream_receipt() -> None:
    with pytest.raises(SciPlex3RunnerError, match="finalized loader receipt"):
        assemble_sciplex3_p1_training_data(
            _SyntheticExactP1Loader(corrupt_final_stream=True),
            REPOSITORY_ROOT,
            batch_size=8_192,
        )


class _EqualityForgingString(str):
    def __eq__(self, other: object) -> bool:
        return True

    def __ne__(self, other: object) -> bool:
        return False

    __hash__ = str.__hash__


class _EqualityForgingInteger(int):
    def __eq__(self, other: object) -> bool:
        return True

    def __ne__(self, other: object) -> bool:
        return False

    __hash__ = int.__hash__


def test_receipt_and_descriptor_constructors_reject_value_identical_scalar_subclasses(
    exact_preparation: tuple[SciPlex3BaselinePreparation, tuple[Path, ...]],
) -> None:
    preparation, _ = exact_preparation
    descriptor = runner_module._exact_p1_descriptor()
    with pytest.raises(ContractViolationError, match="record_count must be an exact integer"):
        replace(descriptor, record_count=_EqualityForgingInteger(descriptor.record_count))

    finalized = preparation.finalized_count_scan_receipt
    source = SciPlex3P1SourceScanReceipt(
        source_sha256=finalized.source_sha256,
        source_md5=finalized.source_md5,
        source_byte_count=finalized.source_byte_count,
        p1_loader_contract_sha256=finalized.p1_loader_contract_sha256,
        feature_panel_artifact_sha256=finalized.feature_panel_artifact_sha256,
        ordered_feature_keys_sha256=finalized.ordered_feature_keys_sha256,
        record_ids_sha256=finalized.record_ids_sha256,
        record_to_well_sha256=finalized.record_to_well_sha256,
        source_row_indices_sha256=preparation.receipt.source_row_indices_sha256,
        ordered_record_source_well_condition_sha256=(
            finalized.ordered_record_source_well_condition_sha256
        ),
        dataset_manifest_sha256=finalized.dataset_manifest_sha256,
        query_sha256=finalized.query_sha256,
        benchmark_sha256=finalized.benchmark_sha256,
        target_value_schema_sha256=finalized.target_value_schema_sha256,
        scoring_transform_sha256=finalized.scoring_transform_sha256,
        matrix_shape=SCIPLEX3_H5AD_SHAPE,
        matrix_nonzero_count=SCIPLEX3_H5AD_NNZ,
        matrix_value_dtype="int64",
    )
    with pytest.raises(ContractViolationError, match="source_sha256 must be an exact string"):
        replace(source, source_sha256=_EqualityForgingString(source.source_sha256))
    with pytest.raises(ContractViolationError, match="panel_umi_total must be an exact integer"):
        replace(
            finalized,
            panel_umi_total=_EqualityForgingInteger(finalized.panel_umi_total),
        )
    with pytest.raises(SciPlex3RunnerError, match="must be an exact integer"):
        replace(
            preparation.receipt,
            panel_umi_total=_EqualityForgingInteger(preparation.receipt.panel_umi_total),
        )


def test_changed_csr_cannot_use_value_identical_nonexact_receipt_scalars(
    exact_preparation: tuple[SciPlex3BaselinePreparation, tuple[Path, ...]],
) -> None:
    preparation, _ = exact_preparation
    changed_well_index = next(
        index for index, well in enumerate(preparation.training_data.wells) if well.counts.nnz > 0
    )
    original_well = preparation.training_data.wells[changed_well_index]
    changed_values = original_well.counts.values.copy()
    changed_values[0] += 1
    changed_counts = ImmutableCSRCounts(
        indptr=original_well.counts.indptr,
        feature_indices=original_well.counts.feature_indices,
        values=changed_values,
        row_count=original_well.counts.row_count,
    )
    changed_wells = list(preparation.training_data.wells)
    changed_wells[changed_well_index] = replace(original_well, counts=changed_counts)
    changed_training = P1TrainingData(
        preparation.training_data.ordered_feature_keys,
        tuple(changed_wells),
    )
    assert not np.array_equal(
        changed_training.wells[changed_well_index].counts.values,
        original_well.counts.values,
    )

    receipt = preparation.receipt
    with pytest.raises(SciPlex3RunnerError, match="exact nonblank trimmed string"):
        replace(
            receipt,
            runner_panel_count_stream_sha256=_EqualityForgingString(
                receipt.runner_panel_count_stream_sha256
            ),
            panel_umi_total=_EqualityForgingInteger(receipt.panel_umi_total),
        )


def test_fit_artifact_binds_final_scan_exact_code_golden_and_no_admission(
    exact_preparation: tuple[SciPlex3BaselinePreparation, tuple[Path, ...]],
    fitted_and_design: tuple[FittedSciPlex3Baseline, SciPlex3P4PredictionDesign, tuple[Path, ...]],
) -> None:
    preparation, _ = exact_preparation
    fitted, _, _ = fitted_and_design
    payload = fitted.artifact.path.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == fitted.artifact.sha256
    manifest = json.loads(payload)
    assert manifest == runner_module._mutable_json_value(fitted.artifact_manifest)
    assert manifest["finalized_count_scan"] == preparation.finalized_count_scan_manifest()
    binding = manifest["executable_binding"]
    assert binding["implementation_code"]["sha256"] == SCIPLEX3_BASELINE_CODE_SHA256
    assert binding["golden_fixture"]["sha256"] == (SCIPLEX3_BASELINE_GOLDEN_FIXTURE_SHA256)
    assert manifest["safety_boundary"] == {
        "baseline_run_status_issued": False,
        "can_mint_lifecycle_evidence": False,
        "heldout_memberships_read": False,
        "heldout_outcomes_read": False,
        "metric_results_issued": False,
        "scientifically_admissible": False,
        "trusted_workflow_receipt_issued": False,
    }
    assert b'"status":"passed"' not in payload


def test_p4_design_opens_only_post_fit_and_maps_all_treated_and_no_action_cases(
    fitted_and_design: tuple[FittedSciPlex3Baseline, SciPlex3P4PredictionDesign, tuple[Path, ...]],
) -> None:
    fitted, design, observed_reads = fitted_and_design
    assert BENCHMARK_PATH.resolve() in observed_reads
    assert CASES_PATH.resolve() in observed_reads
    assert len(design.p4_targets) == SCIPLEX3_P4_CASE_COUNT
    no_action = tuple(
        target for target in design.p4_targets if isinstance(target.condition, NoAction)
    )
    assert len(no_action) == SCIPLEX3_P4_CONTROL_CASE_COUNT
    assert len(design.p4_targets) - len(no_action) == SCIPLEX3_P4_TREATED_CASE_COUNT
    assert design.prediction_targets_sha256 == SCIPLEX3_P4_PREDICTION_TARGETS_SHA256
    assert design.fitted_state_artifact_sha256 == fitted.artifact.sha256
    assert design.heldout_outcomes_read is False
    assert design.can_mint_lifecycle_evidence is False
    assert design.scientifically_admissible is False


class _ForgedBaseline:
    baseline_id = "exact-condition-rep1-empirical-resampling"

    def __init__(self, feature_keys: tuple[str, ...]) -> None:
        self.ordered_feature_keys = feature_keys

    def fitted_state_manifest(self) -> dict[str, object]:
        return {"baseline_id": self.baseline_id, "forged": True}

    def sample(self, request: object) -> PredictiveRawCountSamples:
        raise AssertionError(f"forged baseline must not sample: {request!r}")


class _ManifestDelegatingForgedBaseline:
    """Returns the real state manifest but would fabricate every predictive draw."""

    def __init__(self, real: SciPlex3RawCountBaseline) -> None:
        self._real = real
        self.baseline_id = real.baseline_id
        self.ordered_feature_keys = real.ordered_feature_keys

    def fitted_state_manifest(self) -> dict[str, object]:
        return self._real.fitted_state_manifest()

    def sample(self, request: BaselineSampleRequest) -> PredictiveRawCountSamples:
        return PredictiveRawCountSamples(
            baseline_id=self.baseline_id,
            target=request.target,
            ordered_feature_keys=self.ordered_feature_keys,
            seed=request.seed,
            samples=np.ones((request.sample_count, 2_000), dtype=np.int64),
        )


def test_post_fit_boundary_rejects_unregistered_structural_protocol_object(
    exact_preparation: tuple[SciPlex3BaselinePreparation, tuple[Path, ...]],
    fitted_and_design: tuple[FittedSciPlex3Baseline, SciPlex3P4PredictionDesign, tuple[Path, ...]],
) -> None:
    preparation, _ = exact_preparation
    fitted, _, _ = fitted_and_design
    forged = FittedSciPlex3Baseline(
        baseline=_ForgedBaseline(preparation.training_data.ordered_feature_keys),
        artifact=fitted.artifact,
        artifact_manifest=fitted.artifact_manifest,
        preparation_fingerprint=fitted.preparation_fingerprint,
    )
    with pytest.raises(SciPlex3RunnerError, match="exact registered"):
        open_sciplex3_p4_prediction_design(preparation, forged)


def test_post_fit_boundary_recomputes_exact_class_fitted_state(
    exact_preparation: tuple[SciPlex3BaselinePreparation, tuple[Path, ...]],
    fitted_and_design: tuple[FittedSciPlex3Baseline, SciPlex3P4PredictionDesign, tuple[Path, ...]],
) -> None:
    preparation, _ = exact_preparation
    fitted, _, _ = fitted_and_design
    assert isinstance(fitted.baseline, ExactConditionRep1EmpiricalResampling)
    changed_pools = dict(fitted.baseline._by_condition)
    changed_pools.pop(
        next(condition for condition in changed_pools if not isinstance(condition, NoAction))
    )
    changed_state = replace(fitted.baseline, _by_condition=MappingProxyType(changed_pools))
    forged = FittedSciPlex3Baseline(
        baseline=changed_state,
        artifact=fitted.artifact,
        artifact_manifest=fitted.artifact_manifest,
        preparation_fingerprint=fitted.preparation_fingerprint,
    )
    with pytest.raises(SciPlex3RunnerError, match="in-memory fitted state"):
        open_sciplex3_p4_prediction_design(preparation, forged)


def test_action_binding_rejects_nonexact_scalar_reconstruction() -> None:
    class _IntSubclass(int):
        pass

    class _TextLike:
        def __str__(self) -> str:
            return "source-label:drug@10nM"

    with pytest.raises(SciPlex3RunnerError, match="exact nonblank"):
        SciPlex3ActionBinding(
            source_condition_id=_TextLike(),  # type: ignore[arg-type]
            query_spec_id="query",
            compound="drug",
            dose_nm=10,
            intervention_kind_key="chebi:1",
        )
    with pytest.raises(SciPlex3RunnerError, match="exact supported integer"):
        SciPlex3ActionBinding(
            source_condition_id="source-label:drug@10nM",
            query_spec_id="query",
            compound="drug",
            dose_nm=_IntSubclass(10),
            intervention_kind_key="chebi:1",
        )


def test_writer_rejects_manifest_delegating_malicious_baseline_before_output(
    exact_preparation: tuple[SciPlex3BaselinePreparation, tuple[Path, ...]],
    fitted_and_design: tuple[FittedSciPlex3Baseline, SciPlex3P4PredictionDesign, tuple[Path, ...]],
    tmp_path: Path,
) -> None:
    preparation, _ = exact_preparation
    fitted, design, _ = fitted_and_design
    forged = FittedSciPlex3Baseline(
        baseline=_ManifestDelegatingForgedBaseline(fitted.baseline),
        artifact=fitted.artifact,
        artifact_manifest=fitted.artifact_manifest,
        preparation_fingerprint=fitted.preparation_fingerprint,
    )
    output = tmp_path / "malicious-predictions"
    with pytest.raises(SciPlex3RunnerError, match="exact registered"):
        SciPlex3PredictionArtifactWriter(preparation, forged, design, output)
    assert not output.exists()


def test_prediction_writer_has_only_internal_generation_and_reauthenticates_shards(
    exact_preparation: tuple[SciPlex3BaselinePreparation, tuple[Path, ...]],
    fitted_and_design: tuple[FittedSciPlex3Baseline, SciPlex3P4PredictionDesign, tuple[Path, ...]],
    tmp_path: Path,
) -> None:
    preparation, _ = exact_preparation
    fitted, design, _ = fitted_and_design
    writer = SciPlex3PredictionArtifactWriter(preparation, fitted, design, tmp_path / "predictions")
    assert not hasattr(writer, "write")
    assert SCIPLEX3_BASELINE_SEEDS == (0, 1, 2, 3, 4)
    entries = writer.sample_and_write(design.p4_targets[0], 0)
    assert len(entries) == (
        SCIPLEX3_BASELINE_SAMPLES_PER_CASE_PER_SEED // SCIPLEX3_PREDICTION_SHARD_DRAW_COUNT
    )
    assert all(entry.shape == (SCIPLEX3_PREDICTION_SHARD_DRAW_COUNT, 2_000) for entry in entries)
    assert all(entry.dtype == "<i8" for entry in entries)
    with pytest.raises(SciPlex3RunnerError, match="duplicate prediction"):
        writer.sample_and_write(design.p4_targets[0], 0)
    with pytest.raises(SciPlex3RunnerError, match="target or seed"):
        writer.sample_and_write(design.p4_targets[0], 5)
    with pytest.raises(SciPlex3RunnerError, match="manifest remains sealed"):
        writer.finalize()
    first_path = tmp_path / "predictions" / entries[0].relative_path
    first_path.write_bytes(b"corrupt")
    with pytest.raises(SciPlex3RunnerError, match="SHA-256 drift"):
        writer._reauthenticate_shards(entries)


def test_same_class_mutable_state_is_copied_before_writer_verification(
    exact_preparation: tuple[SciPlex3BaselinePreparation, tuple[Path, ...]],
    fitted_and_design: tuple[FittedSciPlex3Baseline, SciPlex3P4PredictionDesign, tuple[Path, ...]],
    tmp_path: Path,
) -> None:
    preparation, _ = exact_preparation
    fitted, design, _ = fitted_and_design
    assert isinstance(fitted.baseline, ExactConditionRep1EmpiricalResampling)
    caller_pools = {key: list(value) for key, value in fitted.baseline._by_condition.items()}
    reconstructed = ExactConditionRep1EmpiricalResampling(
        fitted.baseline.ordered_feature_keys,
        caller_pools,  # type: ignore[arg-type]
    )
    rebound = FittedSciPlex3Baseline(
        baseline=reconstructed,
        artifact=fitted.artifact,
        artifact_manifest=fitted.artifact_manifest,
        preparation_fingerprint=fitted.preparation_fingerprint,
    )
    writer = SciPlex3PredictionArtifactWriter(
        preparation, rebound, design, tmp_path / "copied-state"
    )
    caller_pools.clear()
    entries = writer.sample_and_write(design.p4_targets[0], 0)
    assert len(entries) == 4


def test_writer_reauthenticates_fitted_state_immediately_before_sampling(
    exact_preparation: tuple[SciPlex3BaselinePreparation, tuple[Path, ...]],
    fitted_and_design: tuple[FittedSciPlex3Baseline, SciPlex3P4PredictionDesign, tuple[Path, ...]],
    tmp_path: Path,
) -> None:
    preparation, _ = exact_preparation
    fitted, design, _ = fitted_and_design
    writer = SciPlex3PredictionArtifactWriter(
        preparation, fitted, design, tmp_path / "state-reauth"
    )
    assert isinstance(fitted.baseline, ExactConditionRep1EmpiricalResampling)
    changed_pools = dict(fitted.baseline._by_condition)
    changed_pools.pop(
        next(condition for condition in changed_pools if not isinstance(condition, NoAction))
    )
    changed = replace(fitted.baseline, _by_condition=changed_pools)
    writer._fitted = replace(fitted, baseline=changed)
    with pytest.raises(SciPlex3RunnerError, match="in-memory fitted state"):
        writer.sample_and_write(design.p4_targets[0], 0)
    assert writer.shard_entries == ()


def test_writer_rehashes_exact_target_design_after_initialization(
    exact_preparation: tuple[SciPlex3BaselinePreparation, tuple[Path, ...]],
    fitted_and_design: tuple[FittedSciPlex3Baseline, SciPlex3P4PredictionDesign, tuple[Path, ...]],
    tmp_path: Path,
) -> None:
    class _MutableCondition:
        def __init__(self) -> None:
            self.compound = "drug"
            self.dose_nm = 10

    preparation, _ = exact_preparation
    fitted, design, _ = fitted_and_design
    local_targets = tuple(replace(target) for target in design.p4_targets)
    local_design = replace(design, p4_targets=local_targets)
    writer = SciPlex3PredictionArtifactWriter(
        preparation, fitted, local_design, tmp_path / "target-reauth"
    )
    object.__setattr__(local_targets[0], "condition", _MutableCondition())
    with pytest.raises(SciPlex3RunnerError, match="condition"):
        writer.sample_and_write(local_targets[0], 0)
    assert writer.shard_entries == ()


def test_prediction_target_constructor_rejects_mutable_condition_lookalike() -> None:
    class _MutableCondition:
        compound = "drug"
        dose_nm = 10

    with pytest.raises(SciPlex3BaselineError, match="exact immutable"):
        PredictionTarget(
            case_id="case",
            target_well_id="well",
            plate_id="plate",
            partition_id="p4-untouched-test",
            condition=_MutableCondition(),  # type: ignore[arg-type]
        )


def test_public_runner_dataclasses_snapshot_mutable_container_inputs(
    exact_preparation: tuple[SciPlex3BaselinePreparation, tuple[Path, ...]],
    fitted_and_design: tuple[FittedSciPlex3Baseline, SciPlex3P4PredictionDesign, tuple[Path, ...]],
) -> None:
    preparation, _ = exact_preparation
    fitted, design, _ = fitted_and_design
    source_actions = dict(preparation.design.actions_by_source_condition)
    query_actions = dict(preparation.design.actions_by_query_spec)
    copied_p1_design = replace(
        preparation.design,
        actions_by_source_condition=source_actions,
        actions_by_query_spec=query_actions,
    )
    targets = list(design.p4_targets)
    copied_p4_design = replace(design, p4_targets=targets)  # type: ignore[arg-type]
    manifest = json.loads(fitted.artifact.path.read_bytes())
    caller_runtime = manifest["runtime"]
    caller_roles = manifest["finalized_count_scan"]["accessed_partition_roles"]
    copied_fitted = FittedSciPlex3Baseline(
        baseline=fitted.baseline,
        artifact=fitted.artifact,
        artifact_manifest=manifest,
        preparation_fingerprint=fitted.preparation_fingerprint,
    )
    shape = [128, SCIPLEX3_FEATURE_COUNT]
    shard = PredictionShardEntry(
        relative_path="shards/case-0000/seed-0/draws-000-127.i64le",
        sha256="0" * 64,
        byte_count=128 * SCIPLEX3_FEATURE_COUNT * 8,
        baseline_id=fitted.baseline.baseline_id,
        case_id=design.p4_targets[0].case_id,
        target_well_id=design.p4_targets[0].target_well_id,
        partition_id="p4-untouched-test",
        seed=0,
        rng_algorithm="numpy-pcg64dxsm-v1",
        draw_start=0,
        draw_stop_exclusive=128,
        shape=shape,  # type: ignore[arg-type]
        dtype="<i8",
        ordered_feature_keys_sha256=SCIPLEX3_ORDERED_FEATURE_KEYS_SHA256,
    )

    source_actions.clear()
    query_actions.clear()
    targets.clear()
    caller_runtime["python_version"] = "mutated"
    caller_roles[0] = "mutated"
    manifest.clear()
    shape[0] = 1
    assert len(copied_p1_design.actions_by_source_condition) == SCIPLEX3_ACTION_ENTRY_COUNT
    assert len(copied_p1_design.actions_by_query_spec) == SCIPLEX3_ACTION_ENTRY_COUNT
    assert len(copied_p4_design.p4_targets) == SCIPLEX3_P4_CASE_COUNT
    assert copied_fitted.artifact_manifest
    assert shard.shape == (128, SCIPLEX3_FEATURE_COUNT)
    frozen_runtime = copied_fitted.artifact_manifest["runtime"]
    frozen_scan = copied_fitted.artifact_manifest["finalized_count_scan"]
    assert isinstance(frozen_runtime, Mapping)
    assert isinstance(frozen_scan, Mapping)
    frozen_roles = frozen_scan["accessed_partition_roles"]
    assert isinstance(frozen_roles, tuple)
    with pytest.raises(TypeError):
        frozen_runtime["python_version"] = "mutated"  # type: ignore[index]
    with pytest.raises(TypeError):
        frozen_roles[0] = "mutated"  # type: ignore[index]


def test_public_reconstruction_cannot_bypass_finalized_scan_gate(
    exact_preparation: tuple[SciPlex3BaselinePreparation, tuple[Path, ...]],
    tmp_path: Path,
) -> None:
    preparation, _ = exact_preparation
    corrupted_final = replace(
        preparation.finalized_count_scan_receipt,
        close_reverification_completed=False,
    )
    forged_receipt = replace(
        preparation.receipt,
        finalized_count_scan_fingerprint=corrupted_final.fingerprint,
    )
    forged = SciPlex3BaselinePreparation(
        training_data=preparation.training_data,
        receipt=forged_receipt,
        finalized_count_scan_receipt=corrupted_final,
        design=preparation.design,
        repository_root=preparation.repository_root,
    )
    with pytest.raises(SciPlex3RunnerError, match="baseline preparation"):
        fit_and_write_sciplex3_baseline(
            forged,
            "exact-condition-rep1-empirical-resampling",
            tmp_path / "forged-fit",
        )


def test_fit_rejects_contradictory_reconstructed_finalized_receipt(
    exact_preparation: tuple[SciPlex3BaselinePreparation, tuple[Path, ...]],
    tmp_path: Path,
) -> None:
    preparation, _ = exact_preparation
    corrupted_final = replace(
        preparation.finalized_count_scan_receipt,
        accessed_partition_roles=("p4-untouched-test",),
        record_count=1,
        source_sha256="0" * 64,
    )
    forged_receipt = replace(
        preparation.receipt,
        finalized_count_scan_fingerprint=corrupted_final.fingerprint,
    )
    forged = replace(
        preparation,
        receipt=forged_receipt,
        finalized_count_scan_receipt=corrupted_final,
    )
    with pytest.raises(SciPlex3RunnerError, match="finalized receipt"):
        fit_and_write_sciplex3_baseline(
            forged,
            "exact-condition-rep1-empirical-resampling",
            tmp_path / "contradictory-final-receipt",
        )


def test_fit_reloads_exact_p1_safe_design_instead_of_trusting_public_mapping(
    exact_preparation: tuple[SciPlex3BaselinePreparation, tuple[Path, ...]],
    tmp_path: Path,
) -> None:
    preparation, _ = exact_preparation
    condition_id, action = next(iter(preparation.design.actions_by_source_condition.items()))
    forged_action = replace(action, query_spec_id=f"forged-{action.query_spec_id}")
    by_condition = dict(preparation.design.actions_by_source_condition)
    by_spec = dict(preparation.design.actions_by_query_spec)
    by_condition[condition_id] = forged_action
    del by_spec[action.query_spec_id]
    by_spec[forged_action.query_spec_id] = forged_action
    forged_design = replace(
        preparation.design,
        actions_by_source_condition=by_condition,
        actions_by_query_spec=by_spec,
    )
    forged = replace(preparation, design=forged_design)
    with pytest.raises(SciPlex3RunnerError, match="query/action/scoring"):
        fit_and_write_sciplex3_baseline(
            forged,
            "exact-condition-rep1-empirical-resampling",
            tmp_path / "forged-design-fit",
        )


def test_fit_rejects_shadowed_or_changed_loaded_implementation(
    exact_preparation: tuple[SciPlex3BaselinePreparation, tuple[Path, ...]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _golden_runtime_or_skip()
    preparation, _ = exact_preparation
    shadow = tmp_path / "shadowed_sciplex3_baselines.py"
    shadow.write_bytes(
        (REPOSITORY_ROOT / "src/cellstate/evaluation/sciplex3_baselines.py").read_bytes()
    )
    monkeypatch.setattr(runner_module, "_IMPORTED_BASELINE_CODE_PATH", shadow.resolve())
    with pytest.raises(SciPlex3RunnerError, match="loaded implementation path"):
        fit_and_write_sciplex3_baseline(
            preparation,
            "exact-condition-rep1-empirical-resampling",
            tmp_path / "shadowed-fit",
        )
    monkeypatch.undo()
    monkeypatch.setattr(runner_module, "_IMPORTED_RUNNER_CODE_SHA256", "0" * 64)
    with pytest.raises(SciPlex3RunnerError, match="changed since import"):
        fit_and_write_sciplex3_baseline(
            preparation,
            "exact-condition-rep1-empirical-resampling",
            tmp_path / "changed-runner-fit",
        )


def test_fit_rehashes_in_memory_csr_against_authenticated_count_stream(
    exact_preparation: tuple[SciPlex3BaselinePreparation, tuple[Path, ...]],
    tmp_path: Path,
) -> None:
    preparation, _ = exact_preparation
    original_well = preparation.training_data.wells[0]
    altered_values = np.asarray(original_well.counts.values, dtype=np.int64).copy()
    altered_values[0] += 1
    altered_counts = ImmutableCSRCounts(
        indptr=original_well.counts.indptr,
        feature_indices=original_well.counts.feature_indices,
        values=altered_values,
        row_count=original_well.counts.row_count,
    )
    altered_well = replace(original_well, counts=altered_counts)
    altered_training = P1TrainingData(
        ordered_feature_keys=preparation.training_data.ordered_feature_keys,
        wells=(altered_well, *preparation.training_data.wells[1:]),
    )
    forged = replace(preparation, training_data=altered_training)
    with pytest.raises(SciPlex3RunnerError, match="in-memory CSR"):
        fit_and_write_sciplex3_baseline(
            forged,
            "exact-condition-rep1-empirical-resampling",
            tmp_path / "altered-count-fit",
        )


def test_prediction_design_mappings_are_immutable(
    fitted_and_design: tuple[FittedSciPlex3Baseline, SciPlex3P4PredictionDesign, tuple[Path, ...]],
    exact_preparation: tuple[SciPlex3BaselinePreparation, tuple[Path, ...]],
) -> None:
    _, design, _ = fitted_and_design
    preparation, _ = exact_preparation
    assert isinstance(preparation.design.actions_by_query_spec, Mapping)
    assert isinstance(preparation.design.actions_by_query_spec, MappingProxyType)
    with pytest.raises(TypeError):
        preparation.design.actions_by_query_spec["forged"] = next(  # type: ignore[index]
            iter(preparation.design.actions_by_query_spec.values())
        )
    assert tuple(target.case_id for target in design.p4_targets) == tuple(
        sorted(target.case_id for target in design.p4_targets)
    )
