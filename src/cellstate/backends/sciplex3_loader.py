"""Immutable, partition-aware reader for the frozen sci-Plex3 K562 source.

The loader is deliberately narrower than a biological backend.  It authenticates one corrected
H5AD, proves its design rows against the checked-in preparation ledgers, and yields raw count
batches on the frozen TRAIN-derived feature panel.  It does not fit a model, score a benchmark,
construct a public biological response, or construct a :class:`CellStateBelief`.

``h5py`` remains optional.  It is imported only when :meth:`SciPlex3K562H5ADLoader.open` is called.
The source is opened once: the exact seekable binary object that is hashed is then passed to
``h5py.File``.  On close, after the HDF5 handle is released, the same descriptor is hashed again
and its filesystem identity is rechecked.  Any uncertainty or drift fails closed.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import platform
import sys
from collections.abc import Iterator, Mapping
from contextlib import suppress
from dataclasses import asdict, dataclass
from pathlib import Path
from types import TracebackType
from typing import Any, BinaryIO, Protocol, Self, cast, runtime_checkable

import numpy as np
import numpy.typing as npt

from cellstate.backends.sciplex3_k562 import (
    SCIPLEX3_K562_BENCHMARK_SHA256,
    SCIPLEX3_K562_MANIFEST_SHA256,
    SCIPLEX3_K562_QUERY_SHA256,
    SCIPLEX3_K562_TARGET_VALUE_SCHEMA_SHA256,
    PopulationComponentAccessPurpose,
)
from cellstate.data.benchmarks import BenchmarkPartitionRole
from cellstate.errors import ContractViolationError

SCIPLEX3_SOURCE_FILENAME = "SrivatsanTrapnell2020_sciplex3.h5ad"
SCIPLEX3_SOURCE_BYTE_COUNT = 2_526_631_614
SCIPLEX3_SOURCE_MD5 = "c9e70629505d98c7ca1a837f62b14e89"
SCIPLEX3_SOURCE_SHA256 = "603ed16c5e25401c8a7f5bb0b2b045179701017d65dcfc6aeea71722a66cd10a"

SCIPLEX3_H5AD_SHAPE = (799_317, 110_983)
SCIPLEX3_H5AD_NNZ = 1_007_419_688
SCIPLEX3_MATRIX_DTYPE = "int64"
SCIPLEX3_FEATURE_COUNT = 2_000
SCIPLEX3_ORDERED_FEATURE_KEYS_SHA256 = (
    "8b9e0e71d9bfc79a5e1db29f73bc30006bf117a29db8abf63b1607403629401f"
)
SCIPLEX3_SOURCE_FEATURE_AXIS_SHA256 = (
    "4fef9247abbb9053260d02f24b8fc6ace057fcda966b8fd0d7bc5daa674aa313"
)
SCIPLEX3_P1_LOADER_CONTRACT_SHA256 = (
    "3de5be54b60ba1403995ba79d122ee8232218be5c027da1bf530cb610ae80f90"
)
SCIPLEX3_FEATURE_PANEL_ARTIFACT_SHA256 = (
    "3935f749cf7feb45883f8ddadcbbbffc71749c7452126e7f53eba16f32585b66"
)
SCIPLEX3_SCORING_TRANSFORM_SHA256 = (
    "6968740ed833e482b83b73d6315c9ef4c1caca0ba1d9bb92ce09ffdca86c8f57"
)

_PREPARATION_DIRECTORY = Path("benchmarks/artifacts/sciplex3-k562-24h-v1")
_P1_LOADER_CONTRACT_PATH = Path(
    "benchmarks/vertical-a/sciplex3-k562-24h-v1/support/p1-loader-contract.json"
)
_SOURCE_VERIFICATION = "source-verification.json"
_FEATURE_PANEL = "feature-panel.json"

_REQUIRED_OBSERVATION_FIELDS = frozenset(
    {
        "_index",
        "cell_line",
        "dose_unit",
        "dose_value",
        "ncounts",
        "perturbation",
        "plate",
        "replicate",
        "time",
        "well",
    }
)
_REQUIRED_FEATURE_FIELDS = frozenset({"ensembl_id", "gene_symbol"})
_SUPPORTED_DOSES_NM = frozenset({10, 100, 1_000, 10_000})

_PURPOSE_BINDINGS: Mapping[
    PopulationComponentAccessPurpose,
    tuple[str, str, BenchmarkPartitionRole],
] = {
    PopulationComponentAccessPurpose.TRAIN_PARAMETERS: (
        "p1-train",
        "train",
        BenchmarkPartitionRole.TRAIN,
    ),
    PopulationComponentAccessPurpose.FIT_CALIBRATION: (
        "p2-calibration",
        "calibration",
        BenchmarkPartitionRole.CALIBRATION,
    ),
    PopulationComponentAccessPurpose.MODEL_SELECTION: (
        "p3-model-selection-validation",
        "model_selection_validation",
        BenchmarkPartitionRole.MODEL_SELECTION_VALIDATION,
    ),
    PopulationComponentAccessPurpose.UNTOUCHED_EVALUATION: (
        "p4-untouched-test",
        "untouched_test",
        BenchmarkPartitionRole.UNTOUCHED_TEST,
    ),
}

_EXPECTED_P1_RECORD_COUNT = 94_785
_EXPECTED_P1_WELL_COUNT = 768
_EXPECTED_P1_TREATED_WELL_COUNT = 752
_EXPECTED_P1_CONTROL_WELL_COUNT = 16
_P1_PLATES = tuple(f"plate{index}" for index in range(1, 9))

_COUNT_STREAM_ENCODING = (
    "canonical_json_utf8_array_of_"
    "[record_id,source_row_index,composite_well_id,condition_id,"
    "[[panel_feature_index,count],...],panel_total]_v1"
)
_ACCESSED_COUNT_DATASETS = ("X.data", "X.indices", "X.indptr", "obs.ncounts")


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _require_exact_primitive_fields(
    value: object,
    *,
    string_fields: tuple[str, ...] = (),
    integer_fields: tuple[str, ...] = (),
    boolean_fields: tuple[str, ...] = (),
) -> None:
    for name in string_fields:
        item = getattr(value, name)
        if type(item) is not str:
            raise ContractViolationError(f"{name} must be an exact string")
    for name in integer_fields:
        if type(getattr(value, name)) is not int:
            raise ContractViolationError(f"{name} must be an exact integer")
    for name in boolean_fields:
        if type(getattr(value, name)) is not bool:
            raise ContractViolationError(f"{name} must be an exact boolean")


def _require_exact_tuple(
    value: object,
    *,
    name: str,
    item_type: type[object],
    length: int | None = None,
) -> None:
    if type(value) is not tuple:
        raise ContractViolationError(f"{name} must be an exact tuple")
    items = cast(tuple[object, ...], value)
    if length is not None and len(items) != length:
        raise ContractViolationError(f"{name} has the wrong tuple length")
    if any(type(item) is not item_type for item in items):
        raise ContractViolationError(f"{name} contains a non-exact scalar")


def _decimal_string_membership_payload(rows: npt.ArrayLike) -> bytes:
    """Reproduce the preparation artifact's unordered decimal-string membership encoding.

    The original feature-selection artifact used ``hash_string_array`` with its default
    lexicographic sorting.  That hash proves the selected source-row *set*; it is deliberately
    separate from the numeric source-axis and record-emission order digests carried by loader
    receipts.
    """

    values = [str(int(row)) for row in np.asarray(rows, dtype=np.int64).tolist()]
    return _canonical_json_bytes(sorted(values))


def _read_loader_implementation_sha256() -> str:
    try:
        return _sha256(Path(__file__).read_bytes())
    except OSError as error:
        raise ContractViolationError("cannot authenticate the loader implementation") from error


_LOADER_IMPLEMENTATION_IMPORT_SHA256 = _read_loader_implementation_sha256()


def _decode_scalar(value: object) -> str:
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ContractViolationError("H5AD contains non-UTF-8 metadata") from error
    return str(value)


def _readonly_int64(values: npt.ArrayLike) -> npt.NDArray[np.int64]:
    raw = np.asarray(values)
    if raw.dtype.kind not in {"i", "u"}:
        raise ContractViolationError("immutable loader arrays must contain exact integer values")
    if raw.dtype.kind == "u" and bool(np.any(raw > np.iinfo(np.int64).max)):
        raise ContractViolationError("immutable loader array exceeds signed 64-bit range")
    contiguous = np.ascontiguousarray(raw, dtype=np.int64)
    return np.frombuffer(contiguous.tobytes(), dtype=np.int64).reshape(contiguous.shape)


@dataclass(frozen=True)
class SciPlex3FeaturePanel:
    """Exact ordered TRAIN-derived feature surface used by every count batch."""

    source_feature_indices: tuple[int, ...]
    ordered_feature_keys: tuple[str, ...]
    ordered_feature_keys_sha256: str
    source_feature_axis_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_feature_indices", tuple(self.source_feature_indices))
        object.__setattr__(self, "ordered_feature_keys", tuple(self.ordered_feature_keys))
        _require_exact_tuple(
            self.source_feature_indices,
            name="source_feature_indices",
            item_type=int,
        )
        _require_exact_tuple(
            self.ordered_feature_keys,
            name="ordered_feature_keys",
            item_type=str,
        )
        _require_exact_primitive_fields(
            self,
            string_fields=("ordered_feature_keys_sha256", "source_feature_axis_sha256"),
        )
        if len(self.source_feature_indices) != SCIPLEX3_FEATURE_COUNT:
            raise ContractViolationError(
                "sci-Plex3 feature panel does not contain exactly 2,000 rows"
            )
        if len(self.ordered_feature_keys) != SCIPLEX3_FEATURE_COUNT:
            raise ContractViolationError("sci-Plex3 feature-key axis does not contain 2,000 rows")
        if len(set(self.source_feature_indices)) != SCIPLEX3_FEATURE_COUNT:
            raise ContractViolationError("sci-Plex3 source feature indices are not unique")
        if len(set(self.ordered_feature_keys)) != SCIPLEX3_FEATURE_COUNT:
            raise ContractViolationError("sci-Plex3 ordered feature keys are not unique")
        if any(index < 0 for index in self.source_feature_indices):
            raise ContractViolationError("sci-Plex3 source feature indices must be nonnegative")
        if any(not key or key != key.strip() for key in self.ordered_feature_keys):
            raise ContractViolationError("sci-Plex3 ordered feature keys must be nonblank/trimmed")


@dataclass(frozen=True)
class SciPlex3PartitionDescriptor:
    """Read-only identity and lifecycle provenance for one frozen partition."""

    partition_id: str
    artifact_role: str
    benchmark_role: BenchmarkPartitionRole
    access_purpose: PopulationComponentAccessPurpose
    record_count: int
    well_count: int
    record_ids_sha256: str
    record_to_well_sha256: str
    source_sha256: str
    loader_contract_sha256: str
    dataset_manifest_sha256: str
    query_sha256: str
    benchmark_sha256: str
    target_value_schema_sha256: str
    scoring_transform_sha256: str
    feature_panel_artifact_sha256: str
    ordered_feature_keys_sha256: str
    count_access_sealed: bool
    can_mint_lifecycle_evidence: bool = False
    scientifically_admissible: bool = False

    def __post_init__(self) -> None:
        _require_exact_primitive_fields(
            self,
            string_fields=(
                "partition_id",
                "artifact_role",
                "record_ids_sha256",
                "record_to_well_sha256",
                "source_sha256",
                "loader_contract_sha256",
                "dataset_manifest_sha256",
                "query_sha256",
                "benchmark_sha256",
                "target_value_schema_sha256",
                "scoring_transform_sha256",
                "feature_panel_artifact_sha256",
                "ordered_feature_keys_sha256",
            ),
            integer_fields=("record_count", "well_count"),
            boolean_fields=(
                "count_access_sealed",
                "can_mint_lifecycle_evidence",
                "scientifically_admissible",
            ),
        )
        if type(self.benchmark_role) is not BenchmarkPartitionRole:
            raise ContractViolationError("benchmark_role must be an exact BenchmarkPartitionRole")
        if type(self.access_purpose) is not PopulationComponentAccessPurpose:
            raise ContractViolationError(
                "access_purpose must be an exact PopulationComponentAccessPurpose"
            )


@dataclass(frozen=True)
class SciPlex3SparseCountBatch:
    """One immutable CSR batch on panel-position columns ``[0, 2000)``.

    Arrays are copied into immutable ``bytes``-backed buffers.  A caller therefore cannot mutate
    the loader's storage, a later batch, or a previously handed-off batch by toggling NumPy's
    ``WRITEABLE`` flag.
    """

    partition: SciPlex3PartitionDescriptor
    batch_index: int
    record_ids: tuple[str, ...]
    composite_well_ids: tuple[str, ...]
    condition_ids: tuple[str, ...]
    source_row_indices: npt.NDArray[np.int64]
    indptr: npt.NDArray[np.int64]
    feature_indices: npt.NDArray[np.int64]
    counts: npt.NDArray[np.int64]
    panel_totals: npt.NDArray[np.int64]

    def __post_init__(self) -> None:
        if type(self.partition) is not SciPlex3PartitionDescriptor:
            raise ContractViolationError("sparse batch partition must use the exact descriptor")
        if type(self.batch_index) is not int or self.batch_index < 0:
            raise ContractViolationError("sparse batch index must be an exact nonnegative integer")
        object.__setattr__(self, "record_ids", tuple(self.record_ids))
        object.__setattr__(self, "composite_well_ids", tuple(self.composite_well_ids))
        object.__setattr__(self, "condition_ids", tuple(self.condition_ids))
        for name in ("record_ids", "composite_well_ids", "condition_ids"):
            values = getattr(self, name)
            _require_exact_tuple(values, name=name, item_type=str)
            if any(not value or value != value.strip() for value in values):
                raise ContractViolationError(f"{name} must contain nonblank trimmed strings")
        object.__setattr__(self, "source_row_indices", _readonly_int64(self.source_row_indices))
        object.__setattr__(self, "indptr", _readonly_int64(self.indptr))
        object.__setattr__(self, "feature_indices", _readonly_int64(self.feature_indices))
        object.__setattr__(self, "counts", _readonly_int64(self.counts))
        object.__setattr__(self, "panel_totals", _readonly_int64(self.panel_totals))
        row_count = len(self.record_ids)
        if row_count == 0:
            raise ContractViolationError("sparse count batches cannot be empty")
        if not (
            len(self.composite_well_ids)
            == len(self.condition_ids)
            == len(self.source_row_indices)
            == len(self.panel_totals)
            == row_count
        ):
            raise ContractViolationError("sparse count batch row metadata are misaligned")
        if self.indptr.shape != (row_count + 1,):
            raise ContractViolationError("sparse count batch indptr has the wrong shape")
        if int(self.indptr[0]) != 0 or np.any(np.diff(self.indptr) < 0):
            raise ContractViolationError("sparse count batch indptr is malformed")
        if int(self.indptr[-1]) != len(self.counts) or len(self.counts) != len(
            self.feature_indices
        ):
            raise ContractViolationError("sparse count batch CSR arrays are misaligned")
        if np.any(self.source_row_indices < 0):
            raise ContractViolationError("sparse count batch source rows cannot be negative")
        if np.any(self.counts < 0):
            raise ContractViolationError("sparse count batch contains negative UMI counts")
        if np.any(self.panel_totals < 0):
            raise ContractViolationError("panel count-vector totals cannot be negative")
        if np.any(self.feature_indices < 0) or np.any(
            self.feature_indices >= SCIPLEX3_FEATURE_COUNT
        ):
            raise ContractViolationError("sparse count batch contains an out-of-panel column")
        for row in range(row_count):
            start = int(self.indptr[row])
            stop = int(self.indptr[row + 1])
            if stop - start > 1 and np.any(np.diff(self.feature_indices[start:stop]) <= 0):
                raise ContractViolationError("sparse batch columns must be unique and ordered")
            if int(self.counts[start:stop].sum(dtype=np.int64)) != int(self.panel_totals[row]):
                raise ContractViolationError("sparse batch panel total does not match its CSR row")

    @property
    def shape(self) -> tuple[int, int]:
        return (len(self.record_ids), SCIPLEX3_FEATURE_COUNT)


@dataclass(frozen=True)
class SciPlex3P1SourceScanReceipt:
    """Initial source/p1-metadata authentication, before counts or close verification.

    Despite the retained compatibility name, this is deliberately not the completed count-scan
    receipt.  ``count_scan_complete`` and ``close_reverification_completed`` are invariantly
    false.  Only :class:`SciPlex3P1FinalizedCountScanReceipt`, issued by
    :meth:`SciPlex3K562H5ADLoader.finalize_parameter_training_count_scan`, proves full p1 count
    coverage and successful final source authentication.
    """

    source_sha256: str
    source_md5: str
    source_byte_count: int
    p1_loader_contract_sha256: str
    feature_panel_artifact_sha256: str
    ordered_feature_keys_sha256: str
    record_ids_sha256: str
    record_to_well_sha256: str
    source_row_indices_sha256: str
    ordered_record_source_well_condition_sha256: str
    dataset_manifest_sha256: str
    query_sha256: str
    benchmark_sha256: str
    target_value_schema_sha256: str
    scoring_transform_sha256: str
    matrix_shape: tuple[int, int]
    matrix_nonzero_count: int
    matrix_value_dtype: str
    partition_id: str = "p1-train"
    access_purpose: str = "train_parameters"
    loader_interface_id: str = "cellstate.sciplex3-training-data-loader.v1"
    heldout_memberships_parsed: bool = False
    heldout_outcome_values_parsed: bool = False
    lifecycle_evidence_issued: bool = False
    scientifically_admissible: bool = False
    trusted_workflow_receipt_present: bool = False
    close_reverification_required: bool = True
    count_scan_complete: bool = False
    close_reverification_completed: bool = False
    count_records_consumed: int = 0
    count_batches_consumed: int = 0

    def __post_init__(self) -> None:
        _require_exact_primitive_fields(
            self,
            string_fields=(
                "source_sha256",
                "source_md5",
                "p1_loader_contract_sha256",
                "feature_panel_artifact_sha256",
                "ordered_feature_keys_sha256",
                "record_ids_sha256",
                "record_to_well_sha256",
                "source_row_indices_sha256",
                "ordered_record_source_well_condition_sha256",
                "dataset_manifest_sha256",
                "query_sha256",
                "benchmark_sha256",
                "target_value_schema_sha256",
                "scoring_transform_sha256",
                "matrix_value_dtype",
                "partition_id",
                "access_purpose",
                "loader_interface_id",
            ),
            integer_fields=(
                "source_byte_count",
                "matrix_nonzero_count",
                "count_records_consumed",
                "count_batches_consumed",
            ),
            boolean_fields=(
                "heldout_memberships_parsed",
                "heldout_outcome_values_parsed",
                "lifecycle_evidence_issued",
                "scientifically_admissible",
                "trusted_workflow_receipt_present",
                "close_reverification_required",
                "count_scan_complete",
                "close_reverification_completed",
            ),
        )
        _require_exact_tuple(self.matrix_shape, name="matrix_shape", item_type=int, length=2)

    @property
    def fingerprint(self) -> str:
        return _sha256(
            _canonical_json_bytes(
                {
                    "access_purpose": self.access_purpose,
                    "benchmark_sha256": self.benchmark_sha256,
                    "close_reverification_required": self.close_reverification_required,
                    "close_reverification_completed": self.close_reverification_completed,
                    "count_batches_consumed": self.count_batches_consumed,
                    "count_records_consumed": self.count_records_consumed,
                    "count_scan_complete": self.count_scan_complete,
                    "dataset_manifest_sha256": self.dataset_manifest_sha256,
                    "feature_panel_artifact_sha256": self.feature_panel_artifact_sha256,
                    "heldout_memberships_parsed": self.heldout_memberships_parsed,
                    "heldout_outcome_values_parsed": self.heldout_outcome_values_parsed,
                    "lifecycle_evidence_issued": self.lifecycle_evidence_issued,
                    "loader_interface_id": self.loader_interface_id,
                    "matrix_nonzero_count": self.matrix_nonzero_count,
                    "matrix_shape": list(self.matrix_shape),
                    "matrix_value_dtype": self.matrix_value_dtype,
                    "ordered_feature_keys_sha256": self.ordered_feature_keys_sha256,
                    "ordered_record_source_well_condition_sha256": (
                        self.ordered_record_source_well_condition_sha256
                    ),
                    "p1_loader_contract_sha256": self.p1_loader_contract_sha256,
                    "partition_id": self.partition_id,
                    "query_sha256": self.query_sha256,
                    "record_ids_sha256": self.record_ids_sha256,
                    "record_to_well_sha256": self.record_to_well_sha256,
                    "scientifically_admissible": self.scientifically_admissible,
                    "scoring_transform_sha256": self.scoring_transform_sha256,
                    "source_byte_count": self.source_byte_count,
                    "source_md5": self.source_md5,
                    "source_row_indices_sha256": self.source_row_indices_sha256,
                    "source_sha256": self.source_sha256,
                    "target_value_schema_sha256": self.target_value_schema_sha256,
                    "trusted_workflow_receipt_present": self.trusted_workflow_receipt_present,
                }
            )
        )


@dataclass(frozen=True)
class SciPlex3P1FinalizedCountScanReceipt:
    """Non-admissible proof of an exact full p1 count scan and close reauthentication."""

    initial_source_authentication_fingerprint: str
    source_sha256: str
    source_md5: str
    source_byte_count: int
    source_descriptor_identity_before: tuple[int, int, int, int, int]
    source_descriptor_identity_after: tuple[int, int, int, int, int]
    loader_implementation_sha256: str
    p1_loader_contract_sha256: str
    dataset_manifest_sha256: str
    query_sha256: str
    benchmark_sha256: str
    target_value_schema_sha256: str
    scoring_transform_sha256: str
    feature_panel_artifact_sha256: str
    ordered_feature_keys_sha256: str
    record_ids_sha256: str
    record_to_well_sha256: str
    emitted_source_row_indices_sha256: str
    ordered_record_source_well_condition_sha256: str
    count_stream_encoding: str
    panel_count_stream_sha256: str
    record_count: int
    well_count: int
    treated_well_count: int
    control_well_count: int
    batch_count: int
    panel_nonzero_count: int
    zero_panel_record_count: int
    panel_umi_total: int
    full_source_umi_total: int
    python_version: str
    python_implementation: str
    numpy_version: str
    h5py_version: str
    hdf5_version: str
    artifact_schema: str = "sciplex3-k562-p1-finalized-count-scan-receipt"
    artifact_schema_version: str = "1.0.0"
    loader_interface_id: str = "cellstate.sciplex3-training-data-loader.v1"
    partition_id: str = "p1-train"
    access_purpose: str = "train_parameters"
    accessed_partition_roles: tuple[str, ...] = ("p1-train",)
    accessed_count_datasets: tuple[str, ...] = _ACCESSED_COUNT_DATASETS
    exact_record_coverage: bool = True
    count_scan_complete: bool = True
    source_descriptor_reverified: bool = True
    close_reverification_completed: bool = True
    finalized: bool = True
    heldout_memberships_parsed: bool = False
    heldout_outcome_values_parsed: bool = False
    trusted_workflow_receipt_present: bool = False
    lifecycle_evidence_issued: bool = False
    scientifically_admissible: bool = False

    def __post_init__(self) -> None:
        _require_exact_primitive_fields(
            self,
            string_fields=(
                "initial_source_authentication_fingerprint",
                "source_sha256",
                "source_md5",
                "loader_implementation_sha256",
                "p1_loader_contract_sha256",
                "dataset_manifest_sha256",
                "query_sha256",
                "benchmark_sha256",
                "target_value_schema_sha256",
                "scoring_transform_sha256",
                "feature_panel_artifact_sha256",
                "ordered_feature_keys_sha256",
                "record_ids_sha256",
                "record_to_well_sha256",
                "emitted_source_row_indices_sha256",
                "ordered_record_source_well_condition_sha256",
                "count_stream_encoding",
                "panel_count_stream_sha256",
                "python_version",
                "python_implementation",
                "numpy_version",
                "h5py_version",
                "hdf5_version",
                "artifact_schema",
                "artifact_schema_version",
                "loader_interface_id",
                "partition_id",
                "access_purpose",
            ),
            integer_fields=(
                "source_byte_count",
                "record_count",
                "well_count",
                "treated_well_count",
                "control_well_count",
                "batch_count",
                "panel_nonzero_count",
                "zero_panel_record_count",
                "panel_umi_total",
                "full_source_umi_total",
            ),
            boolean_fields=(
                "exact_record_coverage",
                "count_scan_complete",
                "source_descriptor_reverified",
                "close_reverification_completed",
                "finalized",
                "heldout_memberships_parsed",
                "heldout_outcome_values_parsed",
                "trusted_workflow_receipt_present",
                "lifecycle_evidence_issued",
                "scientifically_admissible",
            ),
        )
        _require_exact_tuple(
            self.source_descriptor_identity_before,
            name="source_descriptor_identity_before",
            item_type=int,
            length=5,
        )
        _require_exact_tuple(
            self.source_descriptor_identity_after,
            name="source_descriptor_identity_after",
            item_type=int,
            length=5,
        )
        _require_exact_tuple(
            self.accessed_partition_roles,
            name="accessed_partition_roles",
            item_type=str,
        )
        _require_exact_tuple(
            self.accessed_count_datasets,
            name="accessed_count_datasets",
            item_type=str,
        )

    @property
    def fingerprint(self) -> str:
        """Return the canonical content identity of this deterministic receipt."""

        return _sha256(_canonical_json_bytes(asdict(self)))


@runtime_checkable
class SciPlex3TrainingDataLoader(Protocol):
    """Narrow interface future trusted receipts may bind independently of implementation."""

    @property
    def feature_panel(self) -> SciPlex3FeaturePanel: ...

    @property
    def access_purpose(self) -> PopulationComponentAccessPurpose: ...

    @property
    def source_scan_receipt(self) -> SciPlex3P1SourceScanReceipt: ...

    def describe_partition(self) -> SciPlex3PartitionDescriptor: ...

    def iter_parameter_training_batches(
        self,
        *,
        batch_size: int = 512,
        partition_id: str = "p1-train",
    ) -> Iterator[SciPlex3SparseCountBatch]: ...

    def finalize_parameter_training_count_scan(
        self,
    ) -> SciPlex3P1FinalizedCountScanReceipt: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class _SourceStat:
    device: int
    inode: int
    size: int
    mtime_ns: int
    ctime_ns: int

    @classmethod
    def from_stat(cls, value: os.stat_result) -> Self:
        return cls(
            device=value.st_dev,
            inode=value.st_ino,
            size=value.st_size,
            mtime_ns=value.st_mtime_ns,
            ctime_ns=value.st_ctime_ns,
        )

    @property
    def identity(self) -> tuple[int, int, int, int, int]:
        return (self.device, self.inode, self.size, self.mtime_ns, self.ctime_ns)


@dataclass(frozen=True)
class _PartitionMembership:
    descriptor: SciPlex3PartitionDescriptor
    record_ids: tuple[str, ...]
    well_ids: tuple[str, ...]
    plate_ids: tuple[str, ...]
    record_to_well: tuple[tuple[str, str], ...]
    well_to_condition: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class _ObservationIndex:
    row_by_record: Mapping[str, int]
    well_by_record: Mapping[str, str]
    condition_by_record: Mapping[str, str]
    source_row_indices_sha256: str
    ordered_record_source_well_condition_sha256: str


class _CountScanState:
    """Mutable internal accumulator; no proof is issued until close reauthentication succeeds."""

    def __init__(self) -> None:
        self.started = False
        self.iterator_exhausted = False
        self.invalid = False
        self.next_record_position = 0
        self.batch_count = 0
        self.panel_nonzero_count = 0
        self.zero_panel_record_count = 0
        self.panel_umi_total = 0
        self.full_source_umi_total = 0
        self.source_rows: list[int] = []
        self.ordered_binding_hasher = hashlib.sha256(b"[")
        self.panel_count_stream_hasher = hashlib.sha256(b"[")

    @staticmethod
    def _append_item(hasher: Any, position: int, value: object) -> None:
        if position:
            hasher.update(b",")
        hasher.update(_canonical_json_bytes(value))

    def append(
        self,
        *,
        record_id: str,
        source_row: int,
        well_id: str,
        condition_id: str,
        panel_pairs: list[list[int]],
        panel_total: int,
        full_source_total: int,
    ) -> None:
        if panel_total < 0 or full_source_total <= 0:
            raise ContractViolationError("count-scan totals are outside the raw-count domain")
        position = self.next_record_position
        self._append_item(
            self.ordered_binding_hasher,
            position,
            [record_id, source_row, well_id, condition_id],
        )
        self._append_item(
            self.panel_count_stream_hasher,
            position,
            [
                record_id,
                source_row,
                well_id,
                condition_id,
                panel_pairs,
                panel_total,
            ],
        )
        self.source_rows.append(source_row)
        self.panel_nonzero_count += len(panel_pairs)
        self.zero_panel_record_count += panel_total == 0
        self.panel_umi_total += panel_total
        self.full_source_umi_total += full_source_total
        self.next_record_position += 1

    @staticmethod
    def _finish(hasher: Any) -> str:
        completed = hasher.copy()
        completed.update(b"]")
        return cast(str, completed.hexdigest())

    @property
    def ordered_binding_sha256(self) -> str:
        return self._finish(self.ordered_binding_hasher)

    @property
    def panel_count_stream_sha256(self) -> str:
        return self._finish(self.panel_count_stream_hasher)


class SciPlex3K562H5ADLoader:
    """Context-managed exact-source reader for lifecycle-scoped sci-Plex3 partitions."""

    _source_path: Path
    _source_stream: BinaryIO | None
    _h5ad: Any | None
    _initial_stat: _SourceStat | None
    _closed: bool
    _matrix: Any | None
    _matrix_indptr_node: Any | None
    _ncounts_node: Any | None
    _panel: SciPlex3FeaturePanel
    _panel_position_by_source: dict[int, int]
    _access_purpose: PopulationComponentAccessPurpose
    _membership: _PartitionMembership
    _observations: _ObservationIndex
    _source_scan_receipt: SciPlex3P1SourceScanReceipt | None
    _finalized_count_scan_receipt: SciPlex3P1FinalizedCountScanReceipt | None
    _count_scan: _CountScanState
    _loader_implementation_sha256: str
    _h5py_version: str
    _hdf5_version: str

    def __init__(
        self,
        source_path: Path,
        repository_root: Path,
        *,
        access_purpose: PopulationComponentAccessPurpose,
        partition_id: str | None = None,
    ) -> None:
        opened = type(self).open_for_purpose(
            source_path,
            repository_root,
            access_purpose=access_purpose,
            partition_id=partition_id,
        )
        self.__dict__.update(opened.__dict__)

    @classmethod
    def open_for_purpose(
        cls,
        source_path: Path,
        repository_root: Path,
        *,
        access_purpose: PopulationComponentAccessPurpose,
        partition_id: str | None = None,
    ) -> Self:
        """Authenticate ``source_path`` into one permanently purpose-bound session.

        ``repository_root`` must contain the exact checked-in preparation directory.  Neither
        source expectations nor artifact identities are caller-configurable.  The requested
        purpose selects an exact partition through the pinned local registry; it never relabels a
        caller-selected partition.  p4 is unavailable until a future trusted locked-evaluation
        grant exists.
        """

        try:
            expected_partition_id, _, _ = _PURPOSE_BINDINGS[access_purpose]
        except (KeyError, TypeError) as error:
            raise ValueError("unknown population-component access purpose") from error
        if partition_id is not None and partition_id != expected_partition_id:
            raise ContractViolationError(
                f"{access_purpose.value} is bound only to {expected_partition_id}; "
                f"refusing to open {partition_id}"
            )
        if access_purpose is not PopulationComponentAccessPurpose.TRAIN_PARAMETERS:
            future_grant = {
                PopulationComponentAccessPurpose.FIT_CALIBRATION: (
                    "TrustedCalibrationAccessGrant bound to an exact TRAINED_CANDIDATE"
                ),
                PopulationComponentAccessPurpose.MODEL_SELECTION: (
                    "TrustedModelSelectionAccessGrant bound to an exact CALIBRATED_CANDIDATE"
                ),
                PopulationComponentAccessPurpose.UNTOUCHED_EVALUATION: (
                    "LockedEvaluationAccessGrant bound to an exact MODEL_SELECTED_FROZEN candidate"
                ),
            }[access_purpose]
            raise ContractViolationError(
                f"{expected_partition_id} is hard sealed: opening it requires a future trusted "
                f"{future_grant}"
            )

        loader = cls.__new__(cls)
        loader._source_path = Path(source_path)
        loader._source_stream = None
        loader._h5ad = None
        loader._initial_stat = None
        loader._closed = True
        loader._matrix = None
        loader._matrix_indptr_node = None
        loader._ncounts_node = None
        loader._panel_position_by_source = {}
        loader._access_purpose = access_purpose
        loader._observations = _ObservationIndex({}, {}, {}, "", "")
        loader._source_scan_receipt = None
        loader._finalized_count_scan_receipt = None
        loader._count_scan = _CountScanState()
        loader._loader_implementation_sha256 = _read_loader_implementation_sha256()
        if loader._loader_implementation_sha256 != _LOADER_IMPLEMENTATION_IMPORT_SHA256:
            raise ContractViolationError("loader implementation changed after module import")
        loader._h5py_version = ""
        loader._hdf5_version = ""

        artifacts = loader._authenticate_p1_artifacts(Path(repository_root))
        stream: BinaryIO | None = None
        h5ad: Any | None = None
        try:
            stream = loader._source_path.open("rb")
            loader._source_stream = stream
            loader._initial_stat = loader._authenticate_open_source(stream)
            h5py = loader._import_h5py()
            loader._h5py_version, loader._hdf5_version = loader._hdf5_runtime_versions(h5py)
            stream.seek(0)
            h5ad = h5py.File(stream, "r")
            loader._h5ad = h5ad
            loader._closed = False
            loader._validate_h5ad(artifacts)
            return loader
        except BaseException:
            if h5ad is not None:
                with suppress(BaseException):
                    h5ad.close()
            if stream is not None:
                with suppress(BaseException):
                    stream.close()
            loader._h5ad = None
            loader._source_stream = None
            loader._closed = True
            raise

    @classmethod
    def open(
        cls,
        source_path: Path,
        repository_root: Path,
        *,
        access_purpose: PopulationComponentAccessPurpose,
        partition_id: str | None = None,
    ) -> Self:
        """Alias for :meth:`open_for_purpose`; an unscoped loader cannot be opened."""

        return cls.open_for_purpose(
            source_path,
            repository_root,
            access_purpose=access_purpose,
            partition_id=partition_id,
        )

    @classmethod
    def training_partition_descriptor(
        cls,
        repository_root: Path,
    ) -> SciPlex3PartitionDescriptor:
        """Authenticate and describe p1 without opening H5AD or any held-out ledger."""

        loader = cls.__new__(cls)
        artifacts = loader._authenticate_p1_artifacts(Path(repository_root))
        contract = loader._json_object(artifacts["loader_contract"], "p1 loader contract")
        return loader._load_p1_membership(artifacts, contract).descriptor

    @staticmethod
    def _import_h5py() -> Any:
        try:
            return importlib.import_module("h5py")
        except ModuleNotFoundError as error:
            raise RuntimeError(
                "reading the sci-Plex3 H5AD requires optional h5py; install h5py>=3.11"
            ) from error

    @staticmethod
    def _hdf5_runtime_versions(h5py: Any) -> tuple[str, str]:
        h5py_version = getattr(h5py, "__version__", None)
        hdf5_version = getattr(getattr(h5py, "version", None), "hdf5_version", None)
        if not isinstance(h5py_version, str) or not h5py_version:
            raise ContractViolationError("h5py runtime does not expose an exact version")
        if not isinstance(hdf5_version, str) or not hdf5_version:
            raise ContractViolationError("h5py runtime does not expose its HDF5 version")
        return h5py_version, hdf5_version

    @property
    def feature_panel(self) -> SciPlex3FeaturePanel:
        self._require_open()
        return self._panel

    @property
    def access_purpose(self) -> PopulationComponentAccessPurpose:
        self._require_open()
        return self._access_purpose

    @property
    def source_scan_receipt(self) -> SciPlex3P1SourceScanReceipt:
        self._require_open()
        receipt = self._source_scan_receipt
        if receipt is None:
            raise ContractViolationError("p1 source scan receipt is unavailable")
        return receipt

    @property
    def finalized_count_scan_receipt(self) -> SciPlex3P1FinalizedCountScanReceipt:
        """Return the proof issued only after exact count coverage and successful close rehash."""

        if not self._closed:
            raise ContractViolationError(
                "finalized p1 count-scan receipt is unavailable before source close verification"
            )
        receipt = self._finalized_count_scan_receipt
        if receipt is None:
            raise ContractViolationError("no finalized p1 count scan was completed")
        return receipt

    def describe_partition(self) -> SciPlex3PartitionDescriptor:
        """Return the one pinned descriptor permanently bound to this session."""

        self._require_open()
        return self._membership.descriptor

    def iter_parameter_training_batches(
        self,
        *,
        batch_size: int = 512,
        partition_id: str = "p1-train",
    ) -> Iterator[SciPlex3SparseCountBatch]:
        """Read only ``p1-train`` for candidate parameter fitting.

        Held-out roles cannot be named through this method and a p2/p3 session cannot invoke it.
        """

        if self._access_purpose is not PopulationComponentAccessPurpose.TRAIN_PARAMETERS:
            raise ContractViolationError(
                "parameter-training reads require a session permanently bound to p1-train"
            )
        return self.iter_batches(batch_size=batch_size, partition_id=partition_id)

    def finalize_parameter_training_count_scan(
        self,
    ) -> SciPlex3P1FinalizedCountScanReceipt:
        """Close/reverify after one exact full p1 stream, then issue a non-admissible receipt.

        This is the sole count-scan finalization surface.  Partial, repeated, out-of-order, or
        failed iteration cannot reach source close, much less obtain a receipt.
        """

        self._require_open()
        self._assert_count_scan_complete()
        self.close()
        return self.finalized_count_scan_receipt

    def iter_batches(
        self,
        *,
        batch_size: int = 512,
        partition_id: str | None = None,
    ) -> Iterator[SciPlex3SparseCountBatch]:
        """Yield deterministic immutable batches for exactly one lifecycle purpose.

        Record order is the canonical order in the authenticated membership artifact, never H5AD
        row order.  The session purpose and partition were frozen before HDF5 was opened.  An
        optional partition identifier can only assert that already-derived identity.
        """

        self._require_open()
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if partition_id is not None and partition_id != self._membership.descriptor.partition_id:
            raise ContractViolationError(
                f"session is bound only to {self._membership.descriptor.partition_id}; "
                f"refusing count access for {partition_id}"
            )
        if self._count_scan.started:
            self._count_scan.invalid = True
            raise ContractViolationError(
                "a p1 count-scan session permits exactly one canonical batch iterator"
            )
        self._count_scan.started = True
        return self._iter_membership_batches(self._membership, batch_size=batch_size)

    def _iter_membership_batches(
        self,
        membership: _PartitionMembership,
        *,
        batch_size: int,
    ) -> Iterator[SciPlex3SparseCountBatch]:
        record_ids = membership.record_ids
        try:
            for batch_index, start in enumerate(range(0, len(record_ids), batch_size)):
                self._require_open()
                selected = record_ids[start : start + batch_size]
                yield self._read_batch(membership.descriptor, batch_index, selected)
        except BaseException:
            self._count_scan.invalid = True
            raise
        else:
            self._count_scan.iterator_exhausted = True

    def _read_batch(
        self,
        descriptor: SciPlex3PartitionDescriptor,
        batch_index: int,
        record_ids: tuple[str, ...],
    ) -> SciPlex3SparseCountBatch:
        matrix = self._matrix
        indptr_node = self._matrix_indptr_node
        ncounts_node = self._ncounts_node
        if matrix is None or indptr_node is None or ncounts_node is None:
            raise ContractViolationError("sci-Plex3 matrix handle is unavailable")
        data_node = matrix["data"]
        index_node = matrix["indices"]
        output_indptr = [0]
        output_indices: list[int] = []
        output_counts: list[int] = []
        source_rows: list[int] = []
        well_ids: list[str] = []
        condition_ids: list[str] = []
        panel_totals: list[int] = []
        full_source_totals: list[int] = []

        for record_id in record_ids:
            row = self._observations.row_by_record[record_id]
            source_rows.append(row)
            well_ids.append(self._observations.well_by_record[record_id])
            condition_ids.append(self._observations.condition_by_record[record_id])
            source_start = int(indptr_node[row])
            source_stop = int(indptr_node[row + 1])
            if source_start < 0 or source_stop < source_start or source_stop > SCIPLEX3_H5AD_NNZ:
                raise ContractViolationError("authorized source CSR row pointer is malformed")
            raw_counts = np.asarray(data_node[slice(source_start, source_stop)])
            raw_indices = np.asarray(index_node[slice(source_start, source_stop)])
            if raw_counts.dtype != np.dtype(SCIPLEX3_MATRIX_DTYPE):
                raise ContractViolationError(
                    "source count slice changed from exact int64 UMI dtype"
                )
            if raw_indices.dtype.kind not in {"i", "u"}:
                raise ContractViolationError("source CSR indices are not integers")
            counts = np.asarray(raw_counts, dtype=np.int64)
            columns = np.asarray(raw_indices, dtype=np.int64)
            if len(counts) != len(columns):
                raise ContractViolationError("source CSR data and index slices are misaligned")
            if np.any(counts < 0):
                raise ContractViolationError("source contains a negative raw UMI count")
            if np.any(columns < 0) or np.any(columns >= SCIPLEX3_H5AD_SHAPE[1]):
                raise ContractViolationError("source CSR column lies outside the feature axis")
            if len(columns) > 1 and np.any(np.diff(columns) <= 0):
                raise ContractViolationError("source CSR columns are not unique and ordered")
            full_total = int(counts.sum(dtype=np.int64))
            if full_total <= 0:
                raise ContractViolationError("K562 source row has a nonpositive raw-count total")
            try:
                expected_library_size = float(ncounts_node[row])
            except (TypeError, ValueError) as error:
                raise ContractViolationError(
                    "authorized obs ncounts value is not numeric"
                ) from error
            if (
                not np.isfinite(expected_library_size)
                or not expected_library_size.is_integer()
                or expected_library_size <= 0
            ):
                raise ContractViolationError("authorized obs ncounts must be a positive integer")
            if full_total != int(expected_library_size):
                raise ContractViolationError("source CSR row total differs from obs ncounts")
            full_source_totals.append(full_total)

            projected = [
                (self._panel_position_by_source[int(column)], int(count))
                for column, count in zip(columns, counts, strict=True)
                if int(column) in self._panel_position_by_source and int(count) != 0
            ]
            projected.sort()
            row_total = sum(count for _, count in projected)
            output_indices.extend(column for column, _ in projected)
            output_counts.extend(count for _, count in projected)
            output_indptr.append(len(output_counts))
            panel_totals.append(row_total)

        batch = SciPlex3SparseCountBatch(
            partition=descriptor,
            batch_index=batch_index,
            record_ids=record_ids,
            composite_well_ids=tuple(well_ids),
            condition_ids=tuple(condition_ids),
            source_row_indices=np.asarray(source_rows, dtype=np.int64),
            indptr=np.asarray(output_indptr, dtype=np.int64),
            feature_indices=np.asarray(output_indices, dtype=np.int64),
            counts=np.asarray(output_counts, dtype=np.int64),
            panel_totals=np.asarray(panel_totals, dtype=np.int64),
        )
        self._record_count_scan_batch(batch, tuple(full_source_totals))
        return batch

    def _record_count_scan_batch(
        self,
        batch: SciPlex3SparseCountBatch,
        full_source_totals: tuple[int, ...],
    ) -> None:
        state = self._count_scan
        try:
            if not state.started or state.invalid or state.iterator_exhausted:
                raise ContractViolationError("p1 count scan is not in an appendable state")
            if batch.batch_index != state.batch_count:
                raise ContractViolationError("p1 batch order differs from the canonical scan")
            if len(full_source_totals) != len(batch.record_ids):
                raise ContractViolationError("p1 full-source totals are misaligned")
            for row_index, record_id in enumerate(batch.record_ids):
                position = state.next_record_position
                if position >= len(self._membership.record_ids):
                    raise ContractViolationError("p1 count scan emitted too many records")
                if record_id != self._membership.record_ids[position]:
                    raise ContractViolationError("p1 count scan record order is not canonical")
                source_row = int(batch.source_row_indices[row_index])
                well_id = batch.composite_well_ids[row_index]
                condition_id = batch.condition_ids[row_index]
                if (
                    source_row != self._observations.row_by_record[record_id]
                    or well_id != self._observations.well_by_record[record_id]
                    or condition_id != self._observations.condition_by_record[record_id]
                ):
                    raise ContractViolationError(
                        "p1 emitted row binding differs from authentication"
                    )
                start = int(batch.indptr[row_index])
                stop = int(batch.indptr[row_index + 1])
                panel_pairs = [
                    [int(feature_index), int(count)]
                    for feature_index, count in zip(
                        batch.feature_indices[start:stop],
                        batch.counts[start:stop],
                        strict=True,
                    )
                ]
                state.append(
                    record_id=record_id,
                    source_row=source_row,
                    well_id=well_id,
                    condition_id=condition_id,
                    panel_pairs=panel_pairs,
                    panel_total=int(batch.panel_totals[row_index]),
                    full_source_total=full_source_totals[row_index],
                )
            state.batch_count += 1
        except BaseException:
            state.invalid = True
            raise

    def _assert_count_scan_complete(self) -> None:
        state = self._count_scan
        receipt = self._source_scan_receipt
        if (
            not state.started
            or not state.iterator_exhausted
            or state.invalid
            or state.batch_count <= 0
            or state.next_record_position != _EXPECTED_P1_RECORD_COUNT
            or len(state.source_rows) != _EXPECTED_P1_RECORD_COUNT
            or len(set(state.source_rows)) != _EXPECTED_P1_RECORD_COUNT
        ):
            raise ContractViolationError(
                "finalization requires one fully exhausted exact p1 count iterator"
            )
        if receipt is None:
            raise ContractViolationError("initial p1 source authentication receipt is absent")
        if state.ordered_binding_sha256 != receipt.ordered_record_source_well_condition_sha256:
            raise ContractViolationError(
                "emitted p1 row binding digest differs from authentication"
            )
        source_order_digest = _sha256(
            _canonical_json_bytes([str(row) for row in sorted(state.source_rows)])
        )
        if source_order_digest != receipt.source_row_indices_sha256:
            raise ContractViolationError(
                "emitted p1 source-row coverage differs from authentication"
            )
        if state.panel_nonzero_count <= 0 or state.panel_umi_total <= 0:
            raise ContractViolationError("completed p1 count scan has no positive panel counts")
        if state.full_source_umi_total < state.panel_umi_total:
            raise ContractViolationError("p1 panel UMI total exceeds the full-source total")

    def _build_finalized_count_scan_receipt(
        self,
        final_stat: _SourceStat,
    ) -> SciPlex3P1FinalizedCountScanReceipt:
        self._assert_count_scan_complete()
        initial_receipt = self._source_scan_receipt
        initial_stat = self._initial_stat
        if initial_receipt is None or initial_stat is None:
            raise ContractViolationError("p1 initial authentication state is unavailable")
        final_loader_sha256 = _read_loader_implementation_sha256()
        if final_loader_sha256 != self._loader_implementation_sha256:
            raise ContractViolationError("loader implementation changed during the count scan")

        control_well_count = sum(
            condition_id == "source-control@0nM"
            for _, condition_id in self._membership.well_to_condition
        )
        treated_well_count = len(self._membership.well_to_condition) - control_well_count
        if (
            treated_well_count != _EXPECTED_P1_TREATED_WELL_COUNT
            or control_well_count != _EXPECTED_P1_CONTROL_WELL_COUNT
        ):
            raise ContractViolationError("p1 treated/control well coverage drifted")
        state = self._count_scan
        emitted_source_rows_sha256 = _sha256(_canonical_json_bytes(state.source_rows))
        return SciPlex3P1FinalizedCountScanReceipt(
            initial_source_authentication_fingerprint=initial_receipt.fingerprint,
            source_sha256=SCIPLEX3_SOURCE_SHA256,
            source_md5=SCIPLEX3_SOURCE_MD5,
            source_byte_count=SCIPLEX3_SOURCE_BYTE_COUNT,
            source_descriptor_identity_before=initial_stat.identity,
            source_descriptor_identity_after=final_stat.identity,
            loader_implementation_sha256=final_loader_sha256,
            p1_loader_contract_sha256=SCIPLEX3_P1_LOADER_CONTRACT_SHA256,
            dataset_manifest_sha256=SCIPLEX3_K562_MANIFEST_SHA256,
            query_sha256=SCIPLEX3_K562_QUERY_SHA256,
            benchmark_sha256=SCIPLEX3_K562_BENCHMARK_SHA256,
            target_value_schema_sha256=SCIPLEX3_K562_TARGET_VALUE_SCHEMA_SHA256,
            scoring_transform_sha256=SCIPLEX3_SCORING_TRANSFORM_SHA256,
            feature_panel_artifact_sha256=SCIPLEX3_FEATURE_PANEL_ARTIFACT_SHA256,
            ordered_feature_keys_sha256=SCIPLEX3_ORDERED_FEATURE_KEYS_SHA256,
            record_ids_sha256=self._membership.descriptor.record_ids_sha256,
            record_to_well_sha256=self._membership.descriptor.record_to_well_sha256,
            emitted_source_row_indices_sha256=emitted_source_rows_sha256,
            ordered_record_source_well_condition_sha256=state.ordered_binding_sha256,
            count_stream_encoding=_COUNT_STREAM_ENCODING,
            panel_count_stream_sha256=state.panel_count_stream_sha256,
            record_count=state.next_record_position,
            well_count=len(self._membership.well_ids),
            treated_well_count=treated_well_count,
            control_well_count=control_well_count,
            batch_count=state.batch_count,
            panel_nonzero_count=state.panel_nonzero_count,
            zero_panel_record_count=state.zero_panel_record_count,
            panel_umi_total=state.panel_umi_total,
            full_source_umi_total=state.full_source_umi_total,
            python_version=sys.version,
            python_implementation=platform.python_implementation(),
            numpy_version=np.__version__,
            h5py_version=self._h5py_version,
            hdf5_version=self._hdf5_version,
        )

    def close(self) -> None:
        """Close HDF5, then reauthenticate the same descriptor and its filesystem identity."""

        if self._closed:
            return
        h5ad = self._h5ad
        stream = self._source_stream
        initial_stat = self._initial_stat
        self._closed = True
        self._h5ad = None
        self._matrix = None
        self._matrix_indptr_node = None
        self._ncounts_node = None
        close_error: BaseException | None = None
        try:
            if h5ad is None:
                raise ContractViolationError("open loader lost its HDF5 handle")
            h5ad.close()
        except BaseException as error:
            close_error = error

        verification_error: BaseException | None = None
        final_stat: _SourceStat | None = None
        try:
            if stream is None or initial_stat is None or stream.closed:
                raise ContractViolationError("source descriptor closed before final verification")
            final_stat = self._reauthenticate_close_source(stream, initial_stat)
        except BaseException as error:
            verification_error = error
        finally:
            if stream is not None and not stream.closed:
                stream.close()
            self._source_stream = None

        if verification_error is not None:
            raise ContractViolationError("sci-Plex3 source drifted during H5AD access") from (
                verification_error
            )
        if close_error is not None:
            raise ContractViolationError("failed to close the sci-Plex3 HDF5 handle cleanly") from (
                close_error
            )
        state = self._count_scan
        if (
            state.started
            and state.iterator_exhausted
            and not state.invalid
            and state.next_record_position == _EXPECTED_P1_RECORD_COUNT
        ):
            if final_stat is None:
                raise ContractViolationError("final source descriptor identity is unavailable")
            self._finalized_count_scan_receipt = self._build_finalized_count_scan_receipt(
                final_stat
            )

    def __enter__(self) -> Self:
        self._require_open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc_value, traceback
        self.close()

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("sci-Plex3 loader is closed")

    def _authenticate_open_source(self, stream: BinaryIO) -> _SourceStat:
        try:
            descriptor_before = _SourceStat.from_stat(os.fstat(stream.fileno()))
            path_before = _SourceStat.from_stat(self._source_path.stat())
        except OSError as error:
            raise ContractViolationError("cannot stat the sci-Plex3 source descriptor") from error
        if descriptor_before != path_before:
            raise ContractViolationError(
                "source path and opened descriptor do not identify one file"
            )
        if descriptor_before.size != SCIPLEX3_SOURCE_BYTE_COUNT:
            raise ContractViolationError(
                f"sci-Plex3 source byte count mismatch: expected {SCIPLEX3_SOURCE_BYTE_COUNT}, "
                f"observed {descriptor_before.size}"
            )
        sha256, md5, byte_count = self._digest_stream(stream)
        try:
            descriptor_after = _SourceStat.from_stat(os.fstat(stream.fileno()))
            path_after = _SourceStat.from_stat(self._source_path.stat())
        except OSError as error:
            raise ContractViolationError("cannot restat the sci-Plex3 source descriptor") from error
        if descriptor_before != descriptor_after or descriptor_before != path_after:
            raise ContractViolationError("sci-Plex3 source changed during initial authentication")
        if byte_count != SCIPLEX3_SOURCE_BYTE_COUNT:
            raise ContractViolationError("sci-Plex3 source length changed while hashing")
        if md5 != SCIPLEX3_SOURCE_MD5 or sha256 != SCIPLEX3_SOURCE_SHA256:
            raise ContractViolationError(
                "sci-Plex3 source digest differs from the corrected release"
            )
        return descriptor_before

    def _reauthenticate_close_source(
        self,
        stream: BinaryIO,
        initial_stat: _SourceStat,
    ) -> _SourceStat:
        try:
            before = _SourceStat.from_stat(os.fstat(stream.fileno()))
        except OSError as error:
            raise ContractViolationError("cannot stat source before final digest") from error
        if before != initial_stat:
            raise ContractViolationError("source stat changed while the H5AD was open")
        sha256, md5, byte_count = self._digest_stream(stream)
        try:
            after = _SourceStat.from_stat(os.fstat(stream.fileno()))
            path_after = _SourceStat.from_stat(self._source_path.stat())
        except OSError as error:
            raise ContractViolationError("cannot stat source after final digest") from error
        if before != after or initial_stat != after or initial_stat != path_after:
            raise ContractViolationError("source identity changed during final authentication")
        if (
            byte_count != SCIPLEX3_SOURCE_BYTE_COUNT
            or md5 != SCIPLEX3_SOURCE_MD5
            or sha256 != SCIPLEX3_SOURCE_SHA256
        ):
            raise ContractViolationError("source content changed while the H5AD was open")
        return after

    @staticmethod
    def _digest_stream(
        stream: BinaryIO,
        *,
        chunk_size: int = 8 * 1024 * 1024,
    ) -> tuple[str, str, int]:
        if not stream.seekable() or not stream.readable():
            raise ContractViolationError("sci-Plex3 source must be a seekable readable binary file")
        sha256 = hashlib.sha256()
        md5 = hashlib.md5(usedforsecurity=False)
        byte_count = 0
        stream.seek(0)
        while block := stream.read(chunk_size):
            if not isinstance(block, bytes):
                raise ContractViolationError("sci-Plex3 source did not produce binary bytes")
            sha256.update(block)
            md5.update(block)
            byte_count += len(block)
        stream.seek(0)
        return sha256.hexdigest(), md5.hexdigest(), byte_count

    def _authenticate_p1_artifacts(self, repository_root: Path) -> dict[str, bytes]:
        contract_path = repository_root / _P1_LOADER_CONTRACT_PATH
        try:
            contract_payload = contract_path.read_bytes()
        except OSError as error:
            raise ContractViolationError("missing sci-Plex3 p1 loader contract") from error
        if _sha256(contract_payload) != SCIPLEX3_P1_LOADER_CONTRACT_SHA256:
            raise ContractViolationError("sci-Plex3 p1 loader contract drifted")
        contract = self._json_object(contract_payload, "p1 loader contract")
        if contract_payload != _canonical_json_bytes(contract):
            raise ContractViolationError("sci-Plex3 p1 loader contract is not canonical JSON")
        self._validate_p1_loader_contract(contract)

        raw_artifacts = self._mapping(contract.get("artifacts"), "p1 contract artifacts")
        expected_keys = {
            "feature_panel",
            "plate_ids",
            "record_ids",
            "record_to_well",
            "source_verification",
            "well_ids",
            "well_to_condition",
        }
        if set(raw_artifacts) != expected_keys:
            raise ContractViolationError("p1 loader contract artifact closure is not exact")
        retained = {"loader_contract": contract_payload}
        for key in sorted(expected_keys):
            reference = self._mapping(raw_artifacts[key], f"p1 artifact reference {key}")
            relative = reference.get("relative_path")
            byte_count = reference.get("byte_count")
            digest = reference.get("sha256")
            if (
                not isinstance(relative, str)
                or not relative
                or isinstance(byte_count, bool)
                or not isinstance(byte_count, int)
                or byte_count < 0
                or not isinstance(digest, str)
                or len(digest) != 64
            ):
                raise ContractViolationError(f"malformed p1 artifact reference: {key}")
            relative_path = Path(relative)
            if relative_path.is_absolute() or ".." in relative_path.parts:
                raise ContractViolationError("p1 artifact path escapes its trusted root")
            if relative_path.parts[0] == "benchmarks":
                path = repository_root / relative_path
            else:
                path = repository_root / _PREPARATION_DIRECTORY / relative_path
            try:
                payload = path.read_bytes()
            except OSError as error:
                raise ContractViolationError(f"missing p1 artifact: {key}") from error
            if len(payload) != byte_count or _sha256(payload) != digest:
                raise ContractViolationError(f"content-addressed p1 artifact drift: {key}")
            retained[key] = payload
        return retained

    def _validate_h5ad(self, artifacts: Mapping[str, bytes]) -> None:
        contract = self._json_object(artifacts["loader_contract"], "p1 loader contract")
        source_spec = self._json_object(artifacts["source_verification"], _SOURCE_VERIFICATION)
        feature_panel = self._json_object(artifacts["feature_panel"], _FEATURE_PANEL)
        self._validate_p1_summaries(source_spec, feature_panel)

        h5ad = self._h5ad
        if h5ad is None:
            raise ContractViolationError("H5AD handle is absent")
        if not {"X", "obs", "var"} <= set(h5ad.keys()):
            raise ContractViolationError("source H5AD is missing X, obs, or var")
        matrix = h5ad["X"]
        try:
            shape = tuple(int(value) for value in matrix.attrs["shape"])
        except (KeyError, TypeError, ValueError) as error:
            raise ContractViolationError(
                "source CSR matrix lacks an exact shape attribute"
            ) from error
        if shape != SCIPLEX3_H5AD_SHAPE:
            raise ContractViolationError(f"source matrix shape mismatch: {shape}")
        encoding_type = matrix.attrs.get("encoding-type")
        if _decode_scalar(encoding_type) != "csr_matrix":
            raise ContractViolationError("source X is not an AnnData CSR matrix")
        if tuple(matrix.keys()) != ("data", "indices", "indptr"):
            raise ContractViolationError(
                "source CSR group must contain exactly data/indices/indptr"
            )
        data_node = matrix["data"]
        index_node = matrix["indices"]
        indptr_node = matrix["indptr"]
        if np.dtype(data_node.dtype) != np.dtype(SCIPLEX3_MATRIX_DTYPE):
            raise ContractViolationError("corrected source X.data must have exact int64 dtype")
        if np.dtype(index_node.dtype).kind not in {"i", "u"} or np.dtype(
            indptr_node.dtype
        ).kind not in {"i", "u"}:
            raise ContractViolationError("source CSR indices and indptr must be integer arrays")
        if len(data_node) != SCIPLEX3_H5AD_NNZ or len(index_node) != SCIPLEX3_H5AD_NNZ:
            raise ContractViolationError("source CSR nonzero count differs from corrected release")
        if len(indptr_node) != SCIPLEX3_H5AD_SHAPE[0] + 1:
            raise ContractViolationError("source CSR indptr has the wrong structural length")
        self._matrix = matrix
        self._matrix_indptr_node = indptr_node

        obs = h5ad["obs"]
        if not set(obs.keys()) >= _REQUIRED_OBSERVATION_FIELDS:
            raise ContractViolationError("source obs lacks a required corrected-release field")
        ncounts_node = obs["ncounts"]
        if len(ncounts_node) != SCIPLEX3_H5AD_SHAPE[0] or np.dtype(ncounts_node.dtype).kind not in {
            "f",
            "i",
            "u",
        }:
            raise ContractViolationError("source obs.ncounts has the wrong structural schema")
        self._ncounts_node = ncounts_node
        var = h5ad["var"]
        if not set(var.keys()) >= _REQUIRED_FEATURE_FIELDS:
            raise ContractViolationError("source var lacks corrected feature metadata")
        self._panel = self._validate_feature_axis(var, source_spec, feature_panel)
        self._panel_position_by_source = {
            source_index: panel_index
            for panel_index, source_index in enumerate(self._panel.source_feature_indices)
        }

        membership = self._load_p1_membership(artifacts, contract)
        observations = self._validate_p1_observations(obs, membership, feature_panel)
        self._membership = membership
        self._observations = observations
        self._source_scan_receipt = SciPlex3P1SourceScanReceipt(
            source_sha256=SCIPLEX3_SOURCE_SHA256,
            source_md5=SCIPLEX3_SOURCE_MD5,
            source_byte_count=SCIPLEX3_SOURCE_BYTE_COUNT,
            p1_loader_contract_sha256=SCIPLEX3_P1_LOADER_CONTRACT_SHA256,
            feature_panel_artifact_sha256=SCIPLEX3_FEATURE_PANEL_ARTIFACT_SHA256,
            ordered_feature_keys_sha256=SCIPLEX3_ORDERED_FEATURE_KEYS_SHA256,
            record_ids_sha256=membership.descriptor.record_ids_sha256,
            record_to_well_sha256=membership.descriptor.record_to_well_sha256,
            source_row_indices_sha256=observations.source_row_indices_sha256,
            ordered_record_source_well_condition_sha256=(
                observations.ordered_record_source_well_condition_sha256
            ),
            dataset_manifest_sha256=SCIPLEX3_K562_MANIFEST_SHA256,
            query_sha256=SCIPLEX3_K562_QUERY_SHA256,
            benchmark_sha256=SCIPLEX3_K562_BENCHMARK_SHA256,
            target_value_schema_sha256=SCIPLEX3_K562_TARGET_VALUE_SCHEMA_SHA256,
            scoring_transform_sha256=SCIPLEX3_SCORING_TRANSFORM_SHA256,
            matrix_shape=SCIPLEX3_H5AD_SHAPE,
            matrix_nonzero_count=SCIPLEX3_H5AD_NNZ,
            matrix_value_dtype=SCIPLEX3_MATRIX_DTYPE,
        )

    def _validate_p1_loader_contract(self, contract: Mapping[str, object]) -> None:
        if (
            contract.get("artifact_schema") != "sciplex3-k562-p1-loader-contract"
            or contract.get("artifact_schema_version") != "1.0.0"
            or contract.get("heldout_memberships_referenced") is not False
            or contract.get("loader_outputs_can_mint_lifecycle_evidence") is not False
            or contract.get("scientifically_admissible_without_trusted_workflow_receipt")
            is not False
        ):
            raise ContractViolationError("p1 loader contract scope or safety boundary drifted")
        source = self._mapping(contract.get("source"), "p1 contract source")
        if source != {
            "filename": SCIPLEX3_SOURCE_FILENAME,
            "byte_count": SCIPLEX3_SOURCE_BYTE_COUNT,
            "md5": SCIPLEX3_SOURCE_MD5,
            "sha256": SCIPLEX3_SOURCE_SHA256,
        }:
            raise ContractViolationError("p1 loader contract source identity drifted")
        structure = self._mapping(contract.get("h5ad_structure"), "p1 H5AD structure")
        expected_structure = {
            "matrix_encoding": "csr_matrix",
            "matrix_shape": list(SCIPLEX3_H5AD_SHAPE),
            "matrix_nonzero_count": SCIPLEX3_H5AD_NNZ,
            "matrix_value_dtype": SCIPLEX3_MATRIX_DTYPE,
            "required_observation_fields": sorted(_REQUIRED_OBSERVATION_FIELDS),
            "source_feature_axis_sha256": SCIPLEX3_SOURCE_FEATURE_AXIS_SHA256,
        }
        if structure != expected_structure:
            raise ContractViolationError("p1 loader contract H5AD structure drifted")
        feature = self._mapping(contract.get("feature_panel"), "p1 contract feature panel")
        if feature != {
            "feature_count": SCIPLEX3_FEATURE_COUNT,
            "ordered_feature_keys_sha256": SCIPLEX3_ORDERED_FEATURE_KEYS_SHA256,
            "source_feature_axis_sha256": SCIPLEX3_SOURCE_FEATURE_AXIS_SHA256,
        }:
            raise ContractViolationError("p1 loader contract feature identity drifted")
        partition = self._mapping(contract.get("partition"), "p1 contract partition")
        if partition != {
            "access_purpose": "train_parameters",
            "artifact_role": "train",
            "partition_id": "p1-train",
            "record_count": _EXPECTED_P1_RECORD_COUNT,
            "selector": {
                "cell_line": "K562",
                "plates": list(_P1_PLATES),
                "replicate": "rep1",
            },
            "well_count": _EXPECTED_P1_WELL_COUNT,
        }:
            raise ContractViolationError("p1 loader contract partition identity drifted")
        bindings = self._mapping(contract.get("bindings"), "p1 contract bindings")
        if bindings != {
            "benchmark_sha256": SCIPLEX3_K562_BENCHMARK_SHA256,
            "dataset_manifest_sha256": SCIPLEX3_K562_MANIFEST_SHA256,
            "query_sha256": SCIPLEX3_K562_QUERY_SHA256,
            "scoring_transform_sha256": SCIPLEX3_SCORING_TRANSFORM_SHA256,
            "target_value_schema_sha256": SCIPLEX3_K562_TARGET_VALUE_SCHEMA_SHA256,
        }:
            raise ContractViolationError("p1 loader contract semantic bindings drifted")

    def _validate_p1_summaries(
        self,
        source_spec: Mapping[str, object],
        feature_panel: Mapping[str, object],
    ) -> None:
        if source_spec.get("artifact_schema") != "sciplex3-k562-source-verification":
            raise ContractViolationError("source verification has the wrong schema")
        source = self._mapping(source_spec.get("source"), "source verification source")
        expected_source = {
            "filename": SCIPLEX3_SOURCE_FILENAME,
            "byte_count": SCIPLEX3_SOURCE_BYTE_COUNT,
            "md5": SCIPLEX3_SOURCE_MD5,
            "sha256": SCIPLEX3_SOURCE_SHA256,
        }
        for key, expected in expected_source.items():
            if source.get(key) != expected:
                raise ContractViolationError(f"source verification {key} differs from frozen value")
        structure = self._mapping(
            source_spec.get("h5ad_structure"), "source verification H5AD structure"
        )
        if (
            structure.get("matrix_encoding") != "csr_matrix"
            or structure.get("matrix_shape") != list(SCIPLEX3_H5AD_SHAPE)
            or structure.get("matrix_nonzero_count") != SCIPLEX3_H5AD_NNZ
            or structure.get("matrix_value_dtype") != SCIPLEX3_MATRIX_DTYPE
            or structure.get("required_observation_fields") != sorted(_REQUIRED_OBSERVATION_FIELDS)
            or structure.get("source_feature_axis_sha256") != SCIPLEX3_SOURCE_FEATURE_AXIS_SHA256
        ):
            raise ContractViolationError("source verification H5AD contract drifted")
        if (
            feature_panel.get("artifact_schema") != "sciplex3-k562-train-feature-panel"
            or feature_panel.get("source_sha256") != SCIPLEX3_SOURCE_SHA256
            or feature_panel.get("feature_count") != SCIPLEX3_FEATURE_COUNT
            or feature_panel.get("ordered_feature_keys_sha256")
            != SCIPLEX3_ORDERED_FEATURE_KEYS_SHA256
        ):
            raise ContractViolationError("frozen feature-panel contract drifted")
        selection = self._mapping(feature_panel.get("feature_selection"), "feature selection")
        if (
            selection.get("selection_partition_role") != "train"
            or selection.get("count_accessed_partition_roles") != ["train"]
            or selection.get("heldout_count_rows_accessed") != 0
        ):
            raise ContractViolationError("feature panel is not proven TRAIN-only")

    def _validate_feature_axis(
        self,
        var: Any,
        source_spec: Mapping[str, object],
        feature_panel: Mapping[str, object],
    ) -> SciPlex3FeaturePanel:
        rows = np.arange(SCIPLEX3_H5AD_SHAPE[1], dtype=np.int64)
        raw_ensembl = self._decode_column_rows(var, "ensembl_id", rows, allow_missing=True)
        raw_symbols = self._decode_column_rows(var, "gene_symbol", rows, allow_missing=False)
        if len(raw_ensembl) != SCIPLEX3_H5AD_SHAPE[1] or len(raw_symbols) != SCIPLEX3_H5AD_SHAPE[1]:
            raise ContractViolationError("source feature metadata do not cover the matrix axis")
        axis_entries = [
            [index, ensembl, symbol]
            for index, (ensembl, symbol) in enumerate(zip(raw_ensembl, raw_symbols, strict=True))
        ]
        axis_digest = _sha256(_canonical_json_bytes(axis_entries))
        structure = self._mapping(
            source_spec.get("h5ad_structure"), "source verification H5AD structure"
        )
        if axis_digest != SCIPLEX3_SOURCE_FEATURE_AXIS_SHA256 or axis_digest != structure.get(
            "source_feature_axis_sha256"
        ):
            raise ContractViolationError("source feature axis differs from corrected release")

        raw_features = feature_panel.get("features")
        if not isinstance(raw_features, list) or len(raw_features) != SCIPLEX3_FEATURE_COUNT:
            raise ContractViolationError("feature panel does not contain exactly 2,000 definitions")
        indices: list[int] = []
        keys: list[str] = []
        for expected_rank, raw_feature in enumerate(raw_features, start=1):
            feature = self._mapping(raw_feature, "feature panel row")
            source_index = feature.get("source_feature_index")
            ensembl = feature.get("ensembl_id")
            symbol = feature.get("gene_symbol")
            if (
                feature.get("rank") != expected_rank
                or isinstance(source_index, bool)
                or not isinstance(source_index, int)
                or not 0 <= source_index < SCIPLEX3_H5AD_SHAPE[1]
                or not isinstance(ensembl, str)
                or not isinstance(symbol, str)
            ):
                raise ContractViolationError("malformed ordered feature-panel row")
            if raw_ensembl[source_index] != ensembl or raw_symbols[source_index] != symbol:
                raise ContractViolationError(
                    "feature panel differs from the authenticated var axis"
                )
            indices.append(source_index)
            keys.append(f"{ensembl}|{symbol}")
        keys_payload = _canonical_json_bytes(keys)
        digest = _sha256(keys_payload)
        if (
            digest != SCIPLEX3_ORDERED_FEATURE_KEYS_SHA256
            or digest != feature_panel.get("ordered_feature_keys_sha256")
            or len(keys_payload) != feature_panel.get("ordered_feature_keys_encoded_byte_count")
        ):
            raise ContractViolationError("ordered feature-panel identity mismatch")
        return SciPlex3FeaturePanel(
            source_feature_indices=tuple(indices),
            ordered_feature_keys=tuple(keys),
            ordered_feature_keys_sha256=digest,
            source_feature_axis_sha256=axis_digest,
        )

    def _load_p1_membership(
        self,
        artifacts: Mapping[str, bytes],
        contract: Mapping[str, object],
    ) -> _PartitionMembership:
        record_ids = self._string_array(artifacts["record_ids"], "p1 record IDs")
        well_ids = self._string_array(artifacts["well_ids"], "p1 well IDs")
        plate_ids = self._string_array(artifacts["plate_ids"], "p1 plate IDs")
        record_to_well = self._string_pairs(artifacts["record_to_well"], "p1 record-to-well")
        well_to_condition = self._string_pairs(
            artifacts["well_to_condition"], "p1 well-to-condition"
        )
        if len(record_ids) != _EXPECTED_P1_RECORD_COUNT or len(well_ids) != _EXPECTED_P1_WELL_COUNT:
            raise ContractViolationError("p1 materialized membership count drift")
        if plate_ids != _P1_PLATES:
            raise ContractViolationError("p1 materialized plate membership drift")
        if tuple(left for left, _ in record_to_well) != record_ids:
            raise ContractViolationError("p1 record-to-well domain mismatch")
        if tuple(left for left, _ in well_to_condition) != well_ids:
            raise ContractViolationError("p1 well-to-condition domain mismatch")
        references = self._mapping(contract.get("artifacts"), "p1 artifact references")
        record_reference = self._mapping(references.get("record_ids"), "p1 record reference")
        mapping_reference = self._mapping(
            references.get("record_to_well"), "p1 record-to-well reference"
        )
        descriptor = SciPlex3PartitionDescriptor(
            partition_id="p1-train",
            artifact_role="train",
            benchmark_role=BenchmarkPartitionRole.TRAIN,
            access_purpose=PopulationComponentAccessPurpose.TRAIN_PARAMETERS,
            record_count=_EXPECTED_P1_RECORD_COUNT,
            well_count=_EXPECTED_P1_WELL_COUNT,
            record_ids_sha256=cast(str, record_reference["sha256"]),
            record_to_well_sha256=cast(str, mapping_reference["sha256"]),
            source_sha256=SCIPLEX3_SOURCE_SHA256,
            loader_contract_sha256=SCIPLEX3_P1_LOADER_CONTRACT_SHA256,
            dataset_manifest_sha256=SCIPLEX3_K562_MANIFEST_SHA256,
            query_sha256=SCIPLEX3_K562_QUERY_SHA256,
            benchmark_sha256=SCIPLEX3_K562_BENCHMARK_SHA256,
            target_value_schema_sha256=SCIPLEX3_K562_TARGET_VALUE_SCHEMA_SHA256,
            scoring_transform_sha256=SCIPLEX3_SCORING_TRANSFORM_SHA256,
            feature_panel_artifact_sha256=SCIPLEX3_FEATURE_PANEL_ARTIFACT_SHA256,
            ordered_feature_keys_sha256=SCIPLEX3_ORDERED_FEATURE_KEYS_SHA256,
            count_access_sealed=False,
        )
        return _PartitionMembership(
            descriptor=descriptor,
            record_ids=record_ids,
            well_ids=well_ids,
            plate_ids=plate_ids,
            record_to_well=record_to_well,
            well_to_condition=well_to_condition,
        )

    def _validate_p1_observations(
        self,
        obs: Any,
        membership: _PartitionMembership,
        feature_panel: Mapping[str, object],
    ) -> _ObservationIndex:
        row_count = SCIPLEX3_H5AD_SHAPE[0]
        rep1_rows = self._rows_in(obs, "replicate", {"rep1"}, row_count)
        p1_plate_rows = self._rows_in(
            obs,
            "plate",
            set(_P1_PLATES),
            row_count,
        )
        p1_design_rows = np.intersect1d(
            rep1_rows,
            p1_plate_rows,
            assume_unique=True,
        ).astype(np.int64, copy=False)
        p1_design_cell_lines = self._decode_column_rows(
            obs,
            "cell_line",
            p1_design_rows,
            allow_missing=False,
        )
        p1_rows = p1_design_rows[np.asarray(p1_design_cell_lines, dtype=object) == "K562"]
        if len(p1_rows) != _EXPECTED_P1_RECORD_COUNT:
            raise ContractViolationError("source p1 design membership differs from frozen contract")
        record_ids = cast(
            list[str],
            self._decode_column_rows(obs, "_index", p1_rows, allow_missing=False),
        )
        dose_units = cast(
            list[str],
            self._decode_column_rows(obs, "dose_unit", p1_rows, allow_missing=False),
        )
        perturbations = cast(
            list[str],
            self._decode_column_rows(obs, "perturbation", p1_rows, allow_missing=False),
        )
        plates = cast(
            list[str],
            self._decode_column_rows(obs, "plate", p1_rows, allow_missing=False),
        )
        replicates = cast(
            list[str],
            self._decode_column_rows(obs, "replicate", p1_rows, allow_missing=False),
        )
        wells = cast(
            list[str],
            self._decode_column_rows(obs, "well", p1_rows, allow_missing=False),
        )
        doses = self._numeric_column_rows(obs, "dose_value", p1_rows)
        times = self._numeric_column_rows(obs, "time", p1_rows)
        if len(set(record_ids)) != len(record_ids):
            raise ContractViolationError("source p1 observation identifiers are not unique")
        if (
            set(dose_units) != {"nM"}
            or set(times.tolist()) != {24.0}
            or set(replicates) != {"rep1"}
            or set(plates) != set(membership.plate_ids)
        ):
            raise ContractViolationError("p1 design metadata differ from frozen scope")

        actual: list[tuple[str, str, str]] = []
        row_by_record: dict[str, int] = {}
        well_by_record: dict[str, str] = {}
        condition_by_record: dict[str, str] = {}
        well_metadata: dict[str, tuple[str, str, int, bool, str]] = {}
        for position, source_row in enumerate(p1_rows.tolist()):
            record_id = record_ids[position]
            plate = plates[position]
            well = wells[position]
            composite_well = _canonical_json_bytes([plate, well]).decode("utf-8")
            raw_dose = float(doses[position])
            if not np.isfinite(raw_dose) or not raw_dose.is_integer():
                raise ContractViolationError("K562 dose contains a nonintegral value")
            dose_nm = int(raw_dose)
            source_label = perturbations[position]
            normalized_label = source_label.strip()
            if not normalized_label:
                raise ContractViolationError("K562 perturbation label is blank")
            is_control = normalized_label == "control"
            if is_control:
                if dose_nm != 0:
                    raise ContractViolationError("vehicle controls must have zero dose")
                condition = "source-control@0nM"
            else:
                if dose_nm not in _SUPPORTED_DOSES_NM:
                    raise ContractViolationError("treated p1 row uses an unsupported dose")
                condition = f"source-label:{normalized_label}@{dose_nm}nM"
            metadata = (source_label, normalized_label, dose_nm, is_control, condition)
            previous = well_metadata.setdefault(composite_well, metadata)
            if previous != metadata:
                raise ContractViolationError("one p1 well has conflicting design metadata")
            actual.append((record_id, composite_well, condition))
            row_by_record[record_id] = int(source_row)
            well_by_record[record_id] = composite_well
            condition_by_record[record_id] = condition

        if len(well_metadata) != _EXPECTED_P1_WELL_COUNT:
            raise ContractViolationError("source p1 composite-well count drifted")
        ordered = sorted(actual)
        actual_records = tuple(item[0] for item in ordered)
        actual_record_to_well = tuple((item[0], item[1]) for item in ordered)
        actual_well_to_condition = tuple(sorted({(item[1], item[2]) for item in ordered}))
        actual_wells = tuple(left for left, _ in actual_well_to_condition)
        if actual_records != membership.record_ids:
            raise ContractViolationError("source p1 record membership differs from ledger")
        if actual_record_to_well != membership.record_to_well:
            raise ContractViolationError("source p1 record-to-well mapping differs from ledger")
        if actual_wells != membership.well_ids:
            raise ContractViolationError("source p1 well membership differs from ledger")
        if actual_well_to_condition != membership.well_to_condition:
            raise ContractViolationError("source p1 condition mapping differs from ledger")

        selection = self._mapping(feature_panel.get("feature_selection"), "feature selection")
        source_rows_payload = _canonical_json_bytes([str(row) for row in p1_rows.tolist()])
        feature_selection_rows_payload = _decimal_string_membership_payload(p1_rows)
        if (
            selection.get("accessed_source_row_count") != len(p1_rows)
            or selection.get("accessed_source_row_indices_sha256")
            != _sha256(feature_selection_rows_payload)
            or selection.get("accessed_source_row_indices_encoded_byte_count")
            != len(feature_selection_rows_payload)
            or selection.get("train_record_ids_sha256") != membership.descriptor.record_ids_sha256
        ):
            raise ContractViolationError("p1 deterministic source-row order identity mismatch")
        ordered_binding_payload = _canonical_json_bytes(
            [
                [
                    record,
                    row_by_record[record],
                    well_by_record[record],
                    condition_by_record[record],
                ]
                for record in membership.record_ids
            ]
        )
        return _ObservationIndex(
            {record: row_by_record[record] for record in membership.record_ids},
            {record: well_by_record[record] for record in membership.record_ids},
            {record: condition_by_record[record] for record in membership.record_ids},
            _sha256(source_rows_payload),
            _sha256(ordered_binding_payload),
        )

    @staticmethod
    def _json_object(payload: bytes, name: str) -> Mapping[str, object]:
        try:
            value = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ContractViolationError(f"invalid JSON artifact: {name}") from error
        if not isinstance(value, dict):
            raise ContractViolationError(f"artifact must be a JSON object: {name}")
        return cast(Mapping[str, object], value)

    @staticmethod
    def _mapping(value: object, name: str) -> Mapping[str, object]:
        if not isinstance(value, dict):
            raise ContractViolationError(f"{name} must be an object")
        return cast(Mapping[str, object], value)

    @staticmethod
    def _string_array(payload: bytes, name: str) -> tuple[str, ...]:
        try:
            value = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ContractViolationError(f"invalid {name} string-array membership") from error
        if (
            not isinstance(value, list)
            or any(not isinstance(item, str) for item in value)
            or value != sorted(value)
            or len(value) != len(set(value))
            or payload != _canonical_json_bytes(value)
        ):
            raise ContractViolationError(f"{name} string-array membership is not canonical")
        return tuple(cast(list[str], value))

    @staticmethod
    def _string_pairs(payload: bytes, name: str) -> tuple[tuple[str, str], ...]:
        try:
            value = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ContractViolationError(f"invalid {name} pair membership") from error
        if not isinstance(value, list) or payload != _canonical_json_bytes(value):
            raise ContractViolationError(f"{name} pair membership is not canonical")
        pairs: list[tuple[str, str]] = []
        for raw_pair in value:
            if (
                not isinstance(raw_pair, list)
                or len(raw_pair) != 2
                or not all(isinstance(item, str) for item in raw_pair)
            ):
                raise ContractViolationError(f"{name} pair membership contains a malformed row")
            pairs.append((cast(str, raw_pair[0]), cast(str, raw_pair[1])))
        if pairs != sorted(pairs) or len({left for left, _ in pairs}) != len(pairs):
            raise ContractViolationError(f"{name} pair membership is not sorted or unique")
        return tuple(pairs)

    def _rows_in(
        self,
        group: Any,
        name: str,
        targets: set[str],
        expected_length: int,
    ) -> npt.NDArray[np.int64]:
        if not targets:
            raise ContractViolationError(f"obs.{name} selector cannot be empty")
        node = group[name]
        encoding_type = _decode_scalar(node.attrs.get("encoding-type"))
        if encoding_type == "categorical":
            if set(node.keys()) != {"categories", "codes"}:
                raise ContractViolationError(f"categorical obs.{name} has an invalid schema")
            categories = self._decode_raw_strings(np.asarray(node["categories"][:]))
            matches = {index for index, value in enumerate(categories) if value in targets}
            if len(matches) != len(targets):
                raise ContractViolationError(f"obs.{name} lacks an exact selector category")
            codes = np.asarray(node["codes"][:], dtype=np.int64)
            if codes.shape != (expected_length,):
                raise ContractViolationError(f"obs.{name} does not cover the source row axis")
            if np.any(codes < -1) or np.any(codes >= len(categories)):
                raise ContractViolationError(f"obs.{name} has out-of-range categorical codes")
            return np.flatnonzero(np.isin(codes, tuple(matches))).astype(np.int64, copy=False)
        raw = self._decode_raw_strings(np.asarray(node[:]))
        if len(raw) != expected_length:
            raise ContractViolationError(f"obs.{name} does not cover the source row axis")
        observed_targets = set(raw) & targets
        if observed_targets != targets:
            raise ContractViolationError(f"obs.{name} lacks an exact selector value")
        return np.flatnonzero(np.isin(np.asarray(raw, dtype=object), tuple(targets))).astype(
            np.int64,
            copy=False,
        )

    def _decode_column_rows(
        self,
        group: Any,
        name: str,
        rows: npt.NDArray[np.int64],
        *,
        allow_missing: bool,
    ) -> list[str | None]:
        node = group[name]
        encoding_type = _decode_scalar(node.attrs.get("encoding-type"))
        if encoding_type == "categorical":
            if set(node.keys()) != {"categories", "codes"}:
                raise ContractViolationError(f"categorical column {name} has an invalid schema")
            categories = self._decode_raw_strings(np.asarray(node["categories"][:]))
            codes = np.asarray(self._read_dataset_rows(node["codes"], rows), dtype=np.int64)
            if np.any(codes < -1) or np.any(codes >= len(categories)):
                raise ContractViolationError(f"categorical column {name} has invalid codes")
            values: list[str | None] = [
                None if int(code) == -1 else categories[int(code)] for code in codes
            ]
        else:
            raw = self._read_dataset_rows(node, rows)
            values = [
                None if value is None else _decode_scalar(value)
                for value in np.asarray(raw, dtype=object).tolist()
            ]
        if not allow_missing and any(value is None or value == "" for value in values):
            raise ContractViolationError(f"column {name} contains a missing/empty value")
        return values

    def _numeric_column_rows(
        self,
        group: Any,
        name: str,
        rows: npt.NDArray[np.int64],
    ) -> npt.NDArray[np.float64]:
        node = group[name]
        encoding_type = _decode_scalar(node.attrs.get("encoding-type"))
        if encoding_type == "categorical":
            categories = np.asarray(node["categories"][:], dtype=np.float64)
            codes = np.asarray(self._read_dataset_rows(node["codes"], rows), dtype=np.int64)
            if np.any(codes < 0) or np.any(codes >= len(categories)):
                raise ContractViolationError(f"numeric categorical column {name} has invalid codes")
            return np.asarray(categories[codes], dtype=np.float64)
        try:
            return np.asarray(self._read_dataset_rows(node, rows), dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise ContractViolationError(f"column {name} is not numeric") from error

    @staticmethod
    def _read_dataset_rows(dataset: Any, rows: npt.NDArray[np.int64]) -> npt.NDArray[Any]:
        if len(rows) == 0:
            return np.asarray(dataset[slice(0, 0)])
        pieces: list[npt.NDArray[Any]] = []
        start = int(rows[0])
        previous = start
        for raw_row in rows[1:]:
            row = int(raw_row)
            if row != previous + 1:
                pieces.append(np.asarray(dataset[slice(start, previous + 1)]))
                start = row
            previous = row
        pieces.append(np.asarray(dataset[slice(start, previous + 1)]))
        return pieces[0] if len(pieces) == 1 else np.concatenate(pieces)

    @staticmethod
    def _decode_raw_strings(values: npt.NDArray[Any]) -> list[str]:
        decoded = [_decode_scalar(value) for value in values.tolist()]
        if any(not value for value in decoded):
            raise ContractViolationError("H5AD string metadata contain an empty value")
        return decoded


__all__ = [
    "SCIPLEX3_FEATURE_COUNT",
    "SCIPLEX3_FEATURE_PANEL_ARTIFACT_SHA256",
    "SCIPLEX3_H5AD_NNZ",
    "SCIPLEX3_H5AD_SHAPE",
    "SCIPLEX3_ORDERED_FEATURE_KEYS_SHA256",
    "SCIPLEX3_P1_LOADER_CONTRACT_SHA256",
    "SCIPLEX3_SCORING_TRANSFORM_SHA256",
    "SCIPLEX3_SOURCE_BYTE_COUNT",
    "SCIPLEX3_SOURCE_FILENAME",
    "SCIPLEX3_SOURCE_MD5",
    "SCIPLEX3_SOURCE_SHA256",
    "SciPlex3FeaturePanel",
    "SciPlex3K562H5ADLoader",
    "SciPlex3P1FinalizedCountScanReceipt",
    "SciPlex3P1SourceScanReceipt",
    "SciPlex3PartitionDescriptor",
    "SciPlex3SparseCountBatch",
    "SciPlex3TrainingDataLoader",
]
