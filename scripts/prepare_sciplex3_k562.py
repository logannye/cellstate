#!/usr/bin/env python3
"""Prepare content-addressed sci-Plex3 K562 benchmark inputs.

This script deliberately stops before constructing a ``DatasetManifest`` or a benchmark
definition.  It verifies one exact public source artifact, derives immutable K562 population and
partition memberships, and selects a count-aware RNA feature panel using TRAIN rows only.

The large H5AD remains outside Git.  Outputs are small JSON ledgers containing counts, exact
membership hashes, per-well descendant hashes, and the ordered feature panel; no count matrix or
normalized observation is written.

``h5py`` is an optional preparation dependency rather than a runtime dependency of ``cellstate``::

    uv run --with 'h5py>=3.11' --no-sync python scripts/prepare_sciplex3_k562.py \
        --source /path/to/SrivatsanTrapnell2020_sciplex3.h5ad
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

SOURCE_FILENAME = "SrivatsanTrapnell2020_sciplex3.h5ad"
SOURCE_URI = (
    "https://zenodo.org/api/records/13350497/files/SrivatsanTrapnell2020_sciplex3.h5ad/content"
)
SOURCE_ACCESSION = "10.5281/zenodo.13350497"
SOURCE_RELEASE = "1.4"
SOURCE_LICENSE = "CC-BY-4.0"
EXPECTED_SOURCE_BYTE_COUNT = 2_526_631_614
EXPECTED_SOURCE_MD5 = "c9e70629505d98c7ca1a837f62b14e89"
EXPECTED_SOURCE_SHA256 = "603ed16c5e25401c8a7f5bb0b2b045179701017d65dcfc6aeea71722a66cd10a"

EXPECTED_H5AD_SHAPE = (799_317, 110_983)
EXPECTED_H5AD_NNZ = 1_007_419_688
EXPECTED_K562_RECORDS = 173_652
EXPECTED_K562_WELLS = 1_536
EXPECTED_TREATED_WELLS = 1_504
EXPECTED_CONTROL_WELLS = 32
EXPECTED_COMPOUNDS = 188
EXPECTED_DOSES_NM = (10, 100, 1_000, 10_000)

GENERATOR_ID = "cellstate.prepare-sciplex3-k562"
GENERATOR_VERSION = "1.0.0"
ARTIFACT_SCHEMA_VERSION = "1.0.0"
FEATURE_SELECTION_ID = "train-logcp10k-robust-dispersion-hvg"
FEATURE_SELECTION_VERSION = "1.0.0"
FEATURE_PANEL_SIZE = 2_000
FEATURE_DETECTION_FRACTION = 0.005
FEATURE_MEAN_BIN_COUNT = 20
FEATURE_ROW_BATCH_SIZE = 512
LOGCP10K_SCALE = 10_000.0

SUMMARY_FILENAMES = (
    "source-verification.json",
    "k562-universe.json",
    "partitions.json",
    "well-groups.json",
    "intervention-labels.json",
    "feature-panel.json",
)


@dataclass(frozen=True)
class PartitionRule:
    """One immutable partition selector over pre-outcome design metadata."""

    role: str
    replicate: str
    plates: tuple[str, ...]


PARTITION_RULES = (
    PartitionRule("train", "rep1", tuple(f"plate{index}" for index in range(1, 9))),
    PartitionRule("calibration", "rep2", ("plate25", "plate26")),
    PartitionRule("model_selection_validation", "rep2", ("plate27", "plate28")),
    PartitionRule(
        "untouched_test",
        "rep2",
        ("plate29", "plate30", "plate31", "plate32"),
    ),
)

MEMBERSHIP_KINDS = (
    "record-ids",
    "well-ids",
    "plate-ids",
    "record-to-well",
    "well-to-condition",
)
MEMBERSHIP_SCOPES = ("universe", *(rule.role for rule in PARTITION_RULES))
MEMBERSHIP_FILENAMES = tuple(
    f"memberships/{scope}-{kind}.json" for scope in MEMBERSHIP_SCOPES for kind in MEMBERSHIP_KINDS
)
MAPPING_FILENAMES = ("mappings/source-label-to-normalized-label.json",)
OUTPUT_FILENAMES = (*SUMMARY_FILENAMES, *MEMBERSHIP_FILENAMES, *MAPPING_FILENAMES)

EXPECTED_PARTITION_COUNTS: Mapping[str, Mapping[str, int]] = {
    "train": {
        "record_count": 94_785,
        "well_count": 768,
        "treated_well_count": 752,
        "control_well_count": 16,
        "compound_count": 188,
    },
    "calibration": {
        "record_count": 18_001,
        "well_count": 192,
        "treated_well_count": 188,
        "control_well_count": 4,
        "compound_count": 47,
    },
    "model_selection_validation": {
        "record_count": 20_481,
        "well_count": 192,
        "treated_well_count": 188,
        "control_well_count": 4,
        "compound_count": 47,
    },
    "untouched_test": {
        "record_count": 40_385,
        "well_count": 384,
        "treated_well_count": 376,
        "control_well_count": 8,
        "compound_count": 94,
    },
}

TECHNICAL_GENE_PATTERN = re.compile(r"^(?:MT-|RPL|RPS)", flags=re.IGNORECASE)

StringArray = npt.NDArray[np.object_]
IntegerArray = npt.NDArray[np.integer[Any]]
FloatArray = npt.NDArray[np.floating[Any]]


class Sliceable1D(Protocol):
    """The small surface shared by NumPy and h5py one-dimensional arrays."""

    def __getitem__(self, key: slice) -> npt.NDArray[Any]: ...


@dataclass(frozen=True)
class FeatureStatistics:
    """TRAIN-only sufficient statistics used for deterministic feature selection."""

    detection_count: npt.NDArray[np.int64]
    raw_count_total: npt.NDArray[np.int64]
    log_sum: npt.NDArray[np.float64]
    log_sum_squares: npt.NDArray[np.float64]
    accessed_rows: tuple[int, ...]
    accessed_library_count: int
    accessed_count_total: int


def canonical_json_bytes(value: object) -> bytes:
    """Encode canonical compact JSON used by all semantic membership hashes."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def hash_string_array(values: Iterable[str], *, sort_values: bool = True) -> tuple[str, int]:
    """Hash a canonical JSON string array and return its hash and encoded byte count."""

    materialized = list(values)
    if sort_values:
        materialized.sort()
    if len(materialized) != len(set(materialized)):
        raise ValueError("canonical string-array membership must be unique")
    encoded = canonical_json_bytes(materialized)
    return hashlib.sha256(encoded).hexdigest(), len(encoded)


def hash_string_pairs(
    values: Iterable[tuple[str, str]],
    *,
    sort_values: bool = True,
) -> tuple[str, int]:
    """Hash a canonical JSON array of two-string descendant/group pairs."""

    materialized = list(values)
    if sort_values:
        materialized.sort()
    if len(materialized) != len({left for left, _ in materialized}):
        raise ValueError("descendant mapping must contain each record exactly once")
    encoded = canonical_json_bytes(materialized)
    return hashlib.sha256(encoded).hexdigest(), len(encoded)


def digest_file(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> tuple[str, str, int]:
    """Calculate SHA-256, MD5, and byte count in one sequential source read."""

    sha256 = hashlib.sha256()
    md5 = hashlib.md5(usedforsecurity=False)
    byte_count = 0
    with path.open("rb") as stream:
        while block := stream.read(chunk_size):
            sha256.update(block)
            md5.update(block)
            byte_count += len(block)
    return sha256.hexdigest(), md5.hexdigest(), byte_count


def _decode_strings(values: npt.NDArray[Any]) -> StringArray:
    return np.asarray(
        [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in values],
        dtype=object,
    )


def decode_h5ad_column(group: Any, name: str) -> npt.NDArray[Any]:
    """Decode an AnnData 0.2 categorical or ordinary array without importing AnnData."""

    node = group[name]
    encoding_type = node.attrs.get("encoding-type")
    if isinstance(encoding_type, bytes):
        encoding_type = encoding_type.decode("utf-8")
    if encoding_type == "categorical":
        categories = _decode_strings(node["categories"][:])
        codes = np.asarray(node["codes"][:], dtype=np.int64)
        if np.any(codes < -1) or np.any(codes >= len(categories)):
            raise ValueError(f"{name} contains out-of-range categorical codes")
        decoded = np.full(len(codes), None, dtype=object)
        observed = codes >= 0
        decoded[observed] = categories[codes[observed]]
        return decoded
    values = node[:]
    if getattr(values, "dtype", None) is not None and values.dtype.kind in {"O", "S", "U"}:
        return _decode_strings(values)
    return np.asarray(values)


def partition_role(replicate: str, plate: str) -> str:
    """Resolve a design row to exactly one frozen partition without observing its counts."""

    matches = [
        rule.role
        for rule in PARTITION_RULES
        if replicate == rule.replicate and plate in rule.plates
    ]
    if len(matches) != 1:
        raise ValueError(f"K562 row ({replicate!r}, {plate!r}) does not resolve to one partition")
    return matches[0]


def composite_well_id(plate: str, well: str) -> str:
    """Encode the typed composite-source-field WELL identity as canonical JSON."""

    if not plate or not well or plate != plate.strip() or well != well.strip():
        raise ValueError("plate and well source identifiers must be canonical")
    return canonical_json_bytes([plate, well]).decode("utf-8")


def normalized_perturbation_label(value: str) -> str:
    """Trim source presentation whitespace without fuzzy or ontology matching."""

    normalized = value.strip()
    if not normalized:
        raise ValueError("perturbation label must be nonempty")
    return normalized


def condition_id(perturbation: str, dose_nm: int, *, is_control: bool) -> str:
    """Build a source-scoped condition identity; this is not a chemical ontology mapping."""

    if is_control:
        if perturbation != "control" or dose_nm != 0:
            raise ValueError("control wells must use perturbation=control and dose=0")
        return "source-control@0nM"
    if perturbation == "control" or dose_nm not in EXPECTED_DOSES_NM:
        raise ValueError("treated wells must use a supported noncontrol compound and dose")
    return f"source-label:{perturbation}@{dose_nm}nM"


def compute_train_feature_statistics(
    *,
    data: Sliceable1D,
    indices: Sliceable1D,
    indptr: IntegerArray,
    train_rows: Sequence[int],
    n_features: int,
    expected_train_library_sizes: npt.NDArray[Any] | None = None,
    row_batch_size: int = FEATURE_ROW_BATCH_SIZE,
) -> FeatureStatistics:
    """Accumulate logCP10k statistics while requesting only explicitly supplied TRAIN rows.

    The function never slices a span of rows and never receives a held-out mask. Each read is the
    exact CSR interval of one TRAIN row. Fixed-size batching changes only accumulation efficiency,
    not which source counts are requested.
    """

    if row_batch_size <= 0:
        raise ValueError("row_batch_size must be positive")
    rows = tuple(int(row) for row in train_rows)
    if not rows or tuple(sorted(rows)) != rows or len(rows) != len(set(rows)):
        raise ValueError("TRAIN row indices must be nonempty, unique, and sorted")
    if rows[0] < 0 or rows[-1] + 1 >= len(indptr):
        raise ValueError("TRAIN row index lies outside the CSR row axis")
    if n_features <= 0:
        raise ValueError("feature count must be positive")
    if expected_train_library_sizes is not None and len(expected_train_library_sizes) != len(rows):
        raise ValueError("expected TRAIN library sizes must align to TRAIN rows")

    detection = np.zeros(n_features, dtype=np.int64)
    raw_total = np.zeros(n_features, dtype=np.int64)
    log_sum = np.zeros(n_features, dtype=np.float64)
    log_sum_squares = np.zeros(n_features, dtype=np.float64)
    total_counts = 0

    for batch_start in range(0, len(rows), row_batch_size):
        batch_rows = rows[batch_start : batch_start + row_batch_size]
        batch_columns: list[npt.NDArray[np.int64]] = []
        batch_raw: list[npt.NDArray[np.int64]] = []
        batch_log: list[npt.NDArray[np.float64]] = []

        for batch_offset, row in enumerate(batch_rows):
            start = int(indptr[row])
            stop = int(indptr[row + 1])
            row_counts = np.asarray(data[slice(start, stop)], dtype=np.int64)
            row_columns = np.asarray(indices[slice(start, stop)], dtype=np.int64)
            if len(row_counts) != len(row_columns):
                raise ValueError("CSR data and index slices differ in length")
            if np.any(row_counts <= 0):
                raise ValueError("raw sparse counts must be positive integers")
            if np.any(row_columns < 0) or np.any(row_columns >= n_features):
                raise ValueError("CSR feature index lies outside the source axis")
            if len(row_columns) > 1 and np.any(np.diff(row_columns) <= 0):
                raise ValueError("CSR row feature indices must be strictly increasing")
            library_size = int(row_counts.sum(dtype=np.int64))
            if library_size <= 0:
                raise ValueError("TRAIN cells must have a positive raw-count library size")
            if expected_train_library_sizes is not None:
                train_position = batch_start + batch_offset
                observed = float(expected_train_library_sizes[train_position])
                if not math.isfinite(observed) or observed != library_size:
                    raise ValueError(
                        f"CSR row sum disagrees with source ncounts for row {row}: "
                        f"{library_size} != {observed}"
                    )
            transformed = np.log1p(LOGCP10K_SCALE * row_counts / library_size)
            batch_columns.append(row_columns)
            batch_raw.append(row_counts)
            batch_log.append(np.asarray(transformed, dtype=np.float64))
            total_counts += library_size

        columns = np.concatenate(batch_columns)
        raw = np.concatenate(batch_raw)
        transformed = np.concatenate(batch_log)
        detection += np.bincount(columns, minlength=n_features).astype(np.int64)
        raw_total += np.bincount(columns, weights=raw, minlength=n_features).astype(np.int64)
        log_sum += np.bincount(columns, weights=transformed, minlength=n_features)
        log_sum_squares += np.bincount(
            columns,
            weights=transformed * transformed,
            minlength=n_features,
        )

    return FeatureStatistics(
        detection_count=detection,
        raw_count_total=raw_total,
        log_sum=log_sum,
        log_sum_squares=log_sum_squares,
        accessed_rows=rows,
        accessed_library_count=len(rows),
        accessed_count_total=total_counts,
    )


def _mean_bin_scores(
    *,
    mean: FloatArray,
    dispersion: FloatArray,
    eligible_indices: Sequence[int],
    stable_keys: Sequence[tuple[str, str, int]],
    bin_count: int,
) -> npt.NDArray[np.float64]:
    """Robustly standardize dispersion in deterministic equal-frequency mean bins."""

    if bin_count <= 0:
        raise ValueError("mean-bin count must be positive")
    ordered = sorted(eligible_indices, key=lambda index: (mean[index], stable_keys[index]))
    scores = np.full(len(mean), -np.inf, dtype=np.float64)
    raw_bins = np.array_split(
        np.asarray(ordered, dtype=np.int64),
        min(bin_count, len(ordered)),
    )
    for raw_bin in raw_bins:
        if len(raw_bin) == 0:
            continue
        values = dispersion[raw_bin]
        center = float(np.median(values))
        scale = 1.4826 * float(np.median(np.abs(values - center)))
        if not math.isfinite(scale) or scale <= 1e-12:
            scale = float(np.std(values, ddof=1)) if len(values) > 1 else 1.0
        if not math.isfinite(scale) or scale <= 1e-12:
            scale = 1.0
        scores[raw_bin] = (values - center) / scale
    return scores


def select_train_feature_panel(
    *,
    statistics: FeatureStatistics,
    ensembl_ids: Sequence[str],
    gene_symbols: Sequence[str],
    train_cell_count: int,
    panel_size: int = FEATURE_PANEL_SIZE,
    minimum_detection_fraction: float = FEATURE_DETECTION_FRACTION,
    mean_bin_count: int = FEATURE_MEAN_BIN_COUNT,
) -> tuple[list[dict[str, object]], dict[str, int]]:
    """Select a stable mean-aware, variance-rich panel from TRAIN logCP10k statistics."""

    n_features = len(ensembl_ids)
    if len(gene_symbols) != n_features:
        raise ValueError("source gene-symbol and Ensembl axes differ in length")
    if train_cell_count != statistics.accessed_library_count:
        raise ValueError("feature-statistics row count does not match TRAIN membership")
    if not 0 < minimum_detection_fraction <= 1:
        raise ValueError("minimum detection fraction must lie in (0, 1]")
    if panel_size <= 0:
        raise ValueError("feature-panel size must be positive")

    minimum_detection_count = math.ceil(minimum_detection_fraction * train_cell_count)
    ensembl_multiplicity = Counter(ensembl_ids)
    mean = statistics.log_sum / train_cell_count
    numerator = statistics.log_sum_squares - (
        statistics.log_sum * statistics.log_sum / train_cell_count
    )
    variance = np.maximum(numerator / max(train_cell_count - 1, 1), 0.0)
    dispersion = np.divide(
        variance,
        mean,
        out=np.zeros_like(variance),
        where=mean > 0,
    )

    exclusion_counts: Counter[str] = Counter()
    eligible: list[int] = []
    stable_keys: list[tuple[str, str, int]] = []
    for index, (ensembl_id, symbol) in enumerate(zip(ensembl_ids, gene_symbols, strict=True)):
        stable_keys.append((ensembl_id, symbol, index))
        if not ensembl_id.startswith("ENSG"):
            exclusion_counts["nonhuman_or_noncanonical_ensembl_id"] += 1
        elif ensembl_multiplicity[ensembl_id] != 1:
            exclusion_counts["duplicate_ensembl_id"] += 1
        elif TECHNICAL_GENE_PATTERN.match(symbol):
            exclusion_counts["mitochondrial_or_ribosomal_symbol"] += 1
        elif statistics.detection_count[index] < minimum_detection_count:
            exclusion_counts["below_train_detection_threshold"] += 1
        elif not math.isfinite(mean[index]) or not math.isfinite(variance[index]):
            exclusion_counts["nonfinite_train_statistic"] += 1
        elif mean[index] <= 0 or variance[index] <= 0:
            exclusion_counts["zero_train_mean_or_variance"] += 1
        else:
            eligible.append(index)

    if len(eligible) < panel_size:
        raise ValueError(
            f"only {len(eligible)} source features pass TRAIN-only eligibility; "
            f"cannot select {panel_size}"
        )
    normalized_score = _mean_bin_scores(
        mean=np.asarray(mean, dtype=np.float64),
        dispersion=np.asarray(dispersion, dtype=np.float64),
        eligible_indices=eligible,
        stable_keys=stable_keys,
        bin_count=mean_bin_count,
    )
    ordered = sorted(
        eligible,
        key=lambda index: (
            -normalized_score[index],
            -variance[index],
            -int(statistics.detection_count[index]),
            stable_keys[index],
        ),
    )[:panel_size]

    panel: list[dict[str, object]] = []
    for rank, index in enumerate(ordered, start=1):
        panel.append(
            {
                "rank": rank,
                "source_feature_index": index,
                "ensembl_id": ensembl_ids[index],
                "gene_symbol": gene_symbols[index],
                "train_detection_count": int(statistics.detection_count[index]),
                "train_detection_fraction": round(
                    float(statistics.detection_count[index] / train_cell_count), 12
                ),
                "train_raw_count_total": int(statistics.raw_count_total[index]),
                "train_logcp10k_mean": round(float(mean[index]), 12),
                "train_logcp10k_variance": round(float(variance[index]), 12),
                "train_logcp10k_dispersion": round(float(dispersion[index]), 12),
                "train_mean_bin_robust_score": round(float(normalized_score[index]), 12),
            }
        )

    selected_keys = [(item["ensembl_id"], item["gene_symbol"]) for item in panel]
    if len(selected_keys) != len(set(selected_keys)):
        raise ValueError("selected feature panel must be unique")
    eligibility_summary = dict(sorted(exclusion_counts.items()))
    eligibility_summary["eligible_feature_count"] = len(eligible)
    eligibility_summary["minimum_train_detection_count"] = minimum_detection_count
    eligibility_summary["selected_feature_count"] = len(panel)
    return panel, eligibility_summary


def _require_exact_mapping(
    actual: Mapping[str, int],
    expected: Mapping[str, int],
    name: str,
) -> None:
    for key, expected_value in expected.items():
        actual_value = actual.get(key)
        if actual_value != expected_value:
            raise ValueError(
                f"{name} {key} mismatch: expected {expected_value}, got {actual_value}"
            )


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_canonical_json(path: Path, value: object) -> None:
    """Write exact canonical bytes suitable for a typed membership artifact."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value))


def _artifact_record(path: Path, root: Path) -> dict[str, object]:
    content = path.read_bytes()
    return {
        "relative_path": path.relative_to(root).as_posix(),
        "media_type": "application/json",
        "byte_count": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def _membership_artifact_reference(
    *,
    relative_path: str,
    sha256: str,
    byte_count: int,
    encoding: str,
) -> dict[str, object]:
    return {
        "relative_path": relative_path,
        "media_type": "application/json",
        "encoding": encoding,
        "sha256": sha256,
        "byte_count": byte_count,
    }


def _import_h5py() -> Any:
    try:
        return importlib.import_module("h5py")
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "sci-Plex3 preparation requires optional h5py; run with "
            "`uv run --with 'h5py>=3.11' --no-sync python ...`"
        ) from error


def prepare(source: Path, output_root: Path) -> dict[str, object]:
    """Verify the corrected source and write deterministic preparation artifacts."""

    if source.name != SOURCE_FILENAME:
        raise ValueError(f"source filename must be {SOURCE_FILENAME!r}")
    source_sha256, source_md5, source_byte_count = digest_file(source)
    if source_byte_count != EXPECTED_SOURCE_BYTE_COUNT:
        raise ValueError(
            f"source byte count mismatch: {source_byte_count} != {EXPECTED_SOURCE_BYTE_COUNT}"
        )
    if source_md5 != EXPECTED_SOURCE_MD5:
        raise ValueError(f"source MD5 mismatch: {source_md5} != {EXPECTED_SOURCE_MD5}")
    if source_sha256 != EXPECTED_SOURCE_SHA256:
        raise ValueError(f"source SHA-256 mismatch: {source_sha256} != {EXPECTED_SOURCE_SHA256}")

    h5py = _import_h5py()
    script_path = Path(__file__).resolve()
    script_sha256 = hashlib.sha256(script_path.read_bytes()).hexdigest()
    membership_payloads: dict[str, object] = {}
    mapping_payloads: dict[str, object] = {}

    with h5py.File(source, "r") as h5ad:
        required_root_keys = {"X", "obs", "var"}
        if not required_root_keys <= set(h5ad.keys()):
            raise ValueError("source H5AD is missing X, obs, or var")
        matrix = h5ad["X"]
        matrix_shape = tuple(int(value) for value in matrix.attrs["shape"])
        if matrix_shape != EXPECTED_H5AD_SHAPE:
            raise ValueError(f"source matrix shape mismatch: {matrix_shape}")
        if matrix.attrs.get("encoding-type") not in {"csr_matrix", b"csr_matrix"}:
            raise ValueError("source X must use CSR encoding")
        if tuple(matrix.keys()) != ("data", "indices", "indptr"):
            raise ValueError("source CSR group must contain data, indices, and indptr")
        if len(matrix["data"]) != EXPECTED_H5AD_NNZ or len(matrix["indices"]) != EXPECTED_H5AD_NNZ:
            raise ValueError("source CSR nonzero count differs from the corrected release")
        if matrix["data"].dtype.kind not in {"i", "u"}:
            raise ValueError("source X must retain integer UMI counts")
        indptr = np.asarray(matrix["indptr"][:], dtype=np.int64)
        if len(indptr) != matrix_shape[0] + 1 or indptr[0] != 0:
            raise ValueError("source CSR indptr is malformed")
        if np.any(np.diff(indptr) < 0) or int(indptr[-1]) != EXPECTED_H5AD_NNZ:
            raise ValueError("source CSR indptr is not monotone or does not cover X.data")

        obs = h5ad["obs"]
        required_obs_columns = {
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
        if not required_obs_columns <= set(obs.keys()):
            raise ValueError(f"source obs lacks required fields: {required_obs_columns - set(obs)}")
        record_ids = np.asarray(decode_h5ad_column(obs, "_index"), dtype=object)
        cell_lines = np.asarray(decode_h5ad_column(obs, "cell_line"), dtype=object)
        dose_units = np.asarray(decode_h5ad_column(obs, "dose_unit"), dtype=object)
        dose_values = np.asarray(decode_h5ad_column(obs, "dose_value"), dtype=np.float64)
        perturbations = np.asarray(decode_h5ad_column(obs, "perturbation"), dtype=object)
        plates = np.asarray(decode_h5ad_column(obs, "plate"), dtype=object)
        replicates = np.asarray(decode_h5ad_column(obs, "replicate"), dtype=object)
        times = np.asarray(decode_h5ad_column(obs, "time"), dtype=np.float64)
        wells = np.asarray(decode_h5ad_column(obs, "well"), dtype=object)
        if any(
            len(values) != matrix_shape[0]
            for values in (
                record_ids,
                cell_lines,
                dose_units,
                dose_values,
                perturbations,
                plates,
                replicates,
                times,
                wells,
            )
        ):
            raise ValueError("source observation columns do not cover the matrix row axis")
        if len(set(record_ids.tolist())) != len(record_ids):
            raise ValueError("source observation identifiers must be globally unique")

        var = h5ad["var"]
        if not {"ensembl_id", "gene_symbol"} <= set(var.keys()):
            raise ValueError("source var lacks Ensembl IDs or gene symbols")
        source_ensembl_ids = decode_h5ad_column(var, "ensembl_id").tolist()
        source_gene_symbols = decode_h5ad_column(var, "gene_symbol").tolist()
        if any(value is not None and not isinstance(value, str) for value in source_ensembl_ids):
            raise ValueError("source Ensembl identifiers must be strings or AnnData missing values")
        if any(not isinstance(value, str) or not value for value in source_gene_symbols):
            raise ValueError("source gene symbols must be nonempty strings")
        ensembl_ids = [value if value is not None else "" for value in source_ensembl_ids]
        gene_symbols = [str(value) for value in source_gene_symbols]
        if len(ensembl_ids) != matrix_shape[1] or len(gene_symbols) != matrix_shape[1]:
            raise ValueError("source feature metadata do not cover the matrix column axis")
        if len(set(gene_symbols)) != len(gene_symbols):
            raise ValueError("corrected source gene-symbol axis must be unique")

        k562_rows = np.flatnonzero(cell_lines == "K562")
        if len(k562_rows) != EXPECTED_K562_RECORDS:
            raise ValueError(f"K562 record count mismatch: {len(k562_rows)}")
        required_k562_values = {
            "record ID": record_ids,
            "dose unit": dose_units,
            "perturbation": perturbations,
            "plate": plates,
            "replicate": replicates,
            "well": wells,
        }
        for field_name, values in required_k562_values.items():
            if any(value is None for value in values[k562_rows].tolist()):
                raise ValueError(f"K562 {field_name} contains an AnnData missing value")
        if set(dose_units[k562_rows].tolist()) != {"nM"}:
            raise ValueError("K562 dose units are not uniformly nM")
        if set(times[k562_rows].tolist()) != {24.0}:
            raise ValueError("K562 observations are not uniformly at the 24-hour endpoint")
        if set(replicates[k562_rows].tolist()) != {"rep1", "rep2"}:
            raise ValueError("K562 source does not expose the expected two replicate screens")

        partition_rows: dict[str, list[int]] = {rule.role: [] for rule in PARTITION_RULES}
        records_by_well: dict[str, list[str]] = defaultdict(list)
        row_indices_by_well: dict[str, list[int]] = defaultdict(list)
        metadata_by_well: dict[str, tuple[str, str, str, str, int, bool, str]] = {}
        descendant_pairs: list[tuple[str, str]] = []

        for row in k562_rows.tolist():
            record_id = str(record_ids[row])
            plate = str(plates[row])
            well = str(wells[row])
            replicate = str(replicates[row])
            role = partition_role(replicate, plate)
            population_id = composite_well_id(plate, well)
            raw_dose = float(dose_values[row])
            if not raw_dose.is_integer():
                raise ValueError(f"nonintegral K562 dose value in row {row}: {raw_dose}")
            dose_nm = int(raw_dose)
            source_label = str(perturbations[row])
            perturbation = normalized_perturbation_label(source_label)
            is_control = perturbation == "control"
            scoped_condition_id = condition_id(perturbation, dose_nm, is_control=is_control)
            metadata = (
                replicate,
                role,
                source_label,
                perturbation,
                dose_nm,
                is_control,
                scoped_condition_id,
            )
            previous = metadata_by_well.setdefault(population_id, metadata)
            if previous != metadata:
                raise ValueError(f"well {population_id!r} contains conflicting design metadata")
            partition_rows[role].append(row)
            records_by_well[population_id].append(record_id)
            row_indices_by_well[population_id].append(row)
            descendant_pairs.append((record_id, population_id))

        if len(records_by_well) != EXPECTED_K562_WELLS:
            raise ValueError(f"K562 well count mismatch: {len(records_by_well)}")
        if sum(len(values) for values in records_by_well.values()) != EXPECTED_K562_RECORDS:
            raise ValueError("K562 well descendants do not cover the exact record universe")

        well_groups: list[dict[str, object]] = []
        for population_id in sorted(records_by_well):
            plate, well = json.loads(population_id)
            (
                replicate,
                role,
                source_label,
                perturbation,
                dose_nm,
                is_control,
                scoped_condition_id,
            ) = metadata_by_well[population_id]
            member_ids = records_by_well[population_id]
            member_sha256, member_bytes = hash_string_array(member_ids)
            source_rows = sorted(row_indices_by_well[population_id])
            source_row_sha256, source_row_bytes = hash_string_array(str(row) for row in source_rows)
            well_groups.append(
                {
                    "composite_well_id": population_id,
                    "plate": plate,
                    "well": well,
                    "replicate": replicate,
                    "partition_role": role,
                    "source_perturbation_label": source_label,
                    "normalized_perturbation_label": perturbation,
                    "dose_value_nm": dose_nm,
                    "is_vehicle_control": is_control,
                    "source_scoped_condition_id": scoped_condition_id,
                    "record_count": len(member_ids),
                    "record_ids_encoding": "canonical_json_utf8_string_array_v1",
                    "record_ids_sha256": member_sha256,
                    "record_ids_encoded_byte_count": member_bytes,
                    "source_row_indices_encoding": "canonical_json_utf8_decimal_string_array_v1",
                    "source_row_indices_sha256": source_row_sha256,
                    "source_row_indices_encoded_byte_count": source_row_bytes,
                }
            )

        partition_payloads: list[dict[str, object]] = []
        partition_compounds: dict[str, set[str]] = {}
        partition_plates: dict[str, set[str]] = {}
        for rule in PARTITION_RULES:
            role_groups = [group for group in well_groups if group["partition_role"] == rule.role]
            role_well_ids = [str(group["composite_well_id"]) for group in role_groups]
            role_record_ids = [
                record_id
                for population_id in role_well_ids
                for record_id in records_by_well[population_id]
            ]
            role_pairs = [
                (record_id, population_id)
                for population_id in role_well_ids
                for record_id in records_by_well[population_id]
            ]
            treated_groups = [group for group in role_groups if not group["is_vehicle_control"]]
            control_groups = [group for group in role_groups if group["is_vehicle_control"]]
            source_labels = {str(group["source_perturbation_label"]) for group in treated_groups}
            compounds = {str(group["normalized_perturbation_label"]) for group in treated_groups}
            conditions = {str(group["source_scoped_condition_id"]) for group in treated_groups}
            observed_plates = {str(group["plate"]) for group in role_groups}
            role_condition_pairs = [
                (
                    str(group["composite_well_id"]),
                    str(group["source_scoped_condition_id"]),
                )
                for group in role_groups
            ]
            if observed_plates != set(rule.plates):
                raise ValueError(f"{rule.role} plate membership differs from the frozen selector")
            record_sha256, record_bytes = hash_string_array(role_record_ids)
            well_sha256, well_bytes = hash_string_array(role_well_ids)
            mapping_sha256, mapping_bytes = hash_string_pairs(role_pairs)
            plate_sha256, plate_bytes = hash_string_array(observed_plates)
            condition_mapping_sha256, condition_mapping_bytes = hash_string_pairs(
                role_condition_pairs
            )
            source_label_sha256, source_label_bytes = hash_string_array(source_labels)
            normalized_label_sha256, normalized_label_bytes = hash_string_array(compounds)
            role_membership_paths = {
                kind: f"memberships/{rule.role}-{kind}.json" for kind in MEMBERSHIP_KINDS
            }
            membership_payloads.update(
                {
                    role_membership_paths["record-ids"]: sorted(role_record_ids),
                    role_membership_paths["well-ids"]: sorted(role_well_ids),
                    role_membership_paths["plate-ids"]: sorted(observed_plates),
                    role_membership_paths["record-to-well"]: sorted(role_pairs),
                    role_membership_paths["well-to-condition"]: sorted(role_condition_pairs),
                }
            )
            counts = {
                "record_count": len(role_record_ids),
                "well_count": len(role_groups),
                "treated_well_count": len(treated_groups),
                "control_well_count": len(control_groups),
                "compound_count": len(compounds),
            }
            _require_exact_mapping(counts, EXPECTED_PARTITION_COUNTS[rule.role], rule.role)
            if len(conditions) != len(treated_groups):
                raise ValueError(f"{rule.role} contains duplicated treated condition wells")
            partition_compounds[rule.role] = compounds
            partition_plates[rule.role] = observed_plates
            partition_payloads.append(
                {
                    "partition_role": rule.role,
                    "selector": {
                        "replicate": rule.replicate,
                        "plate": list(rule.plates),
                    },
                    **counts,
                    "record_ids_encoding": "canonical_json_utf8_string_array_v1",
                    "record_ids_sha256": record_sha256,
                    "record_ids_encoded_byte_count": record_bytes,
                    "composite_well_ids_encoding": "canonical_json_utf8_string_array_v1",
                    "composite_well_ids_sha256": well_sha256,
                    "composite_well_ids_encoded_byte_count": well_bytes,
                    "record_to_well_encoding": "canonical_json_utf8_string_pair_array_v1",
                    "record_to_well_sha256": mapping_sha256,
                    "record_to_well_encoded_byte_count": mapping_bytes,
                    "plate_ids_encoding": "canonical_json_utf8_string_array_v1",
                    "plate_ids_sha256": plate_sha256,
                    "plate_ids_encoded_byte_count": plate_bytes,
                    "well_to_condition_encoding": ("canonical_json_utf8_string_pair_array_v1"),
                    "well_to_condition_sha256": condition_mapping_sha256,
                    "well_to_condition_encoded_byte_count": condition_mapping_bytes,
                    "condition_group_count": len(set(dict(role_condition_pairs).values())),
                    "source_perturbation_labels_encoding": ("canonical_json_utf8_string_array_v1"),
                    "source_perturbation_labels_sha256": source_label_sha256,
                    "source_perturbation_labels_encoded_byte_count": source_label_bytes,
                    "normalized_perturbation_labels_encoding": (
                        "canonical_json_utf8_string_array_v1"
                    ),
                    "normalized_perturbation_labels_sha256": normalized_label_sha256,
                    "normalized_perturbation_labels_encoded_byte_count": (normalized_label_bytes),
                    "dose_values_nm": list(EXPECTED_DOSES_NM),
                    "membership_artifacts": {
                        "record_ids": _membership_artifact_reference(
                            relative_path=role_membership_paths["record-ids"],
                            sha256=record_sha256,
                            byte_count=record_bytes,
                            encoding="canonical_json_utf8_string_array_v1",
                        ),
                        "well_ids": _membership_artifact_reference(
                            relative_path=role_membership_paths["well-ids"],
                            sha256=well_sha256,
                            byte_count=well_bytes,
                            encoding="canonical_json_utf8_string_array_v1",
                        ),
                        "plate_ids": _membership_artifact_reference(
                            relative_path=role_membership_paths["plate-ids"],
                            sha256=plate_sha256,
                            byte_count=plate_bytes,
                            encoding="canonical_json_utf8_string_array_v1",
                        ),
                        "record_to_well": _membership_artifact_reference(
                            relative_path=role_membership_paths["record-to-well"],
                            sha256=mapping_sha256,
                            byte_count=mapping_bytes,
                            encoding="canonical_json_utf8_string_pair_array_v1",
                        ),
                        "well_to_condition": _membership_artifact_reference(
                            relative_path=role_membership_paths["well-to-condition"],
                            sha256=condition_mapping_sha256,
                            byte_count=condition_mapping_bytes,
                            encoding="canonical_json_utf8_string_pair_array_v1",
                        ),
                    },
                }
            )

        role_names = [rule.role for rule in PARTITION_RULES]
        for index, left in enumerate(role_names):
            for right in role_names[index + 1 :]:
                if partition_plates[left] & partition_plates[right]:
                    raise ValueError(f"protected culture plates overlap: {left}, {right}")
        heldout_roles = (
            "calibration",
            "model_selection_validation",
            "untouched_test",
        )
        for index, left in enumerate(heldout_roles):
            for right in heldout_roles[index + 1 :]:
                if partition_compounds[left] & partition_compounds[right]:
                    raise ValueError(f"rep2 compound groups overlap: {left}, {right}")
        heldout_union = set().union(*(partition_compounds[role] for role in heldout_roles))
        if len(heldout_union) != EXPECTED_COMPOUNDS:
            raise ValueError("rep2 compound partitions do not cover all 188 compounds")
        if partition_compounds["train"] != heldout_union:
            raise ValueError("each held-out compound must have an exact rep1 TRAIN counterpart")

        k562_record_ids = [str(record_ids[row]) for row in k562_rows]
        universe_record_sha256, universe_record_bytes = hash_string_array(k562_record_ids)
        universe_well_sha256, universe_well_bytes = hash_string_array(records_by_well)
        descendant_sha256, descendant_bytes = hash_string_pairs(descendant_pairs)
        source_row_sha256, source_row_bytes = hash_string_array(
            str(row) for row in k562_rows.tolist()
        )
        universe_plate_ids = sorted({str(group["plate"]) for group in well_groups})
        universe_condition_pairs = sorted(
            (
                str(group["composite_well_id"]),
                str(group["source_scoped_condition_id"]),
            )
            for group in well_groups
        )
        condition_well_counts = Counter(
            condition_id for _, condition_id in universe_condition_pairs
        )
        condition_replicates: dict[str, set[str]] = defaultdict(set)
        for group in well_groups:
            condition_replicates[str(group["source_scoped_condition_id"])].add(
                str(group["replicate"])
            )
        treated_condition_ids = set(condition_well_counts) - {"source-control@0nM"}
        if len(treated_condition_ids) != EXPECTED_TREATED_WELLS // 2:
            raise ValueError("replicated design does not contain 752 treated conditions")
        if {condition_well_counts[value] for value in treated_condition_ids} != {2}:
            raise ValueError("each treated compound-dose condition must contain exactly two wells")
        if {frozenset(condition_replicates[value]) for value in treated_condition_ids} != {
            frozenset({"rep1", "rep2"})
        }:
            raise ValueError("each treated condition must contain one well in each replicate")
        if condition_well_counts["source-control@0nM"] != EXPECTED_CONTROL_WELLS:
            raise ValueError("vehicle condition must contain the expected 32 control wells")
        universe_plate_sha256, universe_plate_bytes = hash_string_array(universe_plate_ids)
        universe_condition_sha256, universe_condition_bytes = hash_string_pairs(
            universe_condition_pairs
        )
        source_label_pairs = sorted(
            {
                (
                    str(group["source_perturbation_label"]),
                    str(group["normalized_perturbation_label"]),
                )
                for group in well_groups
            }
        )
        source_label_mapping_sha256, source_label_mapping_bytes = hash_string_pairs(
            source_label_pairs
        )
        if len(source_label_pairs) != EXPECTED_COMPOUNDS + 1:
            raise ValueError("source label mapping does not cover 188 compounds plus control")
        if len({normalized for _, normalized in source_label_pairs}) != len(source_label_pairs):
            raise ValueError("whitespace normalization creates a perturbation-label collision")
        source_label_mapping_path = MAPPING_FILENAMES[0]
        mapping_payloads[source_label_mapping_path] = source_label_pairs
        universe_membership_paths = {
            kind: f"memberships/universe-{kind}.json" for kind in MEMBERSHIP_KINDS
        }
        membership_payloads.update(
            {
                universe_membership_paths["record-ids"]: sorted(k562_record_ids),
                universe_membership_paths["well-ids"]: sorted(records_by_well),
                universe_membership_paths["plate-ids"]: universe_plate_ids,
                universe_membership_paths["record-to-well"]: sorted(descendant_pairs),
                universe_membership_paths["well-to-condition"]: universe_condition_pairs,
            }
        )

        treated_well_count = sum(not bool(group["is_vehicle_control"]) for group in well_groups)
        control_well_count = sum(bool(group["is_vehicle_control"]) for group in well_groups)
        if treated_well_count != EXPECTED_TREATED_WELLS:
            raise ValueError("K562 treated-well count differs from the complete replicated design")
        if control_well_count != EXPECTED_CONTROL_WELLS:
            raise ValueError("K562 control-well count differs from the complete replicated design")

        train_rows = tuple(sorted(partition_rows["train"]))
        train_record_ids = [str(record_ids[row]) for row in train_rows]
        train_record_sha256, _ = hash_string_array(train_record_ids)
        if train_record_sha256 != next(
            str(partition["record_ids_sha256"])
            for partition in partition_payloads
            if partition["partition_role"] == "train"
        ):
            raise ValueError("TRAIN count access does not match frozen TRAIN record membership")
        statistics = compute_train_feature_statistics(
            data=matrix["data"],
            indices=matrix["indices"],
            indptr=indptr,
            train_rows=train_rows,
            n_features=matrix_shape[1],
        )
        accessed_row_sha256, accessed_row_bytes = hash_string_array(
            str(row) for row in statistics.accessed_rows
        )
        if accessed_row_sha256 != hash_string_array(str(row) for row in train_rows)[0]:
            raise ValueError("count reader accessed a row set other than exact TRAIN")
        panel, eligibility_summary = select_train_feature_panel(
            statistics=statistics,
            ensembl_ids=ensembl_ids,
            gene_symbols=gene_symbols,
            train_cell_count=len(train_rows),
        )
        panel_keys = [f"{item['ensembl_id']}|{item['gene_symbol']}" for item in panel]
        panel_sha256, panel_encoded_bytes = hash_string_array(panel_keys, sort_values=False)
        feature_axis_entries = [
            [index, ensembl_id, symbol]
            for index, (ensembl_id, symbol) in enumerate(
                zip(source_ensembl_ids, gene_symbols, strict=True)
            )
        ]
        feature_axis_bytes = canonical_json_bytes(feature_axis_entries)
        feature_axis_sha256 = hashlib.sha256(feature_axis_bytes).hexdigest()

        source_payload = {
            "artifact_schema": "sciplex3-k562-source-verification",
            "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
            "generator": {
                "generator_id": GENERATOR_ID,
                "generator_version": GENERATOR_VERSION,
                "script_sha256": script_sha256,
            },
            "source": {
                "filename": SOURCE_FILENAME,
                "uri": SOURCE_URI,
                "accession": SOURCE_ACCESSION,
                "release": SOURCE_RELEASE,
                "license": SOURCE_LICENSE,
                "byte_count": source_byte_count,
                "md5": source_md5,
                "sha256": source_sha256,
            },
            "h5ad_structure": {
                "matrix_encoding": "csr_matrix",
                "matrix_shape": list(matrix_shape),
                "matrix_nonzero_count": EXPECTED_H5AD_NNZ,
                "matrix_value_dtype": str(matrix["data"].dtype),
                "matrix_values_are_raw_integer_umi_counts": True,
                "required_observation_fields": sorted(required_obs_columns),
                "source_feature_count": matrix_shape[1],
                "source_features_with_ensembl_id": sum(bool(value) for value in ensembl_ids),
                "source_features_with_human_ensembl_id": sum(
                    value.startswith("ENSG") for value in ensembl_ids
                ),
                "source_features_with_mouse_ensembl_id": sum(
                    value.startswith("ENSMUSG") for value in ensembl_ids
                ),
                "source_feature_axis_encoding": (
                    "canonical_json_utf8_index_ensembl_symbol_triple_array_v1"
                ),
                "source_feature_axis_sha256": feature_axis_sha256,
                "source_feature_axis_encoded_byte_count": len(feature_axis_bytes),
            },
            "limitations": [
                "Source verification does not authorize training or benchmark use.",
                "The H5AD is a corrected secondary harmonization of GEO GSE139944.",
                "Dose units and randomized-design evidence require primary-source binding.",
            ],
        }
        universe_payload = {
            "artifact_schema": "sciplex3-k562-record-universe",
            "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
            "source_sha256": source_sha256,
            "selector": {"cell_line": "K562"},
            "record_id_field": "_index",
            "record_count": len(k562_record_ids),
            "record_ids_encoding": "canonical_json_utf8_string_array_v1",
            "record_ids_sha256": universe_record_sha256,
            "record_ids_encoded_byte_count": universe_record_bytes,
            "source_row_indices_encoding": "canonical_json_utf8_decimal_string_array_v1",
            "source_row_indices_sha256": source_row_sha256,
            "source_row_indices_encoded_byte_count": source_row_bytes,
            "population_unit": "independently_treated_well",
            "composite_well_identity": {
                "kind": "COMPOSITE_SOURCE_FIELDS",
                "source_fields": ["plate", "well"],
                "encoding": "canonical_json_utf8_string_array_v1",
                "definition": ("compact JSON string array of the obs field values [plate, well]"),
            },
            "composite_well_count": len(records_by_well),
            "composite_well_ids_encoding": "canonical_json_utf8_string_array_v1",
            "composite_well_ids_sha256": universe_well_sha256,
            "composite_well_ids_encoded_byte_count": universe_well_bytes,
            "record_to_well_encoding": "canonical_json_utf8_string_pair_array_v1",
            "record_to_well_sha256": descendant_sha256,
            "record_to_well_encoded_byte_count": descendant_bytes,
            "plate_ids_encoding": "canonical_json_utf8_string_array_v1",
            "plate_ids_sha256": universe_plate_sha256,
            "plate_ids_encoded_byte_count": universe_plate_bytes,
            "well_to_condition_encoding": "canonical_json_utf8_string_pair_array_v1",
            "well_to_condition_sha256": universe_condition_sha256,
            "well_to_condition_encoded_byte_count": universe_condition_bytes,
            "condition_group_count": len(set(dict(universe_condition_pairs).values())),
            "treated_condition_group_count": len(treated_condition_ids),
            "treated_wells_per_condition": 2,
            "treated_condition_replicate_values": ["rep1", "rep2"],
            "membership_artifacts": {
                "record_ids": _membership_artifact_reference(
                    relative_path=universe_membership_paths["record-ids"],
                    sha256=universe_record_sha256,
                    byte_count=universe_record_bytes,
                    encoding="canonical_json_utf8_string_array_v1",
                ),
                "well_ids": _membership_artifact_reference(
                    relative_path=universe_membership_paths["well-ids"],
                    sha256=universe_well_sha256,
                    byte_count=universe_well_bytes,
                    encoding="canonical_json_utf8_string_array_v1",
                ),
                "plate_ids": _membership_artifact_reference(
                    relative_path=universe_membership_paths["plate-ids"],
                    sha256=universe_plate_sha256,
                    byte_count=universe_plate_bytes,
                    encoding="canonical_json_utf8_string_array_v1",
                ),
                "record_to_well": _membership_artifact_reference(
                    relative_path=universe_membership_paths["record-to-well"],
                    sha256=descendant_sha256,
                    byte_count=descendant_bytes,
                    encoding="canonical_json_utf8_string_pair_array_v1",
                ),
                "well_to_condition": _membership_artifact_reference(
                    relative_path=universe_membership_paths["well-to-condition"],
                    sha256=universe_condition_sha256,
                    byte_count=universe_condition_bytes,
                    encoding="canonical_json_utf8_string_pair_array_v1",
                ),
            },
            "replicate_values": ["rep1", "rep2"],
            "endpoint_hours": 24,
            "dose_unit": "nM",
            "dose_values_nm": list(EXPECTED_DOSES_NM),
            "treated_well_count": treated_well_count,
            "control_well_count": control_well_count,
            "source_perturbation_count": EXPECTED_COMPOUNDS,
            "semantic_boundary": {
                "cells_are_destructive_samples": True,
                "cells_are_independent_biological_replicates": False,
                "wells_are_intervention_randomization_population_and_metric_units": True,
                "plates_are_split_assignment_and_protected_parent_units": True,
            },
        }
        partitions_payload = {
            "artifact_schema": "sciplex3-k562-frozen-partitions",
            "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
            "source_sha256": source_sha256,
            "universe_record_ids_sha256": universe_record_sha256,
            "assignment_unit": "plate",
            "record_unit": "_index",
            "protected_parent_unit": "plate",
            "metric_evaluation_unit": {
                "level": "WELL",
                "identity_kind": "COMPOSITE_SOURCE_FIELDS",
                "source_fields": ["plate", "well"],
                "encoding": "canonical_json_utf8_string_array_v1",
            },
            "partition_rule_uses_only_preoutcome_design_metadata": True,
            "partitions": partition_payloads,
            "cross_partition_checks": {
                "protected_plate_overlap_count": 0,
                "record_overlap_count": 0,
                "well_overlap_count": 0,
                "rep2_compound_overlap_count": 0,
                "rep2_compound_union_count": len(heldout_union),
                "all_evaluation_conditions_have_rep1_train_counterparts": True,
            },
            "scientific_scope": {
                "tests_independent_well_replicate_transfer": True,
                "tests_unseen_compounds": False,
                "tests_external_study_transport": False,
                "tests_pretreatment_molecular_state": False,
            },
        }
        well_groups_payload = {
            "artifact_schema": "sciplex3-k562-well-descendant-groups",
            "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
            "source_sha256": source_sha256,
            "group_count": len(well_groups),
            "record_to_well_sha256": descendant_sha256,
            "record_to_well_artifact": _membership_artifact_reference(
                relative_path=universe_membership_paths["record-to-well"],
                sha256=descendant_sha256,
                byte_count=descendant_bytes,
                encoding="canonical_json_utf8_string_pair_array_v1",
            ),
            "well_to_condition_artifact": _membership_artifact_reference(
                relative_path=universe_membership_paths["well-to-condition"],
                sha256=universe_condition_sha256,
                byte_count=universe_condition_bytes,
                encoding="canonical_json_utf8_string_pair_array_v1",
            ),
            "groups": well_groups,
        }
        intervention_labels_payload = {
            "artifact_schema": "sciplex3-k562-source-label-normalization",
            "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
            "source_sha256": source_sha256,
            "source_field": "obs.perturbation",
            "source_label_count_including_control": len(source_label_pairs),
            "normalized_label_count_including_control": len(
                {normalized for _, normalized in source_label_pairs}
            ),
            "labels_changed_by_normalization": sum(
                source != normalized for source, normalized in source_label_pairs
            ),
            "normalization": {
                "operation": "Python str.strip() with default Unicode whitespace",
                "case_folding": False,
                "fuzzy_matching": False,
                "chemical_name_resolution": False,
                "ontology_mapping": False,
            },
            "mapping_artifact": _membership_artifact_reference(
                relative_path=source_label_mapping_path,
                sha256=source_label_mapping_sha256,
                byte_count=source_label_mapping_bytes,
                encoding="canonical_json_utf8_string_pair_array_v1",
            ),
            "condition_identity": (
                "source-label:<normalized source label>@<integer dose>nM; "
                "source-control@0nM for vehicle"
            ),
            "limitations": [
                "Normalized labels remain source-scoped presentation strings.",
                (
                    "This artifact does not establish chemical identity, mechanism, "
                    "or ontology equivalence."
                ),
                (
                    "Cross-study intervention transport requires a separately reviewed "
                    "mapping artifact."
                ),
            ],
        }
        feature_payload = {
            "artifact_schema": "sciplex3-k562-train-feature-panel",
            "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
            "source_sha256": source_sha256,
            "feature_selection": {
                "selection_id": FEATURE_SELECTION_ID,
                "selection_version": FEATURE_SELECTION_VERSION,
                "selection_partition_role": "train",
                "train_record_count": len(train_rows),
                "train_record_ids_sha256": train_record_sha256,
                "count_accessed_partition_roles": ["train"],
                "heldout_count_rows_accessed": 0,
                "accessed_source_row_count": statistics.accessed_library_count,
                "accessed_source_row_indices_encoding": (
                    "canonical_json_utf8_decimal_string_array_v1"
                ),
                "accessed_source_row_indices_sha256": accessed_row_sha256,
                "accessed_source_row_indices_encoded_byte_count": accessed_row_bytes,
                "accessed_raw_umi_count_total": statistics.accessed_count_total,
                "minimum_detection_fraction": FEATURE_DETECTION_FRACTION,
                "mean_bin_count": FEATURE_MEAN_BIN_COUNT,
                "score": (
                    "robust z-score of logCP10k variance/mean within deterministic "
                    "equal-frequency train-mean bins"
                ),
                "technical_symbol_exclusion_regex": TECHNICAL_GENE_PATTERN.pattern,
                "ensembl_species_policy": (
                    "retain human ENSG features only for the human K562 target"
                ),
                "duplicate_ensembl_policy": "exclude all non-unique source Ensembl IDs",
                "tie_break_order": [
                    "descending robust score",
                    "descending train logCP10k variance",
                    "descending train detection count",
                    "ascending Ensembl ID",
                    "ascending gene symbol",
                    "ascending source feature index",
                ],
                "eligibility_summary": eligibility_summary,
            },
            "transformation": {
                "transformation_id": "natural-log-cp10k",
                "transformation_version": "1.0.0",
                "input": "raw integer UMI count vector for one source record",
                "library_size": "sum of all source-axis UMI counts for that record",
                "formula": "log1p(10000 * count / library_size)",
                "log_base": "e",
                "scale": 10_000,
                "zero_library_policy": "error",
                "fit_statistics": "none",
                "preparation_or_fit_requires_heldout_counts": False,
                "evaluation_requires_counts_for_each_evaluated_record": True,
            },
            "ordered_feature_key_definition": "Ensembl ID + '|' + source gene symbol",
            "ordered_feature_keys_encoding": "canonical_json_utf8_string_array_v1_ordered",
            "ordered_feature_keys_sha256": panel_sha256,
            "ordered_feature_keys_encoded_byte_count": panel_encoded_bytes,
            "feature_count": len(panel),
            "features": panel,
            "limitations": [
                "The panel is a benchmark measurement target, not a universal cellular state.",
                "TRAIN-only variability selection cannot validate pre-treatment state sufficiency.",
                "The derived logCP10k scale is an assay summary, not latent molecular abundance.",
            ],
        }

    output_root.mkdir(parents=True, exist_ok=True)
    payloads = {
        "source-verification.json": source_payload,
        "k562-universe.json": universe_payload,
        "partitions.json": partitions_payload,
        "well-groups.json": well_groups_payload,
        "intervention-labels.json": intervention_labels_payload,
        "feature-panel.json": feature_payload,
    }
    for filename, payload in payloads.items():
        _write_json(output_root / filename, payload)
    if set(membership_payloads) != set(MEMBERSHIP_FILENAMES):
        raise ValueError("materialized membership payload set differs from the frozen file set")
    for filename, payload in membership_payloads.items():
        _write_canonical_json(output_root / filename, payload)
    if set(mapping_payloads) != set(MAPPING_FILENAMES):
        raise ValueError("materialized mapping payload set differs from the frozen file set")
    for filename, payload in mapping_payloads.items():
        _write_canonical_json(output_root / filename, payload)

    artifact_records = [
        _artifact_record(output_root / filename, output_root) for filename in OUTPUT_FILENAMES
    ]
    index_payload = {
        "artifact_schema": "sciplex3-k562-preparation-index",
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "generator_id": GENERATOR_ID,
        "generator_version": GENERATOR_VERSION,
        "generator_script_sha256": script_sha256,
        "source_sha256": source_sha256,
        "artifacts": artifact_records,
        "contains_source_matrix": False,
        "contains_normalized_matrix": False,
        "contains_materialized_membership_arrays": True,
        "membership_artifacts_are_canonical_json_bytes": True,
        "admission_decision": "not_made",
    }
    _write_json(output_root / "artifact-index.json", index_payload)
    return index_payload


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help=f"Exact corrected scPerturb {SOURCE_RELEASE} {SOURCE_FILENAME}",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("benchmarks/artifacts/sciplex3-k562-24h-v1"),
        help="Directory for small content-addressed JSON artifacts",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    result = prepare(args.source.resolve(), args.output_root.resolve())
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
