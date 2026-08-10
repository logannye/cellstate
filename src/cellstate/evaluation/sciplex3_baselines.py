"""Frozen-scope probabilistic baselines for the sci-Plex3 K562 endpoint.

These algorithms predict raw recovered-nucleus counts on the exact ordered 2,000-feature
panel.  They are deliberately not hidden-state estimators and never construct a
``CellStateBelief``.  Every fitted parameter is derived from :class:`P1TrainingData`.

The matched-vehicle baseline identifies the exact treated p1 counterpart and resamples only
the p1 vehicle wells on that counterpart's source plate.  Prediction requests contain design
metadata but no outcome rows, so p2, p3, or p4 observations cannot enter any baseline.

The fixed statistical semantics are:

* empirical baselines choose a well uniformly, then a nucleus uniformly within that well;
* negative-binomial models use a Gamma--Poisson parameterization with
  ``variance = mean + dispersion * mean**2``;
* exact-condition means receive one half of an equal-well p1 global pseudo-well and their
  dispersions receive two global pseudo-wells;
* hierarchical means and dispersions combine one exact-condition well-unit, two
  compound-level well-units, and one global well-unit;
* the low-rank model applies a rank-eight (or explicitly smaller) truncated SVD to the
  matrix of log1p condition effects relative to equal-weight, same-plate p1 vehicles, and
  samples with the equal-well global dispersion; and
* nearest dose minimizes absolute log10-dose distance, breaking ties toward the lower dose.

No statement in this module implies that a baseline has passed the real frozen benchmark.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import ClassVar, Literal, Protocol, cast, runtime_checkable

import numpy as np
from numpy.typing import NDArray

SCIPLEX3_FEATURE_COUNT = 2_000
SCIPLEX3_BASELINE_IMPLEMENTATION_VERSION = "1.0.0"
SCIPLEX3_ORDERED_FEATURE_KEYS_SHA256 = (
    "8b9e0e71d9bfc79a5e1db29f73bc30006bf117a29db8abf63b1607403629401f"
)
EXACT_MEAN_GLOBAL_PSEUDO_WELLS = 0.5
EXACT_DISPERSION_GLOBAL_PSEUDO_WELLS = 2.0
HIERARCHICAL_CONDITION_WELLS = 1.0
HIERARCHICAL_COMPOUND_PSEUDO_WELLS = 2.0
HIERARCHICAL_GLOBAL_PSEUDO_WELLS = 1.0
DEFAULT_LOW_RANK = 8
LOW_RANK_MEAN_DECIMALS = 12
LOW_RANK_SVD_TIE_RTOL = 1e-12
LOW_RANK_NUMERIC_CONTRACT = (
    "full-matrices-false SVD; max-absolute-loading positive sign; fail on retained-boundary "
    "singular-value ties at rtol=1e-12; round raw predictive means to 12 decimals"
)
RNG_ALGORITHM: Literal["numpy-pcg64dxsm-v1"] = "numpy-pcg64dxsm-v1"
MAX_ZERO_TOTAL_REDRAWS = 32
SCIPLEX3_BASELINE_SAMPLES_PER_CASE_PER_SEED = 512
SCIPLEX3_BASELINE_SEEDS = (0, 1, 2, 3, 4)
NEAREST_DOSE_TIE_ATOL = 1e-12

IntArray = NDArray[np.int64]
FloatArray = NDArray[np.float64]


class SciPlex3BaselineError(ValueError):
    """Raised when baseline data or a prediction request violates the frozen contract."""


def _canonical_text(value: object, *, name: str) -> str:
    if type(value) is not str:
        raise SciPlex3BaselineError(f"{name} must be a string")
    if not value or value != value.strip():
        raise SciPlex3BaselineError(f"{name} must be nonblank and trimmed")
    return value


def _immutable_int_matrix(value: object, *, name: str) -> IntArray:
    raw = np.asarray(value)
    if raw.ndim != 2:
        raise SciPlex3BaselineError(f"{name} must be a two-dimensional count matrix")
    if raw.shape[0] == 0 or raw.shape[1] != SCIPLEX3_FEATURE_COUNT:
        raise SciPlex3BaselineError(
            f"{name} must contain at least one row and exactly {SCIPLEX3_FEATURE_COUNT} features"
        )
    if raw.dtype.kind not in {"i", "u"}:
        raise SciPlex3BaselineError(f"{name} must contain integer counts (booleans are invalid)")
    if raw.dtype.kind == "u" and bool(np.any(raw > np.iinfo(np.int64).max)):
        raise SciPlex3BaselineError(f"{name} contains a count outside signed 64-bit range")
    converted = np.asarray(raw, dtype=np.int64, order="C")
    if bool(np.any(converted < 0)):
        raise SciPlex3BaselineError(f"{name} contains a negative count")
    # A bytes-backed view cannot be made writeable again, unlike an owning ndarray whose flag
    # could be reset by a caller.  This gives frozen dataclasses genuinely immutable payloads.
    frozen = np.frombuffer(converted.tobytes(order="C"), dtype=np.int64).reshape(converted.shape)
    return frozen


def _immutable_float_vector(value: object, *, name: str) -> FloatArray:
    raw = np.asarray(value, dtype=np.float64)
    if raw.shape != (SCIPLEX3_FEATURE_COUNT,):
        raise SciPlex3BaselineError(
            f"{name} must contain exactly {SCIPLEX3_FEATURE_COUNT} feature values"
        )
    if not bool(np.all(np.isfinite(raw))) or bool(np.any(raw < 0.0)):
        raise SciPlex3BaselineError(f"{name} must contain finite nonnegative values")
    return np.frombuffer(raw.tobytes(order="C"), dtype=np.float64)


def _immutable_int_vector(value: object, *, name: str) -> IntArray:
    raw = np.asarray(value)
    if raw.ndim != 1 or raw.dtype.kind not in {"i", "u"}:
        raise SciPlex3BaselineError(f"{name} must be a one-dimensional integer array")
    if raw.dtype.kind == "u" and bool(np.any(raw > np.iinfo(np.int64).max)):
        raise SciPlex3BaselineError(f"{name} contains a value outside signed 64-bit range")
    converted = np.asarray(raw, dtype=np.int64, order="C")
    return np.frombuffer(converted.tobytes(order="C"), dtype=np.int64)


def _validate_feature_keys(keys: tuple[str, ...]) -> tuple[str, ...]:
    if len(keys) != SCIPLEX3_FEATURE_COUNT:
        raise SciPlex3BaselineError(
            f"ordered feature panel must contain exactly {SCIPLEX3_FEATURE_COUNT} keys"
        )
    for key in keys:
        _canonical_text(key, name="ordered feature key")
    if len(set(keys)) != len(keys):
        raise SciPlex3BaselineError("ordered feature keys must be unique")
    return keys


@dataclass(frozen=True, slots=True, order=True)
class CompoundDose:
    """One treated compound-dose condition; vehicles are represented separately."""

    compound: str
    dose_nm: int

    def __post_init__(self) -> None:
        _canonical_text(self.compound, name="compound")
        if type(self.dose_nm) is not int:
            raise SciPlex3BaselineError("dose_nm must be an integer number of nanomolar")
        if self.dose_nm <= 0:
            raise SciPlex3BaselineError("dose_nm must be positive for a treated condition")


@dataclass(frozen=True, slots=True)
class NoAction:
    """The assigned vehicle/no-active-compound condition (zero modeled active dose)."""


NO_ACTION = NoAction()
TargetCondition = CompoundDose | NoAction


def _canonical_target_condition(value: object, *, name: str) -> TargetCondition:
    if type(value) is CompoundDose:
        return value
    if type(value) is NoAction:
        return NO_ACTION
    raise SciPlex3BaselineError(f"{name} must be an exact immutable CompoundDose or NoAction")


def _canonical_json_sha256(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _array_manifest(value: NDArray[np.generic]) -> dict[str, object]:
    if value.dtype.kind in {"i", "u"}:
        canonical = np.asarray(value, dtype="<i8", order="C")
        dtype = "little-endian-int64"
    elif value.dtype.kind == "f":
        canonical = np.asarray(value, dtype="<f8", order="C")
        dtype = "little-endian-float64"
    else:
        raise SciPlex3BaselineError("fitted-state array has an unsupported dtype")
    return {
        "dtype": dtype,
        "shape": list(canonical.shape),
        "sha256": hashlib.sha256(canonical.tobytes(order="C")).hexdigest(),
    }


def _condition_manifest(condition: TargetCondition) -> dict[str, object]:
    condition = _canonical_target_condition(condition, name="fitted-state condition")
    if type(condition) is NoAction:
        return {"kind": "no_action", "active_dose_nm": 0}
    compound_dose = cast(CompoundDose, condition)
    return {
        "kind": "compound_dose",
        "compound": compound_dose.compound,
        "dose_nm": compound_dose.dose_nm,
    }


def _condition_sort_key(condition: TargetCondition) -> tuple[str, int]:
    condition = _canonical_target_condition(condition, name="fitted-state condition")
    if type(condition) is NoAction:
        return ("", 0)
    compound_dose = cast(CompoundDose, condition)
    return (compound_dose.compound, compound_dose.dose_nm)


def _well_state_manifest(well: P1WellCounts) -> dict[str, object]:
    counts = well.counts
    return {
        "well_id": well.well_id,
        "plate_id": well.plate_id,
        "condition": _condition_manifest(well.condition or NO_ACTION),
        "partition_id": well.partition_id,
        "replicate": well.replicate,
        "record_count": counts.row_count,
        "record_ids_sha256": _canonical_json_sha256(list(well.record_ids)),
        "source_row_indices": _array_manifest(np.asarray(well.source_row_indices, dtype=np.int64)),
        "csr": {
            "feature_count": SCIPLEX3_FEATURE_COUNT,
            "nnz": counts.nnz,
            "indptr": _array_manifest(counts.indptr),
            "feature_indices": _array_manifest(counts.feature_indices),
            "values": _array_manifest(counts.values),
        },
    }


def _pool_state_manifest(
    pools: Mapping[TargetCondition, tuple[P1WellCounts, ...]],
) -> list[dict[str, object]]:
    return [
        {
            "condition": _condition_manifest(condition),
            "wells": [_well_state_manifest(well) for well in pools[condition]],
        }
        for condition in sorted(pools, key=_condition_sort_key)
    ]


def _parameter_state_manifest(
    parameters: Mapping[TargetCondition, FloatArray],
) -> list[dict[str, object]]:
    return [
        {
            "condition": _condition_manifest(condition),
            "values": _array_manifest(parameters[condition]),
        }
        for condition in sorted(parameters, key=_condition_sort_key)
    ]


def _base_fitted_manifest(
    *,
    baseline_id: str,
    ordered_feature_keys: tuple[str, ...],
    semantics: dict[str, object],
    fitted_state: dict[str, object],
) -> dict[str, object]:
    return {
        "manifest_schema": "sciplex3-baseline-fitted-state-v1",
        "implementation_version": SCIPLEX3_BASELINE_IMPLEMENTATION_VERSION,
        "baseline_id": baseline_id,
        "feature_count": SCIPLEX3_FEATURE_COUNT,
        "ordered_feature_keys_sha256": _canonical_json_sha256(list(ordered_feature_keys)),
        "expected_frozen_ordered_feature_keys_sha256": SCIPLEX3_ORDERED_FEATURE_KEYS_SHA256,
        "rng_algorithm": RNG_ALGORITHM,
        "samples_per_case_per_seed": SCIPLEX3_BASELINE_SAMPLES_PER_CASE_PER_SEED,
        "seeds": list(SCIPLEX3_BASELINE_SEEDS),
        "semantics": semantics,
        "fitted_state": fitted_state,
    }


@dataclass(frozen=True, slots=True, eq=False)
class ImmutableCSRCounts:
    """A bytes-backed CSR matrix whose columns are ordered panel positions 0..1999.

    This is the bounded adapter for the loader's immutable CSR batches.  A trusted runner may
    split one batch into well-local CSR slices without ever materializing the full p1 dense
    matrix.  Baselines densify at most one 2,000-value row or one well-level summary at a time.
    """

    indptr: IntArray
    feature_indices: IntArray
    values: IntArray
    row_count: int

    def __post_init__(self) -> None:
        if type(self.row_count) is not int:
            raise SciPlex3BaselineError("CSR row_count must be an integer")
        if self.row_count <= 0:
            raise SciPlex3BaselineError("CSR row_count must be positive")
        indptr = _immutable_int_vector(self.indptr, name="CSR indptr")
        indices = _immutable_int_vector(self.feature_indices, name="CSR feature_indices")
        values = _immutable_int_vector(self.values, name="CSR values")
        if indptr.shape != (self.row_count + 1,):
            raise SciPlex3BaselineError("CSR indptr length must equal row_count + 1")
        if indptr[0] != 0 or bool(np.any(np.diff(indptr) < 0)):
            raise SciPlex3BaselineError("CSR indptr must start at zero and be nondecreasing")
        if indptr[-1] != len(indices) or len(indices) != len(values):
            raise SciPlex3BaselineError("CSR indptr, indices, and values lengths disagree")
        if bool(np.any(indices < 0)) or bool(np.any(indices >= SCIPLEX3_FEATURE_COUNT)):
            raise SciPlex3BaselineError("CSR feature index is outside the ordered panel")
        if bool(np.any(values <= 0)):
            raise SciPlex3BaselineError("CSR stored counts must be strictly positive")
        for row_index in range(self.row_count):
            start, stop = int(indptr[row_index]), int(indptr[row_index + 1])
            row_indices = indices[start:stop]
            if bool(np.any(np.diff(row_indices) <= 0)):
                raise SciPlex3BaselineError(
                    "CSR feature indices must be strictly increasing within every row"
                )
        object.__setattr__(self, "indptr", indptr)
        object.__setattr__(self, "feature_indices", indices)
        object.__setattr__(self, "values", values)

    @classmethod
    def from_dense(cls, counts: object) -> ImmutableCSRCounts:
        """Create the same immutable representation from a small dense fixture or adapter."""

        dense = _immutable_int_matrix(counts, name="dense raw counts")
        row_indices, feature_indices = np.nonzero(dense)
        row_sizes = np.bincount(row_indices, minlength=dense.shape[0])
        indptr = np.concatenate((np.asarray([0], dtype=np.int64), np.cumsum(row_sizes)))
        values = dense[row_indices, feature_indices]
        return cls(indptr, feature_indices, values, dense.shape[0])

    @property
    def nnz(self) -> int:
        return len(self.values)

    def row_dense(self, row_index: int) -> IntArray:
        if not 0 <= row_index < self.row_count:
            raise SciPlex3BaselineError("CSR row index is out of range")
        output = np.zeros(SCIPLEX3_FEATURE_COUNT, dtype=np.int64)
        start, stop = int(self.indptr[row_index]), int(self.indptr[row_index + 1])
        output[self.feature_indices[start:stop]] = self.values[start:stop]
        return output

    def feature_mean(self) -> FloatArray:
        sums = np.zeros(SCIPLEX3_FEATURE_COUNT, dtype=np.float64)
        np.add.at(sums, self.feature_indices, self.values.astype(np.float64))
        return sums / float(self.row_count)

    def feature_dispersion(self) -> FloatArray:
        if self.row_count == 1:
            return np.zeros(SCIPLEX3_FEATURE_COUNT, dtype=np.float64)
        sums = np.zeros(SCIPLEX3_FEATURE_COUNT, dtype=np.float64)
        squared_sums = np.zeros(SCIPLEX3_FEATURE_COUNT, dtype=np.float64)
        float_values = self.values.astype(np.float64)
        np.add.at(sums, self.feature_indices, float_values)
        np.add.at(squared_sums, self.feature_indices, np.square(float_values))
        mean = sums / float(self.row_count)
        variance = np.maximum(
            (squared_sums - float(self.row_count) * np.square(mean)) / float(self.row_count - 1),
            0.0,
        )
        dispersion = np.zeros(SCIPLEX3_FEATURE_COUNT, dtype=np.float64)
        nonzero = mean > 0.0
        dispersion[nonzero] = np.maximum(
            (variance[nonzero] - mean[nonzero]) / np.square(mean[nonzero]), 0.0
        )
        return dispersion


@dataclass(frozen=True, slots=True, eq=False)
class P1WellCounts:
    """Immutable raw panel rows for one independent p1 replicate-1 well.

    ``condition=None`` denotes a vehicle-control well.  There is intentionally no constructor
    for another fitting partition or replicate.
    """

    well_id: str
    plate_id: str
    condition: CompoundDose | None
    counts: ImmutableCSRCounts
    record_ids: tuple[str, ...]
    source_row_indices: tuple[int, ...]
    partition_id: Literal["p1-train"] = "p1-train"
    replicate: Literal["rep1"] = "rep1"

    def __post_init__(self) -> None:
        _canonical_text(self.well_id, name="well_id")
        _canonical_text(self.plate_id, name="plate_id")
        _canonical_text(self.partition_id, name="partition_id")
        _canonical_text(self.replicate, name="replicate")
        if self.partition_id != "p1-train":
            raise SciPlex3BaselineError("baseline fitting accepts only the p1-train partition")
        if self.replicate != "rep1":
            raise SciPlex3BaselineError("baseline fitting accepts only replicate rep1")
        if self.condition is not None and type(self.condition) is not CompoundDose:
            raise SciPlex3BaselineError(
                "p1 well condition must be an exact immutable CompoundDose or None"
            )
        if type(self.counts) is not ImmutableCSRCounts:
            raise SciPlex3BaselineError(
                "p1 well counts must use the immutable CSR loader-compatible representation"
            )
        object.__setattr__(self, "record_ids", tuple(self.record_ids))
        object.__setattr__(self, "source_row_indices", tuple(self.source_row_indices))
        if len(self.record_ids) != self.counts.row_count:
            raise SciPlex3BaselineError("p1 record ID count must equal the CSR row count")
        for record_id in self.record_ids:
            _canonical_text(record_id, name="p1 record ID")
        if len(set(self.record_ids)) != len(self.record_ids):
            raise SciPlex3BaselineError("p1 record IDs must be unique within a well")
        if self.record_ids != tuple(sorted(self.record_ids)):
            raise SciPlex3BaselineError("p1 record rows must be in canonical record-ID order")
        if len(self.source_row_indices) != self.counts.row_count:
            raise SciPlex3BaselineError("p1 source-row count must equal the CSR row count")
        if any(type(value) is not int or value < 0 for value in self.source_row_indices) or len(
            set(self.source_row_indices)
        ) != len(self.source_row_indices):
            raise SciPlex3BaselineError("p1 source-row indices must be unique nonnegative integers")

    @property
    def is_vehicle(self) -> bool:
        return self.condition is None


@dataclass(frozen=True, slots=True)
class P1TrainingData:
    """Closed p1-only fitting surface shared by all fitted baselines."""

    ordered_feature_keys: tuple[str, ...]
    wells: tuple[P1WellCounts, ...]

    def __post_init__(self) -> None:
        ordered_feature_keys = tuple(self.ordered_feature_keys)
        wells = tuple(self.wells)
        if any(type(well) is not P1WellCounts for well in wells):
            raise SciPlex3BaselineError("p1 training wells must be exact P1WellCounts instances")
        object.__setattr__(self, "ordered_feature_keys", ordered_feature_keys)
        object.__setattr__(self, "wells", tuple(sorted(wells, key=lambda item: item.well_id)))
        _validate_feature_keys(self.ordered_feature_keys)
        if not self.wells:
            raise SciPlex3BaselineError("p1 training data must contain wells")
        well_ids = tuple(well.well_id for well in self.wells)
        if len(set(well_ids)) != len(well_ids):
            raise SciPlex3BaselineError("p1 training well IDs must be unique")
        record_ids = tuple(record_id for well in self.wells for record_id in well.record_ids)
        if len(set(record_ids)) != len(record_ids):
            raise SciPlex3BaselineError("p1 training record IDs must be globally unique")
        source_rows = tuple(
            source_row for well in self.wells for source_row in well.source_row_indices
        )
        if len(set(source_rows)) != len(source_rows):
            raise SciPlex3BaselineError("p1 training source-row indices must be globally unique")
        treated = tuple(well for well in self.wells if not well.is_vehicle)
        vehicles = tuple(well for well in self.wells if well.is_vehicle)
        if not treated:
            raise SciPlex3BaselineError("p1 training data has no treated condition support")
        if not vehicles:
            raise SciPlex3BaselineError("p1 training data has no vehicle controls")
        vehicle_plates = {well.plate_id for well in vehicles}
        missing_control_plates = sorted({well.plate_id for well in treated} - vehicle_plates)
        if missing_control_plates:
            raise SciPlex3BaselineError(
                "p1 treated wells are missing same-plate vehicle controls: "
                + ", ".join(missing_control_plates)
            )


@dataclass(frozen=True, slots=True)
class PredictionTarget:
    """Outcome-free evaluation design metadata supplied to a baseline."""

    case_id: str
    target_well_id: str
    plate_id: str
    partition_id: str
    condition: TargetCondition

    def __post_init__(self) -> None:
        for name in ("case_id", "target_well_id", "plate_id", "partition_id"):
            _canonical_text(getattr(self, name), name=name)
        object.__setattr__(
            self,
            "condition",
            _canonical_target_condition(self.condition, name="prediction target condition"),
        )


@dataclass(frozen=True, slots=True)
class BaselineSampleRequest:
    """Seed-bound, outcome-free raw-count sampling request."""

    target: PredictionTarget
    sample_count: int
    seed: int

    def __post_init__(self) -> None:
        if type(self.target) is not PredictionTarget:
            raise SciPlex3BaselineError("sample request target must be an exact PredictionTarget")
        if type(self.sample_count) is not int:
            raise SciPlex3BaselineError("sample_count must be an integer")
        if self.sample_count <= 0:
            raise SciPlex3BaselineError("sample_count must be positive")
        if type(self.seed) is not int:
            raise SciPlex3BaselineError("seed must be an integer")
        if not 0 <= self.seed <= np.iinfo(np.uint64).max:
            raise SciPlex3BaselineError("seed must be in unsigned 64-bit range")


@dataclass(frozen=True, slots=True, eq=False)
class PredictiveRawCountSamples:
    """Immutable raw-count samples; this is intentionally not a belief-state object."""

    baseline_id: str
    target: PredictionTarget
    ordered_feature_keys: tuple[str, ...]
    seed: int
    samples: IntArray
    rng_algorithm: Literal["numpy-pcg64dxsm-v1"] = RNG_ALGORITHM

    def __post_init__(self) -> None:
        _canonical_text(self.baseline_id, name="baseline_id")
        if type(self.target) is not PredictionTarget:
            raise SciPlex3BaselineError("predictive target must be an exact PredictionTarget")
        ordered_feature_keys = tuple(self.ordered_feature_keys)
        object.__setattr__(self, "ordered_feature_keys", ordered_feature_keys)
        _validate_feature_keys(ordered_feature_keys)
        if type(self.seed) is not int:
            raise SciPlex3BaselineError("predictive seed must be an integer")
        if not 0 <= self.seed <= np.iinfo(np.uint64).max:
            raise SciPlex3BaselineError("predictive seed must be in unsigned 64-bit range")
        _canonical_text(self.rng_algorithm, name="rng_algorithm")
        if self.rng_algorithm != RNG_ALGORITHM:
            raise SciPlex3BaselineError(f"rng_algorithm must be {RNG_ALGORITHM!r}")
        samples = _immutable_int_matrix(self.samples, name="predictive raw-count samples")
        if bool(np.any(np.sum(samples, axis=1, dtype=np.int64) <= 0)):
            raise SciPlex3BaselineError("predictive raw-count samples contain a zero-total panel")
        object.__setattr__(
            self,
            "samples",
            samples,
        )


@runtime_checkable
class SciPlex3RawCountBaseline(Protocol):
    """Common narrow interface for raw-panel probabilistic baselines."""

    @property
    def baseline_id(self) -> str:
        """Frozen identifier used by the benchmark declaration."""

    @property
    def ordered_feature_keys(self) -> tuple[str, ...]:
        """Exact feature order emitted by every sample."""

    def sample(self, request: BaselineSampleRequest) -> PredictiveRawCountSamples:
        """Return deterministic seed-bound raw-count samples for one treated condition."""

    def fitted_state_manifest(self) -> dict[str, object]:
        """Return canonical-JSON-compatible fitted state and frozen semantics."""


def _wells_by_condition(
    training: P1TrainingData,
) -> Mapping[TargetCondition, tuple[P1WellCounts, ...]]:
    mutable: defaultdict[TargetCondition, list[P1WellCounts]] = defaultdict(list)
    for well in training.wells:
        mutable[well.condition if well.condition is not None else NO_ACTION].append(well)
    return MappingProxyType({key: tuple(value) for key, value in mutable.items()})


def _vehicles_by_plate(training: P1TrainingData) -> Mapping[str, tuple[P1WellCounts, ...]]:
    mutable: defaultdict[str, list[P1WellCounts]] = defaultdict(list)
    for well in training.wells:
        if well.is_vehicle:
            mutable[well.plate_id].append(well)
    return MappingProxyType({key: tuple(value) for key, value in mutable.items()})


def _freeze_ordered_feature_keys(value: object) -> tuple[str, ...]:
    try:
        keys = tuple(cast(Sequence[object], value))
    except TypeError as error:
        raise SciPlex3BaselineError(
            "ordered feature keys must be an iterable of strings"
        ) from error
    return _validate_feature_keys(cast(tuple[str, ...], keys))


def _freeze_condition_pools(
    value: Mapping[TargetCondition, tuple[P1WellCounts, ...]],
    *,
    name: str,
) -> Mapping[TargetCondition, tuple[P1WellCounts, ...]]:
    try:
        snapshot = dict(value)
    except (TypeError, ValueError) as error:
        raise SciPlex3BaselineError(f"{name} must be a condition-to-wells mapping") from error
    if not snapshot:
        raise SciPlex3BaselineError(f"{name} must contain at least one condition")
    mutable: dict[TargetCondition, tuple[P1WellCounts, ...]] = {}
    all_well_ids: set[str] = set()
    for raw_condition, raw_wells in snapshot.items():
        condition = _canonical_target_condition(raw_condition, name=f"{name} key")
        if condition in mutable:
            raise SciPlex3BaselineError(f"{name} contains duplicate canonical conditions")
        try:
            wells = tuple(raw_wells)
        except TypeError as error:
            raise SciPlex3BaselineError(f"{name} values must be iterables of wells") from error
        if not wells or any(type(well) is not P1WellCounts for well in wells):
            raise SciPlex3BaselineError(f"{name} values must contain exact P1WellCounts instances")
        wells = tuple(sorted(wells, key=lambda well: well.well_id))
        well_ids = {well.well_id for well in wells}
        if len(well_ids) != len(wells) or all_well_ids.intersection(well_ids):
            raise SciPlex3BaselineError(f"{name} contains duplicate wells")
        all_well_ids.update(well_ids)
        if type(condition) is NoAction:
            matches = all(well.condition is None for well in wells)
        else:
            matches = all(well.condition == condition for well in wells)
        if not matches:
            raise SciPlex3BaselineError(f"{name} contains a well under the wrong condition")
        mutable[condition] = wells
    ordered = {
        condition: mutable[condition] for condition in sorted(mutable, key=_condition_sort_key)
    }
    return MappingProxyType(ordered)


def _freeze_vehicle_pools(
    value: Mapping[str, tuple[P1WellCounts, ...]],
    *,
    name: str,
) -> Mapping[str, tuple[P1WellCounts, ...]]:
    try:
        snapshot = dict(value)
    except (TypeError, ValueError) as error:
        raise SciPlex3BaselineError(f"{name} must be a plate-to-wells mapping") from error
    if not snapshot:
        raise SciPlex3BaselineError(f"{name} must contain at least one plate")
    mutable: dict[str, tuple[P1WellCounts, ...]] = {}
    all_well_ids: set[str] = set()
    for raw_plate_id, raw_wells in snapshot.items():
        plate_id = _canonical_text(raw_plate_id, name=f"{name} plate ID")
        try:
            wells = tuple(raw_wells)
        except TypeError as error:
            raise SciPlex3BaselineError(f"{name} values must be iterables of wells") from error
        if not wells or any(type(well) is not P1WellCounts for well in wells):
            raise SciPlex3BaselineError(f"{name} values must contain exact P1WellCounts instances")
        wells = tuple(sorted(wells, key=lambda well: well.well_id))
        well_ids = {well.well_id for well in wells}
        if len(well_ids) != len(wells) or all_well_ids.intersection(well_ids):
            raise SciPlex3BaselineError(f"{name} contains duplicate wells")
        all_well_ids.update(well_ids)
        if any(not well.is_vehicle or well.plate_id != plate_id for well in wells):
            raise SciPlex3BaselineError(f"{name} contains a nonvehicle or wrong-plate well")
        mutable[plate_id] = wells
    return MappingProxyType({plate_id: mutable[plate_id] for plate_id in sorted(mutable)})


def _vehicles_from_condition_pools(
    pools: Mapping[TargetCondition, tuple[P1WellCounts, ...]],
) -> Mapping[str, tuple[P1WellCounts, ...]]:
    vehicles = pools.get(NO_ACTION)
    if vehicles is None:
        raise SciPlex3BaselineError("p1 condition pools omit no-action vehicle support")
    mutable: defaultdict[str, list[P1WellCounts]] = defaultdict(list)
    for well in vehicles:
        mutable[well.plate_id].append(well)
    return _freeze_vehicle_pools(
        {plate_id: tuple(wells) for plate_id, wells in mutable.items()},
        name="derived p1 vehicle pools",
    )


def _plate_pool_state_manifest(
    pools: Mapping[str, tuple[P1WellCounts, ...]],
) -> list[dict[str, object]]:
    return [
        {
            "plate_id": plate_id,
            "wells": [_well_state_manifest(well) for well in pools[plate_id]],
        }
        for plate_id in sorted(pools)
    ]


def _freeze_parameter_mapping(
    value: Mapping[TargetCondition, FloatArray],
    *,
    name: str,
) -> Mapping[TargetCondition, FloatArray]:
    try:
        snapshot = dict(value)
    except (TypeError, ValueError) as error:
        raise SciPlex3BaselineError(f"{name} must be a condition-to-vector mapping") from error
    if not snapshot:
        raise SciPlex3BaselineError(f"{name} must contain at least one condition")
    mutable: dict[TargetCondition, FloatArray] = {}
    for raw_condition, raw_values in snapshot.items():
        condition = _canonical_target_condition(raw_condition, name=f"{name} key")
        if condition in mutable:
            raise SciPlex3BaselineError(f"{name} contains duplicate canonical conditions")
        mutable[condition] = _immutable_float_vector(raw_values, name=f"{name} values")
    return MappingProxyType(
        {condition: mutable[condition] for condition in sorted(mutable, key=_condition_sort_key)}
    )


def _validate_sample_request(request: object) -> BaselineSampleRequest:
    if type(request) is not BaselineSampleRequest:
        raise SciPlex3BaselineError("baseline sampling requires an exact BaselineSampleRequest")
    return request


def _lookup_condition(
    by_condition: Mapping[TargetCondition, tuple[P1WellCounts, ...]], condition: TargetCondition
) -> tuple[P1WellCounts, ...]:
    wells = by_condition.get(condition)
    if wells is not None:
        return wells
    if isinstance(condition, NoAction):
        raise SciPlex3BaselineError("p1 no-action vehicle support is missing")
    supported_doses = sorted(
        key.dose_nm
        for key in by_condition
        if isinstance(key, CompoundDose) and key.compound == condition.compound
    )
    if not supported_doses:
        raise SciPlex3BaselineError(f"p1 compound support is missing for {condition.compound!r}")
    raise SciPlex3BaselineError(
        f"p1 dose support is missing for {condition.compound!r} at {condition.dose_nm} nM; "
        f"supported doses are {supported_doses}"
    )


def _equal_well_mean(wells: Sequence[P1WellCounts]) -> FloatArray:
    if not wells:
        raise SciPlex3BaselineError("at least one independent well is required")
    means = np.stack([well.counts.feature_mean() for well in wells])
    return np.asarray(np.mean(means, axis=0), dtype=np.float64)


def _well_dispersion(well: P1WellCounts) -> FloatArray:
    return well.counts.feature_dispersion()


def _equal_well_dispersion(wells: Sequence[P1WellCounts]) -> FloatArray:
    if not wells:
        raise SciPlex3BaselineError("at least one independent well is required")
    values = np.stack([_well_dispersion(well) for well in wells])
    return np.asarray(np.mean(values, axis=0), dtype=np.float64)


def _resample_equal_wells(
    wells: Sequence[P1WellCounts],
    *,
    sample_count: int,
    generator: np.random.Generator,
) -> IntArray:
    if not wells:
        raise SciPlex3BaselineError("resampling support contains no wells")
    well_draws = generator.integers(0, len(wells), size=sample_count)
    output = np.empty((sample_count, SCIPLEX3_FEATURE_COUNT), dtype=np.int64)
    for output_index, well_index in enumerate(well_draws):
        well = wells[int(well_index)]
        for _ in range(MAX_ZERO_TOTAL_REDRAWS + 1):
            row_index = int(generator.integers(0, well.counts.row_count))
            candidate = well.counts.row_dense(row_index)
            if int(np.sum(candidate, dtype=np.int64)) > 0:
                output[output_index] = candidate
                break
        else:
            raise SciPlex3BaselineError(
                "empirical sampler exhausted bounded same-well positive-panel redraws"
            )
    return output


def _generator(seed: int) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64DXSM(seed))


def _canonicalize_svd(left: FloatArray, right: FloatArray) -> tuple[FloatArray, FloatArray]:
    canonical_left = np.asarray(left, dtype=np.float64).copy()
    canonical_right = np.asarray(right, dtype=np.float64).copy()
    for component in range(canonical_right.shape[0]):
        anchor = int(np.argmax(np.abs(canonical_right[component])))
        if canonical_right[component, anchor] < 0.0:
            canonical_right[component] *= -1.0
            canonical_left[:, component] *= -1.0
    return canonical_left, canonical_right


def _sample_gamma_poisson(
    mean: FloatArray,
    dispersion: FloatArray,
    *,
    sample_count: int,
    generator: np.random.Generator,
) -> IntArray:
    if mean.shape != (SCIPLEX3_FEATURE_COUNT,) or dispersion.shape != mean.shape:
        raise SciPlex3BaselineError("negative-binomial parameters have the wrong panel shape")
    if (
        not bool(np.all(np.isfinite(mean)))
        or not bool(np.all(np.isfinite(dispersion)))
        or bool(np.any(mean < 0.0))
        or bool(np.any(dispersion < 0.0))
    ):
        raise SciPlex3BaselineError("negative-binomial parameters must be finite and nonnegative")
    samples = _draw_gamma_poisson(mean, dispersion, sample_count=sample_count, generator=generator)
    zero_total = np.sum(samples, axis=1, dtype=np.int64) <= 0
    for _ in range(MAX_ZERO_TOTAL_REDRAWS):
        redraw_count = int(np.sum(zero_total))
        if redraw_count == 0:
            return samples
        samples[zero_total] = _draw_gamma_poisson(
            mean, dispersion, sample_count=redraw_count, generator=generator
        )
        zero_total = np.sum(samples, axis=1, dtype=np.int64) <= 0
    if not bool(np.any(zero_total)):
        return samples
    raise SciPlex3BaselineError(
        "negative-binomial sampler exhausted bounded zero-total panel redraws"
    )


def _draw_gamma_poisson(
    mean: FloatArray,
    dispersion: FloatArray,
    *,
    sample_count: int,
    generator: np.random.Generator,
) -> IntArray:
    rates = np.broadcast_to(mean, (sample_count, SCIPLEX3_FEATURE_COUNT)).copy()
    overdispersed = dispersion > 0.0
    if bool(np.any(overdispersed)):
        alpha = dispersion[overdispersed]
        rates[:, overdispersed] = generator.gamma(
            shape=1.0 / alpha,
            scale=mean[overdispersed] * alpha,
            size=(sample_count, int(np.sum(overdispersed))),
        )
    return np.asarray(generator.poisson(rates), dtype=np.int64)


def _make_output(
    *,
    baseline_id: str,
    ordered_feature_keys: tuple[str, ...],
    request: BaselineSampleRequest,
    samples: IntArray,
) -> PredictiveRawCountSamples:
    if samples.shape[0] != request.sample_count:
        raise SciPlex3BaselineError("baseline returned the wrong number of predictive samples")
    return PredictiveRawCountSamples(
        baseline_id=baseline_id,
        target=request.target,
        ordered_feature_keys=ordered_feature_keys,
        seed=request.seed,
        samples=samples,
    )


@dataclass(frozen=True, slots=True)
class MatchedVehicleResampling:
    """Uniform-well resampling of same-plate p1 controls for the exact p1 condition."""

    baseline_id: ClassVar[str] = "matched-vehicle-resampling"
    ordered_feature_keys: tuple[str, ...]
    _by_condition: Mapping[TargetCondition, tuple[P1WellCounts, ...]] = field(repr=False)
    _vehicles_by_plate: Mapping[str, tuple[P1WellCounts, ...]] = field(repr=False)

    def __post_init__(self) -> None:
        ordered_feature_keys = _freeze_ordered_feature_keys(self.ordered_feature_keys)
        by_condition = _freeze_condition_pools(
            self._by_condition, name="matched-vehicle condition pools"
        )
        vehicles_by_plate = _freeze_vehicle_pools(
            self._vehicles_by_plate, name="matched-vehicle plate pools"
        )
        derived = _vehicles_from_condition_pools(by_condition)
        if _plate_pool_state_manifest(vehicles_by_plate) != _plate_pool_state_manifest(derived):
            raise SciPlex3BaselineError(
                "matched-vehicle plate pools differ from authenticated no-action support"
            )
        object.__setattr__(self, "ordered_feature_keys", ordered_feature_keys)
        object.__setattr__(self, "_by_condition", by_condition)
        object.__setattr__(self, "_vehicles_by_plate", vehicles_by_plate)

    @classmethod
    def fit(cls, training: P1TrainingData) -> MatchedVehicleResampling:
        """Freeze only p1 condition-to-plate and p1 vehicle observations."""

        return cls(
            ordered_feature_keys=training.ordered_feature_keys,
            _by_condition=_wells_by_condition(training),
            _vehicles_by_plate=_vehicles_by_plate(training),
        )

    def fitted_state_manifest(self) -> dict[str, object]:
        return _base_fitted_manifest(
            baseline_id=self.baseline_id,
            ordered_feature_keys=self.ordered_feature_keys,
            semantics={
                "sampling": "uniform-p1-vehicle-well-then-uniform-nucleus",
                "prediction_support": "conditioned-on-positive-panel-total",
                "maximum_zero_total_redraws": MAX_ZERO_TOTAL_REDRAWS,
                "treated_matching": "same-plate vehicles of exact p1 condition counterpart",
                "no_action_matching": "all p1 vehicle wells",
            },
            fitted_state={
                "p1_condition_pools": _pool_state_manifest(self._by_condition),
                "p1_vehicle_pools_by_plate": _plate_pool_state_manifest(self._vehicles_by_plate),
            },
        )

    def sample(self, request: BaselineSampleRequest) -> PredictiveRawCountSamples:
        request = _validate_sample_request(request)
        condition_wells = _lookup_condition(self._by_condition, request.target.condition)
        if isinstance(request.target.condition, NoAction):
            condition_wells = tuple(
                well for wells in self._vehicles_by_plate.values() for well in wells
            )
        control_wells: list[P1WellCounts] = []
        for plate_id in sorted({well.plate_id for well in condition_wells}):
            controls = self._vehicles_by_plate.get(plate_id)
            if controls is None:
                raise SciPlex3BaselineError(
                    f"p1 matched vehicle support is missing for plate {plate_id!r}"
                )
            control_wells.extend(controls)
        samples = _resample_equal_wells(
            control_wells,
            sample_count=request.sample_count,
            generator=_generator(request.seed),
        )
        return _make_output(
            baseline_id=self.baseline_id,
            ordered_feature_keys=self.ordered_feature_keys,
            request=request,
            samples=samples,
        )


@dataclass(frozen=True, slots=True)
class ExactConditionRep1EmpiricalResampling:
    """Uniform-well empirical resampling from the exact p1 compound-dose support."""

    baseline_id: ClassVar[str] = "exact-condition-rep1-empirical-resampling"
    ordered_feature_keys: tuple[str, ...]
    _by_condition: Mapping[TargetCondition, tuple[P1WellCounts, ...]] = field(repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "ordered_feature_keys", _freeze_ordered_feature_keys(self.ordered_feature_keys)
        )
        object.__setattr__(
            self,
            "_by_condition",
            _freeze_condition_pools(self._by_condition, name="exact empirical condition pools"),
        )

    @classmethod
    def fit(cls, training: P1TrainingData) -> ExactConditionRep1EmpiricalResampling:
        return cls(training.ordered_feature_keys, _wells_by_condition(training))

    def fitted_state_manifest(self) -> dict[str, object]:
        return _base_fitted_manifest(
            baseline_id=self.baseline_id,
            ordered_feature_keys=self.ordered_feature_keys,
            semantics={
                "sampling": "uniform-exact-p1-well-then-uniform-nucleus",
                "prediction_support": "conditioned-on-positive-panel-total",
                "maximum_zero_total_redraws": MAX_ZERO_TOTAL_REDRAWS,
            },
            fitted_state={"p1_condition_pools": _pool_state_manifest(self._by_condition)},
        )

    def sample(self, request: BaselineSampleRequest) -> PredictiveRawCountSamples:
        request = _validate_sample_request(request)
        wells = _lookup_condition(self._by_condition, request.target.condition)
        samples = _resample_equal_wells(
            wells,
            sample_count=request.sample_count,
            generator=_generator(request.seed),
        )
        return _make_output(
            baseline_id=self.baseline_id,
            ordered_feature_keys=self.ordered_feature_keys,
            request=request,
            samples=samples,
        )


@dataclass(frozen=True, slots=True)
class ExactConditionNegativeBinomial:
    """Per-condition Gamma--Poisson with fixed equal-well global smoothing."""

    baseline_id: ClassVar[str] = "exact-condition-negative-binomial"
    ordered_feature_keys: tuple[str, ...]
    _means: Mapping[TargetCondition, FloatArray] = field(repr=False)
    _dispersions: Mapping[TargetCondition, FloatArray] = field(repr=False)

    def __post_init__(self) -> None:
        ordered_feature_keys = _freeze_ordered_feature_keys(self.ordered_feature_keys)
        means = _freeze_parameter_mapping(self._means, name="exact-condition means")
        dispersions = _freeze_parameter_mapping(
            self._dispersions, name="exact-condition dispersions"
        )
        if means.keys() != dispersions.keys():
            raise SciPlex3BaselineError("exact-condition mean and dispersion supports differ")
        object.__setattr__(self, "ordered_feature_keys", ordered_feature_keys)
        object.__setattr__(self, "_means", means)
        object.__setattr__(self, "_dispersions", dispersions)

    @classmethod
    def fit(cls, training: P1TrainingData) -> ExactConditionNegativeBinomial:
        by_condition = _wells_by_condition(training)
        global_mean = _equal_well_mean(training.wells)
        global_dispersion = _equal_well_dispersion(training.wells)
        means: dict[TargetCondition, FloatArray] = {}
        dispersions: dict[TargetCondition, FloatArray] = {}
        for condition, wells in by_condition.items():
            condition_mean = _equal_well_mean(wells)
            condition_dispersion = _equal_well_dispersion(wells)
            means[condition] = _immutable_float_vector(
                (condition_mean + EXACT_MEAN_GLOBAL_PSEUDO_WELLS * global_mean)
                / (1.0 + EXACT_MEAN_GLOBAL_PSEUDO_WELLS),
                name="exact-condition mean",
            )
            dispersions[condition] = _immutable_float_vector(
                (condition_dispersion + EXACT_DISPERSION_GLOBAL_PSEUDO_WELLS * global_dispersion)
                / (1.0 + EXACT_DISPERSION_GLOBAL_PSEUDO_WELLS),
                name="exact-condition dispersion",
            )
        return cls(
            training.ordered_feature_keys,
            MappingProxyType(means),
            MappingProxyType(dispersions),
        )

    def predictive_mean(self, condition: TargetCondition) -> FloatArray:
        _lookup_parameter_support(self._means, condition)
        return self._means[condition]

    def fitted_state_manifest(self) -> dict[str, object]:
        return _base_fitted_manifest(
            baseline_id=self.baseline_id,
            ordered_feature_keys=self.ordered_feature_keys,
            semantics={
                "distribution": "independent-feature-gamma-poisson-conditioned-positive-panel",
                "mean_global_pseudo_wells": EXACT_MEAN_GLOBAL_PSEUDO_WELLS,
                "dispersion_global_pseudo_wells": EXACT_DISPERSION_GLOBAL_PSEUDO_WELLS,
                "maximum_zero_total_redraws": MAX_ZERO_TOTAL_REDRAWS,
            },
            fitted_state={
                "means": _parameter_state_manifest(self._means),
                "dispersions": _parameter_state_manifest(self._dispersions),
            },
        )

    def sample(self, request: BaselineSampleRequest) -> PredictiveRawCountSamples:
        request = _validate_sample_request(request)
        condition = request.target.condition
        mean = self.predictive_mean(condition)
        samples = _sample_gamma_poisson(
            mean,
            self._dispersions[condition],
            sample_count=request.sample_count,
            generator=_generator(request.seed),
        )
        return _make_output(
            baseline_id=self.baseline_id,
            ordered_feature_keys=self.ordered_feature_keys,
            request=request,
            samples=samples,
        )


def _lookup_parameter_support(
    parameters: Mapping[TargetCondition, FloatArray], condition: TargetCondition
) -> None:
    if condition in parameters:
        return
    if isinstance(condition, NoAction):
        raise SciPlex3BaselineError("p1 no-action vehicle support is missing")
    supported_doses = sorted(
        key.dose_nm
        for key in parameters
        if isinstance(key, CompoundDose) and key.compound == condition.compound
    )
    if not supported_doses:
        raise SciPlex3BaselineError(f"p1 compound support is missing for {condition.compound!r}")
    raise SciPlex3BaselineError(
        f"p1 dose support is missing for {condition.compound!r} at {condition.dose_nm} nM; "
        f"supported doses are {supported_doses}"
    )


@dataclass(frozen=True, slots=True)
class HierarchicalWellNegativeBinomial:
    """Well-unit Gamma--Poisson shrinkage across condition, compound, and global levels."""

    baseline_id: ClassVar[str] = "hierarchical-well-negative-binomial"
    ordered_feature_keys: tuple[str, ...]
    _means: Mapping[TargetCondition, FloatArray] = field(repr=False)
    _dispersions: Mapping[TargetCondition, FloatArray] = field(repr=False)

    def __post_init__(self) -> None:
        ordered_feature_keys = _freeze_ordered_feature_keys(self.ordered_feature_keys)
        means = _freeze_parameter_mapping(self._means, name="hierarchical means")
        dispersions = _freeze_parameter_mapping(self._dispersions, name="hierarchical dispersions")
        if means.keys() != dispersions.keys():
            raise SciPlex3BaselineError("hierarchical mean and dispersion supports differ")
        object.__setattr__(self, "ordered_feature_keys", ordered_feature_keys)
        object.__setattr__(self, "_means", means)
        object.__setattr__(self, "_dispersions", dispersions)

    @classmethod
    def fit(cls, training: P1TrainingData) -> HierarchicalWellNegativeBinomial:
        by_condition = _wells_by_condition(training)
        by_compound: defaultdict[str, list[P1WellCounts]] = defaultdict(list)
        for condition, wells in by_condition.items():
            if isinstance(condition, CompoundDose):
                by_compound[condition.compound].extend(wells)
        global_mean = _equal_well_mean(training.wells)
        global_dispersion = _equal_well_dispersion(training.wells)
        denominator = (
            HIERARCHICAL_CONDITION_WELLS
            + HIERARCHICAL_COMPOUND_PSEUDO_WELLS
            + HIERARCHICAL_GLOBAL_PSEUDO_WELLS
        )
        means: dict[TargetCondition, FloatArray] = {}
        dispersions: dict[TargetCondition, FloatArray] = {}
        for condition, wells in by_condition.items():
            compound_wells = (
                by_compound[condition.compound]
                if isinstance(condition, CompoundDose)
                else list(wells)
            )
            means[condition] = _immutable_float_vector(
                (
                    HIERARCHICAL_CONDITION_WELLS * _equal_well_mean(wells)
                    + HIERARCHICAL_COMPOUND_PSEUDO_WELLS * _equal_well_mean(compound_wells)
                    + HIERARCHICAL_GLOBAL_PSEUDO_WELLS * global_mean
                )
                / denominator,
                name="hierarchical mean",
            )
            dispersions[condition] = _immutable_float_vector(
                (
                    HIERARCHICAL_CONDITION_WELLS * _equal_well_dispersion(wells)
                    + HIERARCHICAL_COMPOUND_PSEUDO_WELLS * _equal_well_dispersion(compound_wells)
                    + HIERARCHICAL_GLOBAL_PSEUDO_WELLS * global_dispersion
                )
                / denominator,
                name="hierarchical dispersion",
            )
        return cls(
            training.ordered_feature_keys,
            MappingProxyType(means),
            MappingProxyType(dispersions),
        )

    def predictive_mean(self, condition: TargetCondition) -> FloatArray:
        _lookup_parameter_support(self._means, condition)
        return self._means[condition]

    def fitted_state_manifest(self) -> dict[str, object]:
        return _base_fitted_manifest(
            baseline_id=self.baseline_id,
            ordered_feature_keys=self.ordered_feature_keys,
            semantics={
                "distribution": "independent-feature-gamma-poisson-conditioned-positive-panel",
                "condition_wells": HIERARCHICAL_CONDITION_WELLS,
                "compound_pseudo_wells": HIERARCHICAL_COMPOUND_PSEUDO_WELLS,
                "global_pseudo_wells": HIERARCHICAL_GLOBAL_PSEUDO_WELLS,
                "maximum_zero_total_redraws": MAX_ZERO_TOTAL_REDRAWS,
            },
            fitted_state={
                "means": _parameter_state_manifest(self._means),
                "dispersions": _parameter_state_manifest(self._dispersions),
            },
        )

    def sample(self, request: BaselineSampleRequest) -> PredictiveRawCountSamples:
        request = _validate_sample_request(request)
        condition = request.target.condition
        mean = self.predictive_mean(condition)
        samples = _sample_gamma_poisson(
            mean,
            self._dispersions[condition],
            sample_count=request.sample_count,
            generator=_generator(request.seed),
        )
        return _make_output(
            baseline_id=self.baseline_id,
            ordered_feature_keys=self.ordered_feature_keys,
            request=request,
            samples=samples,
        )


@dataclass(frozen=True, slots=True)
class LowRankCompoundDoseResponse:
    """Truncated-SVD model of p1 log1p compound-dose effects over matched p1 vehicles."""

    baseline_id: ClassVar[str] = "low-rank-compound-dose-response"
    ordered_feature_keys: tuple[str, ...]
    rank: int
    _means: Mapping[TargetCondition, FloatArray] = field(repr=False)
    _global_dispersion: FloatArray = field(repr=False)

    def __post_init__(self) -> None:
        if type(self.rank) is not int or self.rank <= 0:
            raise SciPlex3BaselineError("low-rank response rank must be a positive integer")
        ordered_feature_keys = _freeze_ordered_feature_keys(self.ordered_feature_keys)
        means = _freeze_parameter_mapping(self._means, name="low-rank means")
        treated_count = sum(type(condition) is CompoundDose for condition in means)
        if NO_ACTION not in means or self.rank > min(treated_count, SCIPLEX3_FEATURE_COUNT):
            raise SciPlex3BaselineError("low-rank response rank or no-action support is invalid")
        dispersion = _immutable_float_vector(
            self._global_dispersion, name="low-rank global dispersion"
        )
        object.__setattr__(self, "ordered_feature_keys", ordered_feature_keys)
        object.__setattr__(self, "_means", means)
        object.__setattr__(self, "_global_dispersion", dispersion)

    @classmethod
    def fit(
        cls, training: P1TrainingData, *, rank: int = DEFAULT_LOW_RANK
    ) -> LowRankCompoundDoseResponse:
        if type(rank) is not int or rank <= 0:
            raise SciPlex3BaselineError("low-rank response rank must be a positive integer")
        by_condition = _wells_by_condition(training)
        by_vehicle_plate = _vehicles_by_plate(training)
        conditions = tuple(sorted(key for key in by_condition if isinstance(key, CompoundDose)))
        if len(conditions) < 2:
            raise SciPlex3BaselineError("low-rank response requires at least two conditions")
        for compound in sorted({condition.compound for condition in conditions}):
            dose_count = len({item.dose_nm for item in conditions if item.compound == compound})
            if dose_count < 2:
                raise SciPlex3BaselineError(
                    f"low-rank response lacks multiple dose supports for {compound!r}"
                )
        effect_rows: list[FloatArray] = []
        vehicle_means: list[FloatArray] = []
        for condition in conditions:
            condition_wells = by_condition[condition]
            condition_mean = _equal_well_mean(condition_wells)
            matched_controls: list[P1WellCounts] = []
            for plate_id in sorted({well.plate_id for well in condition_wells}):
                controls = by_vehicle_plate.get(plate_id)
                if controls is None:
                    raise SciPlex3BaselineError(
                        f"low-rank response is missing p1 vehicle controls for plate {plate_id!r}"
                    )
                matched_controls.extend(controls)
            vehicle_mean = _equal_well_mean(matched_controls)
            vehicle_means.append(vehicle_mean)
            effect_rows.append(np.log1p(condition_mean) - np.log1p(vehicle_mean))
        effects = np.stack(effect_rows)
        maximum_rank = min(effects.shape)
        effective_rank = min(rank, maximum_rank)
        left, singular_values, right = np.linalg.svd(effects, full_matrices=False)
        left, right = _canonicalize_svd(left, right)
        if effective_rank < len(singular_values) and np.isclose(
            singular_values[effective_rank - 1],
            singular_values[effective_rank],
            rtol=LOW_RANK_SVD_TIE_RTOL,
            atol=0.0,
        ):
            raise SciPlex3BaselineError(
                "low-rank SVD has a degenerate singular-value tie at the retained boundary"
            )
        reconstructed = (left[:, :effective_rank] * singular_values[:effective_rank]) @ right[
            :effective_rank
        ]
        means: dict[TargetCondition, FloatArray] = {
            NO_ACTION: _immutable_float_vector(
                _equal_well_mean(by_condition[NO_ACTION]), name="low-rank no-action mean"
            )
        }
        for row_index, condition in enumerate(conditions):
            log_mean = np.log1p(vehicle_means[row_index]) + reconstructed[row_index]
            # expm1 can produce minute negatives when both terms are near zero.
            predicted = np.round(
                np.maximum(np.expm1(log_mean), 0.0), decimals=LOW_RANK_MEAN_DECIMALS
            )
            means[condition] = _immutable_float_vector(predicted, name="low-rank predictive mean")
        return cls(
            ordered_feature_keys=training.ordered_feature_keys,
            rank=effective_rank,
            _means=MappingProxyType(means),
            _global_dispersion=_immutable_float_vector(
                _equal_well_dispersion(training.wells), name="global well dispersion"
            ),
        )

    def predictive_mean(self, condition: TargetCondition) -> FloatArray:
        _lookup_parameter_support(self._means, condition)
        return self._means[condition]

    def fitted_state_manifest(self) -> dict[str, object]:
        return _base_fitted_manifest(
            baseline_id=self.baseline_id,
            ordered_feature_keys=self.ordered_feature_keys,
            semantics={
                "distribution": "independent-feature-gamma-poisson-conditioned-positive-panel",
                "requested_default_rank": DEFAULT_LOW_RANK,
                "effective_rank": self.rank,
                "numeric_contract": LOW_RANK_NUMERIC_CONTRACT,
                "mean_rounding_decimals": LOW_RANK_MEAN_DECIMALS,
                "svd_boundary_tie_rtol": LOW_RANK_SVD_TIE_RTOL,
                "maximum_zero_total_redraws": MAX_ZERO_TOTAL_REDRAWS,
            },
            fitted_state={
                "reconstructed_means": _parameter_state_manifest(self._means),
                "global_dispersion": _array_manifest(self._global_dispersion),
            },
        )

    def sample(self, request: BaselineSampleRequest) -> PredictiveRawCountSamples:
        request = _validate_sample_request(request)
        samples = _sample_gamma_poisson(
            self.predictive_mean(request.target.condition),
            self._global_dispersion,
            sample_count=request.sample_count,
            generator=_generator(request.seed),
        )
        return _make_output(
            baseline_id=self.baseline_id,
            ordered_feature_keys=self.ordered_feature_keys,
            request=request,
            samples=samples,
        )


@dataclass(frozen=True, slots=True)
class NearestSupportedDose:
    """Secondary empirical baseline at the nearest same-compound p1 log-dose."""

    baseline_id: ClassVar[str] = "nearest-supported-dose"
    ordered_feature_keys: tuple[str, ...]
    _by_condition: Mapping[TargetCondition, tuple[P1WellCounts, ...]] = field(repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "ordered_feature_keys", _freeze_ordered_feature_keys(self.ordered_feature_keys)
        )
        object.__setattr__(
            self,
            "_by_condition",
            _freeze_condition_pools(self._by_condition, name="nearest-dose condition pools"),
        )

    @classmethod
    def fit(cls, training: P1TrainingData) -> NearestSupportedDose:
        return cls(training.ordered_feature_keys, _wells_by_condition(training))

    def fitted_state_manifest(self) -> dict[str, object]:
        return _base_fitted_manifest(
            baseline_id=self.baseline_id,
            ordered_feature_keys=self.ordered_feature_keys,
            semantics={
                "sampling": "uniform-selected-p1-well-then-uniform-nucleus",
                "distance": "absolute-log10-dose",
                "exclude_exact_requested_dose": True,
                "prediction_support": "conditioned-on-positive-panel-total",
                "maximum_zero_total_redraws": MAX_ZERO_TOTAL_REDRAWS,
                "tie_break": "lower-dose",
                "tie_absolute_tolerance": NEAREST_DOSE_TIE_ATOL,
            },
            fitted_state={"p1_condition_pools": _pool_state_manifest(self._by_condition)},
        )

    def supported_condition(self, condition: TargetCondition) -> TargetCondition:
        if isinstance(condition, NoAction):
            _lookup_condition(self._by_condition, condition)
            return NO_ACTION
        candidates = tuple(
            key
            for key in self._by_condition
            if isinstance(key, CompoundDose)
            and key.compound == condition.compound
            and key.dose_nm != condition.dose_nm
        )
        if not candidates:
            compound_present = any(
                isinstance(key, CompoundDose) and key.compound == condition.compound
                for key in self._by_condition
            )
            if compound_present:
                raise SciPlex3BaselineError(
                    f"p1 alternate-dose support is missing for {condition.compound!r}"
                )
            raise SciPlex3BaselineError(
                f"p1 compound support is missing for {condition.compound!r}"
            )
        distances = {
            item: abs(np.log10(item.dose_nm) - np.log10(condition.dose_nm)) for item in candidates
        }
        minimum_distance = min(distances.values())
        tied = tuple(
            item
            for item, distance in distances.items()
            if bool(np.isclose(distance, minimum_distance, rtol=0.0, atol=NEAREST_DOSE_TIE_ATOL))
        )
        return min(tied, key=lambda item: item.dose_nm)

    def sample(self, request: BaselineSampleRequest) -> PredictiveRawCountSamples:
        request = _validate_sample_request(request)
        supported = self.supported_condition(request.target.condition)
        samples = _resample_equal_wells(
            self._by_condition[supported],
            sample_count=request.sample_count,
            generator=_generator(request.seed),
        )
        return _make_output(
            baseline_id=self.baseline_id,
            ordered_feature_keys=self.ordered_feature_keys,
            request=request,
            samples=samples,
        )


SciPlex3BaselineImplementation = (
    MatchedVehicleResampling
    | ExactConditionRep1EmpiricalResampling
    | ExactConditionNegativeBinomial
    | HierarchicalWellNegativeBinomial
    | LowRankCompoundDoseResponse
    | NearestSupportedDose
)

SCIPLEX3_BASELINE_IMPLEMENTATIONS: Mapping[str, type[SciPlex3BaselineImplementation]] = (
    MappingProxyType(
        {
            ExactConditionNegativeBinomial.baseline_id: ExactConditionNegativeBinomial,
            ExactConditionRep1EmpiricalResampling.baseline_id: (
                ExactConditionRep1EmpiricalResampling
            ),
            HierarchicalWellNegativeBinomial.baseline_id: HierarchicalWellNegativeBinomial,
            LowRankCompoundDoseResponse.baseline_id: LowRankCompoundDoseResponse,
            MatchedVehicleResampling.baseline_id: MatchedVehicleResampling,
            NearestSupportedDose.baseline_id: NearestSupportedDose,
        }
    )
)


def fit_sciplex3_baseline_suite(
    training: P1TrainingData, *, low_rank: int = DEFAULT_LOW_RANK
) -> Mapping[str, SciPlex3RawCountBaseline]:
    """Fit the five mandatory baselines and the secondary alternate-dose baseline."""

    fitted: dict[str, SciPlex3RawCountBaseline] = {
        ExactConditionNegativeBinomial.baseline_id: ExactConditionNegativeBinomial.fit(training),
        ExactConditionRep1EmpiricalResampling.baseline_id: (
            ExactConditionRep1EmpiricalResampling.fit(training)
        ),
        HierarchicalWellNegativeBinomial.baseline_id: (
            HierarchicalWellNegativeBinomial.fit(training)
        ),
        LowRankCompoundDoseResponse.baseline_id: LowRankCompoundDoseResponse.fit(
            training, rank=low_rank
        ),
        MatchedVehicleResampling.baseline_id: MatchedVehicleResampling.fit(training),
        NearestSupportedDose.baseline_id: NearestSupportedDose.fit(training),
    }
    return MappingProxyType(fitted)


__all__ = [
    "DEFAULT_LOW_RANK",
    "LOW_RANK_MEAN_DECIMALS",
    "LOW_RANK_NUMERIC_CONTRACT",
    "LOW_RANK_SVD_TIE_RTOL",
    "MAX_ZERO_TOTAL_REDRAWS",
    "NEAREST_DOSE_TIE_ATOL",
    "NO_ACTION",
    "RNG_ALGORITHM",
    "SCIPLEX3_BASELINE_IMPLEMENTATIONS",
    "SCIPLEX3_BASELINE_IMPLEMENTATION_VERSION",
    "SCIPLEX3_BASELINE_SAMPLES_PER_CASE_PER_SEED",
    "SCIPLEX3_BASELINE_SEEDS",
    "SCIPLEX3_FEATURE_COUNT",
    "SCIPLEX3_ORDERED_FEATURE_KEYS_SHA256",
    "BaselineSampleRequest",
    "CompoundDose",
    "ExactConditionNegativeBinomial",
    "ExactConditionRep1EmpiricalResampling",
    "HierarchicalWellNegativeBinomial",
    "ImmutableCSRCounts",
    "LowRankCompoundDoseResponse",
    "MatchedVehicleResampling",
    "NearestSupportedDose",
    "NoAction",
    "P1TrainingData",
    "P1WellCounts",
    "PredictionTarget",
    "PredictiveRawCountSamples",
    "SciPlex3BaselineError",
    "SciPlex3BaselineImplementation",
    "SciPlex3RawCountBaseline",
    "TargetCondition",
    "fit_sciplex3_baseline_suite",
]
