from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import fields
from pathlib import Path
from typing import Literal, cast

import numpy as np
import pytest
from numpy.typing import NDArray

import cellstate.evaluation.sciplex3_baselines as baseline_module
from cellstate.evaluation.sciplex3_baselines import (
    EXACT_MEAN_GLOBAL_PSEUDO_WELLS,
    HIERARCHICAL_COMPOUND_PSEUDO_WELLS,
    HIERARCHICAL_CONDITION_WELLS,
    HIERARCHICAL_GLOBAL_PSEUDO_WELLS,
    MAX_ZERO_TOTAL_REDRAWS,
    NO_ACTION,
    RNG_ALGORITHM,
    SCIPLEX3_BASELINE_IMPLEMENTATION_VERSION,
    SCIPLEX3_BASELINE_IMPLEMENTATIONS,
    SCIPLEX3_BASELINE_SAMPLES_PER_CASE_PER_SEED,
    SCIPLEX3_BASELINE_SEEDS,
    SCIPLEX3_FEATURE_COUNT,
    SCIPLEX3_ORDERED_FEATURE_KEYS_SHA256,
    BaselineSampleRequest,
    CompoundDose,
    ExactConditionNegativeBinomial,
    ExactConditionRep1EmpiricalResampling,
    HierarchicalWellNegativeBinomial,
    ImmutableCSRCounts,
    LowRankCompoundDoseResponse,
    MatchedVehicleResampling,
    NearestSupportedDose,
    P1TrainingData,
    P1WellCounts,
    PredictionTarget,
    PredictiveRawCountSamples,
    SciPlex3BaselineError,
    SciPlex3RawCountBaseline,
    TargetCondition,
    fit_sciplex3_baseline_suite,
)

IntArray = NDArray[np.int64]


def _keys() -> tuple[str, ...]:
    return tuple(f"feature-{index:04d}" for index in range(SCIPLEX3_FEATURE_COUNT))


def _counts(*rows: tuple[int, ...]) -> IntArray:
    matrix = np.zeros((len(rows), SCIPLEX3_FEATURE_COUNT), dtype=np.int64)
    for row_index, row in enumerate(rows):
        matrix[row_index, : len(row)] = row
    return matrix


def _well(
    well_id: str,
    plate_id: str,
    condition: CompoundDose | None,
    *rows: tuple[int, ...],
) -> P1WellCounts:
    source_base = int(hashlib.sha256(well_id.encode()).hexdigest()[:12], 16) * 1_000
    return P1WellCounts(
        well_id=well_id,
        plate_id=plate_id,
        condition=condition,
        counts=ImmutableCSRCounts.from_dense(_counts(*rows)),
        record_ids=tuple(f"{well_id}-record-{index:04d}" for index in range(len(rows))),
        source_row_indices=tuple(source_base + index for index in range(len(rows))),
    )


def _training() -> P1TrainingData:
    a10 = CompoundDose("compound-a", 10)
    a100 = CompoundDose("compound-a", 100)
    b10 = CompoundDose("compound-b", 10)
    b100 = CompoundDose("compound-b", 100)
    return P1TrainingData(
        ordered_feature_keys=_keys(),
        wells=(
            _well("vehicle-a-1", "plate-a", None, (1, 2, 1), (2, 1, 1), (1, 1, 2)),
            _well("vehicle-a-2", "plate-a", None, (30, 1, 1), (31, 2, 1)),
            _well("a-10", "plate-a", a10, (7, 2, 1), (9, 3, 2), (8, 4, 1)),
            _well("a-100", "plate-a", a100, (13, 3, 2), (15, 4, 3), (14, 5, 2)),
            _well("vehicle-b", "plate-b", None, (2, 3, 1), (3, 2, 1), (2, 2, 2)),
            _well("b-10", "plate-b", b10, (4, 9, 1), (5, 8, 2), (6, 10, 1)),
            _well("b-100", "plate-b", b100, (5, 15, 2), (7, 16, 2), (6, 14, 3)),
        ),
    )


def _target(condition: TargetCondition | None = None) -> PredictionTarget:
    return PredictionTarget(
        case_id="case-p4-a10",
        target_well_id="p4-treated-a10",
        plate_id="plate-p4",
        partition_id="p4-untouched-test",
        condition=condition or CompoundDose("compound-a", 10),
    )


def _request(
    condition: TargetCondition | None = None,
    *,
    sample_count: int = 16,
    seed: int = 20260810,
) -> BaselineSampleRequest:
    return BaselineSampleRequest(
        target=_target(condition),
        sample_count=sample_count,
        seed=seed,
    )


@pytest.mark.parametrize(
    ("compound", "dose", "message"),
    [
        ("", 10, "compound"),
        (" padded ", 10, "compound"),
        ("drug", 0, "positive"),
        ("drug", -1, "positive"),
        ("drug", True, "integer"),
        ("drug", 1.5, "integer"),
    ],
)
def test_compound_dose_is_strict(compound: str, dose: object, message: str) -> None:
    with pytest.raises(SciPlex3BaselineError, match=message):
        CompoundDose(compound, cast(int, dose))


def test_loader_compatible_csr_is_ordered_immutable_and_sparse() -> None:
    source = _counts((1, 2, 3), (4, 5, 6))
    counts = ImmutableCSRCounts.from_dense(source)
    well = P1WellCounts("well", "plate", None, counts, ("record-0", "record-1"), (0, 1))
    source[0, 0] = 999

    assert well.is_vehicle
    assert well.counts.row_dense(0)[:3].tolist() == [1, 2, 3]
    assert well.counts.nnz == 6
    assert not well.counts.values.flags.writeable
    with pytest.raises(ValueError):
        well.counts.values.setflags(write=True)
    with pytest.raises(SciPlex3BaselineError, match="out of range"):
        well.counts.row_dense(2)


@pytest.mark.parametrize(
    ("counts", "message"),
    [
        (np.zeros(SCIPLEX3_FEATURE_COUNT, dtype=np.int64), "two-dimensional"),
        (np.zeros((0, SCIPLEX3_FEATURE_COUNT), dtype=np.int64), "at least one row"),
        (np.ones((1, SCIPLEX3_FEATURE_COUNT - 1), dtype=np.int64), "exactly 2000"),
        (np.ones((1, SCIPLEX3_FEATURE_COUNT), dtype=np.float64), "integer counts"),
        (np.ones((1, SCIPLEX3_FEATURE_COUNT), dtype=np.bool_), "integer counts"),
        (-np.ones((1, SCIPLEX3_FEATURE_COUNT), dtype=np.int64), "negative"),
    ],
)
def test_p1_well_rejects_invalid_count_matrices(counts: object, message: str) -> None:
    with pytest.raises(SciPlex3BaselineError, match=message):
        ImmutableCSRCounts.from_dense(counts)


def test_p1_well_rejects_unsigned_overflow_partition_replicate_and_bad_ids() -> None:
    overflow = np.zeros((1, SCIPLEX3_FEATURE_COUNT), dtype=np.uint64)
    overflow[0, 0] = np.iinfo(np.uint64).max
    with pytest.raises(SciPlex3BaselineError, match="64-bit"):
        ImmutableCSRCounts.from_dense(overflow)
    valid = ImmutableCSRCounts.from_dense(_counts((1,)))
    with pytest.raises(SciPlex3BaselineError, match="immutable CSR"):
        P1WellCounts(
            "well",
            "plate",
            None,
            cast(ImmutableCSRCounts, _counts((1,))),
            ("record",),
            (0,),
        )
    with pytest.raises(SciPlex3BaselineError, match="p1-train"):
        P1WellCounts(
            "well",
            "plate",
            None,
            valid,
            ("record",),
            (0,),
            partition_id=cast("Literal['p1-train']", "p2-calibration"),
        )
    with pytest.raises(SciPlex3BaselineError, match="rep1"):
        P1WellCounts(
            "well",
            "plate",
            None,
            valid,
            ("record",),
            (0,),
            replicate=cast("Literal['rep1']", "rep2"),
        )
    with pytest.raises(SciPlex3BaselineError, match="well_id"):
        P1WellCounts(" bad ", "plate", None, valid, ("record",), (0,))


def test_csr_loader_adapter_validates_structure_panel_order_and_totals() -> None:
    counts = ImmutableCSRCounts(
        indptr=np.asarray([0, 2, 3], dtype=np.int64),
        feature_indices=np.asarray([0, 2, 1], dtype=np.int64),
        values=np.asarray([2, 4, 3], dtype=np.int64),
        row_count=2,
    )
    assert counts.row_dense(0)[:4].tolist() == [2, 0, 4, 0]
    assert counts.row_dense(1)[:4].tolist() == [0, 3, 0, 0]
    assert np.allclose(counts.feature_mean()[:4], [1.0, 1.5, 2.0, 0.0])
    dense = np.stack([counts.row_dense(0), counts.row_dense(1)])
    mean = np.mean(dense, axis=0)
    variance = np.var(dense, axis=0, ddof=1)
    expected_dispersion = np.zeros(SCIPLEX3_FEATURE_COUNT)
    nonzero = mean > 0
    expected_dispersion[nonzero] = np.maximum(
        (variance[nonzero] - mean[nonzero]) / np.square(mean[nonzero]), 0.0
    )
    assert np.allclose(counts.feature_dispersion(), expected_dispersion)


def test_csr_loader_adapter_rejects_malformed_sparse_arrays() -> None:
    array = np.asarray
    with pytest.raises(SciPlex3BaselineError, match="row_count must be an integer"):
        ImmutableCSRCounts(array([0, 1]), array([0]), array([1]), cast(int, True))
    with pytest.raises(SciPlex3BaselineError, match="positive"):
        ImmutableCSRCounts(array([0]), array([], dtype=int), array([], dtype=int), 0)
    with pytest.raises(SciPlex3BaselineError, match="one-dimensional integer"):
        ImmutableCSRCounts(array([0.0, 1.0]), array([0]), array([1]), 1)
    with pytest.raises(SciPlex3BaselineError, match=r"row_count \+ 1"):
        ImmutableCSRCounts(array([0, 1, 1]), array([0]), array([1]), 1)
    with pytest.raises(SciPlex3BaselineError, match="start at zero"):
        ImmutableCSRCounts(array([1, 1]), array([], dtype=int), array([], dtype=int), 1)
    with pytest.raises(SciPlex3BaselineError, match="lengths disagree"):
        ImmutableCSRCounts(array([0, 2]), array([0]), array([1]), 1)
    with pytest.raises(SciPlex3BaselineError, match="outside the ordered panel"):
        ImmutableCSRCounts(array([0, 1]), array([SCIPLEX3_FEATURE_COUNT]), array([1]), 1)
    with pytest.raises(SciPlex3BaselineError, match="strictly positive"):
        ImmutableCSRCounts(array([0, 1]), array([0]), array([0]), 1)
    retained_zero = ImmutableCSRCounts(array([0, 0, 1]), array([0]), array([1]), 2)
    assert retained_zero.row_dense(0).sum() == 0
    assert retained_zero.row_dense(1).sum() == 1
    with pytest.raises(SciPlex3BaselineError, match="strictly increasing"):
        ImmutableCSRCounts(array([0, 2]), array([1, 0]), array([1, 1]), 1)


def test_training_surface_rejects_bad_feature_panels_and_missing_roles() -> None:
    vehicle = _well("vehicle", "plate", None, (1,))
    treated = _well("treated", "plate", CompoundDose("drug", 10), (2,))
    with pytest.raises(SciPlex3BaselineError, match="exactly 2000"):
        P1TrainingData(("only-one",), (vehicle, treated))
    blank_keys = list(_keys())
    blank_keys[0] = ""
    with pytest.raises(SciPlex3BaselineError, match="nonblank"):
        P1TrainingData(tuple(blank_keys), (vehicle, treated))
    duplicate_keys = list(_keys())
    duplicate_keys[1] = duplicate_keys[0]
    with pytest.raises(SciPlex3BaselineError, match="unique"):
        P1TrainingData(tuple(duplicate_keys), (vehicle, treated))
    with pytest.raises(SciPlex3BaselineError, match="must contain wells"):
        P1TrainingData(_keys(), ())
    with pytest.raises(SciPlex3BaselineError, match="no treated"):
        P1TrainingData(_keys(), (vehicle,))
    with pytest.raises(SciPlex3BaselineError, match="no vehicle"):
        P1TrainingData(_keys(), (treated,))


def test_training_surface_rejects_duplicate_wells_and_missing_plate_control() -> None:
    vehicle = _well("same-id", "plate-a", None, (1,))
    duplicate = _well("same-id", "plate-a", CompoundDose("drug", 10), (2,))
    with pytest.raises(SciPlex3BaselineError, match="well IDs must be unique"):
        P1TrainingData(_keys(), (vehicle, duplicate))
    other_plate = _well("treated", "plate-b", CompoundDose("drug", 10), (2,))
    with pytest.raises(SciPlex3BaselineError, match="plate-b"):
        P1TrainingData(_keys(), (vehicle, other_plate))


def test_p1_rows_require_canonical_record_and_source_provenance() -> None:
    counts = ImmutableCSRCounts.from_dense(_counts((1,), (2,)))
    with pytest.raises(SciPlex3BaselineError, match="record ID count"):
        P1WellCounts("w", "p", None, counts, ("one",), (1, 2))
    with pytest.raises(SciPlex3BaselineError, match="record IDs must be unique"):
        P1WellCounts("w", "p", None, counts, ("same", "same"), (1, 2))
    with pytest.raises(SciPlex3BaselineError, match="canonical record-ID order"):
        P1WellCounts("w", "p", None, counts, ("z", "a"), (1, 2))
    with pytest.raises(SciPlex3BaselineError, match="source-row count"):
        P1WellCounts("w", "p", None, counts, ("a", "b"), (1,))
    with pytest.raises(SciPlex3BaselineError, match="unique nonnegative"):
        P1WellCounts("w", "p", None, counts, ("a", "b"), (1, 1))

    vehicle = P1WellCounts("v", "p", None, counts, ("a", "b"), (1, 2))
    treated = P1WellCounts("t", "p", CompoundDose("drug", 10), counts, ("a", "c"), (3, 4))
    with pytest.raises(SciPlex3BaselineError, match="globally unique"):
        P1TrainingData(_keys(), (vehicle, treated))
    treated_rows = P1WellCounts("t", "p", CompoundDose("drug", 10), counts, ("c", "d"), (2, 4))
    with pytest.raises(SciPlex3BaselineError, match="source-row indices"):
        P1TrainingData(_keys(), (vehicle, treated_rows))


def test_prediction_target_has_only_outcome_free_design_metadata() -> None:
    assert {item.name for item in fields(PredictionTarget)} == {
        "case_id",
        "target_well_id",
        "plate_id",
        "partition_id",
        "condition",
    }
    with pytest.raises(SciPlex3BaselineError, match="case_id"):
        PredictionTarget(" bad ", "well", "plate", "p4", CompoundDose("drug", 10))


class _TextLike:
    def __str__(self) -> str:
        return "apparently-valid"


class _MutableCondition:
    def __init__(self) -> None:
        self.compound = "drug"
        self.dose_nm = 10


class _IntSubclass(int):
    pass


def test_public_data_surfaces_reject_mutable_or_structural_lookalikes() -> None:
    mutable_condition = cast(TargetCondition, _MutableCondition())
    text_like = cast(str, _TextLike())
    with pytest.raises(SciPlex3BaselineError, match="must be a string"):
        PredictionTarget(text_like, "well", "plate", "p4", CompoundDose("drug", 10))
    with pytest.raises(SciPlex3BaselineError, match="exact immutable"):
        PredictionTarget("case", "well", "plate", "p4", mutable_condition)
    with pytest.raises(SciPlex3BaselineError, match="integer"):
        CompoundDose("drug", _IntSubclass(10))

    counts = ImmutableCSRCounts.from_dense(_counts((1,)))
    with pytest.raises(SciPlex3BaselineError, match="exact immutable"):
        P1WellCounts("well", "plate", mutable_condition, counts, ("record",), (0,))
    with pytest.raises(SciPlex3BaselineError, match="exact P1WellCounts"):
        P1TrainingData(_keys(), cast(tuple[P1WellCounts, ...], (object(),)))
    with pytest.raises(SciPlex3BaselineError, match="exact PredictionTarget"):
        BaselineSampleRequest(cast(PredictionTarget, object()), 1, 0)
    with pytest.raises(SciPlex3BaselineError, match="exact PredictionTarget"):
        PredictiveRawCountSamples(
            "baseline",
            cast(PredictionTarget, object()),
            _keys(),
            0,
            _counts((1,)),
        )


@pytest.mark.parametrize(
    ("sample_count", "seed", "message"),
    [
        (0, 1, "positive"),
        (-1, 1, "positive"),
        (True, 1, "sample_count must be an integer"),
        (1, True, "seed must be an integer"),
        (1, -1, "unsigned 64-bit"),
        (1, 2**64, "unsigned 64-bit"),
    ],
)
def test_sample_request_validates_count_and_seed(
    sample_count: object, seed: object, message: str
) -> None:
    with pytest.raises(SciPlex3BaselineError, match=message):
        BaselineSampleRequest(_target(), cast(int, sample_count), cast(int, seed))


def test_frozen_monte_carlo_schedule_and_rng_are_explicit() -> None:
    assert SCIPLEX3_BASELINE_SAMPLES_PER_CASE_PER_SEED == 512
    assert SCIPLEX3_BASELINE_SEEDS == (0, 1, 2, 3, 4)
    assert RNG_ALGORITHM == "numpy-pcg64dxsm-v1"


def test_every_fitted_state_manifest_is_json_compatible_complete_and_order_stable() -> None:
    training = _training()
    reordered = P1TrainingData(training.ordered_feature_keys, tuple(reversed(training.wells)))
    first: tuple[SciPlex3RawCountBaseline, ...] = (
        MatchedVehicleResampling.fit(training),
        ExactConditionRep1EmpiricalResampling.fit(training),
        ExactConditionNegativeBinomial.fit(training),
        HierarchicalWellNegativeBinomial.fit(training),
        LowRankCompoundDoseResponse.fit(training, rank=1),
        NearestSupportedDose.fit(training),
    )
    second: tuple[SciPlex3RawCountBaseline, ...] = (
        MatchedVehicleResampling.fit(reordered),
        ExactConditionRep1EmpiricalResampling.fit(reordered),
        ExactConditionNegativeBinomial.fit(reordered),
        HierarchicalWellNegativeBinomial.fit(reordered),
        LowRankCompoundDoseResponse.fit(reordered, rank=1),
        NearestSupportedDose.fit(reordered),
    )
    for left, right in zip(first, second, strict=True):
        manifest = left.fitted_state_manifest()
        assert manifest == right.fitted_state_manifest()
        assert manifest["baseline_id"] == left.baseline_id
        assert manifest["implementation_version"] == SCIPLEX3_BASELINE_IMPLEMENTATION_VERSION
        assert manifest["rng_algorithm"] == RNG_ALGORITHM
        assert manifest["samples_per_case_per_seed"] == 512
        assert manifest["seeds"] == [0, 1, 2, 3, 4]
        encoded = json.dumps(
            manifest,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        assert (
            hashlib.sha256(encoded.encode()).hexdigest()
            == hashlib.sha256(encoded.encode()).hexdigest()
        )
    matched_state = first[0].fitted_state_manifest()["fitted_state"]
    assert isinstance(matched_state, dict)
    assert "p1_vehicle_pools_by_plate" in matched_state


def test_all_six_baselines_deep_freeze_public_constructor_state() -> None:
    training = _training()
    request = _request(sample_count=8, seed=19)

    matched_fit = MatchedVehicleResampling.fit(training)
    matched_conditions = {key: list(value) for key, value in matched_fit._by_condition.items()}
    matched_vehicles = {key: list(value) for key, value in matched_fit._vehicles_by_plate.items()}
    matched = MatchedVehicleResampling(
        matched_fit.ordered_feature_keys,
        cast(Mapping[TargetCondition, tuple[P1WellCounts, ...]], matched_conditions),
        cast(Mapping[str, tuple[P1WellCounts, ...]], matched_vehicles),
    )
    matched_manifest = matched.fitted_state_manifest()
    matched_samples = matched.sample(request).samples
    matched_conditions.clear()
    matched_vehicles.clear()
    assert matched.fitted_state_manifest() == matched_manifest
    assert np.array_equal(matched.sample(request).samples, matched_samples)

    for baseline_type in (
        ExactConditionRep1EmpiricalResampling,
        NearestSupportedDose,
    ):
        fitted = baseline_type.fit(training)
        mutable_pools = {key: list(value) for key, value in fitted._by_condition.items()}
        reconstructed = baseline_type(
            fitted.ordered_feature_keys,
            cast(Mapping[TargetCondition, tuple[P1WellCounts, ...]], mutable_pools),
        )
        manifest = reconstructed.fitted_state_manifest()
        samples = reconstructed.sample(request).samples
        mutable_pools.clear()
        assert reconstructed.fitted_state_manifest() == manifest
        assert np.array_equal(reconstructed.sample(request).samples, samples)

    for baseline_type in (
        ExactConditionNegativeBinomial,
        HierarchicalWellNegativeBinomial,
    ):
        fitted = baseline_type.fit(training)
        mutable_means = {key: value.copy() for key, value in fitted._means.items()}
        mutable_dispersions = {key: value.copy() for key, value in fitted._dispersions.items()}
        reconstructed = baseline_type(
            fitted.ordered_feature_keys,
            mutable_means,
            mutable_dispersions,
        )
        manifest = reconstructed.fitted_state_manifest()
        samples = reconstructed.sample(request).samples
        for values in (*mutable_means.values(), *mutable_dispersions.values()):
            values.fill(999.0)
        mutable_means.clear()
        mutable_dispersions.clear()
        assert reconstructed.fitted_state_manifest() == manifest
        assert np.array_equal(reconstructed.sample(request).samples, samples)

    low_rank_fit = LowRankCompoundDoseResponse.fit(training, rank=1)
    mutable_means = {key: value.copy() for key, value in low_rank_fit._means.items()}
    mutable_dispersion = low_rank_fit._global_dispersion.copy()
    low_rank = LowRankCompoundDoseResponse(
        low_rank_fit.ordered_feature_keys,
        low_rank_fit.rank,
        mutable_means,
        mutable_dispersion,
    )
    low_rank_manifest = low_rank.fitted_state_manifest()
    low_rank_samples = low_rank.sample(request).samples
    for values in mutable_means.values():
        values.fill(999.0)
    mutable_dispersion.fill(999.0)
    mutable_means.clear()
    assert low_rank.fitted_state_manifest() == low_rank_manifest
    assert np.array_equal(low_rank.sample(request).samples, low_rank_samples)


def test_matched_vehicle_reconstruction_rejects_forged_hidden_plate_pools() -> None:
    baseline = MatchedVehicleResampling.fit(_training())
    forged = dict(baseline._vehicles_by_plate)
    forged["plate-a"], forged["plate-b"] = forged["plate-b"], forged["plate-a"]
    with pytest.raises(SciPlex3BaselineError, match=r"wrong-plate|authenticated no-action"):
        MatchedVehicleResampling(
            baseline.ordered_feature_keys,
            baseline._by_condition,
            forged,
        )


def test_all_baselines_implement_raw_count_protocol() -> None:
    training = _training()
    baselines: tuple[SciPlex3RawCountBaseline, ...] = (
        MatchedVehicleResampling.fit(training),
        ExactConditionRep1EmpiricalResampling.fit(training),
        ExactConditionNegativeBinomial.fit(training),
        HierarchicalWellNegativeBinomial.fit(training),
        LowRankCompoundDoseResponse.fit(training, rank=1),
        NearestSupportedDose.fit(training),
    )
    assert all(isinstance(baseline, SciPlex3RawCountBaseline) for baseline in baselines)
    assert {baseline.baseline_id for baseline in baselines} == {
        "matched-vehicle-resampling",
        "exact-condition-rep1-empirical-resampling",
        "exact-condition-negative-binomial",
        "hierarchical-well-negative-binomial",
        "low-rank-compound-dose-response",
        "nearest-supported-dose",
    }


def test_baseline_registry_and_suite_entrypoint_are_exact_and_immutable() -> None:
    expected = {
        "exact-condition-negative-binomial": ExactConditionNegativeBinomial,
        "exact-condition-rep1-empirical-resampling": ExactConditionRep1EmpiricalResampling,
        "hierarchical-well-negative-binomial": HierarchicalWellNegativeBinomial,
        "low-rank-compound-dose-response": LowRankCompoundDoseResponse,
        "matched-vehicle-resampling": MatchedVehicleResampling,
        "nearest-supported-dose": NearestSupportedDose,
    }
    assert dict(SCIPLEX3_BASELINE_IMPLEMENTATIONS) == expected
    with pytest.raises(TypeError):
        cast(dict[str, object], SCIPLEX3_BASELINE_IMPLEMENTATIONS)["extra"] = object

    suite = fit_sciplex3_baseline_suite(_training(), low_rank=1)
    assert set(suite) == set(expected)
    assert all(isinstance(item, SciPlex3RawCountBaseline) for item in suite.values())
    with pytest.raises(TypeError):
        cast(dict[str, object], suite)["extra"] = object


def test_every_baseline_represents_and_samples_no_action_cases_from_p1() -> None:
    training = _training()
    baselines: tuple[SciPlex3RawCountBaseline, ...] = (
        MatchedVehicleResampling.fit(training),
        ExactConditionRep1EmpiricalResampling.fit(training),
        ExactConditionNegativeBinomial.fit(training),
        HierarchicalWellNegativeBinomial.fit(training),
        LowRankCompoundDoseResponse.fit(training, rank=1),
        NearestSupportedDose.fit(training),
    )
    request = _request(NO_ACTION, sample_count=32, seed=4)
    for baseline in baselines:
        output = baseline.sample(request)
        assert output.target.condition is NO_ACTION
        assert output.samples.shape == (32, SCIPLEX3_FEATURE_COUNT)
        assert bool(np.all(np.sum(output.samples, axis=1, dtype=np.int64) > 0))


def test_frozen_p4_case_roles_are_all_representable_including_eight_controls() -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "benchmarks/vertical-a/sciplex3-k562-24h-v1/support/evaluation-cases.json"
    )
    raw_cases = cast(list[dict[str, object]], json.loads(path.read_text()))
    p4_cases = [case for case in raw_cases if case["partition_id"] == "p4-untouched-test"]
    targets: list[PredictionTarget] = []
    for case in p4_cases:
        actions = cast(list[str], case["intervention_spec_ids"])
        condition: TargetCondition = CompoundDose(actions[0], 1) if actions else NO_ACTION
        targets.append(
            PredictionTarget(
                case_id=cast(str, case["case_id"]),
                target_well_id=cast(str, case["evaluation_unit_id"]),
                plate_id=cast(str, case["matching_stratum_id"]),
                partition_id=cast(str, case["partition_id"]),
                condition=condition,
            )
        )
    assert len(targets) == 384
    assert sum(isinstance(target.condition, type(NO_ACTION)) for target in targets) == 8


def test_feature_manifest_hash_matches_the_authoritative_ordered_panel() -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "benchmarks/artifacts/sciplex3-k562-24h-v1/feature-panel.json"
    )
    artifact = cast(dict[str, object], json.loads(path.read_text()))
    features = cast(list[dict[str, object]], artifact["features"])
    keys = tuple(f"{item['ensembl_id']}|{item['gene_symbol']}" for item in features)
    training = P1TrainingData(keys, _training().wells)
    manifest = ExactConditionRep1EmpiricalResampling.fit(training).fitted_state_manifest()
    assert manifest["ordered_feature_keys_sha256"] == SCIPLEX3_ORDERED_FEATURE_KEYS_SHA256
    assert manifest["ordered_feature_keys_sha256"] == artifact["ordered_feature_keys_sha256"]


def test_matched_vehicle_resampling_is_seeded_equal_well_and_golden() -> None:
    baseline = MatchedVehicleResampling.fit(_training())
    request = _request(sample_count=12, seed=7)
    first = baseline.sample(request)
    second = baseline.sample(request)

    assert np.array_equal(first.samples, second.samples)
    assert first.samples[:, :3].tolist() == [
        [30, 1, 1],
        [1, 1, 2],
        [2, 1, 1],
        [31, 2, 1],
        [30, 1, 1],
        [1, 2, 1],
        [1, 2, 1],
        [1, 1, 2],
        [30, 1, 1],
        [30, 1, 1],
        [31, 2, 1],
        [1, 2, 1],
    ]
    assert first.baseline_id == baseline.baseline_id
    assert first.ordered_feature_keys == _keys()
    assert first.samples.dtype == np.int64
    assert not first.samples.flags.writeable

    many = baseline.sample(_request(sample_count=2_000, seed=8))
    second_well_fraction = float(np.mean(many.samples[:, 0] >= 30))
    assert 0.45 < second_well_fraction < 0.55


def test_requests_have_no_observation_or_comparator_input_surface() -> None:
    assert {item.name for item in fields(BaselineSampleRequest)} == {
        "target",
        "sample_count",
        "seed",
    }


def test_exact_empirical_is_seeded_and_uses_only_exact_p1_condition() -> None:
    baseline = ExactConditionRep1EmpiricalResampling.fit(_training())
    result = baseline.sample(_request(sample_count=10, seed=3))
    assert result.samples[:, :3].tolist() == [
        [8, 4, 1],
        [8, 4, 1],
        [8, 4, 1],
        [8, 4, 1],
        [8, 4, 1],
        [7, 2, 1],
        [9, 3, 2],
        [9, 3, 2],
        [9, 3, 2],
        [8, 4, 1],
    ]
    assert set(result.samples[:, 0]) <= {7, 8, 9}


def test_empirical_prediction_conditions_on_positive_panel_without_dropping_training_rows() -> None:
    vehicle = _well("vehicle", "plate", None, (1,))
    treated_counts = np.zeros((2, SCIPLEX3_FEATURE_COUNT), dtype=np.int64)
    treated_counts[1, 0] = 5
    treated = P1WellCounts(
        "treated",
        "plate",
        CompoundDose("drug", 10),
        ImmutableCSRCounts.from_dense(treated_counts),
        ("a-zero-panel-record", "b-positive-panel-record"),
        (1, 2),
    )
    training = P1TrainingData(_keys(), (vehicle, treated))
    baseline = ExactConditionRep1EmpiricalResampling.fit(training)
    output = baseline.sample(_request(CompoundDose("drug", 10), sample_count=512, seed=0))
    assert np.all(np.sum(output.samples, axis=1, dtype=np.int64) > 0)
    assert baseline.fitted_state_manifest()["fitted_state"]

    all_zero = P1WellCounts(
        "all-zero",
        "plate",
        CompoundDose("zero-drug", 10),
        ImmutableCSRCounts.from_dense(np.zeros((1, SCIPLEX3_FEATURE_COUNT), dtype=np.int64)),
        ("all-zero-record",),
        (3,),
    )
    zero_baseline = ExactConditionRep1EmpiricalResampling.fit(
        P1TrainingData(_keys(), (vehicle, all_zero))
    )
    with pytest.raises(SciPlex3BaselineError, match="bounded same-well"):
        zero_baseline.sample(_request(CompoundDose("zero-drug", 10), sample_count=1, seed=0))


@pytest.mark.parametrize(
    "baseline_factory",
    [
        ExactConditionRep1EmpiricalResampling.fit,
        ExactConditionNegativeBinomial.fit,
        HierarchicalWellNegativeBinomial.fit,
        lambda data: LowRankCompoundDoseResponse.fit(data, rank=1),
    ],
)
def test_exact_support_baselines_fail_missing_compound_and_dose(
    baseline_factory: Callable[[P1TrainingData], object],
) -> None:
    baseline = cast(SciPlex3RawCountBaseline, baseline_factory(_training()))
    with pytest.raises(SciPlex3BaselineError, match="compound support"):
        baseline.sample(_request(CompoundDose("unknown", 10)))
    with pytest.raises(SciPlex3BaselineError, match="dose support"):
        baseline.sample(_request(CompoundDose("compound-a", 20)))


def test_exact_negative_binomial_has_frozen_equal_well_smoothing_and_golden() -> None:
    training = _training()
    condition = CompoundDose("compound-a", 10)
    baseline = ExactConditionNegativeBinomial.fit(training)
    condition_well = next(well for well in training.wells if well.condition == condition)
    global_mean = np.mean(np.stack([well.counts.feature_mean() for well in training.wells]), axis=0)
    expected = (
        condition_well.counts.feature_mean() + EXACT_MEAN_GLOBAL_PSEUDO_WELLS * global_mean
    ) / (1.0 + EXACT_MEAN_GLOBAL_PSEUDO_WELLS)
    assert np.allclose(baseline.predictive_mean(condition), expected)
    assert not baseline.predictive_mean(condition).flags.writeable

    output = baseline.sample(_request(sample_count=8, seed=11))
    assert output.samples[:, :5].tolist() == [
        [10, 3, 2, 0, 0],
        [5, 2, 3, 0, 0],
        [4, 2, 1, 0, 0],
        [11, 2, 1, 0, 0],
        [8, 3, 1, 0, 0],
        [9, 5, 3, 0, 0],
        [7, 3, 0, 0, 0],
        [9, 3, 1, 0, 0],
    ]


def test_hierarchical_nb_uses_fixed_condition_compound_global_well_weights() -> None:
    training = _training()
    condition = CompoundDose("compound-a", 10)
    baseline = HierarchicalWellNegativeBinomial.fit(training)
    condition_wells = [well for well in training.wells if well.condition == condition]
    compound_wells = [
        well
        for well in training.wells
        if well.condition is not None and well.condition.compound == condition.compound
    ]

    def equal_well_mean(wells: list[P1WellCounts]) -> NDArray[np.float64]:
        return np.asarray(
            np.mean(np.stack([well.counts.feature_mean() for well in wells]), axis=0),
            dtype=np.float64,
        )

    expected = (
        HIERARCHICAL_CONDITION_WELLS * equal_well_mean(condition_wells)
        + HIERARCHICAL_COMPOUND_PSEUDO_WELLS * equal_well_mean(compound_wells)
        + HIERARCHICAL_GLOBAL_PSEUDO_WELLS * equal_well_mean(list(training.wells))
    ) / (
        HIERARCHICAL_CONDITION_WELLS
        + HIERARCHICAL_COMPOUND_PSEUDO_WELLS
        + HIERARCHICAL_GLOBAL_PSEUDO_WELLS
    )
    assert np.allclose(baseline.predictive_mean(condition), expected)
    result = baseline.sample(_request(sample_count=10, seed=23))
    assert result.samples.shape == (10, SCIPLEX3_FEATURE_COUNT)
    assert bool(np.all(np.sum(result.samples, axis=1) > 0))


def test_low_rank_full_reconstruction_recovers_condition_means() -> None:
    training = _training()
    baseline = LowRankCompoundDoseResponse.fit(training, rank=99)
    assert baseline.rank == 4
    for condition in (
        CompoundDose("compound-a", 10),
        CompoundDose("compound-a", 100),
        CompoundDose("compound-b", 10),
        CompoundDose("compound-b", 100),
    ):
        well = next(item for item in training.wells if item.condition == condition)
        assert np.allclose(
            baseline.predictive_mean(condition), well.counts.feature_mean(), atol=1e-10
        )
    result = baseline.sample(_request(sample_count=7, seed=17))
    assert result.samples.shape == (7, SCIPLEX3_FEATURE_COUNT)


def test_low_rank_response_validates_rank_condition_and_dose_support() -> None:
    training = _training()
    for rank in (0, -1, True):
        with pytest.raises(SciPlex3BaselineError, match="positive integer"):
            LowRankCompoundDoseResponse.fit(training, rank=rank)
    vehicle = _well("vehicle", "plate", None, (1, 1))
    one_condition = _well("treated", "plate", CompoundDose("drug", 10), (2, 1))
    with pytest.raises(SciPlex3BaselineError, match="at least two conditions"):
        LowRankCompoundDoseResponse.fit(P1TrainingData(_keys(), (vehicle, one_condition)))
    other = _well("other", "plate", CompoundDose("other-drug", 10), (1, 2))
    with pytest.raises(SciPlex3BaselineError, match="multiple dose supports"):
        LowRankCompoundDoseResponse.fit(P1TrainingData(_keys(), (vehicle, one_condition, other)))

    degenerate = P1TrainingData(
        _keys(),
        (
            _well("v", "p", None, (1, 1)),
            _well("a10", "p", CompoundDose("a", 10), (3, 1)),
            _well("a100", "p", CompoundDose("a", 100), (1, 3)),
            _well("b10", "p", CompoundDose("b", 10), (3, 1)),
            _well("b100", "p", CompoundDose("b", 100), (1, 3)),
        ),
    )
    with pytest.raises(SciPlex3BaselineError, match="degenerate singular-value tie"):
        LowRankCompoundDoseResponse.fit(degenerate, rank=1)


def test_nearest_supported_dose_uses_log_distance_and_lower_tie_break() -> None:
    baseline = NearestSupportedDose.fit(_training())
    assert baseline.supported_condition(CompoundDose("compound-a", 31)) == CompoundDose(
        "compound-a", 10
    )
    assert baseline.supported_condition(CompoundDose("compound-a", 32)) == CompoundDose(
        "compound-a", 100
    )
    assert baseline.supported_condition(CompoundDose("compound-a", 10)) == CompoundDose(
        "compound-a", 100
    )
    assert baseline.supported_condition(NO_ACTION) is NO_ACTION
    result = baseline.sample(_request(CompoundDose("compound-a", 31), sample_count=20, seed=5))
    assert set(result.samples[:, 0]) <= {7, 8, 9}
    with pytest.raises(SciPlex3BaselineError, match="compound support"):
        baseline.supported_condition(CompoundDose("absent", 10))

    vehicle = _well("vehicle", "plate", None, (1,))
    low = _well("low", "plate", CompoundDose("tied-drug", 10), (2,))
    high = _well("high", "plate", CompoundDose("tied-drug", 90), (3,))
    tied = NearestSupportedDose.fit(P1TrainingData(_keys(), (vehicle, low, high)))
    assert tied.supported_condition(CompoundDose("tied-drug", 30)) == CompoundDose("tied-drug", 10)
    one_dose = NearestSupportedDose.fit(P1TrainingData(_keys(), (vehicle, low)))
    with pytest.raises(SciPlex3BaselineError, match="alternate-dose support"):
        one_dose.supported_condition(CompoundDose("tied-drug", 10))


def test_predictive_output_rejects_zero_totals_wrong_count_and_is_immutable() -> None:
    valid = _counts((1,))
    output = PredictiveRawCountSamples("baseline", _target(), _keys(), 1, valid)
    valid[0, 0] = 10
    assert output.samples[0, 0] == 1
    with pytest.raises(SciPlex3BaselineError, match="baseline_id"):
        PredictiveRawCountSamples(" bad ", _target(), _keys(), 1, _counts((1,)))
    with pytest.raises(SciPlex3BaselineError, match="zero-total"):
        PredictiveRawCountSamples(
            "baseline",
            _target(),
            _keys(),
            1,
            np.zeros((1, SCIPLEX3_FEATURE_COUNT), dtype=np.int64),
        )
    with pytest.raises(SciPlex3BaselineError, match="rng_algorithm"):
        PredictiveRawCountSamples(
            "baseline",
            _target(),
            _keys(),
            1,
            _counts((1,)),
            rng_algorithm=cast(Literal["numpy-pcg64dxsm-v1"], "default-rng"),
        )


def test_nb_generation_deterministically_redraws_zero_total_panels() -> None:
    vehicle = _well("vehicle", "plate", None, (1,))
    treated = _well("treated", "plate", CompoundDose("drug", 10), (1,))
    baseline = ExactConditionNegativeBinomial.fit(P1TrainingData(_keys(), (vehicle, treated)))
    request = _request(CompoundDose("drug", 10), sample_count=512, seed=0)
    first = baseline.sample(request)
    second = baseline.sample(request)
    assert np.array_equal(first.samples, second.samples)
    assert bool(np.all(np.sum(first.samples, axis=1, dtype=np.int64) > 0))
    assert first.rng_algorithm == RNG_ALGORITHM


def test_nb_accepts_a_positive_panel_on_the_final_allowed_redraw(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def delayed_positive(
        mean: NDArray[np.float64],
        dispersion: NDArray[np.float64],
        *,
        sample_count: int,
        generator: np.random.Generator,
    ) -> IntArray:
        del mean, dispersion, generator
        nonlocal calls
        calls += 1
        result = np.zeros((sample_count, SCIPLEX3_FEATURE_COUNT), dtype=np.int64)
        if calls == MAX_ZERO_TOTAL_REDRAWS + 1:
            result[:, 0] = 1
        return result

    monkeypatch.setattr(baseline_module, "_draw_gamma_poisson", delayed_positive)
    result = baseline_module._sample_gamma_poisson(
        np.ones(SCIPLEX3_FEATURE_COUNT, dtype=np.float64),
        np.zeros(SCIPLEX3_FEATURE_COUNT, dtype=np.float64),
        sample_count=1,
        generator=np.random.Generator(np.random.PCG64DXSM(0)),
    )
    assert calls == MAX_ZERO_TOTAL_REDRAWS + 1
    assert result[0, 0] == 1
