from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import subprocess
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import MappingProxyType
from typing import cast

import numpy as np
import pytest

import cellstate.evaluation.sciplex3_candidate as candidate_module
from cellstate.evaluation.sciplex3_baselines import (
    NO_ACTION,
    SCIPLEX3_FEATURE_COUNT,
    CompoundDose,
    ImmutableCSRCounts,
    P1TrainingData,
    P1WellCounts,
    PredictionTarget,
)
from cellstate.evaluation.sciplex3_candidate import (
    SCIPLEX3_CANDIDATE_DOSES_NM,
    SCIPLEX3_CANDIDATE_FIXED_FACTOR_SHAPE,
    SCIPLEX3_CANDIDATE_GOLDEN_MODEL_SHA256,
    SCIPLEX3_CANDIDATE_GOLDEN_SAMPLE_SHA256,
    SCIPLEX3_CANDIDATE_MODEL_ID,
    SCIPLEX3_CANDIDATE_MODEL_SCHEMA_VERSION,
    SCIPLEX3_CANDIDATE_OUTPUT_MODEL_SCHEMA_SHA256,
    SCIPLEX3_CANDIDATE_SPECIFICATION_SHA256,
    SCIPLEX3_CANDIDATE_TAU_GRID,
    CandidateRawCountSamples,
    CandidateSampleRequest,
    SciPlex3CandidateError,
    SciPlex3CandidateInitialEquilibration,
    SciPlex3CandidateTraceEntry,
    SciPlex3GammaPoissonCandidate,
    SciPlex3P1ActionBinding,
    SciPlex3P1DesignBindings,
    SciPlex3P1VehicleBinding,
    build_sciplex3_synthetic_golden_candidate,
    candidate_golden_model_bytes,
    candidate_model_schema_manifest,
    candidate_specification_manifest,
    load_sciplex3_candidate,
    training_data_fingerprint,
    verify_sciplex3_candidate_golden,
)


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()


def _tensor_array(payload: dict[str, object], name: str, dtype: str) -> np.ndarray:
    tensors = cast(dict[str, object], payload["tensors"])
    manifest = cast(dict[str, object], tensors[name])
    shape = tuple(cast(list[int], manifest["shape"]))
    return np.frombuffer(base64.b64decode(cast(str, manifest["data_base64"])), dtype=dtype).reshape(
        shape
    )


def _set_tensor(payload: dict[str, object], name: str, value: np.ndarray, dtype: str) -> None:
    canonical = np.asarray(value, dtype=dtype, order="C")
    raw = canonical.tobytes(order="C")
    tensors = cast(dict[str, object], payload["tensors"])
    manifest = cast(dict[str, object], tensors[name])
    manifest["data_base64"] = base64.b64encode(raw).decode()
    manifest["sha256"] = hashlib.sha256(raw).hexdigest()
    manifest["shape"] = list(canonical.shape)


def _candidate() -> SciPlex3GammaPoissonCandidate:
    return load_sciplex3_candidate(
        candidate_golden_model_bytes(),
        expected_sha256=SCIPLEX3_CANDIDATE_GOLDEN_MODEL_SHA256,
    )


def _target(
    *,
    compound: str = "golden-compound-000",
    dose_nm: int = 10,
    partition_id: str = "p4-untouched-test",
    plate_id: str = "future-plate-a",
    case_id: str = "case-a",
    target_well_id: str = "well-a",
) -> PredictionTarget:
    return PredictionTarget(
        case_id=case_id,
        target_well_id=target_well_id,
        plate_id=plate_id,
        partition_id=partition_id,
        condition=CompoundDose(compound, dose_nm),
    )


def _request(
    *,
    compound: str = "golden-compound-000",
    dose_nm: int = 10,
    partition_id: str = "p4-untouched-test",
    plate_id: str = "future-plate-a",
    case_id: str = "case-a",
    target_well_id: str = "well-a",
) -> CandidateSampleRequest:
    return CandidateSampleRequest(
        target=_target(
            compound=compound,
            dose_nm=dose_nm,
            partition_id=partition_id,
            plate_id=plate_id,
            case_id=case_id,
            target_well_id=target_well_id,
        ),
        sample_count=24,
        seed=20260810,
    )


def _design() -> SciPlex3P1DesignBindings:
    plates = tuple(f"plate-{index}" for index in range(8))
    vehicles = tuple(
        SciPlex3P1VehicleBinding(
            plate_id=plate,
            well_ids=(f"{plate}-vehicle-b", f"{plate}-vehicle-a"),
        )
        for plate in plates
    )
    actions = []
    for compound_index in range(188):
        compound = f"compound-{compound_index:03d}"
        for dose_index, dose_nm in enumerate(SCIPLEX3_CANDIDATE_DOSES_NM):
            plate = plates[(compound_index + dose_index) % len(plates)]
            actions.append(
                SciPlex3P1ActionBinding(
                    compound=compound,
                    dose_nm=dose_nm,
                    well_id=f"{compound}-{dose_nm}",
                    plate_id=plate,
                )
            )
    return SciPlex3P1DesignBindings(tuple(reversed(actions)), tuple(reversed(vehicles)))


def _small_training(*, include_zero_row: bool) -> P1TrainingData:
    keys = tuple(f"feature-{index:04d}" for index in range(SCIPLEX3_FEATURE_COUNT))
    vehicle = np.zeros((2 if include_zero_row else 1, SCIPLEX3_FEATURE_COUNT), dtype=np.int64)
    vehicle[-1, :2] = (2, 1)
    treated = np.zeros((1, SCIPLEX3_FEATURE_COUNT), dtype=np.int64)
    treated[0, :2] = (4, 3)
    return P1TrainingData(
        keys,
        (
            P1WellCounts(
                "vehicle",
                "plate",
                None,
                ImmutableCSRCounts.from_dense(vehicle),
                tuple(f"vehicle-{index}" for index in range(len(vehicle))),
                tuple(range(len(vehicle))),
            ),
            P1WellCounts(
                "treated",
                "plate",
                CompoundDose("compound", 10),
                ImmutableCSRCounts.from_dense(treated),
                ("treated-0",),
                (100,),
            ),
        ),
    )


def _mini_full_topology() -> candidate_module._ValidatedTrainingDesign:
    design = _design()
    wells: list[P1WellCounts] = []
    source_index = 0
    for action_index, binding in enumerate(design.actions):
        record_ids: tuple[str, ...]
        source_rows: tuple[int, ...]
        if action_index == 0:
            counts = ImmutableCSRCounts(
                np.asarray([0, 0, 1]),
                np.asarray([0]),
                np.asarray([2]),
                2,
            )
            record_ids = (f"record-{source_index:06d}", f"record-{source_index + 1:06d}")
            source_rows = (source_index, source_index + 1)
            source_index += 2
        else:
            feature_index = action_index % SCIPLEX3_FEATURE_COUNT
            counts = ImmutableCSRCounts(
                np.asarray([0, 1]),
                np.asarray([feature_index]),
                np.asarray([1 + action_index % 3]),
                1,
            )
            record_ids = (f"record-{source_index:06d}",)
            source_rows = (source_index,)
            source_index += 1
        wells.append(
            P1WellCounts(
                binding.well_id,
                binding.plate_id,
                binding.condition,
                counts,
                record_ids,
                source_rows,
            )
        )
    for vehicle in design.vehicles:
        for well_id in vehicle.well_ids:
            feature_index = source_index % SCIPLEX3_FEATURE_COUNT
            wells.append(
                P1WellCounts(
                    well_id,
                    vehicle.plate_id,
                    None,
                    ImmutableCSRCounts(
                        np.asarray([0, 1]),
                        np.asarray([feature_index]),
                        np.asarray([2]),
                        1,
                    ),
                    (f"record-{source_index:06d}",),
                    (source_index,),
                )
            )
            source_index += 1
    action_indices = np.arange(752, dtype=np.int64).reshape(188, 4)
    plate_lookup = {plate: index for index, plate in enumerate(design.plate_ids)}
    action_plate_indices = np.asarray(
        [
            [plate_lookup[action.plate_id] for action in design.actions[index : index + 4]]
            for index in range(0, 752, 4)
        ],
        dtype=np.int64,
    )
    vehicle_indices = np.arange(752, 768, dtype=np.int64).reshape(8, 2)
    training_well_plate_indices = np.empty(768, dtype=np.int64)
    training_well_plate_indices[action_indices] = action_plate_indices
    for plate_index in range(8):
        training_well_plate_indices[vehicle_indices[plate_index]] = plate_index
    return candidate_module._ValidatedTrainingDesign(
        wells=tuple(wells),
        well_index_by_id=MappingProxyType(
            {well.well_id: index for index, well in enumerate(wells)}
        ),
        action_well_indices=action_indices,
        action_plate_indices=action_plate_indices,
        vehicle_well_indices=vehicle_indices,
        training_well_plate_indices=training_well_plate_indices,
        record_count=source_index,
        zero_panel_record_count=1,
    )


def _tiny_validated(*, count: int) -> candidate_module._ValidatedTrainingDesign:
    if count:
        counts = ImmutableCSRCounts(
            np.asarray([0, 1]),
            np.asarray([0]),
            np.asarray([count]),
            1,
        )
    else:
        counts = ImmutableCSRCounts(
            np.asarray([0, 0]),
            np.asarray([], dtype=np.int64),
            np.asarray([], dtype=np.int64),
            1,
        )
    well = P1WellCounts("tiny", "plate", None, counts, ("record",), (0,))
    return candidate_module._ValidatedTrainingDesign(
        wells=(well,),
        well_index_by_id=MappingProxyType({"tiny": 0}),
        action_well_indices=np.zeros((188, 4), dtype=np.int64),
        action_plate_indices=np.zeros((188, 4), dtype=np.int64),
        vehicle_well_indices=np.zeros((8, 2), dtype=np.int64),
        training_well_plate_indices=np.zeros(1, dtype=np.int64),
        record_count=1,
        zero_panel_record_count=int(count == 0),
    )


@pytest.fixture(scope="module")
def exact_sparse_training() -> tuple[P1TrainingData, SciPlex3P1DesignBindings]:
    design = _design()
    repository_root = Path(__file__).resolve().parents[1]
    panel = json.loads(
        (
            repository_root / "benchmarks/artifacts/sciplex3-k562-24h-v1/feature-panel.json"
        ).read_bytes()
    )
    keys = tuple(
        f"{feature['ensembl_id']}|{feature['gene_symbol']}" for feature in panel["features"]
    )
    conditions: dict[str, tuple[str, CompoundDose | None]] = {
        binding.well_id: (binding.plate_id, binding.condition) for binding in design.actions
    }
    for vehicle in design.vehicles:
        for well_id in vehicle.well_ids:
            conditions[well_id] = (vehicle.plate_id, None)
    ordered_well_ids = tuple(sorted(conditions))
    base_rows, remainder = divmod(94_785, 768)
    wells: list[P1WellCounts] = []
    source_offset = 0
    for well_index, well_id in enumerate(ordered_well_ids):
        row_count = base_rows + int(well_index < remainder)
        zero_rows = 7 if well_index == 0 else 0
        increments = np.ones(row_count, dtype=np.int64)
        increments[:zero_rows] = 0
        indptr = np.concatenate((np.asarray([0]), np.cumsum(increments)))
        nonzero_count = row_count - zero_rows
        feature_indices = np.full(nonzero_count, well_index % 2_000, dtype=np.int64)
        values = np.ones(nonzero_count, dtype=np.int64)
        plate_id, condition = conditions[well_id]
        wells.append(
            P1WellCounts(
                well_id,
                plate_id,
                condition,
                ImmutableCSRCounts(indptr, feature_indices, values, row_count),
                tuple(
                    f"record-{index:06d}"
                    for index in range(source_offset, source_offset + row_count)
                ),
                tuple(range(source_offset, source_offset + row_count)),
            )
        )
        source_offset += row_count
    assert source_offset == 94_785
    return P1TrainingData(keys, tuple(wells)), design


def test_frozen_specification_and_output_schema_have_exact_identities() -> None:
    assert hashlib.sha256(_canonical_bytes(candidate_specification_manifest())).hexdigest() == (
        SCIPLEX3_CANDIDATE_SPECIFICATION_SHA256
    )
    assert hashlib.sha256(_canonical_bytes(candidate_model_schema_manifest())).hexdigest() == (
        SCIPLEX3_CANDIDATE_OUTPUT_MODEL_SCHEMA_SHA256
    )
    specification = candidate_specification_manifest()
    assert specification["model_id"] == SCIPLEX3_CANDIDATE_MODEL_ID
    assert specification["model_schema_version"] == "4.0.0"
    assert (
        cast(dict[str, object], specification["distribution"])["observed_target_depth_conditioning"]
        is False
    )
    assert cast(dict[str, object], specification["fit"])["zero_panel_rows_retained"] is True
    reference_runtime = cast(dict[str, object], specification["reference_runtime"])
    assert reference_runtime["blas_name"] == "scipy-openblas"
    assert reference_runtime["blas_version"] == "0.3.31.188.0"
    action_model = cast(dict[str, object], specification["action_model"])
    assert action_model["rho_normalization"] == (
        "arithmetic-mean-over-eight-plates-equals-one-per-factor"
    )
    assert "log-arithmetic-mean-of-eight" in cast(str, action_model["alpha"])
    unseen_plate = cast(dict[str, object], specification["unseen_plate"])
    assert unseen_plate["family"] == "uniform-whole-p1-rho-row"
    assert unseen_plate["context_count"] == 8
    assert unseen_plate["factor_independent_draws"] is False
    assert unseen_plate["parametric_lognormal"] is False
    fit = cast(dict[str, object], specification["fit"])
    assert fit["factor_shape_mode"] == "fixed-not-estimated"
    assert fit["fixed_factor_shape"] == SCIPLEX3_CANDIDATE_FIXED_FACTOR_SHAPE
    assert "pooled_shape_solver" not in fit
    inner = cast(dict[str, object], fit["inner_equilibration"])
    assert inner["residual_tolerance"] == 1e-8
    assert inner["consecutive_passing_sweeps"] == 2
    assert inner["minimum_sweeps"] == 2
    assert inner["maximum_sweeps"] == 50
    schedule = cast(dict[str, object], fit["synchronized_schedule"])
    assert schedule["trace_one_relative_reference"] == "stored-initial-elbo-L0"
    uncertainty = cast(dict[str, object], specification["uncertainty_scope"])
    assert uncertainty == {
        "included": [
            "poisson-umi-shot-noise",
            "fixed-continuous-gamma-factor-activation-heterogeneity",
            "uniform-whole-p1-rho-row-unseen-plate-context",
        ],
        "excluded_or_unclaimed": [
            "technical-capture-attribution-q-removed",
            "fitted-parameter-or-model-uncertainty",
            "identifiable-action-specific-between-well-variance-one-p1-well-per-action",
            "intervention-realization",
            "viability-or-survival",
            "mechanism-pathway-or-cell-state-interpretation",
            "causal-or-transport-uncertainty",
        ],
    }
    claim_ceiling = cast(dict[str, object], specification["claim_ceiling"])
    assert claim_ceiling["factor_semantics"] == "statistical-assay-response-factors-only"
    assert claim_ceiling["never_interpret_as"] == [
        "cell-types",
        "pathways",
        "hidden-cell-state",
    ]
    assert claim_ceiling["predictive_status"] == ("uncalibrated-p1-fit-predictive-association-only")
    assert claim_ceiling["population_scope"] == "recovered-k562-nuclei"
    assert claim_ceiling["time_scope"] == "24-hours"
    assert claim_ceiling["support_scope"] == "exact-panel-actions-and-doses-only"
    assert claim_ceiling["claims_not_made"] == [
        "causality",
        "intervention-realization",
        "viability-or-survival",
        "mechanism-or-pathway",
        "novel-action-dose-time-system-or-transport",
    ]
    calibration = cast(dict[str, object], specification["calibration_declaration_only"])
    assert calibration["tau_grid"] == list(SCIPLEX3_CANDIDATE_TAU_GRID)
    assert calibration["tau_grid_exponents"] == list(range(-20, 7))
    assert 0.1 / SCIPLEX3_CANDIDATE_TAU_GRID[-1] ** 2 > 0.05


def test_v4_family_has_fixed_shape_inner_witnesses_and_empirical_plate_context() -> None:
    candidate = _candidate()
    payload = json.loads(candidate.canonical_model_bytes())
    tensors = cast(dict[str, object], payload["tensors"])
    behavior = candidate.behavior_manifest()
    distribution = cast(dict[str, object], candidate_specification_manifest()["distribution"])

    assert type(candidate.initial_equilibration) is SciPlex3CandidateInitialEquilibration
    assert all(type(item) is SciPlex3CandidateTraceEntry for item in candidate.trace)
    assert candidate.implementation_version == "4.0.0"
    assert candidate.factor_shape.hex() == "0x1.999999999999ap-4"
    assert set(tensors) == {
        "action_well_indices",
        "alpha",
        "basis",
        "delta",
        "factor_contributions",
        "factor_shape",
        "mean_activation",
        "rho",
        "training_well_plate_indices",
        "vehicle_well_indices",
    }
    assert "capture_shape" not in tensors
    assert "factor_shapes" not in tensors
    assert set(behavior) == {
        "all_parameters_finite",
        "can_mint_lifecycle_evidence",
        "capture_latent_present",
        "factor_contribution_shares",
        "factor_order_stable",
        "factor_shape_estimated",
        "factor_shape_mode",
        "final_elbo",
        "fixed_factor_shape",
        "fit_converged",
        "heldout_memberships_read",
        "heldout_outcomes_read",
        "initial_elbo",
        "initial_equilibration_sha256",
        "initial_factor_order",
        "initial_inner_sweep_count_histogram",
        "initial_maximum_inner_sweeps",
        "initial_maximum_terminal_elog_residual",
        "initial_maximum_terminal_shape_residual",
        "inner_all_batches_converged",
        "inner_batch_count",
        "inner_equilibration_performed",
        "loading_rank_ratio",
        "maximum_inner_sweeps",
        "maximum_terminal_elog_residual",
        "maximum_terminal_shape_residual",
        "mean_activation_rank_ratio",
        "minimum_factor_contribution_share",
        "model_schema_version",
        "outer_iteration_count",
        "plate_context_count",
        "plate_context_factorwise_mean_one",
        "plate_context_family",
        "scientifically_admissible",
        "terminal_elbo_relative_changes",
        "training_partition_ids",
    }
    assert behavior["capture_latent_present"] is False
    assert behavior["can_mint_lifecycle_evidence"] is False
    assert behavior["scientifically_admissible"] is False
    assert behavior["heldout_memberships_read"] is False
    assert behavior["heldout_outcomes_read"] is False
    assert behavior["factor_shape_estimated"] is False
    assert behavior["factor_shape_mode"] == "fixed"
    assert behavior["fixed_factor_shape"] == candidate.factor_shape
    assert behavior["factor_order_stable"] is True
    assert behavior["inner_equilibration_performed"] is True
    assert behavior["inner_all_batches_converged"] is True
    assert behavior["plate_context_family"] == "uniform-whole-p1-rho-row"
    assert behavior["plate_context_count"] == 8
    assert behavior["plate_context_factorwise_mean_one"] is True
    assert behavior["model_schema_version"] == SCIPLEX3_CANDIDATE_MODEL_SCHEMA_VERSION
    shares = cast(list[float], behavior["factor_contribution_shares"])
    assert math.fsum(shares) == pytest.approx(1.0, rel=0.0, abs=2e-16)
    assert behavior["minimum_factor_contribution_share"] == min(shares)
    assert distribution["capture_multiplier"] == "fixed-one-no-random-variable"
    assert distribution["factor_shape"] == "fixed-r_theta-0.1-not-estimated"
    assert candidate.fitted_state_manifest()["model_schema_version"] == "4.0.0"
    assert set(payload["initial_equilibration"]) == {
        "elbo",
        "factor_order",
        "inner_sweep_count_histogram",
        "maximum_inner_sweeps",
        "maximum_terminal_elog_residual",
        "maximum_terminal_shape_residual",
    }


def test_loader_rejects_stale_v1_through_v3_identities_and_removed_tensors() -> None:
    for stale_identity in (
        "sciplex3-gamma-poisson-candidate-model-v1",
        "sciplex3-gamma-poisson-pooled-factor-candidate-model-v2",
        "sciplex3-gamma-poisson-fixed-factor-candidate-model-v3",
    ):
        stale_schema = json.loads(candidate_golden_model_bytes())
        stale_schema["model_schema"] = stale_identity
        with pytest.raises(SciPlex3CandidateError, match="identity or scientific specification"):
            SciPlex3GammaPoissonCandidate.from_canonical_model_bytes(_canonical_bytes(stale_schema))

    for removed_name in ("capture_shape", "factor_shapes", "plate_sigma"):
        stale_tensor = json.loads(candidate_golden_model_bytes())
        stale_tensor["tensors"][removed_name] = stale_tensor["tensors"]["factor_shape"]
        with pytest.raises(SciPlex3CandidateError, match="keys differ"):
            SciPlex3GammaPoissonCandidate.from_canonical_model_bytes(_canonical_bytes(stale_tensor))

    missing_version = json.loads(candidate_golden_model_bytes())
    del missing_version["model_schema_version"]
    with pytest.raises(SciPlex3CandidateError, match="keys differ"):
        SciPlex3GammaPoissonCandidate.from_canonical_model_bytes(_canonical_bytes(missing_version))

    for wrong_version in ("3.0.0", 4, True):
        wrong = json.loads(candidate_golden_model_bytes())
        wrong["model_schema_version"] = wrong_version
        with pytest.raises(SciPlex3CandidateError, match="identity or scientific specification"):
            SciPlex3GammaPoissonCandidate.from_canonical_model_bytes(_canonical_bytes(wrong))

    schema = candidate_model_schema_manifest()
    assert schema["model_schema_version"] == "4.0.0"
    assert "model_schema_version" in cast(list[str], schema["required_model_keys"])


def test_design_bindings_are_exact_sorted_and_deeply_immutable() -> None:
    design = _design()
    assert len(design.actions) == 752
    assert len(design.compounds) == 188
    assert design.plate_ids == tuple(sorted(design.plate_ids))
    assert design.vehicles[0].well_ids == tuple(sorted(design.vehicles[0].well_ids))
    assert len(design.fingerprint) == 64
    with pytest.raises(FrozenInstanceError):
        design.context_id = "different"  # type: ignore[misc]


def test_design_rejects_missing_action_duplicate_vehicle_and_context_drift() -> None:
    design = _design()
    with pytest.raises(SciPlex3CandidateError, match="752"):
        SciPlex3P1DesignBindings(design.actions[:-1], design.vehicles)
    with pytest.raises(SciPlex3CandidateError, match="two vehicle"):
        SciPlex3P1VehicleBinding("plate", ("same", "same"))
    with pytest.raises(SciPlex3CandidateError, match="context"):
        SciPlex3P1DesignBindings(design.actions, design.vehicles, "another-context")


def test_canonical_model_round_trip_is_byte_identical_and_deeply_frozen() -> None:
    candidate = _candidate()
    model_bytes = candidate.canonical_model_bytes()
    reloaded = SciPlex3GammaPoissonCandidate.from_canonical_model_bytes(model_bytes)

    assert reloaded.canonical_model_bytes() == model_bytes
    assert reloaded.model_artifact_sha256 == SCIPLEX3_CANDIDATE_GOLDEN_MODEL_SHA256
    assert reloaded.behavior_manifest()["fit_converged"] is True
    assert reloaded.behavior_manifest()["all_parameters_finite"] is True
    assert not reloaded._basis.flags.writeable
    assert not reloaded._delta.flags.writeable
    assert not reloaded._action_well_indices.flags.writeable
    assert not reloaded._training_well_plate_indices.flags.writeable
    assert not reloaded._mean_activation.flags.writeable
    with pytest.raises(ValueError):
        reloaded._basis.setflags(write=True)
    with pytest.raises(ValueError):
        reloaded._delta[0, 0, 0] = 99.0


def test_exact_loader_authenticates_external_digest_before_parsing() -> None:
    payload = candidate_golden_model_bytes()
    with pytest.raises(SciPlex3CandidateError, match="externally bound"):
        load_sciplex3_candidate(payload, expected_sha256="0" * 64)
    with pytest.raises(SciPlex3CandidateError, match="lowercase SHA"):
        load_sciplex3_candidate(payload, expected_sha256="invalid")
    with pytest.raises(SciPlex3CandidateError, match="immutable bytes"):
        SciPlex3GammaPoissonCandidate.from_canonical_model_bytes(bytearray(payload))  # type: ignore[arg-type]


def test_loader_rejects_noncanonical_json_unknown_fields_and_tensor_drift() -> None:
    payload = json.loads(candidate_golden_model_bytes())
    with pytest.raises(SciPlex3CandidateError, match="canonical JSON"):
        SciPlex3GammaPoissonCandidate.from_canonical_model_bytes(
            json.dumps(payload, sort_keys=True).encode()
        )

    extra = json.loads(candidate_golden_model_bytes())
    extra["unexpected"] = True
    with pytest.raises(SciPlex3CandidateError, match="keys differ"):
        SciPlex3GammaPoissonCandidate.from_canonical_model_bytes(_canonical_bytes(extra))

    drift = json.loads(candidate_golden_model_bytes())
    tensor = drift["tensors"]["basis"]
    raw = bytearray(base64.b64decode(tensor["data_base64"]))
    raw[0] ^= 1
    tensor["data_base64"] = base64.b64encode(raw).decode()
    with pytest.raises(SciPlex3CandidateError, match="SHA-256 mismatch"):
        SciPlex3GammaPoissonCandidate.from_canonical_model_bytes(_canonical_bytes(drift))


def test_loader_rejects_authenticated_nonfinite_tensor_and_wrong_shape() -> None:
    nonfinite = json.loads(candidate_golden_model_bytes())
    tensor = nonfinite["tensors"]["alpha"]
    raw = bytearray(base64.b64decode(tensor["data_base64"]))
    raw[:8] = np.asarray([np.nan], dtype="<f8").tobytes()
    tensor["data_base64"] = base64.b64encode(raw).decode()
    tensor["sha256"] = hashlib.sha256(raw).hexdigest()
    with pytest.raises(SciPlex3CandidateError, match="nonfinite"):
        SciPlex3GammaPoissonCandidate.from_canonical_model_bytes(_canonical_bytes(nonfinite))

    wrong_shape = json.loads(candidate_golden_model_bytes())
    wrong_shape["tensors"]["rho"]["shape"] = [16, 8]
    with pytest.raises(SciPlex3CandidateError, match="dtype or shape"):
        SciPlex3GammaPoissonCandidate.from_canonical_model_bytes(_canonical_bytes(wrong_shape))


def test_loader_recomputes_trace_and_rejects_forgery_nonconvergence_and_decrease() -> None:
    forged = json.loads(candidate_golden_model_bytes())
    forged["trace"][1]["relative_change"] += 1e-6
    with pytest.raises(SciPlex3CandidateError, match="forged relative"):
        SciPlex3GammaPoissonCandidate.from_canonical_model_bytes(_canonical_bytes(forged))

    nonconvergent = json.loads(candidate_golden_model_bytes())
    replacement = (-92.9, -92.8, -92.7)
    for index, elbo in zip(range(7, 10), replacement, strict=True):
        previous = nonconvergent["trace"][index - 1]["elbo"]
        nonconvergent["trace"][index]["elbo"] = elbo
        nonconvergent["trace"][index]["relative_change"] = abs(elbo - previous) / max(
            1.0, abs(previous)
        )
    nonconvergent["behavior"]["final_elbo"] = replacement[-1]
    with pytest.raises(SciPlex3CandidateError, match="convergence streak"):
        SciPlex3GammaPoissonCandidate.from_canonical_model_bytes(_canonical_bytes(nonconvergent))

    decreasing = json.loads(candidate_golden_model_bytes())
    previous = decreasing["trace"][4]["elbo"]
    decreasing["trace"][5]["elbo"] = -94.0
    decreasing["trace"][5]["relative_change"] = abs(-94.0 - previous) / max(1.0, abs(previous))
    with pytest.raises(SciPlex3CandidateError, match="material ELBO decrease"):
        SciPlex3GammaPoissonCandidate.from_canonical_model_bytes(_canonical_bytes(decreasing))


def test_loader_recomputes_initial_relative_inner_witnesses_and_terminal_order() -> None:
    forged_initial = json.loads(candidate_golden_model_bytes())
    forged_initial["initial_equilibration"]["elbo"] -= 1.0
    with pytest.raises(SciPlex3CandidateError, match="forged relative"):
        SciPlex3GammaPoissonCandidate.from_canonical_model_bytes(_canonical_bytes(forged_initial))
    unstable = json.loads(candidate_golden_model_bytes())
    unstable["trace"][-1]["factor_order"][0:2] = [1, 0]
    with pytest.raises(SciPlex3CandidateError, match="stable factor-order"):
        SciPlex3GammaPoissonCandidate.from_canonical_model_bytes(_canonical_bytes(unstable))

    wrong_fixed_shape = json.loads(candidate_golden_model_bytes())
    shape = _tensor_array(wrong_fixed_shape, "factor_shape", "<f8").copy()
    shape[0] = np.nextafter(shape[0], np.inf)
    _set_tensor(
        wrong_fixed_shape,
        "factor_shape",
        shape,
        "<f8",
    )
    with pytest.raises(SciPlex3CandidateError, match=r"bit-exact fixed 0\.1"):
        SciPlex3GammaPoissonCandidate.from_canonical_model_bytes(
            _canonical_bytes(wrong_fixed_shape)
        )

    wrong_histogram = json.loads(candidate_golden_model_bytes())
    wrong_histogram["trace"][0]["inner_sweep_count_histogram"][1] -= 1
    with pytest.raises(SciPlex3CandidateError, match="disagree on the p1 batch count"):
        SciPlex3GammaPoissonCandidate.from_canonical_model_bytes(_canonical_bytes(wrong_histogram))

    failed_residual = json.loads(candidate_golden_model_bytes())
    failed_residual["trace"][0]["maximum_terminal_elog_residual"] = 1.1e-8
    with pytest.raises(SciPlex3CandidateError, match="pass the inner tolerance"):
        SciPlex3GammaPoissonCandidate.from_canonical_model_bytes(_canonical_bytes(failed_residual))


def test_loader_reconstructs_closed_world_topology_and_mean_activation_exactly() -> None:
    duplicate = json.loads(candidate_golden_model_bytes())
    action_indices = _tensor_array(duplicate, "action_well_indices", "<i8").copy()
    action_indices[0, 0] = action_indices[0, 1]
    _set_tensor(duplicate, "action_well_indices", action_indices, "<i8")
    with pytest.raises(SciPlex3CandidateError, match="partition all 768"):
        SciPlex3GammaPoissonCandidate.from_canonical_model_bytes(_canonical_bytes(duplicate))

    wrong_vehicle_plate = json.loads(candidate_golden_model_bytes())
    plate_indices = _tensor_array(wrong_vehicle_plate, "training_well_plate_indices", "<i8").copy()
    vehicle_indices = _tensor_array(wrong_vehicle_plate, "vehicle_well_indices", "<i8")
    plate_indices[int(vehicle_indices[0, 0])] = 1
    _set_tensor(wrong_vehicle_plate, "training_well_plate_indices", plate_indices, "<i8")
    with pytest.raises(SciPlex3CandidateError, match="vehicle topology"):
        SciPlex3GammaPoissonCandidate.from_canonical_model_bytes(
            _canonical_bytes(wrong_vehicle_plate)
        )

    activation_drift = json.loads(candidate_golden_model_bytes())
    activation = _tensor_array(activation_drift, "mean_activation", "<f8").copy()
    activation[0, 0] = np.nextafter(activation[0, 0], np.inf)
    _set_tensor(activation_drift, "mean_activation", activation, "<f8")
    with pytest.raises(SciPlex3CandidateError, match="differs from the sealed p1 topology"):
        SciPlex3GammaPoissonCandidate.from_canonical_model_bytes(_canonical_bytes(activation_drift))

    contribution_drift = json.loads(candidate_golden_model_bytes())
    contribution = _tensor_array(contribution_drift, "factor_contributions", "<f8").copy()
    contribution[0] = np.nextafter(contribution[0], np.inf)
    _set_tensor(contribution_drift, "factor_contributions", contribution, "<f8")
    with pytest.raises(SciPlex3CandidateError, match="equal-well predictive means"):
        SciPlex3GammaPoissonCandidate.from_canonical_model_bytes(
            _canonical_bytes(contribution_drift)
        )

    reordered_ids = json.loads(candidate_golden_model_bytes())
    reordered_ids["training_well_ids"][0:2] = reversed(reordered_ids["training_well_ids"][0:2])
    with pytest.raises(SciPlex3CandidateError, match="canonically sorted"):
        SciPlex3GammaPoissonCandidate.from_canonical_model_bytes(_canonical_bytes(reordered_ids))


def test_rho_remains_a_positive_normalized_empirical_context_tensor() -> None:
    stale_context = json.loads(candidate_golden_model_bytes())
    rho = _tensor_array(stale_context, "rho", "<f8").copy()
    rho[0, 0] += 1e-6
    _set_tensor(stale_context, "rho", rho, "<f8")
    with pytest.raises(SciPlex3CandidateError, match="arithmetic mean one"):
        SciPlex3GammaPoissonCandidate.from_canonical_model_bytes(_canonical_bytes(stale_context))

    nonpositive = json.loads(candidate_golden_model_bytes())
    rho = _tensor_array(nonpositive, "rho", "<f8").copy()
    rho[0, 0] = 0.0
    _set_tensor(nonpositive, "rho", rho, "<f8")
    with pytest.raises(SciPlex3CandidateError, match="strictly positive"):
        SciPlex3GammaPoissonCandidate.from_canonical_model_bytes(_canonical_bytes(nonpositive))

    exact_bound = json.loads(candidate_golden_model_bytes())
    rho = _tensor_array(exact_bound, "rho", "<f8").copy()
    rho[:, 0] = np.finfo(np.float64).eps
    rho[0, 0] = 8.0
    _set_tensor(exact_bound, "rho", rho, "<f8")
    with pytest.raises(SciPlex3CandidateError, match="strictly less than eight"):
        SciPlex3GammaPoissonCandidate.from_canonical_model_bytes(_canonical_bytes(exact_bound))

    just_below = np.ones((8, 16), dtype=np.float64)
    just_below[0, 0] = np.nextafter(8.0, -np.inf)
    just_below[1:, 0] = (8.0 - just_below[0, 0]) / 7.0
    accepted = candidate_module._power_normalized_plate_contexts(just_below, 1.0)
    assert np.array_equal(accepted, just_below)


def test_audit_matrix_contributions_and_rank_ratios_are_portably_canonical() -> None:
    candidate = _candidate()
    raw = candidate_module._reconstruct_mean_activation(
        candidate._alpha,
        candidate._rho,
        candidate._delta,
        candidate._training_well_plate_indices,
        candidate._action_well_indices,
        candidate._vehicle_well_indices,
    )
    canonical = candidate_module._canonical_audit_matrix(raw)
    assert np.array_equal(canonical, candidate._mean_activation)
    manual = np.asarray(
        [
            math.fsum(float(canonical[row, factor]) for row in range(768)) / 768.0
            for factor in range(16)
        ]
    )
    assert np.array_equal(candidate_module._factor_contributions(raw), manual)
    behavior = candidate.behavior_manifest()
    for name in ("loading_rank_ratio", "mean_activation_rank_ratio"):
        ratio = cast(float, behavior[name])
        assert ratio == round(ratio, 12)
        assert ratio > (
            candidate_module.SCIPLEX3_CANDIDATE_IDENTIFIABILITY_THRESHOLD
            + candidate_module.SCIPLEX3_CANDIDATE_IDENTIFIABILITY_MARGIN
        )
    with pytest.raises(SciPlex3CandidateError, match="ambiguous quantization boundary"):
        candidate_module._canonical_quantized_scalar(1.5e-12, name="test scalar")


def test_rounded_identifiability_gates_loading_activation_and_contribution_share() -> None:
    loading_rank = json.loads(candidate_golden_model_bytes())
    basis = _tensor_array(loading_rank, "basis", "<f8").copy()
    basis[:] = basis[0]
    _set_tensor(loading_rank, "basis", basis, "<f8")
    with pytest.raises(SciPlex3CandidateError, match="loading matrix fails"):
        SciPlex3GammaPoissonCandidate.from_canonical_model_bytes(_canonical_bytes(loading_rank))

    activation_rank = json.loads(candidate_golden_model_bytes())
    alpha = np.log(np.linspace(2.5, 0.75, 16, dtype=np.float64))
    rho = np.ones((8, 16), dtype=np.float64)
    delta = np.zeros((188, 4, 16), dtype=np.float64)
    mean_activation = candidate_module._canonical_audit_matrix(np.tile(np.exp(alpha), (768, 1)))
    contributions = candidate_module._factor_contributions(mean_activation)
    _set_tensor(activation_rank, "alpha", alpha, "<f8")
    _set_tensor(activation_rank, "rho", rho, "<f8")
    _set_tensor(activation_rank, "delta", delta, "<f8")
    _set_tensor(activation_rank, "mean_activation", mean_activation, "<f8")
    _set_tensor(activation_rank, "factor_contributions", contributions, "<f8")
    with pytest.raises(SciPlex3CandidateError, match="mean activation matrix fails"):
        SciPlex3GammaPoissonCandidate.from_canonical_model_bytes(_canonical_bytes(activation_rank))

    tiny_share = json.loads(candidate_golden_model_bytes())
    alpha = _tensor_array(tiny_share, "alpha", "<f8").copy()
    rho = _tensor_array(tiny_share, "rho", "<f8")
    delta = _tensor_array(tiny_share, "delta", "<f8")
    plate_indices = _tensor_array(tiny_share, "training_well_plate_indices", "<i8")
    action_indices = _tensor_array(tiny_share, "action_well_indices", "<i8")
    vehicle_indices = _tensor_array(tiny_share, "vehicle_well_indices", "<i8")
    alpha[-1] = math.log(1e-9)
    mean_activation = candidate_module._canonical_audit_matrix(
        candidate_module._reconstruct_mean_activation(
            alpha,
            rho,
            delta,
            plate_indices,
            action_indices,
            vehicle_indices,
        )
    )
    _set_tensor(tiny_share, "alpha", alpha, "<f8")
    _set_tensor(tiny_share, "mean_activation", mean_activation, "<f8")
    _set_tensor(
        tiny_share,
        "factor_contributions",
        candidate_module._factor_contributions(mean_activation),
        "<f8",
    )
    with pytest.raises(SciPlex3CandidateError, match="contribution shares fail"):
        SciPlex3GammaPoissonCandidate.from_canonical_model_bytes(_canonical_bytes(tiny_share))


def test_canonical_factor_ties_use_loading_digest_and_duplicate_keys_fail() -> None:
    candidate = _candidate()
    basis = candidate._basis.copy()
    contributions = candidate._factor_contributions.copy()
    contributions[1] = contributions[0]
    order = candidate_module._canonical_factor_order(basis, contributions)
    digests = [
        hashlib.sha256(
            np.asarray(np.round(basis[index], 12), dtype="<f8").tobytes(order="C")
        ).hexdigest()
        for index in (0, 1)
    ]
    assert order[:2].tolist() == sorted((0, 1), key=lambda index: digests[index])

    basis[1] = basis[0]
    with pytest.raises(SciPlex3CandidateError, match="duplicate canonical factor keys"):
        candidate_module._canonical_factor_order(basis, contributions)


def test_sampling_is_deterministic_positive_raw_int64_and_generates_future_total() -> None:
    candidate = _candidate()
    request = _request()
    first = cast(CandidateRawCountSamples, candidate.sample(request))
    second = cast(CandidateRawCountSamples, candidate.sample(request))

    assert first.candidate_id == SCIPLEX3_CANDIDATE_MODEL_ID
    assert first.samples.dtype == np.dtype("int64")
    assert np.array_equal(first.samples, second.samples)
    totals = np.sum(first.samples, axis=1, dtype=np.int64)
    assert bool(np.all(totals > 0))
    assert np.unique(totals).size > 1
    assert not first.samples.flags.writeable


def test_count_substreams_are_case_local_while_whole_plate_context_is_shared() -> None:
    candidate = _candidate()
    first_request = _request(case_id="case-a", target_well_id="well-a")
    second_request = _request(case_id="case-b", target_well_id="well-b")
    first = cast(CandidateRawCountSamples, candidate.sample(first_request))
    second = cast(CandidateRawCountSamples, candidate.sample(second_request))

    assert np.array_equal(
        candidate._plate_context_row(first_request.target, first_request.seed),
        candidate._plate_context_row(second_request.target, second_request.seed),
    )
    assert candidate._plate_context_index(
        first_request.target, first_request.seed
    ) == candidate._plate_context_index(second_request.target, second_request.seed)
    other_action = _request(
        compound="golden-compound-001",
        dose_nm=100,
        case_id="case-c",
        target_well_id="well-c",
    )
    assert np.array_equal(
        candidate._plate_context_row(first_request.target, first_request.seed),
        candidate._plate_context_row(other_action.target, other_action.seed),
    )
    assert not np.array_equal(first.samples, second.samples)


def test_empirical_plate_context_rows_are_reachable_mean_one_bounded_and_not_normal() -> None:
    candidate = _candidate()
    assert np.allclose(np.mean(candidate._rho, axis=0), 1.0, rtol=0.0, atol=5e-13)
    assert bool(np.all(candidate._rho > 0.0))
    assert bool(np.all(candidate._rho < 8.0))
    target = _target(plate_id="future-plate-reachability")
    reached = {candidate._plate_context_index(target, seed) for seed in range(10_000)}
    assert reached == set(range(8))
    for seed in range(64):
        index = candidate._plate_context_index(target, seed)
        assert np.array_equal(candidate._plate_context_row(target, seed), candidate._rho[index])
    mutated_payload = json.loads(candidate.canonical_model_bytes())
    mutated_payload["training"]["training_data_sha256"] = "0" * 64
    mutated = SciPlex3GammaPoissonCandidate.from_canonical_model_bytes(
        _canonical_bytes(mutated_payload)
    )
    assert any(
        candidate._plate_context_index(target, seed) != mutated._plate_context_index(target, seed)
        for seed in range(64)
    )
    assert ".normal(" not in Path(candidate_module.__file__).read_text()


def test_declared_tau_power_transform_is_mean_preserving_and_guarded() -> None:
    candidate = _candidate()
    tau_one = candidate_module._power_normalized_plate_contexts(candidate._rho, 1.0)
    assert np.array_equal(tau_one, candidate._rho)
    for tau in SCIPLEX3_CANDIDATE_TAU_GRID:
        contexts = candidate_module._power_normalized_plate_contexts(candidate._rho, tau)
        assert bool(np.all(contexts > 0.0))
        assert bool(np.all(contexts < 8.0))
        for factor_index in range(16):
            assert math.fsum(float(value) for value in contexts[:, factor_index]) / 8.0 == (
                pytest.approx(1.0, rel=0.0, abs=5e-13)
            )
        assert candidate_module._factor_shape_for_tau(tau) > 0.05
    with pytest.raises(SciPlex3CandidateError, match="exact declared"):
        candidate_module._factor_shape_for_tau(math.exp(7.0 / 20.0))


@pytest.mark.parametrize(
    "target",
    [
        _target(compound="unsupported"),
        _target(dose_nm=11),
        _target(partition_id="p1-train"),
    ],
)
def test_unsupported_action_dose_or_context_fails_closed(target: PredictionTarget) -> None:
    candidate = _candidate()
    assert not candidate.supports(target)
    with pytest.raises(SciPlex3CandidateError, match="unsupported"):
        candidate.sample(CandidateSampleRequest(target, 2, 0))


def test_no_action_is_supported_but_only_in_declared_target_context() -> None:
    candidate = _candidate()
    target = PredictionTarget(
        "case", "well", "future-plate", "p3-model-selection-validation", NO_ACTION
    )
    assert candidate.supports(target)
    result = cast(CandidateRawCountSamples, candidate.sample(CandidateSampleRequest(target, 3, 2)))
    assert result.samples.shape == (3, SCIPLEX3_FEATURE_COUNT)


def test_wrong_request_and_result_provenance_are_rejected() -> None:
    candidate = _candidate()
    with pytest.raises(SciPlex3CandidateError, match="exact CandidateSampleRequest"):
        candidate.sample(object())
    valid = cast(CandidateRawCountSamples, candidate.sample(_request()))
    with pytest.raises(SciPlex3CandidateError, match="model ID"):
        CandidateRawCountSamples(
            "baseline-like-id",
            valid.target,
            valid.ordered_feature_keys,
            valid.seed,
            valid.samples,
        )


def test_golden_fixture_has_fixed_model_and_sample_identities() -> None:
    candidate = build_sciplex3_synthetic_golden_candidate()
    assert SCIPLEX3_CANDIDATE_GOLDEN_MODEL_SHA256 == (
        "d3a5cb630ad4344bda04945a865682531860508ccb8473fa57fb2397c8297be1"
    )
    assert SCIPLEX3_CANDIDATE_GOLDEN_SAMPLE_SHA256 == (
        "6e01b44440c04ccc0ae1540de57d26c3723ff0b12f7e8033d4c980818776fa9f"
    )
    assert candidate.model_artifact_sha256 == SCIPLEX3_CANDIDATE_GOLDEN_MODEL_SHA256
    assert hashlib.sha256(candidate.golden_sample().samples.tobytes()).hexdigest() == (
        SCIPLEX3_CANDIDATE_GOLDEN_SAMPLE_SHA256
    )
    assert verify_sciplex3_candidate_golden(candidate)

    mutated = json.loads(candidate.canonical_model_bytes())
    mutated["training"]["training_data_sha256"] = "0" * 64
    mutated_bytes = _canonical_bytes(mutated)
    mutated_candidate = SciPlex3GammaPoissonCandidate.load_exact(
        mutated_bytes, expected_sha256=hashlib.sha256(mutated_bytes).hexdigest()
    )
    assert not verify_sciplex3_candidate_golden(mutated_candidate)


def test_candidate_import_does_not_eagerly_construct_the_golden_fixture() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    prior_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = str(repository_root / "src")
    if prior_pythonpath:
        environment["PYTHONPATH"] += os.pathsep + prior_pythonpath
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import numpy as np\n"
                "def forbidden(*args, **kwargs):\n"
                "    raise RuntimeError('golden fixture constructed during import')\n"
                "np.sin = forbidden\n"
                "import cellstate.evaluation.sciplex3_candidate as candidate\n"
                "print(candidate.SCIPLEX3_CANDIDATE_GOLDEN_MODEL_SHA256)\n"
            ),
        ],
        cwd=repository_root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert probe.returncode == 0, probe.stderr
    assert probe.stdout.strip() == SCIPLEX3_CANDIDATE_GOLDEN_MODEL_SHA256


def test_training_fingerprint_retains_zero_rows_and_is_order_independent_by_well_id() -> None:
    with_zero = _small_training(include_zero_row=True)
    without_zero = _small_training(include_zero_row=False)
    assert training_data_fingerprint(with_zero) != training_data_fingerprint(without_zero)
    reordered = P1TrainingData(with_zero.ordered_feature_keys, tuple(reversed(with_zero.wells)))
    assert training_data_fingerprint(with_zero) == training_data_fingerprint(reordered)
    with pytest.raises(SciPlex3CandidateError, match="exact P1TrainingData"):
        training_data_fingerprint(cast(P1TrainingData, object()))


def test_frozen_dose_newton_is_deterministic_and_shape_estimation_paths_are_absent() -> None:
    baseline = np.log(np.asarray([1.0, 1.1, 0.9, 1.2], dtype=np.float64))
    means = np.asarray([1.2, 1.4, 1.1, 1.8], dtype=np.float64)
    initial = np.log(means) - baseline
    first = candidate_module._fit_dose_block(
        baseline, means, SCIPLEX3_CANDIDATE_FIXED_FACTOR_SHAPE, initial
    )
    second = candidate_module._fit_dose_block(
        baseline, means, SCIPLEX3_CANDIDATE_FIXED_FACTOR_SHAPE, initial
    )
    assert np.array_equal(first, second)
    assert np.all(np.isfinite(first))
    assert not hasattr(candidate_module, "_solve_gamma_shape")
    assert not hasattr(candidate_module, "_shape_statistics")


def test_action_block_receives_only_the_bit_exact_fixed_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validated = _mini_full_topology()
    scores = np.tile(np.linspace(2.0, 0.5, 16), (768, 1))
    observed_shapes: list[float] = []

    def dose_block(
        _baseline: np.ndarray,
        _means: np.ndarray,
        shape: float,
        initial: np.ndarray,
    ) -> np.ndarray:
        observed_shapes.append(shape)
        return initial

    monkeypatch.setattr(candidate_module, "_fit_dose_block", dose_block)
    candidate_module._update_action_block(scores, validated)
    assert len(observed_shapes) == 188 * 16
    assert {value.hex() for value in observed_shapes} == {"0x1.999999999999ap-4"}


def test_dose_newton_numerical_stationarity_is_exactly_bounded() -> None:
    objective = 100.0
    threshold = (
        candidate_module.SCIPLEX3_CANDIDATE_BLOCK_DECREMENT_EPS_MULTIPLIER
        * np.finfo(np.float64).eps
        * objective
    )
    assert candidate_module._dose_numerically_stationary(threshold, objective)
    assert not candidate_module._dose_numerically_stationary(
        np.nextafter(threshold, np.inf), objective
    )
    assert not candidate_module._dose_numerically_stationary(-1.0, objective)
    assert not candidate_module._dose_numerically_stationary(np.inf, objective)
    action_spec = cast(dict[str, object], candidate_specification_manifest()["action_model"])
    stationary = cast(dict[str, object], action_spec["newton_numerical_stationarity"])
    assert stationary["decrement_epsilon_multiplier"] == 64.0


def test_real_preflight_dose_block_accepts_only_frozen_numerical_stationarity_route() -> None:
    baseline = np.asarray(
        [-0.22813592586453113, -0.22813592586453113, -16.226261752433263, -16.226261752433263]
    )
    means = np.asarray(
        [8.78793098505065e-08, 1.1874999948046873e-07, 9.708333290859374e-08, 2.5035882787842283]
    )
    initial = np.asarray(
        [-16.019165516913258, -15.718109472542128, 0.07856562732365191, 17.143986766746117]
    )
    fitted = candidate_module._fit_dose_block(baseline, means, 1.0, initial)
    assert np.allclose(
        fitted,
        [-8.57759897, -1.9922019, 5.73818128, 15.25658825],
        rtol=0.0,
        atol=5e-9,
    )
    assert candidate_module._dose_objective(fitted, baseline, means, 1.0) == pytest.approx(
        29.28071187699736, rel=0.0, abs=1e-13
    )


def test_nndsvd_relative_score_floor_is_positive_mass_preserving_and_does_not_floor_b() -> None:
    scores = np.zeros((768, 16), dtype=np.float64)
    scores[:, 0] = np.linspace(1.0, 2.0, 768)
    row_totals = np.sum(scores, axis=1)
    basis = np.zeros((16, 2_000), dtype=np.float64)
    basis[:, 0] = 1.0

    positive_scores, unchanged_basis = candidate_module._regularize_nndsvd_initialization(
        scores, basis, row_totals
    )

    assert bool(np.all(positive_scores > 0.0))
    assert np.allclose(np.sum(positive_scores, axis=1), row_totals, rtol=1e-15, atol=0.0)
    assert np.array_equal(unchanged_basis, basis)
    assert bool(
        np.all(
            candidate_module.SCIPLEX3_CANDIDATE_DIRICHLET_CONCENTRATION
            + candidate_module.SCIPLEX3_CANDIDATE_LAMBDA_INITIAL_MASS * unchanged_basis
            > 0.0
        )
    )
    fit_spec = cast(dict[str, object], candidate_specification_manifest()["fit"])
    smoothing = cast(dict[str, object], fit_spec["initialization_score_smoothing"])
    assert smoothing["floor_fraction_of_well_total_per_factor"] == 1e-8
    assert smoothing["loading_floor"] is False


def test_topology_validator_counts_zero_rows_instead_of_filtering_them() -> None:
    training = _small_training(include_zero_row=True)
    with pytest.raises(SciPlex3CandidateError, match="768"):
        candidate_module._validate_training_design(training, _design())


def test_compact_full_topology_executes_sparse_cavi_elbo_and_factor_canonicalization() -> None:
    validated = _mini_full_topology()
    factor_scale = np.linspace(2.0, 0.5, 16, dtype=np.float64)
    well_scores = np.tile(factor_scale, (768, 1))
    for compound_index in range(188):
        for dose_index in range(4):
            well_index = int(validated.action_well_indices[compound_index, dose_index])
            well_scores[well_index] *= np.exp(0.01 * (dose_index - 1.5) * np.linspace(0.5, 1.0, 16))

    alpha, rho, delta, well_means = candidate_module._update_action_block(well_scores, validated)
    assert np.allclose(np.mean(rho, axis=0), 1.0, rtol=0.0, atol=1e-15)
    assert np.all(np.isfinite(delta))
    assert np.array_equal(
        candidate_module._well_factor_means(alpha, rho, delta, validated), well_means
    )

    state = candidate_module._initialize_local_state(validated, well_scores)
    loading_concentration = np.full((16, 2_000), 0.3, dtype=np.float64)
    for factor_index in range(16):
        loading_concentration[factor_index, factor_index] += 1_000.0
    sufficient = candidate_module._cavi_pass(
        validated,
        state,
        loading_concentration,
        well_means,
    )
    assert sufficient.loading_counts.shape == (16, 2_000)
    assert np.sum(sufficient.loading_counts) > 0.0
    assert np.all(state.theta_shape[0] == SCIPLEX3_CANDIDATE_FIXED_FACTOR_SHAPE)
    assert np.any(state.theta_shape[1] > SCIPLEX3_CANDIDATE_FIXED_FACTOR_SHAPE)
    assert np.allclose(
        state.theta_rate[0],
        SCIPLEX3_CANDIDATE_FIXED_FACTOR_SHAPE / well_means[0] + 1.0,
        rtol=0.0,
        atol=0.0,
    )
    assert 2 <= sufficient.maximum_inner_sweeps <= 50
    assert sufficient.maximum_terminal_shape_residual <= 1e-8
    assert sufficient.maximum_terminal_elog_residual <= 1e-8
    assert len(sufficient.inner_sweep_count_histogram) == 50
    assert sufficient.inner_sweep_count_histogram[0] == 0
    assert sum(sufficient.inner_sweep_count_histogram) == 768
    assert not hasattr(state, "capture_shape")

    updated_loading = 0.3 + sufficient.loading_counts
    next_alpha, next_rho, next_delta, next_means = candidate_module._update_action_block(
        sufficient.well_theta_means, validated
    )
    assert np.all(np.isfinite(next_alpha))
    assert np.allclose(np.mean(next_rho, axis=0), 1.0, rtol=0.0, atol=1e-15)
    elbo = candidate_module._elbo(
        validated,
        state,
        sufficient,
        updated_loading,
        next_means,
        next_delta,
    )
    assert np.isfinite(elbo)
    basis = updated_loading / np.sum(updated_loading, axis=1, keepdims=True)
    contributions = candidate_module._factor_contributions(next_means)
    order = candidate_module._canonical_factor_order(basis, contributions)
    assert sorted(order.tolist()) == list(range(16))


def test_inner_equilibration_preserves_zero_rows_and_requires_two_passing_sweeps() -> None:
    validated = _tiny_validated(count=0)
    means = np.ones((1, 16), dtype=np.float64)
    state = candidate_module._initialize_local_state(validated, means)
    loading = np.full((16, 2_000), 0.3, dtype=np.float64)
    sufficient = candidate_module._cavi_pass(validated, state, loading, means)

    assert np.array_equal(
        state.theta_shape,
        np.full((1, 16), SCIPLEX3_CANDIDATE_FIXED_FACTOR_SHAPE),
    )
    assert np.array_equal(
        state.theta_rate,
        np.full((1, 16), SCIPLEX3_CANDIDATE_FIXED_FACTOR_SHAPE + 1.0),
    )
    assert sufficient.maximum_inner_sweeps == 2
    assert sufficient.inner_sweep_count_histogram[1] == 1
    assert sufficient.maximum_terminal_shape_residual == 0.0
    assert sufficient.maximum_terminal_elog_residual == 0.0
    assert np.sum(sufficient.loading_counts) == 0.0


def test_inner_equilibration_conserves_counts_and_fails_closed_at_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validated = _tiny_validated(count=5)
    means = np.ones((1, 16), dtype=np.float64)
    loading = np.full((16, 2_000), 0.3, dtype=np.float64)
    state = candidate_module._initialize_local_state(validated, means)
    sufficient = candidate_module._cavi_pass(validated, state, loading, means)
    omega = 94_785.0 / 768.0
    assert math.fsum(float(value) for value in sufficient.loading_counts.ravel()) == (
        pytest.approx(omega * 5.0, rel=0.0, abs=2e-12)
    )
    assert math.fsum(float(value) for value in state.theta_shape[0]) == pytest.approx(
        16.0 * SCIPLEX3_CANDIDATE_FIXED_FACTOR_SHAPE + 5.0,
        rel=0.0,
        abs=2e-15,
    )

    state = candidate_module._initialize_local_state(validated, means)
    monkeypatch.setattr(candidate_module, "SCIPLEX3_CANDIDATE_MAX_INNER_SWEEPS", 2)
    with pytest.raises(SciPlex3CandidateError, match="failed to converge within 50 sweeps"):
        candidate_module._cavi_pass(validated, state, loading, means)


def test_global_equal_well_weighted_mass_reconciliation_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validated = _tiny_validated(count=5)
    means = np.ones((1, 16), dtype=np.float64)
    loading = np.full((16, 2_000), 0.3, dtype=np.float64)
    state = candidate_module._initialize_local_state(validated, means)
    original_bincount = np.bincount

    def drifting_bincount(
        values: np.ndarray, *, weights: np.ndarray | None = None, minlength: int = 0
    ) -> np.ndarray:
        result = original_bincount(values, weights=weights, minlength=minlength)
        if minlength == SCIPLEX3_FEATURE_COUNT:
            result = result.copy()
            result[0] += 1e-3
        return result

    monkeypatch.setattr(np, "bincount", drifting_bincount)
    with pytest.raises(SciPlex3CandidateError, match="global equal-well-weighted count mass"):
        candidate_module._cavi_pass(validated, state, loading, means)


def test_internal_numeric_helpers_reject_degenerate_inputs() -> None:
    with pytest.raises(SciPlex3CandidateError, match="positivity-floor inputs"):
        candidate_module._regularize_nndsvd_initialization(
            np.zeros((1, 16)), np.zeros((16, 2_000)), np.ones(1)
        )
    with pytest.raises(SciPlex3CandidateError, match="dose block inputs"):
        candidate_module._fit_dose_block(np.zeros(4), np.zeros(4), 1.0, np.zeros(4))
    with pytest.raises(SciPlex3CandidateError, match="exact declared"):
        candidate_module._power_normalized_plate_contexts(np.ones((8, 16)), np.nan)


def test_exact_sparse_topology_validation_succeeds_and_retains_seven_zero_rows(
    exact_sparse_training: tuple[P1TrainingData, SciPlex3P1DesignBindings],
) -> None:
    training, design = exact_sparse_training
    validated = candidate_module._validate_training_design(training, design)
    assert validated.record_count == 94_785
    assert validated.zero_panel_record_count == 7
    assert validated.action_well_indices.shape == (188, 4)
    assert validated.vehicle_well_indices.shape == (8, 2)
    assert len(set(validated.action_well_indices.ravel().tolist())) == 752


def test_real_shape_nndsvd_initialization_is_deterministic_positive_and_total_preserving() -> None:
    generator = np.random.Generator(np.random.PCG64DXSM(12))
    left = generator.random((768, 20))
    right = generator.random((20, 2_000))
    well_means = left @ right
    first_scores, first_basis = candidate_module._nndsvd_initialization(well_means)
    second_scores, second_basis = candidate_module._nndsvd_initialization(well_means)
    assert np.array_equal(first_scores, second_scores)
    assert np.array_equal(first_basis, second_basis)
    assert bool(np.all(first_scores > 0.0))
    assert np.allclose(
        np.sum(first_scores, axis=1), np.sum(well_means, axis=1), rtol=1e-15, atol=0.0
    )
    assert np.allclose(np.sum(first_basis, axis=1), 1.0, rtol=0.0, atol=1e-15)


def test_fitter_orchestration_converges_seals_real_summary_and_reloads(
    exact_sparse_training: tuple[P1TrainingData, SciPlex3P1DesignBindings],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    training, design = exact_sparse_training
    golden = build_sciplex3_synthetic_golden_candidate()
    validated = candidate_module._validate_training_design(training, design)
    basis = golden._basis.copy()
    alpha = golden._alpha.copy()
    rho = golden._rho.copy()
    delta = golden._delta.copy()
    well_means = candidate_module._reconstruct_mean_activation(
        alpha,
        rho,
        delta,
        validated.training_well_plate_indices,
        validated.action_well_indices,
        validated.vehicle_well_indices,
    )
    scores = well_means.copy()
    loading_counts = 1_000.0 * basis
    elbos = iter(
        (
            -110.0,
            -100.0,
            -95.0,
            -93.5,
            -93.1,
            -93.01,
            -93.001,
            -93.000010,
            -93.000005,
            -93.000001,
            -93.0,
        )
    )
    histogram = (0, 768, *([0] * 48))
    call_order: list[str] = []

    def action_block(*_: object) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        call_order.append("M")
        return alpha.copy(), rho.copy(), delta.copy(), well_means.copy()

    def cavi_pass(*_: object) -> candidate_module._PassSufficientStatistics:
        call_order.append("E")
        return candidate_module._PassSufficientStatistics(
            loading_counts=loading_counts.copy(),
            well_theta_means=well_means.copy(),
            allocation_entropy=0.0,
            poisson_factorial=0.0,
            theta_count_elog=0.0,
            maximum_inner_sweeps=2,
            maximum_terminal_shape_residual=1e-12,
            maximum_terminal_elog_residual=2e-12,
            inner_sweep_count_histogram=histogram,
        )

    def elbo(*_: object) -> float:
        call_order.append("L")
        return next(elbos)

    monkeypatch.setattr(
        candidate_module,
        "_nndsvd_initialization",
        lambda _: (scores.copy(), basis.copy()),
    )
    monkeypatch.setattr(
        candidate_module,
        "_update_action_block",
        action_block,
    )
    monkeypatch.setattr(candidate_module, "_cavi_pass", cavi_pass)
    monkeypatch.setattr(candidate_module, "_elbo", elbo)
    monkeypatch.setattr(
        candidate_module,
        "_factor_contributions",
        lambda means: np.mean(means, axis=0, dtype=np.float64),
    )

    candidate = candidate_module._fit_sciplex3_candidate_exact(training, design)
    assert candidate.training_summary.provenance == "real-p1"
    assert candidate.training_summary.record_count == 94_785
    assert candidate.training_summary.zero_panel_record_count == 7
    assert candidate.behavior_manifest()["outer_iteration_count"] == 10
    assert candidate.initial_equilibration.elbo == -110.0
    assert candidate.trace[0].relative_change == pytest.approx(10.0 / 110.0)
    assert call_order[:3] == ["M", "E", "L"]
    assert call_order[3:] == ["M", "E", "L"] * 10
    payload = candidate.canonical_model_bytes()
    assert (
        SciPlex3GammaPoissonCandidate.load_exact(
            payload, expected_sha256=hashlib.sha256(payload).hexdigest()
        ).canonical_model_bytes()
        == payload
    )


def test_public_fit_entrypoints_preserve_exact_class(monkeypatch: pytest.MonkeyPatch) -> None:
    golden = _candidate()
    monkeypatch.setattr(candidate_module, "_fit_sciplex3_candidate_exact", lambda *_: golden)
    training = cast(P1TrainingData, object())
    design = cast(SciPlex3P1DesignBindings, object())
    assert SciPlex3GammaPoissonCandidate.fit(training, design) is golden
    assert candidate_module.fit_sciplex3_candidate(training, design) is golden
    monkeypatch.setattr(candidate_module, "_fit_sciplex3_candidate_exact", lambda *_: object())
    with pytest.raises(SciPlex3CandidateError, match="substituted"):
        SciPlex3GammaPoissonCandidate.fit(training, design)
