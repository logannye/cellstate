from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import subprocess
import sys
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from types import MappingProxyType
from typing import cast

import numpy as np
import pytest

import cellstate.evaluation.sciplex3_candidate as candidate_module
import cellstate.evaluation.sciplex3_sampling_v5 as sampling_v5_module
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
    SCIPLEX3_CANDIDATE_V5_NEUTRAL_CONTEXT_ID,
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
from cellstate.evaluation.sciplex3_candidate_v5 import (
    SciPlex3V5Design,
    fixed_q_full_elbo_action_context,
)
from cellstate.evaluation.sciplex3_sampling_v5 import (
    SCIPLEX3_V5_MAX_SAMPLE_COUNT,
    SCIPLEX3_V5_SAMPLING_CONTRACT_SHA256,
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
    assert specification["model_schema_version"] == "5.0.0"
    assert (
        cast(dict[str, object], specification["distribution"])["observed_target_depth_conditioning"]
        is False
    )
    distribution = cast(dict[str, object], specification["distribution"])
    assert distribution["positive_panel_conditioning"] == (
        "exact-zero-truncated-compound-poisson-log-series"
    )
    assert distribution["positive_panel_rejection_redraws"] is False
    support = cast(dict[str, object], specification["support"])
    assert support["maximum_samples_per_request"] == SCIPLEX3_V5_MAX_SAMPLE_COUNT
    assert support["sampling_contract_sha256"] == SCIPLEX3_V5_SAMPLING_CONTRACT_SHA256
    assert support["supports_argument"] == "exact-CandidateSampleRequest-not-target-only"
    assert cast(dict[str, object], specification["fit"])["zero_panel_rows_retained"] is True
    reference_runtime = cast(dict[str, object], specification["reference_runtime"])
    assert reference_runtime["blas_name"] == "scipy-openblas"
    assert reference_runtime["blas_version"] == "0.3.31.188.0"
    action_model = cast(dict[str, object], specification["action_model"])
    assert action_model["rho_normalization"] == (
        "arithmetic-mean-over-eight-plates-equals-one-per-factor"
    )
    assert action_model["alpha"] == "factorwise-logmeanexp-of-eight-fitted-plate-intercepts"
    unseen_plate = cast(dict[str, object], specification["unseen_plate"])
    assert unseen_plate["family"] == "neutral-unit-context"
    assert unseen_plate["context_count"] == 1
    assert unseen_plate["context_id"] == SCIPLEX3_CANDIDATE_V5_NEUTRAL_CONTEXT_ID
    assert unseen_plate["factor_multiplier"] == 1.0
    assert unseen_plate["deterministic"] is True
    assert unseen_plate["p1_rho_sampling_input"] is False
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
        ],
        "excluded_or_unclaimed": [
            "technical-capture-attribution-q-removed",
            "fitted-parameter-or-model-uncertainty",
            "identifiable-action-specific-between-well-variance-one-p1-well-per-action",
            "unseen-plate-context-variation-neutralized",
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
    assert calibration["active_training_candidate_tau"] == 1.0
    assert calibration["unseen_plate_context_transform"] == "unit-context-invariant-under-tau"
    assert 0.1 / SCIPLEX3_CANDIDATE_TAU_GRID[-1] ** 2 > 0.05


def test_active_family_has_fixed_shape_inner_witnesses_and_v5_sampling_support() -> None:
    candidate = build_sciplex3_synthetic_golden_candidate()
    payload = json.loads(candidate.canonical_model_bytes())
    tensors = cast(dict[str, object], payload["tensors"])
    behavior = candidate.behavior_manifest()
    distribution = cast(dict[str, object], candidate_specification_manifest()["distribution"])

    assert type(candidate.initial_equilibration) is SciPlex3CandidateInitialEquilibration
    assert all(type(item) is SciPlex3CandidateTraceEntry for item in candidate.trace)
    assert candidate.implementation_version == "5.0.0"
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
        "sampling_active_calibration_state_sha256",
        "sampling_contract_sha256",
        "sampling_envelope_combination_count",
        "sampling_envelope_maximum_compound_poisson_intensity",
        "sampling_envelope_maximum_request_count",
        "sampling_envelope_rejection_reasons",
        "sampling_envelope_request_failure_budget_log",
        "sampling_envelope_supported",
        "sampling_envelope_worst_request_tail_log_upper_bound",
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
    assert behavior["plate_context_family"] == "neutral-unit-context"
    assert behavior["plate_context_count"] == 1
    assert behavior["plate_context_factorwise_mean_one"] is True
    assert behavior["sampling_contract_sha256"] == SCIPLEX3_V5_SAMPLING_CONTRACT_SHA256
    assert behavior["sampling_envelope_combination_count"] == 753 * 1 * 27
    assert behavior["sampling_envelope_maximum_request_count"] == 512
    assert behavior["sampling_envelope_rejection_reasons"] == []
    assert behavior["sampling_envelope_supported"] is True
    assert cast(float, behavior["sampling_envelope_worst_request_tail_log_upper_bound"]) < 0.0
    assert len(cast(str, behavior["sampling_active_calibration_state_sha256"])) == 64
    assert behavior["model_schema_version"] == SCIPLEX3_CANDIDATE_MODEL_SCHEMA_VERSION
    shares = cast(list[float], behavior["factor_contribution_shares"])
    assert math.fsum(shares) == pytest.approx(1.0, rel=0.0, abs=2e-16)
    assert behavior["minimum_factor_contribution_share"] == min(shares)
    assert distribution["capture_multiplier"] == "fixed-one-no-random-variable"
    assert distribution["factor_shape"] == "fixed-r_theta-0.1-not-estimated"
    assert candidate.fitted_state_manifest()["model_schema_version"] == "5.0.0"
    assert set(payload["initial_equilibration"]) == {
        "elbo",
        "factor_order",
        "inner_sweep_count_histogram",
        "maximum_inner_sweeps",
        "maximum_terminal_elog_residual",
        "maximum_terminal_shape_residual",
    }


def test_loader_rejects_stale_v1_through_v4_identities_and_removed_tensors() -> None:
    for stale_identity in (
        "sciplex3-gamma-poisson-candidate-model-v1",
        "sciplex3-gamma-poisson-pooled-factor-candidate-model-v2",
        "sciplex3-gamma-poisson-fixed-factor-candidate-model-v3",
        "sciplex3-gamma-poisson-fixed-r0p1-empirical-plate-candidate-model-v4",
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

    for wrong_version in ("3.0.0", "4.0.0", 5, True):
        wrong = json.loads(candidate_golden_model_bytes())
        wrong["model_schema_version"] = wrong_version
        with pytest.raises(SciPlex3CandidateError, match="identity or scientific specification"):
            SciPlex3GammaPoissonCandidate.from_canonical_model_bytes(_canonical_bytes(wrong))

    schema = candidate_model_schema_manifest()
    assert schema["model_schema_version"] == "5.0.0"
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


def test_rho_remains_a_positive_normalized_training_nuisance_tensor() -> None:
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
    candidate = build_sciplex3_synthetic_golden_candidate()
    request = _request()
    first = cast(CandidateRawCountSamples, candidate.sample(request))
    second = cast(CandidateRawCountSamples, candidate.sample(request))

    assert first.candidate_id == SCIPLEX3_CANDIDATE_MODEL_ID
    assert first.samples.dtype == np.dtype("int64")
    assert np.array_equal(first.samples, second.samples)
    totals = np.asarray([sum(int(value) for value in row) for row in first.samples])
    assert bool(np.all(totals > 0))
    assert np.unique(totals).size > 1
    assert not first.samples.flags.writeable
    assert first.model_artifact_sha256 == candidate.model_artifact_sha256
    assert first.sampling_contract_sha256 == SCIPLEX3_V5_SAMPLING_CONTRACT_SHA256
    assert first.target_fingerprint == candidate_module._v5_target_fingerprint(request.target)
    assert first.context_id == SCIPLEX3_CANDIDATE_V5_NEUTRAL_CONTEXT_ID
    assert len(first.calibration_state_sha256) == 64


def test_sampling_envelope_is_built_once_then_reused_by_support_and_sample(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    construction_count = 0
    original_builder = sampling_v5_module.build_sampling_envelope_certificate

    def counted_builder(
        parameters: sampling_v5_module.V5SamplingParameters,
    ) -> sampling_v5_module.V5SamplingEnvelopeCertificate:
        nonlocal construction_count
        construction_count += 1
        return original_builder(parameters)

    monkeypatch.setattr(
        sampling_v5_module,
        "build_sampling_envelope_certificate",
        counted_builder,
    )
    candidate = build_sciplex3_synthetic_golden_candidate()
    sampler = candidate._v5_runtime_sampler()
    request = _request()

    assert construction_count == 1
    assert sampler is candidate._v5_runtime_sampler()
    assert sampler.parameters is candidate._v5_sampling_parameters_cache
    assert sampler.envelope_certificate is candidate._v5_sampling_envelope_certificate_cache
    assert sampler.parameters.model_artifact_sha256 == candidate.model_artifact_sha256
    assert candidate.supports(request)
    candidate.sample(request)
    candidate.behavior_manifest()
    assert construction_count == 1


def test_candidate_construction_rejects_an_unsafe_global_sampling_envelope() -> None:
    candidate = build_sciplex3_synthetic_golden_candidate()
    shifted_alpha = candidate._alpha + 46.0
    shifted_activation = candidate_module._canonical_audit_matrix(
        candidate_module._reconstruct_mean_activation(
            shifted_alpha,
            candidate._rho,
            candidate._delta,
            candidate._training_well_plate_indices,
            candidate._action_well_indices,
            candidate._vehicle_well_indices,
        )
    )

    with pytest.raises(SciPlex3CandidateError, match="complete v5 sampling envelope"):
        replace(
            candidate,
            _alpha=shifted_alpha,
            _mean_activation=shifted_activation,
            _factor_contributions=candidate_module._factor_contributions(shifted_activation),
        )


def test_count_substreams_are_case_local_and_prefix_stable() -> None:
    candidate = build_sciplex3_synthetic_golden_candidate()
    first_request = _request(case_id="case-a", target_well_id="well-a")
    second_request = _request(case_id="case-b", target_well_id="well-b")
    first = cast(CandidateRawCountSamples, candidate.sample(first_request))
    second = cast(CandidateRawCountSamples, candidate.sample(second_request))
    longer = cast(
        CandidateRawCountSamples,
        candidate.sample(
            CandidateSampleRequest(
                target=first_request.target,
                sample_count=first_request.sample_count + 11,
                seed=first_request.seed,
            )
        ),
    )

    assert np.array_equal(first.samples, longer.samples[: first_request.sample_count])
    assert not np.array_equal(first.samples, second.samples)
    assert first.context_id == second.context_id == SCIPLEX3_CANDIDATE_V5_NEUTRAL_CONTEXT_ID


def test_unseen_plate_context_is_one_neutral_row_for_every_tau_and_never_uses_rho() -> None:
    candidate = build_sciplex3_synthetic_golden_candidate()
    parameters = candidate._v5_sampling_parameters(model_artifact_sha256="0" * 64)

    assert parameters.context_ids == (SCIPLEX3_CANDIDATE_V5_NEUTRAL_CONTEXT_ID,)
    assert parameters.context_multipliers.shape == (27, 1, 16)
    assert np.array_equal(parameters.context_multipliers, np.ones((27, 1, 16)))
    assert not np.shares_memory(parameters.context_multipliers, candidate._rho)
    assert ".normal(" not in Path(candidate_module.__file__).read_text()


def test_declared_tau_states_change_only_shape_under_neutral_sampling_context() -> None:
    candidate = build_sciplex3_synthetic_golden_candidate()
    parameters = candidate._v5_sampling_parameters(model_artifact_sha256="0" * 64)

    assert parameters.calibration_taus == SCIPLEX3_CANDIDATE_TAU_GRID
    assert parameters.active_tau == 1.0
    for index, tau in enumerate(SCIPLEX3_CANDIDATE_TAU_GRID):
        assert parameters.factor_shapes[index] == candidate_module._factor_shape_for_tau(tau)
        assert np.array_equal(parameters.context_multipliers[index], np.ones((1, 16)))
        assert parameters.factor_shapes[index] > 0.05
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
    candidate = build_sciplex3_synthetic_golden_candidate()
    assert not candidate.supports(target)
    request = CandidateSampleRequest(target, 2, 0)
    assert not candidate.supports(request)
    with pytest.raises(SciPlex3CandidateError, match="unsupported"):
        candidate.sample(request)


def test_no_action_is_supported_but_only_in_declared_target_context() -> None:
    candidate = build_sciplex3_synthetic_golden_candidate()
    target = PredictionTarget(
        "case", "well", "future-plate", "p3-model-selection-validation", NO_ACTION
    )
    request = CandidateSampleRequest(target, 3, 2)
    assert not candidate.supports(target)
    assert candidate.supports(request)
    result = cast(CandidateRawCountSamples, candidate.sample(request))
    assert result.samples.shape == (3, SCIPLEX3_FEATURE_COUNT)


def test_sample_count_512_is_supported_and_513_fails_before_sampling() -> None:
    candidate = build_sciplex3_synthetic_golden_candidate()
    target = _target()

    assert candidate.supports(CandidateSampleRequest(target, 512, 0))
    oversized = CandidateSampleRequest(target, 513, 0)
    assert not candidate.supports(oversized)
    with pytest.raises(SciPlex3CandidateError, match="count"):
        candidate.sample(oversized)


def test_wrong_request_and_result_provenance_are_rejected() -> None:
    candidate = build_sciplex3_synthetic_golden_candidate()
    with pytest.raises(SciPlex3CandidateError, match="exact CandidateSampleRequest"):
        candidate.sample(object())
    valid = cast(CandidateRawCountSamples, candidate.sample(_request()))
    with pytest.raises(SciPlex3CandidateError, match="model ID"):
        CandidateRawCountSamples(
            candidate_id="baseline-like-id",
            target=valid.target,
            ordered_feature_keys=valid.ordered_feature_keys,
            model_artifact_sha256=valid.model_artifact_sha256,
            calibration_state_sha256=valid.calibration_state_sha256,
            sampling_contract_sha256=valid.sampling_contract_sha256,
            target_fingerprint=valid.target_fingerprint,
            context_id=valid.context_id,
            seed=valid.seed,
            samples=valid.samples,
        )
    with pytest.raises(SciPlex3CandidateError, match="contract provenance"):
        CandidateRawCountSamples(
            candidate_id=valid.candidate_id,
            target=valid.target,
            ordered_feature_keys=valid.ordered_feature_keys,
            model_artifact_sha256=valid.model_artifact_sha256,
            calibration_state_sha256=valid.calibration_state_sha256,
            sampling_contract_sha256="f" * 64,
            target_fingerprint=valid.target_fingerprint,
            context_id=valid.context_id,
            seed=valid.seed,
            samples=valid.samples,
        )


def test_golden_fixture_has_fixed_model_and_sample_identities() -> None:
    candidate = build_sciplex3_synthetic_golden_candidate()
    assert SCIPLEX3_CANDIDATE_GOLDEN_MODEL_SHA256 == (
        "e5f81e28b8f4efbf5cffd64afa326d380b4c071450fd02339b4a46102a2e70a2"
    )
    assert SCIPLEX3_CANDIDATE_GOLDEN_SAMPLE_SHA256 == (
        "9c364005b01142bcfe74a8400ab4b9209ed635b74a703167969aa5e8d80b9e2c"
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


def test_active_action_context_m_step_is_deterministic_all_well_and_v5_scaled() -> None:
    validated = _mini_full_topology()
    scores = np.ones((768, 16), dtype=np.float64)
    plate_effect = np.linspace(-0.3, 0.3, 8, dtype=np.float64)
    dose_effect = np.asarray((-0.15, -0.05, 0.08, 0.22), dtype=np.float64)
    for compound_index in range(188):
        for dose_index in range(4):
            well_index = int(validated.action_well_indices[compound_index, dose_index])
            plate_index = int(validated.training_well_plate_indices[well_index])
            scores[well_index] = math.exp(
                float(plate_effect[plate_index] + dose_effect[dose_index])
            )

    first = candidate_module._update_action_block(scores, validated)
    repeated = candidate_module._update_action_block(scores, validated)
    second = candidate_module._update_action_block(
        scores,
        validated,
        initial=(first[0], first[1], first[2]),
    )

    for first_tensor, repeated_tensor, second_tensor in zip(first, repeated, second, strict=True):
        assert np.array_equal(first_tensor, repeated_tensor)
        assert np.allclose(first_tensor, second_tensor, rtol=0.0, atol=1e-8)
    assert np.max(np.abs(np.log(first[1]))) > 0.1
    assert np.allclose(np.mean(first[1], axis=0), 1.0, rtol=0.0, atol=5e-13)
    action_spec = cast(dict[str, object], candidate_specification_manifest()["action_model"])
    assert action_spec["canonical_objective_version"] == "5.0.0"
    assert action_spec["equal_well_scale"] == 94_785 / 768
    assert action_spec["plate_intercept_fit"] == (
        "all-768-wells-including-treated-and-vehicle-wells"
    )
    assert action_spec["terminal_gradient_tolerance"] == 3e-4
    assert not hasattr(candidate_module, "_fit_dose_block")
    assert not hasattr(candidate_module, "_dose_objective")
    assert not hasattr(candidate_module, "_solve_gamma_shape")
    assert not hasattr(candidate_module, "_shape_statistics")


def test_active_action_context_m_step_rejects_nonpositive_posterior_means() -> None:
    validated = _mini_full_topology()
    with pytest.raises(SciPlex3CandidateError, match="v5 action/context"):
        candidate_module._update_action_block(np.zeros((768, 16)), validated)


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
        candidate_module._v5_action_parameters(next_alpha, next_rho, next_delta),
    )
    assert np.isfinite(elbo)
    basis = updated_loading / np.sum(updated_loading, axis=1, keepdims=True)
    contributions = candidate_module._factor_contributions(next_means)
    order = candidate_module._canonical_factor_order(basis, contributions)
    assert sorted(order.tolist()) == list(range(16))


def test_tracked_elbo_parameter_difference_is_v5_q_under_unequal_well_row_counts() -> None:
    validated = _mini_full_topology()
    row_counts = tuple(well.counts.row_count for well in validated.wells)
    assert set(row_counts) == {1, 2}
    assert sum(row_counts) == validated.record_count == 769

    row = np.arange(validated.record_count, dtype=np.float64)[:, None]
    factor = np.arange(16, dtype=np.float64)[None, :]
    state = candidate_module._LocalVariationalState(
        theta_shape=0.2 + 0.01 * ((row + 2.0 * factor) % 17.0),
        theta_rate=0.7 + 0.02 * ((3.0 * row + factor) % 13.0),
    )
    posterior = np.empty((768, 16), dtype=np.float64)
    offset = 0
    for well_index, row_count in enumerate(row_counts):
        stop = offset + row_count
        posterior[well_index] = np.mean(
            state.theta_shape[offset:stop] / state.theta_rate[offset:stop], axis=0
        )
        offset = stop

    loading = np.full((16, SCIPLEX3_FEATURE_COUNT), 0.3, dtype=np.float64)
    sufficient = candidate_module._PassSufficientStatistics(
        loading_counts=np.zeros_like(loading),
        well_theta_means=posterior,
        allocation_entropy=0.0,
        poisson_factorial=0.0,
        theta_count_elog=0.0,
        maximum_inner_sweeps=2,
        maximum_terminal_shape_residual=0.0,
        maximum_terminal_elog_residual=0.0,
        inner_sweep_count_histogram=(0, 768, *([0] * 48)),
    )
    alpha_a = np.linspace(-0.2, 0.2, 16, dtype=np.float64)
    rho_a = np.ones((8, 16), dtype=np.float64)
    delta_a = np.zeros((188, 4, 16), dtype=np.float64)
    raw_rho_b = np.exp(0.12 * np.sin(np.arange(8 * 16, dtype=np.float64).reshape(8, 16) / 11.0))
    rho_b = raw_rho_b / np.mean(raw_rho_b, axis=0, keepdims=True)
    alpha_b = alpha_a + np.linspace(-0.04, 0.05, 16, dtype=np.float64)
    delta_b = 0.025 * np.sin(np.arange(188 * 4 * 16, dtype=np.float64).reshape(188, 4, 16) / 29.0)
    parameters_a = candidate_module._v5_action_parameters(alpha_a, rho_a, delta_a)
    parameters_b = candidate_module._v5_action_parameters(alpha_b, rho_b, delta_b)

    elbo_a = candidate_module._elbo(validated, state, sufficient, loading, parameters_a)
    elbo_b = candidate_module._elbo(validated, state, sufficient, loading, parameters_b)
    objective_design = SciPlex3V5Design(
        validated.training_well_plate_indices,
        validated.action_well_indices,
        validated.vehicle_well_indices,
    )
    q_a = fixed_q_full_elbo_action_context(posterior, parameters_a, objective_design)
    q_b = fixed_q_full_elbo_action_context(posterior, parameters_b, objective_design)

    assert elbo_b - elbo_a == pytest.approx(q_b - q_a, rel=0.0, abs=2e-10)


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
    with pytest.raises(SciPlex3CandidateError, match="exact declared"):
        candidate_module._factor_shape_for_tau(np.nan)


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

    def action_block(
        *_: object, **__: object
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
