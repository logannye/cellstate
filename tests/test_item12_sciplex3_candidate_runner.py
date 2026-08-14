from __future__ import annotations

import importlib
import json
import shutil
from dataclasses import replace
from pathlib import Path
from typing import cast

import numpy as np
import pytest
from test_item11_sciplex3_runner import _SyntheticExactP1Loader

import cellstate.evaluation.sciplex3_candidate as candidate_module
import cellstate.evaluation.sciplex3_candidate_runner as runner
from cellstate.backends.training import TrainedCandidateFactory
from cellstate.data.benchmarks import BenchmarkPartitionRole
from cellstate.evaluation.sciplex3_baselines import (
    CompoundDose,
    ImmutableCSRCounts,
    P1TrainingData,
)
from cellstate.evaluation.sciplex3_candidate import (
    SCIPLEX3_CANDIDATE_FACTOR_COUNT,
    SCIPLEX3_CANDIDATE_FIXED_FACTOR_SHAPE,
    SCIPLEX3_CANDIDATE_GOLDEN_MODEL_SHA256,
    SCIPLEX3_CANDIDATE_GOLDEN_SAMPLE_SHA256,
    SCIPLEX3_CANDIDATE_MAX_INNER_SWEEPS,
    SCIPLEX3_CANDIDATE_REFERENCE_RUNTIME,
    SCIPLEX3_CANDIDATE_V5_NEUTRAL_CONTEXT_ID,
    SciPlex3CandidateError,
    SciPlex3CandidateInitialEquilibration,
    SciPlex3CandidateTraceEntry,
    SciPlex3CandidateTrainingSummary,
    SciPlex3GammaPoissonCandidate,
    build_sciplex3_synthetic_golden_candidate,
    training_data_fingerprint,
)
from cellstate.evaluation.sciplex3_runner import (
    LocalContentAddressedArtifact,
    SciPlex3BaselinePreparation,
    assemble_sciplex3_p1_training_data,
)
from cellstate.evaluation.sciplex3_sampling_v5 import (
    SCIPLEX3_V5_MAX_SAMPLE_COUNT,
    SCIPLEX3_V5_REQUEST_FAILURE_BUDGET_LOG,
    SCIPLEX3_V5_SAMPLING_CONTRACT_SHA256,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_DIGEST = "a" * 64


def _reference_runtime() -> dict[str, object]:
    return dict(SCIPLEX3_CANDIDATE_REFERENCE_RUNTIME)


@pytest.fixture(autouse=True)
def exact_reference_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runner, "_observed_runtime", _reference_runtime)


@pytest.fixture(scope="module")
def exact_preparation() -> SciPlex3BaselinePreparation:
    return assemble_sciplex3_p1_training_data(
        _SyntheticExactP1Loader(), REPOSITORY_ROOT, batch_size=4_096
    )


def _candidate_for(preparation: SciPlex3BaselinePreparation) -> SciPlex3GammaPoissonCandidate:
    design = runner._candidate_design(preparation)
    compounds = design.compounds
    plates = design.plate_ids
    (
        training_well_ids,
        training_well_plate_indices,
        action_well_indices,
        vehicle_well_indices,
    ) = runner._expected_candidate_topology(preparation, design)
    template = build_sciplex3_synthetic_golden_candidate()
    basis = template._basis.copy()
    alpha = template._alpha.copy()
    rho = template._rho.copy()
    delta = template._delta.copy()
    mean_activation = candidate_module._canonical_audit_matrix(
        candidate_module._reconstruct_mean_activation(
            alpha,
            rho,
            delta,
            training_well_plate_indices,
            action_well_indices,
            vehicle_well_indices,
        )
    )
    contributions = candidate_module._factor_contributions(mean_activation)
    order = candidate_module._canonical_factor_order(basis, contributions)
    basis = basis[order]
    alpha = alpha[order]
    rho = rho[:, order]
    delta = delta[:, :, order]
    mean_activation = candidate_module._canonical_audit_matrix(
        candidate_module._reconstruct_mean_activation(
            alpha,
            rho,
            delta,
            training_well_plate_indices,
            action_well_indices,
            vehicle_well_indices,
        )
    )
    contributions = candidate_module._factor_contributions(mean_activation)
    elbos = (
        -1_000.0,
        -900.0,
        -850.0,
        -840.0,
        -839.0,
        -838.9,
        -838.89,
        -838.88999,
        -838.889989,
        -838.8899889,
    )
    initial_elbo = -1_100.0
    inner_batch_count = runner._expected_inner_batch_count(preparation)
    inner_histogram = (0, inner_batch_count) + (0,) * (SCIPLEX3_CANDIDATE_MAX_INNER_SWEEPS - 2)
    initial_equilibration = SciPlex3CandidateInitialEquilibration(
        elbo=initial_elbo,
        factor_order=tuple(range(SCIPLEX3_CANDIDATE_FACTOR_COUNT)),
        maximum_inner_sweeps=2,
        maximum_terminal_shape_residual=1e-9,
        maximum_terminal_elog_residual=2e-9,
        inner_sweep_count_histogram=inner_histogram,
    )
    trace = tuple(
        SciPlex3CandidateTraceEntry(
            iteration=index + 1,
            elbo=elbo,
            relative_change=abs(elbo - (initial_elbo if index == 0 else elbos[index - 1]))
            / max(1.0, abs(initial_elbo if index == 0 else elbos[index - 1])),
            factor_order=tuple(range(SCIPLEX3_CANDIDATE_FACTOR_COUNT)),
            maximum_inner_sweeps=2,
            maximum_terminal_shape_residual=1e-9,
            maximum_terminal_elog_residual=2e-9,
            inner_sweep_count_histogram=inner_histogram,
        )
        for index, elbo in enumerate(elbos)
    )
    return SciPlex3GammaPoissonCandidate(
        ordered_feature_keys=preparation.training_data.ordered_feature_keys,
        compounds=compounds,
        plate_ids=plates,
        training_well_ids=training_well_ids,
        _basis=basis,
        _alpha=alpha,
        _rho=rho,
        _delta=delta,
        _factor_shape=np.asarray([SCIPLEX3_CANDIDATE_FIXED_FACTOR_SHAPE], dtype=np.float64),
        _factor_contributions=contributions,
        _mean_activation=mean_activation,
        _training_well_plate_indices=training_well_plate_indices,
        _action_well_indices=action_well_indices,
        _vehicle_well_indices=vehicle_well_indices,
        initial_equilibration=initial_equilibration,
        trace=trace,
        training_summary=SciPlex3CandidateTrainingSummary(
            record_count=preparation.receipt.record_count,
            well_count=preparation.receipt.well_count,
            zero_panel_record_count=preparation.receipt.zero_panel_record_count,
            design_sha256=design.fingerprint,
            training_data_sha256=training_data_fingerprint(preparation.training_data),
            provenance="real-p1",
        ),
    )


@pytest.fixture(scope="module")
def exact_candidate(
    exact_preparation: SciPlex3BaselinePreparation,
) -> SciPlex3GammaPoissonCandidate:
    return _candidate_for(exact_preparation)


@pytest.fixture(scope="module")
def sealed_plan(
    exact_preparation: SciPlex3BaselinePreparation,
    tmp_path_factory: pytest.TempPathFactory,
) -> runner.SealedSciPlex3CandidateTrainingPlan:
    patch = pytest.MonkeyPatch()
    patch.setattr(runner, "_observed_runtime", _reference_runtime)
    try:
        plan = runner.build_sciplex3_candidate_training_plan(
            exact_preparation,
            benchmark_fingerprint="b" * 64,
            support_envelope_fingerprint="c" * 64,
        )
        return runner.seal_sciplex3_candidate_training_plan(
            exact_preparation,
            plan,
            tmp_path_factory.mktemp("item12-plan") / "sealed",
        )
    finally:
        patch.undo()


@pytest.fixture(scope="module")
def fitted_candidate(
    exact_preparation: SciPlex3BaselinePreparation,
    exact_candidate: SciPlex3GammaPoissonCandidate,
    sealed_plan: runner.SealedSciPlex3CandidateTrainingPlan,
    tmp_path_factory: pytest.TempPathFactory,
) -> runner.FittedSciPlex3Candidate:
    patch = pytest.MonkeyPatch()
    patch.setattr(runner, "_observed_runtime", _reference_runtime)
    patch.setattr(runner, "_fit_exact_candidate", lambda *_: exact_candidate)
    try:
        return runner.fit_and_write_sciplex3_candidate(
            exact_preparation,
            sealed_plan,
            tmp_path_factory.mktemp("item12-fit") / "fit",
        )
    finally:
        patch.undo()


def test_contained_runtime_lock_binds_builder_archive_and_layer_closure() -> None:
    policy, code_closure, _, image_lock = runner.contained_training_contracts(REPOSITORY_ROOT)

    assert image_lock.runtime_image == policy.runtime_image
    assert policy.worker_command[:3] == ("--signal=TERM", "--kill-after=5s", "3540")
    assert image_lock.training_code_closure_sha256 == code_closure.fingerprint
    assert image_lock.archive_sha256 == (
        "37c2fa5846acfbd8357476859bd7f8f0ac6591261d79c2f6f46f0aa22fb76454"
    )
    assert image_lock.oci_index_digest == (
        "sha256:e0f0afd6c66197a37d0ab7a05e7cccfe5990da1fd8497e175fdf3ab909a67812"
    )
    assert image_lock.config_digest == (
        "sha256:80ed48f278d7a46c0ae7811285efc69181ae59872a358cc9b176079aa09f3cc8"
    )
    assert image_lock.builder.buildx_version == "v0.28.0"
    assert image_lock.builder.buildx_commit == "b1281b81bba797b21d9eaf256e6a13eb14419836"
    assert image_lock.builder.buildkit_version == "v0.24.0"
    assert image_lock.builder.output_options == ("type=oci",)
    assert len(image_lock.layers) == 6
    assert image_lock.layers[4].digest == (
        "sha256:4eaeda62bd74078a1cd0f387c18cac3c1273826cbda1222ba571bf4e06b26533"
    )
    assert image_lock.layers[4].byte_count == 67_847_890


@pytest.mark.parametrize("target", ("archive", "builder", "layer"))
def test_contained_runtime_lock_rejects_provenance_substitution(
    target: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_read = runner._read_bytes

    def substituted_read(path: Path, *, name: str) -> bytes:
        payload = original_read(path, name=name)
        if name != "runtime image provenance lock":
            return payload
        provenance = json.loads(payload)
        if target == "archive":
            provenance["archive_sha256"] = "f" * 64
        elif target == "builder":
            provenance["builder"]["buildx_commit"] = "f" * 40
        else:
            provenance["layers"][4]["byte_count"] += 1
        return runner._canonical_json(provenance)

    monkeypatch.setattr(runner, "_read_bytes", substituted_read)
    with pytest.raises(runner.SciPlex3CandidateRunnerError, match="contradicts its exact files"):
        runner.contained_training_contracts(REPOSITORY_ROOT)


def test_plan_and_fit_are_exact_p1_non_authorizing_artifacts(
    exact_preparation: SciPlex3BaselinePreparation,
    sealed_plan: runner.SealedSciPlex3CandidateTrainingPlan,
    fitted_candidate: runner.FittedSciPlex3Candidate,
) -> None:
    plan = sealed_plan.plan
    assert plan.plan_id == runner.SCIPLEX3_CANDIDATE_TRAINING_PLAN_ID
    assert plan.plan_version == "5.0.0"
    assert plan.trainer_implementation.implementation_version == "5.0.0"
    assert plan.candidate_factory_implementation.implementation_version == "5.0.0"
    assert plan.training_partition_ids == ("p1-train",)
    assert plan.training_partition_roles == (BenchmarkPartitionRole.TRAIN,)
    assert plan.future_calibration_plan is None
    assert plan.p1_count_stream_sha256 == exact_preparation.receipt.runner_panel_count_stream_sha256
    assert plan.p1_assembly_fingerprint == exact_preparation.receipt.fingerprint
    assert plan.candidate_factory_implementation.entrypoint == (
        "cellstate.evaluation.sciplex3_candidate:SciPlex3GammaPoissonCandidate"
    )
    assert {item.path.name for item in sealed_plan.support_artifacts} == {
        "candidate-specification.json",
        "contained-execution-policy.json",
        "output-model-schema.json",
        "p1-count-stream-descriptor.json",
        "publication-generation-seed.json",
        "runtime-lock.json",
        "runtime-image-lock.json",
        "training-code-closure.json",
        "training-execution-input-closure.json",
    }
    assert {
        fitted_candidate.model_artifact.path.name,
        fitted_candidate.observation_artifact.path.name,
    } == {
        "candidate-model.json",
        "training-execution-observation.json",
    }
    observation = fitted_candidate.observation
    assert observation.training_code_closure_sha256 == plan.training_code_closure.sha256
    assert observation.training_execution_input_closure_sha256 == (
        plan.training_execution_input_closure.sha256
    )
    assert observation.software_golden_model_sha256 == SCIPLEX3_CANDIDATE_GOLDEN_MODEL_SHA256
    assert observation.software_golden_sample_sha256 == SCIPLEX3_CANDIDATE_GOLDEN_SAMPLE_SHA256
    assert observation.fit_converged is True
    assert observation.factor_order_stable is True
    assert observation.capture_latent_present is False
    assert observation.artifact_schema_version == "5.0.0"
    assert observation.candidate_model_schema_version == "5.0.0"
    assert observation.factor_shape_mode == "fixed"
    assert observation.factor_shape_estimated is False
    assert observation.fixed_factor_shape == SCIPLEX3_CANDIDATE_FIXED_FACTOR_SHAPE
    assert observation.fixed_factor_shape == fitted_candidate.candidate.factor_shape
    assert observation.inner_equilibration_performed is True
    assert observation.inner_all_batches_converged is True
    assert observation.inner_batch_count == sum(
        fitted_candidate.candidate.initial_equilibration.inner_sweep_count_histogram
    )
    initial_sweeps = sum(
        (index + 1) * count
        for index, count in enumerate(
            fitted_candidate.candidate.initial_equilibration.inner_sweep_count_histogram
        )
    )
    traced_sweeps = sum(
        (index + 1) * count
        for item in fitted_candidate.candidate.trace
        for index, count in enumerate(item.inner_sweep_count_histogram)
    )
    assert observation.total_inner_sweep_count == initial_sweeps + traced_sweeps
    assert observation.initial_elbo == fitted_candidate.candidate.initial_equilibration.elbo
    assert observation.initial_factor_order == (
        fitted_candidate.candidate.initial_equilibration.factor_order
    )
    assert observation.plate_context_family == "neutral-unit-context"
    assert observation.plate_context_id == SCIPLEX3_CANDIDATE_V5_NEUTRAL_CONTEXT_ID
    assert observation.plate_context_count == 1
    assert observation.plate_context_factorwise_mean_one is True
    assert observation.sampling_conditioning == (
        "exact-positive-panel-via-zero-truncated-compound-poisson"
    )
    assert observation.sampling_request_support == ("exact-CandidateSampleRequest-not-target-only")
    assert observation.sampling_contract_sha256 == SCIPLEX3_V5_SAMPLING_CONTRACT_SHA256
    assert observation.sampling_envelope_supported is True
    assert observation.sampling_envelope_combination_count == 753 * 1 * 27
    assert observation.sampling_envelope_maximum_request_count == SCIPLEX3_V5_MAX_SAMPLE_COUNT
    assert observation.sampling_envelope_rejection_reasons == ()
    assert observation.sampling_envelope_request_failure_budget_log == (
        SCIPLEX3_V5_REQUEST_FAILURE_BUDGET_LOG
    )
    assert observation.sampling_envelope_worst_request_tail_log_upper_bound <= (
        SCIPLEX3_V5_REQUEST_FAILURE_BUDGET_LOG
    )
    assert observation.candidate_objective_code_sha256 == (
        runner._IMPORTED_CANDIDATE_V5_CODE_SHA256
    )
    assert observation.candidate_sampling_code_sha256 == (runner._IMPORTED_SAMPLING_V5_CODE_SHA256)
    assert observation.plate_sigma_present is False
    fitted_state = fitted_candidate.candidate.fitted_state_manifest()
    assert observation.initial_equilibration_sha256 == fitted_state["initial_equilibration_sha256"]
    assert (
        observation.inner_equilibration_trace_sha256
        == fitted_state["inner_equilibration_trace_sha256"]
    )
    tensor_sha256 = cast(dict[str, str], fitted_state["tensor_sha256"])
    assert observation.training_nuisance_rho_sha256 == tensor_sha256["rho"]
    sampler = fitted_candidate.candidate._v5_runtime_sampler()
    assert observation.sampling_active_calibration_state_sha256 == (
        sampler.parameters.active_calibration_state_sha256
    )
    assert observation.sampling_envelope_certificate_sha256 == (
        sampler.envelope_certificate.fingerprint
    )
    assert np.array_equal(
        sampler.parameters.context_multipliers,
        np.ones((27, 1, SCIPLEX3_CANDIDATE_FACTOR_COUNT)),
    )
    assert not np.shares_memory(
        sampler.parameters.context_multipliers,
        fitted_candidate.candidate._rho,
    )
    assert len(observation.terminal_elbo_relative_changes) == 3
    assert not hasattr(observation, "terminal_factor_shape_log_changes")
    assert not hasattr(observation, "final_factor_shape")
    assert observation.golden_reproduced is True
    assert observation.heldout_artifacts_resolved is False
    assert observation.heldout_memberships_read is False
    assert observation.heldout_outcomes_read is False
    assert observation.calibration_performed is False
    assert observation.model_selection_performed is False
    assert observation.metrics_computed is False
    assert observation.can_mint_lifecycle_evidence is False
    assert observation.scientifically_admissible is False
    assert (
        runner.verify_sciplex3_candidate_fit(
            exact_preparation, sealed_plan, fitted_candidate
        ).fingerprint
        == observation.fingerprint
    )


def test_factory_binding_resolves_nonvacuous_training_interface(
    sealed_plan: runner.SealedSciPlex3CandidateTrainingPlan,
    exact_candidate: SciPlex3GammaPoissonCandidate,
) -> None:
    module_name, class_name = cast(
        str, sealed_plan.plan.candidate_factory_implementation.entrypoint
    ).split(":", maxsplit=1)
    resolved = getattr(importlib.import_module(module_name), class_name)
    assert resolved is SciPlex3GammaPoissonCandidate
    assert isinstance(exact_candidate, TrainedCandidateFactory)
    model_bytes = exact_candidate.model_bytes()
    reloaded = resolved.load_exact(
        model_bytes, expected_sha256=exact_candidate.model_artifact_sha256
    )
    assert type(reloaded) is SciPlex3GammaPoissonCandidate
    assert reloaded.model_bytes() == model_bytes
    assert reloaded.behavior_manifest()["fit_converged"] is True
    assert reloaded.golden_sample().target.partition_id == "p1-train"


def test_altered_csr_cannot_reuse_the_sealed_plan(
    exact_preparation: SciPlex3BaselinePreparation,
    sealed_plan: runner.SealedSciPlex3CandidateTrainingPlan,
    tmp_path: Path,
) -> None:
    original = next(
        well for well in exact_preparation.training_data.wells if well.counts.values.size
    )
    changed_values = original.counts.values.copy()
    changed_values[0] += 1
    changed_counts = ImmutableCSRCounts(
        row_count=original.counts.row_count,
        indptr=original.counts.indptr,
        feature_indices=original.counts.feature_indices,
        values=changed_values,
    )
    changed_well = replace(original, counts=changed_counts)
    training = P1TrainingData(
        ordered_feature_keys=exact_preparation.training_data.ordered_feature_keys,
        wells=tuple(
            changed_well if well.well_id == original.well_id else well
            for well in exact_preparation.training_data.wells
        ),
    )
    changed = replace(exact_preparation, training_data=training)
    output = tmp_path / "altered-csr"
    with pytest.raises(Exception, match=r"CSR|count stream|p1 counts"):
        runner.fit_and_write_sciplex3_candidate(changed, sealed_plan, output)
    assert not output.exists()


def test_structural_sealed_plan_lookalike_is_rejected(
    exact_preparation: SciPlex3BaselinePreparation,
    tmp_path: Path,
) -> None:
    class _Lookalike:
        preparation_fingerprint = exact_preparation.receipt.fingerprint

    output = tmp_path / "lookalike"
    with pytest.raises(runner.SciPlex3CandidateRunnerError, match="exact sealed"):
        runner.fit_and_write_sciplex3_candidate(
            exact_preparation,
            cast(runner.SealedSciPlex3CandidateTrainingPlan, _Lookalike()),
            output,
        )
    assert not output.exists()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("optimization_seed", 7),
        ("plan_version", "1.0.0"),
        ("plan_version", "2.0.0"),
        ("plan_version", "3.0.0"),
        ("training_partition_ids", ("p2-calibration",)),
        ("training_partition_roles", (BenchmarkPartitionRole.CALIBRATION,)),
    ],
)
def test_stale_or_heldout_plan_is_rejected_before_fit(
    exact_preparation: SciPlex3BaselinePreparation,
    sealed_plan: runner.SealedSciPlex3CandidateTrainingPlan,
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    changed_plan = sealed_plan.plan.model_copy(update={field: value})
    changed = replace(sealed_plan, plan=changed_plan)
    output = tmp_path / field
    with pytest.raises(runner.SciPlex3CandidateRunnerError, match=r"stale|differs"):
        runner.fit_and_write_sciplex3_candidate(exact_preparation, changed, output)
    assert not output.exists()


def test_candidate_code_drift_is_rejected_before_fit(
    exact_preparation: SciPlex3BaselinePreparation,
    sealed_plan: runner.SealedSciPlex3CandidateTrainingPlan,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(runner, "_IMPORTED_CANDIDATE_CODE_SHA256", "0" * 64)
    output = tmp_path / "code-drift"
    with pytest.raises(runner.SciPlex3CandidateRunnerError, match="changed since module import"):
        runner.fit_and_write_sciplex3_candidate(exact_preparation, sealed_plan, output)
    assert not output.exists()


def test_candidate_runner_code_drift_is_rejected_before_fit(
    exact_preparation: SciPlex3BaselinePreparation,
    sealed_plan: runner.SealedSciPlex3CandidateTrainingPlan,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(runner, "_IMPORTED_RUNNER_CODE_SHA256", "0" * 64)
    output = tmp_path / "runner-code-drift"
    with pytest.raises(runner.SciPlex3CandidateRunnerError, match="changed since module import"):
        runner.fit_and_write_sciplex3_candidate(exact_preparation, sealed_plan, output)
    assert not output.exists()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("numpy_version", "0.0.0"),
        ("blas_name", "mkl"),
        ("blas_version", "0.0.0"),
    ],
)
def test_runtime_drift_is_rejected_before_fit(
    exact_preparation: SciPlex3BaselinePreparation,
    sealed_plan: runner.SealedSciPlex3CandidateTrainingPlan,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    field: str,
    value: str,
) -> None:
    drifted = _reference_runtime()
    drifted[field] = value
    monkeypatch.setattr(runner, "_observed_runtime", lambda: drifted)
    output = tmp_path / f"runtime-drift-{field}"
    with pytest.raises(runner.SciPlex3CandidateRunnerError, match="exact frozen"):
        runner.fit_and_write_sciplex3_candidate(exact_preparation, sealed_plan, output)
    assert not output.exists()


def test_off_reference_runtime_fails_before_lazy_golden_construction(
    exact_preparation: SciPlex3BaselinePreparation,
    sealed_plan: runner.SealedSciPlex3CandidateTrainingPlan,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    drifted = _reference_runtime()
    drifted["python_version"] = "3.13.7"
    monkeypatch.setattr(runner, "_observed_runtime", lambda: drifted)

    def unexpected_golden() -> None:
        pytest.fail("golden construction ran before the exact runtime gate")

    monkeypatch.setattr(runner, "_verify_factory_golden", unexpected_golden)
    output = tmp_path / "off-reference-before-golden"
    with pytest.raises(runner.SciPlex3CandidateRunnerError, match="exact frozen"):
        runner.fit_and_write_sciplex3_candidate(exact_preparation, sealed_plan, output)
    assert not output.exists()


@pytest.mark.parametrize(
    "failure",
    [SciPlex3CandidateError("unconverged"), FloatingPointError("nonfinite")],
)
def test_unconverged_or_nonfinite_fit_emits_no_artifact(
    exact_preparation: SciPlex3BaselinePreparation,
    sealed_plan: runner.SealedSciPlex3CandidateTrainingPlan,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    failure: Exception,
) -> None:
    def fail(*_: object) -> SciPlex3GammaPoissonCandidate:
        raise failure

    monkeypatch.setattr(runner, "_fit_exact_candidate", fail)
    output = tmp_path / type(failure).__name__
    with pytest.raises(Exception, match=r"unconverged|nonfinite"):
        runner.fit_and_write_sciplex3_candidate(exact_preparation, sealed_plan, output)
    assert not output.exists()


def test_unsupported_v5_sampling_certificate_emits_no_artifact(
    exact_preparation: SciPlex3BaselinePreparation,
    exact_candidate: SciPlex3GammaPoissonCandidate,
    sealed_plan: runner.SealedSciPlex3CandidateTrainingPlan,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    candidate = SciPlex3GammaPoissonCandidate.load_exact(
        exact_candidate.canonical_model_bytes(),
        expected_sha256=exact_candidate.model_artifact_sha256,
    )
    certificate = candidate._v5_sampling_envelope_certificate_cache
    object.__setattr__(certificate, "supported", False)
    object.__setattr__(certificate, "rejection_reasons", ("forged unsafe envelope",))
    monkeypatch.setattr(runner, "_fit_exact_candidate", lambda *_: candidate)
    output = tmp_path / "unsupported-v5-sampling-envelope"

    with pytest.raises(runner.SciPlex3CandidateRunnerError, match=r"sampling cache|certificate"):
        runner.fit_and_write_sciplex3_candidate(exact_preparation, sealed_plan, output)
    assert not output.exists()


def test_reload_substitution_is_rejected_before_output(
    exact_preparation: SciPlex3BaselinePreparation,
    exact_candidate: SciPlex3GammaPoissonCandidate,
    sealed_plan: runner.SealedSciPlex3CandidateTrainingPlan,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(runner, "_fit_exact_candidate", lambda *_: exact_candidate)
    monkeypatch.setattr(runner, "_load_exact_candidate", lambda *_args, **_kwargs: object())
    output = tmp_path / "reload-substitution"
    with pytest.raises(runner.SciPlex3CandidateRunnerError, match="substituted"):
        runner.fit_and_write_sciplex3_candidate(exact_preparation, sealed_plan, output)
    assert not output.exists()


def test_runner_independently_rejects_forged_v5_topology(
    exact_preparation: SciPlex3BaselinePreparation,
    exact_candidate: SciPlex3GammaPoissonCandidate,
) -> None:
    candidate = SciPlex3GammaPoissonCandidate.load_exact(
        exact_candidate.canonical_model_bytes(),
        expected_sha256=exact_candidate.model_artifact_sha256,
    )
    changed = candidate._action_well_indices.copy()
    changed[0, 0], changed[0, 1] = changed[0, 1], changed[0, 0]
    object.__setattr__(candidate, "_action_well_indices", changed)
    with pytest.raises(runner.SciPlex3CandidateRunnerError, match="sealed topology"):
        runner._validate_candidate_state(
            candidate,
            exact_preparation,
            runner._candidate_design(exact_preparation),
        )


def test_runner_rejects_mutable_fixed_shape_and_training_nuisance_rho_lookalikes(
    exact_preparation: SciPlex3BaselinePreparation,
    exact_candidate: SciPlex3GammaPoissonCandidate,
) -> None:
    design = runner._candidate_design(exact_preparation)
    for field in ("_factor_shape", "_rho"):
        candidate = SciPlex3GammaPoissonCandidate.load_exact(
            exact_candidate.canonical_model_bytes(),
            expected_sha256=exact_candidate.model_artifact_sha256,
        )
        object.__setattr__(candidate, field, getattr(candidate, field).copy())
        with pytest.raises(
            runner.SciPlex3CandidateRunnerError,
            match=r"fixed factor-shape|training nuisance rho",
        ):
            runner._validate_candidate_state(candidate, exact_preparation, design)


def test_runner_independently_rejects_invalid_training_nuisance_rho_means(
    exact_preparation: SciPlex3BaselinePreparation,
    exact_candidate: SciPlex3GammaPoissonCandidate,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = SciPlex3GammaPoissonCandidate.load_exact(
        exact_candidate.canonical_model_bytes(),
        expected_sha256=exact_candidate.model_artifact_sha256,
    )
    changed_rho = candidate._rho.copy()
    changed_rho[0, 0] *= 0.9
    changed_rho.setflags(write=False)
    object.__setattr__(candidate, "_rho", changed_rho)
    monkeypatch.setattr(
        runner,
        "_validate_candidate_topology",
        lambda *_: tuple(float(value) for value in candidate._factor_contributions),
    )
    with pytest.raises(runner.SciPlex3CandidateRunnerError, match="training nuisance rho"):
        runner._validate_candidate_state(
            candidate,
            exact_preparation,
            runner._candidate_design(exact_preparation),
        )


def test_runner_independently_rejects_forged_initial_and_inner_witnesses(
    exact_preparation: SciPlex3BaselinePreparation,
    exact_candidate: SciPlex3GammaPoissonCandidate,
) -> None:
    design = runner._candidate_design(exact_preparation)
    candidate = SciPlex3GammaPoissonCandidate.load_exact(
        exact_candidate.canonical_model_bytes(),
        expected_sha256=exact_candidate.model_artifact_sha256,
    )
    object.__setattr__(
        candidate.initial_equilibration,
        "factor_order",
        (0,) * SCIPLEX3_CANDIDATE_FACTOR_COUNT,
    )
    with pytest.raises(runner.SciPlex3CandidateRunnerError, match="initial factor order"):
        runner._validate_candidate_state(candidate, exact_preparation, design)

    candidate = SciPlex3GammaPoissonCandidate.load_exact(
        exact_candidate.canonical_model_bytes(),
        expected_sha256=exact_candidate.model_artifact_sha256,
    )
    object.__setattr__(
        candidate.trace[0],
        "maximum_terminal_elog_residual",
        2e-8,
    )
    with pytest.raises(runner.SciPlex3CandidateRunnerError, match="inner tolerance"):
        runner._validate_candidate_state(candidate, exact_preparation, design)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("capture_latent_present", True),
        ("factor_order_stable", False),
        ("loading_rank_ratio", 0.0),
        ("outer_iteration_count", 11),
        ("inner_all_batches_converged", False),
        ("plate_context_family", "independent-lognormal"),
        ("sampling_contract_sha256", "f" * 64),
        ("sampling_envelope_supported", False),
        ("sampling_envelope_maximum_request_count", 513),
    ],
)
def test_runner_rederives_v5_behavior_and_sampling_gates(
    exact_preparation: SciPlex3BaselinePreparation,
    exact_candidate: SciPlex3GammaPoissonCandidate,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
) -> None:
    original = SciPlex3GammaPoissonCandidate.behavior_manifest

    def forged(self: SciPlex3GammaPoissonCandidate) -> dict[str, object]:
        behavior = original(self)
        behavior[field] = value
        return behavior

    monkeypatch.setattr(SciPlex3GammaPoissonCandidate, "behavior_manifest", forged)
    with pytest.raises(runner.SciPlex3CandidateRunnerError, match="convergence"):
        runner._validate_candidate_state(
            exact_candidate,
            exact_preparation,
            runner._candidate_design(exact_preparation),
        )


@pytest.mark.parametrize("field", ["loading_rank_ratio", "mean_activation_rank_ratio"])
def test_runner_requires_behavior_rank_ratios_to_equal_recomputed_state(
    exact_preparation: SciPlex3BaselinePreparation,
    exact_candidate: SciPlex3GammaPoissonCandidate,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
) -> None:
    original = SciPlex3GammaPoissonCandidate.behavior_manifest

    def forged(self: SciPlex3GammaPoissonCandidate) -> dict[str, object]:
        behavior = original(self)
        value = cast(float, behavior[field])
        changed = round(value + 1e-12, 12)
        assert changed != value
        behavior[field] = changed
        return behavior

    monkeypatch.setattr(SciPlex3GammaPoissonCandidate, "behavior_manifest", forged)
    with pytest.raises(runner.SciPlex3CandidateRunnerError, match="convergence"):
        runner._validate_candidate_state(
            exact_candidate,
            exact_preparation,
            runner._candidate_design(exact_preparation),
        )


def test_changed_model_bytes_fail_later_verification(
    exact_preparation: SciPlex3BaselinePreparation,
    sealed_plan: runner.SealedSciPlex3CandidateTrainingPlan,
    fitted_candidate: runner.FittedSciPlex3Candidate,
    tmp_path: Path,
) -> None:
    copied_path = tmp_path / "candidate-model.json"
    shutil.copyfile(fitted_candidate.model_artifact.path, copied_path)
    copied_artifact = replace(fitted_candidate.model_artifact, path=copied_path)
    copied = replace(fitted_candidate, model_artifact=copied_artifact)
    with copied_path.open("ab") as stream:
        stream.write(b"\n")
    with pytest.raises(runner.SciPlex3CandidateRunnerError, match="identity drifted"):
        runner.verify_sciplex3_candidate_fit(exact_preparation, sealed_plan, copied)


def test_public_runner_bindings_reject_mutable_or_nonexact_inputs(tmp_path: Path) -> None:
    with pytest.raises(runner.SciPlex3CandidateRunnerError, match="exact shared plan"):
        runner.SealedSciPlex3CandidateTrainingPlan(
            plan=object(),  # type: ignore[arg-type]
            artifact=LocalContentAddressedArtifact(
                tmp_path / "plan", _DIGEST, 1, "application/json"
            ),
            support_artifacts=(),
            preparation_fingerprint=_DIGEST,
        )


def test_runner_helper_boundaries_fail_closed(tmp_path: Path) -> None:
    with pytest.raises(runner.SciPlex3CandidateRunnerError, match="canonical-JSON"):
        runner._canonical_json({"nonfinite": float("nan")})
    with pytest.raises(runner.SciPlex3CandidateRunnerError, match="lowercase SHA-256"):
        runner._exact_sha256("not-a-digest", name="test")
    with pytest.raises(runner.SciPlex3CandidateRunnerError, match="missing test bytes"):
        runner._read_bytes(tmp_path / "missing", name="test bytes")
    with pytest.raises(runner.SciPlex3CandidateRunnerError, match="invalid JSON"):
        runner._json_object(b"{", name="test")
    with pytest.raises(runner.SciPlex3CandidateRunnerError, match="JSON object"):
        runner._json_object(b"[]", name="test")
    with pytest.raises(runner.SciPlex3CandidateRunnerError, match="unsupported exact type"):
        runner._condition_manifest(object())
    assert runner._condition_manifest(CompoundDose("compound", 10)) == {
        "compound": "compound",
        "dose_nm": 10,
        "kind": "compound_dose",
    }

    path = tmp_path / "artifact"
    path.write_bytes(b"x")
    artifact = LocalContentAddressedArtifact(path, _DIGEST, 1, "application/json")
    with pytest.raises(runner.SciPlex3CandidateRunnerError, match="exact local artifact"):
        runner._read_local_artifact(cast(LocalContentAddressedArtifact, object()), name="test")
    with pytest.raises(runner.SciPlex3CandidateRunnerError, match="identity drifted"):
        runner._read_local_artifact(artifact, name="test")


def test_public_observation_and_fit_constructors_reject_drift(
    fitted_candidate: runner.FittedSciPlex3Candidate,
    sealed_plan: runner.SealedSciPlex3CandidateTrainingPlan,
) -> None:
    observation = fitted_candidate.observation
    invalid_updates: tuple[dict[str, object], ...] = (
        {"artifact_schema_version": "1.0.0"},
        {"candidate_model_schema_version": "3.0.0"},
        {"candidate_factory_code_sha256": "0" * 64},
        {"model_artifact_byte_count": 0},
        {"outer_iteration_count": 0},
        {"initial_elbo": float("nan")},
        {"final_elbo": float("nan")},
        {"fixed_factor_shape": 0.2},
        {"initial_factor_order": tuple(reversed(observation.initial_factor_order[:-1]))},
        {"initial_inner_sweep_count_histogram": (0,) * 50},
        {"initial_maximum_terminal_shape_residual": 1.0},
        {"maximum_terminal_elog_residual": 1.0},
        {"total_inner_sweep_count": 0},
        {
            "total_inner_sweep_count": observation.inner_batch_count
            * SCIPLEX3_CANDIDATE_MAX_INNER_SWEEPS
            * (observation.outer_iteration_count + 1)
            + 1
        },
        {"loading_rank_ratio": 0.0},
        {"terminal_elbo_relative_changes": (0.0,)},
        {"factor_order_stable": False},
        {"factor_shape_estimated": True},
        {"inner_all_batches_converged": False},
        {"plate_context_family": "independent-lognormal"},
        {"plate_context_id": "empirical-rho-row"},
        {"plate_context_count": 8},
        {"sampling_contract_sha256": "0" * 64},
        {"sampling_active_calibration_state_sha256": "0" * 64},
        {"sampling_envelope_certificate_sha256": "0" * 64},
        {"sampling_envelope_maximum_request_count": 513},
        {"sampling_envelope_supported": False},
        {"sampling_envelope_rejection_reasons": ("unsafe",)},
        {"candidate_objective_code_sha256": "0" * 64},
        {"candidate_sampling_code_sha256": "0" * 64},
        {"plate_sigma_present": True},
        {"capture_latent_present": True},
        {"model_reloaded": False},
        {"training_partition_ids": ("p2-calibration",)},
        {"heldout_artifacts_resolved": True},
    )
    for update in invalid_updates:
        try:
            replace(observation, **update)  # type: ignore[arg-type]
        except runner.SciPlex3CandidateRunnerError:
            continue
        pytest.fail(f"observation constructor accepted drift: {update!r}")

    with pytest.raises(runner.SciPlex3CandidateRunnerError, match="incomplete or unordered"):
        replace(sealed_plan, support_artifacts=tuple(reversed(sealed_plan.support_artifacts)))
    with pytest.raises(runner.SciPlex3CandidateRunnerError, match="authority flags"):
        replace(sealed_plan, can_mint_lifecycle_evidence=True)  # type: ignore[arg-type]
    with pytest.raises(runner.SciPlex3CandidateRunnerError, match="wrong exact class"):
        replace(fitted_candidate, candidate=object())  # type: ignore[arg-type]
    with pytest.raises(runner.SciPlex3CandidateRunnerError, match="plan fingerprint"):
        replace(fitted_candidate, training_plan_fingerprint="bad")


def test_factory_and_loader_wrapper_failures_are_normalized(
    exact_preparation: SciPlex3BaselinePreparation,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    design = runner._candidate_design(exact_preparation)

    def fail_fit(
        _cls: type[SciPlex3GammaPoissonCandidate],
        _training: P1TrainingData,
        _design: object,
    ) -> SciPlex3GammaPoissonCandidate:
        raise SciPlex3CandidateError("unconverged")

    monkeypatch.setattr(SciPlex3GammaPoissonCandidate, "fit", classmethod(fail_fit))
    with pytest.raises(runner.SciPlex3CandidateRunnerError, match="fitting failed"):
        runner._fit_exact_candidate(exact_preparation, design)
    with pytest.raises(runner.SciPlex3CandidateRunnerError, match="failed exact reload"):
        runner._load_exact_candidate(b"{}", expected_sha256=_DIGEST)
    with pytest.raises(runner.SciPlex3CandidateRunnerError, match="wrong exact class"):
        runner._sample_identity(cast(SciPlex3GammaPoissonCandidate, object()))
