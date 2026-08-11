"""Deterministic builder checks for the p1-trained, still non-runnable component."""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from cellstate.backends import (
    TRAINED_CANDIDATE_FACTORY_INTERFACE,
    BiologicalStagePort,
    BundleContractKind,
    CandidateTrainingPlan,
    ComponentLifecycleStage,
    PortDisposition,
    PortImplementationBinding,
    PortImplementationKind,
    assess_biological_model_bundle,
)
from cellstate.data import BenchmarkArtifact, ContentAddressedArtifact, DatasetManifest
from cellstate.domain import StateQuery
from cellstate.domain.common import canonical_json_bytes
from cellstate.evaluation.sciplex3_candidate import (
    SCIPLEX3_CANDIDATE_FIXED_FACTOR_SHAPE,
    SCIPLEX3_CANDIDATE_GOLDEN_MODEL_SHA256,
    SCIPLEX3_CANDIDATE_GOLDEN_SAMPLE_SHA256,
    SCIPLEX3_CANDIDATE_IMPLEMENTATION_VERSION,
    SCIPLEX3_CANDIDATE_MODEL_ID,
    SCIPLEX3_CANDIDATE_MODEL_SCHEMA,
    SCIPLEX3_CANDIDATE_MODEL_SCHEMA_VERSION,
    SCIPLEX3_CANDIDATE_OUTPUT_MODEL_SCHEMA_SHA256,
    SCIPLEX3_CANDIDATE_REFERENCE_RUNTIME,
    SCIPLEX3_CANDIDATE_SPECIFICATION_SHA256,
    SciPlex3CandidateTrainingSummary,
    build_sciplex3_synthetic_golden_candidate,
    candidate_model_schema_manifest,
    candidate_specification_manifest,
)
from cellstate.evaluation.sciplex3_candidate_runner import (
    SCIPLEX3_CANDIDATE_RUNNER_IMPLEMENTATION_VERSION,
    SCIPLEX3_CANDIDATE_TRAINING_PLAN_ID,
    SCIPLEX3_CANDIDATE_TRAINING_PLAN_VERSION,
    SciPlex3CandidateTrainingObservation,
    _sample_identity,
)

ROOT = Path(__file__).resolve().parents[1]
RAW_BASE = "https://raw.githubusercontent.com/logannye/cellstate/main"
sys.path.insert(0, str(ROOT / "scripts"))
import build_sciplex3_k562_trained_candidate as builder  # noqa: E402
import materialize_sciplex3_k562_p1_candidate as materializer  # noqa: E402


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _artifact(
    path: Path,
    payload: bytes,
    *,
    artifact_id: str,
    media_type: str = "application/json",
) -> ContentAddressedArtifact:
    return ContentAddressedArtifact(
        artifact_id=artifact_id,
        uri=f"{RAW_BASE}/{path.as_posix()}",
        sha256=_sha256(payload),
        byte_count=len(payload),
        media_type=media_type,
    )


def _repository_json(path: Path) -> dict[str, object]:
    value = json.loads((ROOT / path).read_bytes())
    assert type(value) is dict
    return value


def _exact_feature_keys() -> tuple[str, ...]:
    panel = _repository_json(builder.FEATURE_PANEL_PATH)
    features = panel["features"]
    assert type(features) is list
    return tuple(f"{item['ensembl_id']}|{item['gene_symbol']}" for item in features)


def _write_synthetic_runner_closure(
    tmp_path: Path,
) -> tuple[Path, Path, CandidateTrainingPlan]:
    candidate_directory = tmp_path / "candidate"
    support_directory = tmp_path / "support"
    candidate_directory.mkdir(parents=True)
    support_directory.mkdir(parents=True)

    support = builder.build_trained_candidate_support_envelope()
    query = StateQuery.model_validate_json((ROOT / builder.QUERY_PATH).read_bytes())
    benchmark = BenchmarkArtifact.model_validate_json((ROOT / builder.BENCHMARK_PATH).read_bytes())
    item11_scan_payload = (
        ROOT / builder.ITEM11_DIRECTORY / "p1-finalized-count-scan-receipt.json"
    ).read_bytes()
    scan = json.loads(item11_scan_payload)
    scan["python_version"] = "3.11.15 (synthetic Linux x86_64 reference)"
    scan_payload = canonical_json_bytes(scan)
    assembly = json.loads(
        (ROOT / builder.ITEM11_DIRECTORY / "p1-assembly-receipt.json").read_bytes()
    )
    assembly["finalized_count_scan_fingerprint"] = _sha256(scan_payload)
    assembly_payload = canonical_json_bytes(assembly)
    item11_materialization = _repository_json(
        builder.ITEM11_DIRECTORY / "materialization-manifest.json"
    )
    exact_bindings = item11_materialization["exact_bindings"]
    assert type(exact_bindings) is dict
    (candidate_directory / builder.FINALIZED_SCAN_FILENAME).write_bytes(scan_payload)
    (candidate_directory / builder.ASSEMBLY_FILENAME).write_bytes(assembly_payload)

    design_sha256 = "7" * 64
    candidate = replace(
        build_sciplex3_synthetic_golden_candidate(),
        ordered_feature_keys=_exact_feature_keys(),
        training_summary=SciPlex3CandidateTrainingSummary(
            record_count=scan["record_count"],
            well_count=scan["well_count"],
            zero_panel_record_count=scan["zero_panel_record_count"],
            design_sha256=design_sha256,
            training_data_sha256="8" * 64,
            provenance="real-p1",
        ),
    )
    model_payload = candidate.canonical_model_bytes()
    specification_payload = canonical_json_bytes(candidate_specification_manifest())
    model_schema_payload = canonical_json_bytes(candidate_model_schema_manifest())
    runtime_payload = canonical_json_bytes(
        {
            "artifact_schema": "sciplex3-candidate-runtime-lock",
            "artifact_schema_version": "1.0.0",
            "runtime": dict(SCIPLEX3_CANDIDATE_REFERENCE_RUNTIME),
            "thread_environment": {key: "1" for key in builder._THREAD_ENVIRONMENT_KEYS},
        }
    )
    descriptor_payload = canonical_json_bytes(
        {
            "artifact_schema": "sciplex3-p1-candidate-count-stream-descriptor",
            "artifact_schema_version": "1.0.0",
            "assembly_fingerprint": _sha256(assembly_payload),
            "candidate_design_fingerprint": design_sha256,
            "count_stream_encoding": scan["count_stream_encoding"],
            "finalized_count_scan_fingerprint": _sha256(scan_payload),
            "ordered_feature_keys_sha256": scan["ordered_feature_keys_sha256"],
            "panel_count_stream_sha256": scan["panel_count_stream_sha256"],
            "panel_nonzero_count": scan["panel_nonzero_count"],
            "panel_umi_total": scan["panel_umi_total"],
            "record_count": scan["record_count"],
            "training_partition_ids": ["p1-train"],
            "well_count": scan["well_count"],
            "zero_panel_record_count": scan["zero_panel_record_count"],
            "authority": {
                "can_mint_lifecycle_evidence": False,
                "heldout_memberships_read": False,
                "heldout_outcomes_read": False,
                "scientifically_admissible": False,
            },
        }
    )
    sealed_payloads = {
        "candidate-specification.json": specification_payload,
        "output-model-schema.json": model_schema_payload,
        "p1-count-stream-descriptor.json": descriptor_payload,
        "runtime-lock.json": runtime_payload,
    }
    for filename, payload in sealed_payloads.items():
        (support_directory / filename).write_bytes(payload)

    loader_payload = (ROOT / builder.P1_LOADER_CONTRACT_PATH).read_bytes()
    candidate_code = (ROOT / builder.CANDIDATE_CODE_PATH).read_bytes()
    runner_code = (ROOT / builder.CANDIDATE_RUNNER_CODE_PATH).read_bytes()
    plan = CandidateTrainingPlan(
        plan_id=SCIPLEX3_CANDIDATE_TRAINING_PLAN_ID,
        plan_version=SCIPLEX3_CANDIDATE_TRAINING_PLAN_VERSION,
        query_fingerprint=query.fingerprint,
        benchmark_fingerprint=benchmark.fingerprint,
        support_envelope_fingerprint=support.fingerprint,
        training_partition_ids=("p1-train",),
        p1_loader_contract=_artifact(
            builder.P1_LOADER_CONTRACT_PATH,
            loader_payload,
            artifact_id="sciplex3-item12-p1-loader-contract",
        ),
        p1_count_stream=_artifact(
            builder.P1_COUNT_STREAM_DESCRIPTOR_PATH,
            descriptor_payload,
            artifact_id="sciplex3-item12-p1-count-stream-descriptor",
        ),
        p1_count_stream_sha256=scan["panel_count_stream_sha256"],
        p1_finalized_count_scan_fingerprint=_sha256(scan_payload),
        p1_assembly_fingerprint=_sha256(assembly_payload),
        p1_design_fingerprint=design_sha256,
        ordered_feature_keys_sha256=scan["ordered_feature_keys_sha256"],
        action_binding_sha256=exact_bindings["action_domain_sha256"],
        target_value_schema_sha256=exact_bindings["target_value_schema_sha256"],
        candidate_specification=_artifact(
            builder.CANDIDATE_SPECIFICATION_PATH,
            specification_payload,
            artifact_id="sciplex3-item12-candidate-specification",
        ),
        output_model_schema=_artifact(
            builder.CANDIDATE_OUTPUT_MODEL_SCHEMA_PATH,
            model_schema_payload,
            artifact_id="sciplex3-item12-output-model-schema",
        ),
        runtime_lock=_artifact(
            builder.CANDIDATE_RUNTIME_LOCK_PATH,
            runtime_payload,
            artifact_id="sciplex3-item12-runtime-lock",
        ),
        trainer_implementation=PortImplementationBinding(
            implementation_id="cellstate.sciplex3-candidate-runner",
            implementation_version=SCIPLEX3_CANDIDATE_RUNNER_IMPLEMENTATION_VERSION,
            interface=(
                "cellstate.evaluation.sciplex3_candidate_runner.fit_and_write_sciplex3_candidate"
            ),
            kind=PortImplementationKind.PYTHON_ENTRY_POINT,
            code_artifact=_artifact(
                builder.CANDIDATE_RUNNER_CODE_PATH,
                runner_code,
                artifact_id="sciplex3-item12-candidate-runner-code",
                media_type="text/x-python",
            ),
            entrypoint=(
                "cellstate.evaluation.sciplex3_candidate_runner:fit_and_write_sciplex3_candidate"
            ),
        ),
        candidate_factory_implementation=PortImplementationBinding(
            implementation_id="cellstate.sciplex3-gamma-poisson-candidate-factory",
            implementation_version=SCIPLEX3_CANDIDATE_IMPLEMENTATION_VERSION,
            interface=TRAINED_CANDIDATE_FACTORY_INTERFACE,
            kind=PortImplementationKind.PYTHON_ENTRY_POINT,
            code_artifact=_artifact(
                builder.CANDIDATE_CODE_PATH,
                candidate_code,
                artifact_id="sciplex3-item12-candidate-factory-code",
                media_type="text/x-python",
            ),
            entrypoint=("cellstate.evaluation.sciplex3_candidate:SciPlex3GammaPoissonCandidate"),
        ),
        optimization_seed=0,
        deterministic_thread_count=1,
    )
    plan_payload = canonical_json_bytes(plan.model_dump(mode="json"))
    (candidate_directory / builder.PLAN_FILENAME).write_bytes(plan_payload)
    (candidate_directory / builder.MODEL_FILENAME).write_bytes(model_payload)

    golden_request, golden_sample = _sample_identity(candidate)
    behavior = candidate.behavior_manifest()
    fitted_state = candidate.fitted_state_manifest()
    tensor_sha256 = fitted_state["tensor_sha256"]
    assert type(tensor_sha256) is dict
    initial = candidate.initial_equilibration
    total_inner_sweep_count = sum(
        (index + 1) * count for index, count in enumerate(initial.inner_sweep_count_histogram)
    ) + sum(
        (index + 1) * count
        for item in candidate.trace
        for index, count in enumerate(item.inner_sweep_count_histogram)
    )
    observation = SciPlex3CandidateTrainingObservation(
        plan_fingerprint=plan.fingerprint,
        preparation_fingerprint=plan.p1_assembly_fingerprint,
        finalized_count_scan_fingerprint=plan.p1_finalized_count_scan_fingerprint,
        assembly_fingerprint=plan.p1_assembly_fingerprint,
        p1_count_stream_sha256=plan.p1_count_stream_sha256,
        p1_design_fingerprint=plan.p1_design_fingerprint,
        ordered_feature_keys_sha256=plan.ordered_feature_keys_sha256,
        action_binding_sha256=plan.action_binding_sha256,
        candidate_specification_sha256=SCIPLEX3_CANDIDATE_SPECIFICATION_SHA256,
        output_model_schema_sha256=SCIPLEX3_CANDIDATE_OUTPUT_MODEL_SCHEMA_SHA256,
        runtime_lock_sha256=plan.runtime_lock.sha256,
        loader_code_sha256=_sha256((ROOT / builder.LOADER_CODE_PATH).read_bytes()),
        item11_runner_code_sha256=_sha256((ROOT / builder.ITEM11_RUNNER_CODE_PATH).read_bytes()),
        candidate_runner_code_sha256=_sha256(runner_code),
        candidate_factory_code_sha256=_sha256(candidate_code),
        model_artifact_sha256=_sha256(model_payload),
        model_artifact_byte_count=len(model_payload),
        fitted_state_sha256=_sha256(canonical_json_bytes(fitted_state)),
        behavior_sha256=_sha256(canonical_json_bytes(behavior)),
        plate_context_rho_sha256=cast(str, tensor_sha256["rho"]),
        initial_equilibration_sha256=cast(str, fitted_state["initial_equilibration_sha256"]),
        inner_equilibration_trace_sha256=cast(
            str, fitted_state["inner_equilibration_trace_sha256"]
        ),
        golden_request_sha256=golden_request,
        golden_sample_sha256=golden_sample,
        software_golden_model_sha256=SCIPLEX3_CANDIDATE_GOLDEN_MODEL_SHA256,
        software_golden_sample_sha256=SCIPLEX3_CANDIDATE_GOLDEN_SAMPLE_SHA256,
        outer_iteration_count=cast(int, behavior["outer_iteration_count"]),
        initial_elbo=initial.elbo,
        initial_factor_order=initial.factor_order,
        initial_inner_sweep_count_histogram=initial.inner_sweep_count_histogram,
        initial_maximum_inner_sweeps=initial.maximum_inner_sweeps,
        initial_maximum_terminal_shape_residual=initial.maximum_terminal_shape_residual,
        initial_maximum_terminal_elog_residual=initial.maximum_terminal_elog_residual,
        final_elbo=cast(float, behavior["final_elbo"]),
        fixed_factor_shape=cast(float, behavior["fixed_factor_shape"]),
        inner_batch_count=cast(int, behavior["inner_batch_count"]),
        total_inner_sweep_count=total_inner_sweep_count,
        maximum_inner_sweeps=cast(int, behavior["maximum_inner_sweeps"]),
        maximum_terminal_shape_residual=cast(float, behavior["maximum_terminal_shape_residual"]),
        maximum_terminal_elog_residual=cast(float, behavior["maximum_terminal_elog_residual"]),
        loading_rank_ratio=cast(float, behavior["loading_rank_ratio"]),
        mean_activation_rank_ratio=cast(float, behavior["mean_activation_rank_ratio"]),
        minimum_factor_contribution_share=cast(
            float, behavior["minimum_factor_contribution_share"]
        ),
        terminal_elbo_relative_changes=tuple(
            cast(list[float], behavior["terminal_elbo_relative_changes"])
        ),
    )
    observation_payload = canonical_json_bytes(observation.manifest())
    (candidate_directory / builder.OBSERVATION_FILENAME).write_bytes(observation_payload)

    scan_receipt = SimpleNamespace(**scan, fingerprint=_sha256(scan_payload))
    assembly_receipt = SimpleNamespace(**assembly, fingerprint=_sha256(assembly_payload))
    preparation = SimpleNamespace(
        finalized_count_scan_receipt=scan_receipt,
        receipt=assembly_receipt,
        training_data=SimpleNamespace(
            wells=tuple(SimpleNamespace(counts=SimpleNamespace(indptr=(0, 1))) for _ in range(768))
        ),
    )
    vertical = materializer.BENCHMARK_RELATIVE_DIRECTORY / "item12-p1"
    artifact_payloads = {
        "assembly_receipt": (vertical / builder.ASSEMBLY_FILENAME, assembly_payload),
        "candidate_model": (vertical / builder.MODEL_FILENAME, model_payload),
        "candidate_output_model_schema": (
            materializer.SUPPORT_RELATIVE_PATHS["output-model-schema.json"],
            model_schema_payload,
        ),
        "candidate_runtime_lock": (
            materializer.SUPPORT_RELATIVE_PATHS["runtime-lock.json"],
            runtime_payload,
        ),
        "candidate_specification": (
            materializer.SUPPORT_RELATIVE_PATHS["candidate-specification.json"],
            specification_payload,
        ),
        "candidate_training_plan": (vertical / builder.PLAN_FILENAME, plan_payload),
        "finalized_count_scan_receipt": (
            vertical / builder.FINALIZED_SCAN_FILENAME,
            scan_payload,
        ),
        "p1_count_stream_descriptor": (
            materializer.COUNT_DESCRIPTOR_RELATIVE_PATH,
            descriptor_payload,
        ),
        "training_execution_observation": (
            vertical / builder.OBSERVATION_FILENAME,
            observation_payload,
        ),
    }
    materialization = materializer._build_manifest(
        preparation,
        plan,
        observation,
        artifact_payloads,
        materializer._repository_bindings(ROOT),
        support_envelope_fingerprint=support.fingerprint,
    )
    (candidate_directory / builder.MATERIALIZATION_FILENAME).write_bytes(
        canonical_json_bytes(materialization)
    )
    return candidate_directory, support_directory, plan


def test_prefit_support_is_exact_component_model_without_heldout_reads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reads: list[Path] = []
    original_read_bytes = Path.read_bytes

    def recording_read_bytes(path: Path) -> bytes:
        reads.append(path.resolve())
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", recording_read_bytes)
    support = builder.build_trained_candidate_support_envelope()
    assert support.bundle_kind is BundleContractKind.COMPONENT_MODEL
    assert support.envelope_version == "4.0.0-trained-candidate"
    assert support.runtime_operations == ()
    assert (
        support.fingerprint
        == hashlib.sha256(canonical_json_bytes(support.model_dump(mode="json"))).hexdigest()
    )
    assert reads == [(ROOT / builder.SUPPORT_ENVELOPE_PATH).resolve()]
    assert not any(
        token in path.as_posix()
        for path in reads
        for token in ("/memberships/", "/contexts/", "evaluation-cases")
    )
    assert not tuple(tmp_path.iterdir())


def test_builder_derives_only_deterministic_trained_candidate_declarations(
    tmp_path: Path,
) -> None:
    candidate_directory, support_directory, plan = _write_synthetic_runner_closure(tmp_path)
    built = builder.build_trained_candidate(
        candidate_directory,
        sealed_support_directory=support_directory,
    )
    assert built.training_plan == plan
    assert built.bundle.bundle_kind is BundleContractKind.COMPONENT_MODEL
    assert built.bundle.bundle_version == "4.0.0-trained-candidate"
    assert built.bundle.model_artifact == built.model_artifact
    assert built.bundle.training_run is not None
    assert built.bundle.validation_evidence == ()
    assert built.bundle.operation_implementations == ()
    assert built.training_run.training_partition_ids == ("p1-train",)
    assert built.training_run.run_version == "4.0.0"
    assert built.training_run.calibration_partition_ids == ()
    assert built.training_run.model_selection_validation_partition_ids == ()
    assert built.p1_evidence.p2_membership_read is False
    assert built.p1_evidence.p3_membership_read is False
    assert built.p1_evidence.p4_membership_read is False

    ports = {binding.port: binding for binding in built.bundle.ports}
    assert (
        ports[BiologicalStagePort.POPULATION_ASSAY_RESPONSE_DISTRIBUTION_MODEL].disposition
        is PortDisposition.PROVIDED
    )
    assert all(
        ports[port].disposition is PortDisposition.REQUIRED
        for port in built.support_envelope.required_ports
        if port is not BiologicalStagePort.POPULATION_ASSAY_RESPONSE_DISTRIBUTION_MODEL
    )
    artifact_ids = {
        artifact.artifact_id for artifact in built.training_run.training_evidence_artifacts
    }
    assert "sciplex3-item12-reviewed-dataset-manifest" in artifact_ids
    assert "sciplex3-item12-source-verification" in artifact_ids
    assert "sciplex3-item12-p1-source-workflow-resolution" in artifact_ids
    assert not any(
        forbidden in artifact_id
        for artifact_id in artifact_ids
        for forbidden in (
            "source-selection-receipt",
            "fit-receipt",
            "interface-receipt",
            "artifact-resolution-receipt",
        )
    )

    model = json.loads((candidate_directory / builder.MODEL_FILENAME).read_bytes())
    assert model["model_id"] == SCIPLEX3_CANDIDATE_MODEL_ID
    assert model["model_schema"] == SCIPLEX3_CANDIDATE_MODEL_SCHEMA
    assert model["implementation_version"] == "4.0.0"
    assert model["model_schema_version"] == SCIPLEX3_CANDIDATE_MODEL_SCHEMA_VERSION
    assert model["tensors"]["factor_shape"]["shape"] == [1]
    assert "rho" in model["tensors"]
    assert "plate_sigma" not in model["tensors"]
    assert "capture" not in model["tensors"]
    assert "q" not in model["tensors"]
    assert model["initial_equilibration"]

    observation = json.loads((candidate_directory / builder.OBSERVATION_FILENAME).read_bytes())
    assert observation["artifact_schema_version"] == "4.0.0"
    assert observation["candidate_model_schema_version"] == SCIPLEX3_CANDIDATE_MODEL_SCHEMA_VERSION
    assert observation["factor_order_stable"] is True
    assert observation["factor_shape_mode"] == "fixed"
    assert observation["factor_shape_estimated"] is False
    assert observation["fixed_factor_shape"] == SCIPLEX3_CANDIDATE_FIXED_FACTOR_SHAPE
    assert observation["inner_equilibration_performed"] is True
    assert observation["inner_all_batches_converged"] is True
    assert observation["plate_context_family"] == "uniform-whole-p1-rho-row"
    assert observation["capture_latent_present"] is False
    assert observation["plate_sigma_present"] is False


def test_deterministic_outputs_round_trip_but_do_not_self_mint_lifecycle(
    tmp_path: Path,
) -> None:
    candidate_directory, support_directory, _ = _write_synthetic_runner_closure(tmp_path)
    built = builder.build_trained_candidate(
        candidate_directory,
        sealed_support_directory=support_directory,
    )
    output_root = tmp_path / "output"
    builder.emit_trained_candidate_build(built, repository_root=output_root, check=False)
    builder.emit_trained_candidate_build(built, repository_root=output_root, check=True)

    query = StateQuery.model_validate_json((ROOT / builder.QUERY_PATH).read_bytes())
    benchmark = BenchmarkArtifact.model_validate_json((ROOT / builder.BENCHMARK_PATH).read_bytes())
    manifest = DatasetManifest.model_validate_json(
        (ROOT / builder.DATASET_MANIFEST_PATH).read_bytes()
    )
    readiness = assess_biological_model_bundle(
        built.bundle,
        query=query,
        benchmark=benchmark,
        manifests={
            binding.binding_id: manifest for binding in benchmark.definition.evidence_bindings
        },
        support_envelope=built.support_envelope,
        training_run=built.training_run,
    )
    assert readiness.component_model_declared
    assert not readiness.trained_candidate_verified
    assert readiness.lifecycle_stage is ComponentLifecycleStage.SCAFFOLD
    assert not readiness.component_execution_allowed
    assert not readiness.runtime_registration_allowed

    stale_path = output_root / builder.ITEM12_DIRECTORY / builder.MODEL_FILENAME
    stale_path.write_bytes(stale_path.read_bytes() + b"\n")
    with pytest.raises(SystemExit, match="stale"):
        builder.emit_trained_candidate_build(built, repository_root=output_root, check=True)


def test_builder_fails_closed_on_model_runtime_or_source_verification_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate_directory, support_directory, _ = _write_synthetic_runner_closure(tmp_path)
    model_path = candidate_directory / builder.MODEL_FILENAME
    model_path.write_bytes(model_path.read_bytes() + b"\n")
    with pytest.raises(builder.SciPlex3TrainedCandidateBuildError, match="source-free checker"):
        builder.build_trained_candidate(
            candidate_directory,
            sealed_support_directory=support_directory,
        )

    candidate_directory, support_directory, _ = _write_synthetic_runner_closure(
        tmp_path / "specification"
    )
    specification_path = support_directory / "candidate-specification.json"
    specification = json.loads(specification_path.read_bytes())
    specification["candidate_specification_schema_version"] = "2.0.0"
    specification_path.write_bytes(canonical_json_bytes(specification))
    with pytest.raises(builder.SciPlex3TrainedCandidateBuildError, match="source-free checker"):
        builder.build_trained_candidate(
            candidate_directory,
            sealed_support_directory=support_directory,
        )

    candidate_directory, support_directory, _ = _write_synthetic_runner_closure(tmp_path / "model")
    model_path = candidate_directory / builder.MODEL_FILENAME
    model = json.loads(model_path.read_bytes())
    model["model_schema"] = "sciplex3-gamma-poisson-pooled-shape-candidate-model-v2"
    model_path.write_bytes(canonical_json_bytes(model))
    with pytest.raises(builder.SciPlex3TrainedCandidateBuildError, match="source-free checker"):
        builder.build_trained_candidate(
            candidate_directory,
            sealed_support_directory=support_directory,
        )

    candidate_directory, support_directory, _ = _write_synthetic_runner_closure(tmp_path / "second")
    runtime_path = support_directory / "runtime-lock.json"
    runtime = json.loads(runtime_path.read_bytes())
    runtime["runtime"]["blas_version"] = "forged"
    runtime_path.write_bytes(canonical_json_bytes(runtime))
    with pytest.raises(builder.SciPlex3TrainedCandidateBuildError, match="source-free checker"):
        builder.build_trained_candidate(
            candidate_directory,
            sealed_support_directory=support_directory,
        )

    candidate_directory, support_directory, _ = _write_synthetic_runner_closure(tmp_path / "third")
    source_verification = _repository_json(builder.SOURCE_VERIFICATION_PATH)
    source = source_verification["source"]
    assert type(source) is dict
    source["release"] = "stale-release"
    forged_source_verification = tmp_path / "forged-source-verification.json"
    forged_source_verification.write_text(
        json.dumps(source_verification, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    )
    monkeypatch.setattr(builder, "SOURCE_VERIFICATION_PATH", forged_source_verification)
    with pytest.raises(builder.SciPlex3TrainedCandidateBuildError, match="selected manifest"):
        builder.build_trained_candidate(
            candidate_directory,
            sealed_support_directory=support_directory,
        )


def test_builder_rejects_pre_v4_materialization_support_and_model(
    tmp_path: Path,
) -> None:
    candidate_directory, support_directory, _ = _write_synthetic_runner_closure(
        tmp_path / "materialization"
    )
    materialization_path = candidate_directory / builder.MATERIALIZATION_FILENAME
    materialization = json.loads(materialization_path.read_bytes())
    materialization["artifact_schema_version"] = "2.0.0"
    materialization_path.write_bytes(canonical_json_bytes(materialization))
    with pytest.raises(builder.SciPlex3TrainedCandidateBuildError, match="source-free checker"):
        builder.build_trained_candidate(
            candidate_directory,
            sealed_support_directory=support_directory,
        )


@pytest.mark.parametrize(
    "mutation",
    (
        "unknown_top_level",
        "artifact_reference_authority",
        "failed_resource_gate",
        "heldout_safety_flag",
        "numeric_false_safety_flag",
        "wrong_scope",
        "wrong_exact_model_binding",
    ),
)
def test_builder_rejects_nonexact_v4_materialization_manifests(
    tmp_path: Path,
    mutation: str,
) -> None:
    candidate_directory, support_directory, _ = _write_synthetic_runner_closure(tmp_path / mutation)
    manifest_path = candidate_directory / builder.MATERIALIZATION_FILENAME
    manifest = json.loads(manifest_path.read_bytes())
    if mutation == "unknown_top_level":
        manifest["admission_authority"] = False
    elif mutation == "artifact_reference_authority":
        manifest["artifacts"]["candidate_model"]["can_mint_lifecycle_evidence"] = False
    elif mutation == "failed_resource_gate":
        manifest["resource_gates"]["fit_peak_rss"]["within_limit"] = False
    elif mutation == "heldout_safety_flag":
        manifest["safety_boundary"]["heldout_memberships_read"] = True
    elif mutation == "numeric_false_safety_flag":
        manifest["safety_boundary"]["heldout_memberships_read"] = 0
    elif mutation == "wrong_scope":
        manifest["scope"]["plate_sigma_present"] = True
    else:
        manifest["exact_bindings"]["model_artifact_sha256"] = "0" * 64
    manifest_path.write_bytes(canonical_json_bytes(manifest))
    with pytest.raises(builder.SciPlex3TrainedCandidateBuildError, match="source-free checker"):
        builder.build_trained_candidate(
            candidate_directory,
            sealed_support_directory=support_directory,
        )
