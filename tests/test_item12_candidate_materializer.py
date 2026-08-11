from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from cellstate.domain.common import canonical_json_bytes

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPOSITORY_ROOT / "scripts/materialize_sciplex3_k562_p1_candidate.py"


def _load_materializer() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "_item12_candidate_materializer_test", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


materializer = _load_materializer()


@dataclass(frozen=True)
class _Receipt:
    fingerprint: str = "1" * 64
    batch_count: int = 1
    control_well_count: int = 16
    count_scan_complete: bool = True
    feature_panel_artifact_sha256: str = "7" * 64
    full_source_umi_total: int = 10
    panel_nonzero_count: int = 5
    panel_umi_total: int = 8
    loader_contract_sha256: str = "8" * 64
    loader_implementation_sha256: str = "9" * 64
    record_count: int = 94_785
    treated_well_count: int = 752
    well_count: int = 768
    zero_panel_record_count: int = 7
    runner_panel_count_stream_sha256: str = "a" * 64
    scoring_transform_sha256: str = "b" * 64


class _Preparation:
    receipt = _Receipt()
    finalized_count_scan_receipt = SimpleNamespace(
        dataset_manifest_sha256="c" * 64,
        fingerprint="2" * 64,
    )
    training_data = SimpleNamespace(wells=(SimpleNamespace(counts=SimpleNamespace(indptr=(0, 1))),))

    @staticmethod
    def finalized_count_scan_manifest() -> dict[str, object]:
        return {"artifact_schema": "test-finalized-scan"}


def _reference(sha256: str = "a" * 64) -> dict[str, object]:
    return {"byte_count": 1, "relative_path": "test", "sha256": sha256}


def _bindings() -> dict[str, object]:
    return {
        "action_domain": _reference("3" * 64),
        "benchmark": _reference("b" * 64),
        "candidate_code": _reference(materializer.EXPECTED_CANDIDATE_CODE_SHA256),
        "candidate_runner_code": _reference(materializer.EXPECTED_CANDIDATE_RUNNER_CODE_SHA256),
        "dataset_manifest": _reference("c" * 64),
        "feature_panel": _reference("7" * 64),
        "item11_runner_code": _reference("c" * 64),
        "loader_code": _reference("9" * 64),
        "loader_contract": _reference("8" * 64),
        "materializer_code": _reference("d" * 64),
        "query": _reference("b" * 64),
        "scoring_transform": _reference("b" * 64),
        "target_value_schema": _reference("e" * 64),
    }


def _canonical_output(root: Path) -> Path:
    return Path(root / materializer.BENCHMARK_RELATIVE_DIRECTORY / "item12-p1")


def _configure_fake_transaction(
    monkeypatch: pytest.MonkeyPatch,
    *,
    final_check_raises: bool,
) -> tuple[str, list[str]]:
    fingerprint = hashlib.sha256(b"fake-item12-manifest").hexdigest()
    calls: list[str] = []
    preparation = _Preparation()
    plan = SimpleNamespace(
        benchmark_fingerprint="b" * 64,
        support_envelope_fingerprint="e" * 64,
    )

    monkeypatch.setattr(materializer, "_require_reference_runtime", lambda: b"runtime-lock")
    monkeypatch.setattr(materializer, "_repository_bindings", lambda _: _bindings())
    monkeypatch.setattr(materializer, "_verify_imported_module_provenance", lambda *_: None)
    monkeypatch.setattr(
        materializer,
        "_planned_support_envelope",
        lambda _: ("e" * 64, b"support-envelope"),
    )

    def prepare(*_: object) -> _Preparation:
        calls.append("prepare")
        return preparation

    def build(*_: object, **__: object) -> SimpleNamespace:
        calls.append("build-plan")
        return plan

    def seal(_preparation: object, _plan: object, directory: Path) -> SimpleNamespace:
        calls.append("seal-plan")
        directory.mkdir()
        plan_path = directory / materializer.TRAINING_PLAN
        plan_path.write_bytes(b"plan")
        names_and_payloads = {
            "candidate-specification.json": b"specification",
            "output-model-schema.json": b"schema",
            "p1-count-stream-descriptor.json": b"counts",
            "runtime-lock.json": b"runtime-lock",
        }
        supports: list[SimpleNamespace] = []
        for name, payload in names_and_payloads.items():
            path = directory / name
            path.write_bytes(payload)
            supports.append(SimpleNamespace(path=path))
        return SimpleNamespace(
            plan=plan,
            preparation_fingerprint=preparation.receipt.fingerprint,
            artifact=SimpleNamespace(path=plan_path),
            support_artifacts=tuple(supports),
        )

    observation = SimpleNamespace()

    def fit(_preparation: object, _sealed: object, directory: Path) -> SimpleNamespace:
        calls.append("fit")
        directory.mkdir()
        model = directory / materializer.CANDIDATE_MODEL
        observed = directory / materializer.TRAINING_OBSERVATION
        model.write_bytes(b"model")
        observed.write_bytes(b"observation")
        return SimpleNamespace(
            model_artifact=SimpleNamespace(path=model),
            observation_artifact=SimpleNamespace(path=observed),
            observation=observation,
        )

    monkeypatch.setattr(materializer, "_prepare_exact_p1", prepare)
    monkeypatch.setattr(materializer, "build_sciplex3_candidate_training_plan", build)
    monkeypatch.setattr(materializer, "seal_sciplex3_candidate_training_plan", seal)
    monkeypatch.setattr(materializer, "fit_and_write_sciplex3_candidate", fit)
    monkeypatch.setattr(materializer, "verify_sciplex3_candidate_fit", lambda *_: observation)
    monkeypatch.setattr(materializer, "_peak_rss_bytes", lambda: 1)
    monkeypatch.setattr(
        materializer,
        "_build_manifest",
        lambda *_args, **_kwargs: {
            "artifact_schema": "fake-item12-materialization",
            "safety_boundary": dict(materializer._SAFETY_BOUNDARY),
        },
    )
    monkeypatch.setattr(materializer, "_check_manifest", lambda *_args, **_kwargs: fingerprint)

    if final_check_raises:

        def fail_check(*_: object, **__: object) -> str:
            raise materializer.CandidateMaterializationError("post-install drift")

        monkeypatch.setattr(materializer, "check_materialization", fail_check)
    else:
        monkeypatch.setattr(
            materializer,
            "check_materialization",
            lambda *_args, **_kwargs: fingerprint,
        )
    return fingerprint, calls


def test_canonical_output_layout_keeps_count_descriptor_outside_vertical_tree() -> None:
    assert materializer.DEFAULT_OUTPUT_DIRECTORY == (
        REPOSITORY_ROOT / "benchmarks/vertical-a/sciplex3-k562-24h-v1/item12-p1"
    )
    assert (
        Path("benchmarks/artifacts/sciplex3-k562-24h-v1/item12-p1/p1-count-stream-descriptor.json")
        == materializer.COUNT_DESCRIPTOR_RELATIVE_PATH
    )
    assert {
        "candidate-specification.json": Path(
            "benchmarks/vertical-a/sciplex3-k562-24h-v1/support/candidate-specification.json"
        ),
        "output-model-schema.json": Path(
            "benchmarks/vertical-a/sciplex3-k562-24h-v1/support/candidate-output-model-schema.json"
        ),
        "runtime-lock.json": Path(
            "benchmarks/vertical-a/sciplex3-k562-24h-v1/support/candidate-runtime-lock.json"
        ),
    } == materializer.SUPPORT_RELATIVE_PATHS


def test_all_direct_materialization_entrypoints_retire_before_source_or_output_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_accessed = False

    def source_access(*_: object) -> None:
        nonlocal source_accessed
        source_accessed = True

    monkeypatch.setattr(materializer, "_prepare_exact_p1", source_access)
    for entrypoint in (
        materializer.materialize,
        materializer._retired_materialize_implementation,
    ):
        with pytest.raises(
            materializer.CandidateMaterializationError,
            match="legacy direct materialization is retired",
        ):
            entrypoint(
                tmp_path / "protected-source.h5ad",
                tmp_path / "canonical-output",
                repository_root=tmp_path,
            )
    assert source_accessed is False
    assert not (tmp_path / "canonical-output").exists()


def test_check_path_never_calls_source_preparation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "repo"
    output = _canonical_output(root)
    output.mkdir(parents=True)
    manifest_payload = canonical_json_bytes({"artifact_schema": "test"})
    (output / materializer.MATERIALIZATION_MANIFEST).write_bytes(manifest_payload)
    monkeypatch.setattr(materializer, "_repository_bindings", lambda _: _bindings())
    monkeypatch.setattr(
        materializer,
        "_prepare_exact_p1",
        lambda *_: pytest.fail("cheap checker attempted source preparation"),
    )
    expected = hashlib.sha256(manifest_payload).hexdigest()
    monkeypatch.setattr(materializer, "_check_manifest", lambda *_args, **_kwargs: expected)
    assert materializer.check_materialization(output, repository_root=root) == expected


def test_manifest_records_only_deterministic_resource_gates_and_false_authority() -> None:
    preparation = _Preparation()
    plan = SimpleNamespace(
        action_binding_sha256="3" * 64,
        benchmark_fingerprint="4" * 64,
        candidate_specification=SimpleNamespace(sha256="5" * 64),
        fingerprint="6" * 64,
        ordered_feature_keys_sha256="7" * 64,
        output_model_schema=SimpleNamespace(sha256="8" * 64),
        p1_count_stream_sha256="9" * 64,
        p1_design_fingerprint="a" * 64,
        query_fingerprint="b" * 64,
        runtime_lock=SimpleNamespace(sha256="c" * 64),
        support_envelope_fingerprint="d" * 64,
        target_value_schema_sha256="e" * 64,
        optimization_seed=0,
    )
    observation = SimpleNamespace(
        initial_equilibration_sha256="1" * 64,
        inner_equilibration_trace_sha256="2" * 64,
        model_artifact_sha256="f" * 64,
        training_nuisance_rho_sha256="3" * 64,
        fingerprint="0" * 64,
    )
    manifest = materializer._build_manifest(
        preparation,
        plan,
        observation,
        {"candidate_model": (Path("candidate-model.json"), b"model")},
        _bindings(),
        support_envelope_fingerprint="d" * 64,
    )
    assert manifest["resource_gates"] == {
        "fit_peak_rss": {
            "limit_bytes": 4 * 1024**3,
            "measurement": "resource.getrusage(RUSAGE_SELF).ru_maxrss Linux KiB",
            "within_limit": True,
        },
        "fit_wall_clock": {
            "limit_seconds": 3_600,
            "measurement": "time.monotonic around fit_and_write_sciplex3_candidate",
            "within_limit": True,
        },
    }
    assert manifest["artifact_schema_version"] == "5.0.0"
    assert manifest["exact_bindings"]["software_golden_model_sha256"] == (
        materializer.SCIPLEX3_CANDIDATE_GOLDEN_MODEL_SHA256
    )
    assert manifest["exact_bindings"]["software_golden_sample_sha256"] == (
        materializer.SCIPLEX3_CANDIDATE_GOLDEN_SAMPLE_SHA256
    )
    assert manifest["exact_bindings"]["candidate_model_schema_version"] == "5.0.0"
    assert manifest["exact_bindings"]["fixed_factor_shape"] == 0.1
    assert manifest["exact_bindings"]["initial_equilibration_sha256"] == "1" * 64
    assert manifest["exact_bindings"]["inner_equilibration_trace_sha256"] == "2" * 64
    assert manifest["exact_bindings"]["training_nuisance_rho_sha256"] == "3" * 64
    assert manifest["scope"] == {
        "access_purpose": "train_parameters",
        "batch_size": 512,
        "candidate_implementation_version": "5.0.0",
        "candidate_model_id": materializer.SCIPLEX3_CANDIDATE_MODEL_ID,
        "candidate_model_schema": materializer.SCIPLEX3_CANDIDATE_MODEL_SCHEMA,
        "candidate_model_schema_version": "5.0.0",
        "capture_latent_present": False,
        "factor_shape_mode": "fixed",
        "feature_count": 2_000,
        "fixed_factor_shape": 0.1,
        "optimization_seed": 0,
        "partition_id": "p1-train",
        "plate_context_family": "neutral-unit-context",
        "plate_sigma_present": False,
    }
    assert manifest["safety_boundary"] == materializer._SAFETY_BOUNDARY
    assert all(
        value is False
        for key, value in manifest["safety_boundary"].items()
        if key != "accessed_partition_roles"
    )
    serialized = canonical_json_bytes(manifest)
    assert b"elapsed" not in serialized
    assert b"observed_rss" not in serialized
    assert b"attestation" not in serialized


def test_nested_artifact_and_receipt_schemas_reject_unknown_or_authority_fields(
    tmp_path: Path,
) -> None:
    reference = {
        "byte_count": 0,
        "media_type": "application/json",
        "relative_path": "artifact.json",
        "sha256": hashlib.sha256(b"").hexdigest(),
        "admission_authority": False,
    }
    with pytest.raises(materializer.CandidateMaterializationError, match="field closure"):
        materializer._resolve_reference(
            reference,
            repository_root=tmp_path,
            overrides=None,
            name="adversarial artifact",
        )

    runtime_lock = {
        "artifact_schema": "sciplex3-candidate-runtime-lock",
        "artifact_schema_version": "1.0.0",
        "runtime": dict(materializer.SCIPLEX3_CANDIDATE_REFERENCE_RUNTIME),
        "thread_environment": {key: "1" for key in materializer._THREAD_ENVIRONMENT_KEYS},
        "can_mint_lifecycle_evidence": False,
    }
    with pytest.raises(materializer.CandidateMaterializationError, match="field closure"):
        materializer._require_exact_fields(
            runtime_lock,
            materializer._RUNTIME_LOCK_FIELDS,
            name="candidate runtime lock",
        )

    descriptor_authority = {
        "can_mint_lifecycle_evidence": False,
        "heldout_memberships_read": False,
        "heldout_outcomes_read": False,
        "scientifically_admissible": False,
        "trusted_workflow_receipt_present": False,
    }
    with pytest.raises(materializer.CandidateMaterializationError, match="field closure"):
        materializer._require_exact_fields(
            descriptor_authority,
            materializer._COUNT_DESCRIPTOR_AUTHORITY_FIELDS,
            name="p1 count-stream authority",
        )

    scan = json.loads(
        (
            REPOSITORY_ROOT
            / materializer.BENCHMARK_RELATIVE_DIRECTORY
            / "item11-p1/p1-finalized-count-scan-receipt.json"
        ).read_bytes()
    )
    assembly = json.loads(
        (
            REPOSITORY_ROOT
            / materializer.BENCHMARK_RELATIVE_DIRECTORY
            / "item11-p1/p1-assembly-receipt.json"
        ).read_bytes()
    )
    materializer._validated_finalized_scan(scan, name="valid finalized p1 scan")
    materializer._validated_assembly(assembly, name="valid p1 assembly")

    forged_scan = dict(scan)
    forged_scan["artifact_schema_version"] = "4.0.0"
    with pytest.raises(materializer.CandidateMaterializationError, match="safety/schema"):
        materializer._validated_finalized_scan(forged_scan, name="forged finalized p1 scan")
    forged_scan = dict(scan)
    forged_scan["admission_authority_issued"] = False
    with pytest.raises(materializer.CandidateMaterializationError, match="field closure"):
        materializer._validated_finalized_scan(forged_scan, name="forged finalized p1 scan")
    forged_scan = dict(scan)
    forged_scan["heldout_memberships_parsed"] = True
    with pytest.raises(materializer.CandidateMaterializationError, match="safety/schema"):
        materializer._validated_finalized_scan(forged_scan, name="forged finalized p1 scan")

    forged_assembly = dict(assembly)
    forged_assembly["candidate_fit_receipt_issued"] = False
    with pytest.raises(materializer.CandidateMaterializationError, match="field closure"):
        materializer._validated_assembly(forged_assembly, name="forged p1 assembly")
    forged_assembly = dict(assembly)
    forged_assembly["heldout_outcomes_read"] = True
    with pytest.raises(materializer.CandidateMaterializationError, match=r"schema|safety"):
        materializer._validated_assembly(forged_assembly, name="forged p1 assembly")


def test_cheap_checker_rejects_a_pre_v4_materialization_header(
    tmp_path: Path,
) -> None:
    manifest = {
        "artifact_schema": "sciplex3-k562-p1-candidate-materialization",
        "artifact_schema_version": "2.0.0",
    }
    with pytest.raises(materializer.CandidateMaterializationError, match="header drifted"):
        materializer._check_manifest(
            manifest,
            canonical_json_bytes(manifest),
            repository_root=tmp_path,
            repository_bindings={},
        )


def test_cheap_checker_reconstructs_a_real_runner_artifact_closure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from test_item11_sciplex3_runner import _SyntheticExactP1Loader
    from test_item12_sciplex3_candidate_runner import _candidate_for, _reference_runtime

    import cellstate.evaluation.sciplex3_candidate_runner as runner
    from cellstate.evaluation.sciplex3_runner import assemble_sciplex3_p1_training_data

    monkeypatch.setattr(runner, "_observed_runtime", _reference_runtime)
    preparation = assemble_sciplex3_p1_training_data(
        _SyntheticExactP1Loader(), REPOSITORY_ROOT, batch_size=4_096
    )
    support_fingerprint, _ = materializer._planned_support_envelope(REPOSITORY_ROOT)
    benchmark_payload = (
        REPOSITORY_ROOT / materializer.BENCHMARK_RELATIVE_DIRECTORY / "benchmark-artifact.json"
    ).read_bytes()
    plan = runner.build_sciplex3_candidate_training_plan(
        preparation,
        benchmark_fingerprint=hashlib.sha256(benchmark_payload).hexdigest(),
        support_envelope_fingerprint=support_fingerprint,
    )
    sealed = runner.seal_sciplex3_candidate_training_plan(preparation, plan, tmp_path / "sealed")
    candidate = _candidate_for(preparation)
    monkeypatch.setattr(runner, "_fit_exact_candidate", lambda *_: candidate)
    fitted = runner.fit_and_write_sciplex3_candidate(preparation, sealed, tmp_path / "fit")
    observation = runner.verify_sciplex3_candidate_fit(preparation, sealed, fitted)

    output_relative = materializer.BENCHMARK_RELATIVE_DIRECTORY / "item12-p1"
    output = REPOSITORY_ROOT / output_relative
    stage_output = tmp_path / "final-output"
    stage_output.mkdir()
    (stage_output / materializer.TRAINING_PLAN).write_bytes(sealed.artifact.path.read_bytes())
    (stage_output / materializer.CANDIDATE_MODEL).write_bytes(
        fitted.model_artifact.path.read_bytes()
    )
    (stage_output / materializer.TRAINING_OBSERVATION).write_bytes(
        fitted.observation_artifact.path.read_bytes()
    )
    (stage_output / materializer.FINALIZED_SCAN_RECEIPT).write_bytes(
        canonical_json_bytes(preparation.finalized_count_scan_manifest())
    )
    (stage_output / materializer.ASSEMBLY_RECEIPT).write_bytes(
        canonical_json_bytes(asdict(preparation.receipt))
    )
    sealed_support = {artifact.path.name: artifact.path for artifact in sealed.support_artifacts}
    stage_support = {
        materializer.SUPPORT_RELATIVE_PATHS[name]: sealed_support[name]
        for name in (
            "candidate-specification.json",
            "output-model-schema.json",
            "runtime-lock.json",
        )
    }
    stage_count = sealed_support["p1-count-stream-descriptor.json"]
    payloads = materializer._artifact_payloads(
        stage_output=stage_output,
        stage_support=stage_support,
        stage_count_descriptor=stage_count,
        output_directory=output,
        repository_root=REPOSITORY_ROOT,
    )
    bindings = materializer._repository_bindings(REPOSITORY_ROOT)
    manifest = materializer._build_manifest(
        preparation,
        plan,
        observation,
        payloads,
        bindings,
        support_envelope_fingerprint=support_fingerprint,
    )
    manifest_payload = canonical_json_bytes(manifest)
    overrides = {
        relative_path: (
            stage_output / relative_path.relative_to(output_relative)
            if relative_path.is_relative_to(output_relative)
            else stage_support.get(relative_path, stage_count)
        )
        for relative_path, _ in payloads.values()
    }
    assert (
        materializer._check_manifest(
            manifest,
            manifest_payload,
            repository_root=REPOSITORY_ROOT,
            repository_bindings=bindings,
            overrides=overrides,
        )
        == hashlib.sha256(manifest_payload).hexdigest()
    )


def test_cli_check_rejects_a_source_argument_without_touching_it(tmp_path: Path) -> None:
    source = tmp_path / "must-not-be-opened.h5ad"
    import subprocess

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--check",
            "--source-h5ad",
            str(source),
        ],
        cwd=REPOSITORY_ROOT,
        env={**os.environ, "PYTHONPATH": str(REPOSITORY_ROOT / "src")},
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "--check never accepts or opens --source-h5ad" in result.stderr
    assert not source.exists()
