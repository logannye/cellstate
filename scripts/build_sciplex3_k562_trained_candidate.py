#!/usr/bin/env python3
"""Build the deterministic, non-runnable sci-Plex3 p1 trained-candidate bundle.

This builder never opens the source H5AD and never issues a trust receipt.  It consumes the
already sealed Item 12 runner outputs, rechecks their exact static closure, and emits ordinary
content-addressed contracts.  HMAC source-selection and fit-semantic receipts remain runtime-only
inputs to ``TrainingVerificationContext``.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, cast

from cellstate.backends.contracts import (
    BiologicalModelBundleContract,
    BiologicalStagePort,
    BiologicalSupportEnvelope,
    BundleContractKind,
    BundleContractReference,
    ModelPortBinding,
    PortDisposition,
    TrainingRunBinding,
    derive_query_prerequisite_report,
)
from cellstate.backends.training import (
    CandidateTrainingPlan,
    P1TrainingEvidence,
)
from cellstate.data import (
    BenchmarkArtifact,
    BenchmarkPartitionRole,
    ContentAddressedArtifact,
    DatasetManifest,
    SourceArtifact,
)
from cellstate.domain.common import canonical_json_bytes
from cellstate.domain.query import StateQuery
from cellstate.evaluation.sciplex3_candidate import (
    SCIPLEX3_CANDIDATE_FIXED_FACTOR_SHAPE,
    SCIPLEX3_CANDIDATE_GOLDEN_MODEL_SHA256,
    SCIPLEX3_CANDIDATE_GOLDEN_SAMPLE_SHA256,
    SCIPLEX3_CANDIDATE_IMPLEMENTATION_VERSION,
    SCIPLEX3_CANDIDATE_MODEL_ID,
    SCIPLEX3_CANDIDATE_MODEL_SCHEMA_VERSION,
    SCIPLEX3_CANDIDATE_OUTPUT_MODEL_SCHEMA_SHA256,
    SCIPLEX3_CANDIDATE_REFERENCE_RUNTIME,
    SCIPLEX3_CANDIDATE_SPECIFICATION_SHA256,
    SciPlex3GammaPoissonCandidate,
    candidate_model_schema_manifest,
    candidate_specification_manifest,
    load_sciplex3_candidate,
)
from cellstate.evaluation.sciplex3_candidate_runner import (
    SCIPLEX3_CANDIDATE_RUNNER_IMPLEMENTATION_VERSION,
    SCIPLEX3_CANDIDATE_TRAINING_PLAN_ID,
    SCIPLEX3_CANDIDATE_TRAINING_PLAN_VERSION,
    SciPlex3CandidateTrainingObservation,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RAW_BASE = "https://raw.githubusercontent.com/logannye/cellstate/main"

COMPONENT_DIRECTORY = Path("backends/vertical-a/sciplex3-k562-24h-v1")
SUPPORT_ENVELOPE_PATH = COMPONENT_DIRECTORY / "support-envelope.json"
TRAINING_RUN_PATH = COMPONENT_DIRECTORY / "training-run.json"
BUNDLE_PATH = COMPONENT_DIRECTORY / "bundle-contract.json"

BENCHMARK_DIRECTORY = Path("benchmarks/vertical-a/sciplex3-k562-24h-v1")
ITEM11_DIRECTORY = BENCHMARK_DIRECTORY / "item11-p1"
ITEM12_DIRECTORY = BENCHMARK_DIRECTORY / "item12-p1"
QUERY_PATH = BENCHMARK_DIRECTORY / "state-query.json"
BENCHMARK_PATH = BENCHMARK_DIRECTORY / "benchmark-artifact.json"
P1_LOADER_CONTRACT_PATH = BENCHMARK_DIRECTORY / "support/p1-loader-contract.json"
CANDIDATE_SPECIFICATION_PATH = BENCHMARK_DIRECTORY / "support/candidate-specification.json"
CANDIDATE_OUTPUT_MODEL_SCHEMA_PATH = (
    BENCHMARK_DIRECTORY / "support/candidate-output-model-schema.json"
)
CANDIDATE_RUNTIME_LOCK_PATH = BENCHMARK_DIRECTORY / "support/candidate-runtime-lock.json"
ACTION_DOMAIN_PATH = BENCHMARK_DIRECTORY / "support/action-domain-mapping.json"
TARGET_VALUE_SCHEMA_PATH = BENCHMARK_DIRECTORY / "support/target-value-schema.json"
FEATURE_PANEL_PATH = Path("benchmarks/artifacts/sciplex3-k562-24h-v1/feature-panel.json")
SOURCE_VERIFICATION_PATH = Path(
    "benchmarks/artifacts/sciplex3-k562-24h-v1/source-verification.json"
)
P1_COUNT_STREAM_DESCRIPTOR_PATH = (
    Path("benchmarks/artifacts/sciplex3-k562-24h-v1/item12-p1") / "p1-count-stream-descriptor.json"
)
DATASET_MANIFEST_PATH = Path("data_manifests/reviewed/sciplex3-k562-24h.json")
CANDIDATE_CODE_PATH = Path("src/cellstate/evaluation/sciplex3_candidate.py")
CANDIDATE_RUNNER_CODE_PATH = Path("src/cellstate/evaluation/sciplex3_candidate_runner.py")
ITEM11_RUNNER_CODE_PATH = Path("src/cellstate/evaluation/sciplex3_runner.py")
LOADER_CODE_PATH = Path("src/cellstate/backends/sciplex3_loader.py")
MATERIALIZER_PATH = Path("scripts/materialize_sciplex3_k562_p1_candidate.py")

PLAN_FILENAME = "candidate-training-plan.json"
MODEL_FILENAME = "candidate-model.json"
OBSERVATION_FILENAME = "training-execution-observation.json"
FINALIZED_SCAN_FILENAME = "p1-finalized-count-scan-receipt.json"
ASSEMBLY_FILENAME = "p1-assembly-receipt.json"
MATERIALIZATION_FILENAME = "materialization-manifest.json"
SEALED_SUPPORT_FILENAMES = (
    "candidate-specification.json",
    "output-model-schema.json",
    "p1-count-stream-descriptor.json",
    "runtime-lock.json",
)

_SUPPORT_OUTPUT_BY_INPUT = {
    "candidate-specification.json": CANDIDATE_SPECIFICATION_PATH,
    "output-model-schema.json": CANDIDATE_OUTPUT_MODEL_SCHEMA_PATH,
    "p1-count-stream-descriptor.json": P1_COUNT_STREAM_DESCRIPTOR_PATH,
    "runtime-lock.json": CANDIDATE_RUNTIME_LOCK_PATH,
}
_THREAD_ENVIRONMENT_KEYS = (
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)
EXPECTED_CANDIDATE_CODE_SHA256 = "f316edbc4f3204686d2d9d7a0a7fbc1d809dcac61601416f7f02323dece152b8"
EXPECTED_CANDIDATE_RUNNER_CODE_SHA256 = (
    "7d8ae937d1188979b461a94f39f7a9bddc3c7e793d1c4ce00134722b81a928c4"
)


class SciPlex3TrainedCandidateBuildError(ValueError):
    """Raised when deterministic Item 12 inputs do not close exactly."""


def _load_materializer_checker(repository_root: Path) -> ModuleType:
    path = repository_root / MATERIALIZER_PATH
    spec = importlib.util.spec_from_file_location("_cellstate_item12_materializer_checker", path)
    if spec is None or spec.loader is None:
        raise SciPlex3TrainedCandidateBuildError("cannot load the Item 12 materializer checker")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as error:
        sys.modules.pop(spec.name, None)
        raise SciPlex3TrainedCandidateBuildError(
            "Item 12 materializer checker import failed"
        ) from error
    return module


def _validate_materialization_with_checker(
    candidate_directory: Path,
    repository_root: Path,
    *,
    sealed_support_directory: Path | None,
    count_stream_descriptor_path: Path,
) -> None:
    module = _load_materializer_checker(repository_root)
    checker = getattr(module, "check_materialization_inputs", None)
    if not callable(checker):
        raise SciPlex3TrainedCandidateBuildError(
            "Item 12 materializer lacks the source-free staged checker"
        )
    try:
        fingerprint = cast(Callable[..., str], checker)(
            candidate_directory,
            repository_root=repository_root,
            sealed_support_directory=sealed_support_directory,
            count_stream_descriptor_path=count_stream_descriptor_path,
        )
    except Exception as error:
        raise SciPlex3TrainedCandidateBuildError(
            "Item 12 materialization failed the exact source-free checker"
        ) from error
    manifest_payload = _read(
        candidate_directory / MATERIALIZATION_FILENAME,
        name="Item 12 materialization manifest",
    )
    if fingerprint != _sha256(manifest_payload):
        raise SciPlex3TrainedCandidateBuildError(
            "Item 12 materialization checker returned a contradictory fingerprint"
        )


@dataclass(frozen=True, slots=True)
class SciPlex3TrainedCandidateBuild:
    """Fully validated deterministic outputs; none is a trust or authority receipt."""

    support_envelope: BiologicalSupportEnvelope
    training_plan: CandidateTrainingPlan
    p1_evidence: P1TrainingEvidence
    training_run: TrainingRunBinding
    bundle: BiologicalModelBundleContract
    model_artifact: ContentAddressedArtifact
    training_result_artifact: ContentAddressedArtifact
    workflow_resolution_artifacts: tuple[ContentAddressedArtifact, ...]
    outputs: tuple[tuple[Path, bytes], ...]

    def __post_init__(self) -> None:
        if type(self.outputs) is not tuple or not self.outputs:
            raise SciPlex3TrainedCandidateBuildError("build outputs must be a nonempty tuple")
        paths = tuple(path for path, _ in self.outputs)
        if paths != tuple(sorted(paths, key=lambda value: value.as_posix())):
            raise SciPlex3TrainedCandidateBuildError("build outputs must be canonically sorted")
        if len(paths) != len(set(paths)):
            raise SciPlex3TrainedCandidateBuildError("build output paths must be unique")
        if any(type(payload) is not bytes or not payload for _, payload in self.outputs):
            raise SciPlex3TrainedCandidateBuildError("build output payloads must be exact bytes")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _read(path: Path, *, name: str) -> bytes:
    try:
        return path.read_bytes()
    except OSError as error:
        raise SciPlex3TrainedCandidateBuildError(f"missing {name}: {path}") from error


def _json_object(payload: bytes, *, name: str) -> dict[str, object]:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SciPlex3TrainedCandidateBuildError(f"invalid JSON for {name}") from error
    if type(value) is not dict or any(type(key) is not str for key in value):
        raise SciPlex3TrainedCandidateBuildError(f"{name} must be an exact JSON object")
    if canonical_json_bytes(value) != payload:
        raise SciPlex3TrainedCandidateBuildError(f"{name} must use canonical JSON bytes")
    return cast(dict[str, object], value)


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


def _require_artifact_payload(
    artifact: ContentAddressedArtifact,
    payload: bytes,
    *,
    expected_path: Path,
    expected_artifact_id: str,
    expected_media_type: str,
    name: str,
) -> None:
    expected = _artifact(
        expected_path,
        payload,
        artifact_id=expected_artifact_id,
        media_type=expected_media_type,
    )
    if artifact != expected:
        raise SciPlex3TrainedCandidateBuildError(
            f"{name} declaration differs from exact current bytes or URI"
        )


def _reference(
    *,
    contract_id: str,
    contract_version: str,
    path: Path,
    payload: bytes,
    artifact_id: str,
) -> BundleContractReference:
    return BundleContractReference(
        contract_id=contract_id,
        contract_version=contract_version,
        artifact=_artifact(path, payload, artifact_id=artifact_id),
    )


def build_trained_candidate_support_envelope(
    *, repository_root: Path = REPOSITORY_ROOT
) -> BiologicalSupportEnvelope:
    """Return the pre-fit COMPONENT_MODEL envelope without opening benchmark descendants."""

    # The checked-in envelope already carries authenticated opaque query/benchmark references.
    # Reading the benchmark itself here would parse held-out case metadata before the p1 fit.
    payload = _read(
        Path(repository_root).resolve() / SUPPORT_ENVELOPE_PATH,
        name="existing population-response support envelope",
    )
    try:
        existing = BiologicalSupportEnvelope.model_validate_json(payload)
    except ValueError as error:
        raise SciPlex3TrainedCandidateBuildError("existing support envelope is invalid") from error
    if payload != canonical_json_bytes(existing.model_dump(mode="json")):
        raise SciPlex3TrainedCandidateBuildError("existing support envelope is not canonical JSON")
    result = BiologicalSupportEnvelope.model_validate(
        {
            **existing.model_dump(mode="python"),
            "bundle_kind": BundleContractKind.COMPONENT_MODEL,
            "envelope_version": "4.0.0-trained-candidate",
            "notes": tuple(
                sorted(
                    (
                        "Endpoint output remains a predictive distribution of raw integer UMI "
                        "vectors on the exact ordered 2,000-feature panel.",
                        "Forecast causal status is predictive_association and realized "
                        "intracellular exposure remains unknown.",
                        "One exact p1-trained candidate may be declared, but calibration, model "
                        "selection, validation, and every public runtime operation remain closed.",
                        "The incompatible v4 candidate fixes the Gamma factor shape at 0.1, uses "
                        "one empirical whole-p1 rho row per plate, and has no q/capture or plate "
                        "sigma latent.",
                        "Static plate and assigned-action context are inputs, not a t=0 hidden-"
                        "state prior or endpoint-response lookup.",
                    )
                )
            ),
        }
    )
    return result


def trained_candidate_support_bytes(*, repository_root: Path = REPOSITORY_ROOT) -> bytes:
    """Canonical bytes used by the runner before the final model exists."""

    support = build_trained_candidate_support_envelope(repository_root=repository_root)
    return canonical_json_bytes(support.model_dump(mode="json"))


def _load_plan(candidate_directory: Path) -> tuple[CandidateTrainingPlan, bytes]:
    payload = _read(candidate_directory / PLAN_FILENAME, name="candidate training plan")
    _json_object(payload, name="candidate training plan")
    try:
        plan = CandidateTrainingPlan.model_validate_json(payload)
    except ValueError as error:
        raise SciPlex3TrainedCandidateBuildError("invalid candidate training plan") from error
    if canonical_json_bytes(plan.model_dump(mode="json")) != payload:
        raise SciPlex3TrainedCandidateBuildError("candidate training plan is not canonical")
    return plan, payload


def _load_observation(
    candidate_directory: Path,
) -> tuple[SciPlex3CandidateTrainingObservation, bytes]:
    payload = _read(
        candidate_directory / OBSERVATION_FILENAME,
        name="candidate training execution observation",
    )
    raw = _json_object(payload, name="candidate training execution observation")
    expected_keys = set(SciPlex3CandidateTrainingObservation.__dataclass_fields__)
    if set(raw) != expected_keys:
        raise SciPlex3TrainedCandidateBuildError(
            "candidate training observation has an unexpected schema"
        )
    partition_ids = raw.get("training_partition_ids")
    if type(partition_ids) is not list or partition_ids != ["p1-train"]:
        raise SciPlex3TrainedCandidateBuildError(
            "candidate training observation partition scope drifted"
        )
    raw["training_partition_ids"] = tuple(partition_ids)
    for field_name in (
        "initial_factor_order",
        "initial_inner_sweep_count_histogram",
        "terminal_elbo_relative_changes",
    ):
        values = raw.get(field_name)
        if type(values) is not list:
            raise SciPlex3TrainedCandidateBuildError(
                f"candidate training observation {field_name} must be an exact array"
            )
        raw[field_name] = tuple(values)
    try:
        observation = SciPlex3CandidateTrainingObservation(**cast(Any, raw))
    except (TypeError, ValueError) as error:
        raise SciPlex3TrainedCandidateBuildError(
            "candidate training observation failed exact reconstruction"
        ) from error
    if canonical_json_bytes(observation.manifest()) != payload:
        raise SciPlex3TrainedCandidateBuildError(
            "candidate training observation changed on reconstruction"
        )
    return observation, payload


def _load_repository_json(repository_root: Path, path: Path, *, name: str) -> dict[str, object]:
    return _json_object(_read(repository_root / path, name=name), name=name)


def _validate_plan_closure(
    plan: CandidateTrainingPlan,
    candidate_directory: Path,
    repository_root: Path,
    support: BiologicalSupportEnvelope,
    *,
    sealed_support_directory: Path | None,
    count_stream_descriptor_path: Path | None,
) -> tuple[dict[str, bytes], dict[str, object], dict[str, object], dict[str, object]]:
    query_payload = _read(repository_root / QUERY_PATH, name="frozen state query")
    benchmark_payload = _read(repository_root / BENCHMARK_PATH, name="benchmark artifact")
    query = StateQuery.model_validate_json(query_payload)
    benchmark = BenchmarkArtifact.model_validate_json(benchmark_payload)
    if (
        canonical_json_bytes(query.model_dump(mode="json")) != query_payload
        or canonical_json_bytes(benchmark.model_dump(mode="json")) != benchmark_payload
        or plan.query_fingerprint != query.fingerprint
        or plan.benchmark_fingerprint != benchmark.fingerprint
        or plan.support_envelope_fingerprint != support.fingerprint
        or plan.training_partition_ids != ("p1-train",)
        or plan.training_partition_roles != (BenchmarkPartitionRole.TRAIN,)
        or plan.future_calibration_plan is not None
    ):
        raise SciPlex3TrainedCandidateBuildError(
            "candidate training plan differs from the exact p1 component scope"
        )

    if sealed_support_directory is None:
        sealed_paths = {
            "candidate-specification.json": repository_root / CANDIDATE_SPECIFICATION_PATH,
            "output-model-schema.json": repository_root / CANDIDATE_OUTPUT_MODEL_SCHEMA_PATH,
            "runtime-lock.json": repository_root / CANDIDATE_RUNTIME_LOCK_PATH,
            "p1-count-stream-descriptor.json": (repository_root / P1_COUNT_STREAM_DESCRIPTOR_PATH),
        }
        if count_stream_descriptor_path is not None:
            sealed_paths["p1-count-stream-descriptor.json"] = Path(
                count_stream_descriptor_path
            ).resolve()
    else:
        sealed_support_directory = Path(sealed_support_directory).resolve()
        sealed_paths = {
            filename: sealed_support_directory / filename for filename in SEALED_SUPPORT_FILENAMES
        }
        if count_stream_descriptor_path is not None:
            sealed_paths["p1-count-stream-descriptor.json"] = Path(
                count_stream_descriptor_path
            ).resolve()
    _validate_materialization_with_checker(
        candidate_directory,
        repository_root,
        sealed_support_directory=sealed_support_directory,
        count_stream_descriptor_path=sealed_paths["p1-count-stream-descriptor.json"],
    )
    sealed = {
        filename: _read(sealed_paths[filename], name=filename)
        for filename in SEALED_SUPPORT_FILENAMES
    }
    for filename, payload in sealed.items():
        _json_object(payload, name=filename)
    expected_specification = canonical_json_bytes(candidate_specification_manifest())
    expected_model_schema = canonical_json_bytes(candidate_model_schema_manifest())
    expected_runtime_lock = canonical_json_bytes(
        {
            "artifact_schema": "sciplex3-candidate-runtime-lock",
            "artifact_schema_version": "1.0.0",
            "runtime": dict(SCIPLEX3_CANDIDATE_REFERENCE_RUNTIME),
            "thread_environment": {key: "1" for key in _THREAD_ENVIRONMENT_KEYS},
        }
    )
    if sealed["candidate-specification.json"] != expected_specification:
        raise SciPlex3TrainedCandidateBuildError("candidate specification bytes are stale")
    if sealed["output-model-schema.json"] != expected_model_schema:
        raise SciPlex3TrainedCandidateBuildError("candidate model schema bytes are stale")
    if sealed["runtime-lock.json"] != expected_runtime_lock:
        raise SciPlex3TrainedCandidateBuildError("candidate runtime-lock bytes are stale")
    if (
        _sha256(expected_specification) != SCIPLEX3_CANDIDATE_SPECIFICATION_SHA256
        or _sha256(expected_model_schema) != SCIPLEX3_CANDIDATE_OUTPUT_MODEL_SCHEMA_SHA256
    ):
        raise SciPlex3TrainedCandidateBuildError(
            "candidate support constants differ from current canonical bytes"
        )

    loader_payload = _read(repository_root / P1_LOADER_CONTRACT_PATH, name="p1 loader contract")
    candidate_code = _read(repository_root / CANDIDATE_CODE_PATH, name="candidate code")
    runner_code = _read(repository_root / CANDIDATE_RUNNER_CODE_PATH, name="candidate runner code")
    if _sha256(candidate_code) != EXPECTED_CANDIDATE_CODE_SHA256:
        raise SciPlex3TrainedCandidateBuildError(
            "candidate source differs from the independent v4 freeze"
        )
    if _sha256(runner_code) != EXPECTED_CANDIDATE_RUNNER_CODE_SHA256:
        raise SciPlex3TrainedCandidateBuildError(
            "candidate runner differs from the independent v4 freeze"
        )
    _require_artifact_payload(
        plan.p1_loader_contract,
        loader_payload,
        expected_path=P1_LOADER_CONTRACT_PATH,
        expected_artifact_id="sciplex3-item12-p1-loader-contract",
        expected_media_type="application/json",
        name="p1 loader contract",
    )
    _require_artifact_payload(
        plan.p1_count_stream,
        sealed["p1-count-stream-descriptor.json"],
        expected_path=P1_COUNT_STREAM_DESCRIPTOR_PATH,
        expected_artifact_id="sciplex3-item12-p1-count-stream-descriptor",
        expected_media_type="application/json",
        name="p1 count-stream descriptor",
    )
    _require_artifact_payload(
        plan.candidate_specification,
        expected_specification,
        expected_path=CANDIDATE_SPECIFICATION_PATH,
        expected_artifact_id="sciplex3-item12-candidate-specification",
        expected_media_type="application/json",
        name="candidate specification",
    )
    _require_artifact_payload(
        plan.output_model_schema,
        expected_model_schema,
        expected_path=CANDIDATE_OUTPUT_MODEL_SCHEMA_PATH,
        expected_artifact_id="sciplex3-item12-output-model-schema",
        expected_media_type="application/json",
        name="candidate output-model schema",
    )
    _require_artifact_payload(
        plan.runtime_lock,
        expected_runtime_lock,
        expected_path=CANDIDATE_RUNTIME_LOCK_PATH,
        expected_artifact_id="sciplex3-item12-runtime-lock",
        expected_media_type="application/json",
        name="candidate runtime lock",
    )
    _require_artifact_payload(
        plan.trainer_implementation.code_artifact,
        runner_code,
        expected_path=CANDIDATE_RUNNER_CODE_PATH,
        expected_artifact_id="sciplex3-item12-candidate-runner-code",
        expected_media_type="text/x-python",
        name="candidate runner code",
    )
    _require_artifact_payload(
        plan.candidate_factory_implementation.code_artifact,
        candidate_code,
        expected_path=CANDIDATE_CODE_PATH,
        expected_artifact_id="sciplex3-item12-candidate-factory-code",
        expected_media_type="text/x-python",
        name="candidate factory code",
    )
    if (
        plan.plan_id != SCIPLEX3_CANDIDATE_TRAINING_PLAN_ID
        or plan.plan_version != SCIPLEX3_CANDIDATE_TRAINING_PLAN_VERSION
        or plan.optimization_seed != 0
        or plan.deterministic_thread_count != 1
        or plan.trainer_implementation.implementation_version
        != SCIPLEX3_CANDIDATE_RUNNER_IMPLEMENTATION_VERSION
        or plan.candidate_factory_implementation.implementation_version
        != SCIPLEX3_CANDIDATE_IMPLEMENTATION_VERSION
        or plan.trainer_implementation.entrypoint
        != "cellstate.evaluation.sciplex3_candidate_runner:fit_and_write_sciplex3_candidate"
        or plan.candidate_factory_implementation.entrypoint
        != "cellstate.evaluation.sciplex3_candidate:SciPlex3GammaPoissonCandidate"
    ):
        raise SciPlex3TrainedCandidateBuildError(
            "candidate plan executable or deterministic-fit identity drifted"
        )

    materialization_payload = _read(
        candidate_directory / MATERIALIZATION_FILENAME,
        name="Item 12 materialization manifest",
    )
    assembly_payload = _read(
        candidate_directory / ASSEMBLY_FILENAME,
        name="Item 12 p1 assembly receipt",
    )
    scan_payload = _read(
        candidate_directory / FINALIZED_SCAN_FILENAME,
        name="Item 12 finalized count scan",
    )
    materialization = _json_object(materialization_payload, name="Item 12 materialization manifest")
    assembly = _json_object(assembly_payload, name="Item 12 p1 assembly receipt")
    scan = _json_object(scan_payload, name="Item 12 finalized count scan")
    if (
        materialization.get("artifact_schema") != "sciplex3-k562-p1-candidate-materialization"
        or materialization.get("artifact_schema_version") != "4.0.0"
    ):
        raise SciPlex3TrainedCandidateBuildError(
            "Item 12 materialization is not from the incompatible v4 candidate family"
        )
    item11_materialization = _load_repository_json(
        repository_root,
        ITEM11_DIRECTORY / MATERIALIZATION_FILENAME,
        name="Item 11 materialization manifest",
    )
    item11_assembly = _load_repository_json(
        repository_root,
        ITEM11_DIRECTORY / ASSEMBLY_FILENAME,
        name="Item 11 p1 assembly receipt",
    )
    item11_scan = _load_repository_json(
        repository_root,
        ITEM11_DIRECTORY / FINALIZED_SCAN_FILENAME,
        name="Item 11 finalized count scan",
    )
    descriptor = _json_object(
        sealed["p1-count-stream-descriptor.json"], name="p1 count-stream descriptor"
    )
    exact_bindings = cast(dict[str, object], materialization.get("exact_bindings"))
    p1_scan = cast(dict[str, object], materialization.get("p1_scan"))
    if type(exact_bindings) is not dict or type(p1_scan) is not dict:
        raise SciPlex3TrainedCandidateBuildError("Item 11 manifest structure drifted")
    expected_descriptor = {
        "artifact_schema": "sciplex3-p1-candidate-count-stream-descriptor",
        "artifact_schema_version": "1.0.0",
        "assembly_fingerprint": _sha256(assembly_payload),
        "candidate_design_fingerprint": plan.p1_design_fingerprint,
        "count_stream_encoding": scan.get("count_stream_encoding"),
        "finalized_count_scan_fingerprint": _sha256(scan_payload),
        "ordered_feature_keys_sha256": scan.get("ordered_feature_keys_sha256"),
        "panel_count_stream_sha256": scan.get("panel_count_stream_sha256"),
        "panel_nonzero_count": scan.get("panel_nonzero_count"),
        "panel_umi_total": scan.get("panel_umi_total"),
        "record_count": scan.get("record_count"),
        "training_partition_ids": ["p1-train"],
        "well_count": scan.get("well_count"),
        "zero_panel_record_count": scan.get("zero_panel_record_count"),
        "authority": {
            "can_mint_lifecycle_evidence": False,
            "heldout_memberships_read": False,
            "heldout_outcomes_read": False,
            "scientifically_admissible": False,
        },
    }
    if descriptor != expected_descriptor:
        raise SciPlex3TrainedCandidateBuildError(
            "candidate count-stream descriptor differs from exact Item 11 closure"
        )
    if (
        plan.p1_count_stream_sha256 != scan.get("panel_count_stream_sha256")
        or plan.p1_finalized_count_scan_fingerprint
        != expected_descriptor["finalized_count_scan_fingerprint"]
        or plan.p1_assembly_fingerprint != expected_descriptor["assembly_fingerprint"]
        or plan.ordered_feature_keys_sha256 != scan.get("ordered_feature_keys_sha256")
        or plan.action_binding_sha256 != exact_bindings.get("action_domain_sha256")
        or plan.target_value_schema_sha256 != exact_bindings.get("target_value_schema_sha256")
        or plan.p1_loader_contract.sha256 != exact_bindings.get("loader_contract_sha256")
        or exact_bindings.get("candidate_code_sha256") != EXPECTED_CANDIDATE_CODE_SHA256
        or exact_bindings.get("candidate_runner_code_sha256")
        != EXPECTED_CANDIDATE_RUNNER_CODE_SHA256
        or scan.get("finalized") is not True
        or scan.get("source_descriptor_reverified") is not True
        or scan.get("exact_record_coverage") is not True
        or scan.get("heldout_memberships_parsed") is not False
        or scan.get("heldout_outcome_values_parsed") is not False
        or assembly.get("can_mint_lifecycle_evidence") is not False
        or assembly.get("scientifically_admissible") is not False
        or p1_scan.get("count_scan_complete") is not True
    ):
        raise SciPlex3TrainedCandidateBuildError("candidate plan or Item 11 safety closure drifted")
    item11_exact = cast(dict[str, object], item11_materialization.get("exact_bindings"))
    item11_p1_scan = cast(dict[str, object], item11_materialization.get("p1_scan"))
    if (
        type(item11_exact) is not dict
        or type(item11_p1_scan) is not dict
        or any(
            scan.get(field) != item11_scan.get(field)
            for field in (
                "source_sha256",
                "record_count",
                "well_count",
                "treated_well_count",
                "control_well_count",
                "panel_count_stream_sha256",
                "panel_nonzero_count",
                "panel_umi_total",
                "zero_panel_record_count",
                "ordered_feature_keys_sha256",
                "query_sha256",
                "benchmark_sha256",
                "p1_loader_contract_sha256",
                "target_value_schema_sha256",
            )
        )
        or any(
            p1_scan.get(field) != item11_p1_scan.get(field)
            for field in (
                "record_count",
                "well_count",
                "treated_well_count",
                "control_well_count",
                "panel_count_stream_sha256",
                "panel_nonzero_count",
                "panel_umi_total",
                "zero_panel_record_count",
                "zero_panel_well_count",
            )
        )
        or any(
            exact_bindings.get(field) != item11_exact.get(field)
            for field in (
                "action_domain_sha256",
                "benchmark_sha256",
                "dataset_manifest_sha256",
                "feature_panel_artifact_sha256",
                "loader_contract_sha256",
                "loader_implementation_sha256",
                "query_sha256",
                "scoring_transform_sha256",
                "target_value_schema_sha256",
            )
        )
        or assembly.get("runner_panel_count_stream_sha256")
        != item11_assembly.get("runner_panel_count_stream_sha256")
    ):
        raise SciPlex3TrainedCandidateBuildError(
            "Item 12 p1 conceptual data differs from the frozen Item 11 closure"
        )
    return sealed, materialization, assembly, scan


def _validate_model_and_observation(
    plan: CandidateTrainingPlan,
    observation: SciPlex3CandidateTrainingObservation,
    candidate_directory: Path,
    repository_root: Path,
    materialization: dict[str, object],
    assembly: dict[str, object],
    scan: dict[str, object],
) -> tuple[SciPlex3GammaPoissonCandidate, bytes]:
    model_payload = _read(candidate_directory / MODEL_FILENAME, name="candidate model")
    model_sha256 = _sha256(model_payload)
    try:
        candidate = load_sciplex3_candidate(model_payload, expected_sha256=model_sha256)
    except ValueError as error:
        raise SciPlex3TrainedCandidateBuildError("candidate model failed exact reload") from error
    if type(candidate) is not SciPlex3GammaPoissonCandidate:
        raise SciPlex3TrainedCandidateBuildError("candidate loader substituted another class")
    if candidate.canonical_model_bytes() != model_payload:
        raise SciPlex3TrainedCandidateBuildError("candidate model changed on canonical reload")

    exact_bindings = cast(dict[str, object], materialization["exact_bindings"])
    p1_scan = cast(dict[str, object], materialization["p1_scan"])
    behavior = candidate.behavior_manifest()
    fitted_state = candidate.fitted_state_manifest()
    tensor_sha256 = fitted_state.get("tensor_sha256")
    if (
        type(tensor_sha256) is not dict
        or type(tensor_sha256.get("rho")) is not str
        or "plate_sigma" in tensor_sha256
    ):
        raise SciPlex3TrainedCandidateBuildError(
            "candidate fitted state lacks the exact empirical-rho/no-sigma closure"
        )
    initial = candidate.initial_equilibration
    total_inner_sweep_count = sum(
        (index + 1) * count for index, count in enumerate(initial.inner_sweep_count_histogram)
    ) + sum(
        (index + 1) * count
        for item in candidate.trace
        for index, count in enumerate(item.inner_sweep_count_histogram)
    )
    expected_observation_bindings = {
        "plan_fingerprint": plan.fingerprint,
        "preparation_fingerprint": plan.p1_assembly_fingerprint,
        "finalized_count_scan_fingerprint": plan.p1_finalized_count_scan_fingerprint,
        "assembly_fingerprint": plan.p1_assembly_fingerprint,
        "p1_count_stream_sha256": plan.p1_count_stream_sha256,
        "p1_design_fingerprint": plan.p1_design_fingerprint,
        "ordered_feature_keys_sha256": plan.ordered_feature_keys_sha256,
        "action_binding_sha256": plan.action_binding_sha256,
        "candidate_specification_sha256": plan.candidate_specification.sha256,
        "output_model_schema_sha256": plan.output_model_schema.sha256,
        "runtime_lock_sha256": plan.runtime_lock.sha256,
        "loader_code_sha256": _sha256(
            _read(repository_root / LOADER_CODE_PATH, name="sci-Plex3 loader code")
        ),
        "item11_runner_code_sha256": _sha256(
            _read(repository_root / ITEM11_RUNNER_CODE_PATH, name="Item 11 runner code")
        ),
        "candidate_runner_code_sha256": plan.trainer_implementation.code_artifact.sha256,
        "candidate_factory_code_sha256": plan.candidate_factory_implementation.code_artifact.sha256,
        "model_artifact_sha256": model_sha256,
        "model_artifact_byte_count": len(model_payload),
        "fitted_state_sha256": _sha256(canonical_json_bytes(fitted_state)),
        "behavior_sha256": _sha256(canonical_json_bytes(behavior)),
        "plate_context_rho_sha256": tensor_sha256["rho"],
        "initial_equilibration_sha256": fitted_state.get("initial_equilibration_sha256"),
        "inner_equilibration_trace_sha256": fitted_state.get("inner_equilibration_trace_sha256"),
        "software_golden_model_sha256": SCIPLEX3_CANDIDATE_GOLDEN_MODEL_SHA256,
        "software_golden_sample_sha256": SCIPLEX3_CANDIDATE_GOLDEN_SAMPLE_SHA256,
        "outer_iteration_count": behavior.get("outer_iteration_count"),
        "initial_elbo": initial.elbo,
        "initial_factor_order": initial.factor_order,
        "initial_inner_sweep_count_histogram": initial.inner_sweep_count_histogram,
        "initial_maximum_inner_sweeps": initial.maximum_inner_sweeps,
        "initial_maximum_terminal_shape_residual": (initial.maximum_terminal_shape_residual),
        "initial_maximum_terminal_elog_residual": initial.maximum_terminal_elog_residual,
        "final_elbo": behavior.get("final_elbo"),
        "fixed_factor_shape": behavior.get("fixed_factor_shape"),
        "inner_batch_count": behavior.get("inner_batch_count"),
        "total_inner_sweep_count": total_inner_sweep_count,
        "maximum_inner_sweeps": behavior.get("maximum_inner_sweeps"),
        "maximum_terminal_shape_residual": behavior.get("maximum_terminal_shape_residual"),
        "maximum_terminal_elog_residual": behavior.get("maximum_terminal_elog_residual"),
        "loading_rank_ratio": behavior.get("loading_rank_ratio"),
        "mean_activation_rank_ratio": behavior.get("mean_activation_rank_ratio"),
        "minimum_factor_contribution_share": behavior.get("minimum_factor_contribution_share"),
        "terminal_elbo_relative_changes": tuple(
            cast(list[float], behavior.get("terminal_elbo_relative_changes"))
        ),
        "candidate_model_schema_version": behavior.get("model_schema_version"),
        "fit_converged": behavior.get("fit_converged"),
        "all_parameters_finite": behavior.get("all_parameters_finite"),
        "factor_order_stable": behavior.get("factor_order_stable"),
        "factor_shape_mode": behavior.get("factor_shape_mode"),
        "factor_shape_estimated": behavior.get("factor_shape_estimated"),
        "inner_equilibration_performed": behavior.get("inner_equilibration_performed"),
        "inner_all_batches_converged": behavior.get("inner_all_batches_converged"),
        "plate_context_family": behavior.get("plate_context_family"),
        "plate_context_count": behavior.get("plate_context_count"),
        "plate_context_factorwise_mean_one": behavior.get("plate_context_factorwise_mean_one"),
        "capture_latent_present": behavior.get("capture_latent_present"),
        "plate_sigma_present": False,
    }
    for field_name, expected in expected_observation_bindings.items():
        if getattr(observation, field_name) != expected:
            raise SciPlex3TrainedCandidateBuildError(
                f"candidate observation {field_name} differs from current closure"
            )
    # The runner's deterministic four-draw identity is cheap and outcome-free.
    from cellstate.evaluation.sciplex3_candidate_runner import _sample_identity

    golden_request, golden_sample = _sample_identity(candidate)
    if (
        observation.golden_request_sha256 != golden_request
        or observation.golden_sample_sha256 != golden_sample
    ):
        raise SciPlex3TrainedCandidateBuildError("candidate golden sample identity drifted")
    summary = candidate.training_summary
    ordered_keys_sha256 = _sha256(canonical_json_bytes(list(candidate.ordered_feature_keys)))
    if (
        candidate.model_id != SCIPLEX3_CANDIDATE_MODEL_ID
        or summary.record_count != scan.get("record_count")
        or summary.well_count != scan.get("well_count")
        or summary.zero_panel_record_count != scan.get("zero_panel_record_count")
        or summary.design_sha256 != plan.p1_design_fingerprint
        or summary.provenance != "real-p1"
        or ordered_keys_sha256 != plan.ordered_feature_keys_sha256
        or exact_bindings.get("loader_implementation_sha256") != observation.loader_code_sha256
        or p1_scan.get("assembly_fingerprint") != plan.p1_assembly_fingerprint
        or assembly.get("runner_panel_count_stream_sha256") != plan.p1_count_stream_sha256
        or exact_bindings.get("plate_context_rho_sha256") != observation.plate_context_rho_sha256
        or exact_bindings.get("initial_equilibration_sha256")
        != observation.initial_equilibration_sha256
        or exact_bindings.get("inner_equilibration_trace_sha256")
        != observation.inner_equilibration_trace_sha256
        or exact_bindings.get("candidate_model_schema_version")
        != SCIPLEX3_CANDIDATE_MODEL_SCHEMA_VERSION
        or exact_bindings.get("fixed_factor_shape") != SCIPLEX3_CANDIDATE_FIXED_FACTOR_SHAPE
    ):
        raise SciPlex3TrainedCandidateBuildError(
            "candidate model is not bound to the exact Item 11 p1 closure"
        )
    return candidate, model_payload


def _source_for_scan(
    repository_root: Path,
    scan: dict[str, object],
) -> tuple[DatasetManifest, SourceArtifact, bytes]:
    payload = _read(repository_root / DATASET_MANIFEST_PATH, name="reviewed dataset manifest")
    try:
        manifest = DatasetManifest.model_validate_json(payload)
    except ValueError as error:
        raise SciPlex3TrainedCandidateBuildError("reviewed dataset manifest is invalid") from error
    if manifest.canonical_json_bytes != payload:
        raise SciPlex3TrainedCandidateBuildError("reviewed dataset manifest is not canonical")
    matches = tuple(
        source for source in manifest.sources if source.sha256 == scan.get("source_sha256")
    )
    if len(matches) != 1:
        raise SciPlex3TrainedCandidateBuildError(
            "finalized scan source is not unique in the reviewed manifest"
        )
    source = matches[0]
    if source.byte_count != scan.get("source_byte_count"):
        raise SciPlex3TrainedCandidateBuildError("finalized scan source byte count drifted")
    return manifest, source, payload


def _validate_source_verification(
    payload: bytes,
    *,
    source: SourceArtifact,
    scan: dict[str, object],
) -> None:
    try:
        raw_verification = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SciPlex3TrainedCandidateBuildError(
            "invalid JSON for source verification artifact"
        ) from error
    if type(raw_verification) is not dict or any(type(key) is not str for key in raw_verification):
        raise SciPlex3TrainedCandidateBuildError(
            "source verification artifact must be an exact JSON object"
        )
    verification = cast(dict[str, object], raw_verification)
    canonical_pretty = (
        json.dumps(verification, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode()
    if payload != canonical_pretty:
        raise SciPlex3TrainedCandidateBuildError(
            "source verification artifact must use canonical pretty JSON bytes"
        )
    if set(verification) != {
        "artifact_schema",
        "artifact_schema_version",
        "generator",
        "h5ad_structure",
        "limitations",
        "source",
    } or (
        verification["artifact_schema"] != "sciplex3-k562-source-verification"
        or verification["artifact_schema_version"] != "1.0.0"
    ):
        raise SciPlex3TrainedCandidateBuildError("source verification schema drifted")
    generator = verification["generator"]
    structure = verification["h5ad_structure"]
    limitations = verification["limitations"]
    source_identity = verification["source"]
    if (
        type(generator) is not dict
        or set(generator) != {"generator_id", "generator_version", "script_sha256"}
        or any(type(value) is not str or not value for value in generator.values())
        or type(structure) is not dict
        or set(structure)
        != {
            "matrix_encoding",
            "matrix_nonzero_count",
            "matrix_shape",
            "matrix_value_dtype",
            "matrix_values_are_raw_integer_umi_counts",
            "required_observation_fields",
            "source_feature_axis_encoded_byte_count",
            "source_feature_axis_encoding",
            "source_feature_axis_sha256",
            "source_feature_count",
            "source_features_with_ensembl_id",
            "source_features_with_human_ensembl_id",
            "source_features_with_mouse_ensembl_id",
        }
        or type(limitations) is not list
        or not limitations
        or any(type(item) is not str or not item for item in limitations)
        or type(source_identity) is not dict
        or set(source_identity)
        != {
            "accession",
            "byte_count",
            "filename",
            "license",
            "md5",
            "release",
            "sha256",
            "uri",
        }
    ):
        raise SciPlex3TrainedCandidateBuildError("source verification structure drifted")
    string_structure_fields = (
        "matrix_encoding",
        "matrix_value_dtype",
        "source_feature_axis_encoding",
        "source_feature_axis_sha256",
    )
    integer_structure_fields = (
        "matrix_nonzero_count",
        "source_feature_axis_encoded_byte_count",
        "source_feature_count",
        "source_features_with_ensembl_id",
        "source_features_with_human_ensembl_id",
        "source_features_with_mouse_ensembl_id",
    )
    matrix_shape = structure["matrix_shape"]
    observation_fields = structure["required_observation_fields"]
    if (
        any(type(structure[field]) is not str for field in string_structure_fields)
        or any(type(structure[field]) is not int for field in integer_structure_fields)
        or structure["matrix_values_are_raw_integer_umi_counts"] is not True
        or type(matrix_shape) is not list
        or len(matrix_shape) != 2
        or any(type(item) is not int or item <= 0 for item in matrix_shape)
        or type(observation_fields) is not list
        or not observation_fields
        or any(type(item) is not str or not item for item in observation_fields)
    ):
        raise SciPlex3TrainedCandidateBuildError("source verification H5AD schema drifted")
    expected_source_fields = {
        "accession": source.accession,
        "byte_count": source.byte_count,
        "release": source.release,
        "sha256": source.sha256,
        "uri": str(source.uri),
    }
    if (
        any(source_identity.get(key) != value for key, value in expected_source_fields.items())
        or source_identity.get("sha256") != scan.get("source_sha256")
        or source_identity.get("byte_count") != scan.get("source_byte_count")
        or source_identity.get("md5") != scan.get("source_md5")
        or type(source_identity.get("filename")) is not str
        or type(source_identity.get("license")) is not str
    ):
        raise SciPlex3TrainedCandidateBuildError(
            "source verification identity differs from the selected manifest/scan source"
        )


def build_trained_candidate(
    candidate_directory: Path | None = None,
    *,
    repository_root: Path = REPOSITORY_ROOT,
    sealed_support_directory: Path | None = None,
    count_stream_descriptor_path: Path | None = None,
) -> SciPlex3TrainedCandidateBuild:
    """Validate promoted runner outputs and build deterministic trained-candidate contracts.

    By default, inputs are read from their declared canonical repository locations.  The two
    explicit support-path arguments exist only for a caller validating an exclusive staging area;
    checked-in currentness never relies on a retained staging directory.
    """

    repository_root = Path(repository_root).resolve()
    if candidate_directory is None:
        candidate_directory = repository_root / ITEM12_DIRECTORY
    candidate_directory = Path(candidate_directory).resolve()
    support = build_trained_candidate_support_envelope(repository_root=repository_root)
    support_payload = canonical_json_bytes(support.model_dump(mode="json"))
    plan, plan_payload = _load_plan(candidate_directory)
    sealed, materialization, assembly, scan = _validate_plan_closure(
        plan,
        candidate_directory,
        repository_root,
        support,
        sealed_support_directory=sealed_support_directory,
        count_stream_descriptor_path=count_stream_descriptor_path,
    )
    observation, observation_payload = _load_observation(candidate_directory)
    _, model_payload = _validate_model_and_observation(
        plan,
        observation,
        candidate_directory,
        repository_root,
        materialization,
        assembly,
        scan,
    )
    _, source, manifest_payload = _source_for_scan(repository_root, scan)

    finalized_payload = _read(
        candidate_directory / FINALIZED_SCAN_FILENAME,
        name="Item 12 finalized count scan",
    )
    assembly_payload = _read(
        candidate_directory / ASSEMBLY_FILENAME,
        name="Item 12 assembly receipt",
    )
    materialization_payload = _read(
        candidate_directory / MATERIALIZATION_FILENAME,
        name="Item 12 materialization manifest",
    )
    finalized_artifact = _artifact(
        ITEM12_DIRECTORY / FINALIZED_SCAN_FILENAME,
        finalized_payload,
        artifact_id="sciplex3-item12-p1-finalized-count-scan",
    )
    assembly_artifact = _artifact(
        ITEM12_DIRECTORY / ASSEMBLY_FILENAME,
        assembly_payload,
        artifact_id="sciplex3-item12-p1-assembly-receipt",
    )
    materialization_artifact = _artifact(
        ITEM12_DIRECTORY / MATERIALIZATION_FILENAME,
        materialization_payload,
        artifact_id="sciplex3-item12-p1-materialization",
    )
    p1_evidence = P1TrainingEvidence(
        evidence_id="sciplex3-k562-item12-p1-training-evidence",
        evidence_version="1.0.0",
        training_plan_fingerprint=plan.fingerprint,
        partition_ids=("p1-train",),
        source=source,
        finalized_count_scan=finalized_artifact,
        assembly_receipt=assembly_artifact,
        p1_materialization=materialization_artifact,
        count_stream_sha256=plan.p1_count_stream_sha256,
        finalized_count_scan_fingerprint=plan.p1_finalized_count_scan_fingerprint,
        assembly_fingerprint=plan.p1_assembly_fingerprint,
        record_count=cast(int, scan["record_count"]),
        well_count=cast(int, scan["well_count"]),
        treated_well_count=cast(int, scan["treated_well_count"]),
        control_well_count=cast(int, scan["control_well_count"]),
        nnz=cast(int, scan["panel_nonzero_count"]),
        zero_panel_record_count=cast(int, scan["zero_panel_record_count"]),
    )
    evidence_payload = canonical_json_bytes(p1_evidence.model_dump(mode="json"))

    source_verification_payload = _read(
        repository_root / SOURCE_VERIFICATION_PATH, name="source verification artifact"
    )
    _validate_source_verification(
        source_verification_payload,
        source=source,
        scan=scan,
    )
    source_verification = _artifact(
        SOURCE_VERIFICATION_PATH,
        source_verification_payload,
        artifact_id="sciplex3-item12-source-verification",
    )
    manifest_artifact = _artifact(
        DATASET_MANIFEST_PATH,
        manifest_payload,
        artifact_id="sciplex3-item12-reviewed-dataset-manifest",
    )
    workflow_payload = canonical_json_bytes(
        {
            "artifact_schema": "sciplex3-item12-p1-source-workflow-resolution",
            "artifact_schema_version": "1.0.0",
            "access_purpose": "train_parameters",
            "allowed_partition_ids": ["p1-train"],
            "dataset_manifest_sha256": manifest_artifact.sha256,
            "selected_source": source.model_dump(mode="json"),
            "selection_rule": "unique-finalized-scan-sha256-in-reviewed-manifest-v1",
            "source_verification_sha256": source_verification.sha256,
            "training_plan_fingerprint": plan.fingerprint,
            "authority": {
                "can_mint_lifecycle_evidence": False,
                "heldout_artifacts_resolved": False,
                "heldout_memberships_read": False,
                "heldout_outcomes_read": False,
                "scientifically_admissible": False,
                "trusted_receipt": False,
            },
        }
    )
    workflow_artifact = _artifact(
        ITEM12_DIRECTORY / "p1-source-workflow-resolution.json",
        workflow_payload,
        artifact_id="sciplex3-item12-p1-source-workflow-resolution",
    )
    workflow_resolution_artifacts = tuple(
        sorted(
            (manifest_artifact, source_verification, workflow_artifact),
            key=lambda item: item.artifact_id,
        )
    )

    plan_artifact = _artifact(
        ITEM12_DIRECTORY / PLAN_FILENAME,
        plan_payload,
        artifact_id="sciplex3-item12-candidate-training-plan",
    )
    evidence_artifact = _artifact(
        ITEM12_DIRECTORY / "p1-training-evidence.json",
        evidence_payload,
        artifact_id="sciplex3-item12-p1-training-evidence",
    )
    training_result_artifact = _artifact(
        ITEM12_DIRECTORY / OBSERVATION_FILENAME,
        observation_payload,
        artifact_id="sciplex3-item12-candidate-training-result",
    )
    model_artifact = _artifact(
        ITEM12_DIRECTORY / MODEL_FILENAME,
        model_payload,
        artifact_id="sciplex3-item12-p1-trained-candidate-model",
        media_type="application/vnd.cellstate.candidate+json",
    )
    deterministic_training_evidence = tuple(
        sorted(
            {
                item.artifact_id: item
                for item in (
                    plan_artifact,
                    evidence_artifact,
                    plan.p1_loader_contract,
                    plan.p1_count_stream,
                    plan.candidate_specification,
                    plan.output_model_schema,
                    plan.runtime_lock,
                    plan.trainer_implementation.code_artifact,
                    plan.candidate_factory_implementation.code_artifact,
                    finalized_artifact,
                    assembly_artifact,
                    materialization_artifact,
                    training_result_artifact,
                    *workflow_resolution_artifacts,
                )
            }.values(),
            key=lambda item: item.artifact_id,
        )
    )
    training_run = TrainingRunBinding(
        run_id="vertical-a.sciplex3-k562-24h.p1-candidate-training-run",
        run_version="4.0.0",
        query_fingerprint=plan.query_fingerprint,
        benchmark_fingerprint=plan.benchmark_fingerprint,
        support_envelope_fingerprint=plan.support_envelope_fingerprint,
        model_artifact=model_artifact,
        training_partition_ids=("p1-train",),
        training_evidence_artifacts=deterministic_training_evidence,
    )
    training_run_payload = canonical_json_bytes(training_run.model_dump(mode="json"))
    support_reference = _reference(
        contract_id=support.envelope_id,
        contract_version=support.envelope_version,
        path=SUPPORT_ENVELOPE_PATH,
        payload=support_payload,
        artifact_id="sciplex3-k562-population-response-support-envelope",
    )
    training_run_reference = _reference(
        contract_id=training_run.run_id,
        contract_version=training_run.run_version,
        path=TRAINING_RUN_PATH,
        payload=training_run_payload,
        artifact_id="sciplex3-k562-item12-training-run",
    )

    scaffold_bundle_payload = _read(
        repository_root / BUNDLE_PATH, name="existing population-response bundle"
    )
    try:
        scaffold_bundle = BiologicalModelBundleContract.model_validate_json(scaffold_bundle_payload)
    except ValueError as error:
        raise SciPlex3TrainedCandidateBuildError(
            "existing population-response bundle is invalid"
        ) from error
    if scaffold_bundle_payload != canonical_json_bytes(scaffold_bundle.model_dump(mode="json")):
        raise SciPlex3TrainedCandidateBuildError(
            "existing population-response bundle is not canonical JSON"
        )
    ports: list[ModelPortBinding] = []
    for binding in scaffold_bundle.ports:
        if binding.port is BiologicalStagePort.POPULATION_ASSAY_RESPONSE_DISTRIBUTION_MODEL:
            ports.append(
                ModelPortBinding(
                    port=binding.port,
                    disposition=PortDisposition.PROVIDED,
                    implementation=plan.candidate_factory_implementation,
                    rationale=(
                        "Exact p1-trained candidate factory; calibration, validation, and public "
                        "runtime authority remain absent.",
                    ),
                )
            )
        elif binding.port is BiologicalStagePort.MODEL_ENSEMBLE:
            ports.append(
                binding.model_copy(
                    update={
                        "rationale": (
                            "Only one p1-trained candidate is declared; no calibrated or selected "
                            "model ensemble exists.",
                        )
                    }
                )
            )
        else:
            ports.append(binding)
    bundle = BiologicalModelBundleContract(
        bundle_id="vertical-a.sciplex3-k562-24h.population-response",
        bundle_version="4.0.0-trained-candidate",
        bundle_kind=BundleContractKind.COMPONENT_MODEL,
        description=(
            "Exact p1-trained K562 well-context and assigned-action to 24-hour recovered-nucleus "
            "count-distribution candidate; calibration, selection, validation, and public runtime "
            "remain closed."
        ),
        posterior_schema_id=None,
        query=support.query,
        benchmark=support.benchmark,
        support_envelope=support_reference,
        model_artifact=model_artifact,
        training_run=training_run_reference,
        validation_evidence=(),
        ports=tuple(sorted(ports, key=lambda item: item.port.value)),
        operation_implementations=(),
    )
    prerequisite_report = derive_query_prerequisite_report(
        query=StateQuery.model_validate_json(_read(repository_root / QUERY_PATH, name="query")),
        support_envelope=support,
        bundle=bundle,
    )
    if not prerequisite_report.structurally_satisfied:
        raise SciPlex3TrainedCandidateBuildError(
            "trained-candidate bundle violates exact query prerequisites"
        )
    provided = tuple(
        binding.port for binding in bundle.ports if binding.disposition is PortDisposition.PROVIDED
    )
    if provided != (BiologicalStagePort.POPULATION_ASSAY_RESPONSE_DISTRIBUTION_MODEL,):
        raise SciPlex3TrainedCandidateBuildError(
            "trained-candidate bundle must provide exactly the population-response port"
        )
    if (
        bundle.operation_implementations
        or bundle.validation_evidence
        or training_run.calibration_partition_ids
        or training_run.calibration_evidence_artifacts
        or training_run.model_selection_validation_partition_ids
        or training_run.model_selection_evidence_artifacts
        or training_run.model_selection_freeze_artifact is not None
    ):
        raise SciPlex3TrainedCandidateBuildError(
            "trained-candidate build opened a later lifecycle or runtime surface"
        )
    bundle_payload = canonical_json_bytes(bundle.model_dump(mode="json"))

    output_payloads = {
        SUPPORT_ENVELOPE_PATH: support_payload,
        TRAINING_RUN_PATH: training_run_payload,
        BUNDLE_PATH: bundle_payload,
        CANDIDATE_SPECIFICATION_PATH: sealed["candidate-specification.json"],
        CANDIDATE_OUTPUT_MODEL_SCHEMA_PATH: sealed["output-model-schema.json"],
        CANDIDATE_RUNTIME_LOCK_PATH: sealed["runtime-lock.json"],
        P1_COUNT_STREAM_DESCRIPTOR_PATH: sealed["p1-count-stream-descriptor.json"],
        ITEM12_DIRECTORY / PLAN_FILENAME: plan_payload,
        ITEM12_DIRECTORY / MODEL_FILENAME: model_payload,
        ITEM12_DIRECTORY / OBSERVATION_FILENAME: observation_payload,
        ITEM12_DIRECTORY / FINALIZED_SCAN_FILENAME: finalized_payload,
        ITEM12_DIRECTORY / ASSEMBLY_FILENAME: assembly_payload,
        ITEM12_DIRECTORY / MATERIALIZATION_FILENAME: materialization_payload,
        ITEM12_DIRECTORY / "p1-training-evidence.json": evidence_payload,
        ITEM12_DIRECTORY / "p1-source-workflow-resolution.json": workflow_payload,
    }
    return SciPlex3TrainedCandidateBuild(
        support_envelope=support,
        training_plan=plan,
        p1_evidence=p1_evidence,
        training_run=training_run,
        bundle=bundle,
        model_artifact=model_artifact,
        training_result_artifact=training_result_artifact,
        workflow_resolution_artifacts=workflow_resolution_artifacts,
        outputs=tuple(sorted(output_payloads.items(), key=lambda item: item[0].as_posix())),
    )


def emit_trained_candidate_build(
    build: SciPlex3TrainedCandidateBuild,
    *,
    repository_root: Path = REPOSITORY_ROOT,
    check: bool,
) -> None:
    """Write validated outputs atomically per file, or perform a source-free currentness check."""

    repository_root = Path(repository_root).resolve()
    if check:
        for relative_path, payload in build.outputs:
            path = repository_root / relative_path
            if not path.exists() or path.read_bytes() != payload:
                raise SystemExit(f"generated trained-candidate artifact is stale: {path}")
        return

    staged: list[tuple[Path, Path]] = []
    try:
        for relative_path, payload in build.outputs:
            path = repository_root / relative_path
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(f".{path.name}.item12-build.tmp")
            descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
            try:
                with os.fdopen(descriptor, "wb") as handle:
                    handle.write(payload)
                    handle.flush()
                    os.fsync(handle.fileno())
            except BaseException:
                temporary.unlink(missing_ok=True)
                raise
            staged.append((temporary, path))
        for temporary, path in staged:
            temporary.replace(path)
    finally:
        for temporary, _ in staged:
            temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-directory", type=Path)
    parser.add_argument("--sealed-support-directory", type=Path)
    parser.add_argument("--count-stream-descriptor", type=Path)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--print-support-fingerprint", action="store_true")
    args = parser.parse_args()
    support = build_trained_candidate_support_envelope()
    if args.print_support_fingerprint:
        print(f"support_envelope_fingerprint {support.fingerprint}")
        if args.candidate_directory is None:
            return
    build = build_trained_candidate(
        args.candidate_directory,
        sealed_support_directory=args.sealed_support_directory,
        count_stream_descriptor_path=args.count_stream_descriptor,
    )
    emit_trained_candidate_build(build, check=args.check)
    print(f"support_envelope_fingerprint {build.support_envelope.fingerprint}")
    print(f"training_plan_fingerprint {build.training_plan.fingerprint}")
    print(f"training_run_fingerprint {build.training_run.fingerprint}")
    print(f"model_artifact_sha256 {build.model_artifact.sha256}")
    print(f"bundle_fingerprint {build.bundle.fingerprint}")
    print("static_declaration trained_candidate")
    print("trusted_lifecycle_stage scaffold")
    print("runtime_operations 0")
    print("trusted_receipts_written 0")


if __name__ == "__main__":
    main()
