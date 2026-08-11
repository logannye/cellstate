"""P1-only fit boundary for the frozen sci-Plex3 Gamma--Poisson candidate.

The runner consumes the already authenticated Item 11 preparation, an immutable pre-fit
``CandidateTrainingPlan``, and the exact Item 12 candidate factory.  It never resolves a p2, p3,
or p4 artifact, never computes an evaluation metric, and never emits lifecycle or scientific
admission authority.  Its outputs are ordinary content-addressed execution observations.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, Literal, cast

import numpy as np
import scipy

import cellstate.evaluation.sciplex3_candidate as _candidate_module
import cellstate.evaluation.sciplex3_candidate_v5 as _candidate_v5_module
import cellstate.evaluation.sciplex3_runner as _item11
import cellstate.evaluation.sciplex3_sampling_v5 as _sampling_v5_module
from cellstate.backends.contracts import PortImplementationBinding, PortImplementationKind
from cellstate.backends.sciplex3_loader import (
    SCIPLEX3_SOURCE_BYTE_COUNT,
    SCIPLEX3_SOURCE_SHA256,
)
from cellstate.backends.training import (
    TRAINED_CANDIDATE_FACTORY_INTERFACE,
    CandidateTrainingPlan,
    candidate_training_plan_generation_seed_bytes,
)
from cellstate.data.benchmarks import BenchmarkPartitionRole, ContentAddressedArtifact
from cellstate.domain.common import canonical_json_bytes
from cellstate.errors import ContractViolationError
from cellstate.evaluation.sciplex3_baselines import (
    RNG_ALGORITHM,
    CompoundDose,
    NoAction,
)
from cellstate.evaluation.sciplex3_candidate import (
    SCIPLEX3_CANDIDATE_ACTION_COUNT,
    SCIPLEX3_CANDIDATE_BATCH_SIZE,
    SCIPLEX3_CANDIDATE_COMPOUND_COUNT,
    SCIPLEX3_CANDIDATE_CONVERGENCE_RTOL,
    SCIPLEX3_CANDIDATE_CONVERGENCE_STREAK,
    SCIPLEX3_CANDIDATE_DOSES_NM,
    SCIPLEX3_CANDIDATE_FACTOR_COUNT,
    SCIPLEX3_CANDIDATE_FACTOR_ORDER_DECIMALS,
    SCIPLEX3_CANDIDATE_FIXED_FACTOR_SHAPE,
    SCIPLEX3_CANDIDATE_GOLDEN_MODEL_SHA256,
    SCIPLEX3_CANDIDATE_GOLDEN_SAMPLE_SHA256,
    SCIPLEX3_CANDIDATE_IDENTIFIABILITY_MARGIN,
    SCIPLEX3_CANDIDATE_IDENTIFIABILITY_THRESHOLD,
    SCIPLEX3_CANDIDATE_IMPLEMENTATION_VERSION,
    SCIPLEX3_CANDIDATE_INNER_RESIDUAL_TOL,
    SCIPLEX3_CANDIDATE_MAX_INNER_SWEEPS,
    SCIPLEX3_CANDIDATE_MAX_OUTER_ITERATIONS,
    SCIPLEX3_CANDIDATE_MIN_INNER_SWEEPS,
    SCIPLEX3_CANDIDATE_MIN_OUTER_ITERATIONS,
    SCIPLEX3_CANDIDATE_MODEL_ID,
    SCIPLEX3_CANDIDATE_MODEL_SCHEMA_VERSION,
    SCIPLEX3_CANDIDATE_OUTPUT_MODEL_SCHEMA_SHA256,
    SCIPLEX3_CANDIDATE_PLATE_COUNT,
    SCIPLEX3_CANDIDATE_REFERENCE_RUNTIME,
    SCIPLEX3_CANDIDATE_SPECIFICATION_SHA256,
    SCIPLEX3_CANDIDATE_TAU_GRID,
    SCIPLEX3_CANDIDATE_TRAINING_RECORD_COUNT,
    SCIPLEX3_CANDIDATE_TRAINING_WELL_COUNT,
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
    candidate_golden_model_bytes,
    candidate_model_schema_manifest,
    candidate_specification_manifest,
    training_data_fingerprint,
    verify_sciplex3_candidate_golden,
)
from cellstate.evaluation.sciplex3_runner import (
    LocalContentAddressedArtifact,
    SciPlex3BaselinePreparation,
)
from cellstate.evaluation.sciplex3_sampling_v5 import (
    SCIPLEX3_V5_MAX_COMPOUND_POISSON_INTENSITY,
    SCIPLEX3_V5_MAX_SAMPLE_COUNT,
    SCIPLEX3_V5_REQUEST_FAILURE_BUDGET_LOG,
    SCIPLEX3_V5_SAMPLING_CONTRACT_SHA256,
    V5PositiveConditionedSampler,
    V5SamplingEnvelopeCertificate,
    V5SamplingParameters,
)
from cellstate.training.execution import (
    ContainedExecutionPolicy,
    ExecutionInputClosureManifest,
    RuntimeImageIdentity,
    RuntimeImageLock,
    TrainingCodeClosureEntry,
    TrainingCodeClosureManifest,
)
from cellstate.training.publication import generation_id_for_seed

SCIPLEX3_CANDIDATE_RUNNER_IMPLEMENTATION_VERSION: Final = "5.0.0"
SCIPLEX3_CANDIDATE_TRAINING_PLAN_ID: Final = SCIPLEX3_CANDIDATE_MODEL_ID
SCIPLEX3_CANDIDATE_TRAINING_PLAN_VERSION: Final = "5.0.0"
SCIPLEX3_CANDIDATE_OPTIMIZATION_SEED: Final = 0

_P1_PARTITION_ID = "p1-train"
_RAW_BASE = "https://raw.githubusercontent.com/logannye/cellstate/main/"
_CANDIDATE_PUBLICATION_RELATIVE_PATH = (
    "backends/vertical-a/sciplex3-k562-24h-v1/candidate-publication"
)
_LOADER_CONTRACT_RELATIVE_PATH = (
    "benchmarks/vertical-a/sciplex3-k562-24h-v1/support/p1-loader-contract.json"
)
_CANDIDATE_CODE_RELATIVE_PATH = "src/cellstate/evaluation/sciplex3_candidate.py"
_CANDIDATE_V5_CODE_RELATIVE_PATH = "src/cellstate/evaluation/sciplex3_candidate_v5.py"
_SAMPLING_V5_CODE_RELATIVE_PATH = "src/cellstate/evaluation/sciplex3_sampling_v5.py"
_RUNNER_CODE_RELATIVE_PATH = "src/cellstate/evaluation/sciplex3_candidate_runner.py"
_WORKER_CODE_RELATIVE_PATH = "scripts/sciplex3_k562_v5_worker.py"
_SUPERVISOR_CODE_RELATIVE_PATH = "scripts/run_sciplex3_k562_v5_contained.py"
_EXECUTION_CODE_RELATIVE_PATH = "src/cellstate/training/execution.py"
_PUBLICATION_CODE_RELATIVE_PATH = "src/cellstate/training/publication.py"
_MATERIALIZER_CODE_RELATIVE_PATH = "scripts/materialize_sciplex3_k562_p1_candidate.py"
_BUILDER_CODE_RELATIVE_PATH = "scripts/build_sciplex3_k562_trained_candidate.py"
_CONTAINED_POLICY_RELATIVE_PATH = (
    "benchmarks/vertical-a/sciplex3-k562-24h-v1/support/contained-execution-policy.json"
)
_RUNTIME_IMAGE_LOCK_RELATIVE_PATH = (
    "benchmarks/vertical-a/sciplex3-k562-24h-v1/support/candidate-runtime-image-lock.json"
)
_TRAINING_CODE_CLOSURE_RELATIVE_PATH = (
    "benchmarks/vertical-a/sciplex3-k562-24h-v1/support/training-code-closure.json"
)
_EXECUTION_INPUT_CLOSURE_RELATIVE_PATH = (
    "benchmarks/vertical-a/sciplex3-k562-24h-v1/support/training-execution-input-closure.json"
)
_PUBLICATION_GENERATION_SEED_RELATIVE_PATH = (
    "benchmarks/vertical-a/sciplex3-k562-24h-v1/item12-p1/"
    "candidate-publication-generation-seed.json"
)
_CONTAINED_EXECUTION_ID = "sciplex3-k562-v5-fit"
_RUNTIME_IMAGE_REFERENCE = (
    "cellstate-sciplex3-v5-runtime@"
    "sha256:edd451f171161472c1a3bb6a1ae434cdedc5b776e228757ac732522c1035df18"
)
_RUNTIME_IMAGE_DIGEST = "sha256:edd451f171161472c1a3bb6a1ae434cdedc5b776e228757ac732522c1035df18"
_RUNTIME_IMAGE_INDEX_DIGEST = (
    "sha256:ababac344fae7f3d679cf9b3bbf4c46b8f3b169b358566d4abd6e3b0e7b8251e"
)
_RUNTIME_IMAGE_CONFIG_DIGEST = (
    "sha256:b9cdf1e179f149319b038f2f58bb80470c2a1b5bda8f1cf9d2ccbe17fe3b59e5"
)
_RUNTIME_IMAGE_SOURCE_DATE_EPOCH = 1_786_406_400
_THREAD_ENVIRONMENT_KEYS = (
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)
_SUPPORT_FILENAMES = (
    "candidate-specification.json",
    "contained-execution-policy.json",
    "output-model-schema.json",
    "p1-count-stream-descriptor.json",
    "publication-generation-seed.json",
    "runtime-lock.json",
    "runtime-image-lock.json",
    "training-code-closure.json",
    "training-execution-input-closure.json",
)

if _candidate_module.__file__ is None:  # pragma: no cover - import boundary
    raise ImportError("loaded sci-Plex3 candidate module has no source path")
_IMPORTED_CANDIDATE_CODE_PATH: Final = Path(_candidate_module.__file__).resolve()
_IMPORTED_CANDIDATE_CODE_SHA256: Final = hashlib.sha256(
    _IMPORTED_CANDIDATE_CODE_PATH.read_bytes()
).hexdigest()
if _candidate_v5_module.__file__ is None or _sampling_v5_module.__file__ is None:
    raise ImportError("loaded sci-Plex3 v5 objective or sampler module has no source path")
_IMPORTED_CANDIDATE_V5_CODE_PATH: Final = Path(_candidate_v5_module.__file__).resolve()
_IMPORTED_CANDIDATE_V5_CODE_SHA256: Final = hashlib.sha256(
    _IMPORTED_CANDIDATE_V5_CODE_PATH.read_bytes()
).hexdigest()
_IMPORTED_SAMPLING_V5_CODE_PATH: Final = Path(_sampling_v5_module.__file__).resolve()
_IMPORTED_SAMPLING_V5_CODE_SHA256: Final = hashlib.sha256(
    _IMPORTED_SAMPLING_V5_CODE_PATH.read_bytes()
).hexdigest()
_IMPORTED_RUNNER_CODE_PATH: Final = Path(__file__).resolve()
_IMPORTED_RUNNER_CODE_SHA256: Final = hashlib.sha256(
    _IMPORTED_RUNNER_CODE_PATH.read_bytes()
).hexdigest()


class SciPlex3CandidateRunnerError(ContractViolationError):
    """Raised when the exact p1 candidate fit boundary cannot be reconstructed."""


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_json(value: object) -> bytes:
    try:
        return canonical_json_bytes(value)
    except (TypeError, ValueError) as error:
        raise SciPlex3CandidateRunnerError("value is not canonical-JSON-compatible") from error


def _expected_v5_active_calibration_state_sha256() -> str:
    """Bind the fixed shape, neutral context, and active tau without candidate-owned state."""

    neutral_context = np.ones((1, SCIPLEX3_CANDIDATE_FACTOR_COUNT), dtype="<f8")
    return _sha256(
        _canonical_json(
            {
                "context_multipliers_sha256": _sha256(neutral_context.tobytes(order="C")),
                "context_shape": list(neutral_context.shape),
                "factor_shape_hex": SCIPLEX3_CANDIDATE_FIXED_FACTOR_SHAPE.hex(),
                "sampling_contract_sha256": SCIPLEX3_V5_SAMPLING_CONTRACT_SHA256,
                "tau_hex": (1.0).hex(),
            }
        )
    )


def _exact_sha256(value: object, *, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise SciPlex3CandidateRunnerError(f"{name} must be an exact lowercase SHA-256")
    return value


def _read_bytes(path: Path, *, name: str) -> bytes:
    try:
        return path.read_bytes()
    except OSError as error:
        raise SciPlex3CandidateRunnerError(f"missing {name}: {path}") from error


def _read_local_artifact(artifact: LocalContentAddressedArtifact, *, name: str) -> bytes:
    if type(artifact) is not LocalContentAddressedArtifact:
        raise SciPlex3CandidateRunnerError(f"{name} must use the exact local artifact type")
    payload = _read_bytes(artifact.path, name=name)
    if _sha256(payload) != artifact.sha256 or len(payload) != artifact.byte_count:
        raise SciPlex3CandidateRunnerError(f"{name} content identity drifted")
    if (
        artifact.can_mint_lifecycle_evidence is not False
        or artifact.scientifically_admissible is not False
    ):
        raise SciPlex3CandidateRunnerError(f"{name} authority flags must remain false")
    return payload


def _json_object(payload: bytes, *, name: str) -> dict[str, object]:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SciPlex3CandidateRunnerError(f"invalid JSON for {name}") from error
    if type(value) is not dict or any(type(key) is not str for key in value):
        raise SciPlex3CandidateRunnerError(f"{name} must be an exact JSON object")
    return cast(dict[str, object], value)


def _artifact_for_payload(
    *,
    artifact_id: str,
    relative_uri: str,
    payload: bytes,
    generation_id: str,
    media_type: str = "application/json",
) -> ContentAddressedArtifact:
    generation_id = _exact_sha256(generation_id, name="planned generation ID")
    return ContentAddressedArtifact(
        artifact_id=artifact_id,
        uri=(
            _RAW_BASE
            + _CANDIDATE_PUBLICATION_RELATIVE_PATH
            + f"/generations/{generation_id}/tree/"
            + relative_uri
        ),
        sha256=_sha256(payload),
        byte_count=len(payload),
        media_type=media_type,
    )


def _observed_runtime() -> dict[str, object]:
    thread_values = {key: os.environ.get(key) for key in _THREAD_ENVIRONMENT_KEYS}
    build_dependencies = np.__config__.CONFIG.get("Build Dependencies")
    if not isinstance(build_dependencies, Mapping):
        raise SciPlex3CandidateRunnerError("NumPy build dependencies are unavailable")
    blas = build_dependencies.get("blas")
    if not isinstance(blas, Mapping):
        raise SciPlex3CandidateRunnerError("NumPy BLAS build identity is unavailable")
    blas_name = blas.get("name")
    blas_version = blas.get("version")
    if (
        blas.get("found") is not True
        or type(blas_name) is not str
        or not blas_name
        or blas_name != blas_name.strip()
        or type(blas_version) is not str
        or not blas_version
        or blas_version != blas_version.strip()
    ):
        raise SciPlex3CandidateRunnerError("NumPy BLAS build identity is malformed")
    return {
        "blas_name": blas_name,
        "blas_version": blas_version,
        "byte_order": sys.byteorder,
        "numpy_version": np.__version__,
        "platform_machine": platform.machine(),
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "scipy_version": scipy.__version__,
        "single_thread": all(value == "1" for value in thread_values.values()),
    }


def _runtime_lock_payload() -> bytes:
    observed = _observed_runtime()
    if observed != dict(SCIPLEX3_CANDIDATE_REFERENCE_RUNTIME):
        raise SciPlex3CandidateRunnerError(
            "candidate fitting requires the exact frozen Python/NumPy/SciPy/OpenBLAS/x86_64 "
            "single-thread runtime"
        )
    return _canonical_json(
        {
            "artifact_schema": "sciplex3-candidate-runtime-lock",
            "artifact_schema_version": "1.0.0",
            "runtime": observed,
            "thread_environment": {key: "1" for key in _THREAD_ENVIRONMENT_KEYS},
        }
    )


def _training_code_closure(repository_root: Path) -> TrainingCodeClosureManifest:
    """Inventory every local module plus the exact worker/supervisor entry points."""

    root = Path(repository_root)
    paths = {
        path.relative_to(root).as_posix()
        for path in (root / "src/cellstate").rglob("*.py")
        if path.is_file() and not path.is_symlink()
    }
    paths.update(
        {
            _BUILDER_CODE_RELATIVE_PATH,
            _MATERIALIZER_CODE_RELATIVE_PATH,
            _SUPERVISOR_CODE_RELATIVE_PATH,
            _WORKER_CODE_RELATIVE_PATH,
        }
    )
    entries: list[TrainingCodeClosureEntry] = []
    for relative_path in sorted(paths):
        path = root / relative_path
        try:
            observed = path.lstat()
            payload = path.read_bytes()
        except OSError as error:
            raise SciPlex3CandidateRunnerError(
                f"missing training-code closure entry: {relative_path}"
            ) from error
        if path.is_symlink() or not path.is_file() or observed.st_size != len(payload):
            raise SciPlex3CandidateRunnerError(
                f"training-code closure entry is not one stable regular file: {relative_path}"
            )
        entries.append(
            TrainingCodeClosureEntry(
                relative_path=relative_path,
                sha256=_sha256(payload),
                byte_count=len(payload),
            )
        )
    manifest = TrainingCodeClosureManifest(entries=tuple(entries))
    by_path = {entry.relative_path: entry.sha256 for entry in manifest.entries}
    expected_imports = {
        _CANDIDATE_CODE_RELATIVE_PATH: _IMPORTED_CANDIDATE_CODE_SHA256,
        _CANDIDATE_V5_CODE_RELATIVE_PATH: _IMPORTED_CANDIDATE_V5_CODE_SHA256,
        _SAMPLING_V5_CODE_RELATIVE_PATH: _IMPORTED_SAMPLING_V5_CODE_SHA256,
        _RUNNER_CODE_RELATIVE_PATH: _IMPORTED_RUNNER_CODE_SHA256,
    }
    if any(by_path.get(path) != digest for path, digest in expected_imports.items()):
        raise SciPlex3CandidateRunnerError(
            "loaded v5 candidate modules differ from the exact training-code closure"
        )
    return manifest


def _training_execution_input_closure(
    repository_root: Path,
    code_closure: TrainingCodeClosureManifest,
) -> ExecutionInputClosureManifest:
    """Add only authenticated p1/public control inputs to the complete code closure."""

    root = Path(repository_root)
    loader_contract_path = root / _LOADER_CONTRACT_RELATIVE_PATH
    loader_contract_payload = _read_bytes(loader_contract_path, name="p1 loader contract")
    contract = _json_object(loader_contract_payload, name="p1 loader contract")
    artifact_declarations = contract.get("artifacts")
    if type(artifact_declarations) is not dict:
        raise SciPlex3CandidateRunnerError("p1 loader contract artifact map is malformed")
    input_paths = {
        "backends/vertical-a/sciplex3-k562-24h-v1/support-envelope.json",
        "benchmarks/vertical-a/sciplex3-k562-24h-v1/benchmark-artifact.json",
        "benchmarks/vertical-a/sciplex3-k562-24h-v1/state-query.json",
        "benchmarks/vertical-a/sciplex3-k562-24h-v1/support/action-domain-mapping.json",
        _LOADER_CONTRACT_RELATIVE_PATH,
        "benchmarks/vertical-a/sciplex3-k562-24h-v1/support/scoring-transform.json",
        "benchmarks/vertical-a/sciplex3-k562-24h-v1/support/target-value-schema.json",
        "containers/sciplex3-v5-runtime/Dockerfile",
        "containers/sciplex3-v5-runtime/requirements.lock",
        "containers/sciplex3-v5-runtime/runtime-image-lock.json",
        "data_manifests/reviewed/sciplex3-k562-24h.json",
    }
    declared_identity: dict[str, tuple[str, int]] = {}
    for declaration in artifact_declarations.values():
        if type(declaration) is not dict:
            raise SciPlex3CandidateRunnerError("p1 loader artifact declaration is malformed")
        relative = declaration.get("relative_path")
        digest = declaration.get("sha256")
        byte_count = declaration.get("byte_count")
        if type(relative) is not str or type(digest) is not str or type(byte_count) is not int:
            raise SciPlex3CandidateRunnerError("p1 loader artifact identity is malformed")
        path = (
            relative
            if relative.startswith("benchmarks/")
            else f"benchmarks/artifacts/sciplex3-k562-24h-v1/{relative}"
        )
        input_paths.add(path)
        declared_identity[path] = (digest, byte_count)

    entries = {entry.relative_path: entry for entry in code_closure.entries}
    for relative_path in sorted(input_paths):
        payload = _read_bytes(root / relative_path, name="contained execution input")
        entry = TrainingCodeClosureEntry(
            relative_path=relative_path,
            sha256=_sha256(payload),
            byte_count=len(payload),
        )
        declared = declared_identity.get(relative_path)
        if declared is not None and declared != (entry.sha256, entry.byte_count):
            raise SciPlex3CandidateRunnerError("p1 loader input differs from its contract")
        entries[relative_path] = entry
    return ExecutionInputClosureManifest(
        training_code_closure_sha256=code_closure.fingerprint,
        entries=tuple(entries[path] for path in sorted(entries)),
    )


def _contained_execution_policy(
    training_code_closure_sha256: str,
    execution_input_closure_sha256: str,
) -> ContainedExecutionPolicy:
    return ContainedExecutionPolicy(
        policy_id="sciplex3-k562-v5-contained-fit",
        owner_id="sciplex3-k562-v5",
        runtime_image=RuntimeImageIdentity(
            reference=_RUNTIME_IMAGE_REFERENCE,
            digest=_RUNTIME_IMAGE_DIGEST,
        ),
        training_code_closure_sha256=training_code_closure_sha256,
        execution_input_closure_sha256=execution_input_closure_sha256,
        wall_clock_seconds=3_600,
        cleanup_timeout_seconds=30,
        memory_max_bytes=4 * 1024**3,
        memory_swap_max_bytes=4 * 1024**3,
        pids_limit=256,
        temporary_max_bytes=256 * 1024**2,
        snapshot_max_bytes=3 * 1024**3,
        observed_training_peak_memory_bytes=1_731_055_616,
        source_container_path="/run/cellstate/source/source.h5ad",
        code_container_path="/workspace",
        output_container_path="/run/cellstate/output",
        snapshot_container_path="/run/cellstate/snapshot",
        temporary_container_path="/run/cellstate/tmp",
        workdir="/workspace",
        environment={
            "LANG": "C.UTF-8",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
            "PYTHONPATH": "/workspace/src:/workspace",
            "TMPDIR": "/run/cellstate/tmp",
            "VECLIB_MAXIMUM_THREADS": "1",
        },
        worker_command=(
            "--signal=KILL",
            "--kill-after=5s",
            "3540",
            "/opt/runtime/bin/python",
            _WORKER_CODE_RELATIVE_PATH,
            "--source",
            "/run/cellstate/source/source.h5ad",
            "--output",
            "/run/cellstate/output",
            "--repository-root",
            "/workspace",
            "--execution-id",
            _CONTAINED_EXECUTION_ID,
            "--expected-source-sha256",
            SCIPLEX3_SOURCE_SHA256,
            "--expected-source-byte-count",
            str(SCIPLEX3_SOURCE_BYTE_COUNT),
            "--snapshot-directory",
            "/run/cellstate/snapshot",
            "--snapshot-max-bytes",
            str(3 * 1024**3),
        ),
    )


def contained_training_contracts(
    repository_root: Path,
) -> tuple[
    ContainedExecutionPolicy,
    TrainingCodeClosureManifest,
    ExecutionInputClosureManifest,
    RuntimeImageLock,
]:
    """Reconstruct the source-free policy, executable closure, and image lock."""

    root = Path(repository_root)
    code_closure = _training_code_closure(root)
    input_closure = _training_execution_input_closure(repository_root, code_closure)
    policy = _contained_execution_policy(code_closure.fingerprint, input_closure.fingerprint)
    provenance_path = root / "containers/sciplex3-v5-runtime/runtime-image-lock.json"
    provenance_payload = _read_bytes(provenance_path, name="runtime image provenance lock")
    provenance = _json_object(provenance_payload, name="runtime image provenance lock")
    canonical_provenance = canonical_json_bytes(provenance)
    if provenance_payload not in {canonical_provenance, canonical_provenance + b"\n"}:
        raise SciPlex3CandidateRunnerError("runtime image provenance lock is not canonical JSON")
    build = provenance.get("build")
    if type(build) is not dict:
        raise SciPlex3CandidateRunnerError("runtime image provenance build identity is malformed")
    dockerfile = _read_bytes(
        root / "containers/sciplex3-v5-runtime/Dockerfile", name="runtime Dockerfile"
    )
    requirements = _read_bytes(
        root / "containers/sciplex3-v5-runtime/requirements.lock",
        name="runtime requirements lock",
    )
    if (
        provenance.get("image_reference") != policy.runtime_image.reference
        or provenance.get("image_digest") != policy.runtime_image.digest
        or provenance.get("platform") != policy.runtime_image.platform
        or provenance.get("operating_system") != "linux"
        or provenance.get("architecture") != "amd64"
        or provenance.get("oci_index_digest") != _RUNTIME_IMAGE_INDEX_DIGEST
        or provenance.get("config_digest") != _RUNTIME_IMAGE_CONFIG_DIGEST
        or provenance.get("distribution") != "local-oci-layout-load-required"
        or provenance.get("container_user_mode") != policy.container_user_mode
        or provenance.get("snapshot_volume_initialization") != policy.snapshot_volume_initialization
        or build.get("dockerfile_sha256") != _sha256(dockerfile)
        or build.get("requirements_sha256") != _sha256(requirements)
        or build.get("source_date_epoch") != _RUNTIME_IMAGE_SOURCE_DATE_EPOCH
        or build.get("oci_output") != "type=oci"
        or build.get("provenance_attestation_disabled") is not True
        or build.get("reproducibility_build_count") != 2
    ):
        raise SciPlex3CandidateRunnerError("runtime image provenance contradicts its exact files")
    image_lock = RuntimeImageLock(
        runtime_image=policy.runtime_image,
        container_user_mode=policy.container_user_mode,
        snapshot_volume_initialization=policy.snapshot_volume_initialization,
        training_code_closure_sha256=code_closure.fingerprint,
        image_provenance_sha256=_sha256(provenance_payload),
    )
    return policy, code_closure, input_closure, image_lock


def _current_code_payload(path: Path, imported_sha256: str, *, name: str) -> bytes:
    payload = _read_bytes(path, name=name)
    if _sha256(payload) != imported_sha256:
        raise SciPlex3CandidateRunnerError(f"{name} changed since module import")
    return payload


def _candidate_design(preparation: SciPlex3BaselinePreparation) -> SciPlex3P1DesignBindings:
    condition_to_well: dict[tuple[str, int], tuple[str, str]] = {}
    vehicles_by_plate: dict[str, list[str]] = {}
    for well in preparation.training_data.wells:
        if well.condition is None:
            vehicles_by_plate.setdefault(well.plate_id, []).append(well.well_id)
            continue
        if type(well.condition) is not CompoundDose:
            raise SciPlex3CandidateRunnerError(
                "p1 training condition has an unsupported exact type"
            )
        key = (well.condition.compound, well.condition.dose_nm)
        if key in condition_to_well:
            raise SciPlex3CandidateRunnerError("p1 candidate design repeats a treated condition")
        condition_to_well[key] = (well.well_id, well.plate_id)

    actions: list[SciPlex3P1ActionBinding] = []
    for binding in preparation.design.actions_by_source_condition.values():
        key = (binding.compound, binding.dose_nm)
        try:
            well_id, plate_id = condition_to_well.pop(key)
        except KeyError as error:
            raise SciPlex3CandidateRunnerError(
                "p1 candidate design action lacks one exact treated well"
            ) from error
        actions.append(
            SciPlex3P1ActionBinding(
                compound=binding.compound,
                dose_nm=binding.dose_nm,
                well_id=well_id,
                plate_id=plate_id,
            )
        )
    if condition_to_well:
        raise SciPlex3CandidateRunnerError("p1 training data contains an undeclared action")
    vehicles = tuple(
        SciPlex3P1VehicleBinding(plate_id=plate_id, well_ids=cast(tuple[str, str], tuple(wells)))
        for plate_id, wells in sorted(vehicles_by_plate.items())
    )
    try:
        return SciPlex3P1DesignBindings(actions=tuple(actions), vehicles=vehicles)
    except SciPlex3CandidateError as error:
        raise SciPlex3CandidateRunnerError("p1 candidate design adapter is not exact") from error


def _count_stream_descriptor_payload(
    preparation: SciPlex3BaselinePreparation,
    *,
    design: SciPlex3P1DesignBindings,
) -> bytes:
    identity = _item11._recompute_in_memory_p1_identity(preparation)
    finalized = preparation.finalized_count_scan_receipt
    return _canonical_json(
        {
            "artifact_schema": "sciplex3-p1-candidate-count-stream-descriptor",
            "artifact_schema_version": "1.0.0",
            "assembly_fingerprint": preparation.receipt.fingerprint,
            "candidate_design_fingerprint": design.fingerprint,
            "count_stream_encoding": finalized.count_stream_encoding,
            "finalized_count_scan_fingerprint": finalized.fingerprint,
            "ordered_feature_keys_sha256": identity.ordered_feature_keys_sha256,
            "panel_count_stream_sha256": identity.panel_count_stream_sha256,
            "panel_nonzero_count": identity.panel_nonzero_count,
            "panel_umi_total": identity.panel_umi_total,
            "record_count": identity.record_count,
            "training_partition_ids": [_P1_PARTITION_ID],
            "well_count": identity.well_count,
            "zero_panel_record_count": identity.zero_panel_record_count,
            "authority": {
                "can_mint_lifecycle_evidence": False,
                "heldout_memberships_read": False,
                "heldout_outcomes_read": False,
                "scientifically_admissible": False,
            },
        }
    )


def _output_model_schema_payload() -> bytes:
    payload = _canonical_json(candidate_model_schema_manifest())
    if _sha256(payload) != SCIPLEX3_CANDIDATE_OUTPUT_MODEL_SCHEMA_SHA256:
        raise SciPlex3CandidateRunnerError("candidate output model schema identity drifted")
    return payload


def _verify_factory_golden() -> None:
    payload = candidate_golden_model_bytes()
    if _sha256(payload) != SCIPLEX3_CANDIDATE_GOLDEN_MODEL_SHA256:
        raise SciPlex3CandidateRunnerError("candidate software golden model bytes drifted")
    try:
        candidate = _candidate_module.load_sciplex3_candidate(
            payload, expected_sha256=SCIPLEX3_CANDIDATE_GOLDEN_MODEL_SHA256
        )
    except SciPlex3CandidateError as error:
        raise SciPlex3CandidateRunnerError(
            "candidate software golden model failed reload"
        ) from error
    if (
        type(candidate) is not SciPlex3GammaPoissonCandidate
        or not verify_sciplex3_candidate_golden(candidate)
        or _sha256(candidate.golden_sample().samples.tobytes(order="C"))
        != SCIPLEX3_CANDIDATE_GOLDEN_SAMPLE_SHA256
    ):
        raise SciPlex3CandidateRunnerError("candidate software golden sample identity drifted")


def _expected_plan(
    preparation: SciPlex3BaselinePreparation,
    *,
    benchmark_fingerprint: str,
    support_envelope_fingerprint: str,
) -> tuple[CandidateTrainingPlan, dict[str, bytes]]:
    _item11._validate_preparation(preparation)
    benchmark_fingerprint = _exact_sha256(benchmark_fingerprint, name="benchmark fingerprint")
    support_envelope_fingerprint = _exact_sha256(
        support_envelope_fingerprint, name="support-envelope fingerprint"
    )
    design = _candidate_design(preparation)
    # The runtime gate precedes executable-golden construction so supported nonreference
    # interpreters can import this module while fitting still fails before numerical work.
    runtime_lock_payload = _runtime_lock_payload()
    _verify_factory_golden()
    repository_root = preparation.repository_root
    imported_paths = {
        _CANDIDATE_CODE_RELATIVE_PATH: _IMPORTED_CANDIDATE_CODE_PATH,
        _CANDIDATE_V5_CODE_RELATIVE_PATH: _IMPORTED_CANDIDATE_V5_CODE_PATH,
        _SAMPLING_V5_CODE_RELATIVE_PATH: _IMPORTED_SAMPLING_V5_CODE_PATH,
        _RUNNER_CODE_RELATIVE_PATH: _IMPORTED_RUNNER_CODE_PATH,
    }
    if any(
        (repository_root / relative).resolve() != loaded
        for relative, loaded in imported_paths.items()
    ):
        raise SciPlex3CandidateRunnerError(
            "loaded candidate implementation paths differ from the repository closure"
        )
    loader_payload = _read_bytes(
        repository_root / _LOADER_CONTRACT_RELATIVE_PATH,
        name="p1 loader contract",
    )
    if _sha256(loader_payload) != preparation.receipt.loader_contract_sha256:
        raise SciPlex3CandidateRunnerError(
            "p1 loader contract bytes differ from the assembly receipt"
        )
    candidate_code = _current_code_payload(
        _IMPORTED_CANDIDATE_CODE_PATH,
        _IMPORTED_CANDIDATE_CODE_SHA256,
        name="candidate factory implementation",
    )
    runner_code = _current_code_payload(
        _IMPORTED_RUNNER_CODE_PATH,
        _IMPORTED_RUNNER_CODE_SHA256,
        name="candidate runner implementation",
    )
    item11_runner = _current_code_payload(
        _item11._IMPORTED_RUNNER_CODE_PATH,
        _item11._IMPORTED_RUNNER_CODE_SHA256,
        name="Item 11 runner implementation",
    )
    if _sha256(item11_runner) != _item11._IMPORTED_RUNNER_CODE_SHA256:
        raise SciPlex3CandidateRunnerError("Item 11 runner code identity drifted")
    policy, code_closure, input_closure, image_lock = contained_training_contracts(repository_root)
    specification = _canonical_json(candidate_specification_manifest())
    if _sha256(specification) != SCIPLEX3_CANDIDATE_SPECIFICATION_SHA256:
        raise SciPlex3CandidateRunnerError("candidate specification identity drifted")
    support_payloads = {
        "candidate-specification.json": specification,
        "contained-execution-policy.json": _canonical_json(policy.model_dump(mode="json")),
        "output-model-schema.json": _output_model_schema_payload(),
        "p1-count-stream-descriptor.json": _count_stream_descriptor_payload(
            preparation, design=design
        ),
        "runtime-lock.json": runtime_lock_payload,
        "runtime-image-lock.json": _canonical_json(image_lock.model_dump(mode="json")),
        "training-code-closure.json": _canonical_json(code_closure.model_dump(mode="json")),
        "training-execution-input-closure.json": _canonical_json(
            input_closure.model_dump(mode="json")
        ),
    }
    identity = _item11._recompute_in_memory_p1_identity(preparation)

    def artifact_set(generation_id: str) -> dict[str, ContentAddressedArtifact]:
        return {
            "p1_loader_contract": _artifact_for_payload(
                artifact_id="sciplex3-item12-p1-loader-contract",
                relative_uri=_LOADER_CONTRACT_RELATIVE_PATH,
                payload=loader_payload,
                generation_id=generation_id,
            ),
            "p1_count_stream": _artifact_for_payload(
                artifact_id="sciplex3-item12-p1-count-stream-descriptor",
                relative_uri=(
                    "benchmarks/artifacts/sciplex3-k562-24h-v1/item12-p1/"
                    "p1-count-stream-descriptor.json"
                ),
                payload=support_payloads["p1-count-stream-descriptor.json"],
                generation_id=generation_id,
            ),
            "candidate_specification": _artifact_for_payload(
                artifact_id="sciplex3-item12-candidate-specification",
                relative_uri=(
                    "benchmarks/vertical-a/sciplex3-k562-24h-v1/support/"
                    "candidate-specification.json"
                ),
                payload=support_payloads["candidate-specification.json"],
                generation_id=generation_id,
            ),
            "output_model_schema": _artifact_for_payload(
                artifact_id="sciplex3-item12-output-model-schema",
                relative_uri=(
                    "benchmarks/vertical-a/sciplex3-k562-24h-v1/support/"
                    "candidate-output-model-schema.json"
                ),
                payload=support_payloads["output-model-schema.json"],
                generation_id=generation_id,
            ),
            "runtime_lock": _artifact_for_payload(
                artifact_id="sciplex3-item12-runtime-lock",
                relative_uri=(
                    "benchmarks/vertical-a/sciplex3-k562-24h-v1/support/candidate-runtime-lock.json"
                ),
                payload=support_payloads["runtime-lock.json"],
                generation_id=generation_id,
            ),
            "contained_execution_policy": _artifact_for_payload(
                artifact_id="sciplex3-item12-contained-execution-policy",
                relative_uri=_CONTAINED_POLICY_RELATIVE_PATH,
                payload=support_payloads["contained-execution-policy.json"],
                generation_id=generation_id,
            ),
            "runtime_image_lock": _artifact_for_payload(
                artifact_id="sciplex3-item12-runtime-image-lock",
                relative_uri=_RUNTIME_IMAGE_LOCK_RELATIVE_PATH,
                payload=support_payloads["runtime-image-lock.json"],
                generation_id=generation_id,
            ),
            "training_code_closure": _artifact_for_payload(
                artifact_id="sciplex3-item12-training-code-closure",
                relative_uri=_TRAINING_CODE_CLOSURE_RELATIVE_PATH,
                payload=support_payloads["training-code-closure.json"],
                generation_id=generation_id,
            ),
            "training_execution_input_closure": _artifact_for_payload(
                artifact_id="sciplex3-item12-training-execution-input-closure",
                relative_uri=_EXECUTION_INPUT_CLOSURE_RELATIVE_PATH,
                payload=support_payloads["training-execution-input-closure.json"],
                generation_id=generation_id,
            ),
            "trainer_code": _artifact_for_payload(
                artifact_id="sciplex3-item12-candidate-runner-code",
                relative_uri=_RUNNER_CODE_RELATIVE_PATH,
                payload=runner_code,
                generation_id=generation_id,
                media_type="text/x-python",
            ),
            "factory_code": _artifact_for_payload(
                artifact_id="sciplex3-item12-candidate-factory-code",
                relative_uri=_CANDIDATE_CODE_RELATIVE_PATH,
                payload=candidate_code,
                generation_id=generation_id,
                media_type="text/x-python",
            ),
        }

    def plan_fields(generation_id: str) -> dict[str, object]:
        artifacts = artifact_set(generation_id)
        return {
            "schema_version": "0.1-experimental",
            "plan_id": SCIPLEX3_CANDIDATE_TRAINING_PLAN_ID,
            "plan_version": SCIPLEX3_CANDIDATE_TRAINING_PLAN_VERSION,
            "query_fingerprint": preparation.design.query_fingerprint,
            "benchmark_fingerprint": benchmark_fingerprint,
            "support_envelope_fingerprint": support_envelope_fingerprint,
            "training_partition_ids": (_P1_PARTITION_ID,),
            "training_partition_roles": (BenchmarkPartitionRole.TRAIN,),
            "p1_loader_contract": artifacts["p1_loader_contract"],
            "p1_count_stream": artifacts["p1_count_stream"],
            "p1_count_stream_sha256": identity.panel_count_stream_sha256,
            "p1_finalized_count_scan_fingerprint": (
                preparation.finalized_count_scan_receipt.fingerprint
            ),
            "p1_assembly_fingerprint": preparation.receipt.fingerprint,
            "p1_design_fingerprint": design.fingerprint,
            "ordered_feature_keys_sha256": identity.ordered_feature_keys_sha256,
            "action_binding_sha256": preparation.design.action_domain_sha256,
            "target_value_schema_sha256": preparation.design.target_value_schema_sha256,
            "candidate_specification": artifacts["candidate_specification"],
            "output_model_schema": artifacts["output_model_schema"],
            "runtime_lock": artifacts["runtime_lock"],
            "contained_execution_policy": artifacts["contained_execution_policy"],
            "runtime_image_lock": artifacts["runtime_image_lock"],
            "training_code_closure": artifacts["training_code_closure"],
            "training_execution_input_closure": artifacts["training_execution_input_closure"],
            "trainer_implementation": PortImplementationBinding(
                implementation_id="cellstate.sciplex3-candidate-runner",
                implementation_version=SCIPLEX3_CANDIDATE_RUNNER_IMPLEMENTATION_VERSION,
                interface=(
                    "cellstate.evaluation.sciplex3_candidate_runner.fit_and_write_sciplex3_candidate"
                ),
                kind=PortImplementationKind.PYTHON_ENTRY_POINT,
                code_artifact=artifacts["trainer_code"],
                entrypoint=(
                    "cellstate.evaluation.sciplex3_candidate_runner:fit_and_write_sciplex3_candidate"
                ),
            ),
            "candidate_factory_implementation": PortImplementationBinding(
                implementation_id="cellstate.sciplex3-gamma-poisson-candidate-factory",
                implementation_version=SCIPLEX3_CANDIDATE_IMPLEMENTATION_VERSION,
                interface=TRAINED_CANDIDATE_FACTORY_INTERFACE,
                kind=PortImplementationKind.PYTHON_ENTRY_POINT,
                code_artifact=artifacts["factory_code"],
                entrypoint=(
                    "cellstate.evaluation.sciplex3_candidate:SciPlex3GammaPoissonCandidate"
                ),
            ),
            "optimization_seed": SCIPLEX3_CANDIDATE_OPTIMIZATION_SEED,
            "deterministic_thread_count": 1,
            "future_calibration_plan": None,
        }

    placeholder_generation_id = "0" * 64
    generation_seed = candidate_training_plan_generation_seed_bytes(
        plan_fields(placeholder_generation_id)
    )
    planned_generation_id = generation_id_for_seed(generation_seed)
    final_fields = plan_fields(planned_generation_id)
    publication_generation_seed = _artifact_for_payload(
        artifact_id="sciplex3-item12-publication-generation-seed",
        relative_uri=_PUBLICATION_GENERATION_SEED_RELATIVE_PATH,
        payload=generation_seed,
        generation_id=planned_generation_id,
    )
    plan = CandidateTrainingPlan.model_validate(
        {
            **final_fields,
            "planned_generation_id": planned_generation_id,
            "publication_generation_seed": publication_generation_seed,
        }
    )
    if candidate_training_plan_generation_seed_bytes(plan) != generation_seed:
        raise SciPlex3CandidateRunnerError("rendered plan changed its pre-render generation seed")
    support_payloads["publication-generation-seed.json"] = generation_seed
    return plan, support_payloads


def build_sciplex3_candidate_training_plan(
    preparation: SciPlex3BaselinePreparation,
    *,
    benchmark_fingerprint: str,
    support_envelope_fingerprint: str,
) -> CandidateTrainingPlan:
    """Build the exact pre-fit plan without resolving protected benchmark descendants."""

    plan, _ = _expected_plan(
        preparation,
        benchmark_fingerprint=benchmark_fingerprint,
        support_envelope_fingerprint=support_envelope_fingerprint,
    )
    return plan


@dataclass(frozen=True, slots=True)
class SealedSciPlex3CandidateTrainingPlan:
    """Re-read local plan bytes plus its generated, non-authorizing support descriptors."""

    plan: CandidateTrainingPlan
    artifact: LocalContentAddressedArtifact
    support_artifacts: tuple[LocalContentAddressedArtifact, ...]
    preparation_fingerprint: str
    can_mint_lifecycle_evidence: Literal[False] = False
    scientifically_admissible: Literal[False] = False

    def __post_init__(self) -> None:
        if type(self.plan) is not CandidateTrainingPlan:
            raise SciPlex3CandidateRunnerError("sealed plan must use the exact shared plan type")
        if type(self.artifact) is not LocalContentAddressedArtifact:
            raise SciPlex3CandidateRunnerError("sealed plan artifact has the wrong exact type")
        supports = tuple(self.support_artifacts)
        if any(type(item) is not LocalContentAddressedArtifact for item in supports):
            raise SciPlex3CandidateRunnerError("sealed support artifact has the wrong exact type")
        if tuple(item.path.name for item in supports) != _SUPPORT_FILENAMES:
            raise SciPlex3CandidateRunnerError(
                "sealed support artifacts are incomplete or unordered"
            )
        _exact_sha256(self.preparation_fingerprint, name="sealed preparation fingerprint")
        if (
            self.can_mint_lifecycle_evidence is not False
            or self.scientifically_admissible is not False
        ):
            raise SciPlex3CandidateRunnerError("sealed plan authority flags must be exactly false")
        object.__setattr__(self, "support_artifacts", supports)


def seal_sciplex3_candidate_training_plan(
    preparation: SciPlex3BaselinePreparation,
    plan: CandidateTrainingPlan,
    output_directory: Path,
) -> SealedSciPlex3CandidateTrainingPlan:
    """Validate and persist the pre-fit plan before any candidate fitting begins."""

    if type(plan) is not CandidateTrainingPlan:
        raise SciPlex3CandidateRunnerError("training plan must use the exact shared plan type")
    expected, support_payloads = _expected_plan(
        preparation,
        benchmark_fingerprint=plan.benchmark_fingerprint,
        support_envelope_fingerprint=plan.support_envelope_fingerprint,
    )
    if plan != expected:
        raise SciPlex3CandidateRunnerError(
            "training plan differs from the exact current p1 closure"
        )
    output = _item11._exclusive_directory(Path(output_directory))
    support_artifacts: list[LocalContentAddressedArtifact] = []
    for filename in _SUPPORT_FILENAMES:
        payload = support_payloads[filename]
        path = output / filename
        _item11._write_exclusive(path, payload)
        support_artifacts.append(
            _item11._verify_local_artifact(
                path, expected_payload=payload, media_type="application/json"
            )
        )
    plan_payload = _canonical_json(plan.model_dump(mode="json"))
    plan_path = output / "candidate-training-plan.json"
    _item11._write_exclusive(plan_path, plan_payload)
    artifact = _item11._verify_local_artifact(
        plan_path, expected_payload=plan_payload, media_type="application/json"
    )
    return SealedSciPlex3CandidateTrainingPlan(
        plan=CandidateTrainingPlan.model_validate_json(plan_payload),
        artifact=artifact,
        support_artifacts=tuple(support_artifacts),
        preparation_fingerprint=preparation.receipt.fingerprint,
    )


def _validate_sealed_plan(
    preparation: SciPlex3BaselinePreparation,
    sealed_plan: SealedSciPlex3CandidateTrainingPlan,
) -> tuple[CandidateTrainingPlan, SciPlex3P1DesignBindings]:
    if type(sealed_plan) is not SealedSciPlex3CandidateTrainingPlan:
        raise SciPlex3CandidateRunnerError("fit requires an exact sealed pre-fit plan")
    if sealed_plan.preparation_fingerprint != preparation.receipt.fingerprint:
        raise SciPlex3CandidateRunnerError("sealed plan is bound to another p1 preparation")
    plan = sealed_plan.plan
    expected, support_payloads = _expected_plan(
        preparation,
        benchmark_fingerprint=plan.benchmark_fingerprint,
        support_envelope_fingerprint=plan.support_envelope_fingerprint,
    )
    if plan != expected:
        raise SciPlex3CandidateRunnerError("sealed plan is stale or reconstructed incorrectly")
    plan_payload = _read_local_artifact(sealed_plan.artifact, name="sealed training plan")
    if plan_payload != _canonical_json(plan.model_dump(mode="json")):
        raise SciPlex3CandidateRunnerError("sealed training plan bytes differ from its binding")
    reparsed = CandidateTrainingPlan.model_validate_json(plan_payload)
    if reparsed != expected or reparsed.fingerprint != plan.fingerprint:
        raise SciPlex3CandidateRunnerError("sealed training plan cannot be reconstructed exactly")
    for artifact, filename in zip(sealed_plan.support_artifacts, _SUPPORT_FILENAMES, strict=True):
        payload = _read_local_artifact(artifact, name=f"sealed support {filename}")
        if payload != support_payloads[filename]:
            raise SciPlex3CandidateRunnerError(f"sealed support {filename} is stale")
    return plan, _candidate_design(preparation)


def _condition_manifest(condition: object) -> dict[str, object]:
    if type(condition) is NoAction:
        return {"kind": "no_action"}
    if type(condition) is CompoundDose:
        return {
            "compound": condition.compound,
            "dose_nm": condition.dose_nm,
            "kind": "compound_dose",
        }
    raise SciPlex3CandidateRunnerError("golden condition has an unsupported exact type")


def _golden_request_manifest(sample: CandidateRawCountSamples) -> dict[str, object]:
    target = sample.target
    return {
        "case_id": target.case_id,
        "condition": _condition_manifest(target.condition),
        "partition_id": target.partition_id,
        "plate_id": target.plate_id,
        "sample_count": int(sample.samples.shape[0]),
        "seed": sample.seed,
        "target_well_id": target.target_well_id,
    }


def _sample_identity(candidate: SciPlex3GammaPoissonCandidate) -> tuple[str, str]:
    if type(candidate) is not SciPlex3GammaPoissonCandidate:
        raise SciPlex3CandidateRunnerError("golden candidate has the wrong exact class")
    try:
        sample = candidate.golden_sample()
    except SciPlex3CandidateError as error:
        raise SciPlex3CandidateRunnerError("candidate golden sample failed") from error
    if (
        type(sample) is not CandidateRawCountSamples
        or sample.candidate_id != SCIPLEX3_CANDIDATE_MODEL_ID
        or sample.target.partition_id != _P1_PARTITION_ID
        or sample.rng_algorithm != RNG_ALGORITHM
        or sample.ordered_feature_keys != candidate.ordered_feature_keys
        or sample.samples.shape != (8, len(candidate.ordered_feature_keys))
    ):
        raise SciPlex3CandidateRunnerError("candidate golden sample contract drifted")
    request = CandidateSampleRequest(
        target=sample.target,
        sample_count=int(sample.samples.shape[0]),
        seed=sample.seed,
    )
    sampler = candidate._v5_runtime_sampler()
    certificate = sampler.envelope_certificate
    if (
        candidate.supports(sample.target)
        or not sampler.supports(candidate._v5_sampling_request(request))
        or sample.model_artifact_sha256 != candidate.model_artifact_sha256
        or sample.model_artifact_sha256 != sampler.parameters.model_artifact_sha256
        or sample.calibration_state_sha256 != sampler.parameters.active_calibration_state_sha256
        or sample.sampling_contract_sha256 != SCIPLEX3_V5_SAMPLING_CONTRACT_SHA256
        or sample.target_fingerprint != _candidate_module._v5_target_fingerprint(sample.target)
        or sample.context_id != SCIPLEX3_CANDIDATE_V5_NEUTRAL_CONTEXT_ID
        or certificate.sampling_contract_sha256 != SCIPLEX3_V5_SAMPLING_CONTRACT_SHA256
        or not certificate.supported
    ):
        raise SciPlex3CandidateRunnerError(
            "candidate golden request-level sampling provenance drifted"
        )
    values = np.asarray(sample.samples, dtype="<i8", order="C")
    request_manifest = _golden_request_manifest(sample)
    identity = _sha256(
        _canonical_json(
            {
                "request": request_manifest,
                "rng_algorithm": sample.rng_algorithm,
                "sample_bytes_sha256": _sha256(values.tobytes(order="C")),
                "sample_shape": list(values.shape),
                "sampling_provenance": {
                    "active_calibration_state_sha256": sample.calibration_state_sha256,
                    "context_id": sample.context_id,
                    "envelope_certificate_sha256": certificate.fingerprint,
                    "model_artifact_sha256": sample.model_artifact_sha256,
                    "sampling_contract_sha256": sample.sampling_contract_sha256,
                    "target_fingerprint": sample.target_fingerprint,
                },
            }
        )
    )
    return _sha256(_canonical_json(request_manifest)), identity


def _expected_candidate_topology(
    preparation: SciPlex3BaselinePreparation,
    design: SciPlex3P1DesignBindings,
) -> tuple[tuple[str, ...], np.ndarray, np.ndarray, np.ndarray]:
    wells = preparation.training_data.wells
    training_well_ids = tuple(well.well_id for well in wells)
    well_index = {well_id: index for index, well_id in enumerate(training_well_ids)}
    plate_index = {plate_id: index for index, plate_id in enumerate(design.plate_ids)}
    compound_index = {compound: index for index, compound in enumerate(design.compounds)}
    if (
        len(well_index) != SCIPLEX3_CANDIDATE_TRAINING_WELL_COUNT
        or len(plate_index) != SCIPLEX3_CANDIDATE_PLATE_COUNT
        or len(compound_index) != SCIPLEX3_CANDIDATE_COMPOUND_COUNT
    ):
        raise SciPlex3CandidateRunnerError("candidate p1 topology cardinality drifted")
    try:
        training_well_plate_indices = np.asarray(
            [plate_index[well.plate_id] for well in wells], dtype="<i8"
        )
        action_well_indices = np.empty(
            (SCIPLEX3_CANDIDATE_COMPOUND_COUNT, len(SCIPLEX3_CANDIDATE_DOSES_NM)),
            dtype="<i8",
        )
        for action in design.actions:
            action_well_indices[
                compound_index[action.compound],
                SCIPLEX3_CANDIDATE_DOSES_NM.index(action.dose_nm),
            ] = well_index[action.well_id]
        vehicle_well_indices = np.asarray(
            [[well_index[well_id] for well_id in vehicle.well_ids] for vehicle in design.vehicles],
            dtype="<i8",
        )
    except (KeyError, ValueError) as error:
        raise SciPlex3CandidateRunnerError(
            "candidate p1 topology differs from its exact design"
        ) from error
    return (
        training_well_ids,
        training_well_plate_indices,
        action_well_indices,
        vehicle_well_indices,
    )


def _validate_candidate_topology(
    candidate: SciPlex3GammaPoissonCandidate,
    preparation: SciPlex3BaselinePreparation,
    design: SciPlex3P1DesignBindings,
) -> tuple[float, ...]:
    (
        training_well_ids,
        training_well_plate_indices,
        action_well_indices,
        vehicle_well_indices,
    ) = _expected_candidate_topology(preparation, design)
    if (
        candidate.training_well_ids != training_well_ids
        or not np.array_equal(candidate._training_well_plate_indices, training_well_plate_indices)
        or not np.array_equal(candidate._action_well_indices, action_well_indices)
        or not np.array_equal(candidate._vehicle_well_indices, vehicle_well_indices)
    ):
        raise SciPlex3CandidateRunnerError(
            "candidate sealed topology differs from exact p1 preparation and design"
        )

    reconstructed = np.empty(
        (SCIPLEX3_CANDIDATE_TRAINING_WELL_COUNT, SCIPLEX3_CANDIDATE_FACTOR_COUNT),
        dtype=np.float64,
    )
    base = np.exp(candidate._alpha)[None, :] * candidate._rho
    for plate in range(SCIPLEX3_CANDIDATE_PLATE_COUNT):
        reconstructed[vehicle_well_indices[plate]] = base[plate]
    for compound in range(SCIPLEX3_CANDIDATE_COMPOUND_COUNT):
        for dose in range(len(SCIPLEX3_CANDIDATE_DOSES_NM)):
            well = int(action_well_indices[compound, dose])
            plate = int(training_well_plate_indices[well])
            reconstructed[well] = base[plate] * np.exp(candidate._delta[compound, dose])
    canonical = np.asarray(
        np.round(reconstructed, SCIPLEX3_CANDIDATE_FACTOR_ORDER_DECIMALS),
        dtype="<f8",
        order="C",
    )
    canonical = np.frombuffer(canonical.tobytes(order="C"), dtype="<f8").reshape(canonical.shape)
    if not np.array_equal(candidate._mean_activation, canonical):
        raise SciPlex3CandidateRunnerError(
            "candidate mean-activation witness differs from independently reconstructed p1 means"
        )
    contributions = tuple(
        math.fsum(
            float(canonical[row, factor]) for row in range(SCIPLEX3_CANDIDATE_TRAINING_WELL_COUNT)
        )
        / SCIPLEX3_CANDIDATE_TRAINING_WELL_COUNT
        for factor in range(SCIPLEX3_CANDIDATE_FACTOR_COUNT)
    )
    if not np.array_equal(
        candidate._factor_contributions,
        np.asarray(contributions, dtype=np.float64),
    ):
        raise SciPlex3CandidateRunnerError(
            "candidate factor contributions differ from independent equal-well sums"
        )
    return contributions


def _expected_inner_batch_count(preparation: SciPlex3BaselinePreparation) -> int:
    count = sum(
        (well.counts.row_count + SCIPLEX3_CANDIDATE_BATCH_SIZE - 1) // SCIPLEX3_CANDIDATE_BATCH_SIZE
        for well in preparation.training_data.wells
    )
    if count <= 0:
        raise SciPlex3CandidateRunnerError("p1 preparation has no candidate inner batches")
    return count


def _validate_inner_witness(
    witness: SciPlex3CandidateInitialEquilibration | SciPlex3CandidateTraceEntry,
    *,
    expected_batch_count: int,
    name: str,
) -> int:
    histogram = witness.inner_sweep_count_histogram
    if (
        type(histogram) is not tuple
        or len(histogram) != SCIPLEX3_CANDIDATE_MAX_INNER_SWEEPS
        or any(type(count) is not int or count < 0 for count in histogram)
        or histogram[0] != 0
        or sum(histogram) != expected_batch_count
        or type(witness.maximum_inner_sweeps) is not int
        or not SCIPLEX3_CANDIDATE_MIN_INNER_SWEEPS
        <= witness.maximum_inner_sweeps
        <= SCIPLEX3_CANDIDATE_MAX_INNER_SWEEPS
    ):
        raise SciPlex3CandidateRunnerError(f"{name} inner-sweep histogram is invalid")
    highest_occupied = max(index + 1 for index, count in enumerate(histogram) if count)
    if highest_occupied != witness.maximum_inner_sweeps:
        raise SciPlex3CandidateRunnerError(
            f"{name} maximum inner sweeps differs from its histogram"
        )
    for field_name in (
        "maximum_terminal_shape_residual",
        "maximum_terminal_elog_residual",
    ):
        value = getattr(witness, field_name)
        if (
            type(value) is not float
            or not math.isfinite(value)
            or not 0.0 <= value <= SCIPLEX3_CANDIDATE_INNER_RESIDUAL_TOL
        ):
            raise SciPlex3CandidateRunnerError(
                f"{name} {field_name} does not pass the fixed inner tolerance"
            )
    return sum((index + 1) * count for index, count in enumerate(histogram))


def _validate_factor_order(value: object, *, name: str) -> tuple[int, ...]:
    if (
        type(value) is not tuple
        or len(value) != SCIPLEX3_CANDIDATE_FACTOR_COUNT
        or any(type(index) is not int for index in value)
        or set(value) != set(range(SCIPLEX3_CANDIDATE_FACTOR_COUNT))
    ):
        raise SciPlex3CandidateRunnerError(f"{name} is not an exact factor permutation")
    return cast(tuple[int, ...], value)


def _inner_equilibration_manifest(
    candidate: SciPlex3GammaPoissonCandidate,
) -> list[dict[str, object]]:
    return [
        candidate.initial_equilibration.manifest(),
        *[
            {
                "inner_sweep_count_histogram": list(item.inner_sweep_count_histogram),
                "iteration": item.iteration,
                "maximum_inner_sweeps": item.maximum_inner_sweeps,
                "maximum_terminal_elog_residual": item.maximum_terminal_elog_residual,
                "maximum_terminal_shape_residual": item.maximum_terminal_shape_residual,
            }
            for item in candidate.trace
        ],
    ]


def _validate_candidate_state(
    candidate: SciPlex3GammaPoissonCandidate,
    preparation: SciPlex3BaselinePreparation,
    design: SciPlex3P1DesignBindings,
) -> tuple[dict[str, object], dict[str, object]]:
    if type(candidate) is not SciPlex3GammaPoissonCandidate:
        raise SciPlex3CandidateRunnerError("candidate factory returned a substituted class")
    contributions = _validate_candidate_topology(candidate, preparation, design)
    fixed_shape = np.asarray([SCIPLEX3_CANDIDATE_FIXED_FACTOR_SHAPE], dtype="<f8")
    if (
        type(candidate._factor_shape) is not np.ndarray
        or candidate._factor_shape.dtype.str != "<f8"
        or not candidate._factor_shape.flags.c_contiguous
        or candidate._factor_shape.flags.writeable
        or not np.array_equal(candidate._factor_shape, fixed_shape)
    ):
        raise SciPlex3CandidateRunnerError("candidate fixed factor-shape witness drifted")
    rho = candidate._rho
    if (
        type(rho) is not np.ndarray
        or rho.dtype.str != "<f8"
        or not rho.flags.c_contiguous
        or rho.flags.writeable
        or rho.shape != (SCIPLEX3_CANDIDATE_PLATE_COUNT, SCIPLEX3_CANDIDATE_FACTOR_COUNT)
        or not bool(np.all(np.isfinite(rho)))
        or bool(np.any(rho <= 0.0))
        or bool(np.any(rho >= SCIPLEX3_CANDIDATE_PLATE_COUNT))
    ):
        raise SciPlex3CandidateRunnerError("candidate training nuisance rho is invalid")
    rho_means = np.asarray(
        [
            math.fsum(float(rho[plate, factor]) for plate in range(SCIPLEX3_CANDIDATE_PLATE_COUNT))
            / SCIPLEX3_CANDIDATE_PLATE_COUNT
            for factor in range(SCIPLEX3_CANDIDATE_FACTOR_COUNT)
        ],
        dtype=np.float64,
    )
    if not bool(np.allclose(rho_means, 1.0, rtol=0.0, atol=5e-13)):
        raise SciPlex3CandidateRunnerError("candidate training nuisance rho is invalid")
    training_nuisance_rho_sha256 = _sha256(rho.tobytes(order="C"))

    sampling_parameters = candidate._v5_sampling_parameters_cache
    sampling_sampler = candidate._v5_runtime_sampler()
    sampling_certificate = sampling_sampler.envelope_certificate
    expected_sampling_parameters = candidate._v5_sampling_parameters(
        model_artifact_sha256=sampling_parameters.model_artifact_sha256
    )
    expected_combination_count = (
        (SCIPLEX3_CANDIDATE_ACTION_COUNT + 1)
        * len(expected_sampling_parameters.context_ids)
        * len(SCIPLEX3_CANDIDATE_TAU_GRID)
    )
    if (
        type(sampling_parameters) is not V5SamplingParameters
        or type(sampling_sampler) is not V5PositiveConditionedSampler
        or sampling_sampler is not candidate._v5_runtime_sampler_cache
        or sampling_sampler.parameters is not sampling_parameters
        or type(sampling_certificate) is not V5SamplingEnvelopeCertificate
        or sampling_certificate is not candidate._v5_sampling_envelope_certificate_cache
        or sampling_parameters.parameter_fingerprint
        != expected_sampling_parameters.parameter_fingerprint
        or sampling_parameters.active_calibration_state_sha256
        != expected_sampling_parameters.active_calibration_state_sha256
        or sampling_parameters.context_ids != (SCIPLEX3_CANDIDATE_V5_NEUTRAL_CONTEXT_ID,)
        or sampling_parameters.context_multipliers.shape
        != (len(SCIPLEX3_CANDIDATE_TAU_GRID), 1, SCIPLEX3_CANDIDATE_FACTOR_COUNT)
        or not np.array_equal(
            sampling_parameters.context_multipliers,
            np.ones(
                (len(SCIPLEX3_CANDIDATE_TAU_GRID), 1, SCIPLEX3_CANDIDATE_FACTOR_COUNT),
                dtype=np.float64,
            ),
        )
        or np.shares_memory(sampling_parameters.context_multipliers, rho)
        or sampling_parameters.active_tau != 1.0
        or len(sampling_parameters.action_ids) != SCIPLEX3_CANDIDATE_ACTION_COUNT + 1
        or sampling_certificate.parameter_fingerprint != sampling_parameters.parameter_fingerprint
        or sampling_certificate.sampling_contract_sha256 != SCIPLEX3_V5_SAMPLING_CONTRACT_SHA256
        or sampling_certificate.combination_count != expected_combination_count
        or sampling_certificate.maximum_request_count != SCIPLEX3_V5_MAX_SAMPLE_COUNT
        or sampling_certificate.request_failure_budget_log != SCIPLEX3_V5_REQUEST_FAILURE_BUDGET_LOG
        or sampling_certificate.worst_request_tail_log_upper_bound
        > SCIPLEX3_V5_REQUEST_FAILURE_BUDGET_LOG
        or not 0.0
        < sampling_certificate.maximum_compound_poisson_intensity
        <= SCIPLEX3_V5_MAX_COMPOUND_POISSON_INTENSITY
        or not sampling_certificate.supported
        or sampling_certificate.rejection_reasons
    ):
        raise SciPlex3CandidateRunnerError(
            "candidate neutral-context v5 sampling cache or certificate drifted"
        )

    initial = candidate.initial_equilibration
    if type(initial) is not SciPlex3CandidateInitialEquilibration:
        raise SciPlex3CandidateRunnerError("candidate initial equilibration has the wrong type")
    if type(initial.elbo) is not float or not math.isfinite(initial.elbo):
        raise SciPlex3CandidateRunnerError("candidate initial ELBO is invalid")
    _validate_factor_order(initial.factor_order, name="candidate initial factor order")
    trace = candidate.trace
    if (
        type(trace) is not tuple
        or not SCIPLEX3_CANDIDATE_MIN_OUTER_ITERATIONS
        <= len(trace)
        <= SCIPLEX3_CANDIDATE_MAX_OUTER_ITERATIONS
        or any(type(item) is not SciPlex3CandidateTraceEntry for item in trace)
        or tuple(item.iteration for item in trace) != tuple(range(1, len(trace) + 1))
    ):
        raise SciPlex3CandidateRunnerError("candidate synchronized trace is noncanonical")
    expected_batch_count = _expected_inner_batch_count(preparation)
    _validate_inner_witness(
        initial,
        expected_batch_count=expected_batch_count,
        name="candidate initialization",
    )
    for item in trace:
        if (
            type(item.elbo) is not float
            or not math.isfinite(item.elbo)
            or type(item.relative_change) is not float
            or not math.isfinite(item.relative_change)
            or item.relative_change < 0.0
        ):
            raise SciPlex3CandidateRunnerError("candidate synchronized trace scalar drifted")
        _validate_factor_order(
            item.factor_order,
            name=f"candidate trace iteration {item.iteration} factor order",
        )
        _validate_inner_witness(
            item,
            expected_batch_count=expected_batch_count,
            name=f"candidate trace iteration {item.iteration}",
        )
    previous_elbo = initial.elbo
    for item in trace:
        expected_relative = abs(item.elbo - previous_elbo) / max(1.0, abs(previous_elbo))
        if (
            item.relative_change != expected_relative
            or item.elbo - previous_elbo
            < -_candidate_module.SCIPLEX3_CANDIDATE_ELBO_DECREASE_RTOL
            * max(1.0, abs(previous_elbo))
        ):
            raise SciPlex3CandidateRunnerError("candidate synchronized ELBO trace drifted")
        previous_elbo = item.elbo
    terminal_trace = trace[-SCIPLEX3_CANDIDATE_CONVERGENCE_STREAK:]
    if (
        any(item.relative_change > SCIPLEX3_CANDIDATE_CONVERGENCE_RTOL for item in terminal_trace)
        or len({item.factor_order for item in terminal_trace}) != 1
    ):
        raise SciPlex3CandidateRunnerError("candidate terminal ELBO or order gate failed")

    behavior = candidate.behavior_manifest()
    expected_behavior_keys = {
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
    if set(behavior) != expected_behavior_keys:
        raise SciPlex3CandidateRunnerError("candidate behavior manifest schema drifted")
    iterations = behavior["outer_iteration_count"]
    final_elbo = behavior["final_elbo"]
    loading_rank_ratio = behavior["loading_rank_ratio"]
    mean_activation_rank_ratio = behavior["mean_activation_rank_ratio"]
    minimum_share = behavior["minimum_factor_contribution_share"]
    shares = behavior["factor_contribution_shares"]
    terminal_elbo = behavior["terminal_elbo_relative_changes"]
    contribution_total = math.fsum(contributions)
    expected_shares = [value / contribution_total for value in contributions]
    expected_terminal_elbo = [item.relative_change for item in terminal_trace]
    try:
        expected_loading_rank_ratio = _candidate_module._rounded_matrix_condition_ratio(
            candidate._basis,
            name="runner-recomputed loading matrix",
        )
        expected_mean_activation_rank_ratio = _candidate_module._rounded_matrix_condition_ratio(
            candidate._mean_activation,
            name="runner-recomputed mean activation matrix",
        )
    except SciPlex3CandidateError as error:
        raise SciPlex3CandidateRunnerError(
            "candidate independently recomputed rank gate failed"
        ) from error
    initial_sha256 = _sha256(_canonical_json(initial.manifest()))
    inner_trace_sha256 = _sha256(_canonical_json(_inner_equilibration_manifest(candidate)))
    maximum_inner_sweeps = max(item.maximum_inner_sweeps for item in trace)
    maximum_shape_residual = max(item.maximum_terminal_shape_residual for item in trace)
    maximum_elog_residual = max(item.maximum_terminal_elog_residual for item in trace)
    if (
        behavior["fit_converged"] is not True
        or behavior["all_parameters_finite"] is not True
        or behavior["capture_latent_present"] is not False
        or behavior["factor_shape_estimated"] is not False
        or behavior["factor_shape_mode"] != "fixed"
        or behavior["fixed_factor_shape"] != SCIPLEX3_CANDIDATE_FIXED_FACTOR_SHAPE
        or behavior["factor_order_stable"] is not True
        or behavior["training_partition_ids"] != [_P1_PARTITION_ID]
        or behavior["heldout_memberships_read"] is not False
        or behavior["heldout_outcomes_read"] is not False
        or behavior["can_mint_lifecycle_evidence"] is not False
        or behavior["scientifically_admissible"] is not False
        or behavior["inner_all_batches_converged"] is not True
        or behavior["inner_equilibration_performed"] is not True
        or behavior["inner_batch_count"] != expected_batch_count
        or behavior["model_schema_version"] != SCIPLEX3_CANDIDATE_MODEL_SCHEMA_VERSION
        or behavior["plate_context_count"] != 1
        or behavior["plate_context_factorwise_mean_one"] is not True
        or behavior["plate_context_family"] != "neutral-unit-context"
        or behavior["sampling_active_calibration_state_sha256"]
        != sampling_parameters.active_calibration_state_sha256
        or behavior["sampling_contract_sha256"] != SCIPLEX3_V5_SAMPLING_CONTRACT_SHA256
        or behavior["sampling_envelope_combination_count"] != sampling_certificate.combination_count
        or behavior["sampling_envelope_maximum_compound_poisson_intensity"]
        != sampling_certificate.maximum_compound_poisson_intensity
        or behavior["sampling_envelope_maximum_request_count"]
        != sampling_certificate.maximum_request_count
        or behavior["sampling_envelope_rejection_reasons"]
        != list(sampling_certificate.rejection_reasons)
        or behavior["sampling_envelope_request_failure_budget_log"]
        != sampling_certificate.request_failure_budget_log
        or behavior["sampling_envelope_supported"] is not True
        or behavior["sampling_envelope_worst_request_tail_log_upper_bound"]
        != sampling_certificate.worst_request_tail_log_upper_bound
        or type(iterations) is not int
        or not SCIPLEX3_CANDIDATE_MIN_OUTER_ITERATIONS
        <= iterations
        <= SCIPLEX3_CANDIDATE_MAX_OUTER_ITERATIONS
        or iterations != len(trace)
        or type(final_elbo) is not float
        or not math.isfinite(final_elbo)
        or final_elbo != trace[-1].elbo
        or behavior["initial_elbo"] != initial.elbo
        or behavior["initial_equilibration_sha256"] != initial_sha256
        or behavior["initial_factor_order"] != list(initial.factor_order)
        or behavior["initial_inner_sweep_count_histogram"]
        != list(initial.inner_sweep_count_histogram)
        or behavior["initial_maximum_inner_sweeps"] != initial.maximum_inner_sweeps
        or behavior["initial_maximum_terminal_elog_residual"]
        != initial.maximum_terminal_elog_residual
        or behavior["initial_maximum_terminal_shape_residual"]
        != initial.maximum_terminal_shape_residual
        or behavior["maximum_inner_sweeps"] != maximum_inner_sweeps
        or behavior["maximum_terminal_elog_residual"] != maximum_elog_residual
        or behavior["maximum_terminal_shape_residual"] != maximum_shape_residual
        or type(loading_rank_ratio) is not float
        or type(mean_activation_rank_ratio) is not float
        or type(minimum_share) is not float
        or loading_rank_ratio != expected_loading_rank_ratio
        or mean_activation_rank_ratio != expected_mean_activation_rank_ratio
        or loading_rank_ratio != round(loading_rank_ratio, SCIPLEX3_CANDIDATE_FACTOR_ORDER_DECIMALS)
        or mean_activation_rank_ratio
        != round(mean_activation_rank_ratio, SCIPLEX3_CANDIDATE_FACTOR_ORDER_DECIMALS)
        or not all(
            math.isfinite(value)
            and value
            > SCIPLEX3_CANDIDATE_IDENTIFIABILITY_THRESHOLD
            + SCIPLEX3_CANDIDATE_IDENTIFIABILITY_MARGIN
            for value in (loading_rank_ratio, mean_activation_rank_ratio)
        )
        or type(shares) is not list
        or len(shares) != SCIPLEX3_CANDIDATE_FACTOR_COUNT
        or any(type(value) is not float for value in shares)
        or not np.array_equal(
            np.asarray(shares, dtype=np.float64),
            np.asarray(expected_shares, dtype=np.float64),
        )
        or minimum_share != min(expected_shares)
        or type(terminal_elbo) is not list
        or len(terminal_elbo) != SCIPLEX3_CANDIDATE_CONVERGENCE_STREAK
        or any(
            type(value) is not float
            or not math.isfinite(value)
            or value > SCIPLEX3_CANDIDATE_CONVERGENCE_RTOL
            for value in terminal_elbo
        )
        or terminal_elbo != expected_terminal_elbo
    ):
        raise SciPlex3CandidateRunnerError(
            "candidate convergence, finiteness, or scope gate failed"
        )
    summary = candidate.training_summary
    if (
        summary.record_count != SCIPLEX3_CANDIDATE_TRAINING_RECORD_COUNT
        or summary.record_count != preparation.receipt.record_count
        or summary.well_count != SCIPLEX3_CANDIDATE_TRAINING_WELL_COUNT
        or summary.well_count != preparation.receipt.well_count
        or summary.zero_panel_record_count != preparation.receipt.zero_panel_record_count
        or summary.design_sha256 != design.fingerprint
        or summary.training_data_sha256 != training_data_fingerprint(preparation.training_data)
        or summary.provenance != "real-p1"
        or candidate.ordered_feature_keys != preparation.training_data.ordered_feature_keys
        or candidate.compounds != design.compounds
        or candidate.plate_ids != design.plate_ids
    ):
        raise SciPlex3CandidateRunnerError("candidate fitted state is bound to another p1 input")
    fitted_state = candidate.fitted_state_manifest()
    # Canonical round trips reject mutable containers and nonfinite lookalikes before hashing.
    behavior_copy = _json_object(_canonical_json(behavior), name="candidate behavior")
    fitted_state_copy = _json_object(_canonical_json(fitted_state), name="candidate fitted state")
    tensor_sha256 = fitted_state_copy.get("tensor_sha256")
    if (
        set(fitted_state_copy)
        != {
            "behavior",
            "candidate_specification_sha256",
            "compounds_sha256",
            "implementation_version",
            "initial_equilibration_sha256",
            "inner_equilibration_trace_sha256",
            "model_id",
            "model_schema_version",
            "ordered_feature_keys_sha256",
            "plate_ids_sha256",
            "tensor_sha256",
            "training",
            "training_well_ids_sha256",
        }
        or type(tensor_sha256) is not dict
        or set(cast(dict[str, object], tensor_sha256))
        != {
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
        or fitted_state_copy.get("implementation_version")
        != SCIPLEX3_CANDIDATE_IMPLEMENTATION_VERSION
        or fitted_state_copy.get("model_id") != SCIPLEX3_CANDIDATE_MODEL_ID
        or fitted_state_copy.get("model_schema_version") != SCIPLEX3_CANDIDATE_MODEL_SCHEMA_VERSION
        or fitted_state_copy.get("candidate_specification_sha256")
        != SCIPLEX3_CANDIDATE_SPECIFICATION_SHA256
        or fitted_state_copy.get("initial_equilibration_sha256") != initial_sha256
        or fitted_state_copy.get("inner_equilibration_trace_sha256") != inner_trace_sha256
        or cast(dict[str, object], tensor_sha256).get("rho") != training_nuisance_rho_sha256
        or cast(dict[str, object], tensor_sha256).get("factor_shape")
        != _sha256(fixed_shape.tobytes(order="C"))
    ):
        raise SciPlex3CandidateRunnerError("candidate fitted state specification drifted")
    if sampling_parameters.model_artifact_sha256 != candidate.model_artifact_sha256:
        raise SciPlex3CandidateRunnerError(
            "candidate v5 sampling provenance is bound to another model artifact"
        )
    return behavior_copy, fitted_state_copy


def _fit_exact_candidate(
    preparation: SciPlex3BaselinePreparation,
    design: SciPlex3P1DesignBindings,
) -> SciPlex3GammaPoissonCandidate:
    try:
        candidate = SciPlex3GammaPoissonCandidate.fit(preparation.training_data, design)
    except (SciPlex3CandidateError, ArithmeticError, FloatingPointError) as error:
        raise SciPlex3CandidateRunnerError("exact candidate fitting failed closed") from error
    if type(candidate) is not SciPlex3GammaPoissonCandidate:
        raise SciPlex3CandidateRunnerError("exact candidate factory returned another class")
    return candidate


def _load_exact_candidate(payload: bytes, *, expected_sha256: str) -> SciPlex3GammaPoissonCandidate:
    try:
        candidate = _candidate_module.load_sciplex3_candidate(
            payload, expected_sha256=expected_sha256
        )
    except (SciPlex3CandidateError, ArithmeticError, FloatingPointError) as error:
        raise SciPlex3CandidateRunnerError("sealed candidate model failed exact reload") from error
    if type(candidate) is not SciPlex3GammaPoissonCandidate:
        raise SciPlex3CandidateRunnerError("candidate reload substituted another class")
    return candidate


@dataclass(frozen=True, slots=True)
class SciPlex3CandidateTrainingObservation:
    """Canonical p1 execution facts; deliberately not a trusted workflow receipt."""

    plan_fingerprint: str
    preparation_fingerprint: str
    finalized_count_scan_fingerprint: str
    assembly_fingerprint: str
    p1_count_stream_sha256: str
    p1_design_fingerprint: str
    ordered_feature_keys_sha256: str
    action_binding_sha256: str
    candidate_specification_sha256: str
    output_model_schema_sha256: str
    runtime_lock_sha256: str
    training_code_closure_sha256: str
    training_execution_input_closure_sha256: str
    loader_code_sha256: str
    item11_runner_code_sha256: str
    candidate_runner_code_sha256: str
    candidate_factory_code_sha256: str
    candidate_objective_code_sha256: str
    candidate_sampling_code_sha256: str
    model_artifact_sha256: str
    model_artifact_byte_count: int
    fitted_state_sha256: str
    behavior_sha256: str
    training_nuisance_rho_sha256: str
    sampling_contract_sha256: str
    sampling_active_calibration_state_sha256: str
    sampling_parameter_fingerprint: str
    sampling_envelope_certificate_sha256: str
    initial_equilibration_sha256: str
    inner_equilibration_trace_sha256: str
    golden_request_sha256: str
    golden_sample_sha256: str
    software_golden_model_sha256: str
    software_golden_sample_sha256: str
    outer_iteration_count: int
    initial_elbo: float
    initial_factor_order: tuple[int, ...]
    initial_inner_sweep_count_histogram: tuple[int, ...]
    initial_maximum_inner_sweeps: int
    initial_maximum_terminal_shape_residual: float
    initial_maximum_terminal_elog_residual: float
    final_elbo: float
    fixed_factor_shape: float
    inner_batch_count: int
    total_inner_sweep_count: int
    maximum_inner_sweeps: int
    maximum_terminal_shape_residual: float
    maximum_terminal_elog_residual: float
    loading_rank_ratio: float
    mean_activation_rank_ratio: float
    minimum_factor_contribution_share: float
    terminal_elbo_relative_changes: tuple[float, ...]
    sampling_envelope_combination_count: int
    sampling_envelope_maximum_request_count: int
    sampling_envelope_request_failure_budget_log: float
    sampling_envelope_worst_request_tail_log_upper_bound: float
    sampling_envelope_maximum_compound_poisson_intensity: float
    sampling_envelope_rejection_reasons: tuple[str, ...]
    sampling_envelope_worst_action_id: str
    sampling_envelope_worst_context_id: str
    sampling_envelope_worst_tau_hex: str
    artifact_schema: Literal["sciplex3-candidate-training-execution-observation"] = (
        "sciplex3-candidate-training-execution-observation"
    )
    artifact_schema_version: Literal["5.0.0"] = "5.0.0"
    candidate_model_schema_version: Literal["5.0.0"] = "5.0.0"
    fit_converged: Literal[True] = True
    all_parameters_finite: Literal[True] = True
    factor_order_stable: Literal[True] = True
    factor_shape_mode: Literal["fixed"] = "fixed"
    factor_shape_estimated: Literal[False] = False
    inner_equilibration_performed: Literal[True] = True
    inner_all_batches_converged: Literal[True] = True
    plate_context_family: Literal["neutral-unit-context"] = "neutral-unit-context"
    plate_context_id: Literal["neutral-unit-unseen-plate-context"] = (
        "neutral-unit-unseen-plate-context"
    )
    plate_context_count: Literal[1] = 1
    plate_context_factorwise_mean_one: Literal[True] = True
    sampling_conditioning: Literal["exact-positive-panel-via-zero-truncated-compound-poisson"] = (
        "exact-positive-panel-via-zero-truncated-compound-poisson"
    )
    sampling_request_support: Literal["exact-CandidateSampleRequest-not-target-only"] = (
        "exact-CandidateSampleRequest-not-target-only"
    )
    sampling_envelope_supported: Literal[True] = True
    model_reloaded: Literal[True] = True
    exact_class_reloaded: Literal[True] = True
    golden_reproduced: Literal[True] = True
    training_partition_ids: tuple[Literal["p1-train"], ...] = ("p1-train",)
    heldout_artifacts_resolved: Literal[False] = False
    heldout_memberships_read: Literal[False] = False
    heldout_outcomes_read: Literal[False] = False
    calibration_performed: Literal[False] = False
    model_selection_performed: Literal[False] = False
    metrics_computed: Literal[False] = False
    capture_latent_present: Literal[False] = False
    plate_sigma_present: Literal[False] = False
    can_mint_lifecycle_evidence: Literal[False] = False
    scientifically_admissible: Literal[False] = False

    def __post_init__(self) -> None:
        if (
            type(self.artifact_schema) is not str
            or self.artifact_schema != "sciplex3-candidate-training-execution-observation"
            or type(self.artifact_schema_version) is not str
            or self.artifact_schema_version != "5.0.0"
            or type(self.candidate_model_schema_version) is not str
            or self.candidate_model_schema_version != SCIPLEX3_CANDIDATE_MODEL_SCHEMA_VERSION
        ):
            raise SciPlex3CandidateRunnerError("observation schema identity drifted")
        for name in (
            "plan_fingerprint",
            "preparation_fingerprint",
            "finalized_count_scan_fingerprint",
            "assembly_fingerprint",
            "p1_count_stream_sha256",
            "p1_design_fingerprint",
            "ordered_feature_keys_sha256",
            "action_binding_sha256",
            "candidate_specification_sha256",
            "output_model_schema_sha256",
            "runtime_lock_sha256",
            "training_code_closure_sha256",
            "training_execution_input_closure_sha256",
            "loader_code_sha256",
            "item11_runner_code_sha256",
            "candidate_runner_code_sha256",
            "candidate_factory_code_sha256",
            "candidate_objective_code_sha256",
            "candidate_sampling_code_sha256",
            "model_artifact_sha256",
            "fitted_state_sha256",
            "behavior_sha256",
            "training_nuisance_rho_sha256",
            "sampling_contract_sha256",
            "sampling_active_calibration_state_sha256",
            "sampling_parameter_fingerprint",
            "sampling_envelope_certificate_sha256",
            "initial_equilibration_sha256",
            "inner_equilibration_trace_sha256",
            "golden_request_sha256",
            "golden_sample_sha256",
            "software_golden_model_sha256",
            "software_golden_sample_sha256",
        ):
            _exact_sha256(getattr(self, name), name=name)
        if (
            self.candidate_specification_sha256 != SCIPLEX3_CANDIDATE_SPECIFICATION_SHA256
            or self.output_model_schema_sha256 != SCIPLEX3_CANDIDATE_OUTPUT_MODEL_SCHEMA_SHA256
            or self.loader_code_sha256 != _item11.SCIPLEX3_LOADER_CODE_SHA256
            or self.item11_runner_code_sha256 != _item11._IMPORTED_RUNNER_CODE_SHA256
            or self.candidate_runner_code_sha256 != _IMPORTED_RUNNER_CODE_SHA256
            or self.candidate_factory_code_sha256 != _IMPORTED_CANDIDATE_CODE_SHA256
            or self.candidate_objective_code_sha256 != _IMPORTED_CANDIDATE_V5_CODE_SHA256
            or self.candidate_sampling_code_sha256 != _IMPORTED_SAMPLING_V5_CODE_SHA256
            or self.software_golden_model_sha256 != SCIPLEX3_CANDIDATE_GOLDEN_MODEL_SHA256
            or self.software_golden_sample_sha256 != SCIPLEX3_CANDIDATE_GOLDEN_SAMPLE_SHA256
            or self.sampling_contract_sha256 != SCIPLEX3_V5_SAMPLING_CONTRACT_SHA256
            or self.sampling_active_calibration_state_sha256
            != _expected_v5_active_calibration_state_sha256()
        ):
            raise SciPlex3CandidateRunnerError("observation executable binding drifted")
        if type(self.model_artifact_byte_count) is not int or self.model_artifact_byte_count <= 0:
            raise SciPlex3CandidateRunnerError("model artifact byte count must be positive")
        if (
            type(self.outer_iteration_count) is not int
            or not SCIPLEX3_CANDIDATE_MIN_OUTER_ITERATIONS
            <= self.outer_iteration_count
            <= SCIPLEX3_CANDIDATE_MAX_OUTER_ITERATIONS
            or type(self.initial_elbo) is not float
            or not math.isfinite(self.initial_elbo)
            or type(self.final_elbo) is not float
            or not math.isfinite(self.final_elbo)
            or type(self.fixed_factor_shape) is not float
            or self.fixed_factor_shape != SCIPLEX3_CANDIDATE_FIXED_FACTOR_SHAPE
            or type(self.loading_rank_ratio) is not float
            or type(self.mean_activation_rank_ratio) is not float
            or type(self.minimum_factor_contribution_share) is not float
            or any(
                not math.isfinite(value)
                for value in (
                    self.loading_rank_ratio,
                    self.mean_activation_rank_ratio,
                    self.minimum_factor_contribution_share,
                )
            )
            or self.loading_rank_ratio
            <= SCIPLEX3_CANDIDATE_IDENTIFIABILITY_THRESHOLD
            + SCIPLEX3_CANDIDATE_IDENTIFIABILITY_MARGIN
            or self.mean_activation_rank_ratio
            <= SCIPLEX3_CANDIDATE_IDENTIFIABILITY_THRESHOLD
            + SCIPLEX3_CANDIDATE_IDENTIFIABILITY_MARGIN
            or self.minimum_factor_contribution_share
            <= SCIPLEX3_CANDIDATE_IDENTIFIABILITY_THRESHOLD
            or self.loading_rank_ratio
            != round(
                self.loading_rank_ratio,
                SCIPLEX3_CANDIDATE_FACTOR_ORDER_DECIMALS,
            )
            or self.mean_activation_rank_ratio
            != round(
                self.mean_activation_rank_ratio,
                SCIPLEX3_CANDIDATE_FACTOR_ORDER_DECIMALS,
            )
        ):
            raise SciPlex3CandidateRunnerError("observation convergence scalars are invalid")
        if (
            type(self.initial_factor_order) is not tuple
            or len(self.initial_factor_order) != SCIPLEX3_CANDIDATE_FACTOR_COUNT
            or any(type(index) is not int for index in self.initial_factor_order)
            or set(self.initial_factor_order) != set(range(SCIPLEX3_CANDIDATE_FACTOR_COUNT))
        ):
            raise SciPlex3CandidateRunnerError(
                "observation initial factor order must be an exact permutation"
            )
        initial_histogram = self.initial_inner_sweep_count_histogram
        if (
            type(initial_histogram) is not tuple
            or len(initial_histogram) != SCIPLEX3_CANDIDATE_MAX_INNER_SWEEPS
            or any(type(count) is not int or count < 0 for count in initial_histogram)
            or initial_histogram[0] != 0
            or type(self.inner_batch_count) is not int
            or self.inner_batch_count <= 0
            or sum(initial_histogram) != self.inner_batch_count
            or type(self.initial_maximum_inner_sweeps) is not int
            or not SCIPLEX3_CANDIDATE_MIN_INNER_SWEEPS
            <= self.initial_maximum_inner_sweeps
            <= SCIPLEX3_CANDIDATE_MAX_INNER_SWEEPS
            or max(index + 1 for index, count in enumerate(initial_histogram) if count)
            != self.initial_maximum_inner_sweeps
            or type(self.total_inner_sweep_count) is not int
            or self.total_inner_sweep_count
            < self.inner_batch_count
            * SCIPLEX3_CANDIDATE_MIN_INNER_SWEEPS
            * (self.outer_iteration_count + 1)
            or self.total_inner_sweep_count
            > self.inner_batch_count
            * SCIPLEX3_CANDIDATE_MAX_INNER_SWEEPS
            * (self.outer_iteration_count + 1)
            or type(self.maximum_inner_sweeps) is not int
            or not SCIPLEX3_CANDIDATE_MIN_INNER_SWEEPS
            <= self.maximum_inner_sweeps
            <= SCIPLEX3_CANDIDATE_MAX_INNER_SWEEPS
        ):
            raise SciPlex3CandidateRunnerError(
                "observation inner-equilibration count witnesses are invalid"
            )
        for name in (
            "initial_maximum_terminal_shape_residual",
            "initial_maximum_terminal_elog_residual",
            "maximum_terminal_shape_residual",
            "maximum_terminal_elog_residual",
        ):
            value = getattr(self, name)
            if (
                type(value) is not float
                or not math.isfinite(value)
                or not 0.0 <= value <= SCIPLEX3_CANDIDATE_INNER_RESIDUAL_TOL
            ):
                raise SciPlex3CandidateRunnerError(
                    f"observation {name} does not pass the inner residual tolerance"
                )
        terminal_values = self.terminal_elbo_relative_changes
        if (
            type(terminal_values) is not tuple
            or len(terminal_values) != SCIPLEX3_CANDIDATE_CONVERGENCE_STREAK
            or any(
                type(value) is not float
                or not math.isfinite(value)
                or not 0.0 <= value <= SCIPLEX3_CANDIDATE_CONVERGENCE_RTOL
                for value in terminal_values
            )
        ):
            raise SciPlex3CandidateRunnerError(
                "observation terminal ELBO changes are not an exact convergence streak"
            )
        for name in (
            "fit_converged",
            "all_parameters_finite",
            "factor_order_stable",
            "inner_equilibration_performed",
            "inner_all_batches_converged",
            "plate_context_factorwise_mean_one",
            "sampling_envelope_supported",
            "model_reloaded",
            "exact_class_reloaded",
            "golden_reproduced",
        ):
            if getattr(self, name) is not True:
                raise SciPlex3CandidateRunnerError(f"observation {name} must be exactly true")
        if (
            type(self.factor_shape_mode) is not str
            or self.factor_shape_mode != "fixed"
            or self.factor_shape_estimated is not False
            or type(self.plate_context_family) is not str
            or self.plate_context_family != "neutral-unit-context"
            or type(self.plate_context_id) is not str
            or self.plate_context_id != SCIPLEX3_CANDIDATE_V5_NEUTRAL_CONTEXT_ID
            or type(self.plate_context_count) is not int
            or self.plate_context_count != 1
            or type(self.sampling_conditioning) is not str
            or self.sampling_conditioning
            != "exact-positive-panel-via-zero-truncated-compound-poisson"
            or type(self.sampling_request_support) is not str
            or self.sampling_request_support != "exact-CandidateSampleRequest-not-target-only"
        ):
            raise SciPlex3CandidateRunnerError(
                "observation fixed-shape or neutral-context sampling family drifted"
            )
        if (
            type(self.sampling_envelope_combination_count) is not int
            or self.sampling_envelope_combination_count
            != (SCIPLEX3_CANDIDATE_ACTION_COUNT + 1) * len(SCIPLEX3_CANDIDATE_TAU_GRID)
            or type(self.sampling_envelope_maximum_request_count) is not int
            or self.sampling_envelope_maximum_request_count != SCIPLEX3_V5_MAX_SAMPLE_COUNT
            or type(self.sampling_envelope_request_failure_budget_log) is not float
            or self.sampling_envelope_request_failure_budget_log
            != SCIPLEX3_V5_REQUEST_FAILURE_BUDGET_LOG
            or type(self.sampling_envelope_worst_request_tail_log_upper_bound) is not float
            or not math.isfinite(self.sampling_envelope_worst_request_tail_log_upper_bound)
            or self.sampling_envelope_worst_request_tail_log_upper_bound
            > SCIPLEX3_V5_REQUEST_FAILURE_BUDGET_LOG
            or type(self.sampling_envelope_maximum_compound_poisson_intensity) is not float
            or not math.isfinite(self.sampling_envelope_maximum_compound_poisson_intensity)
            or not 0.0
            < self.sampling_envelope_maximum_compound_poisson_intensity
            <= SCIPLEX3_V5_MAX_COMPOUND_POISSON_INTENSITY
            or type(self.sampling_envelope_rejection_reasons) is not tuple
            or self.sampling_envelope_rejection_reasons
        ):
            raise SciPlex3CandidateRunnerError(
                "observation v5 sampling envelope certificate drifted"
            )
        if (
            type(self.sampling_envelope_worst_action_id) is not str
            or not self.sampling_envelope_worst_action_id
            or self.sampling_envelope_worst_action_id
            != self.sampling_envelope_worst_action_id.strip()
            or self.sampling_envelope_worst_context_id != SCIPLEX3_CANDIDATE_V5_NEUTRAL_CONTEXT_ID
            or self.sampling_envelope_worst_tau_hex
            not in {tau.hex() for tau in SCIPLEX3_CANDIDATE_TAU_GRID}
        ):
            raise SciPlex3CandidateRunnerError(
                "observation v5 sampling envelope witness identity drifted"
            )
        declared_certificate = V5SamplingEnvelopeCertificate(
            parameter_fingerprint=self.sampling_parameter_fingerprint,
            combination_count=self.sampling_envelope_combination_count,
            supported=self.sampling_envelope_supported,
            rejection_reasons=self.sampling_envelope_rejection_reasons,
            maximum_request_count=self.sampling_envelope_maximum_request_count,
            request_failure_budget_log=self.sampling_envelope_request_failure_budget_log,
            worst_request_tail_log_upper_bound=(
                self.sampling_envelope_worst_request_tail_log_upper_bound
            ),
            worst_action_id=self.sampling_envelope_worst_action_id,
            worst_context_id=self.sampling_envelope_worst_context_id,
            worst_tau_hex=self.sampling_envelope_worst_tau_hex,
            maximum_compound_poisson_intensity=(
                self.sampling_envelope_maximum_compound_poisson_intensity
            ),
        )
        if declared_certificate.fingerprint != self.sampling_envelope_certificate_sha256:
            raise SciPlex3CandidateRunnerError(
                "observation v5 sampling envelope certificate digest drifted"
            )
        if type(self.training_partition_ids) is not tuple or self.training_partition_ids != (
            _P1_PARTITION_ID,
        ):
            raise SciPlex3CandidateRunnerError("observation training scope must be exact p1")
        for name in (
            "heldout_artifacts_resolved",
            "heldout_memberships_read",
            "heldout_outcomes_read",
            "calibration_performed",
            "model_selection_performed",
            "metrics_computed",
            "capture_latent_present",
            "plate_sigma_present",
            "can_mint_lifecycle_evidence",
            "scientifically_admissible",
        ):
            if getattr(self, name) is not False:
                raise SciPlex3CandidateRunnerError(f"observation {name} must be exactly false")

    def manifest(self) -> dict[str, object]:
        return {
            name: list(value) if isinstance(value, tuple) else value
            for name in self.__dataclass_fields__
            if (value := getattr(self, name)) is not None
        }

    @property
    def fingerprint(self) -> str:
        return _sha256(_canonical_json(self.manifest()))


def _make_observation(
    preparation: SciPlex3BaselinePreparation,
    plan: CandidateTrainingPlan,
    candidate: SciPlex3GammaPoissonCandidate,
    model_artifact: LocalContentAddressedArtifact,
    *,
    golden_sample_sha256: str,
) -> SciPlex3CandidateTrainingObservation:
    behavior, fitted_state = _validate_candidate_state(
        candidate, preparation, _candidate_design(preparation)
    )
    golden_request_sha256, _ = _sample_identity(candidate)
    initial = candidate.initial_equilibration
    trace = candidate.trace
    tensor_sha256 = cast(dict[str, object], fitted_state["tensor_sha256"])
    sampling_sampler = candidate._v5_runtime_sampler()
    sampling_parameters = sampling_sampler.parameters
    sampling_certificate = sampling_sampler.envelope_certificate
    total_inner_sweep_count = sum(
        (index + 1) * count for index, count in enumerate(initial.inner_sweep_count_histogram)
    ) + sum(
        (index + 1) * count
        for item in trace
        for index, count in enumerate(item.inner_sweep_count_histogram)
    )
    return SciPlex3CandidateTrainingObservation(
        plan_fingerprint=plan.fingerprint,
        preparation_fingerprint=preparation.receipt.fingerprint,
        finalized_count_scan_fingerprint=(preparation.finalized_count_scan_receipt.fingerprint),
        assembly_fingerprint=preparation.receipt.fingerprint,
        p1_count_stream_sha256=plan.p1_count_stream_sha256,
        p1_design_fingerprint=plan.p1_design_fingerprint,
        ordered_feature_keys_sha256=plan.ordered_feature_keys_sha256,
        action_binding_sha256=plan.action_binding_sha256,
        candidate_specification_sha256=plan.candidate_specification.sha256,
        output_model_schema_sha256=plan.output_model_schema.sha256,
        runtime_lock_sha256=plan.runtime_lock.sha256,
        training_code_closure_sha256=plan.training_code_closure.sha256,
        training_execution_input_closure_sha256=(plan.training_execution_input_closure.sha256),
        loader_code_sha256=_item11.SCIPLEX3_LOADER_CODE_SHA256,
        item11_runner_code_sha256=_item11._IMPORTED_RUNNER_CODE_SHA256,
        candidate_runner_code_sha256=_IMPORTED_RUNNER_CODE_SHA256,
        candidate_factory_code_sha256=_IMPORTED_CANDIDATE_CODE_SHA256,
        candidate_objective_code_sha256=_IMPORTED_CANDIDATE_V5_CODE_SHA256,
        candidate_sampling_code_sha256=_IMPORTED_SAMPLING_V5_CODE_SHA256,
        model_artifact_sha256=model_artifact.sha256,
        model_artifact_byte_count=model_artifact.byte_count,
        fitted_state_sha256=_sha256(_canonical_json(fitted_state)),
        behavior_sha256=_sha256(_canonical_json(behavior)),
        training_nuisance_rho_sha256=cast(str, tensor_sha256["rho"]),
        sampling_contract_sha256=SCIPLEX3_V5_SAMPLING_CONTRACT_SHA256,
        sampling_active_calibration_state_sha256=(
            sampling_parameters.active_calibration_state_sha256
        ),
        sampling_parameter_fingerprint=sampling_parameters.parameter_fingerprint,
        sampling_envelope_certificate_sha256=sampling_certificate.fingerprint,
        initial_equilibration_sha256=cast(str, fitted_state["initial_equilibration_sha256"]),
        inner_equilibration_trace_sha256=cast(
            str, fitted_state["inner_equilibration_trace_sha256"]
        ),
        golden_request_sha256=golden_request_sha256,
        golden_sample_sha256=golden_sample_sha256,
        software_golden_model_sha256=SCIPLEX3_CANDIDATE_GOLDEN_MODEL_SHA256,
        software_golden_sample_sha256=SCIPLEX3_CANDIDATE_GOLDEN_SAMPLE_SHA256,
        outer_iteration_count=cast(int, behavior["outer_iteration_count"]),
        initial_elbo=initial.elbo,
        initial_factor_order=initial.factor_order,
        initial_inner_sweep_count_histogram=initial.inner_sweep_count_histogram,
        initial_maximum_inner_sweeps=initial.maximum_inner_sweeps,
        initial_maximum_terminal_shape_residual=(initial.maximum_terminal_shape_residual),
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
        sampling_envelope_combination_count=sampling_certificate.combination_count,
        sampling_envelope_maximum_request_count=sampling_certificate.maximum_request_count,
        sampling_envelope_request_failure_budget_log=(
            sampling_certificate.request_failure_budget_log
        ),
        sampling_envelope_worst_request_tail_log_upper_bound=(
            sampling_certificate.worst_request_tail_log_upper_bound
        ),
        sampling_envelope_maximum_compound_poisson_intensity=(
            sampling_certificate.maximum_compound_poisson_intensity
        ),
        sampling_envelope_rejection_reasons=sampling_certificate.rejection_reasons,
        sampling_envelope_worst_action_id=sampling_certificate.worst_action_id,
        sampling_envelope_worst_context_id=sampling_certificate.worst_context_id,
        sampling_envelope_worst_tau_hex=sampling_certificate.worst_tau_hex,
    )


@dataclass(frozen=True, slots=True)
class FittedSciPlex3Candidate:
    """Exact reloaded candidate plus its acyclic model/observation artifacts."""

    candidate: SciPlex3GammaPoissonCandidate = field(repr=False)
    model_artifact: LocalContentAddressedArtifact
    observation: SciPlex3CandidateTrainingObservation
    observation_artifact: LocalContentAddressedArtifact
    training_plan_fingerprint: str
    can_mint_lifecycle_evidence: Literal[False] = False
    scientifically_admissible: Literal[False] = False

    def __post_init__(self) -> None:
        if type(self.candidate) is not SciPlex3GammaPoissonCandidate:
            raise SciPlex3CandidateRunnerError("fitted candidate has the wrong exact class")
        if (
            type(self.model_artifact) is not LocalContentAddressedArtifact
            or type(self.observation_artifact) is not LocalContentAddressedArtifact
            or type(self.observation) is not SciPlex3CandidateTrainingObservation
        ):
            raise SciPlex3CandidateRunnerError("fitted candidate artifact binding is not exact")
        _exact_sha256(self.training_plan_fingerprint, name="fitted training plan fingerprint")
        if (
            self.observation.plan_fingerprint != self.training_plan_fingerprint
            or self.observation.model_artifact_sha256 != self.model_artifact.sha256
            or self.observation.model_artifact_byte_count != self.model_artifact.byte_count
            or self.can_mint_lifecycle_evidence is not False
            or self.scientifically_admissible is not False
        ):
            raise SciPlex3CandidateRunnerError(
                "fitted candidate binding or authority flags drifted"
            )


def fit_and_write_sciplex3_candidate(
    preparation: SciPlex3BaselinePreparation,
    sealed_plan: SealedSciPlex3CandidateTrainingPlan,
    output_directory: Path,
) -> FittedSciPlex3Candidate:
    """Fit exact p1 bytes, then seal/reload the model before emitting an observation."""

    plan, design = _validate_sealed_plan(preparation, sealed_plan)
    candidate = _fit_exact_candidate(preparation, design)
    _validate_candidate_state(candidate, preparation, design)
    model_payload = candidate.canonical_model_bytes()
    if type(model_payload) is not bytes or not model_payload:
        raise SciPlex3CandidateRunnerError("candidate did not emit exact nonempty model bytes")
    # Loading before any write catches a noncanonical or internally inconsistent serializer.
    first_reload = _load_exact_candidate(model_payload, expected_sha256=_sha256(model_payload))
    if type(first_reload) is not SciPlex3GammaPoissonCandidate:
        raise SciPlex3CandidateRunnerError("candidate first reload substituted another class")
    if first_reload.canonical_model_bytes() != model_payload:
        raise SciPlex3CandidateRunnerError("candidate canonical bytes changed on first reload")
    _, first_golden = _sample_identity(first_reload)
    if _sample_identity(first_reload)[1] != first_golden:
        raise SciPlex3CandidateRunnerError("candidate golden sampling is not deterministic")

    # Reconstruct the complete pre-fit boundary again after fitting and before creating outputs.
    current_plan, current_design = _validate_sealed_plan(preparation, sealed_plan)
    if current_plan != plan or current_design != design:
        raise SciPlex3CandidateRunnerError("p1 plan or design changed during fitting")
    output = _item11._exclusive_directory(Path(output_directory))
    model_path = output / "candidate-model.json"
    _item11._write_exclusive(model_path, model_payload)
    model_artifact = _item11._verify_local_artifact(
        model_path, expected_payload=model_payload, media_type="application/json"
    )
    reread_payload = _read_local_artifact(model_artifact, name="sealed candidate model")
    reloaded = _load_exact_candidate(reread_payload, expected_sha256=model_artifact.sha256)
    if type(reloaded) is not SciPlex3GammaPoissonCandidate:
        raise SciPlex3CandidateRunnerError("candidate sealed reload substituted another class")
    if reloaded.canonical_model_bytes() != reread_payload:
        raise SciPlex3CandidateRunnerError("reloaded candidate bytes differ from sealed model")
    _validate_candidate_state(reloaded, preparation, design)
    _, reproduced = _sample_identity(reloaded)
    if reproduced != first_golden:
        raise SciPlex3CandidateRunnerError("reloaded candidate did not reproduce the golden sample")

    # Re-read and revalidate immediately before the observation binds the model identity.
    final_payload = _read_local_artifact(model_artifact, name="final candidate model")
    final_candidate = _load_exact_candidate(final_payload, expected_sha256=model_artifact.sha256)
    if type(final_candidate) is not SciPlex3GammaPoissonCandidate:
        raise SciPlex3CandidateRunnerError("candidate final reload substituted another class")
    _, final_golden = _sample_identity(final_candidate)
    if final_golden != first_golden:
        raise SciPlex3CandidateRunnerError("candidate changed before observation emission")
    observation = _make_observation(
        preparation,
        plan,
        final_candidate,
        model_artifact,
        golden_sample_sha256=final_golden,
    )
    observation_payload = _canonical_json(observation.manifest())
    observation_path = output / "training-execution-observation.json"
    _item11._write_exclusive(observation_path, observation_payload)
    observation_artifact = _item11._verify_local_artifact(
        observation_path,
        expected_payload=observation_payload,
        media_type="application/json",
    )
    fitted = FittedSciPlex3Candidate(
        candidate=final_candidate,
        model_artifact=model_artifact,
        observation=observation,
        observation_artifact=observation_artifact,
        training_plan_fingerprint=plan.fingerprint,
    )
    verify_sciplex3_candidate_fit(preparation, sealed_plan, fitted)
    return fitted


def verify_sciplex3_candidate_fit(
    preparation: SciPlex3BaselinePreparation,
    sealed_plan: SealedSciPlex3CandidateTrainingPlan,
    fitted: FittedSciPlex3Candidate,
) -> SciPlex3CandidateTrainingObservation:
    """Reconstruct all p1, plan, code, runtime, model, and golden identities."""

    if type(fitted) is not FittedSciPlex3Candidate:
        raise SciPlex3CandidateRunnerError("candidate fit verification requires the exact type")
    plan, design = _validate_sealed_plan(preparation, sealed_plan)
    if fitted.training_plan_fingerprint != plan.fingerprint:
        raise SciPlex3CandidateRunnerError("fitted candidate is bound to another training plan")
    model_payload = _read_local_artifact(fitted.model_artifact, name="candidate model")
    if fitted.candidate.canonical_model_bytes() != model_payload:
        raise SciPlex3CandidateRunnerError("in-memory candidate differs from sealed model bytes")
    candidate = _load_exact_candidate(model_payload, expected_sha256=fitted.model_artifact.sha256)
    if type(candidate) is not SciPlex3GammaPoissonCandidate:
        raise SciPlex3CandidateRunnerError(
            "candidate verification reload substituted another class"
        )
    _validate_candidate_state(candidate, preparation, design)
    _, golden = _sample_identity(candidate)
    expected = _make_observation(
        preparation,
        plan,
        candidate,
        fitted.model_artifact,
        golden_sample_sha256=golden,
    )
    observation_payload = _read_local_artifact(
        fitted.observation_artifact, name="candidate training observation"
    )
    parsed = _json_object(observation_payload, name="candidate training observation")
    if (
        observation_payload != _canonical_json(parsed)
        or parsed != expected.manifest()
        or fitted.observation != expected
        or expected.golden_sample_sha256 != golden
    ):
        raise SciPlex3CandidateRunnerError("candidate training observation is stale or forged")
    return expected


__all__ = [
    "SCIPLEX3_CANDIDATE_RUNNER_IMPLEMENTATION_VERSION",
    "FittedSciPlex3Candidate",
    "SciPlex3CandidateRunnerError",
    "SciPlex3CandidateTrainingObservation",
    "SealedSciPlex3CandidateTrainingPlan",
    "build_sciplex3_candidate_training_plan",
    "fit_and_write_sciplex3_candidate",
    "seal_sciplex3_candidate_training_plan",
    "verify_sciplex3_candidate_fit",
]
