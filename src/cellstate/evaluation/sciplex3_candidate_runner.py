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
import cellstate.evaluation.sciplex3_runner as _item11
from cellstate.backends.contracts import PortImplementationBinding, PortImplementationKind
from cellstate.backends.training import (
    TRAINED_CANDIDATE_FACTORY_INTERFACE,
    CandidateTrainingPlan,
)
from cellstate.data.benchmarks import ContentAddressedArtifact
from cellstate.domain.common import canonical_json_bytes
from cellstate.errors import ContractViolationError
from cellstate.evaluation.sciplex3_baselines import (
    RNG_ALGORITHM,
    CompoundDose,
    NoAction,
)
from cellstate.evaluation.sciplex3_candidate import (
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
    SCIPLEX3_CANDIDATE_TRAINING_RECORD_COUNT,
    SCIPLEX3_CANDIDATE_TRAINING_WELL_COUNT,
    CandidateRawCountSamples,
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

SCIPLEX3_CANDIDATE_RUNNER_IMPLEMENTATION_VERSION: Final = "4.0.0"
SCIPLEX3_CANDIDATE_TRAINING_PLAN_ID: Final = SCIPLEX3_CANDIDATE_MODEL_ID
SCIPLEX3_CANDIDATE_TRAINING_PLAN_VERSION: Final = "4.0.0"
SCIPLEX3_CANDIDATE_OPTIMIZATION_SEED: Final = 0

_P1_PARTITION_ID = "p1-train"
_RAW_BASE = "https://raw.githubusercontent.com/logannye/cellstate/main/"
_LOADER_CONTRACT_RELATIVE_PATH = (
    "benchmarks/vertical-a/sciplex3-k562-24h-v1/support/p1-loader-contract.json"
)
_CANDIDATE_CODE_RELATIVE_PATH = "src/cellstate/evaluation/sciplex3_candidate.py"
_RUNNER_CODE_RELATIVE_PATH = "src/cellstate/evaluation/sciplex3_candidate_runner.py"
_THREAD_ENVIRONMENT_KEYS = (
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)
_SUPPORT_FILENAMES = (
    "candidate-specification.json",
    "output-model-schema.json",
    "p1-count-stream-descriptor.json",
    "runtime-lock.json",
)

if _candidate_module.__file__ is None:  # pragma: no cover - import boundary
    raise ImportError("loaded sci-Plex3 candidate module has no source path")
_IMPORTED_CANDIDATE_CODE_PATH: Final = Path(_candidate_module.__file__).resolve()
_IMPORTED_CANDIDATE_CODE_SHA256: Final = hashlib.sha256(
    _IMPORTED_CANDIDATE_CODE_PATH.read_bytes()
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
    media_type: str = "application/json",
) -> ContentAddressedArtifact:
    return ContentAddressedArtifact(
        artifact_id=artifact_id,
        uri=_RAW_BASE + relative_uri,
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


def _support_payloads(
    preparation: SciPlex3BaselinePreparation,
    *,
    design: SciPlex3P1DesignBindings,
) -> dict[str, bytes]:
    specification = _canonical_json(candidate_specification_manifest())
    if _sha256(specification) != SCIPLEX3_CANDIDATE_SPECIFICATION_SHA256:
        raise SciPlex3CandidateRunnerError("candidate specification identity drifted")
    return {
        "candidate-specification.json": specification,
        "output-model-schema.json": _output_model_schema_payload(),
        "p1-count-stream-descriptor.json": _count_stream_descriptor_payload(
            preparation, design=design
        ),
        "runtime-lock.json": _runtime_lock_payload(),
    }


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
    support_payloads = _support_payloads(preparation, design=design)
    # The runtime gate precedes executable-golden construction so supported nonreference
    # interpreters can import this module while fitting still fails before numerical work.
    _verify_factory_golden()
    repository_root = preparation.repository_root
    if (
        repository_root / _CANDIDATE_CODE_RELATIVE_PATH
    ).resolve() != _IMPORTED_CANDIDATE_CODE_PATH or (
        repository_root / _RUNNER_CODE_RELATIVE_PATH
    ).resolve() != _IMPORTED_RUNNER_CODE_PATH:
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

    p1_loader_contract = _artifact_for_payload(
        artifact_id="sciplex3-item12-p1-loader-contract",
        relative_uri=_LOADER_CONTRACT_RELATIVE_PATH,
        payload=loader_payload,
    )
    p1_count_stream = _artifact_for_payload(
        artifact_id="sciplex3-item12-p1-count-stream-descriptor",
        relative_uri=(
            "benchmarks/artifacts/sciplex3-k562-24h-v1/item12-p1/p1-count-stream-descriptor.json"
        ),
        payload=support_payloads["p1-count-stream-descriptor.json"],
    )
    candidate_specification = _artifact_for_payload(
        artifact_id="sciplex3-item12-candidate-specification",
        relative_uri=(
            "benchmarks/vertical-a/sciplex3-k562-24h-v1/support/candidate-specification.json"
        ),
        payload=support_payloads["candidate-specification.json"],
    )
    output_model_schema = _artifact_for_payload(
        artifact_id="sciplex3-item12-output-model-schema",
        relative_uri=(
            "benchmarks/vertical-a/sciplex3-k562-24h-v1/support/candidate-output-model-schema.json"
        ),
        payload=support_payloads["output-model-schema.json"],
    )
    runtime_lock = _artifact_for_payload(
        artifact_id="sciplex3-item12-runtime-lock",
        relative_uri=(
            "benchmarks/vertical-a/sciplex3-k562-24h-v1/support/candidate-runtime-lock.json"
        ),
        payload=support_payloads["runtime-lock.json"],
    )
    trainer_code = _artifact_for_payload(
        artifact_id="sciplex3-item12-candidate-runner-code",
        relative_uri=_RUNNER_CODE_RELATIVE_PATH,
        payload=runner_code,
        media_type="text/x-python",
    )
    factory_code = _artifact_for_payload(
        artifact_id="sciplex3-item12-candidate-factory-code",
        relative_uri=_CANDIDATE_CODE_RELATIVE_PATH,
        payload=candidate_code,
        media_type="text/x-python",
    )
    identity = _item11._recompute_in_memory_p1_identity(preparation)
    plan = CandidateTrainingPlan(
        plan_id=SCIPLEX3_CANDIDATE_TRAINING_PLAN_ID,
        plan_version=SCIPLEX3_CANDIDATE_TRAINING_PLAN_VERSION,
        query_fingerprint=preparation.design.query_fingerprint,
        benchmark_fingerprint=benchmark_fingerprint,
        support_envelope_fingerprint=support_envelope_fingerprint,
        training_partition_ids=(_P1_PARTITION_ID,),
        p1_loader_contract=p1_loader_contract,
        p1_count_stream=p1_count_stream,
        p1_count_stream_sha256=identity.panel_count_stream_sha256,
        p1_finalized_count_scan_fingerprint=(preparation.finalized_count_scan_receipt.fingerprint),
        p1_assembly_fingerprint=preparation.receipt.fingerprint,
        p1_design_fingerprint=design.fingerprint,
        ordered_feature_keys_sha256=identity.ordered_feature_keys_sha256,
        action_binding_sha256=preparation.design.action_domain_sha256,
        target_value_schema_sha256=preparation.design.target_value_schema_sha256,
        candidate_specification=candidate_specification,
        output_model_schema=output_model_schema,
        runtime_lock=runtime_lock,
        trainer_implementation=PortImplementationBinding(
            implementation_id="cellstate.sciplex3-candidate-runner",
            implementation_version=SCIPLEX3_CANDIDATE_RUNNER_IMPLEMENTATION_VERSION,
            interface=(
                "cellstate.evaluation.sciplex3_candidate_runner.fit_and_write_sciplex3_candidate"
            ),
            kind=PortImplementationKind.PYTHON_ENTRY_POINT,
            code_artifact=trainer_code,
            entrypoint=(
                "cellstate.evaluation.sciplex3_candidate_runner:fit_and_write_sciplex3_candidate"
            ),
        ),
        candidate_factory_implementation=PortImplementationBinding(
            implementation_id="cellstate.sciplex3-gamma-poisson-candidate-factory",
            implementation_version=SCIPLEX3_CANDIDATE_IMPLEMENTATION_VERSION,
            interface=TRAINED_CANDIDATE_FACTORY_INTERFACE,
            kind=PortImplementationKind.PYTHON_ENTRY_POINT,
            code_artifact=factory_code,
            entrypoint=("cellstate.evaluation.sciplex3_candidate:SciPlex3GammaPoissonCandidate"),
        ),
        optimization_seed=SCIPLEX3_CANDIDATE_OPTIMIZATION_SEED,
        deterministic_thread_count=1,
        future_calibration_plan=None,
    )
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
    values = np.asarray(sample.samples, dtype="<i8", order="C")
    request_manifest = _golden_request_manifest(sample)
    identity = _sha256(
        _canonical_json(
            {
                "request": request_manifest,
                "rng_algorithm": sample.rng_algorithm,
                "sample_bytes_sha256": _sha256(values.tobytes(order="C")),
                "sample_shape": list(values.shape),
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
        raise SciPlex3CandidateRunnerError("candidate empirical plate contexts are invalid")
    rho_means = np.asarray(
        [
            math.fsum(float(rho[plate, factor]) for plate in range(SCIPLEX3_CANDIDATE_PLATE_COUNT))
            / SCIPLEX3_CANDIDATE_PLATE_COUNT
            for factor in range(SCIPLEX3_CANDIDATE_FACTOR_COUNT)
        ],
        dtype=np.float64,
    )
    if not bool(np.allclose(rho_means, 1.0, rtol=0.0, atol=5e-13)):
        raise SciPlex3CandidateRunnerError("candidate empirical plate contexts are invalid")
    rho_sha256 = _sha256(rho.tobytes(order="C"))

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
        or behavior["plate_context_count"] != SCIPLEX3_CANDIDATE_PLATE_COUNT
        or behavior["plate_context_factorwise_mean_one"] is not True
        or behavior["plate_context_family"] != "uniform-whole-p1-rho-row"
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
        or cast(dict[str, object], tensor_sha256).get("rho") != rho_sha256
        or cast(dict[str, object], tensor_sha256).get("factor_shape")
        != _sha256(fixed_shape.tobytes(order="C"))
    ):
        raise SciPlex3CandidateRunnerError("candidate fitted state specification drifted")
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
    loader_code_sha256: str
    item11_runner_code_sha256: str
    candidate_runner_code_sha256: str
    candidate_factory_code_sha256: str
    model_artifact_sha256: str
    model_artifact_byte_count: int
    fitted_state_sha256: str
    behavior_sha256: str
    plate_context_rho_sha256: str
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
    artifact_schema: Literal["sciplex3-candidate-training-execution-observation"] = (
        "sciplex3-candidate-training-execution-observation"
    )
    artifact_schema_version: Literal["4.0.0"] = "4.0.0"
    candidate_model_schema_version: Literal["4.0.0"] = "4.0.0"
    fit_converged: Literal[True] = True
    all_parameters_finite: Literal[True] = True
    factor_order_stable: Literal[True] = True
    factor_shape_mode: Literal["fixed"] = "fixed"
    factor_shape_estimated: Literal[False] = False
    inner_equilibration_performed: Literal[True] = True
    inner_all_batches_converged: Literal[True] = True
    plate_context_family: Literal["uniform-whole-p1-rho-row"] = "uniform-whole-p1-rho-row"
    plate_context_count: Literal[8] = 8
    plate_context_factorwise_mean_one: Literal[True] = True
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
            or self.artifact_schema_version != "4.0.0"
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
            "loader_code_sha256",
            "item11_runner_code_sha256",
            "candidate_runner_code_sha256",
            "candidate_factory_code_sha256",
            "model_artifact_sha256",
            "fitted_state_sha256",
            "behavior_sha256",
            "plate_context_rho_sha256",
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
            or self.software_golden_model_sha256 != SCIPLEX3_CANDIDATE_GOLDEN_MODEL_SHA256
            or self.software_golden_sample_sha256 != SCIPLEX3_CANDIDATE_GOLDEN_SAMPLE_SHA256
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
            or self.plate_context_family != "uniform-whole-p1-rho-row"
            or type(self.plate_context_count) is not int
            or self.plate_context_count != SCIPLEX3_CANDIDATE_PLATE_COUNT
        ):
            raise SciPlex3CandidateRunnerError(
                "observation fixed-shape or empirical-plate family drifted"
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
        loader_code_sha256=_item11.SCIPLEX3_LOADER_CODE_SHA256,
        item11_runner_code_sha256=_item11._IMPORTED_RUNNER_CODE_SHA256,
        candidate_runner_code_sha256=_IMPORTED_RUNNER_CODE_SHA256,
        candidate_factory_code_sha256=_IMPORTED_CANDIDATE_CODE_SHA256,
        model_artifact_sha256=model_artifact.sha256,
        model_artifact_byte_count=model_artifact.byte_count,
        fitted_state_sha256=_sha256(_canonical_json(fitted_state)),
        behavior_sha256=_sha256(_canonical_json(behavior)),
        plate_context_rho_sha256=cast(str, tensor_sha256["rho"]),
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
