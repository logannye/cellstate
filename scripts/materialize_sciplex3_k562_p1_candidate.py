#!/usr/bin/env python3
"""Check legacy sci-Plex3 candidate materializations; direct materialization is retired.

Both the public and compatibility materialization entry points fail before path or source access.
Only the Item 12.2 contained v5 supervisor may launch a future source-touching worker. ``--check``
reauthenticates an already published deterministic closure and never opens, stats, or hashes the
source H5AD.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import resource
import sys
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import asdict
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import cellstate.backends.sciplex3_loader as _loader_module
import cellstate.evaluation.sciplex3_candidate as _candidate_module
import cellstate.evaluation.sciplex3_candidate_runner as _candidate_runner_module
import cellstate.evaluation.sciplex3_runner as _item11_runner_module
from cellstate.backends.sciplex3_k562 import PopulationComponentAccessPurpose
from cellstate.backends.sciplex3_loader import (
    SCIPLEX3_SOURCE_BYTE_COUNT,
    SCIPLEX3_SOURCE_FILENAME,
    SCIPLEX3_SOURCE_MD5,
    SCIPLEX3_SOURCE_SHA256,
    SciPlex3K562H5ADLoader,
    SciPlex3P1FinalizedCountScanReceipt,
)
from cellstate.backends.training import CandidateTrainingPlan
from cellstate.domain.common import canonical_json_bytes
from cellstate.evaluation.sciplex3_candidate import (
    SCIPLEX3_CANDIDATE_FIXED_FACTOR_SHAPE,
    SCIPLEX3_CANDIDATE_GOLDEN_MODEL_SHA256,
    SCIPLEX3_CANDIDATE_GOLDEN_SAMPLE_SHA256,
    SCIPLEX3_CANDIDATE_IMPLEMENTATION_VERSION,
    SCIPLEX3_CANDIDATE_MAX_OUTER_ITERATIONS,
    SCIPLEX3_CANDIDATE_MIN_OUTER_ITERATIONS,
    SCIPLEX3_CANDIDATE_MODEL_ID,
    SCIPLEX3_CANDIDATE_MODEL_SCHEMA,
    SCIPLEX3_CANDIDATE_MODEL_SCHEMA_VERSION,
    SCIPLEX3_CANDIDATE_REFERENCE_RUNTIME,
    SciPlex3GammaPoissonCandidate,
)
from cellstate.evaluation.sciplex3_candidate_runner import (
    SciPlex3CandidateTrainingObservation,
    build_sciplex3_candidate_training_plan,
    fit_and_write_sciplex3_candidate,
    seal_sciplex3_candidate_training_plan,
    verify_sciplex3_candidate_fit,
)
from cellstate.evaluation.sciplex3_runner import (
    SciPlex3BaselinePreparation,
    SciPlex3P1AssemblyReceipt,
    assemble_sciplex3_p1_training_data,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_RELATIVE_DIRECTORY = Path("benchmarks/vertical-a/sciplex3-k562-24h-v1")
DEFAULT_OUTPUT_DIRECTORY = REPOSITORY_ROOT / BENCHMARK_RELATIVE_DIRECTORY / "item12-p1"
COUNT_DESCRIPTOR_RELATIVE_PATH = Path(
    "benchmarks/artifacts/sciplex3-k562-24h-v1/item12-p1/p1-count-stream-descriptor.json"
)
SUPPORT_RELATIVE_PATHS: Mapping[str, Path] = {
    "candidate-specification.json": (
        BENCHMARK_RELATIVE_DIRECTORY / "support/candidate-specification.json"
    ),
    "output-model-schema.json": (
        BENCHMARK_RELATIVE_DIRECTORY / "support/candidate-output-model-schema.json"
    ),
    "runtime-lock.json": BENCHMARK_RELATIVE_DIRECTORY / "support/candidate-runtime-lock.json",
}
MATERIALIZATION_MANIFEST = "materialization-manifest.json"
FINALIZED_SCAN_RECEIPT = "p1-finalized-count-scan-receipt.json"
ASSEMBLY_RECEIPT = "p1-assembly-receipt.json"
TRAINING_PLAN = "candidate-training-plan.json"
CANDIDATE_MODEL = "candidate-model.json"
TRAINING_OBSERVATION = "training-execution-observation.json"
MATERIALIZATION_BATCH_SIZE = 512
FIT_WALL_LIMIT_SECONDS = 60 * 60
FIT_RSS_LIMIT_BYTES = 4 * 1024**3
_THREAD_ENVIRONMENT_KEYS = (
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)

# The import boundary captures exact v5 bytes; the contained execution-input manifest binds the
# same digests before Docker imports this module from its sealed code tree.
EXPECTED_CANDIDATE_CODE_SHA256 = _candidate_runner_module._IMPORTED_CANDIDATE_CODE_SHA256
EXPECTED_CANDIDATE_RUNNER_CODE_SHA256 = _candidate_runner_module._IMPORTED_RUNNER_CODE_SHA256

_BUILDER_RELATIVE_PATH = Path("scripts/build_sciplex3_k562_trained_candidate.py")
_REPOSITORY_BINDING_PATHS: Mapping[str, Path] = {
    "action_domain": BENCHMARK_RELATIVE_DIRECTORY / "support/action-domain-mapping.json",
    "benchmark": BENCHMARK_RELATIVE_DIRECTORY / "benchmark-artifact.json",
    "candidate_code": Path("src/cellstate/evaluation/sciplex3_candidate.py"),
    "candidate_runner_code": Path("src/cellstate/evaluation/sciplex3_candidate_runner.py"),
    "dataset_manifest": Path("data_manifests/reviewed/sciplex3-k562-24h.json"),
    "feature_panel": Path("benchmarks/artifacts/sciplex3-k562-24h-v1/feature-panel.json"),
    "item11_runner_code": Path("src/cellstate/evaluation/sciplex3_runner.py"),
    "loader_code": Path("src/cellstate/backends/sciplex3_loader.py"),
    "loader_contract": BENCHMARK_RELATIVE_DIRECTORY / "support/p1-loader-contract.json",
    "materializer_code": Path("scripts/materialize_sciplex3_k562_p1_candidate.py"),
    "query": BENCHMARK_RELATIVE_DIRECTORY / "state-query.json",
    "scoring_transform": BENCHMARK_RELATIVE_DIRECTORY / "support/scoring-transform.json",
    "target_value_schema": BENCHMARK_RELATIVE_DIRECTORY / "support/target-value-schema.json",
    "trained_candidate_builder_code": _BUILDER_RELATIVE_PATH,
}

_SAFETY_BOUNDARY: Mapping[str, object] = {
    "accessed_partition_roles": ["p1-train"],
    "admission_authority_issued": False,
    "calibration_performed": False,
    "can_mint_lifecycle_evidence": False,
    "candidate_fit_receipt_issued": False,
    "heldout_artifacts_resolved": False,
    "heldout_memberships_read": False,
    "heldout_outcomes_read": False,
    "hmac_receipts_persisted": False,
    "lifecycle_evidence_issued": False,
    "metrics_computed": False,
    "model_selection_performed": False,
    "p2_calibration_accessed": False,
    "p3_model_selection_accessed": False,
    "p4_untouched_test_accessed": False,
    "public_runtime_registered": False,
    "scientifically_admissible": False,
    "training_source_selection_receipt_issued": False,
    "trusted_workflow_receipt_issued": False,
}

_ARTIFACT_REFERENCE_FIELDS = frozenset({"byte_count", "media_type", "relative_path", "sha256"})
_RUNTIME_LOCK_FIELDS = frozenset(
    {"artifact_schema", "artifact_schema_version", "runtime", "thread_environment"}
)
_COUNT_DESCRIPTOR_FIELDS = frozenset(
    {
        "artifact_schema",
        "artifact_schema_version",
        "assembly_fingerprint",
        "authority",
        "candidate_design_fingerprint",
        "count_stream_encoding",
        "finalized_count_scan_fingerprint",
        "ordered_feature_keys_sha256",
        "panel_count_stream_sha256",
        "panel_nonzero_count",
        "panel_umi_total",
        "record_count",
        "training_partition_ids",
        "well_count",
        "zero_panel_record_count",
    }
)
_COUNT_DESCRIPTOR_AUTHORITY_FIELDS = frozenset(
    {
        "can_mint_lifecycle_evidence",
        "heldout_memberships_read",
        "heldout_outcomes_read",
        "scientifically_admissible",
    }
)
_FINALIZED_SCAN_FIELDS = frozenset(
    {
        "initial_source_authentication_fingerprint",
        "source_sha256",
        "source_md5",
        "source_byte_count",
        "source_descriptor_identity_before",
        "source_descriptor_identity_after",
        "loader_implementation_sha256",
        "p1_loader_contract_sha256",
        "dataset_manifest_sha256",
        "query_sha256",
        "benchmark_sha256",
        "target_value_schema_sha256",
        "scoring_transform_sha256",
        "feature_panel_artifact_sha256",
        "ordered_feature_keys_sha256",
        "record_ids_sha256",
        "record_to_well_sha256",
        "emitted_source_row_indices_sha256",
        "ordered_record_source_well_condition_sha256",
        "count_stream_encoding",
        "panel_count_stream_sha256",
        "record_count",
        "well_count",
        "treated_well_count",
        "control_well_count",
        "batch_count",
        "panel_nonzero_count",
        "zero_panel_record_count",
        "panel_umi_total",
        "full_source_umi_total",
        "python_version",
        "python_implementation",
        "numpy_version",
        "h5py_version",
        "hdf5_version",
        "artifact_schema",
        "artifact_schema_version",
        "loader_interface_id",
        "partition_id",
        "access_purpose",
        "accessed_partition_roles",
        "accessed_count_datasets",
        "exact_record_coverage",
        "count_scan_complete",
        "source_descriptor_reverified",
        "close_reverification_completed",
        "finalized",
        "heldout_memberships_parsed",
        "heldout_outcome_values_parsed",
        "trusted_workflow_receipt_present",
        "lifecycle_evidence_issued",
        "scientifically_admissible",
    }
)
_ASSEMBLY_FIELDS = frozenset(
    {
        "loader_source_scan_fingerprint",
        "finalized_count_scan_fingerprint",
        "loader_implementation_sha256",
        "loader_contract_sha256",
        "source_sha256",
        "record_count",
        "well_count",
        "treated_well_count",
        "control_well_count",
        "record_ids_sha256",
        "record_to_well_sha256",
        "well_ids_sha256",
        "well_to_condition_sha256",
        "source_row_indices_sha256",
        "emitted_source_row_indices_sha256",
        "ordered_record_source_well_condition_sha256",
        "runner_panel_count_stream_sha256",
        "loader_panel_count_stream_sha256",
        "panel_nonzero_count",
        "zero_panel_record_count",
        "panel_umi_total",
        "full_source_umi_total",
        "batch_count",
        "ordered_feature_keys_sha256",
        "feature_panel_artifact_sha256",
        "action_domain_sha256",
        "query_sha256",
        "benchmark_sha256",
        "scoring_transform_sha256",
        "target_value_schema_sha256",
        "partition_id",
        "access_purpose",
        "exact_record_coverage",
        "count_scan_complete",
        "close_reverification_completed",
        "heldout_memberships_read",
        "heldout_outcomes_read",
        "can_mint_lifecycle_evidence",
        "scientifically_admissible",
    }
)


class CandidateMaterializationError(RuntimeError):
    """Raised when Item 12 cannot be materialized or reauthenticated exactly."""


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _read_bytes(path: Path, *, name: str) -> bytes:
    try:
        return path.read_bytes()
    except OSError as error:
        raise CandidateMaterializationError(f"cannot read {name}: {path}") from error


def _canonical_object(payload: bytes, *, name: str) -> dict[str, object]:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CandidateMaterializationError(f"{name} is not valid JSON") from error
    if type(value) is not dict or any(type(key) is not str for key in value):
        raise CandidateMaterializationError(f"{name} must be one exact JSON object")
    result = cast(dict[str, object], value)
    if canonical_json_bytes(result) != payload:
        raise CandidateMaterializationError(f"{name} is not canonical JSON")
    return result


def _load_canonical_object(path: Path, *, name: str) -> tuple[dict[str, object], bytes]:
    payload = _read_bytes(path, name=name)
    return _canonical_object(payload, name=name), payload


def _as_mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(type(key) is not str for key in value):
        raise CandidateMaterializationError(f"{name} must be an object")
    return cast(Mapping[str, object], value)


def _as_list(value: object, *, name: str) -> list[object]:
    if type(value) is not list:
        raise CandidateMaterializationError(f"{name} must be an array")
    return cast(list[object], value)


def _require_exact_fields(
    mapping: Mapping[str, object], expected: frozenset[str], *, name: str
) -> None:
    if set(mapping) != expected:
        raise CandidateMaterializationError(f"{name} field closure drifted")


def _json_values_are_exact(left: object, right: object) -> bool:
    return canonical_json_bytes(left) == canonical_json_bytes(right)


def _is_exact_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validated_finalized_scan(
    mapping: Mapping[str, object], *, name: str
) -> SciPlex3P1FinalizedCountScanReceipt:
    _require_exact_fields(mapping, _FINALIZED_SCAN_FIELDS, name=name)
    values = dict(mapping)
    for field_name in (
        "source_descriptor_identity_before",
        "source_descriptor_identity_after",
        "accessed_partition_roles",
        "accessed_count_datasets",
    ):
        values[field_name] = tuple(_as_list(values.get(field_name), name=f"{name} {field_name}"))
    try:
        receipt = SciPlex3P1FinalizedCountScanReceipt(**cast(Any, values))
    except Exception as error:
        raise CandidateMaterializationError(f"{name} primitive/container schema drifted") from error
    if canonical_json_bytes(asdict(receipt)) != canonical_json_bytes(dict(mapping)):
        raise CandidateMaterializationError(f"{name} canonical dataclass projection drifted")
    if (
        receipt.artifact_schema != "sciplex3-k562-p1-finalized-count-scan-receipt"
        or receipt.artifact_schema_version != "1.0.0"
        or receipt.loader_interface_id != "cellstate.sciplex3-training-data-loader.v1"
        or receipt.partition_id != "p1-train"
        or receipt.access_purpose != "train_parameters"
        or receipt.accessed_partition_roles != ("p1-train",)
        or receipt.accessed_count_datasets != ("X.data", "X.indices", "X.indptr", "obs.ncounts")
        or receipt.source_descriptor_identity_before != receipt.source_descriptor_identity_after
        or receipt.exact_record_coverage is not True
        or receipt.count_scan_complete is not True
        or receipt.source_descriptor_reverified is not True
        or receipt.close_reverification_completed is not True
        or receipt.finalized is not True
        or receipt.heldout_memberships_parsed is not False
        or receipt.heldout_outcome_values_parsed is not False
        or receipt.trusted_workflow_receipt_present is not False
        or receipt.lifecycle_evidence_issued is not False
        or receipt.scientifically_admissible is not False
    ):
        raise CandidateMaterializationError(f"{name} p1-only safety/schema closure drifted")
    return receipt


def _validated_assembly(mapping: Mapping[str, object], *, name: str) -> SciPlex3P1AssemblyReceipt:
    _require_exact_fields(mapping, _ASSEMBLY_FIELDS, name=name)
    try:
        receipt = SciPlex3P1AssemblyReceipt(**cast(Any, dict(mapping)))
    except Exception as error:
        raise CandidateMaterializationError(f"{name} primitive schema drifted") from error
    if canonical_json_bytes(asdict(receipt)) != canonical_json_bytes(dict(mapping)):
        raise CandidateMaterializationError(f"{name} canonical dataclass projection drifted")
    if (
        receipt.partition_id != "p1-train"
        or receipt.access_purpose != "train_parameters"
        or receipt.exact_record_coverage is not True
        or receipt.count_scan_complete is not True
        or receipt.close_reverification_completed is not True
        or receipt.heldout_memberships_read is not False
        or receipt.heldout_outcomes_read is not False
        or receipt.can_mint_lifecycle_evidence is not False
        or receipt.scientifically_admissible is not False
    ):
        raise CandidateMaterializationError(f"{name} p1-only safety closure drifted")
    return receipt


def _write_exclusive(path: Path, payload: bytes) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            written = stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except OSError as error:
        raise CandidateMaterializationError(f"cannot write staged artifact: {path}") from error
    if written != len(payload) or _read_bytes(path, name="new staged artifact") != payload:
        raise CandidateMaterializationError(f"staged artifact failed immediate re-read: {path}")


def _write_canonical_exclusive(path: Path, value: object) -> None:
    _write_exclusive(path, canonical_json_bytes(value))


def _repository_reference(repository_root: Path, relative_path: Path) -> dict[str, object]:
    payload = _read_bytes(repository_root / relative_path, name=relative_path.as_posix())
    return {
        "byte_count": len(payload),
        "relative_path": relative_path.as_posix(),
        "sha256": _sha256(payload),
    }


def _repository_bindings(repository_root: Path) -> dict[str, object]:
    bindings: dict[str, object] = {
        name: _repository_reference(repository_root, relative_path)
        for name, relative_path in sorted(_REPOSITORY_BINDING_PATHS.items())
    }
    candidate = _as_mapping(bindings["candidate_code"], name="candidate code binding")
    if candidate.get("sha256") != EXPECTED_CANDIDATE_CODE_SHA256:
        raise CandidateMaterializationError(
            "candidate source differs from the independently frozen Item 12 identity"
        )
    runner = _as_mapping(bindings["candidate_runner_code"], name="candidate runner binding")
    if runner.get("sha256") != EXPECTED_CANDIDATE_RUNNER_CODE_SHA256:
        raise CandidateMaterializationError(
            "candidate runner differs from the independently frozen Item 12 identity"
        )
    return bindings


def _verify_imported_module_provenance(
    repository_root: Path, repository_bindings: Mapping[str, object]
) -> None:
    """Bind every source-opening/executing import to current frozen repository bytes."""

    modules: tuple[tuple[str, ModuleType, str, Path, Path | None, str, str | None], ...] = (
        (
            "loader",
            _loader_module,
            "loader_code",
            _REPOSITORY_BINDING_PATHS["loader_code"],
            Path(_loader_module.__file__).resolve() if _loader_module.__file__ else None,
            _loader_module._LOADER_IMPLEMENTATION_IMPORT_SHA256,
            _item11_runner_module.SCIPLEX3_LOADER_CODE_SHA256,
        ),
        (
            "Item 11 runner",
            _item11_runner_module,
            "item11_runner_code",
            _REPOSITORY_BINDING_PATHS["item11_runner_code"],
            _item11_runner_module._IMPORTED_RUNNER_CODE_PATH,
            _item11_runner_module._IMPORTED_RUNNER_CODE_SHA256,
            None,
        ),
        (
            "candidate",
            _candidate_module,
            "candidate_code",
            _REPOSITORY_BINDING_PATHS["candidate_code"],
            _candidate_runner_module._IMPORTED_CANDIDATE_CODE_PATH,
            _candidate_runner_module._IMPORTED_CANDIDATE_CODE_SHA256,
            EXPECTED_CANDIDATE_CODE_SHA256,
        ),
        (
            "candidate runner",
            _candidate_runner_module,
            "candidate_runner_code",
            _REPOSITORY_BINDING_PATHS["candidate_runner_code"],
            _candidate_runner_module._IMPORTED_RUNNER_CODE_PATH,
            _candidate_runner_module._IMPORTED_RUNNER_CODE_SHA256,
            EXPECTED_CANDIDATE_RUNNER_CODE_SHA256,
        ),
    )
    for (
        name,
        module,
        binding_name,
        relative_path,
        imported_path,
        imported_sha256,
        frozen_sha256,
    ) in modules:
        module_file = module.__file__
        if type(module_file) is not str or imported_path is None:
            raise CandidateMaterializationError(f"loaded {name} module has no exact source path")
        repository_path = (repository_root / relative_path).resolve()
        if (
            Path(module_file).resolve() != repository_path
            or Path(imported_path).resolve() != repository_path
        ):
            raise CandidateMaterializationError(
                f"loaded {name} module path differs from the repository closure"
            )
        payload = _read_bytes(repository_path, name=f"loaded {name} module")
        digest = _sha256(payload)
        if (
            digest != _binding_sha256(repository_bindings, binding_name)
            or digest != imported_sha256
            or (frozen_sha256 is not None and digest != frozen_sha256)
        ):
            raise CandidateMaterializationError(
                f"loaded {name} module bytes differ from the imported/frozen closure"
            )
    if (
        SciPlex3K562H5ADLoader is not _loader_module.SciPlex3K562H5ADLoader
        or assemble_sciplex3_p1_training_data
        is not _item11_runner_module.assemble_sciplex3_p1_training_data
        or SciPlex3GammaPoissonCandidate is not _candidate_module.SciPlex3GammaPoissonCandidate
        or build_sciplex3_candidate_training_plan
        is not _candidate_runner_module.build_sciplex3_candidate_training_plan
        or seal_sciplex3_candidate_training_plan
        is not _candidate_runner_module.seal_sciplex3_candidate_training_plan
        or fit_and_write_sciplex3_candidate
        is not _candidate_runner_module.fit_and_write_sciplex3_candidate
        or verify_sciplex3_candidate_fit
        is not _candidate_runner_module.verify_sciplex3_candidate_fit
    ):
        raise CandidateMaterializationError("loaded Item 12 executable symbols were substituted")


def _binding_sha256(bindings: Mapping[str, object], name: str) -> str:
    reference = _as_mapping(bindings.get(name), name=f"repository binding {name}")
    digest = reference.get("sha256")
    if type(digest) is not str or len(digest) != 64:
        raise CandidateMaterializationError(f"repository binding {name} has no SHA-256")
    return digest


def _artifact_reference(
    payload: bytes,
    *,
    relative_path: Path,
    media_type: str = "application/json",
) -> dict[str, object]:
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise CandidateMaterializationError("artifact reference escapes the repository")
    return {
        "byte_count": len(payload),
        "media_type": media_type,
        "relative_path": relative_path.as_posix(),
        "sha256": _sha256(payload),
    }


def _output_relative_path(output_directory: Path, repository_root: Path, filename: str) -> Path:
    try:
        relative_directory = output_directory.relative_to(repository_root)
    except ValueError as error:
        raise CandidateMaterializationError(
            "output directory must remain inside the repository root"
        ) from error
    return relative_directory / filename


def _load_builder(repository_root: Path) -> ModuleType:
    path = repository_root / _BUILDER_RELATIVE_PATH
    if not path.is_file():
        raise CandidateMaterializationError(f"missing trained-candidate builder: {path}")
    spec = importlib.util.spec_from_file_location("_cellstate_item12_builder", path)
    if spec is None or spec.loader is None:
        raise CandidateMaterializationError("cannot load the trained-candidate builder")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as error:
        sys.modules.pop(spec.name, None)
        raise CandidateMaterializationError("trained-candidate builder import failed") from error
    return module


def _planned_support_envelope(repository_root: Path) -> tuple[str, bytes]:
    module = _load_builder(repository_root)
    build = getattr(module, "build_trained_candidate_support_envelope", None)
    render = getattr(module, "trained_candidate_support_bytes", None)
    if not callable(build) or not callable(render):
        raise CandidateMaterializationError(
            "trained-candidate builder lacks the frozen pre-fit support API"
        )
    try:
        envelope = build()
        payload = render()
        fingerprint = envelope.fingerprint
        expected = canonical_json_bytes(envelope.model_dump(mode="json"))
    except Exception as error:
        raise CandidateMaterializationError(
            "cannot construct the pre-fit trained-candidate support envelope"
        ) from error
    if type(payload) is not bytes or payload != expected or _sha256(payload) != fingerprint:
        raise CandidateMaterializationError(
            "pre-fit trained-candidate support bytes are not exact or canonical"
        )
    return cast(str, fingerprint), payload


def _require_reference_runtime() -> bytes:
    # The runner owns the detailed runtime lock (including the exact BLAS implementation).  Keep
    # the Linux gate explicit here so the source is never opened on a merely lookalike platform.
    import cellstate.evaluation.sciplex3_candidate_runner as runner

    if platform.system() != "Linux":
        raise CandidateMaterializationError("candidate fitting requires exact Linux x86_64")
    try:
        payload = runner._runtime_lock_payload()
    except Exception as error:
        raise CandidateMaterializationError("candidate reference-runtime gate failed") from error
    value = _canonical_object(payload, name="candidate runtime lock")
    runtime = _as_mapping(value.get("runtime"), name="candidate runtime-lock identity")
    for key, expected in SCIPLEX3_CANDIDATE_REFERENCE_RUNTIME.items():
        if runtime.get(key) != expected:
            raise CandidateMaterializationError(f"candidate runtime-lock mismatch: {key}")
    return payload


def _prepare_exact_p1(source_h5ad: Path, repository_root: Path) -> SciPlex3BaselinePreparation:
    loader = SciPlex3K562H5ADLoader.open_for_purpose(
        source_h5ad,
        repository_root,
        access_purpose=PopulationComponentAccessPurpose.TRAIN_PARAMETERS,
        partition_id="p1-train",
    )
    try:
        return assemble_sciplex3_p1_training_data(
            loader,
            repository_root,
            batch_size=MATERIALIZATION_BATCH_SIZE,
        )
    finally:
        # Item 11 close() reauthenticates the descriptor and exact source bytes after the complete
        # count stream has been consumed.  A close failure therefore fails the whole transaction.
        loader.close()


def _peak_rss_bytes() -> int:
    observed = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if type(observed) not in (int, float) or observed < 0:
        raise CandidateMaterializationError("cannot measure process peak RSS")
    # The authorized fit runtime is Linux, where ru_maxrss is KiB.
    return int(observed * 1024)


def _copy_verified(source: Path, destination: Path) -> bytes:
    payload = _read_bytes(source, name="runner output")
    _write_exclusive(destination, payload)
    return payload


def _artifact_payloads(
    *,
    stage_output: Path,
    stage_support: Mapping[Path, Path],
    stage_count_descriptor: Path,
    output_directory: Path,
    repository_root: Path,
) -> dict[str, tuple[Path, bytes]]:
    by_role: dict[str, tuple[Path, bytes]] = {
        "assembly_receipt": (
            _output_relative_path(output_directory, repository_root, ASSEMBLY_RECEIPT),
            _read_bytes(stage_output / ASSEMBLY_RECEIPT, name="p1 assembly receipt"),
        ),
        "candidate_model": (
            _output_relative_path(output_directory, repository_root, CANDIDATE_MODEL),
            _read_bytes(stage_output / CANDIDATE_MODEL, name="candidate model"),
        ),
        "candidate_training_plan": (
            _output_relative_path(output_directory, repository_root, TRAINING_PLAN),
            _read_bytes(stage_output / TRAINING_PLAN, name="candidate training plan"),
        ),
        "finalized_count_scan_receipt": (
            _output_relative_path(output_directory, repository_root, FINALIZED_SCAN_RECEIPT),
            _read_bytes(stage_output / FINALIZED_SCAN_RECEIPT, name="finalized p1 scan"),
        ),
        "p1_count_stream_descriptor": (
            COUNT_DESCRIPTOR_RELATIVE_PATH,
            _read_bytes(stage_count_descriptor, name="p1 count-stream descriptor"),
        ),
        "training_execution_observation": (
            _output_relative_path(output_directory, repository_root, TRAINING_OBSERVATION),
            _read_bytes(stage_output / TRAINING_OBSERVATION, name="training observation"),
        ),
    }
    support_roles = {
        "candidate-specification.json": "candidate_specification",
        "output-model-schema.json": "candidate_output_model_schema",
        "runtime-lock.json": "candidate_runtime_lock",
    }
    for runner_name, role in support_roles.items():
        canonical_path = SUPPORT_RELATIVE_PATHS[runner_name]
        by_role[role] = (
            canonical_path,
            _read_bytes(stage_support[canonical_path], name=role.replace("_", " ")),
        )
    return by_role


def _build_manifest(
    preparation: SciPlex3BaselinePreparation,
    plan: CandidateTrainingPlan,
    observation: SciPlex3CandidateTrainingObservation,
    artifact_payloads: Mapping[str, tuple[Path, bytes]],
    repository_bindings: Mapping[str, object],
    *,
    support_envelope_fingerprint: str,
) -> dict[str, object]:
    scan = preparation.finalized_count_scan_receipt
    assembly = preparation.receipt
    zero_panel_well_count = sum(
        int(well.counts.indptr[0]) == int(well.counts.indptr[-1])
        for well in preparation.training_data.wells
    )
    if zero_panel_well_count != 0:
        raise CandidateMaterializationError("exact p1 candidate input contains a zero-panel well")
    artifacts = {
        role: _artifact_reference(payload, relative_path=relative_path)
        for role, (relative_path, payload) in sorted(artifact_payloads.items())
    }
    return {
        "artifact_schema": "sciplex3-k562-p1-candidate-materialization",
        "artifact_schema_version": "5.0.0",
        "artifacts": artifacts,
        "exact_bindings": {
            "action_binding_sha256": plan.action_binding_sha256,
            "action_domain_sha256": plan.action_binding_sha256,
            "assembly_fingerprint": assembly.fingerprint,
            "benchmark_fingerprint": plan.benchmark_fingerprint,
            "benchmark_sha256": plan.benchmark_fingerprint,
            "candidate_code_sha256": _binding_sha256(repository_bindings, "candidate_code"),
            "candidate_runner_code_sha256": _binding_sha256(
                repository_bindings, "candidate_runner_code"
            ),
            "candidate_specification_sha256": plan.candidate_specification.sha256,
            "candidate_training_plan_fingerprint": plan.fingerprint,
            "candidate_model_schema_version": SCIPLEX3_CANDIDATE_MODEL_SCHEMA_VERSION,
            "dataset_manifest_sha256": scan.dataset_manifest_sha256,
            "feature_panel_artifact_sha256": assembly.feature_panel_artifact_sha256,
            "finalized_count_scan_fingerprint": scan.fingerprint,
            "item11_runner_code_sha256": _binding_sha256(repository_bindings, "item11_runner_code"),
            "initial_equilibration_sha256": observation.initial_equilibration_sha256,
            "inner_equilibration_trace_sha256": observation.inner_equilibration_trace_sha256,
            "loader_code_sha256": _binding_sha256(repository_bindings, "loader_code"),
            "loader_contract_sha256": assembly.loader_contract_sha256,
            "loader_implementation_sha256": assembly.loader_implementation_sha256,
            "materializer_code_sha256": _binding_sha256(repository_bindings, "materializer_code"),
            "model_artifact_sha256": observation.model_artifact_sha256,
            "ordered_feature_keys_sha256": plan.ordered_feature_keys_sha256,
            "output_model_schema_sha256": plan.output_model_schema.sha256,
            "p1_count_stream_sha256": plan.p1_count_stream_sha256,
            "p1_design_fingerprint": plan.p1_design_fingerprint,
            "query_fingerprint": plan.query_fingerprint,
            "query_sha256": plan.query_fingerprint,
            "runtime_lock_sha256": plan.runtime_lock.sha256,
            "fixed_factor_shape": SCIPLEX3_CANDIDATE_FIXED_FACTOR_SHAPE,
            "training_nuisance_rho_sha256": observation.training_nuisance_rho_sha256,
            "software_golden_model_sha256": SCIPLEX3_CANDIDATE_GOLDEN_MODEL_SHA256,
            "software_golden_sample_sha256": SCIPLEX3_CANDIDATE_GOLDEN_SAMPLE_SHA256,
            "scoring_transform_sha256": assembly.scoring_transform_sha256,
            "support_envelope_fingerprint": support_envelope_fingerprint,
            "target_value_schema_sha256": plan.target_value_schema_sha256,
            "training_observation_fingerprint": observation.fingerprint,
        },
        "p1_scan": {
            "assembly_fingerprint": assembly.fingerprint,
            "batch_count": assembly.batch_count,
            "control_well_count": assembly.control_well_count,
            "count_scan_complete": assembly.count_scan_complete,
            "finalized_count_scan_fingerprint": scan.fingerprint,
            "full_source_umi_total": assembly.full_source_umi_total,
            "panel_count_stream_sha256": assembly.runner_panel_count_stream_sha256,
            "panel_nonzero_count": assembly.panel_nonzero_count,
            "panel_umi_total": assembly.panel_umi_total,
            "record_count": assembly.record_count,
            "treated_well_count": assembly.treated_well_count,
            "well_count": assembly.well_count,
            "zero_panel_record_count": assembly.zero_panel_record_count,
            "zero_panel_well_count": zero_panel_well_count,
        },
        "repository_bindings": dict(repository_bindings),
        "resource_gates": {
            "aggregate_container_memory": {
                "enforced_by": (
                    "Docker cgroup memory.max and memory.swap.max over the complete "
                    "container process tree"
                ),
                "limit_bytes": FIT_RSS_LIMIT_BYTES,
                "terminal_result_artifact": "contained-training-observation.json",
                "terminal_result_recorded_here": False,
            },
            "aggregate_container_wall_clock": {
                "enforced_by": (
                    "parent Docker deadline plus the in-container source-to-posthash watchdog"
                ),
                "limit_seconds": FIT_WALL_LIMIT_SECONDS,
                "terminal_result_artifact": "contained-training-observation.json",
                "terminal_result_recorded_here": False,
            },
        },
        "safety_boundary": dict(_SAFETY_BOUNDARY),
        "scope": {
            "access_purpose": "train_parameters",
            "batch_size": MATERIALIZATION_BATCH_SIZE,
            "candidate_model_id": SCIPLEX3_CANDIDATE_MODEL_ID,
            "candidate_implementation_version": SCIPLEX3_CANDIDATE_IMPLEMENTATION_VERSION,
            "candidate_model_schema": SCIPLEX3_CANDIDATE_MODEL_SCHEMA,
            "candidate_model_schema_version": SCIPLEX3_CANDIDATE_MODEL_SCHEMA_VERSION,
            "capture_latent_present": False,
            "factor_shape_mode": "fixed",
            "feature_count": 2_000,
            "fixed_factor_shape": SCIPLEX3_CANDIDATE_FIXED_FACTOR_SHAPE,
            "optimization_seed": plan.optimization_seed,
            "partition_id": "p1-train",
            "plate_context_family": "neutral-unit-context",
            "plate_sigma_present": False,
        },
        "source": {
            "byte_count": SCIPLEX3_SOURCE_BYTE_COUNT,
            "filename": SCIPLEX3_SOURCE_FILENAME,
            "md5": SCIPLEX3_SOURCE_MD5,
            "sha256": SCIPLEX3_SOURCE_SHA256,
        },
    }


def _resolve_reference(
    reference: Mapping[str, object],
    *,
    repository_root: Path,
    overrides: Mapping[Path, Path] | None,
    name: str,
) -> tuple[Path, bytes]:
    _require_exact_fields(reference, _ARTIFACT_REFERENCE_FIELDS, name=name)
    relative = reference.get("relative_path")
    digest = reference.get("sha256")
    byte_count = reference.get("byte_count")
    if (
        type(relative) is not str
        or not relative
        or not _is_exact_sha256(digest)
        or type(byte_count) is not int
        or byte_count < 0
        or reference.get("media_type") != "application/json"
    ):
        raise CandidateMaterializationError(f"malformed {name} reference")
    relative_path = Path(relative)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise CandidateMaterializationError(f"{name} reference escapes the repository")
    path = (
        overrides.get(relative_path, repository_root / relative_path)
        if overrides
        else (repository_root / relative_path)
    )
    payload = _read_bytes(path, name=name)
    if len(payload) != byte_count or _sha256(payload) != digest:
        raise CandidateMaterializationError(f"content-addressed {name} drifted")
    return path, payload


def _require_false(mapping: Mapping[str, object], names: tuple[str, ...], *, label: str) -> None:
    if any(mapping.get(name) is not False for name in names):
        raise CandidateMaterializationError(f"{label} crossed the Item 12 authority boundary")


def _check_manifest(
    manifest: Mapping[str, object],
    manifest_payload: bytes,
    *,
    repository_root: Path,
    repository_bindings: Mapping[str, object],
    overrides: Mapping[Path, Path] | None = None,
) -> str:
    if (
        manifest.get("artifact_schema") != "sciplex3-k562-p1-candidate-materialization"
        or manifest.get("artifact_schema_version") != "5.0.0"
    ):
        raise CandidateMaterializationError("Item 12 materialization header drifted")
    if set(manifest) != {
        "artifact_schema",
        "artifact_schema_version",
        "artifacts",
        "exact_bindings",
        "p1_scan",
        "repository_bindings",
        "resource_gates",
        "safety_boundary",
        "scope",
        "source",
    }:
        raise CandidateMaterializationError("Item 12 manifest field closure drifted")
    if not _json_values_are_exact(manifest.get("repository_bindings"), dict(repository_bindings)):
        raise CandidateMaterializationError("Item 12 repository-byte closure is stale")
    if not _json_values_are_exact(
        manifest.get("source"),
        {
            "byte_count": SCIPLEX3_SOURCE_BYTE_COUNT,
            "filename": SCIPLEX3_SOURCE_FILENAME,
            "md5": SCIPLEX3_SOURCE_MD5,
            "sha256": SCIPLEX3_SOURCE_SHA256,
        },
    ):
        raise CandidateMaterializationError("Item 12 source identity drifted")
    safety = _as_mapping(manifest.get("safety_boundary"), name="safety boundary")
    if not _json_values_are_exact(dict(safety), dict(_SAFETY_BOUNDARY)):
        raise CandidateMaterializationError("Item 12 safety boundary is not exact")
    resources = _as_mapping(manifest.get("resource_gates"), name="resource gates")
    if not _json_values_are_exact(
        dict(resources),
        {
            "aggregate_container_memory": {
                "enforced_by": (
                    "Docker cgroup memory.max and memory.swap.max over the complete "
                    "container process tree"
                ),
                "limit_bytes": FIT_RSS_LIMIT_BYTES,
                "terminal_result_artifact": "contained-training-observation.json",
                "terminal_result_recorded_here": False,
            },
            "aggregate_container_wall_clock": {
                "enforced_by": (
                    "parent Docker deadline plus the in-container source-to-posthash watchdog"
                ),
                "limit_seconds": FIT_WALL_LIMIT_SECONDS,
                "terminal_result_artifact": "contained-training-observation.json",
                "terminal_result_recorded_here": False,
            },
        },
    ):
        raise CandidateMaterializationError("Item 12 resource gates drifted")

    artifacts = _as_mapping(manifest.get("artifacts"), name="materialized artifacts")
    expected_roles = {
        "assembly_receipt",
        "candidate_model",
        "candidate_output_model_schema",
        "candidate_runtime_lock",
        "candidate_specification",
        "candidate_training_plan",
        "finalized_count_scan_receipt",
        "p1_count_stream_descriptor",
        "training_execution_observation",
    }
    if set(artifacts) != expected_roles:
        raise CandidateMaterializationError("Item 12 artifact closure is not exact")
    expected_relative_paths = {
        "assembly_receipt": BENCHMARK_RELATIVE_DIRECTORY / "item12-p1" / ASSEMBLY_RECEIPT,
        "candidate_model": BENCHMARK_RELATIVE_DIRECTORY / "item12-p1" / CANDIDATE_MODEL,
        "candidate_output_model_schema": SUPPORT_RELATIVE_PATHS["output-model-schema.json"],
        "candidate_runtime_lock": SUPPORT_RELATIVE_PATHS["runtime-lock.json"],
        "candidate_specification": SUPPORT_RELATIVE_PATHS["candidate-specification.json"],
        "candidate_training_plan": BENCHMARK_RELATIVE_DIRECTORY / "item12-p1" / TRAINING_PLAN,
        "finalized_count_scan_receipt": (
            BENCHMARK_RELATIVE_DIRECTORY / "item12-p1" / FINALIZED_SCAN_RECEIPT
        ),
        "p1_count_stream_descriptor": COUNT_DESCRIPTOR_RELATIVE_PATH,
        "training_execution_observation": (
            BENCHMARK_RELATIVE_DIRECTORY / "item12-p1" / TRAINING_OBSERVATION
        ),
    }
    resolved: dict[str, tuple[Path, bytes]] = {}
    for role in sorted(expected_roles):
        reference = _as_mapping(artifacts.get(role), name=f"{role} reference")
        if reference.get("relative_path") != expected_relative_paths[role].as_posix():
            raise CandidateMaterializationError(f"{role} uses a noncanonical repository path")
        resolved[role] = _resolve_reference(
            reference,
            repository_root=repository_root,
            overrides=overrides,
            name=role.replace("_", " "),
        )

    exact = _as_mapping(manifest.get("exact_bindings"), name="exact bindings")
    plan_payload = resolved["candidate_training_plan"][1]
    try:
        plan = CandidateTrainingPlan.model_validate_json(plan_payload)
    except Exception as error:
        raise CandidateMaterializationError("candidate training plan is invalid") from error
    if canonical_json_bytes(plan.model_dump(mode="json")) != plan_payload:
        raise CandidateMaterializationError("candidate training plan is not canonical")
    if (
        plan.training_partition_ids != ("p1-train",)
        or tuple(role.value for role in plan.training_partition_roles) != ("train",)
        or plan.future_calibration_plan is not None
        or plan.optimization_seed != 0
        or plan.deterministic_thread_count != 1
    ):
        raise CandidateMaterializationError("candidate plan is not exact p1-only training")
    role_to_plan_artifact = {
        "candidate_specification": plan.candidate_specification,
        "candidate_output_model_schema": plan.output_model_schema,
        "candidate_runtime_lock": plan.runtime_lock,
        "p1_count_stream_descriptor": plan.p1_count_stream,
    }
    for role, declared in role_to_plan_artifact.items():
        payload = resolved[role][1]
        if declared.sha256 != _sha256(payload) or declared.byte_count != len(payload):
            raise CandidateMaterializationError(f"training plan {role} binding drifted")
    if (
        plan.trainer_implementation.code_artifact.sha256
        != _binding_sha256(repository_bindings, "candidate_runner_code")
        or plan.candidate_factory_implementation.code_artifact.sha256
        != _binding_sha256(repository_bindings, "candidate_code")
        or plan.benchmark_fingerprint != _binding_sha256(repository_bindings, "benchmark")
        or plan.query_fingerprint != _binding_sha256(repository_bindings, "query")
        or plan.action_binding_sha256 != _binding_sha256(repository_bindings, "action_domain")
        or plan.target_value_schema_sha256
        != _binding_sha256(repository_bindings, "target_value_schema")
        or plan.p1_loader_contract.sha256 != _binding_sha256(repository_bindings, "loader_contract")
    ):
        raise CandidateMaterializationError("candidate plan semantic/code binding drifted")

    runtime_lock = _canonical_object(
        resolved["candidate_runtime_lock"][1], name="candidate runtime lock"
    )
    _require_exact_fields(runtime_lock, _RUNTIME_LOCK_FIELDS, name="candidate runtime lock")
    if (
        runtime_lock.get("artifact_schema") != "sciplex3-candidate-runtime-lock"
        or runtime_lock.get("artifact_schema_version") != "1.0.0"
        or not _json_values_are_exact(
            runtime_lock.get("runtime"), dict(SCIPLEX3_CANDIDATE_REFERENCE_RUNTIME)
        )
        or not _json_values_are_exact(
            runtime_lock.get("thread_environment"),
            {key: "1" for key in _THREAD_ENVIRONMENT_KEYS},
        )
    ):
        raise CandidateMaterializationError("candidate runtime lock drifted")

    count_descriptor = _canonical_object(
        resolved["p1_count_stream_descriptor"][1], name="p1 count-stream descriptor"
    )
    _require_exact_fields(
        count_descriptor, _COUNT_DESCRIPTOR_FIELDS, name="p1 count-stream descriptor"
    )
    count_authority = _as_mapping(
        count_descriptor.get("authority"), name="p1 count-stream authority"
    )
    _require_exact_fields(
        count_authority,
        _COUNT_DESCRIPTOR_AUTHORITY_FIELDS,
        name="p1 count-stream authority",
    )
    if (
        not _json_values_are_exact(
            dict(count_authority),
            {
                "can_mint_lifecycle_evidence": False,
                "heldout_memberships_read": False,
                "heldout_outcomes_read": False,
                "scientifically_admissible": False,
            },
        )
        or count_descriptor.get("artifact_schema")
        != "sciplex3-p1-candidate-count-stream-descriptor"
        or count_descriptor.get("artifact_schema_version") != "1.0.0"
        or count_descriptor.get("training_partition_ids") != ["p1-train"]
        or type(count_descriptor.get("count_stream_encoding")) is not str
        or not cast(str, count_descriptor["count_stream_encoding"])
        or any(
            type(count_descriptor.get(field_name)) is not int
            for field_name in (
                "panel_nonzero_count",
                "panel_umi_total",
                "record_count",
                "well_count",
                "zero_panel_record_count",
            )
        )
        or any(
            not _is_exact_sha256(count_descriptor.get(field_name))
            for field_name in (
                "assembly_fingerprint",
                "candidate_design_fingerprint",
                "finalized_count_scan_fingerprint",
                "ordered_feature_keys_sha256",
                "panel_count_stream_sha256",
            )
        )
        or count_descriptor.get("record_count") != 94_785
        or count_descriptor.get("well_count") != 768
        or count_descriptor.get("zero_panel_record_count") != 7
        or count_descriptor.get("panel_count_stream_sha256") != plan.p1_count_stream_sha256
        or count_descriptor.get("candidate_design_fingerprint") != plan.p1_design_fingerprint
        or count_descriptor.get("ordered_feature_keys_sha256") != plan.ordered_feature_keys_sha256
        or count_descriptor.get("assembly_fingerprint") != plan.p1_assembly_fingerprint
        or count_descriptor.get("finalized_count_scan_fingerprint")
        != plan.p1_finalized_count_scan_fingerprint
    ):
        raise CandidateMaterializationError("p1 count-stream descriptor drifted")
    _require_false(
        count_authority,
        (
            "can_mint_lifecycle_evidence",
            "heldout_memberships_read",
            "heldout_outcomes_read",
            "scientifically_admissible",
        ),
        label="p1 count-stream descriptor",
    )

    scan = _canonical_object(resolved["finalized_count_scan_receipt"][1], name="finalized scan")
    assembly = _canonical_object(resolved["assembly_receipt"][1], name="assembly receipt")
    _validated_finalized_scan(scan, name="finalized p1 scan")
    _validated_assembly(assembly, name="p1 assembly")
    if (
        scan.get("artifact_schema") != "sciplex3-k562-p1-finalized-count-scan-receipt"
        or scan.get("artifact_schema_version") != "1.0.0"
        or scan.get("loader_interface_id") != "cellstate.sciplex3-training-data-loader.v1"
        or scan.get("partition_id") != "p1-train"
        or scan.get("access_purpose") != "train_parameters"
        or scan.get("accessed_partition_roles") != ["p1-train"]
        or scan.get("source_sha256") != SCIPLEX3_SOURCE_SHA256
        or scan.get("source_byte_count") != SCIPLEX3_SOURCE_BYTE_COUNT
        or scan.get("source_md5") != SCIPLEX3_SOURCE_MD5
        or scan.get("dataset_manifest_sha256")
        != _binding_sha256(repository_bindings, "dataset_manifest")
        or scan.get("feature_panel_artifact_sha256")
        != _binding_sha256(repository_bindings, "feature_panel")
        or scan.get("ordered_feature_keys_sha256") != plan.ordered_feature_keys_sha256
        or scan.get("p1_loader_contract_sha256")
        != _binding_sha256(repository_bindings, "loader_contract")
        or scan.get("loader_implementation_sha256")
        != _binding_sha256(repository_bindings, "loader_code")
        or scan.get("scoring_transform_sha256")
        != _binding_sha256(repository_bindings, "scoring_transform")
        or scan.get("query_sha256") != plan.query_fingerprint
        or scan.get("benchmark_sha256") != plan.benchmark_fingerprint
        or scan.get("target_value_schema_sha256") != plan.target_value_schema_sha256
        or scan.get("source_descriptor_identity_before")
        != scan.get("source_descriptor_identity_after")
        or scan.get("record_count") != 94_785
        or scan.get("well_count") != 768
        or scan.get("treated_well_count") != 752
        or scan.get("control_well_count") != 16
        or scan.get("zero_panel_record_count") != 7
        or scan.get("accessed_count_datasets") != ["X.data", "X.indices", "X.indptr", "obs.ncounts"]
        or scan.get("count_scan_complete") is not True
        or scan.get("close_reverification_completed") is not True
        or scan.get("source_descriptor_reverified") is not True
        or scan.get("exact_record_coverage") is not True
        or scan.get("finalized") is not True
    ):
        raise CandidateMaterializationError("finalized p1 scan closure drifted")
    _require_false(
        scan,
        (
            "heldout_memberships_parsed",
            "heldout_outcome_values_parsed",
            "trusted_workflow_receipt_present",
            "lifecycle_evidence_issued",
            "scientifically_admissible",
        ),
        label="finalized p1 scan",
    )
    if (
        assembly.get("partition_id") != "p1-train"
        or assembly.get("access_purpose") != "train_parameters"
        or assembly.get("source_sha256") != SCIPLEX3_SOURCE_SHA256
        or assembly.get("loader_source_scan_fingerprint")
        != scan.get("initial_source_authentication_fingerprint")
        or assembly.get("loader_implementation_sha256") != scan.get("loader_implementation_sha256")
        or assembly.get("loader_contract_sha256") != scan.get("p1_loader_contract_sha256")
        or assembly.get("record_count") != scan.get("record_count")
        or assembly.get("well_count") != scan.get("well_count")
        or assembly.get("treated_well_count") != scan.get("treated_well_count")
        or assembly.get("control_well_count") != scan.get("control_well_count")
        or assembly.get("zero_panel_record_count") != scan.get("zero_panel_record_count")
        or assembly.get("record_ids_sha256") != scan.get("record_ids_sha256")
        or assembly.get("record_to_well_sha256") != scan.get("record_to_well_sha256")
        or assembly.get("emitted_source_row_indices_sha256")
        != scan.get("emitted_source_row_indices_sha256")
        or assembly.get("ordered_record_source_well_condition_sha256")
        != scan.get("ordered_record_source_well_condition_sha256")
        or assembly.get("runner_panel_count_stream_sha256") != scan.get("panel_count_stream_sha256")
        or assembly.get("loader_panel_count_stream_sha256") != scan.get("panel_count_stream_sha256")
        or assembly.get("panel_nonzero_count") != scan.get("panel_nonzero_count")
        or assembly.get("panel_umi_total") != scan.get("panel_umi_total")
        or assembly.get("full_source_umi_total") != scan.get("full_source_umi_total")
        or assembly.get("batch_count") != scan.get("batch_count")
        or assembly.get("ordered_feature_keys_sha256") != scan.get("ordered_feature_keys_sha256")
        or assembly.get("feature_panel_artifact_sha256")
        != scan.get("feature_panel_artifact_sha256")
        or assembly.get("action_domain_sha256") != plan.action_binding_sha256
        or assembly.get("query_sha256") != scan.get("query_sha256")
        or assembly.get("benchmark_sha256") != scan.get("benchmark_sha256")
        or assembly.get("scoring_transform_sha256") != scan.get("scoring_transform_sha256")
        or assembly.get("target_value_schema_sha256") != scan.get("target_value_schema_sha256")
        or assembly.get("count_scan_complete") is not True
        or assembly.get("close_reverification_completed") is not True
        or assembly.get("exact_record_coverage") is not True
        or assembly.get("finalized_count_scan_fingerprint")
        != _sha256(resolved["finalized_count_scan_receipt"][1])
    ):
        raise CandidateMaterializationError("p1 assembly closure drifted")
    _require_false(
        assembly,
        (
            "heldout_memberships_read",
            "heldout_outcomes_read",
            "can_mint_lifecycle_evidence",
            "scientifically_admissible",
        ),
        label="p1 assembly",
    )
    if (
        count_descriptor.get("count_stream_encoding") != scan.get("count_stream_encoding")
        or count_descriptor.get("panel_nonzero_count") != scan.get("panel_nonzero_count")
        or count_descriptor.get("panel_umi_total") != scan.get("panel_umi_total")
        or count_descriptor.get("record_count") != scan.get("record_count")
        or count_descriptor.get("well_count") != scan.get("well_count")
        or count_descriptor.get("zero_panel_record_count") != scan.get("zero_panel_record_count")
    ):
        raise CandidateMaterializationError("p1 count-stream receipt projection drifted")
    if (
        plan.p1_finalized_count_scan_fingerprint
        != _sha256(resolved["finalized_count_scan_receipt"][1])
        or plan.p1_assembly_fingerprint != _sha256(resolved["assembly_receipt"][1])
        or plan.p1_count_stream_sha256 != scan.get("panel_count_stream_sha256")
    ):
        raise CandidateMaterializationError("candidate plan differs from exact p1 receipts")
    expected_scan_summary = {
        "assembly_fingerprint": _sha256(resolved["assembly_receipt"][1]),
        "batch_count": assembly.get("batch_count"),
        "control_well_count": 16,
        "count_scan_complete": True,
        "finalized_count_scan_fingerprint": _sha256(resolved["finalized_count_scan_receipt"][1]),
        "full_source_umi_total": assembly.get("full_source_umi_total"),
        "panel_count_stream_sha256": plan.p1_count_stream_sha256,
        "panel_nonzero_count": assembly.get("panel_nonzero_count"),
        "panel_umi_total": assembly.get("panel_umi_total"),
        "record_count": 94_785,
        "treated_well_count": 752,
        "well_count": 768,
        "zero_panel_record_count": 7,
        "zero_panel_well_count": 0,
    }
    if not _json_values_are_exact(manifest.get("p1_scan"), expected_scan_summary):
        raise CandidateMaterializationError("Item 12 p1 scan summary drifted")
    if not _json_values_are_exact(
        manifest.get("scope"),
        {
            "access_purpose": "train_parameters",
            "batch_size": MATERIALIZATION_BATCH_SIZE,
            "candidate_implementation_version": SCIPLEX3_CANDIDATE_IMPLEMENTATION_VERSION,
            "candidate_model_id": SCIPLEX3_CANDIDATE_MODEL_ID,
            "candidate_model_schema": SCIPLEX3_CANDIDATE_MODEL_SCHEMA,
            "candidate_model_schema_version": SCIPLEX3_CANDIDATE_MODEL_SCHEMA_VERSION,
            "capture_latent_present": False,
            "factor_shape_mode": "fixed",
            "feature_count": 2_000,
            "fixed_factor_shape": SCIPLEX3_CANDIDATE_FIXED_FACTOR_SHAPE,
            "optimization_seed": 0,
            "partition_id": "p1-train",
            "plate_context_family": "neutral-unit-context",
            "plate_sigma_present": False,
        },
    ):
        raise CandidateMaterializationError("Item 12 fit scope drifted")

    model_payload = resolved["candidate_model"][1]
    model_sha256 = _sha256(model_payload)
    try:
        candidate = SciPlex3GammaPoissonCandidate.load_exact(
            model_payload, expected_sha256=model_sha256
        )
    except Exception as error:
        raise CandidateMaterializationError("candidate model failed exact reload") from error
    if type(candidate) is not SciPlex3GammaPoissonCandidate:
        raise CandidateMaterializationError("candidate model reload substituted another class")
    if candidate.model_bytes() != model_payload or candidate.model_artifact_sha256 != model_sha256:
        raise CandidateMaterializationError("candidate model bytes changed on reload")
    behavior = candidate.behavior_manifest()
    fitted_state = candidate.fitted_state_manifest()
    tensor_sha256 = fitted_state.get("tensor_sha256")
    if (
        type(tensor_sha256) is not dict
        or type(tensor_sha256.get("rho")) is not str
        or "plate_sigma" in tensor_sha256
        or behavior.get("model_schema_version") != SCIPLEX3_CANDIDATE_MODEL_SCHEMA_VERSION
        or behavior.get("factor_shape_mode") != "fixed"
        or behavior.get("factor_shape_estimated") is not False
        or behavior.get("fixed_factor_shape") != SCIPLEX3_CANDIDATE_FIXED_FACTOR_SHAPE
        or behavior.get("inner_equilibration_performed") is not True
        or behavior.get("inner_all_batches_converged") is not True
        or behavior.get("plate_context_family") != "neutral-unit-context"
        or behavior.get("plate_context_count") != 1
        or behavior.get("plate_context_factorwise_mean_one") is not True
        or behavior.get("capture_latent_present") is not False
        or behavior.get("fit_converged") is not True
        or behavior.get("all_parameters_finite") is not True
        or behavior.get("training_partition_ids") != ["p1-train"]
        or type(behavior.get("outer_iteration_count")) is not int
        or not SCIPLEX3_CANDIDATE_MIN_OUTER_ITERATIONS
        <= cast(int, behavior["outer_iteration_count"])
        <= SCIPLEX3_CANDIDATE_MAX_OUTER_ITERATIONS
    ):
        raise CandidateMaterializationError("candidate model convergence contract drifted")
    _require_false(
        behavior,
        (
            "heldout_memberships_read",
            "heldout_outcomes_read",
            "can_mint_lifecycle_evidence",
            "scientifically_admissible",
        ),
        label="candidate model",
    )
    summary = candidate.training_summary
    if (
        summary.record_count != 94_785
        or summary.well_count != 768
        or summary.zero_panel_record_count != 7
        or summary.design_sha256 != plan.p1_design_fingerprint
        or summary.provenance != "real-p1"
    ):
        raise CandidateMaterializationError("candidate model training summary drifted")
    initial = candidate.initial_equilibration
    total_inner_sweep_count = sum(
        (index + 1) * count for index, count in enumerate(initial.inner_sweep_count_histogram)
    ) + sum(
        (index + 1) * count
        for item in candidate.trace
        for index, count in enumerate(item.inner_sweep_count_histogram)
    )

    observation_payload = resolved["training_execution_observation"][1]
    observation_value = _canonical_object(observation_payload, name="training observation")
    if set(observation_value) != set(SciPlex3CandidateTrainingObservation.__dataclass_fields__):
        raise CandidateMaterializationError("training observation field closure drifted")
    observation_python = dict(observation_value)
    partition_ids = observation_python.get("training_partition_ids")
    if type(partition_ids) is list:
        observation_python["training_partition_ids"] = tuple(partition_ids)
    for field_name in (
        "initial_factor_order",
        "initial_inner_sweep_count_histogram",
        "sampling_envelope_rejection_reasons",
        "terminal_elbo_relative_changes",
    ):
        values = observation_python.get(field_name)
        if type(values) is list:
            observation_python[field_name] = tuple(values)
    try:
        observation = SciPlex3CandidateTrainingObservation(**cast(Any, observation_python))
    except Exception as error:
        raise CandidateMaterializationError("training observation is invalid") from error
    if (
        observation.plan_fingerprint != plan.fingerprint
        or observation.model_artifact_sha256 != model_sha256
        or observation.model_artifact_byte_count != len(model_payload)
        or observation.finalized_count_scan_fingerprint
        != _sha256(resolved["finalized_count_scan_receipt"][1])
        or observation.assembly_fingerprint != _sha256(resolved["assembly_receipt"][1])
        or observation.p1_count_stream_sha256 != plan.p1_count_stream_sha256
        or observation.candidate_specification_sha256 != plan.candidate_specification.sha256
        or observation.output_model_schema_sha256 != plan.output_model_schema.sha256
        or observation.runtime_lock_sha256 != plan.runtime_lock.sha256
        or observation.training_nuisance_rho_sha256 != tensor_sha256["rho"]
        or observation.initial_equilibration_sha256
        != fitted_state.get("initial_equilibration_sha256")
        or observation.inner_equilibration_trace_sha256
        != fitted_state.get("inner_equilibration_trace_sha256")
        or observation.outer_iteration_count != behavior.get("outer_iteration_count")
        or observation.initial_elbo != initial.elbo
        or observation.initial_factor_order != initial.factor_order
        or observation.initial_inner_sweep_count_histogram != initial.inner_sweep_count_histogram
        or observation.initial_maximum_inner_sweeps != initial.maximum_inner_sweeps
        or observation.initial_maximum_terminal_shape_residual
        != initial.maximum_terminal_shape_residual
        or observation.initial_maximum_terminal_elog_residual
        != initial.maximum_terminal_elog_residual
        or observation.final_elbo != behavior.get("final_elbo")
        or observation.fixed_factor_shape != behavior.get("fixed_factor_shape")
        or observation.inner_batch_count != behavior.get("inner_batch_count")
        or observation.total_inner_sweep_count != total_inner_sweep_count
        or observation.maximum_inner_sweeps != behavior.get("maximum_inner_sweeps")
        or observation.maximum_terminal_shape_residual
        != behavior.get("maximum_terminal_shape_residual")
        or observation.maximum_terminal_elog_residual
        != behavior.get("maximum_terminal_elog_residual")
        or observation.loading_rank_ratio != behavior.get("loading_rank_ratio")
        or observation.mean_activation_rank_ratio != behavior.get("mean_activation_rank_ratio")
        or observation.minimum_factor_contribution_share
        != behavior.get("minimum_factor_contribution_share")
        or list(observation.terminal_elbo_relative_changes)
        != behavior.get("terminal_elbo_relative_changes")
        or observation.candidate_model_schema_version != behavior.get("model_schema_version")
        or observation.factor_order_stable is not behavior.get("factor_order_stable")
        or observation.factor_shape_mode != behavior.get("factor_shape_mode")
        or observation.factor_shape_estimated is not behavior.get("factor_shape_estimated")
        or observation.inner_equilibration_performed
        is not behavior.get("inner_equilibration_performed")
        or observation.inner_all_batches_converged
        is not behavior.get("inner_all_batches_converged")
        or observation.plate_context_family != behavior.get("plate_context_family")
        or observation.plate_context_count != behavior.get("plate_context_count")
        or observation.plate_context_factorwise_mean_one
        is not behavior.get("plate_context_factorwise_mean_one")
        or observation.capture_latent_present is not behavior.get("capture_latent_present")
        or observation.plate_sigma_present is not False
    ):
        raise CandidateMaterializationError("training observation binding drifted")
    import cellstate.evaluation.sciplex3_candidate_runner as runner

    golden_request_sha256, golden_sample_sha256 = runner._sample_identity(candidate)
    if (
        observation.behavior_sha256 != _sha256(canonical_json_bytes(behavior))
        or observation.fitted_state_sha256 != _sha256(canonical_json_bytes(fitted_state))
        or observation.golden_request_sha256 != golden_request_sha256
        or observation.golden_sample_sha256 != golden_sample_sha256
    ):
        raise CandidateMaterializationError(
            "training observation behavior, state, or golden sample drifted"
        )

    expected_exact = {
        "action_binding_sha256": plan.action_binding_sha256,
        "action_domain_sha256": plan.action_binding_sha256,
        "assembly_fingerprint": _sha256(resolved["assembly_receipt"][1]),
        "benchmark_fingerprint": plan.benchmark_fingerprint,
        "benchmark_sha256": plan.benchmark_fingerprint,
        "candidate_code_sha256": _binding_sha256(repository_bindings, "candidate_code"),
        "candidate_runner_code_sha256": _binding_sha256(
            repository_bindings, "candidate_runner_code"
        ),
        "candidate_model_schema_version": SCIPLEX3_CANDIDATE_MODEL_SCHEMA_VERSION,
        "candidate_specification_sha256": plan.candidate_specification.sha256,
        "candidate_training_plan_fingerprint": plan.fingerprint,
        "dataset_manifest_sha256": _binding_sha256(repository_bindings, "dataset_manifest"),
        "feature_panel_artifact_sha256": _binding_sha256(repository_bindings, "feature_panel"),
        "finalized_count_scan_fingerprint": _sha256(resolved["finalized_count_scan_receipt"][1]),
        "fixed_factor_shape": SCIPLEX3_CANDIDATE_FIXED_FACTOR_SHAPE,
        "initial_equilibration_sha256": observation.initial_equilibration_sha256,
        "inner_equilibration_trace_sha256": observation.inner_equilibration_trace_sha256,
        "item11_runner_code_sha256": _binding_sha256(repository_bindings, "item11_runner_code"),
        "loader_code_sha256": _binding_sha256(repository_bindings, "loader_code"),
        "loader_contract_sha256": _binding_sha256(repository_bindings, "loader_contract"),
        "loader_implementation_sha256": _binding_sha256(repository_bindings, "loader_code"),
        "materializer_code_sha256": _binding_sha256(repository_bindings, "materializer_code"),
        "model_artifact_sha256": model_sha256,
        "ordered_feature_keys_sha256": plan.ordered_feature_keys_sha256,
        "output_model_schema_sha256": plan.output_model_schema.sha256,
        "p1_count_stream_sha256": plan.p1_count_stream_sha256,
        "p1_design_fingerprint": plan.p1_design_fingerprint,
        "training_nuisance_rho_sha256": observation.training_nuisance_rho_sha256,
        "query_fingerprint": plan.query_fingerprint,
        "query_sha256": plan.query_fingerprint,
        "runtime_lock_sha256": plan.runtime_lock.sha256,
        "software_golden_model_sha256": SCIPLEX3_CANDIDATE_GOLDEN_MODEL_SHA256,
        "software_golden_sample_sha256": SCIPLEX3_CANDIDATE_GOLDEN_SAMPLE_SHA256,
        "scoring_transform_sha256": _binding_sha256(repository_bindings, "scoring_transform"),
        "support_envelope_fingerprint": plan.support_envelope_fingerprint,
        "target_value_schema_sha256": plan.target_value_schema_sha256,
        "training_observation_fingerprint": observation.fingerprint,
    }
    if not _json_values_are_exact(dict(exact), expected_exact):
        raise CandidateMaterializationError("Item 12 exact bindings drifted")
    return _sha256(manifest_payload)


def check_materialization_inputs(
    output_directory: Path = DEFAULT_OUTPUT_DIRECTORY,
    *,
    repository_root: Path = REPOSITORY_ROOT,
    sealed_support_directory: Path | None = None,
    count_stream_descriptor_path: Path | None = None,
) -> str:
    """Reauthenticate canonical or staged Item 12 bytes without touching the source H5AD."""

    root = Path(repository_root).resolve()
    output = Path(output_directory).resolve()
    manifest, payload = _load_canonical_object(
        output / MATERIALIZATION_MANIFEST,
        name="Item 12 materialization manifest",
    )
    bindings = _repository_bindings(root)
    canonical_output = BENCHMARK_RELATIVE_DIRECTORY / "item12-p1"
    overrides = {
        canonical_output / filename: output / filename
        for filename in (
            ASSEMBLY_RECEIPT,
            CANDIDATE_MODEL,
            FINALIZED_SCAN_RECEIPT,
            TRAINING_OBSERVATION,
            TRAINING_PLAN,
        )
    }
    if sealed_support_directory is not None:
        support_directory = Path(sealed_support_directory).resolve()
        overrides.update(
            {
                relative_path: support_directory / runner_name
                for runner_name, relative_path in SUPPORT_RELATIVE_PATHS.items()
            }
        )
    if count_stream_descriptor_path is not None:
        overrides[COUNT_DESCRIPTOR_RELATIVE_PATH] = Path(count_stream_descriptor_path).resolve()
    return _check_manifest(
        manifest,
        payload,
        repository_root=root,
        repository_bindings=bindings,
        overrides=overrides,
    )


def check_materialization(
    output_directory: Path = DEFAULT_OUTPUT_DIRECTORY,
    *,
    repository_root: Path = REPOSITORY_ROOT,
) -> str:
    """Reauthenticate checked-in Item 12 bytes without resolving or touching the source H5AD."""

    return check_materialization_inputs(output_directory, repository_root=repository_root)


def _install_exclusive_file(source: Path, target: Path) -> None:
    payload = _read_bytes(source, name="staged transaction artifact")
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        os.link(source, target)
    except OSError as error:
        raise CandidateMaterializationError(
            f"cannot install exclusive artifact: {target}"
        ) from error
    try:
        if _read_bytes(target, name="installed transaction artifact") != payload:
            raise CandidateMaterializationError(
                f"installed artifact differs from staging: {target}"
            )
        _fsync_directory(target.parent)
    except BaseException:
        with suppress(FileNotFoundError):
            target.unlink()
        raise


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(path, flags)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError as error:
        raise CandidateMaterializationError(
            f"cannot durably synchronize transaction directory: {path}"
        ) from error


def _remove_empty_parents(path: Path, *, stop: Path) -> None:
    parent = path.parent
    while parent != stop and parent.is_relative_to(stop):
        try:
            parent.rmdir()
        except OSError:
            break
        parent = parent.parent


def materialize(
    source_h5ad: Path,
    output_directory: Path = DEFAULT_OUTPUT_DIRECTORY,
    *,
    repository_root: Path = REPOSITORY_ROOT,
) -> str:
    """Reject the retired in-process source-touching/publication path before source access."""

    del source_h5ad, output_directory, repository_root
    raise CandidateMaterializationError(
        "legacy direct materialization is retired; use the contained v5 supervisor"
    )


def _retired_materialize_implementation(
    source_h5ad: Path,
    output_directory: Path = DEFAULT_OUTPUT_DIRECTORY,
    *,
    repository_root: Path = REPOSITORY_ROOT,
) -> str:
    """Fail closed; retained only as a compatibility name for retirement assertions."""

    del source_h5ad, output_directory, repository_root
    raise CandidateMaterializationError(
        "legacy direct materialization is retired; use the contained v5 supervisor"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--source-h5ad", type=Path)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    args = parser.parse_args()
    if args.check:
        if args.source_h5ad is not None:
            parser.error("--check never accepts or opens --source-h5ad")
        fingerprint = check_materialization(args.output_directory)
        print(f"item12_p1_candidate_materialization_sha256 {fingerprint}")
        return
    if args.source_h5ad is None:
        parser.error("--source-h5ad is required unless --check is used")
    fingerprint = materialize(args.source_h5ad, args.output_directory)
    print(f"item12_p1_candidate_materialization_sha256 {fingerprint}")


if __name__ == "__main__":
    main()
