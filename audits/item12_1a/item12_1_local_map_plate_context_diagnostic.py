#!/usr/bin/env python3
"""Item 12.1b harness for characterizing the exact v4 first-failing local map.

Freezing and source-free testing this executable is the software-only Item 12.1a milestone.  It
does not execute or authorize Item 12.1b; a one-use exact-p1 replay remains separately gated.

This one-shot scratch driver authenticates the frozen software, runtime, Item 11 p1 receipts,
loader contract, and source identity before the source H5AD is resolved.  It then reconstructs the
exact p1 preparation and calls only the runner's private in-memory wrapper around the frozen exact
candidate fitter.  It never creates a training plan, serializes model bytes, samples, emits an
observation, invokes a post-fit behavior/fitted-state API, or writes persistent data.  Before any
source operation, the exact h5py runtime is imported under a one-use allowance for CPython's
``open('/dev/null', None, O_CLOEXEC | O_RDWR)`` containment descriptor.  The allowance is closed
before the loader runs and cannot authorize any other write-capable open.

The frozen fitter itself retains its built-in final validation, but the expected initial inner
failure occurs before that surface.  On that one exact failure this driver copies only the current
batch's strictly necessary numeric map inputs, clears every retained traceback and cause, and then
replays the frozen local map from the bit-exact 0.1 shape.  It emits bounded indices, summaries,
digests, objectives, invariants, local Jacobian spectral radii, and empirical plate-context
concentration diagnostics.  It never emits source rows, feature indices, count arrays, identifiers,
model state or bytes, a plan, an observation, a sample, an artifact, or lifecycle authority.
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib
import io
import json
import math
import os
import signal
import stat
import sys
import time
import traceback as traceback_module
from collections.abc import Mapping, Sequence
from importlib.machinery import ModuleSpec
from importlib.util import find_spec
from itertools import pairwise
from pathlib import Path
from types import ModuleType
from typing import Any, Final, cast

sys.dont_write_bytecode = True

DRIVER_SCHEMA: Final = "sciplex3-item12.1b-local-map-plate-context-replay-harness"
DRIVER_SCHEMA_VERSION: Final = "1.0.0"
ATTEMPT2_DRIVER_SHA256_ORACLE: Final = (
    "795c59296f5cefb1b6dd78a021ea0eb8e795217eda5226becf6c5bf909f6623a"
)
ATTEMPT2_REPORT_SHA256_ORACLE: Final = (
    "66e9debc1a402e7aa68cbc934f7c5f641529eea3187ec15606364c912af8faa8"
)

EXPECTED_ATTEMPT2_TERMINAL_SHAPE_RESIDUAL_HEX: Final = "0x1.fa26f9b70f4e4p-3"
EXPECTED_ATTEMPT2_TERMINAL_ELOG_RESIDUAL_HEX: Final = "0x1.e00ca4aa71e08p+1"
EXPECTED_ATTEMPT2_THETA_SHAPE_SHA256: Final = (
    "2d04db3cb407c623153164c6ce8cd9c1c9012ba138cc8e91a1fb9716defd77b2"
)
EXPECTED_ATTEMPT2_THETA_RATE_SHA256: Final = (
    "244f9c16dd9c6465c7807f828bd7688a486e9d7337c35d4910658f7a39f5e331"
)
EXPECTED_ATTEMPT2_LOADING_CONCENTRATION_SHA256: Final = (
    "7764a0b0c6f365330ca86fbfbfc6c65aae8fb849436b18104ad2121959d6936d"
)
EXPECTED_ATTEMPT2_WELL_FACTOR_MEANS_SHA256: Final = (
    "d200f9cf8730838f688fbf7afc212cecccf721045fee706d9dfc54e60755e974"
)
EXPECTED_ATTEMPT2_RHO_SHA256: Final = (
    "390b4375bd74f1ca6e47407340228055b58b1eb628b26aa3c2eb1db85ec66565"
)
TERMINAL_JACOBIAN_STATE_INDICES: Final = (49, 50)

EXPECTED_H5PY_VERSION: Final = "3.16.0"
EXPECTED_HDF5_VERSION: Final = "2.0.0"
EXPECTED_RUNTIME_PREFIX: Final = "/opt/runtime"
EXPECTED_H5PY_PACKAGE_ROOT: Final = "/opt/runtime/lib/python3.11/site-packages/h5py"
EXPECTED_H5PY_PACKAGE_ORIGIN: Final = "/opt/runtime/lib/python3.11/site-packages/h5py/__init__.py"
EXPECTED_H5PY_ORIGIN_MODULE_COUNT: Final = 40
EXPECTED_H5PY_ORIGIN_MANIFEST_SHA256: Final = (
    "af7e2cbfb651bcaa5e80e589cf20534e7328a81d88369a467363c4e8b2444bfd"
)
EXPECTED_DEVNULL_PATH: Final = "/dev/null"
EXPECTED_DEVNULL_DEVICE_MAJOR: Final = 1
EXPECTED_DEVNULL_DEVICE_MINOR: Final = 3
EXPECTED_DEVNULL_RW_OPEN_FLAGS: Final = 524_290

EXPECTED_CANDIDATE_SHA256: Final = (
    "f316edbc4f3204686d2d9d7a0a7fbc1d809dcac61601416f7f02323dece152b8"
)
EXPECTED_RUNNER_SHA256: Final = "7d8ae937d1188979b461a94f39f7a9bddc3c7e793d1c4ce00134722b81a928c4"
EXPECTED_BUILDER_SHA256: Final = "6768fe97b9bec75f56aa6556a6ad6946c7967b0892dc5d18b07510f8c46c2361"
EXPECTED_MATERIALIZER_SHA256: Final = (
    "d60844733b486fb298eecdbabde4149c2ebacc3251e039198ffc8dd68300c91d"
)
EXPECTED_LOADER_SHA256: Final = "bed3b56f7a91f1bb60f799ea2e28dc31505f196579cb8cb4ff386df5364a979d"
EXPECTED_ITEM11_RUNNER_SHA256: Final = (
    "2896ff33121f7059fb9e6811940b977fdeae254aa8d905af0ea43cbb991e8023"
)
EXPECTED_ITEM11_MATERIALIZER_SHA256: Final = (
    "392bb85368a9fd9842c869aeb9330dc0c3b8a43683f76f9f631a49d3ec7e95a5"
)
EXPECTED_ITEM11_MATERIALIZATION_SHA256: Final = (
    "7dd28d3ddca5d09d81779bfc3e02ec15d09428be354f6972e1ceda20ee1dd0e6"
)
EXPECTED_ITEM11_ASSEMBLY_SHA256: Final = (
    "e9d54381e8edb7cd9922d2a58d026830686bb0c6247146f71c08b81418831e0c"
)
EXPECTED_ITEM11_SCAN_SHA256: Final = (
    "3213ac3da3547db1cbdcb5506a340192048349c9cd880060c45b2a88dc6226cf"
)
EXPECTED_LOADER_CONTRACT_SHA256: Final = (
    "3de5be54b60ba1403995ba79d122ee8232218be5c027da1bf530cb610ae80f90"
)
EXPECTED_SOURCE_VERIFICATION_SHA256: Final = (
    "84d871ebc56a3343bf1277e7a6e7c9cc6dec9f28ba8e93fd4404a4c9b5e7fda7"
)
EXPECTED_DATASET_MANIFEST_SHA256: Final = (
    "6248e63237a4c0c7ae53538666a1294cf1108569792eb54702ec15f439d9cb31"
)
EXPECTED_SPECIFICATION_SHA256: Final = (
    "a783d22a799e0136cfe779cae301b05e35f3f0f3c1cf004e94423a74bbb9bd7a"
)
EXPECTED_OUTPUT_SCHEMA_SHA256: Final = (
    "523dcaceb6de519db8966ac3a6514732446984fa46c4d47716c96727f05e2653"
)
EXPECTED_GOLDEN_MODEL_SHA256: Final = (
    "d3a5cb630ad4344bda04945a865682531860508ccb8473fa57fb2397c8297be1"
)
EXPECTED_GOLDEN_SAMPLE_SHA256: Final = (
    "6e01b44440c04ccc0ae1540de57d26c3723ff0b12f7e8033d4c980818776fa9f"
)
EXPECTED_RUNTIME_LOCK_SHA256: Final = (
    "407752a74dcc33eef41b233804b218254d53015ff7b1a7cfe79af3dcc60cf28d"
)
EXPECTED_PANEL_COUNT_STREAM_SHA256: Final = (
    "d55d0c62cd761d329415c61357b571845512d6dd14b455191848ebac00389fe6"
)
EXPECTED_SOURCE_SHA256: Final = "603ed16c5e25401c8a7f5bb0b2b045179701017d65dcfc6aeea71722a66cd10a"
EXPECTED_SOURCE_MD5: Final = "c9e70629505d98c7ca1a837f62b14e89"
EXPECTED_SOURCE_BYTE_COUNT: Final = 2_526_631_614
EXPECTED_SOURCE_FILENAME: Final = "SrivatsanTrapnell2020_sciplex3.h5ad"

EXPECTED_MODEL_ID: Final = "sciplex3-k562-24h-gamma-poisson-fixed-r0p1-empirical-plate-k16"
EXPECTED_VERSION: Final = "4.0.0"
EXPECTED_MODEL_SCHEMA: Final = (
    "sciplex3-gamma-poisson-fixed-r0p1-empirical-plate-candidate-model-v4"
)
EXPECTED_FIXED_FACTOR_SHAPE_HEX: Final = "0x1.999999999999ap-4"
EXPECTED_FIXED_FACTOR_SHAPE: Final = float.fromhex(EXPECTED_FIXED_FACTOR_SHAPE_HEX)

FIT_WALL_LIMIT_SECONDS: Final = 3_600
FIT_RSS_LIMIT_BYTES: Final = 4 * 1024**3
THREAD_ENVIRONMENT_KEYS: Final = (
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)

REPOSITORY_PATHS: Final[Mapping[str, str]] = {
    "builder": "scripts/build_sciplex3_k562_trained_candidate.py",
    "candidate": "src/cellstate/evaluation/sciplex3_candidate.py",
    "candidate_runner": "src/cellstate/evaluation/sciplex3_candidate_runner.py",
    "dataset_manifest": "data_manifests/reviewed/sciplex3-k562-24h.json",
    "item11_assembly": (
        "benchmarks/vertical-a/sciplex3-k562-24h-v1/item11-p1/p1-assembly-receipt.json"
    ),
    "item11_materialization": (
        "benchmarks/vertical-a/sciplex3-k562-24h-v1/item11-p1/materialization-manifest.json"
    ),
    "item11_materializer": "scripts/materialize_sciplex3_k562_p1_baselines.py",
    "item11_runner": "src/cellstate/evaluation/sciplex3_runner.py",
    "item11_scan": (
        "benchmarks/vertical-a/sciplex3-k562-24h-v1/item11-p1/p1-finalized-count-scan-receipt.json"
    ),
    "loader": "src/cellstate/backends/sciplex3_loader.py",
    "loader_contract": (
        "benchmarks/vertical-a/sciplex3-k562-24h-v1/support/p1-loader-contract.json"
    ),
    "materializer": "scripts/materialize_sciplex3_k562_p1_candidate.py",
    "source_verification": ("benchmarks/artifacts/sciplex3-k562-24h-v1/source-verification.json"),
}

EXPECTED_REPOSITORY_SHA256: Final[Mapping[str, str]] = {
    "builder": EXPECTED_BUILDER_SHA256,
    "candidate": EXPECTED_CANDIDATE_SHA256,
    "candidate_runner": EXPECTED_RUNNER_SHA256,
    "dataset_manifest": EXPECTED_DATASET_MANIFEST_SHA256,
    "item11_assembly": EXPECTED_ITEM11_ASSEMBLY_SHA256,
    "item11_materialization": EXPECTED_ITEM11_MATERIALIZATION_SHA256,
    "item11_materializer": EXPECTED_ITEM11_MATERIALIZER_SHA256,
    "item11_runner": EXPECTED_ITEM11_RUNNER_SHA256,
    "item11_scan": EXPECTED_ITEM11_SCAN_SHA256,
    "loader": EXPECTED_LOADER_SHA256,
    "loader_contract": EXPECTED_LOADER_CONTRACT_SHA256,
    "materializer": EXPECTED_MATERIALIZER_SHA256,
    "source_verification": EXPECTED_SOURCE_VERIFICATION_SHA256,
}

EXPECTED_SCIENTIFIC_FAILURE_MESSAGES: Final[Mapping[str, str]] = {
    "candidate CAVI failed to converge within 50 outer passes": "outer-nonconvergence",
    "candidate inner CAVI failed to converge within 50 sweeps": "inner-nonconvergence",
    "candidate tracked ELBO materially decreased": "material-elbo-decrease",
    "candidate contains duplicate canonical factor keys": "canonical-factor-key-degeneracy",
    "candidate factor contribution shares fail the frozen identifiability gate": (
        "factor-contribution-identifiability"
    ),
    "candidate loading matrix fails the frozen rounded identifiability threshold": (
        "loading-rank-identifiability"
    ),
    "candidate mean activation matrix fails the frozen rounded identifiability threshold": (
        "mean-activation-rank-identifiability"
    ),
    "candidate loading matrix rank ratio lies on an ambiguous quantization boundary": (
        "loading-rank-portability-boundary"
    ),
    "candidate mean activation matrix rank ratio lies on an ambiguous quantization boundary": (
        "mean-activation-rank-portability-boundary"
    ),
}

ASSEMBLY_RUNTIME_DEPENDENT_FIELDS: Final = frozenset(
    {
        "finalized_count_scan_fingerprint",
    }
)
SCAN_RUNTIME_OR_FILESYSTEM_DEPENDENT_FIELDS: Final = frozenset(
    {
        "h5py_version",
        "hdf5_version",
        "numpy_version",
        "python_implementation",
        "python_version",
        "source_descriptor_identity_after",
        "source_descriptor_identity_before",
    }
)

HELDOUT_PATH_MARKERS: Final = (
    "memberships/calibration-",
    "memberships/model_selection_validation-",
    "memberships/untouched_test-",
    "memberships/universe-",
    "benchmarks/artifacts/sciplex3-k562-24h-v1/k562-universe.json",
    "benchmarks/artifacts/sciplex3-k562-24h-v1/partitions.json",
    "support/evaluation-cases.json",
    "support/treated-well-to-matched-controls.json",
    "support/well-groups.json",
    "support/contexts/plate25.json",
    "support/contexts/plate26.json",
    "support/contexts/plate27.json",
    "support/contexts/plate28.json",
    "support/contexts/plate29.json",
    "support/contexts/plate30.json",
    "support/contexts/plate31.json",
    "support/contexts/plate32.json",
)
MUTATING_AUDIT_EVENTS: Final = frozenset(
    {
        "os.chmod",
        "os.chown",
        "os.link",
        "os.mkdir",
        "os.remove",
        "os.rename",
        "os.rmdir",
        "os.symlink",
        "os.truncate",
    }
)


class TrajectoryError(RuntimeError):
    """Raised when the diagnostic contract or frozen closure differs."""


class FitWallLimitError(RuntimeError):
    """Raised by the hard one-shot fit wall timer."""


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    return _sha256(path.read_bytes())


def _exception_identity(error: BaseException) -> dict[str, str]:
    error_type = type(error)
    return {
        "message": str(error),
        "type": f"{error_type.__module__}.{error_type.__qualname__}",
    }


def _validate_exact_json_tree(value: object, *, path: str = "$") -> None:
    value_type = type(value)
    if value is None or value_type in (bool, int, str):
        return
    if value_type is float:
        if not math.isfinite(cast(float, value)):
            raise TrajectoryError(f"nonfinite JSON float at {path}")
        return
    if value_type is list:
        for index, item in enumerate(cast(list[object], value)):
            _validate_exact_json_tree(item, path=f"{path}[{index}]")
        return
    if value_type is dict:
        for key, item in cast(dict[object, object], value).items():
            if type(key) is not str:
                raise TrajectoryError(f"non-exact JSON object key at {path}")
            _validate_exact_json_tree(item, path=f"{path}.{key}")
        return
    raise TrajectoryError(
        f"non-exact JSON value type at {path}: {value_type.__module__}.{value_type.__qualname__}"
    )


def _canonical_json_bytes(value: object) -> bytes:
    _validate_exact_json_tree(value)
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()


def _canonical_json_object(payload: bytes, *, name: str) -> dict[str, object]:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise TrajectoryError(f"{name} is not valid JSON") from error
    if type(value) is not dict or any(type(key) is not str for key in value):
        raise TrajectoryError(f"{name} is not one exact JSON object")
    result = cast(dict[str, object], value)
    if _canonical_json_bytes(result) != payload:
        raise TrajectoryError(f"{name} is not canonical JSON")
    return result


def _float_manifest(value: float) -> dict[str, object]:
    if type(value) is not float or not math.isfinite(value):
        raise TrajectoryError("bounded float manifest requires one finite exact float")
    return {"hex": value.hex(), "value": value}


def _binding_digest(bindings: Mapping[str, object], name: str) -> str:
    raw = bindings.get(name)
    if not isinstance(raw, Mapping):
        raise TrajectoryError(f"repository binding is missing: {name}")
    digest = raw.get("sha256")
    if type(digest) is not str or len(digest) != 64:
        raise TrajectoryError(f"repository binding has no exact SHA-256: {name}")
    return digest


def _wall_limit_handler(_signum: int, _frame: object) -> None:
    raise FitWallLimitError("v4 candidate fit exceeded the hard 3600-second wall limit")


def _audit_event_forbidden(event: str, arguments: tuple[object, ...]) -> tuple[bool, str | None]:
    if event in MUTATING_AUDIT_EVENTS:
        return True, "filesystem-mutation-event"
    if event != "open" or not arguments:
        return False, None
    raw_path = arguments[0]
    normalized_path: str | None = None
    if isinstance(raw_path, (str, bytes, os.PathLike)):
        try:
            normalized_path = os.fsdecode(os.fspath(raw_path)).replace("\\", "/")
        except (TypeError, ValueError):
            normalized_path = None
    if normalized_path is not None and any(
        marker in normalized_path for marker in HELDOUT_PATH_MARKERS
    ):
        return True, "heldout-path-read"
    mode = arguments[1] if len(arguments) > 1 else None
    flags = arguments[2] if len(arguments) > 2 else None
    if type(mode) is str and any(character in mode for character in "wax+"):
        return True, "file-write-mode"
    write_flags = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND
    if type(flags) is int and flags & write_flags:
        return True, "file-write-flags"
    return False, None


def _new_audit_state() -> dict[str, object]:
    return {
        "allowed_devnull_rw_open_count": 0,
        "devnull_allowance_active": False,
        "devnull_allowance_closed_before_source": False,
        "devnull_character_device_identity_pass": False,
        "heldout_path_attempt_count": 0,
        "read_open_event_count": 0,
        "write_attempt_count": 0,
    }


def _exact_devnull_character_device_identity_pass() -> bool:
    try:
        observed = os.lstat(EXPECTED_DEVNULL_PATH)
        return bool(
            stat.S_ISCHR(observed.st_mode)
            and os.major(observed.st_rdev) == EXPECTED_DEVNULL_DEVICE_MAJOR
            and os.minor(observed.st_rdev) == EXPECTED_DEVNULL_DEVICE_MINOR
        )
    except (AttributeError, OSError, TypeError, ValueError):
        return False


def _is_exact_allowed_devnull_rw_open(event: str, arguments: tuple[object, ...]) -> bool:
    return bool(
        event == "open"
        and type(arguments) is tuple
        and len(arguments) == 3
        and type(arguments[0]) is str
        and arguments[0] == EXPECTED_DEVNULL_PATH
        and arguments[1] is None
        and type(arguments[2]) is int
        and arguments[2] == EXPECTED_DEVNULL_RW_OPEN_FLAGS
    )


def _activate_exact_devnull_import_allowance(state: dict[str, object]) -> None:
    if (
        type(state.get("devnull_allowance_active")) is not bool
        or type(state.get("devnull_allowance_closed_before_source")) is not bool
        or type(state.get("devnull_character_device_identity_pass")) is not bool
        or type(state.get("allowed_devnull_rw_open_count")) is not int
    ):
        raise TrajectoryError("diagnostic filesystem audit state is malformed")
    if (
        state["devnull_allowance_active"] is True
        or state["devnull_allowance_closed_before_source"] is True
        or state["allowed_devnull_rw_open_count"] != 0
    ):
        raise TrajectoryError("exact /dev/null import allowance is not fresh")
    runtime_flags = getattr(os, "O_CLOEXEC", None)
    if (
        type(runtime_flags) is not int
        or runtime_flags | os.O_RDWR != EXPECTED_DEVNULL_RW_OPEN_FLAGS
    ):
        raise TrajectoryError("runtime /dev/null import flags differ from the sealed Linux tuple")
    identity_pass = _exact_devnull_character_device_identity_pass()
    state["devnull_character_device_identity_pass"] = identity_pass
    if not identity_pass:
        raise TrajectoryError("/dev/null is not the exact Linux character device 1:3")
    state["devnull_allowance_active"] = True


def _close_exact_devnull_import_allowance(state: dict[str, object]) -> None:
    if state.get("devnull_allowance_active") is not True:
        raise TrajectoryError("exact /dev/null import allowance is not active")
    state["devnull_allowance_active"] = False
    if state.get("allowed_devnull_rw_open_count") != 1:
        raise TrajectoryError("exact h5py import did not consume one /dev/null allowance")
    state["devnull_allowance_closed_before_source"] = True


def _observe_diagnostic_audit_event(
    state: dict[str, object],
    event: str,
    arguments: tuple[object, ...],
) -> None:
    if (
        state.get("devnull_allowance_active") is True
        and state.get("devnull_allowance_closed_before_source") is False
        and state.get("devnull_character_device_identity_pass") is True
        and state.get("allowed_devnull_rw_open_count") == 0
        and _is_exact_allowed_devnull_rw_open(event, arguments)
    ):
        state["allowed_devnull_rw_open_count"] = 1
        return
    forbidden, reason = _audit_event_forbidden(event, arguments)
    if event == "open" and not forbidden:
        state["read_open_event_count"] = cast(int, state["read_open_event_count"]) + 1
    if not forbidden:
        return
    if reason == "heldout-path-read":
        state["heldout_path_attempt_count"] = cast(int, state["heldout_path_attempt_count"]) + 1
    else:
        state["write_attempt_count"] = cast(int, state["write_attempt_count"]) + 1
    raise TrajectoryError(f"forbidden diagnostic filesystem operation: {reason}")


def _install_no_write_or_heldout_audit() -> dict[str, object]:
    state = _new_audit_state()

    def audit(event: str, arguments: tuple[object, ...]) -> None:
        _observe_diagnostic_audit_event(state, event, arguments)

    sys.addaudithook(audit)
    return state


def _invalid_isolation_manifest(
    audit_state: Mapping[str, object] | None,
) -> dict[str, object]:
    observed: dict[str, object] = {}
    for name in (
        "heldout_path_attempt_count",
        "read_open_event_count",
        "write_attempt_count",
    ):
        raw = None if audit_state is None else audit_state.get(name)
        observed[name] = raw if type(raw) is int and raw >= 0 else None
    return {
        "allowed_devnull_rw_open_count": (
            audit_state.get("allowed_devnull_rw_open_count")
            if audit_state is not None
            and type(audit_state.get("allowed_devnull_rw_open_count")) is int
            else None
        ),
        "audit_hook_installed": audit_state is not None,
        "claim_status": "not-established-on-invalid-path",
        "devnull_allowance_closed_before_source": (
            audit_state.get("devnull_allowance_closed_before_source")
            if audit_state is not None
            and type(audit_state.get("devnull_allowance_closed_before_source")) is bool
            else None
        ),
        "devnull_character_device_identity_pass": (
            audit_state.get("devnull_character_device_identity_pass")
            if audit_state is not None
            and type(audit_state.get("devnull_character_device_identity_pass")) is bool
            else None
        ),
        "observed_audit_counters": observed,
        "prohibited_surface_status": "not-observed-to-completion",
    }


def _loaded_h5py_module_names() -> tuple[str, ...]:
    return tuple(
        sorted(
            name
            for name in sys.modules
            if type(name) is str and (name == "h5py" or name.startswith("h5py."))
        )
    )


def _require_exact_runtime_prefix() -> str:
    runtime_prefix = sys.prefix
    if type(runtime_prefix) is not str or runtime_prefix != EXPECTED_RUNTIME_PREFIX:
        raise TrajectoryError("Python runtime prefix differs from the sealed reference runtime")
    return runtime_prefix


def _sealed_h5py_package_paths() -> tuple[Path, Path]:
    package_root = Path(EXPECTED_H5PY_PACKAGE_ROOT)
    package_origin = Path(EXPECTED_H5PY_PACKAGE_ORIGIN)
    try:
        resolved_root = package_root.resolve(strict=True)
        resolved_origin = package_origin.resolve(strict=True)
        root_mode = resolved_root.stat().st_mode
        origin_mode = resolved_origin.stat().st_mode
    except (OSError, RuntimeError) as error:
        raise TrajectoryError("sealed h5py package paths are not real runtime paths") from error
    if (
        resolved_root != package_root
        or not stat.S_ISDIR(root_mode)
        or resolved_origin != package_origin
        or not stat.S_ISREG(origin_mode)
        or resolved_origin.parent != resolved_root
    ):
        raise TrajectoryError("sealed h5py package paths are not exact regular package paths")
    return resolved_root, resolved_origin


def _exact_h5py_spec_identity(specification: object) -> dict[str, object]:
    _require_exact_runtime_prefix()
    package_root, package_origin = _sealed_h5py_package_paths()
    if type(specification) is not ModuleSpec:
        raise TrajectoryError("h5py import specification has the wrong exact type")
    raw_origin = specification.origin
    search_locations = specification.submodule_search_locations
    if type(raw_origin) is not str or raw_origin != EXPECTED_H5PY_PACKAGE_ORIGIN:
        raise TrajectoryError("h5py import specification origin differs from the sealed runtime")
    if (
        type(search_locations) is not list
        or len(search_locations) != 1
        or type(search_locations[0]) is not str
        or search_locations[0] != EXPECTED_H5PY_PACKAGE_ROOT
    ):
        raise TrajectoryError(
            "h5py import specification package path differs from the sealed runtime"
        )
    try:
        resolved_origin = Path(raw_origin).resolve(strict=True)
        resolved_search_root = Path(search_locations[0]).resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise TrajectoryError("h5py import specification paths are not real") from error
    if resolved_origin != package_origin or resolved_search_root != package_root:
        raise TrajectoryError(
            "h5py import specification paths do not resolve to the sealed runtime"
        )
    return {
        "h5py_preimport_spec_origin": raw_origin,
        "h5py_preimport_spec_search_locations": list(search_locations),
        "runtime_prefix": sys.prefix,
    }


def _h5py_origin_manifest(module: ModuleType) -> dict[str, object]:
    package_root, package_origin = _sealed_h5py_package_paths()
    raw_module_origin = getattr(module, "__file__", None)
    raw_module_path = getattr(module, "__path__", None)
    if type(raw_module_origin) is not str or raw_module_origin != EXPECTED_H5PY_PACKAGE_ORIGIN:
        raise TrajectoryError("h5py package origin differs from the sealed runtime")
    if (
        type(raw_module_path) is not list
        or len(raw_module_path) != 1
        or type(raw_module_path[0]) is not str
        or raw_module_path[0] != EXPECTED_H5PY_PACKAGE_ROOT
    ):
        raise TrajectoryError("h5py package path differs from the sealed runtime")
    try:
        resolved_module_origin = Path(raw_module_origin).resolve(strict=True)
        resolved_module_root = Path(raw_module_path[0]).resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise TrajectoryError("h5py package paths are not real") from error
    if resolved_module_origin != package_origin or resolved_module_root != package_root:
        raise TrajectoryError("h5py package paths do not resolve to the sealed runtime")

    specification_identity = _exact_h5py_spec_identity(getattr(module, "__spec__", None))
    manifest: dict[str, str] = {}
    for name in _loaded_h5py_module_names():
        loaded_module = sys.modules.get(name)
        if type(loaded_module) is not ModuleType:
            raise TrajectoryError(f"loaded h5py module has the wrong exact type: {name}")
        raw_file = getattr(loaded_module, "__file__", None)
        if type(raw_file) is not str:
            raise TrajectoryError(f"loaded h5py module has no exact string origin: {name}")
        raw_path = Path(raw_file)
        try:
            resolved_file = raw_path.resolve(strict=True)
            file_mode = resolved_file.stat().st_mode
        except (OSError, RuntimeError) as error:
            raise TrajectoryError(f"loaded h5py module origin is not real: {name}") from error
        if not raw_path.is_absolute() or raw_file != str(resolved_file):
            raise TrajectoryError(
                f"loaded h5py module origin is not an exact resolved path: {name}"
            )
        try:
            relative_file = resolved_file.relative_to(package_root)
        except ValueError as error:
            raise TrajectoryError(
                f"loaded h5py module origin escapes the package root: {name}"
            ) from error
        if (
            not stat.S_ISREG(file_mode)
            or not relative_file.parts
            or relative_file.suffix not in {".py", ".so"}
        ):
            raise TrajectoryError(
                f"loaded h5py module origin is not one regular .py/.so file: {name}"
            )
        manifest[name] = relative_file.as_posix()
    manifest = dict(sorted(manifest.items()))
    manifest_sha256 = _sha256(_canonical_json_bytes(manifest))
    if (
        len(manifest) != EXPECTED_H5PY_ORIGIN_MODULE_COUNT
        or manifest_sha256 != EXPECTED_H5PY_ORIGIN_MANIFEST_SHA256
    ):
        raise TrajectoryError("loaded h5py origin manifest differs from the sealed runtime")
    return {
        **specification_identity,
        "h5py_origin_manifest": manifest,
        "h5py_origin_manifest_module_count": len(manifest),
        "h5py_origin_manifest_sha256": manifest_sha256,
        "h5py_package_origin": raw_module_origin,
        "h5py_package_path": list(raw_module_path),
    }


def _exact_h5py_runtime_identity(module: object) -> dict[str, object]:
    if type(module) is not ModuleType:
        raise TrajectoryError("h5py import did not return one exact module")
    h5py_version = getattr(module, "__version__", None)
    version_surface = getattr(module, "version", None)
    hdf5_version = getattr(version_surface, "hdf5_version", None)
    if type(h5py_version) is not str or h5py_version != EXPECTED_H5PY_VERSION:
        raise TrajectoryError("h5py version differs from the sealed reference runtime")
    if type(hdf5_version) is not str or hdf5_version != EXPECTED_HDF5_VERSION:
        raise TrajectoryError("HDF5 version differs from the sealed reference runtime")
    return {
        **_h5py_origin_manifest(module),
        "h5py_version": h5py_version,
        "hdf5_version": hdf5_version,
    }


def _preimport_exact_h5py(
    audit_state: dict[str, object],
) -> tuple[ModuleType, dict[str, object]]:
    if _loaded_h5py_module_names():
        raise TrajectoryError("h5py must be initially absent before its one-use import allowance")
    _require_exact_runtime_prefix()
    specification_identity = _exact_h5py_spec_identity(find_spec("h5py"))
    _activate_exact_devnull_import_allowance(audit_state)
    module: object | None = None
    try:
        module = importlib.import_module("h5py")
    finally:
        _close_exact_devnull_import_allowance(audit_state)
    identity = _exact_h5py_runtime_identity(module)
    if sys.modules.get("h5py") is not module:
        raise TrajectoryError("h5py import is not bound to its exact cached module")
    return cast(ModuleType, module), {
        **specification_identity,
        **identity,
        "h5py_initially_absent": True,
    }


def _require_cached_exact_h5py(
    module: ModuleType,
    materializer: ModuleType,
    audit_state: Mapping[str, object],
) -> dict[str, object]:
    if (
        audit_state.get("devnull_allowance_active") is not False
        or audit_state.get("devnull_allowance_closed_before_source") is not True
        or audit_state.get("devnull_character_device_identity_pass") is not True
        or audit_state.get("allowed_devnull_rw_open_count") != 1
    ):
        raise TrajectoryError("h5py cache check found an open or invalid /dev/null allowance")
    if sys.modules.get("h5py") is not module:
        raise TrajectoryError("cached h5py module identity changed")
    identity_before_loader_check = _exact_h5py_runtime_identity(module)
    loader_class = getattr(materializer, "SciPlex3K562H5ADLoader", None)
    loader_import = getattr(loader_class, "_import_h5py", None)
    if not callable(loader_import) or loader_import() is not module:
        raise TrajectoryError("the frozen loader does not reuse the exact cached h5py module")
    identity_after_loader_check = _exact_h5py_runtime_identity(module)
    if identity_after_loader_check != identity_before_loader_check:
        raise TrajectoryError("h5py runtime identity changed during the loader cache check")
    return {
        **identity_after_loader_check,
        "h5py_cached_module_identity_pass": True,
    }


class _PoisonSet:
    """Reversible exact-identity sentinels for prohibited in-fit surfaces."""

    def __init__(self) -> None:
        self._bindings: list[tuple[object, str, object]] = []
        self.counts: dict[str, int] = {}
        self._restored = False

    def add(self, owner: object, name: str, *, label: str) -> None:
        namespace = vars(owner)
        if name not in namespace:
            raise TrajectoryError(f"cannot poison missing prohibited surface: {label}")
        original = namespace[name]
        self.counts[label] = 0

        def blocked(*_args: object, **_kwargs: object) -> object:
            self.counts[label] += 1
            raise TrajectoryError(f"prohibited nonissuing surface invoked: {label}")

        if isinstance(original, classmethod):
            replacement: object = classmethod(blocked)
        elif isinstance(original, staticmethod):
            replacement = staticmethod(blocked)
        else:
            replacement = blocked
        self._bindings.append((owner, name, original))
        setattr(owner, name, replacement)

    def restore(self) -> None:
        if self._restored:
            return
        for owner, name, original in reversed(self._bindings):
            setattr(owner, name, original)
        self._restored = True

    def identities_restored(self) -> bool:
        return self._restored and all(
            vars(owner).get(name) is original for owner, name, original in self._bindings
        )

    def manifest(self) -> dict[str, object]:
        return {
            "all_call_counts_zero": all(count == 0 for count in self.counts.values()),
            "bindings_restored_by_identity": self.identities_restored(),
            "call_counts": dict(sorted(self.counts.items())),
        }


def _install_nonissuance_poisons(
    materializer: ModuleType,
    candidate: ModuleType,
    runner: ModuleType,
) -> _PoisonSet:
    poisons = _PoisonSet()
    candidate_class = candidate.SciPlex3GammaPoissonCandidate
    for name in (
        "_payload",
        "_sample_validated",
        "_tensor_manifests",
        "canonical_model_bytes",
        "fitted_state_manifest",
        "from_canonical_model_bytes",
        "golden_sample",
        "load_exact",
        "model_bytes",
        "sample",
    ):
        poisons.add(candidate_class, name, label=f"candidate.{name}")
    for name in (
        "_load_exact_candidate",
        "_make_observation",
        "_sample_identity",
        "_validate_candidate_state",
        "build_sciplex3_candidate_training_plan",
        "fit_and_write_sciplex3_candidate",
        "seal_sciplex3_candidate_training_plan",
    ):
        poisons.add(runner, name, label=f"candidate_runner.{name}")
    poisons.add(materializer, "materialize", label="candidate_materializer.materialize")
    poisons.add(
        runner._item11,
        "fit_and_write_sciplex3_baseline",
        label="item11_runner.fit_and_write_sciplex3_baseline",
    )
    poisons.add(
        runner._item11,
        "open_sciplex3_p4_prediction_design",
        label="item11_runner.open_sciplex3_p4_prediction_design",
    )
    poisons.add(
        materializer.SciPlex3K562H5ADLoader,
        "open_for_purpose",
        label="loader.second_open_for_purpose",
    )
    builder = sys.modules.get("_cellstate_item12_builder")
    if builder is not None:
        for name in ("build_trained_candidate", "emit_trained_candidate_build"):
            if name in vars(builder):
                poisons.add(builder, name, label=f"trained_candidate_builder.{name}")
    return poisons


def _snapshot_output_targets(repository_root: Path, materializer: ModuleType) -> dict[str, object]:
    targets = {
        "count_stream_descriptor": repository_root / materializer.COUNT_DESCRIPTOR_RELATIVE_PATH,
        "item12_output_directory": Path(materializer.DEFAULT_OUTPUT_DIRECTORY),
        **{
            f"sealed_support_{name}": repository_root / path
            for name, path in materializer.SUPPORT_RELATIVE_PATHS.items()
        },
    }
    result: dict[str, object] = {}
    for name, path in sorted(targets.items()):
        exists = path.exists()
        result[name] = {
            "exists": exists,
            "sha256": _file_sha256(path) if exists and path.is_file() else None,
        }
    return result


def _tensor_summary(value: object, np: ModuleType, *, integer: bool = False) -> dict[str, object]:
    dtype = "<i8" if integer else "<f8"
    array = np.asarray(value, dtype=dtype, order="C")
    finite = True if integer else bool(np.all(np.isfinite(array)))
    return {
        "all_finite": finite,
        "byte_count": int(array.nbytes),
        "maximum": int(np.max(array))
        if integer and array.size
        else (float(np.max(array)) if finite and array.size else None),
        "minimum": int(np.min(array))
        if integer and array.size
        else (float(np.min(array)) if finite and array.size else None),
        "sha256": _sha256(array.tobytes(order="C")),
        "shape": [int(item) for item in array.shape],
    }


def _canonical_audit_matrix(value: object, np: ModuleType, *, decimals: int) -> object:
    rounded = np.asarray(np.round(value, decimals), dtype="<f8", order="C")
    return np.frombuffer(rounded.tobytes(order="C"), dtype="<f8").reshape(rounded.shape)


def _rank_diagnostics(
    value: object,
    np: ModuleType,
    *,
    name: str,
    decimals: int,
    threshold: float,
    margin: float,
    boundary_epsilon_multiplier: float,
) -> dict[str, object]:
    canonical = _canonical_audit_matrix(value, np, decimals=decimals)
    singular_values = np.linalg.svd(canonical, compute_uv=False)
    spectrum_valid = bool(
        singular_values.shape == (min(canonical.shape),)
        and np.all(np.isfinite(singular_values))
        and singular_values[0] > 0.0
        and singular_values[-1] >= 0.0
    )
    if not spectrum_valid:
        return {
            "error": f"{name} singular spectrum is invalid",
            "gate_pass": False,
            "spectrum_valid": False,
        }
    raw_ratio = float(singular_values[-1] / singular_values[0])
    quantized = float(np.round(raw_ratio, decimals))
    half_quantum = 0.5 * 10.0**-decimals
    boundary_tolerance = (
        boundary_epsilon_multiplier * float(np.finfo(np.float64).eps) * max(1.0, abs(raw_ratio))
    )
    lower_distance = abs(raw_ratio - (quantized - half_quantum))
    upper_distance = abs(raw_ratio - (quantized + half_quantum))
    boundary_unambiguous = min(lower_distance, upper_distance) > boundary_tolerance
    gate_pass = bool(
        boundary_unambiguous and math.isfinite(quantized) and quantized > threshold + margin
    )
    return {
        "boundary_tolerance": _float_manifest(float(boundary_tolerance)),
        "gate_pass": gate_pass,
        "quantization_boundary_unambiguous": boundary_unambiguous,
        "quantized_ratio": _float_manifest(quantized),
        "raw_ratio": _float_manifest(raw_ratio),
        "scientific_gate": _float_manifest(float(threshold + margin)),
        "singular_maximum": _float_manifest(float(singular_values[0])),
        "singular_minimum": _float_manifest(float(singular_values[-1])),
        "spectrum_sha256": _sha256(
            np.asarray(singular_values, dtype="<f8", order="C").tobytes(order="C")
        ),
        "spectrum_valid": True,
        "to_lower_half_boundary": _float_manifest(float(lower_distance)),
        "to_upper_half_boundary": _float_manifest(float(upper_distance)),
    }


def _independent_contributions(mean_activation: object, candidate: ModuleType) -> object:
    np = candidate.np
    canonical = _canonical_audit_matrix(
        mean_activation,
        np,
        decimals=candidate.SCIPLEX3_CANDIDATE_FACTOR_ORDER_DECIMALS,
    )
    return np.asarray(
        [
            math.fsum(
                float(canonical[row_index, factor_index])
                for row_index in range(candidate.SCIPLEX3_CANDIDATE_TRAINING_WELL_COUNT)
            )
            / candidate.SCIPLEX3_CANDIDATE_TRAINING_WELL_COUNT
            for factor_index in range(candidate.SCIPLEX3_CANDIDATE_FACTOR_COUNT)
        ],
        dtype=np.float64,
    )


def _factor_key_diagnostics(
    basis: object,
    contributions: object,
    candidate: ModuleType,
) -> tuple[dict[str, object], tuple[int, ...]]:
    np = candidate.np
    canonical_basis = _canonical_audit_matrix(
        basis,
        np,
        decimals=candidate.SCIPLEX3_CANDIDATE_FACTOR_ORDER_DECIMALS,
    )
    contribution_array = np.asarray(contributions, dtype=np.float64)
    rounded = np.round(
        contribution_array,
        candidate.SCIPLEX3_CANDIDATE_FACTOR_ORDER_DECIMALS,
    )
    row_digests = tuple(
        _sha256(np.asarray(canonical_basis[index], dtype="<f8", order="C").tobytes(order="C"))
        for index in range(candidate.SCIPLEX3_CANDIDATE_FACTOR_COUNT)
    )
    keys = tuple(
        (-float(rounded[index]), row_digests[index])
        for index in range(candidate.SCIPLEX3_CANDIDATE_FACTOR_COUNT)
    )
    unique = len(set(keys)) == candidate.SCIPLEX3_CANDIDATE_FACTOR_COUNT
    order = tuple(sorted(range(candidate.SCIPLEX3_CANDIDATE_FACTOR_COUNT), key=keys.__getitem__))
    sorted_primary = sorted(float(item) for item in rounded)
    primary_gaps = [right - left for left, right in pairwise(sorted_primary)]
    return (
        {
            "basis_row_digest_sequence_sha256": _sha256(_canonical_json_bytes(list(row_digests))),
            "canonical_key_sequence_sha256": _sha256(
                _canonical_json_bytes([[negative, digest] for negative, digest in keys])
            ),
            "canonical_order_sha256": _sha256(_canonical_json_bytes(list(order))),
            "duplicate_key_count": len(keys) - len(set(keys)),
            "duplicate_rounded_primary_count": len(rounded)
            - len({float(item) for item in rounded}),
            "keys_unique": unique,
            "minimum_sorted_primary_gap": _float_manifest(
                float(min(primary_gaps)) if primary_gaps else 0.0
            ),
            "order": [int(item) for item in order],
        },
        order,
    )


def _contribution_diagnostics(contributions: object, candidate: ModuleType) -> dict[str, object]:
    np = candidate.np
    values = np.asarray(contributions, dtype=np.float64)
    total = math.fsum(float(item) for item in values)
    finite_positive = bool(np.all(np.isfinite(values)) and np.all(values > 0.0) and total > 0.0)
    shares = values / total if finite_positive else np.full_like(values, np.nan)
    share_sum = math.fsum(float(item) for item in shares) if finite_positive else math.nan
    minimum_share = float(np.min(shares)) if finite_positive else math.nan
    share_sum_pass = bool(
        finite_positive
        and math.isclose(
            share_sum,
            1.0,
            rel_tol=0.0,
            abs_tol=8.0 * float(np.finfo(np.float64).eps),
        )
    )
    gate_pass = bool(
        share_sum_pass and minimum_share > candidate.SCIPLEX3_CANDIDATE_IDENTIFIABILITY_THRESHOLD
    )
    return {
        "all_finite_and_positive": finite_positive,
        "contribution_sha256": _sha256(values.astype("<f8").tobytes(order="C")),
        "gate_pass": gate_pass,
        "maximum_contribution": (
            _float_manifest(float(np.max(values))) if finite_positive else None
        ),
        "minimum_contribution": (
            _float_manifest(float(np.min(values))) if finite_positive else None
        ),
        "minimum_share": _float_manifest(minimum_share) if finite_positive else None,
        "share_sha256": (
            _sha256(np.asarray(shares, dtype="<f8", order="C").tobytes(order="C"))
            if finite_positive
            else None
        ),
        "share_sum": _float_manifest(float(share_sum)) if finite_positive else None,
        "share_sum_pass": share_sum_pass,
    }


def _rho_diagnostics(rho: object, candidate: ModuleType) -> dict[str, object]:
    np = candidate.np
    values = np.asarray(rho, dtype=np.float64)
    shape_pass = values.shape == (
        candidate.SCIPLEX3_CANDIDATE_PLATE_COUNT,
        candidate.SCIPLEX3_CANDIDATE_FACTOR_COUNT,
    )
    finite_bounded = bool(
        shape_pass
        and np.all(np.isfinite(values))
        and np.all(values > 0.0)
        and np.all(values < candidate.SCIPLEX3_CANDIDATE_PLATE_COUNT)
    )
    if finite_bounded:
        means = np.asarray(
            [
                math.fsum(
                    float(values[plate_index, factor_index])
                    for plate_index in range(candidate.SCIPLEX3_CANDIDATE_PLATE_COUNT)
                )
                / candidate.SCIPLEX3_CANDIDATE_PLATE_COUNT
                for factor_index in range(candidate.SCIPLEX3_CANDIDATE_FACTOR_COUNT)
            ],
            dtype=np.float64,
        )
        maximum_error = float(np.max(np.abs(means - 1.0)))
    else:
        means = np.asarray([], dtype=np.float64)
        maximum_error = math.inf
    mean_pass = bool(finite_bounded and maximum_error <= 5e-13)
    return {
        "all_finite_strictly_between_zero_and_eight": finite_bounded,
        "factorwise_mean_maximum_absolute_error": (
            _float_manifest(maximum_error) if math.isfinite(maximum_error) else None
        ),
        "factorwise_mean_one_pass": mean_pass,
        "factorwise_mean_sha256": (
            _sha256(means.astype("<f8").tobytes(order="C")) if finite_bounded else None
        ),
        "rho_sha256": (_sha256(values.astype("<f8").tobytes(order="C")) if shape_pass else None),
        "shape_pass": shape_pass,
    }


def _reconstruct_mean_activation(
    alpha: object,
    rho: object,
    delta: object,
    training_well_plate_indices: object,
    action_well_indices: object,
    vehicle_well_indices: object,
    candidate: ModuleType,
) -> object:
    np = candidate.np
    alpha_array = np.asarray(alpha, dtype=np.float64)
    rho_array = np.asarray(rho, dtype=np.float64)
    delta_array = np.asarray(delta, dtype=np.float64)
    plate_indices = np.asarray(training_well_plate_indices, dtype=np.int64)
    action_indices = np.asarray(action_well_indices, dtype=np.int64)
    vehicle_indices = np.asarray(vehicle_well_indices, dtype=np.int64)
    result = np.empty(
        (
            candidate.SCIPLEX3_CANDIDATE_TRAINING_WELL_COUNT,
            candidate.SCIPLEX3_CANDIDATE_FACTOR_COUNT,
        ),
        dtype=np.float64,
    )
    base = np.exp(alpha_array)[None, :] * rho_array
    for plate_index in range(candidate.SCIPLEX3_CANDIDATE_PLATE_COUNT):
        result[vehicle_indices[plate_index]] = base[plate_index]
    for compound_index in range(candidate.SCIPLEX3_CANDIDATE_COMPOUND_COUNT):
        for dose_index in range(len(candidate.SCIPLEX3_CANDIDATE_DOSES_NM)):
            well_index = int(action_indices[compound_index, dose_index])
            plate_index = int(plate_indices[well_index])
            result[well_index] = base[plate_index] * np.exp(delta_array[compound_index, dose_index])
    if not bool(np.all(np.isfinite(result))) or bool(np.any(result <= 0.0)):
        raise TrajectoryError("private parameters reconstruct a nonpositive mean activation")
    return result


def _copy_inner_witness(value: object, expected_type: type[object]) -> dict[str, object]:
    if type(value) is not expected_type:
        raise TrajectoryError("inner-equilibration witness has a substituted exact type")
    return {
        "elbo": float(value.elbo),
        "factor_order": [int(item) for item in value.factor_order],
        "inner_sweep_count_histogram": [int(item) for item in value.inner_sweep_count_histogram],
        "maximum_inner_sweeps": int(value.maximum_inner_sweeps),
        "maximum_terminal_elog_residual": float(value.maximum_terminal_elog_residual),
        "maximum_terminal_shape_residual": float(value.maximum_terminal_shape_residual),
    }


def _copy_trace(raw_trace: object, trace_type: type[object]) -> list[dict[str, object]]:
    if type(raw_trace) not in (list, tuple):
        raise TrajectoryError("exact fitter has no list/tuple convergence trace")
    entries: list[dict[str, object]] = []
    for item in cast(Sequence[object], raw_trace):
        if type(item) is not trace_type:
            raise TrajectoryError("exact fitter trace contains a substituted entry type")
        entries.append(
            {
                "elbo": float(item.elbo),
                "factor_order": [int(value) for value in item.factor_order],
                "inner_sweep_count_histogram": [
                    int(value) for value in item.inner_sweep_count_histogram
                ],
                "iteration": int(item.iteration),
                "maximum_inner_sweeps": int(item.maximum_inner_sweeps),
                "maximum_terminal_elog_residual": float(item.maximum_terminal_elog_residual),
                "maximum_terminal_shape_residual": float(item.maximum_terminal_shape_residual),
                "relative_change": float(item.relative_change),
            }
        )
    return entries


def _inner_witness_pass(
    witness: Mapping[str, object],
    *,
    expected_batch_count: int,
    candidate: ModuleType,
) -> bool:
    factor_order = witness.get("factor_order")
    if (
        type(factor_order) is not list
        or len(factor_order) != candidate.SCIPLEX3_CANDIDATE_FACTOR_COUNT
        or any(type(item) is not int for item in factor_order)
        or set(cast(list[int], factor_order))
        != set(range(candidate.SCIPLEX3_CANDIDATE_FACTOR_COUNT))
    ):
        return False
    histogram = witness.get("inner_sweep_count_histogram")
    if (
        type(histogram) is not list
        or len(histogram) != candidate.SCIPLEX3_CANDIDATE_MAX_INNER_SWEEPS
        or any(type(item) is not int or item < 0 for item in histogram)
        or histogram[0] != 0
        or sum(cast(list[int], histogram)) != expected_batch_count
    ):
        return False
    maximum = witness.get("maximum_inner_sweeps")
    if (
        type(maximum) is not int
        or not candidate.SCIPLEX3_CANDIDATE_MIN_INNER_SWEEPS
        <= maximum
        <= candidate.SCIPLEX3_CANDIDATE_MAX_INNER_SWEEPS
    ):
        return False
    occupied = [index + 1 for index, count in enumerate(cast(list[int], histogram)) if count]
    if not occupied or max(occupied) != maximum:
        return False
    for name in ("maximum_terminal_shape_residual", "maximum_terminal_elog_residual"):
        value = witness.get(name)
        if (
            type(value) is not float
            or not math.isfinite(value)
            or not 0.0 <= value <= candidate.SCIPLEX3_CANDIDATE_INNER_RESIDUAL_TOL
        ):
            return False
    return True


def _trace_diagnostics(
    initial: Mapping[str, object] | None,
    trace: list[dict[str, object]],
    *,
    expected_batch_count: int,
    candidate: ModuleType,
) -> dict[str, object]:
    initial_pass = bool(
        initial is not None
        and type(initial.get("elbo")) is float
        and math.isfinite(cast(float, initial["elbo"]))
        and _inner_witness_pass(
            initial,
            expected_batch_count=expected_batch_count,
            candidate=candidate,
        )
    )
    order_universe = set(range(candidate.SCIPLEX3_CANDIDATE_FACTOR_COUNT))
    iteration_sequence_pass = [entry["iteration"] for entry in trace] == list(
        range(1, len(trace) + 1)
    )
    order_permutation_pass = all(
        type(entry["factor_order"]) is list
        and len(cast(list[object], entry["factor_order"]))
        == candidate.SCIPLEX3_CANDIDATE_FACTOR_COUNT
        and all(type(item) is int for item in cast(list[object], entry["factor_order"]))
        and set(cast(list[int], entry["factor_order"])) == order_universe
        for entry in trace
    )
    all_inner_pass = initial_pass and all(
        _inner_witness_pass(
            entry,
            expected_batch_count=expected_batch_count,
            candidate=candidate,
        )
        for entry in trace
    )
    recurrence_pass = initial_pass
    decrease_pass = initial_pass
    previous = None if initial is None else cast(float, initial["elbo"])
    convergence_windows: list[int] = []
    for index, entry in enumerate(trace):
        current = cast(float, entry["elbo"])
        if previous is None:
            recurrence_pass = False
            decrease_pass = False
            break
        expected_relative = abs(current - previous) / max(1.0, abs(previous))
        recurrence_pass = recurrence_pass and entry["relative_change"] == expected_relative
        decrease_pass = decrease_pass and (
            current - previous
            >= -candidate.SCIPLEX3_CANDIDATE_ELBO_DECREASE_RTOL * max(1.0, abs(previous))
        )
        previous = current
        iteration = index + 1
        if iteration >= candidate.SCIPLEX3_CANDIDATE_MIN_OUTER_ITERATIONS:
            terminal = trace[
                index + 1 - candidate.SCIPLEX3_CANDIDATE_CONVERGENCE_STREAK : index + 1
            ]
            if (
                len(terminal) == candidate.SCIPLEX3_CANDIDATE_CONVERGENCE_STREAK
                and all(
                    cast(float, item["relative_change"])
                    <= candidate.SCIPLEX3_CANDIDATE_CONVERGENCE_RTOL
                    for item in terminal
                )
                and len({tuple(cast(list[int], item["factor_order"])) for item in terminal}) == 1
            ):
                convergence_windows.append(iteration)
    maximum_inner_sweeps = max(
        ([cast(int, initial["maximum_inner_sweeps"])] if initial is not None else [0])
        + [cast(int, item["maximum_inner_sweeps"]) for item in trace]
    )
    maximum_shape_residual = max(
        (
            [cast(float, initial["maximum_terminal_shape_residual"])]
            if initial is not None
            else [0.0]
        )
        + [cast(float, item["maximum_terminal_shape_residual"]) for item in trace]
    )
    maximum_elog_residual = max(
        ([cast(float, initial["maximum_terminal_elog_residual"])] if initial is not None else [0.0])
        + [cast(float, item["maximum_terminal_elog_residual"]) for item in trace]
    )
    return {
        "all_inner_witnesses_pass": all_inner_pass,
        "all_orders_are_permutations": order_permutation_pass,
        "decrease_gate_pass": decrease_pass,
        "earliest_combined_convergence_iteration": (
            min(convergence_windows) if convergence_windows else None
        ),
        "initial_witness_pass": initial_pass,
        "iteration_sequence_pass": iteration_sequence_pass,
        "maximum_inner_sweeps": maximum_inner_sweeps,
        "maximum_terminal_elog_residual": _float_manifest(maximum_elog_residual),
        "maximum_terminal_shape_residual": _float_manifest(maximum_shape_residual),
        "recurrence_pass": recurrence_pass,
        "terminal_combined_convergence_pass": bool(
            convergence_windows and convergence_windows[-1] == len(trace)
        ),
        "trace_entry_count": len(trace),
        "trace_sha256": _sha256(_canonical_json_bytes(trace)),
    }


def _success_validation_pass(
    failure: Mapping[str, object] | None,
    initial: Mapping[str, object] | None,
    trace_diagnostics: Mapping[str, object],
    final_aggregates: Mapping[str, object],
) -> bool:
    return bool(
        failure is None
        and initial is not None
        and trace_diagnostics.get("initial_witness_pass") is True
        and trace_diagnostics.get("iteration_sequence_pass") is True
        and trace_diagnostics.get("all_orders_are_permutations") is True
        and trace_diagnostics.get("all_inner_witnesses_pass") is True
        and trace_diagnostics.get("recurrence_pass") is True
        and trace_diagnostics.get("decrease_gate_pass") is True
        and trace_diagnostics.get("terminal_combined_convergence_pass") is True
        and final_aggregates.get("all_final_private_gates_pass") is True
    )


def _expected_failure_validation_pass(
    failure: Mapping[str, object] | None,
    initial: Mapping[str, object] | None,
    trace: Sequence[Mapping[str, object]],
    trace_diagnostics: Mapping[str, object],
    *,
    maximum_outer_iterations: int,
) -> bool:
    if failure is None:
        return False
    kind = failure.get("kind")
    stage_proof = failure.get("stage_proof")
    if not isinstance(stage_proof, Mapping) or stage_proof.get("pass") is not True:
        return False
    if (
        trace_diagnostics.get("iteration_sequence_pass") is not True
        or trace_diagnostics.get("all_orders_are_permutations") is not True
    ):
        return False
    initial_failure = bool(
        initial is None
        and not trace
        and kind in ("inner-nonconvergence", "canonical-factor-key-degeneracy")
    )
    prior_trace_valid = bool(
        initial is not None
        and trace_diagnostics.get("initial_witness_pass") is True
        and trace_diagnostics.get("all_inner_witnesses_pass") is True
        and trace_diagnostics.get("recurrence_pass") is True
        and trace_diagnostics.get("decrease_gate_pass") is True
    )
    if not initial_failure and not prior_trace_valid:
        return False
    if kind == "outer-nonconvergence":
        return bool(
            len(trace) == maximum_outer_iterations
            and trace_diagnostics.get("terminal_combined_convergence_pass") is False
            and trace_diagnostics.get("earliest_combined_convergence_iteration") is None
        )
    return True


def _expected_inner_batch_count(preparation: object, candidate: ModuleType) -> int:
    count = sum(
        (well.counts.row_count + candidate.SCIPLEX3_CANDIDATE_BATCH_SIZE - 1)
        // candidate.SCIPLEX3_CANDIDATE_BATCH_SIZE
        for well in preparation.training_data.wells
    )
    if type(count) is not int or count <= 0:
        raise TrajectoryError("exact p1 preparation has no candidate inner batches")
    return count


def _private_state_aggregates(
    *,
    basis: object,
    alpha: object,
    rho: object,
    delta: object,
    factor_shape: object,
    contributions: object,
    mean_activation: object,
    training_well_plate_indices: object,
    action_well_indices: object,
    vehicle_well_indices: object,
    candidate: ModuleType,
    expected_terminal_order: Sequence[int] | None,
    require_canonical_axes: bool,
) -> dict[str, object]:
    np = candidate.np
    fixed_shape = np.asarray(factor_shape, dtype="<f8", order="C")
    fixed_shape_pass = bool(
        fixed_shape.shape == (1,)
        and np.array_equal(
            fixed_shape,
            np.asarray([EXPECTED_FIXED_FACTOR_SHAPE], dtype="<f8"),
        )
    )
    reconstructed = _reconstruct_mean_activation(
        alpha,
        rho,
        delta,
        training_well_plate_indices,
        action_well_indices,
        vehicle_well_indices,
        candidate,
    )
    canonical_reconstructed = _canonical_audit_matrix(
        reconstructed,
        np,
        decimals=candidate.SCIPLEX3_CANDIDATE_FACTOR_ORDER_DECIMALS,
    )
    sealed_activation = np.asarray(mean_activation, dtype="<f8", order="C")
    activation_match = bool(np.array_equal(sealed_activation, canonical_reconstructed))
    independent_contributions = _independent_contributions(sealed_activation, candidate)
    sealed_contributions = np.asarray(contributions, dtype=np.float64)
    contributions_match = bool(np.array_equal(sealed_contributions, independent_contributions))
    key_diagnostics, independent_order = _factor_key_diagnostics(
        basis,
        independent_contributions,
        candidate,
    )
    expected_order_match = bool(
        expected_terminal_order is None
        or independent_order == tuple(int(item) for item in expected_terminal_order)
    )
    canonical_axes_pass = bool(
        not require_canonical_axes
        or independent_order == tuple(range(candidate.SCIPLEX3_CANDIDATE_FACTOR_COUNT))
    )
    loading_rank = _rank_diagnostics(
        basis,
        np,
        name="loading matrix",
        decimals=candidate.SCIPLEX3_CANDIDATE_FACTOR_ORDER_DECIMALS,
        threshold=candidate.SCIPLEX3_CANDIDATE_IDENTIFIABILITY_THRESHOLD,
        margin=candidate.SCIPLEX3_CANDIDATE_IDENTIFIABILITY_MARGIN,
        boundary_epsilon_multiplier=(
            candidate.SCIPLEX3_CANDIDATE_QUANTIZATION_BOUNDARY_EPS_MULTIPLIER
        ),
    )
    activation_rank = _rank_diagnostics(
        sealed_activation,
        np,
        name="mean activation matrix",
        decimals=candidate.SCIPLEX3_CANDIDATE_FACTOR_ORDER_DECIMALS,
        threshold=candidate.SCIPLEX3_CANDIDATE_IDENTIFIABILITY_THRESHOLD,
        margin=candidate.SCIPLEX3_CANDIDATE_IDENTIFIABILITY_MARGIN,
        boundary_epsilon_multiplier=(
            candidate.SCIPLEX3_CANDIDATE_QUANTIZATION_BOUNDARY_EPS_MULTIPLIER
        ),
    )
    contribution = _contribution_diagnostics(independent_contributions, candidate)
    rho_diagnostics = _rho_diagnostics(rho, candidate)
    basis_array = np.asarray(basis, dtype=np.float64)
    basis_simplex_pass = bool(
        basis_array.shape
        == (
            candidate.SCIPLEX3_CANDIDATE_FACTOR_COUNT,
            2_000,
        )
        and np.all(np.isfinite(basis_array))
        and np.all(basis_array > 0.0)
        and np.allclose(
            np.sum(basis_array, axis=1),
            1.0,
            rtol=0.0,
            atol=5e-13,
        )
    )
    tensor_summaries = {
        "action_well_indices": _tensor_summary(action_well_indices, np, integer=True),
        "alpha": _tensor_summary(alpha, np),
        "basis": _tensor_summary(basis, np),
        "delta": _tensor_summary(delta, np),
        "factor_shape": _tensor_summary(factor_shape, np),
        "mean_activation": _tensor_summary(mean_activation, np),
        "rho": _tensor_summary(rho, np),
        "training_well_plate_indices": _tensor_summary(
            training_well_plate_indices, np, integer=True
        ),
        "vehicle_well_indices": _tensor_summary(vehicle_well_indices, np, integer=True),
    }
    tensor_finite = all(cast(bool, item["all_finite"]) for item in tensor_summaries.values())
    gate_pass = bool(
        fixed_shape_pass
        and basis_simplex_pass
        and activation_match
        and contributions_match
        and key_diagnostics["keys_unique"] is True
        and expected_order_match
        and canonical_axes_pass
        and loading_rank["gate_pass"] is True
        and activation_rank["gate_pass"] is True
        and contribution["gate_pass"] is True
        and rho_diagnostics["factorwise_mean_one_pass"] is True
        and rho_diagnostics["all_finite_strictly_between_zero_and_eight"] is True
        and tensor_finite
    )
    return {
        "all_final_private_gates_pass": gate_pass,
        "basis_positive_simplex_pass": basis_simplex_pass,
        "canonical_axes_pass": canonical_axes_pass,
        "contribution_diagnostics": contribution,
        "factor_key_diagnostics": key_diagnostics,
        "factor_shape": _float_manifest(float(fixed_shape[0])),
        "fixed_factor_shape_pass": fixed_shape_pass,
        "independent_contributions_match": contributions_match,
        "independent_mean_activation_match": activation_match,
        "independent_order_matches_terminal_trace": expected_order_match,
        "loading_rank": loading_rank,
        "mean_activation_rank": activation_rank,
        "rho_diagnostics": rho_diagnostics,
        "tensor_summaries": tensor_summaries,
        "tensors_all_finite": tensor_finite,
    }


def _expected_topology(
    preparation: object, design: object, candidate: ModuleType
) -> tuple[object, ...]:
    np = candidate.np
    training_well_ids = tuple(well.well_id for well in preparation.training_data.wells)
    well_index = {well_id: index for index, well_id in enumerate(training_well_ids)}
    plate_index = {plate_id: index for index, plate_id in enumerate(design.plate_ids)}
    compound_index = {compound: index for index, compound in enumerate(design.compounds)}
    training_well_plate_indices = np.asarray(
        [plate_index[well.plate_id] for well in preparation.training_data.wells],
        dtype="<i8",
    )
    action_well_indices = np.empty(
        (
            candidate.SCIPLEX3_CANDIDATE_COMPOUND_COUNT,
            len(candidate.SCIPLEX3_CANDIDATE_DOSES_NM),
        ),
        dtype="<i8",
    )
    for action in design.actions:
        action_well_indices[
            compound_index[action.compound],
            candidate.SCIPLEX3_CANDIDATE_DOSES_NM.index(action.dose_nm),
        ] = well_index[action.well_id]
    vehicle_well_indices = np.asarray(
        [[well_index[well_id] for well_id in vehicle.well_ids] for vehicle in design.vehicles],
        dtype="<i8",
    )
    return (
        training_well_ids,
        training_well_plate_indices,
        action_well_indices,
        vehicle_well_indices,
    )


def _success_aggregates(
    fitted: object,
    preparation: object,
    design: object,
    candidate: ModuleType,
) -> tuple[dict[str, object], dict[str, object], list[dict[str, object]]]:
    if type(fitted) is not candidate.SciPlex3GammaPoissonCandidate:
        raise TrajectoryError("in-memory fitter returned a substituted exact candidate class")
    initial = _copy_inner_witness(
        fitted.initial_equilibration,
        candidate.SciPlex3CandidateInitialEquilibration,
    )
    trace = _copy_trace(fitted.trace, candidate.SciPlex3CandidateTraceEntry)
    expected_ids, expected_plate, expected_action, expected_vehicle = _expected_topology(
        preparation,
        design,
        candidate,
    )
    topology_pass = bool(
        fitted.training_well_ids == expected_ids
        and candidate.np.array_equal(fitted._training_well_plate_indices, expected_plate)
        and candidate.np.array_equal(fitted._action_well_indices, expected_action)
        and candidate.np.array_equal(fitted._vehicle_well_indices, expected_vehicle)
    )
    float_private_arrays = (
        fitted._basis,
        fitted._alpha,
        fitted._rho,
        fitted._delta,
        fitted._factor_shape,
        fitted._factor_contributions,
        fitted._mean_activation,
    )
    integer_private_arrays = (
        fitted._training_well_plate_indices,
        fitted._action_well_indices,
        fitted._vehicle_well_indices,
    )
    frozen_private_storage_pass = bool(
        all(
            type(value) is candidate.np.ndarray
            and value.dtype.str == "<f8"
            and value.flags.c_contiguous
            and not value.flags.writeable
            for value in float_private_arrays
        )
        and all(
            type(value) is candidate.np.ndarray
            and value.dtype.str == "<i8"
            and value.flags.c_contiguous
            and not value.flags.writeable
            for value in integer_private_arrays
        )
    )
    state = _private_state_aggregates(
        basis=fitted._basis,
        alpha=fitted._alpha,
        rho=fitted._rho,
        delta=fitted._delta,
        factor_shape=fitted._factor_shape,
        contributions=fitted._factor_contributions,
        mean_activation=fitted._mean_activation,
        training_well_plate_indices=fitted._training_well_plate_indices,
        action_well_indices=fitted._action_well_indices,
        vehicle_well_indices=fitted._vehicle_well_indices,
        candidate=candidate,
        expected_terminal_order=None,
        require_canonical_axes=True,
    )
    training_summary_pass = bool(
        type(fitted.training_summary) is candidate.SciPlex3CandidateTrainingSummary
        and fitted.training_summary.provenance == "real-p1"
        and fitted.training_summary.record_count
        == candidate.SCIPLEX3_CANDIDATE_TRAINING_RECORD_COUNT
        and fitted.training_summary.well_count == candidate.SCIPLEX3_CANDIDATE_TRAINING_WELL_COUNT
        and fitted.training_summary.zero_panel_record_count
        == candidate.SCIPLEX3_CANDIDATE_ZERO_PANEL_RECORD_COUNT
        and fitted.training_summary.design_sha256 == design.fingerprint
        and fitted.training_summary.training_data_sha256
        == candidate.training_data_fingerprint(preparation.training_data)
    )
    state["p1_topology_matches_preparation"] = topology_pass
    state["private_tensor_storage_frozen"] = frozen_private_storage_pass
    state["training_summary_matches_preparation"] = training_summary_pass
    state["all_final_private_gates_pass"] = bool(
        state["all_final_private_gates_pass"]
        and topology_pass
        and frozen_private_storage_pass
        and training_summary_pass
    )
    return initial, state, trace


def _provisional_failure_aggregates(
    frame_locals: Mapping[str, object],
    trace: list[dict[str, object]],
    candidate: ModuleType,
    *,
    failure_kind: str,
    failure_stage: str,
) -> dict[str, object]:
    np = candidate.np
    result: dict[str, object] = {}
    summaries: dict[str, object] = {}
    for name, integer in (
        ("alpha", False),
        ("basis", False),
        ("contributions", False),
        ("current_basis", False),
        ("current_contributions", False),
        ("delta", False),
        ("loading_concentration", False),
        ("mean_activation", False),
        ("rho", False),
        ("well_factor_means", False),
    ):
        if name in frame_locals:
            summaries[name] = _tensor_summary(frame_locals[name], np, integer=integer)
    state = frame_locals.get("state")
    if type(state) is candidate._LocalVariationalState:
        summaries["theta_shape"] = _tensor_summary(state.theta_shape, np)
        summaries["theta_rate"] = _tensor_summary(state.theta_rate, np)
    result["tensor_summaries"] = summaries
    result["all_available_tensors_finite"] = bool(
        summaries and all(cast(bool, item["all_finite"]) for item in summaries.values())
    )
    final_required = {
        "alpha",
        "basis",
        "contributions",
        "delta",
        "mean_activation",
        "rho",
        "validated",
    }
    current_required = {
        "alpha",
        "delta",
        "loading_concentration",
        "rho",
        "validated",
        "well_factor_means",
    }
    if failure_stage == "post-canonical-final-private-gate" and final_required.issubset(
        frame_locals
    ):
        validated = frame_locals["validated"]
        provisional = _private_state_aggregates(
            basis=frame_locals["basis"],
            alpha=frame_locals["alpha"],
            rho=frame_locals["rho"],
            delta=frame_locals["delta"],
            factor_shape=np.asarray([EXPECTED_FIXED_FACTOR_SHAPE], dtype=np.float64),
            contributions=frame_locals["contributions"],
            mean_activation=frame_locals["mean_activation"],
            training_well_plate_indices=validated.training_well_plate_indices,
            action_well_indices=validated.action_well_indices,
            vehicle_well_indices=validated.vehicle_well_indices,
            candidate=candidate,
            expected_terminal_order=None,
            require_canonical_axes=True,
        )
        provisional["failure_kind"] = failure_kind
        provisional["failure_stage"] = failure_stage
        provisional["recomputed_order_matches_current_order"] = None
        provisional["recomputed_order_matches_last_trace"] = None
        provisional["state_axis_stage"] = "post-canonical-final"
        result["provisional_final_state"] = provisional
    elif current_required.issubset(frame_locals):
        validated = frame_locals["validated"]
        loading_concentration = np.asarray(frame_locals["loading_concentration"], dtype=np.float64)
        loading_sums = np.sum(loading_concentration, axis=1)
        recomputed_basis = loading_concentration / loading_sums[:, None]
        reconstructed = _canonical_audit_matrix(
            _reconstruct_mean_activation(
                frame_locals["alpha"],
                frame_locals["rho"],
                frame_locals["delta"],
                validated.training_well_plate_indices,
                validated.action_well_indices,
                validated.vehicle_well_indices,
                candidate,
            ),
            np,
            decimals=candidate.SCIPLEX3_CANDIDATE_FACTOR_ORDER_DECIMALS,
        )
        recomputed_contributions = _independent_contributions(
            frame_locals["well_factor_means"], candidate
        )
        provisional = _private_state_aggregates(
            basis=recomputed_basis,
            alpha=frame_locals["alpha"],
            rho=frame_locals["rho"],
            delta=frame_locals["delta"],
            factor_shape=np.asarray([EXPECTED_FIXED_FACTOR_SHAPE], dtype=np.float64),
            contributions=recomputed_contributions,
            mean_activation=reconstructed,
            training_well_plate_indices=validated.training_well_plate_indices,
            action_well_indices=validated.action_well_indices,
            vehicle_well_indices=validated.vehicle_well_indices,
            candidate=candidate,
            expected_terminal_order=None,
            require_canonical_axes=False,
        )
        recomputed_order = tuple(cast(list[int], provisional["factor_key_diagnostics"]["order"]))
        current_order_match: bool | None = None
        current_order_assignment_completed = bool(
            (
                failure_kind == "material-elbo-decrease"
                and failure_stage == "outer-objective-gate-before-trace-append"
            )
            or (
                failure_kind == "outer-nonconvergence"
                and failure_stage == "outer-cap-after-final-trace-append"
            )
        )
        if current_order_assignment_completed:
            current_order_raw = frame_locals.get("current_order")
            if current_order_raw is not None:
                current_order = tuple(int(item) for item in current_order_raw)
                current_order_match = recomputed_order == current_order
            else:
                current_order_match = False
        last_trace_match: bool | None = None
        if (
            failure_kind == "outer-nonconvergence"
            and failure_stage == "outer-cap-after-final-trace-append"
        ):
            last_trace_match = bool(
                trace and recomputed_order == tuple(cast(list[int], trace[-1]["factor_order"]))
            )
        provisional["all_final_private_gates_pass"] = bool(
            provisional["all_final_private_gates_pass"]
            and current_order_match is not False
            and last_trace_match is not False
        )
        provisional["failure_kind"] = failure_kind
        provisional["failure_stage"] = failure_stage
        provisional["recomputed_order_matches_current_order"] = current_order_match
        provisional["recomputed_order_matches_last_trace"] = last_trace_match
        provisional["state_axis_stage"] = "pre-canonical-outer"
        result["provisional_final_state"] = provisional
    else:
        result["provisional_final_state"] = None
        result["missing_provisional_fields"] = {
            "post_canonical_final": sorted(final_required - set(frame_locals)),
            "pre_canonical_outer": sorted(current_required - set(frame_locals)),
        }
    sufficient = frame_locals.get("current_sufficient", frame_locals.get("previous_sufficient"))
    if type(sufficient) is candidate._PassSufficientStatistics:
        result["latest_sufficient_statistics"] = {
            "allocation_entropy": _float_manifest(float(sufficient.allocation_entropy)),
            "inner_sweep_count_histogram": [
                int(item) for item in sufficient.inner_sweep_count_histogram
            ],
            "loading_counts": _tensor_summary(sufficient.loading_counts, np),
            "maximum_inner_sweeps": int(sufficient.maximum_inner_sweeps),
            "maximum_terminal_elog_residual": _float_manifest(
                float(sufficient.maximum_terminal_elog_residual)
            ),
            "maximum_terminal_shape_residual": _float_manifest(
                float(sufficient.maximum_terminal_shape_residual)
            ),
            "poisson_factorial": _float_manifest(float(sufficient.poisson_factorial)),
            "theta_count_elog": _float_manifest(float(sufficient.theta_count_elog)),
            "well_theta_means": _tensor_summary(sufficient.well_theta_means, np),
        }
    else:
        result["latest_sufficient_statistics"] = None
    return result


@dataclasses.dataclass(slots=True)
class _LocalMapReplayInput:
    """Private copied inputs for one bounded local-map replay.

    These arrays never enter the report.  They are copied before traceback clearing and are
    overwritten and replaced with empty arrays immediately after replay.
    """

    well_index: int
    well_row_count: int
    batch_index: int
    batch_start: int
    batch_stop: int
    global_start: int
    global_stop: int
    expected_log_loading: object
    batch_rate: object
    row_for_entry: object
    feature_indices: object
    values: object
    prior_mean: object
    rho: object
    traceback_current_shape: object
    traceback_proposal_shape: object
    traceback_allocated_counts: object
    traceback_responsibilities: object
    traceback_allocations: object
    traceback_state_terminal_shape: object
    normalized_basis_row_sha256: tuple[str, ...]
    traceback_hashes: dict[str, object]
    copies_isolated: bool
    traceback_current_aliases_terminal_state: bool


@dataclasses.dataclass(slots=True)
class _LocalMapEvaluation:
    proposal_shape: object
    expected_log_theta: object
    allocated_counts: object
    responsibilities: object
    allocations: object
    pre_shape_objective: float
    post_coordinate_objective: float
    pre_shape_objective_fsum: float
    post_coordinate_objective_fsum: float


def _canonical_array_sha256(value: object, np: ModuleType, *, integer: bool = False) -> str:
    dtype = "<i8" if integer else "<f8"
    canonical = np.asarray(value, dtype=dtype, order="C")
    return _sha256(canonical.tobytes(order="C"))


def _copy_c_array(value: object, np: ModuleType, *, integer: bool = False) -> object:
    dtype = np.int64 if integer else np.float64
    return np.array(value, dtype=dtype, order="C", copy=True)


def _combined_state_sha256(shape: object, rate: object, np: ModuleType) -> str:
    shape_bytes = np.asarray(shape, dtype="<f8", order="C").tobytes(order="C")
    rate_bytes = np.asarray(rate, dtype="<f8", order="C").tobytes(order="C")
    return _sha256(b"item12.1-local-state\0shape\0" + shape_bytes + b"\0rate\0" + rate_bytes)


def _combined_allocation_sha256(
    allocated_counts: object,
    responsibilities: object,
    allocations: object,
    np: ModuleType,
) -> str:
    count_bytes = np.asarray(allocated_counts, dtype="<f8", order="C").tobytes(order="C")
    responsibility_bytes = np.asarray(responsibilities, dtype="<f8", order="C").tobytes(order="C")
    allocation_bytes = np.asarray(allocations, dtype="<f8", order="C").tobytes(order="C")
    return _sha256(
        b"item12.1-local-allocation\0counts\0"
        + count_bytes
        + b"\0responsibilities\0"
        + responsibility_bytes
        + b"\0allocations\0"
        + allocation_bytes
    )


def _attempt2_full_state_hashes(
    fit_locals: Mapping[str, object], candidate: ModuleType
) -> dict[str, str]:
    np = candidate.np
    state = fit_locals.get("state")
    if type(state) is not candidate._LocalVariationalState:
        raise TrajectoryError("initial failure lacks the exact local-state type")
    required = ("loading_concentration", "well_factor_means", "rho")
    if any(name not in fit_locals for name in required):
        raise TrajectoryError("initial failure lacks an attempt-2 full-state binding")
    return {
        "loading_concentration_sha256": _canonical_array_sha256(
            fit_locals["loading_concentration"], np
        ),
        "rho_sha256": _canonical_array_sha256(fit_locals["rho"], np),
        "theta_rate_sha256": _canonical_array_sha256(state.theta_rate, np),
        "theta_shape_sha256": _canonical_array_sha256(state.theta_shape, np),
        "well_factor_means_sha256": _canonical_array_sha256(fit_locals["well_factor_means"], np),
    }


def _require_attempt2_baseline(
    fit_locals: Mapping[str, object],
    cavi_locals: Mapping[str, object],
    candidate: ModuleType,
) -> dict[str, object]:
    if (
        fit_locals.get("initial_equilibration") is not None
        or fit_locals.get("iteration") is not None
        or fit_locals.get("trace", []) != []
    ):
        raise TrajectoryError("Item 12.1 accepts only the initial-untraced inner failure")
    if (
        type(cavi_locals.get("sweep_count")) is not int
        or cavi_locals["sweep_count"] != candidate.SCIPLEX3_CANDIDATE_MAX_INNER_SWEEPS
        or type(cavi_locals.get("passing_streak")) is not int
        or cavi_locals["passing_streak"] != 0
    ):
        raise TrajectoryError("initial inner failure cap/streak differs from attempt 2")
    shape_residual = cavi_locals.get("terminal_shape_residual")
    elog_residual = cavi_locals.get("terminal_elog_residual")
    if (
        type(shape_residual) is not float
        or shape_residual.hex() != EXPECTED_ATTEMPT2_TERMINAL_SHAPE_RESIDUAL_HEX
        or type(elog_residual) is not float
        or elog_residual.hex() != EXPECTED_ATTEMPT2_TERMINAL_ELOG_RESIDUAL_HEX
    ):
        raise TrajectoryError("initial inner failure residuals differ from attempt 2")
    hashes = _attempt2_full_state_hashes(fit_locals, candidate)
    expected_hashes = {
        "loading_concentration_sha256": EXPECTED_ATTEMPT2_LOADING_CONCENTRATION_SHA256,
        "rho_sha256": EXPECTED_ATTEMPT2_RHO_SHA256,
        "theta_rate_sha256": EXPECTED_ATTEMPT2_THETA_RATE_SHA256,
        "theta_shape_sha256": EXPECTED_ATTEMPT2_THETA_SHAPE_SHA256,
        "well_factor_means_sha256": EXPECTED_ATTEMPT2_WELL_FACTOR_MEANS_SHA256,
    }
    if hashes != expected_hashes:
        raise TrajectoryError("initial inner failure full state differs from attempt 2")
    return {
        "full_state_hashes": hashes,
        "passing_streak": 0,
        "terminal_elog_residual": _float_manifest(elog_residual),
        "terminal_shape_residual": _float_manifest(shape_residual),
    }


def _capture_local_map_replay_input(
    fit_locals: Mapping[str, object],
    cavi_locals: Mapping[str, object],
    candidate: ModuleType,
) -> tuple[_LocalMapReplayInput, dict[str, object]]:
    np = candidate.np
    baseline = _require_attempt2_baseline(fit_locals, cavi_locals, candidate)
    state = fit_locals["state"]
    integer_names = ("row_for_entry", "feature_indices")
    float_names = (
        "expected_log_loading",
        "batch_rate",
        "values",
        "prior_mean",
        "current_shape",
        "proposal_shape",
        "allocated_counts",
        "responsibilities",
        "allocations",
    )
    if any(name not in cavi_locals for name in (*integer_names, *float_names)):
        raise TrajectoryError("exact CAVI frame lacks a required local-map input")
    well_index = cavi_locals.get("well_index")
    well_row_count = cavi_locals.get("row_count")
    batch_start = cavi_locals.get("batch_start")
    batch_stop = cavi_locals.get("batch_stop")
    global_slice = cavi_locals.get("global_slice")
    offset = cavi_locals.get("offset")
    validated = fit_locals.get("validated")
    if (
        type(well_index) is not int
        or not 0 <= well_index < candidate.SCIPLEX3_CANDIDATE_TRAINING_WELL_COUNT
        or type(well_row_count) is not int
        or well_row_count <= 0
        or type(batch_start) is not int
        or type(batch_stop) is not int
        or not 0 <= batch_start < batch_stop <= well_row_count
        or type(global_slice) is not slice
        or type(global_slice.start) is not int
        or type(global_slice.stop) is not int
        or global_slice.step is not None
        or type(offset) is not int
        or type(validated) is not candidate._ValidatedTrainingDesign
        or well_row_count != validated.wells[well_index].counts.row_count
        or batch_start % candidate.SCIPLEX3_CANDIDATE_BATCH_SIZE != 0
        or batch_stop != min(batch_start + candidate.SCIPLEX3_CANDIDATE_BATCH_SIZE, well_row_count)
        or global_slice.start != offset + batch_start
        or global_slice.stop != offset + batch_stop
    ):
        raise TrajectoryError("exact CAVI frame has invalid canonical batch bounds")
    if (
        cavi_locals.get("validated") is not validated
        or cavi_locals.get("state") is not fit_locals.get("state")
        or cavi_locals.get("loading_concentration") is not fit_locals.get("loading_concentration")
        or cavi_locals.get("well_factor_means") is not fit_locals.get("well_factor_means")
    ):
        raise TrajectoryError("exact CAVI arguments differ from the fit-frame globals")
    batch_size = batch_stop - batch_start
    factor_count = candidate.SCIPLEX3_CANDIDATE_FACTOR_COUNT
    expected_log_loading = cavi_locals["expected_log_loading"]
    batch_rate = cavi_locals["batch_rate"]
    row_for_entry = cavi_locals["row_for_entry"]
    feature_indices = cavi_locals["feature_indices"]
    values = cavi_locals["values"]
    prior_mean = cavi_locals["prior_mean"]
    current_shape = cavi_locals["current_shape"]
    proposal_shape = cavi_locals["proposal_shape"]
    allocated_counts = cavi_locals["allocated_counts"]
    responsibilities = cavi_locals["responsibilities"]
    allocations = cavi_locals["allocations"]
    terminal_shape = state.theta_shape[global_slice]
    if (
        np.asarray(expected_log_loading).shape != (factor_count, candidate.SCIPLEX3_FEATURE_COUNT)
        or np.asarray(batch_rate).shape != (batch_size, factor_count)
        or np.asarray(prior_mean).shape != (factor_count,)
        or np.asarray(current_shape).shape != (batch_size, factor_count)
        or np.asarray(proposal_shape).shape != (batch_size, factor_count)
        or np.asarray(allocated_counts).shape != (batch_size, factor_count)
        or np.asarray(terminal_shape).shape != (batch_size, factor_count)
        or np.asarray(row_for_entry).ndim != 1
        or np.asarray(feature_indices).shape != np.asarray(row_for_entry).shape
        or np.asarray(values).shape != np.asarray(row_for_entry).shape
        or np.asarray(responsibilities).shape != (len(values), factor_count)
        or np.asarray(allocations).shape != (len(values), factor_count)
    ):
        raise TrajectoryError("exact CAVI frame local-map array topology drifted")
    row_array = np.asarray(row_for_entry)
    feature_array = np.asarray(feature_indices)
    value_array = np.asarray(values)
    if (
        bool(np.any(row_array < 0))
        or bool(np.any(row_array >= batch_size))
        or bool(np.any(feature_array < 0))
        or bool(np.any(feature_array >= candidate.SCIPLEX3_FEATURE_COUNT))
        or not bool(np.all(np.isfinite(value_array)))
        or bool(np.any(value_array <= 0.0))
        or not bool(np.all(value_array == np.floor(value_array)))
    ):
        raise TrajectoryError("exact CAVI frame sparse map inputs are invalid")
    expected_fixed_rate = (
        candidate.SCIPLEX3_CANDIDATE_FIXED_FACTOR_SHAPE / np.asarray(prior_mean, dtype=np.float64)
        + 1.0
    )
    if not np.array_equal(
        np.asarray(prior_mean), np.asarray(fit_locals["well_factor_means"])[well_index]
    ):
        raise TrajectoryError("exact CAVI prior mean differs from the fit-frame well mean")
    if (
        not bool(np.all(np.isfinite(expected_fixed_rate)))
        or bool(np.any(expected_fixed_rate <= 1.0))
        or not np.array_equal(
            np.asarray(batch_rate), np.broadcast_to(expected_fixed_rate, (batch_size, factor_count))
        )
    ):
        raise TrajectoryError("exact CAVI frame rate differs from the frozen local map")
    recomputed_expected_log_loading = (
        candidate.digamma(fit_locals["loading_concentration"])
        - candidate.digamma(np.sum(fit_locals["loading_concentration"], axis=1))[:, None]
    )
    if not np.array_equal(np.asarray(expected_log_loading), recomputed_expected_log_loading):
        raise TrajectoryError("exact CAVI expected-log loading differs from frozen globals")
    loading_concentration = np.asarray(
        fit_locals["loading_concentration"], dtype=np.float64, order="C"
    )
    normalized_basis = loading_concentration / np.sum(loading_concentration, axis=1, keepdims=True)
    normalized_basis_row_sha256 = tuple(
        _canonical_array_sha256(normalized_basis[factor_index], np)
        for factor_index in range(factor_count)
    )
    if len(set(normalized_basis_row_sha256)) != factor_count:
        raise TrajectoryError("local-map normalized basis rows lack unique digest bindings")
    current_aliases_terminal = bool(
        np.shares_memory(np.asarray(current_shape), np.asarray(state.theta_shape))
        and np.array_equal(np.asarray(current_shape), np.asarray(terminal_shape))
    )
    if (
        not current_aliases_terminal
        or not np.array_equal(np.asarray(current_shape), np.asarray(proposal_shape))
        or not np.array_equal(np.asarray(proposal_shape), np.asarray(terminal_shape))
    ):
        raise TrajectoryError("traceback terminal shape alias semantics drifted")
    histogram = cavi_locals.get("inner_sweep_count_histogram")
    maximum_prior_sweeps = cavi_locals.get("maximum_inner_sweeps")
    if (
        type(histogram) is not list
        or len(histogram) != candidate.SCIPLEX3_CANDIDATE_MAX_INNER_SWEEPS
        or any(type(value) is not int or value < 0 for value in histogram)
        or histogram[0] != 0
        or type(maximum_prior_sweeps) is not int
    ):
        raise TrajectoryError("initial failure lacks a canonical prior-batch schedule witness")
    canonical_offset = sum(validated.wells[index].counts.row_count for index in range(well_index))
    if offset != canonical_offset:
        raise TrajectoryError("initial failure global offset differs from canonical well order")
    canonical_prior_batch_count = (
        sum(
            (validated.wells[index].counts.row_count + candidate.SCIPLEX3_CANDIDATE_BATCH_SIZE - 1)
            // candidate.SCIPLEX3_CANDIDATE_BATCH_SIZE
            for index in range(well_index)
        )
        + batch_start // candidate.SCIPLEX3_CANDIDATE_BATCH_SIZE
    )
    observed_prior_batch_count = sum(histogram)
    occupied_sweeps = [index + 1 for index, count in enumerate(histogram) if count]
    if observed_prior_batch_count != canonical_prior_batch_count or maximum_prior_sweeps != (
        max(occupied_sweeps) if occupied_sweeps else 0
    ):
        raise TrajectoryError("prior canonical batches lack exact convergence schedule proof")
    baseline["prior_converged_batch_schedule"] = {
        "canonical_prior_batch_count": int(canonical_prior_batch_count),
        "histogram_sha256": _sha256(_canonical_json_bytes(histogram)),
        "maximum_sweeps": maximum_prior_sweeps,
        "nonzero_bins": [
            {"converged_batch_count": histogram[index], "sweep": index + 1}
            for index in range(len(histogram))
            if histogram[index]
        ],
        "observed_prior_batch_count": observed_prior_batch_count,
        "pass": True,
    }
    arrays: dict[str, object] = {}
    try:
        for name in (*integer_names, *float_names):
            arrays[name] = _copy_c_array(cavi_locals[name], np, integer=name in integer_names)
        arrays["rho"] = _copy_c_array(fit_locals["rho"], np)
        arrays["state_terminal_shape"] = _copy_c_array(terminal_shape, np)
    except BaseException:
        for private_array in arrays.values():
            private = np.asarray(private_array)
            if private.flags.writeable:
                private.fill(0)
        raise
    originals = {
        **{name: cavi_locals[name] for name in (*integer_names, *float_names)},
        "rho": fit_locals["rho"],
        "state_terminal_shape": terminal_shape,
    }
    copies_isolated = all(
        not np.shares_memory(np.asarray(arrays[name]), np.asarray(originals[name]))
        for name in arrays
    )
    if not copies_isolated:
        for private_array in arrays.values():
            private = np.asarray(private_array)
            if private.flags.writeable:
                private.fill(0)
        raise TrajectoryError("local-map replay input copy aliases traceback state")
    traceback_hashes = {
        "allocated_counts_sha256": _canonical_array_sha256(allocated_counts, np),
        "allocation_sha256": _canonical_array_sha256(allocations, np),
        "batch_rate_sha256": _canonical_array_sha256(batch_rate, np),
        "current_shape_sha256": _canonical_array_sha256(current_shape, np),
        "expected_log_loading_sha256": _canonical_array_sha256(expected_log_loading, np),
        "feature_index_sha256": _canonical_array_sha256(feature_indices, np, integer=True),
        "prior_mean_sha256": _canonical_array_sha256(prior_mean, np),
        "proposal_shape_sha256": _canonical_array_sha256(proposal_shape, np),
        "responsibility_sha256": _canonical_array_sha256(responsibilities, np),
        "row_for_entry_sha256": _canonical_array_sha256(row_for_entry, np, integer=True),
        "state_terminal_shape_sha256": _canonical_array_sha256(terminal_shape, np),
        "value_sha256": _canonical_array_sha256(values, np),
    }
    copied_hashes = {
        "allocated_counts_sha256": _canonical_array_sha256(arrays["allocated_counts"], np),
        "allocation_sha256": _canonical_array_sha256(arrays["allocations"], np),
        "batch_rate_sha256": _canonical_array_sha256(arrays["batch_rate"], np),
        "current_shape_sha256": _canonical_array_sha256(arrays["current_shape"], np),
        "expected_log_loading_sha256": _canonical_array_sha256(arrays["expected_log_loading"], np),
        "feature_index_sha256": _canonical_array_sha256(
            arrays["feature_indices"], np, integer=True
        ),
        "prior_mean_sha256": _canonical_array_sha256(arrays["prior_mean"], np),
        "proposal_shape_sha256": _canonical_array_sha256(arrays["proposal_shape"], np),
        "responsibility_sha256": _canonical_array_sha256(arrays["responsibilities"], np),
        "row_for_entry_sha256": _canonical_array_sha256(arrays["row_for_entry"], np, integer=True),
        "state_terminal_shape_sha256": _canonical_array_sha256(arrays["state_terminal_shape"], np),
        "value_sha256": _canonical_array_sha256(arrays["values"], np),
    }
    if copied_hashes != traceback_hashes:
        for private_array in arrays.values():
            private = np.asarray(private_array)
            if private.flags.writeable:
                private.fill(0)
        raise TrajectoryError("local-map replay copies differ from traceback inputs")
    replay_input = _LocalMapReplayInput(
        well_index=well_index,
        well_row_count=well_row_count,
        batch_index=batch_start // candidate.SCIPLEX3_CANDIDATE_BATCH_SIZE,
        batch_start=batch_start,
        batch_stop=batch_stop,
        global_start=cast(int, global_slice.start),
        global_stop=cast(int, global_slice.stop),
        expected_log_loading=arrays["expected_log_loading"],
        batch_rate=arrays["batch_rate"],
        row_for_entry=arrays["row_for_entry"],
        feature_indices=arrays["feature_indices"],
        values=arrays["values"],
        prior_mean=arrays["prior_mean"],
        rho=arrays["rho"],
        traceback_current_shape=arrays["current_shape"],
        traceback_proposal_shape=arrays["proposal_shape"],
        traceback_allocated_counts=arrays["allocated_counts"],
        traceback_responsibilities=arrays["responsibilities"],
        traceback_allocations=arrays["allocations"],
        traceback_state_terminal_shape=arrays["state_terminal_shape"],
        normalized_basis_row_sha256=normalized_basis_row_sha256,
        traceback_hashes=traceback_hashes,
        copies_isolated=copies_isolated,
        traceback_current_aliases_terminal_state=current_aliases_terminal,
    )
    return replay_input, baseline


def _local_variational_objective(
    replay_input: _LocalMapReplayInput,
    shape: object,
    allocated_counts: object,
    responsibilities: object,
    allocations: object,
    candidate: ModuleType,
) -> float:
    """Evaluate L(shape, phi) for one fixed-global batch in production summation order."""

    np = candidate.np
    shape_array = np.asarray(shape, dtype=np.float64, order="C")
    rate = np.asarray(replay_input.batch_rate, dtype=np.float64, order="C")
    feature_indices = np.asarray(replay_input.feature_indices, dtype=np.int64, order="C")
    values = np.asarray(replay_input.values, dtype=np.float64, order="C")
    expected_log_loading = np.asarray(
        replay_input.expected_log_loading, dtype=np.float64, order="C"
    )
    responsibility_array = np.asarray(responsibilities, dtype=np.float64, order="C")
    allocation_array = np.asarray(allocations, dtype=np.float64, order="C")
    allocated_array = np.asarray(allocated_counts, dtype=np.float64, order="C")
    expected_log_theta = candidate.digamma(shape_array) - np.log(rate)
    theta_mean = shape_array / rate
    loading_term = (
        0.0
        if not len(values)
        else float(np.sum(allocation_array * expected_log_loading[:, feature_indices].T))
    )
    positive = responsibility_array > 0.0
    allocation_entropy = (
        0.0
        if not len(values)
        else -float(np.sum(allocation_array[positive] * np.log(responsibility_array[positive])))
    )
    poisson_factorial = 0.0 if not len(values) else float(np.sum(candidate.gammaln(values + 1.0)))
    likelihood = (
        float(np.sum(allocated_array * expected_log_theta))
        + loading_term
        + allocation_entropy
        - poisson_factorial
        - float(np.sum(theta_mean))
    )
    factor_shape = candidate.SCIPLEX3_CANDIDATE_FIXED_FACTOR_SHAPE
    prior_mean = np.asarray(replay_input.prior_mean, dtype=np.float64, order="C")
    theta_prior = (
        factor_shape * (math.log(factor_shape) - np.log(prior_mean)[None, :])
        - float(candidate.gammaln(factor_shape))
        + (factor_shape - 1.0) * expected_log_theta
        - factor_shape * theta_mean / prior_mean[None, :]
    )
    unweighted = likelihood + float(
        np.sum(theta_prior) + np.sum(candidate._gamma_entropy(shape_array, rate))
    )
    omega = candidate.SCIPLEX3_CANDIDATE_TRAINING_RECORD_COUNT / (
        candidate.SCIPLEX3_CANDIDATE_TRAINING_WELL_COUNT * replay_input.well_row_count
    )
    objective = omega * unweighted
    if not math.isfinite(objective):
        raise TrajectoryError("local-map variational objective is nonfinite")
    return float(objective)


def _local_variational_objective_fsum(
    replay_input: _LocalMapReplayInput,
    shape: object,
    allocated_counts: object,
    responsibilities: object,
    allocations: object,
    candidate: ModuleType,
) -> float:
    """Independently sum the same local objective with fixed C-order ``math.fsum``."""

    np = candidate.np
    shape_array = np.asarray(shape, dtype=np.float64, order="C")
    rate = np.asarray(replay_input.batch_rate, dtype=np.float64, order="C")
    feature_indices = np.asarray(replay_input.feature_indices, dtype=np.int64, order="C")
    values = np.asarray(replay_input.values, dtype=np.float64, order="C")
    expected_log_loading = np.asarray(
        replay_input.expected_log_loading, dtype=np.float64, order="C"
    )
    responsibility_array = np.asarray(responsibilities, dtype=np.float64, order="C")
    allocation_array = np.asarray(allocations, dtype=np.float64, order="C")
    allocated_array = np.asarray(allocated_counts, dtype=np.float64, order="C")
    expected_log_theta = candidate.digamma(shape_array) - np.log(rate)
    theta_mean = shape_array / rate
    terms: list[float] = [
        math.fsum(
            float(value) for value in (allocated_array * expected_log_theta).ravel(order="C")
        ),
        -math.fsum(float(value) for value in theta_mean.ravel(order="C")),
    ]
    if len(values):
        loading_terms = allocation_array * expected_log_loading[:, feature_indices].T
        terms.append(math.fsum(float(value) for value in loading_terms.ravel(order="C")))
        positive = responsibility_array > 0.0
        terms.append(
            -math.fsum(
                float(value)
                for value in (allocation_array[positive] * np.log(responsibility_array[positive]))
            )
        )
        terms.append(-math.fsum(float(value) for value in candidate.gammaln(values + 1.0)))
    factor_shape = candidate.SCIPLEX3_CANDIDATE_FIXED_FACTOR_SHAPE
    prior_mean = np.asarray(replay_input.prior_mean, dtype=np.float64, order="C")
    theta_prior = (
        factor_shape * (math.log(factor_shape) - np.log(prior_mean)[None, :])
        - float(candidate.gammaln(factor_shape))
        + (factor_shape - 1.0) * expected_log_theta
        - factor_shape * theta_mean / prior_mean[None, :]
    )
    entropy = candidate._gamma_entropy(shape_array, rate)
    terms.append(math.fsum(float(value) for value in theta_prior.ravel(order="C")))
    terms.append(math.fsum(float(value) for value in entropy.ravel(order="C")))
    omega = candidate.SCIPLEX3_CANDIDATE_TRAINING_RECORD_COUNT / (
        candidate.SCIPLEX3_CANDIDATE_TRAINING_WELL_COUNT * replay_input.well_row_count
    )
    objective = omega * math.fsum(terms)
    if not math.isfinite(objective):
        raise TrajectoryError("independent local-map variational objective is nonfinite")
    return float(objective)


def _evaluate_local_map(
    replay_input: _LocalMapReplayInput,
    current_shape: object,
    candidate: ModuleType,
) -> _LocalMapEvaluation:
    np = candidate.np
    shape = np.asarray(current_shape, dtype=np.float64, order="C")
    rate = np.asarray(replay_input.batch_rate, dtype=np.float64, order="C")
    row_for_entry = np.asarray(replay_input.row_for_entry, dtype=np.int64, order="C")
    feature_indices = np.asarray(replay_input.feature_indices, dtype=np.int64, order="C")
    values = np.asarray(replay_input.values, dtype=np.float64, order="C")
    expected_log_loading = np.asarray(
        replay_input.expected_log_loading, dtype=np.float64, order="C"
    )
    expected_log_theta = candidate.digamma(shape) - np.log(rate)
    allocated_counts = np.zeros_like(shape)
    responsibilities = np.empty(
        (len(values), candidate.SCIPLEX3_CANDIDATE_FACTOR_COUNT), dtype=np.float64
    )
    allocations = np.empty_like(responsibilities)
    if len(values):
        logits = expected_log_theta[row_for_entry] + expected_log_loading[:, feature_indices].T
        logits -= np.max(logits, axis=1, keepdims=True)
        responsibilities[:] = np.exp(logits)
        responsibilities /= np.sum(responsibilities, axis=1, keepdims=True)
        allocations[:] = values[:, None] * responsibilities
        np.add.at(allocated_counts, row_for_entry, allocations)
    proposal_shape = candidate.SCIPLEX3_CANDIDATE_FIXED_FACTOR_SHAPE + allocated_counts
    pre_shape_objective = _local_variational_objective(
        replay_input,
        shape,
        allocated_counts,
        responsibilities,
        allocations,
        candidate,
    )
    post_coordinate_objective = _local_variational_objective(
        replay_input,
        proposal_shape,
        allocated_counts,
        responsibilities,
        allocations,
        candidate,
    )
    pre_shape_objective_fsum = _local_variational_objective_fsum(
        replay_input,
        shape,
        allocated_counts,
        responsibilities,
        allocations,
        candidate,
    )
    post_coordinate_objective_fsum = _local_variational_objective_fsum(
        replay_input,
        proposal_shape,
        allocated_counts,
        responsibilities,
        allocations,
        candidate,
    )
    if (
        not bool(np.all(np.isfinite(proposal_shape)))
        or not bool(np.all(np.isfinite(responsibilities)))
        or not bool(np.all(np.isfinite(allocations)))
    ):
        raise TrajectoryError("local-map replay produced a nonfinite value")
    return _LocalMapEvaluation(
        proposal_shape=np.asarray(proposal_shape, dtype=np.float64, order="C"),
        expected_log_theta=np.asarray(expected_log_theta, dtype=np.float64, order="C"),
        allocated_counts=np.asarray(allocated_counts, dtype=np.float64, order="C"),
        responsibilities=np.asarray(responsibilities, dtype=np.float64, order="C"),
        allocations=np.asarray(allocations, dtype=np.float64, order="C"),
        pre_shape_objective=pre_shape_objective,
        post_coordinate_objective=post_coordinate_objective,
        pre_shape_objective_fsum=pre_shape_objective_fsum,
        post_coordinate_objective_fsum=post_coordinate_objective_fsum,
    )


def _relative_distance(left: object, right: object, np: ModuleType) -> tuple[float, float]:
    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    absolute = float(np.linalg.norm(right_array - left_array))
    denominator = max(1.0, float(np.linalg.norm(left_array)), float(np.linalg.norm(right_array)))
    return absolute, absolute / denominator


def _componentwise_shape_distance(left: object, right: object, np: ModuleType) -> float:
    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    denominator = np.maximum(1.0, np.maximum(np.abs(left_array), np.abs(right_array)))
    return float(np.max(np.abs(right_array - left_array) / denominator))


def _elog_distance(left: object, right: object, candidate: ModuleType) -> float:
    np = candidate.np
    return float(
        np.max(
            np.abs(
                candidate.digamma(np.asarray(right, dtype=np.float64))
                - candidate.digamma(np.asarray(left, dtype=np.float64))
            )
        )
    )


def _map_residuals(current: object, proposal: object, candidate: ModuleType) -> tuple[float, float]:
    np = candidate.np
    current_array = np.asarray(current, dtype=np.float64)
    proposal_array = np.asarray(proposal, dtype=np.float64)
    denominator = np.maximum(1.0, np.maximum(np.abs(current_array), np.abs(proposal_array)))
    shape_residual = float(np.max(np.abs(proposal_array - current_array) / denominator))
    elog_residual = float(
        np.max(np.abs(candidate.digamma(proposal_array) - candidate.digamma(current_array)))
    )
    return shape_residual, elog_residual


def _update_cosine_sign(
    current: object, one_step: object, two_step: object, np: ModuleType
) -> tuple[float | None, int]:
    current_array = np.asarray(current, dtype=np.float64)
    one_array = np.asarray(one_step, dtype=np.float64)
    two_array = np.asarray(two_step, dtype=np.float64)
    first = (one_array - current_array) / np.maximum(
        1.0, np.maximum(np.abs(current_array), np.abs(one_array))
    )
    second = (two_array - one_array) / np.maximum(
        1.0, np.maximum(np.abs(one_array), np.abs(two_array))
    )
    first_flat = first.ravel(order="C")
    second_flat = second.ravel(order="C")
    dot = math.fsum(
        float(first_flat[index]) * float(second_flat[index]) for index in range(first_flat.size)
    )
    first_norm = math.sqrt(math.fsum(float(value) * float(value) for value in first_flat))
    second_norm = math.sqrt(math.fsum(float(value) * float(value) for value in second_flat))
    denominator = first_norm * second_norm
    cosine = None if denominator == 0.0 else max(-1.0, min(1.0, dot / denominator))
    sign = 1 if dot > 0.0 else (-1 if dot < 0.0 else 0)
    return (None if cosine is None else float(cosine)), sign


def _objective_change_diagnostic(
    previous: float,
    current: float,
    previous_fsum: float,
    current_fsum: float,
    candidate: ModuleType,
) -> dict[str, object]:
    np = candidate.np
    change = current - previous
    fsum_change = current_fsum - previous_fsum
    production_threshold = candidate.SCIPLEX3_CANDIDATE_ELBO_DECREASE_RTOL * max(1.0, abs(previous))
    fsum_threshold = candidate.SCIPLEX3_CANDIDATE_ELBO_DECREASE_RTOL * max(1.0, abs(previous_fsum))
    production_ambiguity = 64.0 * np.finfo(np.float64).eps * max(1.0, abs(previous), abs(current))
    fsum_ambiguity = (
        64.0 * np.finfo(np.float64).eps * max(1.0, abs(previous_fsum), abs(current_fsum))
    )
    production_material_decrease = bool(change < -production_threshold)
    fsum_material_decrease = bool(fsum_change < -fsum_threshold)
    boundary_ambiguous = bool(
        abs(change + production_threshold) <= production_ambiguity
        or abs(fsum_change + fsum_threshold) <= fsum_ambiguity
        or production_material_decrease != fsum_material_decrease
    )
    return {
        "boundary_or_summation_ambiguous": boundary_ambiguous,
        "change": _float_manifest(float(change)),
        "fsum_change": _float_manifest(float(fsum_change)),
        "fsum_ambiguity_band": _float_manifest(float(fsum_ambiguity)),
        "fsum_material_decrease": fsum_material_decrease,
        "fsum_material_decrease_threshold": _float_manifest(float(fsum_threshold)),
        "production_material_decrease": production_material_decrease,
        "production_ambiguity_band": _float_manifest(float(production_ambiguity)),
        "production_material_decrease_threshold": _float_manifest(float(production_threshold)),
    }


def _mass_invariants(
    replay_input: _LocalMapReplayInput,
    current_shape: object,
    evaluation: _LocalMapEvaluation,
    candidate: ModuleType,
    *,
    input_state_index: int,
) -> dict[str, object]:
    np = candidate.np
    shape = np.asarray(current_shape, dtype=np.float64)
    rate = np.asarray(replay_input.batch_rate, dtype=np.float64)
    responsibilities = np.asarray(evaluation.responsibilities, dtype=np.float64)
    allocated_counts = np.asarray(evaluation.allocated_counts, dtype=np.float64)
    row_for_entry = np.asarray(replay_input.row_for_entry, dtype=np.int64)
    values = np.asarray(replay_input.values, dtype=np.float64)
    row_input_mass = np.bincount(row_for_entry, weights=values, minlength=shape.shape[0])
    row_allocated_mass = np.sum(allocated_counts, axis=1)
    proposal_mass = np.sum(
        np.asarray(evaluation.proposal_shape, dtype=np.float64)
        - candidate.SCIPLEX3_CANDIDATE_FIXED_FACTOR_SHAPE,
        axis=1,
    )
    responsibility_error = (
        0.0 if not len(values) else float(np.max(np.abs(np.sum(responsibilities, axis=1) - 1.0)))
    )
    allocation_error = float(np.max(np.abs(row_allocated_mass - row_input_mass)))
    proposal_mass_error = float(np.max(np.abs(proposal_mass - row_input_mass)))
    tolerance = (
        candidate.SCIPLEX3_CANDIDATE_MASS_EPS_MULTIPLIER
        * np.finfo(np.float64).eps
        * max(1.0, float(np.max(row_input_mass)))
    )
    fixed_rate = (
        candidate.SCIPLEX3_CANDIDATE_FIXED_FACTOR_SHAPE
        / np.asarray(replay_input.prior_mean, dtype=np.float64)
        + 1.0
    )
    incoming_shape_mass_pass = bool(
        np.array_equal(
            shape,
            np.full_like(shape, candidate.SCIPLEX3_CANDIDATE_FIXED_FACTOR_SHAPE),
        )
        if input_state_index == 0
        else np.max(
            np.abs(
                np.sum(shape - candidate.SCIPLEX3_CANDIDATE_FIXED_FACTOR_SHAPE, axis=1)
                - row_input_mass
            )
        )
        <= tolerance
    )
    allocation_finite_pass = bool(
        np.all(np.isfinite(evaluation.allocations)) and np.all(evaluation.allocations >= 0.0)
    )
    rate_range_pass = bool(np.all(np.isfinite(rate)) and np.all(rate > 1.0))
    shape_range_pass = bool(
        np.all(np.isfinite(shape))
        and np.all(shape >= candidate.SCIPLEX3_CANDIDATE_FIXED_FACTOR_SHAPE)
    )
    fixed_rate_pass = bool(np.array_equal(rate, np.broadcast_to(fixed_rate, rate.shape)))
    map_surfaces_finite = bool(
        np.all(np.isfinite(evaluation.expected_log_theta))
        and np.all(np.isfinite(responsibilities))
        and np.all(np.isfinite(allocated_counts))
        and np.all(np.isfinite(evaluation.proposal_shape))
    )
    proposal_elementwise_pass = bool(
        np.array_equal(
            np.asarray(evaluation.proposal_shape, dtype=np.float64),
            candidate.SCIPLEX3_CANDIDATE_FIXED_FACTOR_SHAPE + allocated_counts,
        )
    )
    result = {
        "allocation_nonnegative_finite": bool(allocation_finite_pass),
        "allocation_maximum": _float_manifest(
            0.0 if not evaluation.allocations.size else float(np.max(evaluation.allocations))
        ),
        "allocation_minimum": _float_manifest(
            0.0 if not evaluation.allocations.size else float(np.min(evaluation.allocations))
        ),
        "allocated_count_maximum": _float_manifest(float(np.max(allocated_counts))),
        "allocated_count_minimum": _float_manifest(float(np.min(allocated_counts))),
        "allocated_row_mass_max_abs_error": _float_manifest(allocation_error),
        "allocated_row_mass_pass": bool(allocation_error <= tolerance),
        "fixed_rate_bit_exact": fixed_rate_pass,
        "expected_log_theta_maximum": _float_manifest(float(np.max(evaluation.expected_log_theta))),
        "expected_log_theta_minimum": _float_manifest(float(np.min(evaluation.expected_log_theta))),
        "incoming_shape_mass_pass": incoming_shape_mass_pass,
        "map_numeric_surfaces_all_finite": map_surfaces_finite,
        "proposal_row_mass_max_abs_error": _float_manifest(proposal_mass_error),
        "proposal_row_mass_pass": bool(proposal_mass_error <= tolerance),
        "proposal_shape_equals_fixed_prior_plus_allocated_counts": proposal_elementwise_pass,
        "proposal_shape_maximum": _float_manifest(float(np.max(evaluation.proposal_shape))),
        "proposal_shape_minimum": _float_manifest(float(np.min(evaluation.proposal_shape))),
        "rate_finite_strictly_above_one": rate_range_pass,
        "rate_maximum": _float_manifest(float(np.max(rate))),
        "rate_minimum": _float_manifest(float(np.min(rate))),
        "responsibility_simplex_max_abs_error": _float_manifest(responsibility_error),
        "responsibility_maximum": _float_manifest(
            0.0
            if not evaluation.responsibilities.size
            else float(np.max(evaluation.responsibilities))
        ),
        "responsibility_minimum": _float_manifest(
            0.0
            if not evaluation.responsibilities.size
            else float(np.min(evaluation.responsibilities))
        ),
        "responsibility_simplex_pass": bool(
            responsibility_error
            <= (candidate.SCIPLEX3_CANDIDATE_MASS_EPS_MULTIPLIER * np.finfo(np.float64).eps)
        ),
        "shape_finite_at_or_above_fixed_prior": shape_range_pass,
        "shape_maximum": _float_manifest(float(np.max(shape))),
        "shape_minimum": _float_manifest(float(np.min(shape))),
    }
    result["all_pass"] = bool(
        allocation_finite_pass
        and allocation_error <= tolerance
        and fixed_rate_pass
        and incoming_shape_mass_pass
        and map_surfaces_finite
        and proposal_mass_error <= tolerance
        and proposal_elementwise_pass
        and rate_range_pass
        and responsibility_error
        <= candidate.SCIPLEX3_CANDIDATE_MASS_EPS_MULTIPLIER * np.finfo(np.float64).eps
        and shape_range_pass
    )
    return result


def _worst_row_factor_summary(
    replay_input: _LocalMapReplayInput,
    current_shape: object,
    evaluation: _LocalMapEvaluation,
    candidate: ModuleType,
) -> dict[str, object]:
    np = candidate.np
    shape = np.asarray(current_shape, dtype=np.float64)
    proposal = np.asarray(evaluation.proposal_shape, dtype=np.float64)
    elog_change = np.abs(candidate.digamma(proposal) - candidate.digamma(shape))
    shape_change = np.abs(proposal - shape) / np.maximum(
        1.0, np.maximum(np.abs(shape), np.abs(proposal))
    )
    severity = np.maximum(shape_change, elog_change) / (
        candidate.SCIPLEX3_CANDIDATE_INNER_RESIDUAL_TOL
    )
    flat_index = int(np.argmax(severity.ravel(order="C")))
    row_index, factor_index = np.unravel_index(flat_index, severity.shape, order="C")
    row_for_entry = np.asarray(replay_input.row_for_entry, dtype=np.int64)
    values = np.asarray(replay_input.values, dtype=np.float64)
    row_mask = row_for_entry == row_index
    row_umi = math.fsum(float(value) for value in values[row_mask])
    if not row_umi.is_integer():
        raise TrajectoryError("local-map row UMI total is not integral")
    allocated_count = float(np.asarray(evaluation.allocated_counts)[row_index, factor_index])
    return {
        "Relog": _float_manifest(float(elog_change[row_index, factor_index])),
        "Rshape": _float_manifest(float(shape_change[row_index, factor_index])),
        "allocated_count": _float_manifest(allocated_count),
        "allocated_share_of_row_umi": _float_manifest(
            0.0 if row_umi == 0.0 else allocated_count / row_umi
        ),
        "batch_row_index": int(row_index),
        "current_shape": _float_manifest(float(shape[row_index, factor_index])),
        "factor_index": int(factor_index),
        "fixed_rate": _float_manifest(
            float(np.asarray(replay_input.batch_rate)[row_index, factor_index])
        ),
        "prior_mean": _float_manifest(float(np.asarray(replay_input.prior_mean)[factor_index])),
        "proposal_shape": _float_manifest(float(proposal[row_index, factor_index])),
        "row_nonzero_entry_count": int(np.sum(row_mask)),
        "row_umi_total": int(row_umi),
        "severity_over_tolerance": _float_manifest(float(severity[row_index, factor_index])),
        "well_row_index": replay_input.batch_start + int(row_index),
    }


def _per_row_residual_summary(
    replay_input: _LocalMapReplayInput,
    current: object,
    one_step: object,
    two_step: object,
    candidate: ModuleType,
) -> dict[str, object]:
    np = candidate.np
    current_array = np.asarray(current, dtype=np.float64)
    one_array = np.asarray(one_step, dtype=np.float64)
    two_array = np.asarray(two_step, dtype=np.float64)
    first_denominator = np.maximum(1.0, np.maximum(np.abs(current_array), np.abs(one_array)))
    second_denominator = np.maximum(1.0, np.maximum(np.abs(current_array), np.abs(two_array)))
    d1_shape_by_row = np.max(np.abs(one_array - current_array) / first_denominator, axis=1)
    d2_shape_by_row = np.max(np.abs(two_array - current_array) / second_denominator, axis=1)
    d1_elog_by_row = np.max(
        np.abs(candidate.digamma(one_array) - candidate.digamma(current_array)), axis=1
    )
    d2_elog_by_row = np.max(
        np.abs(candidate.digamma(two_array) - candidate.digamma(current_array)), axis=1
    )
    row_cosines = np.empty(current_array.shape[0], dtype=np.float64)
    row_cosine_defined = np.zeros(current_array.shape[0], dtype=np.int64)
    for row_index in range(current_array.shape[0]):
        cosine, _sign = _update_cosine_sign(
            current_array[row_index], one_array[row_index], two_array[row_index], np
        )
        row_cosines[row_index] = 0.0 if cosine is None else cosine
        row_cosine_defined[row_index] = int(cosine is not None)
    tolerance = candidate.SCIPLEX3_CANDIDATE_INNER_RESIDUAL_TOL
    d1_severity_by_row = np.maximum(d1_shape_by_row, d1_elog_by_row) / tolerance
    d2_severity_by_row = np.maximum(d2_shape_by_row, d2_elog_by_row) / tolerance
    worst_d1_row = int(np.argmax(d1_severity_by_row))
    worst_d2_row = int(np.argmax(d2_severity_by_row))
    defined_rows = np.flatnonzero(row_cosine_defined)
    most_negative_row = (
        None if not len(defined_rows) else int(defined_rows[np.argmin(row_cosines[defined_rows])])
    )
    canonical = np.stack(
        (
            d1_shape_by_row,
            d1_elog_by_row,
            d2_shape_by_row,
            d2_elog_by_row,
            row_cosines,
            row_cosine_defined.astype(np.float64),
        ),
        axis=1,
    )
    return {
        "metric_matrix_sha256": _canonical_array_sha256(canonical, np),
        "most_negative_update_cosine": (
            None
            if most_negative_row is None
            else {
                "batch_row_index": most_negative_row,
                "cosine": _float_manifest(float(row_cosines[most_negative_row])),
                "well_row_index": replay_input.batch_start + most_negative_row,
            }
        ),
        "row_count": int(current_array.shape[0]),
        "worst_one_step": {
            "batch_row_index": worst_d1_row,
            "Relog": _float_manifest(float(d1_elog_by_row[worst_d1_row])),
            "Rshape": _float_manifest(float(d1_shape_by_row[worst_d1_row])),
            "severity_over_tolerance": _float_manifest(float(d1_severity_by_row[worst_d1_row])),
            "well_row_index": replay_input.batch_start + worst_d1_row,
        },
        "worst_two_step": {
            "batch_row_index": worst_d2_row,
            "Relog": _float_manifest(float(d2_elog_by_row[worst_d2_row])),
            "Rshape": _float_manifest(float(d2_shape_by_row[worst_d2_row])),
            "severity_over_tolerance": _float_manifest(float(d2_severity_by_row[worst_d2_row])),
            "well_row_index": replay_input.batch_start + worst_d2_row,
        },
    }


def _row_jacobians(
    replay_input: _LocalMapReplayInput,
    current_shape: object,
    evaluation: _LocalMapEvaluation,
    candidate: ModuleType,
    polygamma_function: Any,
) -> list[object]:
    np = candidate.np
    shape = np.asarray(current_shape, dtype=np.float64)
    row_for_entry = np.asarray(replay_input.row_for_entry, dtype=np.int64)
    values = np.asarray(replay_input.values, dtype=np.float64)
    responsibilities = np.asarray(evaluation.responsibilities, dtype=np.float64)
    zero = np.zeros(
        (candidate.SCIPLEX3_CANDIDATE_FACTOR_COUNT, candidate.SCIPLEX3_CANDIDATE_FACTOR_COUNT),
        dtype=np.float64,
    )
    result: list[object] = []
    for row_index in range(shape.shape[0]):
        positions = np.flatnonzero(row_for_entry == row_index)
        if len(positions):
            probabilities = responsibilities[positions]
            weights = values[positions]
            diagonal = np.diag(np.sum(weights[:, None] * probabilities, axis=0))
            outer = np.einsum("e,ek,el->kl", weights, probabilities, probabilities)
            covariance = diagonal - outer
            jacobian = (
                covariance
                * np.asarray(polygamma_function(1, shape[row_index]), dtype=np.float64)[None, :]
            )
        else:
            jacobian = np.zeros_like(zero)
        result.append(np.asarray(jacobian, dtype=np.float64, order="C"))
    return result


def _jacobian_radius(
    jacobian: object,
    shape: object,
    candidate: ModuleType,
    polygamma_function: Any,
) -> dict[str, object]:
    np = candidate.np
    matrix = np.asarray(jacobian, dtype=np.float64, order="C")
    trigamma = np.asarray(polygamma_function(1, np.asarray(shape, dtype=np.float64)))
    root = np.sqrt(trigamma)
    hessian = matrix / trigamma[None, :]
    symmetric = root[:, None] * hessian * root[None, :]
    symmetric = 0.5 * (symmetric + symmetric.T)
    symmetric_eigenvalues = np.asarray(np.linalg.eigvalsh(symmetric), dtype=np.float64)
    generic_eigenvalues = np.asarray(np.linalg.eigvals(matrix), dtype=np.complex128)
    symmetric_radius = float(np.max(symmetric_eigenvalues))
    generic_radius = float(np.max(np.abs(generic_eigenvalues)))
    ambiguity = (
        64.0 * np.finfo(np.float64).eps * max(1.0, float(np.linalg.norm(symmetric, ord=np.inf)))
    )
    if not all(math.isfinite(value) for value in (symmetric_radius, generic_radius, ambiguity)):
        raise TrajectoryError("local-map Jacobian spectral radius is nonfinite")
    agreement = bool(abs(symmetric_radius - generic_radius) <= ambiguity)
    lower = max(0.0, symmetric_radius - ambiguity)
    upper = symmetric_radius + ambiguity
    return {
        "ambiguity_band": _float_manifest(float(ambiguity)),
        "generic_eigenvalue_radius": _float_manifest(generic_radius),
        "generic_eigenvalue_sha256": _sha256(
            np.asarray(generic_eigenvalues, dtype="<c16", order="C").tobytes(order="C")
        ),
        "lower_bound": _float_manifest(float(lower)),
        "methods_agree_within_ambiguity": agreement,
        "noncontractive": bool(lower > 1.0),
        "spectral_radius": _float_manifest(symmetric_radius),
        "symmetric_eigenvalue_sha256": _canonical_array_sha256(symmetric_eigenvalues, np),
        "spectral_status": (
            "noncontractive" if lower > 1.0 else ("contractive" if upper < 1.0 else "ambiguous")
        ),
        "upper_bound": _float_manifest(float(upper)),
    }


def _maximum_row_jacobian_summary(
    replay_input: _LocalMapReplayInput,
    current_shape: object,
    jacobians: list[object],
    candidate: ModuleType,
    polygamma_function: Any,
    *,
    sweep: int,
) -> dict[str, object]:
    np = candidate.np
    shape = np.asarray(current_shape, dtype=np.float64)
    radii = [
        _jacobian_radius(jacobian, shape[row_index], candidate, polygamma_function)
        for row_index, jacobian in enumerate(jacobians)
    ]
    maximum_row = max(
        range(len(radii)),
        key=lambda index: cast(dict[str, object], radii[index]["spectral_radius"])["value"],
    )
    maximum_jacobian = jacobians[maximum_row]
    radius = radii[maximum_row]
    return {
        "batch_row_index": maximum_row,
        "factor_count": candidate.SCIPLEX3_CANDIDATE_FACTOR_COUNT,
        "jacobian_sha256": _canonical_array_sha256(maximum_jacobian, np),
        "radius": radius,
        "sweep": sweep,
        "well_row_index": replay_input.batch_start + maximum_row,
    }


def _two_step_jacobian_summary(
    replay_input: _LocalMapReplayInput,
    jacobians_49: list[object],
    jacobians_50: list[object],
    candidate: ModuleType,
) -> dict[str, object]:
    np = candidate.np
    maximum_radius = -1.0
    maximum_row = 0
    maximum_composition: object | None = None
    maximum_eigenvalues: object | None = None
    for row_index, (first, second) in enumerate(zip(jacobians_49, jacobians_50, strict=True)):
        composition = np.asarray(second) @ np.asarray(first)
        eigenvalues = np.asarray(np.linalg.eigvals(composition), dtype=np.complex128)
        radius = float(np.max(np.abs(eigenvalues)))
        if not math.isfinite(radius):
            raise TrajectoryError("two-step local-map Jacobian radius is nonfinite")
        if radius > maximum_radius:
            maximum_radius = radius
            maximum_row = row_index
            maximum_composition = composition
            maximum_eigenvalues = eigenvalues
    if maximum_composition is None or maximum_eigenvalues is None:
        raise TrajectoryError("two-step local-map Jacobian has no row block")
    ambiguity = (
        64.0
        * np.finfo(np.float64).eps
        * max(1.0, float(np.linalg.norm(maximum_composition, ord=np.inf)))
    )
    lower = max(0.0, maximum_radius - ambiguity)
    upper = maximum_radius + ambiguity
    return {
        "ambiguity_band": _float_manifest(float(ambiguity)),
        "batch_row_index": maximum_row,
        "composition_order": "J(A50)@J(A49)",
        "factor_count": candidate.SCIPLEX3_CANDIDATE_FACTOR_COUNT,
        "jacobian_sha256": _canonical_array_sha256(maximum_composition, np),
        "lower_bound": _float_manifest(float(lower)),
        "noncontractive": bool(lower > 1.0),
        "spectral_radius": _float_manifest(maximum_radius),
        "eigenvalue_sha256": _sha256(
            np.asarray(maximum_eigenvalues, dtype="<c16", order="C").tobytes(order="C")
        ),
        "spectral_status": (
            "noncontractive" if lower > 1.0 else ("contractive" if upper < 1.0 else "ambiguous")
        ),
        "upper_bound": _float_manifest(float(upper)),
        "well_row_index": replay_input.batch_start + maximum_row,
    }


def _rho_effective_context_diagnostics(
    replay_input: _LocalMapReplayInput, candidate: ModuleType
) -> list[dict[str, object]]:
    np = candidate.np
    values = np.asarray(replay_input.rho, dtype=np.float64, order="C")
    expected_shape = (
        candidate.SCIPLEX3_CANDIDATE_PLATE_COUNT,
        candidate.SCIPLEX3_CANDIDATE_FACTOR_COUNT,
    )
    if (
        values.shape != expected_shape
        or not bool(np.all(np.isfinite(values)))
        or bool(np.any(values <= 0.0))
    ):
        raise TrajectoryError("plate-context diagnostic input is invalid")
    result: list[dict[str, object]] = []
    for factor_index in range(candidate.SCIPLEX3_CANDIDATE_FACTOR_COUNT):
        total = math.fsum(float(values[plate_index, factor_index]) for plate_index in range(8))
        shares = np.asarray(values[:, factor_index] / total, dtype=np.float64)
        share_sum = math.fsum(float(value) for value in shares)
        squared_total = math.fsum(
            float(values[plate_index, factor_index]) * float(values[plate_index, factor_index])
            for plate_index in range(8)
        )
        effective_count = total * total / squared_total
        maximum_index = int(np.argmax(shares))
        maximum_share = float(shares[maximum_index])
        minimum_share = float(np.min(shares))
        epsilon = np.finfo(np.float64).eps
        effective_count_ambiguity = 64.0 * epsilon * max(1.0, abs(effective_count))
        maximum_share_ambiguity = 64.0 * epsilon * max(1.0, abs(maximum_share))
        mean_one_error = abs(total / candidate.SCIPLEX3_CANDIDATE_PLATE_COUNT - 1.0)
        share_sum_error = abs(share_sum - 1.0)
        effective_count_boundary_ambiguous = bool(
            abs(effective_count - 2.0) <= effective_count_ambiguity
        )
        maximum_share_boundary_ambiguous = bool(abs(maximum_share - 0.5) <= maximum_share_ambiguity)
        effective_count_clear_collapse = bool(
            effective_count < 2.0 and not effective_count_boundary_ambiguous
        )
        maximum_share_clear_collapse = bool(
            maximum_share > 0.5 and not maximum_share_boundary_ambiguous
        )
        gates_pass = bool(
            mean_one_error <= 5e-13
            and share_sum_error <= 8.0 * epsilon
            and 1.0 <= effective_count <= float(candidate.SCIPLEX3_CANDIDATE_PLATE_COUNT)
            and 1.0 / candidate.SCIPLEX3_CANDIDATE_PLATE_COUNT <= maximum_share <= 1.0
            and 0.0 < minimum_share <= maximum_share
        )
        result.append(
            {
                "collapse_boundary_ambiguous": bool(
                    effective_count_boundary_ambiguous or maximum_share_boundary_ambiguous
                ),
                "collapsed": bool(effective_count_clear_collapse or maximum_share_clear_collapse),
                "effective_count_below_two": bool(effective_count < 2.0),
                "effective_count_clear_collapse": effective_count_clear_collapse,
                "effective_count_collapse_boundary_ambiguous": (effective_count_boundary_ambiguous),
                "effective_context_count_inverse_simpson": _float_manifest(effective_count),
                "effective_context_count_ambiguity_band": _float_manifest(
                    float(effective_count_ambiguity)
                ),
                "factor_index": factor_index,
                "factorwise_mean": _float_manifest(
                    float(total / candidate.SCIPLEX3_CANDIDATE_PLATE_COUNT)
                ),
                "factorwise_mean_one_abs_error": _float_manifest(float(mean_one_error)),
                "factorwise_sum": _float_manifest(float(total)),
                "gates_pass": gates_pass,
                "maximum_share": _float_manifest(maximum_share),
                "maximum_share_above_one_half": bool(maximum_share > 0.5),
                "maximum_share_ambiguity_band": _float_manifest(float(maximum_share_ambiguity)),
                "maximum_share_clear_collapse": maximum_share_clear_collapse,
                "maximum_share_collapse_boundary_ambiguous": (maximum_share_boundary_ambiguous),
                "maximum_share_context_index": maximum_index,
                "minimum_share": _float_manifest(minimum_share),
                "normalized_basis_row_sha256": replay_input.normalized_basis_row_sha256[
                    factor_index
                ],
                "rho_squared_sum": _float_manifest(float(squared_total)),
                "share_sum": _float_manifest(float(share_sum)),
                "share_sum_abs_error": _float_manifest(float(share_sum_error)),
                "tiny_normalized_share": bool(minimum_share <= math.sqrt(np.finfo(np.float64).eps)),
            }
        )
    return result


def _wipe_replay_input(replay_input: _LocalMapReplayInput, np: ModuleType) -> bool:
    array_fields = (
        "expected_log_loading",
        "batch_rate",
        "row_for_entry",
        "feature_indices",
        "values",
        "prior_mean",
        "rho",
        "traceback_current_shape",
        "traceback_proposal_shape",
        "traceback_allocated_counts",
        "traceback_responsibilities",
        "traceback_allocations",
        "traceback_state_terminal_shape",
    )
    for name in array_fields:
        value = np.asarray(getattr(replay_input, name))
        if value.flags.writeable:
            value.fill(0)
        setattr(replay_input, name, np.empty((0,), dtype=value.dtype))
    return all(np.asarray(getattr(replay_input, name)).size == 0 for name in array_fields)


def _strict_two_cycle_summary(
    replay_input: _LocalMapReplayInput,
    state_49: object,
    state_50: object,
    state_51: object,
    candidate: ModuleType,
) -> dict[str, object]:
    np = candidate.np
    a49 = np.asarray(state_49, dtype=np.float64)
    a50 = np.asarray(state_50, dtype=np.float64)
    a51 = np.asarray(state_51, dtype=np.float64)
    denominator_1 = np.maximum(1.0, np.maximum(np.abs(a49), np.abs(a50)))
    denominator_2 = np.maximum(1.0, np.maximum(np.abs(a49), np.abs(a51)))
    d1_shape = np.max(np.abs(a50 - a49) / denominator_1, axis=1)
    d2_shape = np.max(np.abs(a51 - a49) / denominator_2, axis=1)
    d1_elog = np.max(np.abs(candidate.digamma(a50) - candidate.digamma(a49)), axis=1)
    adjacent_second_denominator = np.maximum(1.0, np.maximum(np.abs(a50), np.abs(a51)))
    adjacent_second_shape = np.max(np.abs(a51 - a50) / adjacent_second_denominator, axis=1)
    adjacent_second_elog = np.max(np.abs(candidate.digamma(a51) - candidate.digamma(a50)), axis=1)
    d2_elog = np.max(np.abs(candidate.digamma(a51) - candidate.digamma(a49)), axis=1)
    cosines: list[float | None] = []
    for row_index in range(a49.shape[0]):
        cosine, _sign = _update_cosine_sign(a49[row_index], a50[row_index], a51[row_index], np)
        cosines.append(cosine)
    qualifying = [
        row_index
        for row_index, cosine in enumerate(cosines)
        if (
            d1_shape[row_index] > candidate.SCIPLEX3_CANDIDATE_INNER_RESIDUAL_TOL
            or d1_elog[row_index] > candidate.SCIPLEX3_CANDIDATE_INNER_RESIDUAL_TOL
        )
        and (
            adjacent_second_shape[row_index] > candidate.SCIPLEX3_CANDIDATE_INNER_RESIDUAL_TOL
            or adjacent_second_elog[row_index] > candidate.SCIPLEX3_CANDIDATE_INNER_RESIDUAL_TOL
        )
        and d2_shape[row_index] <= candidate.SCIPLEX3_CANDIDATE_INNER_RESIDUAL_TOL
        and d2_elog[row_index] <= candidate.SCIPLEX3_CANDIDATE_INNER_RESIDUAL_TOL
        and cosine is not None
        and cosine <= -1.0 + candidate.SCIPLEX3_CANDIDATE_INNER_RESIDUAL_TOL
    ]
    row_index = qualifying[0] if qualifying else int(np.argmin(d2_elog))
    return {
        "A49_array_equal_A51": bool(np.array_equal(a49, a51)),
        "A49_shape_sha256": _canonical_array_sha256(a49, np),
        "A50_shape_sha256": _canonical_array_sha256(a50, np),
        "A51_shape_sha256": _canonical_array_sha256(a51, np),
        "adjacent_second_Relog": _float_manifest(float(adjacent_second_elog[row_index])),
        "adjacent_second_Rshape": _float_manifest(float(adjacent_second_shape[row_index])),
        "batch_row_index": row_index,
        "cosine": (
            None if cosines[row_index] is None else _float_manifest(cast(float, cosines[row_index]))
        ),
        "one_step_Relog": _float_manifest(float(d1_elog[row_index])),
        "one_step_Rshape": _float_manifest(float(d1_shape[row_index])),
        "same_row_strict_two_cycle": bool(qualifying),
        "two_step_Relog": _float_manifest(float(d2_elog[row_index])),
        "two_step_Rshape": _float_manifest(float(d2_shape[row_index])),
        "well_row_index": replay_input.batch_start + row_index,
    }


def _scientific_route_branches(
    *,
    core_replay_valid: bool,
    objective_material_decrease: bool,
    objective_ambiguous: bool,
    strict_two_cycle: bool,
    jacobian_noncontractive: bool,
    jacobian_ambiguous: bool,
    context_collapsed_or_tiny: bool,
    context_ambiguous: bool,
) -> list[str]:
    if not core_replay_valid:
        return ["invalid-or-inexact-replay-no-conclusion"]
    branches: list[str] = []
    if objective_material_decrease:
        branches.append("implementation-fix")
    if (
        not objective_material_decrease
        and not objective_ambiguous
        and (strict_two_cycle or jacobian_noncontractive)
    ):
        branches.append("v5-safeguarded-local-solver")
    if context_collapsed_or_tiny:
        branches.append("v5-plate-regularization-or-neutralization")
    if not branches:
        branches.append("inconclusive")
    if objective_ambiguous or jacobian_ambiguous or context_ambiguous:
        branches.append("ambiguity-no-conclusion-for-affected-gate")
    return branches


def _replay_local_map_diagnostic(
    replay_input: _LocalMapReplayInput,
    candidate: ModuleType,
) -> dict[str, object]:
    np = candidate.np
    special = importlib.import_module("scipy.special")
    polygamma_function = getattr(special, "polygamma", None)
    if not callable(polygamma_function):
        raise TrajectoryError("reference SciPy special module lacks polygamma")
    batch_size = replay_input.batch_stop - replay_input.batch_start
    factor_count = candidate.SCIPLEX3_CANDIDATE_FACTOR_COUNT
    initial_shape = np.full(
        (batch_size, factor_count),
        candidate.SCIPLEX3_CANDIDATE_FIXED_FACTOR_SHAPE,
        dtype=np.float64,
    )
    exact_initial = np.full_like(initial_shape, float.fromhex(EXPECTED_FIXED_FACTOR_SHAPE_HEX))
    if not np.array_equal(initial_shape, exact_initial):
        raise TrajectoryError("local-map replay does not start at bit-exact 0.1")
    current = initial_shape
    sweeps: list[dict[str, object]] = []
    jacobian_blocks: dict[int, list[object]] = {}
    jacobian_summaries: list[dict[str, object]] = []
    previous_post_objective: float | None = None
    previous_post_objective_fsum: float | None = None
    objective_diagnostics: list[dict[str, object]] = []
    production_sweep_50: dict[str, object] | None = None
    strict_two_cycle: dict[str, object] | None = None
    terminal_state_match = False
    evaluation = _evaluate_local_map(replay_input, current, candidate)
    map_evaluation_count = 1
    terminal_evaluation: _LocalMapEvaluation | None = None
    for sweep in range(1, candidate.SCIPLEX3_CANDIDATE_MAX_INNER_SWEEPS + 1):
        two_step = _evaluate_local_map(replay_input, evaluation.proposal_shape, candidate)
        map_evaluation_count += 1
        shape_residual, elog_residual = _map_residuals(
            current, evaluation.proposal_shape, candidate
        )
        one_absolute, one_relative = _relative_distance(current, evaluation.proposal_shape, np)
        two_absolute, two_relative = _relative_distance(current, two_step.proposal_shape, np)
        cosine, sign = _update_cosine_sign(
            current, evaluation.proposal_shape, two_step.proposal_shape, np
        )
        two_step_distance = {
            "absolute_l2": _float_manifest(two_absolute),
            "elog_max": _float_manifest(
                _elog_distance(current, two_step.proposal_shape, candidate)
            ),
            "relative_l2": _float_manifest(two_relative),
            "shape_componentwise_max": _float_manifest(
                _componentwise_shape_distance(current, two_step.proposal_shape, np)
            ),
        }
        per_row_residual_summary = _per_row_residual_summary(
            replay_input,
            current,
            evaluation.proposal_shape,
            two_step.proposal_shape,
            candidate,
        )
        shape_objective_change = _objective_change_diagnostic(
            evaluation.pre_shape_objective,
            evaluation.post_coordinate_objective,
            evaluation.pre_shape_objective_fsum,
            evaluation.post_coordinate_objective_fsum,
            candidate,
        )
        allocation_objective_change = (
            None
            if previous_post_objective is None or previous_post_objective_fsum is None
            else _objective_change_diagnostic(
                previous_post_objective,
                evaluation.pre_shape_objective,
                previous_post_objective_fsum,
                evaluation.pre_shape_objective_fsum,
                candidate,
            )
        )
        total_post_to_post_change = (
            None
            if previous_post_objective is None or previous_post_objective_fsum is None
            else _objective_change_diagnostic(
                previous_post_objective,
                evaluation.post_coordinate_objective,
                previous_post_objective_fsum,
                evaluation.post_coordinate_objective_fsum,
                candidate,
            )
        )
        previous_post_objective = evaluation.post_coordinate_objective
        previous_post_objective_fsum = evaluation.post_coordinate_objective_fsum
        objective_diagnostics.append(shape_objective_change)
        if allocation_objective_change is not None:
            objective_diagnostics.append(allocation_objective_change)
        if total_post_to_post_change is not None:
            objective_diagnostics.append(total_post_to_post_change)
        invariants = _mass_invariants(
            replay_input,
            current,
            evaluation,
            candidate,
            input_state_index=sweep - 1,
        )
        entry = {
            "Rshape": _float_manifest(shape_residual),
            "Relog": _float_manifest(elog_residual),
            "allocation_sha256": _canonical_array_sha256(evaluation.allocations, np),
            "allocated_counts_sha256": _canonical_array_sha256(evaluation.allocated_counts, np),
            "allocation_coordinate_objective_change": allocation_objective_change,
            "invariants": invariants,
            "post_coordinate_objective": _float_manifest(evaluation.post_coordinate_objective),
            "post_coordinate_objective_fsum": _float_manifest(
                evaluation.post_coordinate_objective_fsum
            ),
            "pre_shape_objective": _float_manifest(evaluation.pre_shape_objective),
            "pre_shape_objective_fsum": _float_manifest(evaluation.pre_shape_objective_fsum),
            "one_step_distance": {
                "absolute_l2": _float_manifest(one_absolute),
                "elog_max": _float_manifest(elog_residual),
                "relative_l2": _float_manifest(one_relative),
                "shape_componentwise_max": _float_manifest(shape_residual),
            },
            "outgoing_proposal_shape_sha256": _canonical_array_sha256(
                evaluation.proposal_shape, np
            ),
            "responsibility_sha256": _canonical_array_sha256(evaluation.responsibilities, np),
            "batch_rate_sha256": _canonical_array_sha256(replay_input.batch_rate, np),
            "shape_sha256": _canonical_array_sha256(current, np),
            "state_sha256": _combined_state_sha256(current, replay_input.batch_rate, np),
            "sweep": sweep,
            "input_state_index": sweep - 1,
            "output_state_index": sweep,
            "shape_coordinate_objective_change": shape_objective_change,
            "synchronized_local_variational_objective": _float_manifest(
                evaluation.post_coordinate_objective
            ),
            "synchronized_allocation_state_sha256": _combined_allocation_sha256(
                evaluation.allocated_counts,
                evaluation.responsibilities,
                evaluation.allocations,
                np,
            ),
            "total_post_to_post_objective_change": total_post_to_post_change,
            "two_step_distance": two_step_distance,
            "update_cosine": None if cosine is None else _float_manifest(cosine),
            "update_dot_sign": sign,
            "per_row_residual_summary": per_row_residual_summary,
            "worst_row_factor": _worst_row_factor_summary(
                replay_input, current, evaluation, candidate
            ),
        }
        sweeps.append(entry)
        blocks = _row_jacobians(
            replay_input,
            current,
            evaluation,
            candidate,
            polygamma_function,
        )
        if sweep == candidate.SCIPLEX3_CANDIDATE_MAX_INNER_SWEEPS:
            jacobian_blocks[49] = blocks
        jacobian_summary = _maximum_row_jacobian_summary(
            replay_input,
            current,
            blocks,
            candidate,
            polygamma_function,
            sweep=sweep,
        )
        jacobian_summary["input_state_index"] = sweep - 1
        jacobian_summaries.append(jacobian_summary)
        if sweep == candidate.SCIPLEX3_CANDIDATE_MAX_INNER_SWEEPS:
            exact_allocated = np.array_equal(
                evaluation.allocated_counts, replay_input.traceback_allocated_counts
            )
            exact_responsibilities = np.array_equal(
                evaluation.responsibilities, replay_input.traceback_responsibilities
            )
            exact_allocations = np.array_equal(
                evaluation.allocations, replay_input.traceback_allocations
            )
            exact_proposal = np.array_equal(
                evaluation.proposal_shape, replay_input.traceback_proposal_shape
            )
            exact_terminal = np.array_equal(
                evaluation.proposal_shape, replay_input.traceback_state_terminal_shape
            )
            exact_post_assignment_current = np.array_equal(
                evaluation.proposal_shape, replay_input.traceback_current_shape
            )
            residual_match = bool(
                shape_residual.hex() == EXPECTED_ATTEMPT2_TERMINAL_SHAPE_RESIDUAL_HEX
                and elog_residual.hex() == EXPECTED_ATTEMPT2_TERMINAL_ELOG_RESIDUAL_HEX
            )
            production_sweep_50 = {
                "all_exact": bool(
                    exact_allocated
                    and exact_responsibilities
                    and exact_allocations
                    and exact_proposal
                    and exact_terminal
                    and exact_post_assignment_current
                    and residual_match
                ),
                "allocated_counts_exact": exact_allocated,
                "allocated_counts_sha256": _canonical_array_sha256(evaluation.allocated_counts, np),
                "allocation_exact": exact_allocations,
                "allocation_sha256": _canonical_array_sha256(evaluation.allocations, np),
                "pre_update_current_shape_sha256": _canonical_array_sha256(current, np),
                "post_assignment_current_shape_exact": exact_post_assignment_current,
                "post_assignment_current_shape_sha256": _canonical_array_sha256(
                    evaluation.proposal_shape, np
                ),
                "proposal_exact": exact_proposal,
                "proposal_shape_sha256": _canonical_array_sha256(evaluation.proposal_shape, np),
                "residual_hex_exact": residual_match,
                "responsibility_exact": exact_responsibilities,
                "responsibility_sha256": _canonical_array_sha256(evaluation.responsibilities, np),
                "state_terminal_shape_exact": exact_terminal,
                "traceback_current_is_post_assignment_alias": (
                    replay_input.traceback_current_aliases_terminal_state
                ),
                "traceback_current_shape_sha256": replay_input.traceback_hashes[
                    "current_shape_sha256"
                ],
            }
            strict_two_cycle = _strict_two_cycle_summary(
                replay_input,
                current,
                evaluation.proposal_shape,
                two_step.proposal_shape,
                candidate,
            )
        current = np.asarray(evaluation.proposal_shape, dtype=np.float64, order="C").copy()
        if sweep == candidate.SCIPLEX3_CANDIDATE_MAX_INNER_SWEEPS:
            terminal_state_match = bool(
                np.array_equal(current, replay_input.traceback_current_shape)
                and np.array_equal(current, replay_input.traceback_proposal_shape)
                and np.array_equal(current, replay_input.traceback_state_terminal_shape)
            )
            terminal_evaluation = two_step
        else:
            evaluation = two_step
    if production_sweep_50 is None or strict_two_cycle is None or terminal_evaluation is None:
        raise TrajectoryError("local-map replay omitted frozen production sweep 50 diagnostics")
    terminal_blocks = _row_jacobians(
        replay_input,
        current,
        terminal_evaluation,
        candidate,
        polygamma_function,
    )
    jacobian_blocks[50] = terminal_blocks
    terminal_jacobian_summary = _maximum_row_jacobian_summary(
        replay_input,
        current,
        terminal_blocks,
        candidate,
        polygamma_function,
        sweep=50,
    )
    terminal_jacobian_summary.pop("sweep")
    terminal_jacobian_summary["state_index"] = 50
    terminal_jacobian_summary["production_sweep"] = None
    terminal_jacobian_summary["shape_sha256"] = _canonical_array_sha256(current, np)
    if set(jacobian_blocks) != set(TERMINAL_JACOBIAN_STATE_INDICES):
        raise TrajectoryError("local-map replay omitted a declared Jacobian state")
    two_step_jacobian = _two_step_jacobian_summary(
        replay_input,
        jacobian_blocks[49],
        jacobian_blocks[50],
        candidate,
    )
    diagnostic_lookahead = {
        "allocated_counts_sha256": _canonical_array_sha256(
            terminal_evaluation.allocated_counts, np
        ),
        "allocation_sha256": _canonical_array_sha256(terminal_evaluation.allocations, np),
        "input_shape_sha256": _canonical_array_sha256(current, np),
        "input_state_index": 50,
        "output_state_index": 51,
        "proposal_A51_shape_sha256": _canonical_array_sha256(
            terminal_evaluation.proposal_shape, np
        ),
        "responsibility_sha256": _canonical_array_sha256(terminal_evaluation.responsibilities, np),
    }
    del evaluation, terminal_evaluation
    rho_diagnostics = _rho_effective_context_diagnostics(replay_input, candidate)
    all_invariants = all(cast(dict[str, object], item["invariants"])["all_pass"] for item in sweeps)
    objective_ambiguous = any(
        cast(bool, item["boundary_or_summation_ambiguous"]) for item in objective_diagnostics
    )
    objective_material_decrease = any(
        cast(bool, item["production_material_decrease"])
        and cast(bool, item["fsum_material_decrease"])
        and not cast(bool, item["boundary_or_summation_ambiguous"])
        for item in objective_diagnostics
    )
    terminal_radius = cast(dict[str, object], terminal_jacobian_summary["radius"])
    jacobian_noncontractive = (
        any(
            cast(dict[str, object], item["radius"])["noncontractive"] is True
            and cast(dict[str, object], item["radius"])["methods_agree_within_ambiguity"] is True
            for item in jacobian_summaries
        )
        or (
            terminal_radius["noncontractive"] is True
            and terminal_radius["methods_agree_within_ambiguity"] is True
        )
        or two_step_jacobian["noncontractive"] is True
    )
    jacobian_ambiguous = (
        any(
            cast(dict[str, object], item["radius"])["spectral_status"] == "ambiguous"
            or cast(dict[str, object], item["radius"])["methods_agree_within_ambiguity"] is not True
            for item in jacobian_summaries
        )
        or terminal_radius["spectral_status"] == "ambiguous"
        or terminal_radius["methods_agree_within_ambiguity"] is not True
        or two_step_jacobian["spectral_status"] == "ambiguous"
    )
    context_collapsed = any(
        cast(bool, item["collapsed"]) and cast(bool, item["gates_pass"]) for item in rho_diagnostics
    )
    context_tiny = any(
        cast(bool, item["tiny_normalized_share"]) and cast(bool, item["gates_pass"])
        for item in rho_diagnostics
    )
    rho_gates_all_pass = all(cast(bool, item["gates_pass"]) for item in rho_diagnostics)
    context_ambiguous = any(
        cast(bool, item["collapse_boundary_ambiguous"]) for item in rho_diagnostics
    )
    row_for_entry = np.asarray(replay_input.row_for_entry, dtype=np.int64)
    replay_values = np.asarray(replay_input.values, dtype=np.float64)
    row_nnz = np.bincount(row_for_entry, minlength=batch_size)
    row_umi = np.bincount(row_for_entry, weights=replay_values, minlength=batch_size)
    map_state_chain_exact = bool(
        all(
            cast(str, sweeps[index]["outgoing_proposal_shape_sha256"])
            == cast(str, sweeps[index + 1]["shape_sha256"])
            for index in range(len(sweeps) - 1)
        )
        and cast(str, sweeps[-1]["outgoing_proposal_shape_sha256"])
        == cast(str, terminal_jacobian_summary["shape_sha256"])
        == cast(str, diagnostic_lookahead["input_shape_sha256"])
        and cast(str, diagnostic_lookahead["proposal_A51_shape_sha256"])
        == cast(str, strict_two_cycle["A51_shape_sha256"])
    )
    core_replay_valid = bool(
        production_sweep_50["all_exact"] is True
        and terminal_state_match
        and replay_input.copies_isolated
        and all_invariants
        and map_evaluation_count == candidate.SCIPLEX3_CANDIDATE_MAX_INNER_SWEEPS + 1
        and map_state_chain_exact
        and rho_gates_all_pass
    )
    scientific_branches = _scientific_route_branches(
        core_replay_valid=core_replay_valid,
        objective_material_decrease=objective_material_decrease,
        objective_ambiguous=objective_ambiguous,
        strict_two_cycle=strict_two_cycle["same_row_strict_two_cycle"] is True,
        jacobian_noncontractive=jacobian_noncontractive,
        jacobian_ambiguous=jacobian_ambiguous,
        context_collapsed_or_tiny=context_tiny or context_collapsed,
        context_ambiguous=context_ambiguous,
    )
    diagnostic = {
        "bounded_failing_batch": {
            "batch_index": replay_input.batch_index,
            "batch_row_count": batch_size,
            "batch_start": replay_input.batch_start,
            "batch_stop": replay_input.batch_stop,
            "global_record_start": replay_input.global_start,
            "global_record_stop": replay_input.global_stop,
            "nonzero_entry_count": int(replay_values.size),
            "row_nnz_maximum": int(np.max(row_nnz)),
            "row_nnz_minimum": int(np.min(row_nnz)),
            "row_umi_maximum": int(np.max(row_umi)),
            "row_umi_minimum": int(np.min(row_umi)),
            "umi_total": int(math.fsum(float(value) for value in replay_values)),
            "well_index": replay_input.well_index,
            "well_row_count": replay_input.well_row_count,
        },
        "definitions": {
            "jacobian": (
                "analytic J=H*diag(trigamma(A)); one-step radius uses the similar symmetric "
                "sqrt(D)*H*sqrt(D), cross-checked against generic eigenvalues"
            ),
            "local_variational_objective": (
                "fixed-global augmented Gamma-Poisson local ELBO; each map records "
                "for sweep s=1..50 pre-shape L(A[s-1],phi_s) and post-coordinate "
                "L(A_s,phi_s)"
            ),
            "rho_effective_context_count": (
                "inverse Simpson of shares rho[plate,factor]/sum_plate(rho[:,factor])"
            ),
            "sweep_index": (
                "per_sweep entries 1..50 are production maps A[s-1]->A[s]; state 0 is "
                "bit-exact prior shape 0.1; the sole diagnostic lookahead F(A50) yields A51"
            ),
        },
        "initial_shape_all_entries_bit_exact_fixed_0p1": bool(
            np.array_equal(initial_shape, exact_initial)
        ),
        "diagnostic_lookahead_F_A50_to_A51": diagnostic_lookahead,
        "per_sweep_max_row_jacobian_spectral_radii": jacobian_summaries,
        "local_map_evaluation_count": map_evaluation_count,
        "map_state_digest_chain_exact": map_state_chain_exact,
        "objective_summary": {
            "any_ambiguity": objective_ambiguous,
            "any_material_decrease": objective_material_decrease,
            "comparison_count": len(objective_diagnostics),
        },
        "per_sweep": sweeps,
        "plate_contexts": rho_diagnostics,
        "production_sweep_50_replay": production_sweep_50,
        "scientific_route_indicators": {
            "branches": scientific_branches,
            "cap_or_tolerance_relaxation_authorized": False,
            "context_collapse_or_tiny_share": context_collapsed or context_tiny,
            "jacobian_noncontractive": jacobian_noncontractive,
            "objective_material_decrease": objective_material_decrease,
            "strict_same_row_two_cycle": strict_two_cycle["same_row_strict_two_cycle"],
        },
        "strict_two_cycle": strict_two_cycle,
        "terminal_state_jacobian_spectral_radius": terminal_jacobian_summary,
        "terminal_state_exact_traceback_current_proposal_and_state_slice": terminal_state_match,
        "traceback_copy_manifest": {
            "copies_isolated": replay_input.copies_isolated,
            "hashes": dict(replay_input.traceback_hashes),
        },
        "two_step_row_jacobian_spectral_radius": two_step_jacobian,
    }
    diagnostic["pass"] = bool(
        diagnostic["initial_shape_all_entries_bit_exact_fixed_0p1"]
        and all_invariants
        and map_evaluation_count == candidate.SCIPLEX3_CANDIDATE_MAX_INNER_SWEEPS + 1
        and map_state_chain_exact
        and len(sweeps) == candidate.SCIPLEX3_CANDIDATE_MAX_INNER_SWEEPS
        and [cast(int, item["sweep"]) for item in sweeps]
        == list(range(1, candidate.SCIPLEX3_CANDIDATE_MAX_INNER_SWEEPS + 1))
        and [cast(int, item["input_state_index"]) for item in sweeps]
        == list(range(candidate.SCIPLEX3_CANDIDATE_MAX_INNER_SWEEPS))
        and [cast(int, item["output_state_index"]) for item in sweeps]
        == list(range(1, candidate.SCIPLEX3_CANDIDATE_MAX_INNER_SWEEPS + 1))
        and len(jacobian_summaries) == candidate.SCIPLEX3_CANDIDATE_MAX_INNER_SWEEPS
        and [cast(int, item["sweep"]) for item in jacobian_summaries]
        == list(range(1, candidate.SCIPLEX3_CANDIDATE_MAX_INNER_SWEEPS + 1))
        and [cast(int, item["input_state_index"]) for item in jacobian_summaries]
        == list(range(candidate.SCIPLEX3_CANDIDATE_MAX_INNER_SWEEPS))
        and terminal_jacobian_summary["state_index"]
        == candidate.SCIPLEX3_CANDIDATE_MAX_INNER_SWEEPS
        and terminal_jacobian_summary["production_sweep"] is None
        and len(rho_diagnostics) == candidate.SCIPLEX3_CANDIDATE_FACTOR_COUNT
        and rho_gates_all_pass
        and production_sweep_50["all_exact"] is True
        and terminal_state_match
        and replay_input.copies_isolated
    )
    return diagnostic


def _consume_item12_1_initial_failure(
    outer_error: BaseException,
    candidate: ModuleType,
    runner: ModuleType,
) -> tuple[dict[str, object], dict[str, object]]:
    outer_traceback = outer_error.__traceback__
    cause = outer_error.__cause__
    cause_traceback = None if cause is None else cause.__traceback__
    replay_input: _LocalMapReplayInput | None = None
    failure: dict[str, object] | None = None
    baseline: dict[str, object] | None = None
    fit_frames: list[Any] = []
    cavi_frames: list[Any] = []
    try:
        if (
            type(outer_error) is not runner.SciPlex3CandidateRunnerError
            or str(outer_error) != "exact candidate fitting failed closed"
            or type(cause) is not candidate.SciPlex3CandidateError
            or str(cause) != "candidate inner CAVI failed to converge within 50 sweeps"
        ):
            raise TrajectoryError("Item 12.1 lacks the exact runner/initial-inner cause chain")
        fit_code = candidate._fit_sciplex3_candidate_exact.__code__
        cavi_code = candidate._cavi_pass.__code__
        current_traceback = cause_traceback
        while current_traceback is not None:
            if current_traceback.tb_frame.f_code is fit_code:
                fit_frames.append(current_traceback.tb_frame)
            if current_traceback.tb_frame.f_code is cavi_code:
                cavi_frames.append(current_traceback.tb_frame)
            current_traceback = current_traceback.tb_next
        if len(fit_frames) != 1 or len(cavi_frames) != 1:
            raise TrajectoryError("Item 12.1 requires one exact fit and one exact CAVI frame")
        replay_input, baseline = _capture_local_map_replay_input(
            fit_frames[0].f_locals,
            cavi_frames[0].f_locals,
            candidate,
        )
        failure = {
            "cause": _exception_identity(cause),
            "exact_cavi_code_frame_count": 1,
            "exact_fit_code_frame_count": 1,
            "kind": "inner-nonconvergence",
            "outer": _exception_identity(outer_error),
            "stage": "initial-untraced-equilibration",
        }
    finally:
        if cause_traceback is not None:
            traceback_module.clear_frames(cause_traceback)
        if outer_traceback is not None:
            traceback_module.clear_frames(outer_traceback)
        if cause is not None:
            cause.__traceback__ = None
            cause.__cause__ = None
            cause.__context__ = None
        outer_error.__traceback__ = None
        outer_error.__cause__ = None
        outer_error.__context__ = None
    if replay_input is None or failure is None or baseline is None:
        raise TrajectoryError("Item 12.1 failure capture did not complete")
    traceback_cleared = bool(
        outer_error.__traceback__ is None
        and outer_error.__cause__ is None
        and outer_error.__context__ is None
        and (
            cause is None
            or (
                cause.__traceback__ is None
                and cause.__cause__ is None
                and cause.__context__ is None
            )
        )
        and all(not frame.f_locals for frame in (*fit_frames, *cavi_frames))
    )
    if not traceback_cleared:
        raw_copies_destroyed = _wipe_replay_input(replay_input, candidate.np)
        if not raw_copies_destroyed:
            raise TrajectoryError(
                "traceback cleanup proof failed before replay and private copies were not wiped"
            )
        raise TrajectoryError("traceback cleanup proof failed before local-map replay")
    try:
        diagnostic = _replay_local_map_diagnostic(replay_input, candidate)
        diagnostic["attempt2_baseline"] = baseline
        diagnostic["tracebacks_and_causes_cleared_before_replay"] = traceback_cleared
        diagnostic["pass"] = bool(diagnostic["pass"] and traceback_cleared)
    finally:
        raw_copies_destroyed = _wipe_replay_input(replay_input, candidate.np)
    diagnostic["raw_replay_copies_destroyed_before_report"] = raw_copies_destroyed
    diagnostic["pass"] = bool(diagnostic["pass"] and raw_copies_destroyed)
    if not raw_copies_destroyed:
        route = cast(dict[str, object], diagnostic["scientific_route_indicators"])
        route["branches"] = ["invalid-containment-no-scientific-conclusion"]
    return failure, diagnostic


def _consume_fit_failure(
    outer_error: BaseException,
    candidate: ModuleType,
    runner: ModuleType,
) -> tuple[
    dict[str, object],
    dict[str, object] | None,
    list[dict[str, object]],
    dict[str, object],
]:
    outer_traceback = outer_error.__traceback__
    cause = outer_error.__cause__
    cause_traceback = None if cause is None else cause.__traceback__
    try:
        if (
            type(outer_error) is not runner.SciPlex3CandidateRunnerError
            or str(outer_error) != "exact candidate fitting failed closed"
            or type(cause) is not candidate.SciPlex3CandidateError
        ):
            raise TrajectoryError("fit failure lacks the exact runner/candidate cause chain")
        failure_kind = EXPECTED_SCIENTIFIC_FAILURE_MESSAGES.get(str(cause))
        if failure_kind is None:
            raise TrajectoryError("candidate error is not a frozen expected scientific failure")
        fit_code = candidate._fit_sciplex3_candidate_exact.__code__
        fit_frames = []
        cavi_frames = []
        cavi_code = candidate._cavi_pass.__code__
        current = cause_traceback
        while current is not None:
            if current.tb_frame.f_code is fit_code:
                fit_frames.append(current.tb_frame)
            if current.tb_frame.f_code is cavi_code:
                cavi_frames.append(current.tb_frame)
            current = current.tb_next
        if len(fit_frames) != 1:
            raise TrajectoryError("candidate cause lacks exactly one exact-fit traceback frame")
        frame_locals = fit_frames[0].f_locals
        initial_raw = frame_locals.get("initial_equilibration")
        initial = (
            None
            if initial_raw is None
            else _copy_inner_witness(
                initial_raw,
                candidate.SciPlex3CandidateInitialEquilibration,
            )
        )
        raw_trace = frame_locals.get("trace", [])
        trace = _copy_trace(raw_trace, candidate.SciPlex3CandidateTraceEntry)
        fit_iteration = frame_locals.get("iteration")
        if failure_kind == "inner-nonconvergence":
            if len(cavi_frames) != 1:
                raise TrajectoryError(
                    "inner nonconvergence lacks exactly one exact CAVI traceback frame"
                )
            cavi_locals = cavi_frames[0].f_locals
            sweep_count = cavi_locals.get("sweep_count")
            passing_streak = cavi_locals.get("passing_streak")
            shape_residual = cavi_locals.get("terminal_shape_residual")
            elog_residual = cavi_locals.get("terminal_elog_residual")
            inner_cap_proof = bool(
                type(sweep_count) is int
                and sweep_count == candidate.SCIPLEX3_CANDIDATE_MAX_INNER_SWEEPS
                and type(passing_streak) is int
                and passing_streak < candidate.SCIPLEX3_CANDIDATE_INNER_CONVERGENCE_STREAK
                and type(shape_residual) is float
                and math.isfinite(shape_residual)
                and type(elog_residual) is float
                and math.isfinite(elog_residual)
            )
            if initial is None:
                stage_name = "initial-untraced-equilibration"
                schedule_proof = not trace and fit_iteration is None
            else:
                stage_name = "traced-outer-e-step-before-append"
                schedule_proof = bool(
                    type(fit_iteration) is int
                    and fit_iteration == len(trace) + 1
                    and 1 <= fit_iteration <= candidate.SCIPLEX3_CANDIDATE_MAX_OUTER_ITERATIONS
                )
            stage_proof: dict[str, object] = {
                "failed_outer_iteration": (fit_iteration if type(fit_iteration) is int else None),
                "inner_passing_streak": (passing_streak if type(passing_streak) is int else None),
                "inner_sweep_cap_reached": inner_cap_proof,
                "inner_sweep_count": sweep_count if type(sweep_count) is int else None,
                "pass": inner_cap_proof and schedule_proof,
                "prior_trace_entry_count": len(trace),
                "stage": stage_name,
                "terminal_elog_residual": (
                    _float_manifest(elog_residual)
                    if type(elog_residual) is float and math.isfinite(elog_residual)
                    else None
                ),
                "terminal_shape_residual": (
                    _float_manifest(shape_residual)
                    if type(shape_residual) is float and math.isfinite(shape_residual)
                    else None
                ),
            }
        elif failure_kind == "outer-nonconvergence":
            stage_proof = {
                "failed_outer_iteration": (fit_iteration if type(fit_iteration) is int else None),
                "pass": bool(
                    type(fit_iteration) is int
                    and fit_iteration == candidate.SCIPLEX3_CANDIDATE_MAX_OUTER_ITERATIONS
                    and len(trace) == candidate.SCIPLEX3_CANDIDATE_MAX_OUTER_ITERATIONS
                ),
                "prior_trace_entry_count": len(trace),
                "stage": "outer-cap-after-final-trace-append",
            }
        elif failure_kind == "material-elbo-decrease":
            stage_proof = {
                "failed_outer_iteration": (fit_iteration if type(fit_iteration) is int else None),
                "pass": bool(
                    initial is not None
                    and type(fit_iteration) is int
                    and fit_iteration == len(trace) + 1
                ),
                "prior_trace_entry_count": len(trace),
                "stage": "outer-objective-gate-before-trace-append",
            }
        else:
            terminal = trace[-candidate.SCIPLEX3_CANDIDATE_CONVERGENCE_STREAK :]
            terminal_convergence_pass = bool(
                len(trace) >= candidate.SCIPLEX3_CANDIDATE_MIN_OUTER_ITERATIONS
                and len(terminal) == candidate.SCIPLEX3_CANDIDATE_CONVERGENCE_STREAK
                and all(
                    cast(float, item["relative_change"])
                    <= candidate.SCIPLEX3_CANDIDATE_CONVERGENCE_RTOL
                    for item in terminal
                )
                and len({tuple(cast(list[int], item["factor_order"])) for item in terminal}) == 1
            )
            post_canonical_final = {
                "basis",
                "contributions",
                "mean_activation",
            }.issubset(frame_locals)
            pre_canonical_final = {
                "basis",
                "contributions",
            }.issubset(frame_locals)
            current_outer = {
                "current_basis",
                "current_contributions",
                "well_factor_means",
            }.issubset(frame_locals)
            initial_key_gate = bool(
                initial is None
                and not trace
                and fit_iteration is None
                and {"initial_basis_mean", "well_factor_means"}.issubset(frame_locals)
            )
            if post_canonical_final:
                stage_name = "post-canonical-final-private-gate"
                schedule_proof = terminal_convergence_pass
            elif pre_canonical_final:
                stage_name = "post-convergence-pre-canonical-factor-key-gate"
                schedule_proof = terminal_convergence_pass
            elif current_outer:
                stage_name = "traced-outer-factor-key-gate-before-append"
                schedule_proof = bool(
                    initial is not None
                    and type(fit_iteration) is int
                    and fit_iteration == len(trace) + 1
                )
            elif initial_key_gate:
                stage_name = "initial-untraced-factor-key-gate"
                schedule_proof = True
            else:
                stage_name = "unresolved-scientific-gate-stage"
                schedule_proof = False
            stage_proof = {
                "failed_outer_iteration": (fit_iteration if type(fit_iteration) is int else None),
                "pass": schedule_proof,
                "prior_trace_entry_count": len(trace),
                "stage": stage_name,
                "terminal_convergence_pass": terminal_convergence_pass,
            }
        aggregates = _provisional_failure_aggregates(
            frame_locals,
            trace,
            candidate,
            failure_kind=failure_kind,
            failure_stage=cast(str, stage_proof["stage"]),
        )
        failure = {
            "cause": _exception_identity(cause),
            "exact_fit_code_frame_count": 1,
            "kind": failure_kind,
            "outer": _exception_identity(outer_error),
            "stage_proof": stage_proof,
        }
        return failure, initial, trace, aggregates
    finally:
        if cause_traceback is not None:
            traceback_module.clear_frames(cause_traceback)
        if outer_traceback is not None:
            traceback_module.clear_frames(outer_traceback)
        if cause is not None:
            cause.__traceback__ = None
            cause.__cause__ = None
            cause.__context__ = None
        outer_error.__traceback__ = None
        outer_error.__cause__ = None
        outer_error.__context__ = None
        del cause, cause_traceback, outer_traceback


def _load_canonical_receipt(
    path: Path, *, expected_sha256: str, name: str
) -> tuple[dict[str, object], bytes]:
    payload = path.read_bytes()
    if _sha256(payload) != expected_sha256:
        raise TrajectoryError(f"{name} content identity drifted")
    return _canonical_json_object(payload, name=name), payload


def _pre_source_closure(
    repository_root: Path,
    materializer: ModuleType,
    candidate: ModuleType,
    runner: ModuleType,
    item11_checker: ModuleType,
) -> tuple[dict[str, object], bytes, dict[str, object]]:
    if materializer.FIT_WALL_LIMIT_SECONDS != FIT_WALL_LIMIT_SECONDS:
        raise TrajectoryError("materializer wall limit drifted")
    if materializer.FIT_RSS_LIMIT_BYTES != FIT_RSS_LIMIT_BYTES:
        raise TrajectoryError("materializer RSS limit drifted")
    for key in THREAD_ENVIRONMENT_KEYS:
        if os.environ.get(key) != "1":
            raise TrajectoryError(f"thread environment is not exactly one: {key}")
    observed_repository = {
        name: _file_sha256(repository_root / relative_path)
        for name, relative_path in REPOSITORY_PATHS.items()
    }
    if observed_repository != dict(EXPECTED_REPOSITORY_SHA256):
        raise TrajectoryError("frozen repository/source-free receipt closure drifted")
    if item11_checker.check_materialization() != EXPECTED_ITEM11_MATERIALIZATION_SHA256:
        raise TrajectoryError("Item 11 checked materialization fingerprint drifted")
    runtime = materializer._require_reference_runtime()
    if _sha256(runtime) != EXPECTED_RUNTIME_LOCK_SHA256:
        raise TrajectoryError("candidate runtime-lock identity drifted")
    bindings = materializer._repository_bindings(repository_root)
    materializer._verify_imported_module_provenance(repository_root, bindings)
    support_fingerprint, support_payload = materializer._planned_support_envelope(repository_root)
    if support_fingerprint != _sha256(support_payload):
        raise TrajectoryError("planned support envelope identity is inconsistent")
    if (
        _binding_digest(bindings, "candidate_code") != EXPECTED_CANDIDATE_SHA256
        or _binding_digest(bindings, "candidate_runner_code") != EXPECTED_RUNNER_SHA256
        or _binding_digest(bindings, "trained_candidate_builder_code") != EXPECTED_BUILDER_SHA256
        or _binding_digest(bindings, "materializer_code") != EXPECTED_MATERIALIZER_SHA256
        or _binding_digest(bindings, "loader_code") != EXPECTED_LOADER_SHA256
        or _binding_digest(bindings, "item11_runner_code") != EXPECTED_ITEM11_RUNNER_SHA256
        or _binding_digest(bindings, "loader_contract") != EXPECTED_LOADER_CONTRACT_SHA256
        or _binding_digest(bindings, "dataset_manifest") != EXPECTED_DATASET_MANIFEST_SHA256
    ):
        raise TrajectoryError("materializer repository binding closure drifted")
    if (
        candidate.SCIPLEX3_CANDIDATE_MODEL_ID != EXPECTED_MODEL_ID
        or candidate.SCIPLEX3_CANDIDATE_IMPLEMENTATION_VERSION != EXPECTED_VERSION
        or candidate.SCIPLEX3_CANDIDATE_MODEL_SCHEMA != EXPECTED_MODEL_SCHEMA
        or candidate.SCIPLEX3_CANDIDATE_MODEL_SCHEMA_VERSION != EXPECTED_VERSION
        or candidate.SCIPLEX3_CANDIDATE_FIXED_FACTOR_SHAPE.hex() != EXPECTED_FIXED_FACTOR_SHAPE_HEX
        or candidate.SCIPLEX3_CANDIDATE_SPECIFICATION_SHA256 != EXPECTED_SPECIFICATION_SHA256
        or candidate.SCIPLEX3_CANDIDATE_OUTPUT_MODEL_SCHEMA_SHA256 != EXPECTED_OUTPUT_SCHEMA_SHA256
        or candidate.SCIPLEX3_CANDIDATE_GOLDEN_MODEL_SHA256 != EXPECTED_GOLDEN_MODEL_SHA256
        or candidate.SCIPLEX3_CANDIDATE_GOLDEN_SAMPLE_SHA256 != EXPECTED_GOLDEN_SAMPLE_SHA256
        or runner._IMPORTED_CANDIDATE_CODE_SHA256 != EXPECTED_CANDIDATE_SHA256
        or runner._IMPORTED_RUNNER_CODE_SHA256 != EXPECTED_RUNNER_SHA256
    ):
        raise TrajectoryError("candidate v4 model/spec/schema/golden identity drifted")
    return (
        dict(bindings),
        runtime,
        {
            "golden_execution_performed": False,
            "golden_model_sha256": EXPECTED_GOLDEN_MODEL_SHA256,
            "golden_sample_sha256": EXPECTED_GOLDEN_SAMPLE_SHA256,
            "output_model_schema_sha256": EXPECTED_OUTPUT_SCHEMA_SHA256,
            "specification_sha256": EXPECTED_SPECIFICATION_SHA256,
            "support_envelope_sha256": support_fingerprint,
        },
    )


def _json_projection(value: object) -> object:
    if value is None or type(value) in (bool, int, float, str):
        return value
    if type(value) in (list, tuple):
        return [_json_projection(item) for item in cast(Sequence[object], value)]
    if type(value) is dict:
        if any(type(key) is not str for key in cast(dict[object, object], value)):
            raise TrajectoryError("receipt projection contains a non-exact string key")
        return {
            cast(str, key): _json_projection(item)
            for key, item in cast(dict[object, object], value).items()
        }
    raise TrajectoryError("receipt projection contains a non-JSON primitive")


def _runtime_neutral_receipt_comparison(
    fresh_assembly: Mapping[str, object],
    sealed_assembly: Mapping[str, object],
    fresh_scan: Mapping[str, object],
    sealed_scan: Mapping[str, object],
) -> dict[str, object]:
    assembly_fields = set(sealed_assembly) - set(ASSEMBLY_RUNTIME_DEPENDENT_FIELDS)
    scan_fields = set(sealed_scan) - set(SCAN_RUNTIME_OR_FILESYSTEM_DEPENDENT_FIELDS)
    if set(fresh_assembly) != set(sealed_assembly) or set(fresh_scan) != set(sealed_scan):
        raise TrajectoryError("fresh and sealed p1 receipt field closures differ")
    projected_fresh_assembly = {
        key: _json_projection(fresh_assembly[key]) for key in sorted(assembly_fields)
    }
    projected_sealed_assembly = {
        key: _json_projection(sealed_assembly[key]) for key in sorted(assembly_fields)
    }
    projected_fresh_scan = {key: _json_projection(fresh_scan[key]) for key in sorted(scan_fields)}
    projected_sealed_scan = {key: _json_projection(sealed_scan[key]) for key in sorted(scan_fields)}
    fresh_assembly_bytes = _canonical_json_bytes(projected_fresh_assembly)
    sealed_assembly_bytes = _canonical_json_bytes(projected_sealed_assembly)
    fresh_scan_bytes = _canonical_json_bytes(projected_fresh_scan)
    sealed_scan_bytes = _canonical_json_bytes(projected_sealed_scan)
    assembly_match = fresh_assembly_bytes == sealed_assembly_bytes
    scan_match = fresh_scan_bytes == sealed_scan_bytes
    return {
        "assembly_runtime_dependent_fields_excluded": sorted(ASSEMBLY_RUNTIME_DEPENDENT_FIELDS),
        "assembly_runtime_neutral_match": assembly_match,
        "assembly_runtime_neutral_sha256": _sha256(fresh_assembly_bytes),
        "scan_runtime_or_filesystem_dependent_fields_excluded": sorted(
            SCAN_RUNTIME_OR_FILESYSTEM_DEPENDENT_FIELDS
        ),
        "scan_runtime_neutral_match": scan_match,
        "scan_runtime_neutral_sha256": _sha256(fresh_scan_bytes),
    }


def _preparation_closure(
    preparation: object,
    repository_root: Path,
    candidate: ModuleType,
    runner: ModuleType,
) -> dict[str, object]:
    runner._item11._validate_preparation(preparation)
    assembly_path = repository_root / REPOSITORY_PATHS["item11_assembly"]
    scan_path = repository_root / REPOSITORY_PATHS["item11_scan"]
    assembly, _assembly_payload = _load_canonical_receipt(
        assembly_path,
        expected_sha256=EXPECTED_ITEM11_ASSEMBLY_SHA256,
        name="Item 11 p1 assembly receipt",
    )
    scan, _scan_payload = _load_canonical_receipt(
        scan_path,
        expected_sha256=EXPECTED_ITEM11_SCAN_SHA256,
        name="Item 11 finalized p1 scan receipt",
    )
    fresh_assembly = cast(dict[str, object], dataclasses.asdict(preparation.receipt))
    fresh_scan = cast(dict[str, object], preparation.finalized_count_scan_manifest())
    receipt_comparison = _runtime_neutral_receipt_comparison(
        fresh_assembly,
        assembly,
        fresh_scan,
        scan,
    )
    if (
        receipt_comparison["assembly_runtime_neutral_match"] is not True
        or receipt_comparison["scan_runtime_neutral_match"] is not True
    ):
        raise TrajectoryError(
            "fresh p1 preparation differs from runtime-neutral sealed Item 11 concepts"
        )
    source_receipt = preparation.finalized_count_scan_receipt
    if (
        source_receipt.partition_id != "p1-train"
        or source_receipt.accessed_partition_roles != ("p1-train",)
        or source_receipt.source_sha256 != EXPECTED_SOURCE_SHA256
        or source_receipt.source_md5 != EXPECTED_SOURCE_MD5
        or source_receipt.source_byte_count != EXPECTED_SOURCE_BYTE_COUNT
        or source_receipt.panel_count_stream_sha256 != EXPECTED_PANEL_COUNT_STREAM_SHA256
        or source_receipt.source_descriptor_identity_before
        != source_receipt.source_descriptor_identity_after
        or source_receipt.close_reverification_completed is not True
        or source_receipt.source_descriptor_reverified is not True
        or source_receipt.finalized is not True
        or source_receipt.heldout_memberships_parsed is not False
        or source_receipt.heldout_outcome_values_parsed is not False
        or source_receipt.lifecycle_evidence_issued is not False
        or source_receipt.scientifically_admissible is not False
        or preparation.receipt.heldout_memberships_read is not False
        or preparation.receipt.heldout_outcomes_read is not False
        or preparation.receipt.can_mint_lifecycle_evidence is not False
        or preparation.receipt.scientifically_admissible is not False
        or preparation.receipt.runner_panel_count_stream_sha256
        != EXPECTED_PANEL_COUNT_STREAM_SHA256
        or preparation.receipt.loader_panel_count_stream_sha256
        != EXPECTED_PANEL_COUNT_STREAM_SHA256
    ):
        raise TrajectoryError("real p1 preparation safety/provenance closure drifted")
    if (
        assembly.get("runner_panel_count_stream_sha256") != EXPECTED_PANEL_COUNT_STREAM_SHA256
        or scan.get("panel_count_stream_sha256") != EXPECTED_PANEL_COUNT_STREAM_SHA256
    ):
        raise TrajectoryError("sealed conceptual p1 count-stream identity drifted")
    return {
        "assembly_fingerprint": preparation.receipt.fingerprint,
        "candidate_training_data_sha256": candidate.training_data_fingerprint(
            preparation.training_data
        ),
        "close_reverification_completed": True,
        "conceptual_p1_count_stream_sha256": EXPECTED_PANEL_COUNT_STREAM_SHA256,
        "finalized_count_scan_fingerprint": (preparation.finalized_count_scan_receipt.fingerprint),
        "record_count": int(preparation.receipt.record_count),
        "runtime_neutral_item11_receipt_comparison": receipt_comparison,
        "source_descriptor_reverified": True,
        "source_sha256": EXPECTED_SOURCE_SHA256,
        "well_count": int(preparation.receipt.well_count),
        "zero_panel_record_count": int(preparation.receipt.zero_panel_record_count),
    }


def _module_path_pass(module: ModuleType, expected: Path) -> bool:
    path = getattr(module, "__file__", None)
    return type(path) is str and Path(path).resolve() == expected.resolve()


def _run(repository_argument: str, source_argument: str) -> tuple[dict[str, object], int]:
    phase = "argument-validation"
    driver_path = Path(__file__).resolve()
    driver_sha256_before = _file_sha256(driver_path)
    audit_state: dict[str, object] | None = None
    original_sys_path = list(sys.path)
    loaded_names = (
        "cellstate.backends.sciplex3_loader",
        "cellstate.evaluation.sciplex3_candidate",
        "cellstate.evaluation.sciplex3_candidate_runner",
        "cellstate.evaluation.sciplex3_runner",
        "scripts.materialize_sciplex3_k562_p1_baselines",
        "scripts.materialize_sciplex3_k562_p1_candidate",
    )
    try:
        repository_root = Path(repository_argument).resolve(strict=True)
        if not repository_root.is_dir():
            raise TrajectoryError("repository argument is not a directory")
        source_h5ad = Path(source_argument)
        if source_h5ad.name != EXPECTED_SOURCE_FILENAME:
            raise TrajectoryError("source argument has the wrong sealed filename")
        if any(name in sys.modules for name in loaded_names):
            raise TrajectoryError(
                "diagnostic requires a fresh process with no preloaded fit modules"
            )
        if _loaded_h5py_module_names():
            raise TrajectoryError("diagnostic requires h5py to be initially absent")
        sys.path.insert(0, str(repository_root))
        sys.path.insert(0, str(repository_root / "src"))
        audit_state = _install_no_write_or_heldout_audit()

        phase = "source-free-runtime-code-provenance-preflight"
        materializer = importlib.import_module("scripts.materialize_sciplex3_k562_p1_candidate")
        item11_checker = importlib.import_module("scripts.materialize_sciplex3_k562_p1_baselines")
        candidate = importlib.import_module("cellstate.evaluation.sciplex3_candidate")
        runner = importlib.import_module("cellstate.evaluation.sciplex3_candidate_runner")
        module_paths_pass = bool(
            _module_path_pass(
                materializer,
                repository_root / REPOSITORY_PATHS["materializer"],
            )
            and _module_path_pass(
                item11_checker,
                repository_root / REPOSITORY_PATHS["item11_materializer"],
            )
            and _module_path_pass(
                candidate,
                repository_root / REPOSITORY_PATHS["candidate"],
            )
            and _module_path_pass(
                runner,
                repository_root / REPOSITORY_PATHS["candidate_runner"],
            )
        )
        if not module_paths_pass:
            raise TrajectoryError("imported fit modules are bound to another checkout")
        bindings_before, runtime_before, software_identities = _pre_source_closure(
            repository_root,
            materializer,
            candidate,
            runner,
            item11_checker,
        )

        phase = "source-free-exact-h5py-preimport"
        h5py_module, h5py_preimport_identity = _preimport_exact_h5py(audit_state)
        h5py_cache_before_source = _require_cached_exact_h5py(
            h5py_module,
            materializer,
            audit_state,
        )
        software_identities = {
            **software_identities,
            **h5py_preimport_identity,
            **h5py_cache_before_source,
        }
        output_targets_before = _snapshot_output_targets(repository_root, materializer)

        phase = "exact-p1-source-open-assembly-close-reauthentication"
        preparation = materializer._prepare_exact_p1(source_h5ad, repository_root)
        h5py_cache_after_preparation = _require_cached_exact_h5py(
            h5py_module,
            materializer,
            audit_state,
        )
        preparation_identity = _preparation_closure(
            preparation,
            repository_root,
            candidate,
            runner,
        )
        design = runner._candidate_design(preparation)
        expected_batch_count = _expected_inner_batch_count(preparation, candidate)

        phase = "exact-v4-in-memory-fit"
        poisons = _install_nonissuance_poisons(materializer, candidate, runner)
        prior_handler = signal.getsignal(signal.SIGALRM)
        prior_timer = signal.getitimer(signal.ITIMER_REAL)
        if prior_timer != (0.0, 0.0):
            poisons.restore()
            raise TrajectoryError("diagnostic process already has a real-time interval timer")
        fitted: Any | None = None
        failure: dict[str, object] | None = None
        local_diagnostic: dict[str, object] | None = None
        fit_error: BaseException | None = None
        fit_started = time.monotonic()
        try:
            try:
                signal.signal(signal.SIGALRM, _wall_limit_handler)
                signal.setitimer(signal.ITIMER_REAL, float(FIT_WALL_LIMIT_SECONDS))
                try:
                    fitted = runner._fit_exact_candidate(preparation, design)
                except BaseException as error:
                    fit_error = error
                phase = "bounded-traceback-copy-clear-and-local-map-replay"
                if fit_error is not None:
                    failure, local_diagnostic = _consume_item12_1_initial_failure(
                        fit_error,
                        candidate,
                        runner,
                    )
                    fit_error = None
                elif fitted is not None:
                    del fitted
                    fitted = None
                    raise TrajectoryError("Item 12.1 expected the exact initial inner failure")
                else:
                    raise TrajectoryError("Item 12.1 fitter returned neither result nor failure")
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0.0)
                signal.signal(signal.SIGALRM, prior_handler)
        finally:
            poisons.restore()
        fit_elapsed_seconds = time.monotonic() - fit_started
        peak_rss_bytes = materializer._peak_rss_bytes()

        phase = "post-fit-integrity-recheck"
        bindings_after = materializer._repository_bindings(repository_root)
        runtime_after = materializer._require_reference_runtime()
        item11_after = item11_checker.check_materialization()
        output_targets_after = _snapshot_output_targets(repository_root, materializer)
        driver_sha256_after = _file_sha256(driver_path)
        h5py_cache_after_fit = _require_cached_exact_h5py(
            h5py_module,
            materializer,
            audit_state,
        )
        resource_gates = {
            "fit_capture_replay_elapsed_seconds": _float_manifest(float(fit_elapsed_seconds)),
            "fit_capture_replay_wall_limit_pass": fit_elapsed_seconds <= FIT_WALL_LIMIT_SECONDS,
            "fit_capture_replay_wall_limit_seconds": FIT_WALL_LIMIT_SECONDS,
            "peak_rss_bytes": peak_rss_bytes,
            "peak_rss_limit_bytes": FIT_RSS_LIMIT_BYTES,
            "peak_rss_limit_pass": peak_rss_bytes <= FIT_RSS_LIMIT_BYTES,
        }
        isolation = {
            "allowed_devnull_rw_open_count": cast(
                int, audit_state["allowed_devnull_rw_open_count"]
            ),
            "behavior_api_invoked_by_driver": False,
            "candidate_object_discarded": True,
            "devnull_allowance_closed_before_source": cast(
                bool, audit_state["devnull_allowance_closed_before_source"]
            ),
            "devnull_character_device_identity_pass": cast(
                bool, audit_state["devnull_character_device_identity_pass"]
            ),
            "fitted_state_api_invoked": False,
            "frozen_fitter_internal_behavior_validation_preserved": True,
            "golden_model_or_sample_executed": False,
            "heldout_artifacts_resolved": False,
            "heldout_memberships_read": False,
            "heldout_outcomes_read": False,
            "lifecycle_evidence_issued": False,
            "materializer_invoked": False,
            "model_bytes_created": False,
            "p2_calibration_read": False,
            "p3_model_selection_read": False,
            "p4_untouched_test_read": False,
            "plan_created": False,
            "prohibited_surface_poison_manifest": poisons.manifest(),
            "public_runtime_registered": False,
            "sample_api_invoked": False,
            "scientifically_admissible": False,
            "training_observation_created": False,
            "write_attempt_count": cast(int, audit_state["write_attempt_count"]),
        }
        integrity = {
            "builder_sha256": _binding_digest(bindings_after, "trained_candidate_builder_code"),
            "candidate_sha256": _binding_digest(bindings_after, "candidate_code"),
            "driver_sha256": driver_sha256_after,
            "driver_unchanged": driver_sha256_after == driver_sha256_before,
            "heldout_path_attempt_count": cast(int, audit_state["heldout_path_attempt_count"]),
            "h5py_cached_module_unchanged": bool(
                h5py_cache_before_source == h5py_cache_after_preparation == h5py_cache_after_fit
            ),
            "h5py_origin_manifest_module_count": h5py_cache_after_fit[
                "h5py_origin_manifest_module_count"
            ],
            "h5py_origin_manifest_sha256": h5py_cache_after_fit["h5py_origin_manifest_sha256"],
            "item11_checked_materialization_sha256": item11_after,
            "item11_runner_sha256": _binding_digest(bindings_after, "item11_runner_code"),
            "loader_contract_sha256": _binding_digest(bindings_after, "loader_contract"),
            "loader_sha256": _binding_digest(bindings_after, "loader_code"),
            "materializer_sha256": _binding_digest(bindings_after, "materializer_code"),
            "module_paths_pass": module_paths_pass,
            "output_targets_unchanged": output_targets_after == output_targets_before,
            "repository_bindings_unchanged": bindings_after == bindings_before,
            "runner_sha256": _binding_digest(bindings_after, "candidate_runner_code"),
            "runtime_lock_sha256": _sha256(runtime_after),
            "runtime_unchanged": runtime_after == runtime_before,
        }
        integrity_pass = bool(
            integrity["driver_unchanged"]
            and integrity["module_paths_pass"]
            and integrity["repository_bindings_unchanged"]
            and integrity["output_targets_unchanged"]
            and integrity["runtime_unchanged"]
            and integrity["h5py_cached_module_unchanged"]
            and integrity["h5py_origin_manifest_module_count"] == EXPECTED_H5PY_ORIGIN_MODULE_COUNT
            and integrity["h5py_origin_manifest_sha256"] == EXPECTED_H5PY_ORIGIN_MANIFEST_SHA256
            and integrity["runtime_lock_sha256"] == EXPECTED_RUNTIME_LOCK_SHA256
            and integrity["item11_checked_materialization_sha256"]
            == EXPECTED_ITEM11_MATERIALIZATION_SHA256
            and integrity["heldout_path_attempt_count"] == 0
            and isolation["allowed_devnull_rw_open_count"] == 1
            and isolation["devnull_allowance_closed_before_source"] is True
            and isolation["devnull_character_device_identity_pass"] is True
            and isolation["write_attempt_count"] == 0
            and cast(dict[str, object], isolation["prohibited_surface_poison_manifest"])[
                "all_call_counts_zero"
            ]
            is True
            and cast(dict[str, object], isolation["prohibited_surface_poison_manifest"])[
                "bindings_restored_by_identity"
            ]
            is True
            and resource_gates["fit_capture_replay_wall_limit_pass"]
            and resource_gates["peak_rss_limit_pass"]
        )
        expected_failure_valid = bool(
            failure is not None
            and failure.get("kind") == "inner-nonconvergence"
            and failure.get("stage") == "initial-untraced-equilibration"
            and local_diagnostic is not None
            and local_diagnostic.get("pass") is True
        )
        if expected_failure_valid and integrity_pass:
            status = "validated-item12.1b-local-map-plate-context-replay-no-artifact-issued"
            exit_code = 2
        else:
            status = "invalid-item12.1b-diagnostic-no-artifact-issued"
            exit_code = 1
        report = {
            "attempt2_driver_sha256_historical_oracle": ATTEMPT2_DRIVER_SHA256_ORACLE,
            "attempt2_report_sha256_historical_oracle": ATTEMPT2_REPORT_SHA256_ORACLE,
            "exact_attempt2_initial_failure_replayed": expected_failure_valid,
            "failure": failure,
            "input_bindings": {
                "candidate_design_fingerprint": design.fingerprint,
                "candidate_inner_batch_count": expected_batch_count,
                **preparation_identity,
            },
            "integrity": integrity,
            "integrity_pass": integrity_pass,
            "isolation_and_nonauthority": isolation,
            "local_map_plate_context_diagnostic": local_diagnostic,
            "resource_gates": resource_gates,
            "schema": DRIVER_SCHEMA,
            "schema_version": DRIVER_SCHEMA_VERSION,
            "software_identities": software_identities,
            "status": status,
        }
        del preparation, design
        return report, exit_code
    except BaseException as error:
        return (
            {
                "diagnostic_error": _exception_identity(error),
                "driver_sha256": driver_sha256_before,
                "isolation_and_nonauthority": _invalid_isolation_manifest(audit_state),
                "phase": phase,
                "schema": DRIVER_SCHEMA,
                "schema_version": DRIVER_SCHEMA_VERSION,
                "status": "diagnostic-preflight-extraction-or-integrity-failure-no-artifact-issued",
            },
            1,
        )
    finally:
        sys.path[:] = original_sys_path


def main() -> int:
    if len(sys.argv) != 3:
        report: dict[str, object] = {
            "diagnostic_error": {
                "message": (
                    "usage: item12_1_local_map_plate_context_diagnostic.py "
                    "REPOSITORY_ROOT SOURCE_H5AD"
                ),
                "type": f"{TrajectoryError.__module__}.{TrajectoryError.__qualname__}",
            },
            "schema": DRIVER_SCHEMA,
            "schema_version": DRIVER_SCHEMA_VERSION,
            "status": "diagnostic-argument-failure-no-artifact-issued",
        }
        exit_code = 64
        suppressed_stdout = ""
        suppressed_stderr = ""
    else:
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        sys.stdout = stdout_capture
        sys.stderr = stderr_capture
        try:
            report, exit_code = _run(sys.argv[1], sys.argv[2])
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr
        suppressed_stdout = stdout_capture.getvalue()
        suppressed_stderr = stderr_capture.getvalue()
    report["suppressed_dependency_stderr"] = {
        "byte_count": len(suppressed_stderr.encode()),
        "sha256": _sha256(suppressed_stderr.encode()),
    }
    report["suppressed_dependency_stdout"] = {
        "byte_count": len(suppressed_stdout.encode()),
        "sha256": _sha256(suppressed_stdout.encode()),
    }
    payload = _canonical_json_bytes(report)
    sys.stdout.buffer.write(payload + b"\n")
    sys.stdout.buffer.flush()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
