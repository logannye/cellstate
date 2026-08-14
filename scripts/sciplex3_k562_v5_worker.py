#!/usr/bin/env python3
"""Internal-only contained sci-Plex3 worker; refuses execution outside the frozen topology."""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import stat
import sys
from contextlib import suppress
from dataclasses import asdict
from pathlib import Path
from typing import Literal

from cellstate.backends.sciplex3_loader import (
    SCIPLEX3_SOURCE_BYTE_COUNT,
    SCIPLEX3_SOURCE_SHA256,
)
from cellstate.domain.common import canonical_json_bytes
from cellstate.training.execution import (
    ContainedTrainingWorkerObservation,
    ContainedTrainingWorkerTerminalReport,
    inventory_staged_training_tree,
)

_CONTAINED_SOURCE_PATH = Path("/run/cellstate/source/source.h5ad")
_CONTAINED_OUTPUT_PATH = Path("/run/cellstate/output")
_CONTAINED_REPOSITORY_ROOT = Path("/workspace")
_CONTAINED_SNAPSHOT_DIRECTORY = Path("/run/cellstate/snapshot")
_CONTAINED_TEMPORARY_DIRECTORY = Path("/run/cellstate/tmp")
_CONTAINED_EXECUTION_ID = "sciplex3-k562-v5-fit"
_CONTAINED_SNAPSHOT_MAX_BYTES = 3 * 1024**3
_CONTAINED_MEMORY_MAX_BYTES = 4 * 1024**3
_CONTAINED_PIDS_MAX = 256
_PROC_CONTROL_MAX_BYTES = 1024 * 1024


class _ReportedWorkerFailure(RuntimeError):
    """Internal signal emitted only after the sanitized terminal report is durable."""


class _SnapshotFailure(RuntimeError):
    """Fixed internal snapshot failure carrying only conservative descriptor state."""

    def __init__(
        self,
        *,
        source_access_started: bool,
        protected_source_acquired: bool,
        snapshot_descriptor: int | None,
    ) -> None:
        super().__init__("private source snapshot failed")
        self.source_access_started = source_access_started
        self.protected_source_acquired = protected_source_acquired
        self.snapshot_descriptor = snapshot_descriptor


def _read_bounded_control_text(path: Path) -> str:
    """Read one public kernel/container control file without touching protected source bytes."""

    try:
        payload = path.read_bytes()
    except OSError as error:
        raise RuntimeError("contained worker control topology is unavailable") from error
    if not payload or len(payload) > _PROC_CONTROL_MAX_BYTES or b"\x00" in payload:
        raise RuntimeError("contained worker control topology is invalid")
    try:
        return payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise RuntimeError("contained worker control topology is not text") from error


def _decode_mountinfo_path(value: str) -> str:
    for encoded, decoded in (("\\040", " "), ("\\011", "\t"), ("\\012", "\n"), ("\\134", "\\")):
        value = value.replace(encoded, decoded)
    return value


def _mountinfo_topology(payload: str) -> dict[str, tuple[frozenset[str], str]]:
    topology: dict[str, tuple[frozenset[str], str]] = {}
    for line in payload.splitlines():
        left, separator, right = line.partition(" - ")
        left_fields = left.split()
        right_fields = right.split()
        if not separator or len(left_fields) < 6 or len(right_fields) < 3:
            raise RuntimeError("contained worker mount topology is malformed")
        mount_point = _decode_mountinfo_path(left_fields[4])
        if mount_point in topology:
            raise RuntimeError("contained worker mount topology has duplicate targets")
        topology[mount_point] = (frozenset(left_fields[5].split(",")), right_fields[0])
    return topology


def _verify_contained_worker_topology() -> None:
    """Fail closed unless the official Docker topology and cgroup policy are active.

    This is a confinement check, not a defense against a hostile root host, which can fabricate
    procfs and mount state.  The authorized host supervisor remains the authority boundary.
    """

    if sys.platform != "linux":
        raise RuntimeError("contained worker requires the approved Linux runtime")
    try:
        docker_marker = Path("/.dockerenv").lstat()
    except OSError as error:
        raise RuntimeError("contained worker is outside Docker") from error
    if not stat.S_ISREG(docker_marker.st_mode):
        raise RuntimeError("contained worker Docker marker is invalid")

    topology = _mountinfo_topology(_read_bounded_control_text(Path("/proc/self/mountinfo")))
    required_mounts = {
        "/": ("ro", None),
        _CONTAINED_SOURCE_PATH.as_posix(): ("ro", None),
        _CONTAINED_REPOSITORY_ROOT.as_posix(): ("ro", None),
        _CONTAINED_OUTPUT_PATH.as_posix(): ("rw", None),
        _CONTAINED_SNAPSHOT_DIRECTORY.as_posix(): ("rw", None),
        _CONTAINED_TEMPORARY_DIRECTORY.as_posix(): ("rw", "tmpfs"),
        "/sys/fs/cgroup": ("ro", "cgroup2"),
    }
    for mount_point, (required_option, expected_filesystem) in required_mounts.items():
        observed = topology.get(mount_point)
        if observed is None:
            raise RuntimeError("contained worker lacks one required isolated mount")
        options, filesystem = observed
        if required_option not in options or (
            expected_filesystem is not None and filesystem != expected_filesystem
        ):
            raise RuntimeError("contained worker mount permissions or type differ")
    temporary_options = topology[_CONTAINED_TEMPORARY_DIRECTORY.as_posix()][0]
    if not {"rw", "nosuid", "nodev", "noexec"}.issubset(temporary_options):
        raise RuntimeError("contained worker temporary mount is insufficiently restricted")

    if _read_bounded_control_text(Path("/proc/self/cgroup")).strip() not in {"0::/"}:
        raise RuntimeError("contained worker is outside its private cgroup-v2 namespace")
    expected_limits = {
        Path("/sys/fs/cgroup/memory.max"): str(_CONTAINED_MEMORY_MAX_BYTES),
        Path("/sys/fs/cgroup/memory.swap.max"): "0",
        Path("/sys/fs/cgroup/pids.max"): str(_CONTAINED_PIDS_MAX),
    }
    if any(
        _read_bounded_control_text(path).strip() != value for path, value in expected_limits.items()
    ):
        raise RuntimeError("contained worker cgroup limits differ from the approved policy")

    status: dict[str, str] = {}
    for line in _read_bounded_control_text(Path("/proc/self/status")).splitlines():
        key, separator, value = line.partition(":")
        if separator:
            status[key] = value.strip()
    if (
        status.get("NoNewPrivs") != "1"
        or status.get("CapEff") != "0000000000000000"
        or status.get("CapPrm") != "0000000000000000"
        or status.get("CapBnd") != "0000000000000000"
    ):
        raise RuntimeError("contained worker process privileges differ from policy")
    ipv4_routes = _read_bounded_control_text(Path("/proc/net/route")).splitlines()
    ipv6_routes = _read_bounded_control_text(Path("/proc/net/ipv6_route")).splitlines()
    if any(
        fields and fields[0] != "lo" for line in ipv4_routes[1:] if (fields := line.split())
    ) or any(fields and fields[-1] != "lo" for line in ipv6_routes if (fields := line.split())):
        raise RuntimeError("contained worker has a non-loopback network route")


def _require_exact_contained_invocation(
    *,
    source_path: Path,
    output_path: Path,
    repository_root: Path,
    execution_id: str,
    expected_source_sha256: str,
    expected_source_byte_count: int,
    snapshot_directory: Path,
    snapshot_max_bytes: int,
) -> None:
    exact = (
        source_path == _CONTAINED_SOURCE_PATH
        and output_path == _CONTAINED_OUTPUT_PATH
        and repository_root == _CONTAINED_REPOSITORY_ROOT
        and execution_id == _CONTAINED_EXECUTION_ID
        and expected_source_sha256 == SCIPLEX3_SOURCE_SHA256
        and expected_source_byte_count == SCIPLEX3_SOURCE_BYTE_COUNT
        and snapshot_directory == _CONTAINED_SNAPSHOT_DIRECTORY
        and snapshot_max_bytes == _CONTAINED_SNAPSHOT_MAX_BYTES
    )
    if not exact:
        raise RuntimeError("worker invocation differs from the approved contained contract")
    _verify_contained_worker_topology()


def _hash_descriptor(descriptor: int) -> tuple[str, int]:
    digest = hashlib.sha256()
    byte_count = 0
    before = os.fstat(descriptor)
    if not stat.S_ISREG(before.st_mode):
        raise RuntimeError("source snapshot descriptor is not a regular file")
    os.lseek(descriptor, 0, os.SEEK_SET)
    while True:
        chunk = os.read(descriptor, 1024 * 1024)
        if not chunk:
            break
        digest.update(chunk)
        byte_count += len(chunk)
    after = os.fstat(descriptor)
    if (before.st_dev, before.st_ino, before.st_size) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
    ):
        raise RuntimeError("source snapshot changed while being authenticated")
    os.lseek(descriptor, 0, os.SEEK_SET)
    return digest.hexdigest(), byte_count


def _snapshot_pinned_source(
    source: Path,
    snapshot_directory: Path,
    *,
    snapshot_max_bytes: int,
) -> tuple[int, str, int]:
    """Copy the Docker-pinned bind once, then hold an unlinked immutable private inode."""

    read_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        source_descriptor = os.open(source, read_flags)
    except BaseException:
        raise _SnapshotFailure(
            source_access_started=False,
            protected_source_acquired=False,
            snapshot_descriptor=None,
        ) from None
    snapshot_path = snapshot_directory / ".contained-source-snapshot"
    write_flags = (
        os.O_RDWR
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    snapshot_descriptor = -1
    read_descriptor = -1
    try:
        source_before = os.fstat(source_descriptor)
        if not stat.S_ISREG(source_before.st_mode):
            raise RuntimeError("Docker source bind is not one regular file")
        if source_before.st_size > snapshot_max_bytes:
            raise RuntimeError("Docker source bind exceeds the bounded snapshot volume budget")
        snapshot_descriptor = os.open(snapshot_path, write_flags, 0o600)
        digest = hashlib.sha256()
        byte_count = 0
        while True:
            chunk = os.read(source_descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            byte_count += len(chunk)
            if byte_count > snapshot_max_bytes:
                raise RuntimeError("source snapshot crossed its exact volume budget")
            offset = 0
            while offset < len(chunk):
                written = os.write(snapshot_descriptor, chunk[offset:])
                if written <= 0:
                    raise RuntimeError("short private source-snapshot write")
                offset += written
        source_after = os.fstat(source_descriptor)
        if (
            source_before.st_dev,
            source_before.st_ino,
            source_before.st_size,
        ) != (source_after.st_dev, source_after.st_ino, source_after.st_size):
            raise RuntimeError("Docker source bind changed while snapshotting")
        os.fsync(snapshot_descriptor)
        os.fchmod(snapshot_descriptor, 0o400)
        read_descriptor = os.open(snapshot_path, read_flags)
        snapshot_path.unlink()
        os.close(snapshot_descriptor)
        snapshot_descriptor = -1
        returned_descriptor = read_descriptor
        read_descriptor = -1
        return returned_descriptor, digest.hexdigest(), byte_count
    except BaseException:
        failure_descriptor: int | None = None
        if read_descriptor >= 0:
            failure_descriptor = read_descriptor
            read_descriptor = -1
            if snapshot_descriptor >= 0:
                with suppress(OSError):
                    os.close(snapshot_descriptor)
                snapshot_descriptor = -1
        elif snapshot_descriptor >= 0:
            failure_descriptor = snapshot_descriptor
            snapshot_descriptor = -1
        with suppress(OSError):
            snapshot_path.unlink()
        raise _SnapshotFailure(
            source_access_started=True,
            protected_source_acquired=True,
            snapshot_descriptor=failure_descriptor,
        ) from None
    finally:
        with suppress(OSError):
            os.close(source_descriptor)


def _write_exclusive(path: Path, payload: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o400)
    try:
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise RuntimeError("short contained-worker observation write")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    directory_descriptor = os.open(
        path.parent,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_DIRECTORY", 0),
    )
    try:
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)


def _failure_code_for_phase(
    phase: Literal[
        "source_authentication",
        "contract_authentication",
        "preparation",
        "training",
        "stage_finalization",
        "completed",
    ],
) -> Literal[
    "source_authentication_failed",
    "contract_authentication_failed",
    "preparation_failed",
    "training_failed",
    "stage_finalization_failed",
]:
    if phase == "source_authentication":
        return "source_authentication_failed"
    if phase == "contract_authentication":
        return "contract_authentication_failed"
    if phase == "preparation":
        return "preparation_failed"
    if phase == "training":
        return "training_failed"
    return "stage_finalization_failed"


def run_worker(
    *,
    source_path: Path,
    output_path: Path,
    repository_root: Path,
    execution_id: str,
    expected_source_sha256: str,
    expected_source_byte_count: int,
    snapshot_directory: Path,
    snapshot_max_bytes: int,
) -> ContainedTrainingWorkerTerminalReport:
    """Verify confinement, authenticate the pinned bind, fit v5, and leave only a stage."""

    _require_exact_contained_invocation(
        source_path=source_path,
        output_path=output_path,
        repository_root=repository_root,
        execution_id=execution_id,
        expected_source_sha256=expected_source_sha256,
        expected_source_byte_count=expected_source_byte_count,
        snapshot_directory=snapshot_directory,
        snapshot_max_bytes=snapshot_max_bytes,
    )
    phase: Literal[
        "source_authentication",
        "contract_authentication",
        "preparation",
        "training",
        "stage_finalization",
        "completed",
    ] = "source_authentication"
    failure_code: (
        Literal[
            "source_authentication_failed",
            "contract_authentication_failed",
            "preparation_failed",
            "training_failed",
            "stage_finalization_failed",
            "source_post_authentication_failed",
            "stage_inventory_failed",
        ]
        | None
    ) = None
    pre_sha256: str | None = None
    pre_byte_count: int | None = None
    post_sha256: str | None = None
    post_byte_count: int | None = None
    source_access_started = False
    protected_source_acquired = False
    source_post_authentication_attempted = False
    snapshot_descriptor = -1
    copied_sha256: str | None = None
    copied_byte_count: int | None = None
    policy = None
    code_closure = None
    input_closure = None
    runtime_image_lock = None
    plan = None
    try:
        snapshot_descriptor, copied_sha256, copied_byte_count = _snapshot_pinned_source(
            source_path,
            snapshot_directory,
            snapshot_max_bytes=snapshot_max_bytes,
        )
        source_access_started = True
        protected_source_acquired = True
    except _SnapshotFailure as snapshot_failure:
        source_access_started = snapshot_failure.source_access_started
        protected_source_acquired = snapshot_failure.protected_source_acquired
        snapshot_descriptor = (
            snapshot_failure.snapshot_descriptor
            if snapshot_failure.snapshot_descriptor is not None
            else -1
        )
        failure_code = "source_authentication_failed"
    try:
        if failure_code is not None:
            raise RuntimeError("private source snapshot did not complete")
        pre_sha256, pre_byte_count = _hash_descriptor(snapshot_descriptor)
        if (pre_sha256, pre_byte_count) != (copied_sha256, copied_byte_count) or (
            pre_sha256,
            pre_byte_count,
        ) != (expected_source_sha256, expected_source_byte_count):
            raise RuntimeError("pinned source bytes differ before contained training")

        # Import source-touching and numerical dependencies only after the private descriptor is
        # independently authenticated inside Docker's aggregate cgroup.
        phase = "contract_authentication"
        import scripts.materialize_sciplex3_k562_p1_candidate as materializer
        from cellstate.evaluation import sciplex3_candidate_runner as runner

        policy, code_closure, input_closure, runtime_image_lock = (
            runner.contained_training_contracts(repository_root)
        )
        snapshot_proc_path = Path(f"/proc/self/fd/{snapshot_descriptor}")
        phase = "preparation"
        preparation = materializer._prepare_exact_p1(snapshot_proc_path, repository_root)
        repository_bindings = materializer._repository_bindings(repository_root)
        support_fingerprint, _ = materializer._planned_support_envelope(repository_root)
        plan = runner.build_sciplex3_candidate_training_plan(
            preparation,
            benchmark_fingerprint=materializer._binding_sha256(repository_bindings, "benchmark"),
            support_envelope_fingerprint=support_fingerprint,
        )

        sealed = runner.seal_sciplex3_candidate_training_plan(
            preparation, plan, output_path / "sealed-plan"
        )
        phase = "training"
        fitted = runner.fit_and_write_sciplex3_candidate(preparation, sealed, output_path / "fit")
        observation = runner.verify_sciplex3_candidate_fit(preparation, sealed, fitted)
        phase = "stage_finalization"
        shutil.copyfile(sealed.artifact.path, output_path / "candidate-training-plan.json")
        shutil.copyfile(fitted.model_artifact.path, output_path / "candidate-model.json")
        shutil.copyfile(
            fitted.observation_artifact.path,
            output_path / "training-execution-observation.json",
        )
        _write_exclusive(
            output_path / "p1-finalized-count-scan-receipt.json",
            canonical_json_bytes(preparation.finalized_count_scan_manifest()),
        )
        _write_exclusive(
            output_path / "p1-assembly-receipt.json",
            canonical_json_bytes(asdict(preparation.receipt)),
        )

        sealed_support = {
            artifact.path.name: artifact.path for artifact in sealed.support_artifacts
        }
        required_legacy_support = {
            "candidate-specification.json",
            "output-model-schema.json",
            "p1-count-stream-descriptor.json",
            "runtime-lock.json",
        }
        if not required_legacy_support.issubset(sealed_support):
            raise RuntimeError("sealed contained plan lacks the materialization support closure")
        vertical = materializer.BENCHMARK_RELATIVE_DIRECTORY / "item12-p1"
        artifact_payloads = {
            "assembly_receipt": (
                vertical / "p1-assembly-receipt.json",
                (output_path / "p1-assembly-receipt.json").read_bytes(),
            ),
            "candidate_model": (
                vertical / "candidate-model.json",
                (output_path / "candidate-model.json").read_bytes(),
            ),
            "candidate_output_model_schema": (
                materializer.SUPPORT_RELATIVE_PATHS["output-model-schema.json"],
                sealed_support["output-model-schema.json"].read_bytes(),
            ),
            "candidate_runtime_lock": (
                materializer.SUPPORT_RELATIVE_PATHS["runtime-lock.json"],
                sealed_support["runtime-lock.json"].read_bytes(),
            ),
            "candidate_specification": (
                materializer.SUPPORT_RELATIVE_PATHS["candidate-specification.json"],
                sealed_support["candidate-specification.json"].read_bytes(),
            ),
            "candidate_training_plan": (
                vertical / "candidate-training-plan.json",
                (output_path / "candidate-training-plan.json").read_bytes(),
            ),
            "finalized_count_scan_receipt": (
                vertical / "p1-finalized-count-scan-receipt.json",
                (output_path / "p1-finalized-count-scan-receipt.json").read_bytes(),
            ),
            "p1_count_stream_descriptor": (
                materializer.COUNT_DESCRIPTOR_RELATIVE_PATH,
                sealed_support["p1-count-stream-descriptor.json"].read_bytes(),
            ),
            "training_execution_observation": (
                vertical / "training-execution-observation.json",
                (output_path / "training-execution-observation.json").read_bytes(),
            ),
        }
        materialization = materializer._build_manifest(
            preparation,
            plan,
            observation,
            artifact_payloads,
            repository_bindings,
            support_envelope_fingerprint=support_fingerprint,
        )
        _write_exclusive(
            output_path / "materialization-manifest.json",
            canonical_json_bytes(materialization),
        )
    except BaseException:
        failure_code = _failure_code_for_phase(phase)
    finally:
        if snapshot_descriptor >= 0:
            source_post_authentication_attempted = True
            try:
                post_sha256, post_byte_count = _hash_descriptor(snapshot_descriptor)
            except BaseException:
                failure_code = "source_post_authentication_failed"
            finally:
                os.close(snapshot_descriptor)

    source_matches_expected = (
        pre_sha256 == post_sha256 == expected_source_sha256
        and pre_byte_count == post_byte_count == expected_source_byte_count
    )
    if not source_matches_expected and failure_code is None:
        failure_code = "source_post_authentication_failed"
    try:
        staged_inventory = inventory_staged_training_tree(output_path)
        staged_tree_sha256: str | None = staged_inventory.fingerprint
        staged_file_count: int | None = len(staged_inventory.entries)
        staged_inventory_status: Literal["verified", "unavailable"] = "verified"
    except BaseException:
        staged_inventory = None
        staged_tree_sha256 = None
        staged_file_count = None
        staged_inventory_status = "unavailable"
        if failure_code is None:
            failure_code = "stage_inventory_failed"

    success_observation: ContainedTrainingWorkerObservation | None = None
    if failure_code is None:
        if (
            policy is None
            or code_closure is None
            or input_closure is None
            or runtime_image_lock is None
            or plan is None
            or pre_sha256 is None
            or pre_byte_count is None
            or post_sha256 is None
            or post_byte_count is None
            or staged_inventory is None
        ):
            failure_code = "stage_finalization_failed"
        else:
            try:
                success_observation = ContainedTrainingWorkerObservation(
                    execution_id=execution_id,
                    training_plan_fingerprint=plan.fingerprint,
                    policy_fingerprint=policy.fingerprint,
                    runtime_image_digest=runtime_image_lock.runtime_image.digest,
                    training_code_closure_sha256=code_closure.fingerprint,
                    execution_input_closure_sha256=input_closure.fingerprint,
                    expected_source_sha256=expected_source_sha256,
                    source_pre_sha256=pre_sha256,
                    source_post_sha256=post_sha256,
                    expected_source_byte_count=expected_source_byte_count,
                    source_pre_byte_count=pre_byte_count,
                    source_post_byte_count=post_byte_count,
                    staged_inventory=staged_inventory,
                    staged_tree_sha256=staged_inventory.fingerprint,
                    staged_file_count=len(staged_inventory.entries),
                )
            except BaseException:
                failure_code = "stage_finalization_failed"

    if failure_code is None:
        phase = "completed"
        outcome: Literal["success", "worker_failure"] = "success"
        report_failure_code: Literal[
            "none",
            "source_authentication_failed",
            "contract_authentication_failed",
            "preparation_failed",
            "training_failed",
            "stage_finalization_failed",
            "source_post_authentication_failed",
            "stage_inventory_failed",
        ] = "none"
    else:
        outcome = "worker_failure"
        report_failure_code = failure_code
    terminal_report = ContainedTrainingWorkerTerminalReport(
        execution_id=execution_id,
        outcome=outcome,
        terminal_phase=phase,
        failure_code=report_failure_code,
        expected_source_sha256=expected_source_sha256,
        expected_source_byte_count=expected_source_byte_count,
        source_pre_sha256=pre_sha256,
        source_pre_byte_count=pre_byte_count,
        source_post_sha256=post_sha256,
        source_post_byte_count=post_byte_count,
        source_access_started=source_access_started,
        protected_source_acquired=protected_source_acquired,
        source_post_authentication_attempted=source_post_authentication_attempted,
        source_post_authentication_completed=post_sha256 is not None,
        source_matches_expected=source_matches_expected,
        staged_inventory_status=staged_inventory_status,
        staged_tree_sha256=staged_tree_sha256,
        staged_file_count=staged_file_count,
        success_observation=success_observation,
    )
    _write_exclusive(
        output_path / "contained-worker-observation.json",
        canonical_json_bytes(terminal_report.model_dump(mode="json")),
    )
    if terminal_report.outcome != "success":
        raise _ReportedWorkerFailure("contained worker failed; see sanitized terminal report")
    return terminal_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--execution-id", required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--expected-source-byte-count", type=int, required=True)
    parser.add_argument("--snapshot-directory", type=Path, required=True)
    parser.add_argument("--snapshot-max-bytes", type=int, required=True)
    args = parser.parse_args()
    try:
        run_worker(
            source_path=args.source,
            output_path=args.output,
            repository_root=args.repository_root,
            execution_id=args.execution_id,
            expected_source_sha256=args.expected_source_sha256,
            expected_source_byte_count=args.expected_source_byte_count,
            snapshot_directory=args.snapshot_directory,
            snapshot_max_bytes=args.snapshot_max_bytes,
        )
    except BaseException:
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()
