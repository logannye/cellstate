#!/usr/bin/env python3
"""Authorized host supervisor for one nonissuing, contained sci-Plex3 v5 fit."""

from __future__ import annotations

import argparse
import os
import stat
from pathlib import Path
from typing import Literal

from cellstate.backends.sciplex3_loader import (
    SCIPLEX3_SOURCE_BYTE_COUNT,
    SCIPLEX3_SOURCE_SHA256,
)
from cellstate.domain.common import canonical_json_bytes
from cellstate.evaluation.sciplex3_candidate_runner import contained_training_contracts
from cellstate.training.execution import (
    PARENT_TERMINAL_REPORT_MAX_BYTES,
    WORKER_TERMINAL_REPORT_MAX_BYTES,
    CanonicalPublicationTreeIdentity,
    ContainedExecutionError,
    ContainedExecutionObservation,
    ContainedTrainingObservation,
    ContainedTrainingTerminalObservation,
    ContainedTrainingWorkerTerminalReport,
    DockerExecutor,
    ExecutionStageAlreadyClaimed,
    StagedTrainingInventory,
    canonical_publication_tree_identity,
    inventory_staged_training_tree,
    seal_staged_training_tree,
)
from cellstate.training.item12_3_authorization import (
    Item123ExecutionStart,
    VerifiedItem123ExecutionCapability,
    verify_capability_for_execution,
    verify_execution_start,
)

CONTAINED_EXECUTION_ID = "sciplex3-k562-v5-fit"
CANONICAL_PUBLICATION_RELATIVE_TEXT: Literal[
    "backends/vertical-a/sciplex3-k562-24h-v1/candidate-publication"
] = "backends/vertical-a/sciplex3-k562-24h-v1/candidate-publication"
CANONICAL_PUBLICATION_RELATIVE_PATH = Path(CANONICAL_PUBLICATION_RELATIVE_TEXT)
PARENT_TERMINAL_REPORT_FILENAME = "contained-training-terminal-observation.json"
WORKER_TERMINAL_REPORT_FILENAME = "contained-worker-observation.json"
SUCCESS_OBSERVATION_FILENAME = "contained-training-observation.json"
_PARENT_TERMINAL_INVENTORY_MAX_ENTRIES = 64

_SEALED_SUPPORT_FILENAMES = (
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
_SUCCESS_STAGE_PATHS = tuple(
    sorted(
        (
            "candidate-model.json",
            "candidate-training-plan.json",
            "fit/candidate-model.json",
            "fit/training-execution-observation.json",
            "materialization-manifest.json",
            "p1-assembly-receipt.json",
            "p1-finalized-count-scan-receipt.json",
            "sealed-plan/candidate-training-plan.json",
            *(f"sealed-plan/{name}" for name in _SEALED_SUPPORT_FILENAMES),
            "training-execution-observation.json",
        )
    )
)


def _write_exclusive(path: Path, payload: bytes) -> None:
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(path, flags, 0o400)
        try:
            offset = 0
            while offset < len(payload):
                written = os.write(descriptor, payload[offset:])
                if written <= 0:
                    raise ContainedExecutionError("short parent terminal-evidence write")
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
    except OSError as error:
        raise ContainedExecutionError("cannot durably write parent terminal evidence") from error


def _read_bounded_worker_report(path: Path) -> ContainedTrainingWorkerTerminalReport:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        before = path.lstat()
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_size <= 0
            or before.st_size > WORKER_TERMINAL_REPORT_MAX_BYTES
        ):
            raise ContainedExecutionError("worker terminal report violates its exact byte bound")
        descriptor = os.open(path, flags)
        try:
            opened = os.fstat(descriptor)
            chunks: list[bytes] = []
            byte_count = 0
            while chunk := os.read(
                descriptor,
                min(64 * 1024, WORKER_TERMINAL_REPORT_MAX_BYTES + 1 - byte_count),
            ):
                chunks.append(chunk)
                byte_count += len(chunk)
                if byte_count > WORKER_TERMINAL_REPORT_MAX_BYTES:
                    break
            after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
    except FileNotFoundError:
        raise
    except OSError as error:
        raise ContainedExecutionError("cannot read worker terminal report") from error
    payload = b"".join(chunks)
    if (
        (before.st_dev, before.st_ino, before.st_size)
        != (opened.st_dev, opened.st_ino, opened.st_size)
        or (opened.st_dev, opened.st_ino, opened.st_size)
        != (after.st_dev, after.st_ino, after.st_size)
        or len(payload) != opened.st_size
    ):
        raise ContainedExecutionError("worker terminal report changed while being read")
    try:
        report = ContainedTrainingWorkerTerminalReport.model_validate_json(payload)
    except ValueError as error:
        raise ContainedExecutionError("worker terminal report is invalid") from error
    if canonical_json_bytes(report.model_dump(mode="json")) != payload:
        raise ContainedExecutionError("worker terminal report is not canonical JSON")
    return report


def _semantic_stage_fingerprint(
    output: Path,
    *,
    repository_root: Path,
    inventory: StagedTrainingInventory,
) -> str:
    paths = tuple(entry.relative_path for entry in inventory.entries)
    if paths != _SUCCESS_STAGE_PATHS:
        raise ContainedExecutionError("successful worker stage has a noncanonical file closure")
    entries = {entry.relative_path: entry for entry in inventory.entries}
    if (
        entries["candidate-model.json"].sha256 != entries["fit/candidate-model.json"].sha256
        or entries["candidate-training-plan.json"].sha256
        != entries["sealed-plan/candidate-training-plan.json"].sha256
        or entries["training-execution-observation.json"].sha256
        != entries["fit/training-execution-observation.json"].sha256
    ):
        raise ContainedExecutionError("successful worker stage duplicates different artifact bytes")

    # This checker reauthenticates only staged and checked-in public control bytes.  It has no
    # protected-source locator and cannot publish.
    import scripts.materialize_sciplex3_k562_p1_candidate as materializer

    try:
        fingerprint = materializer.check_materialization_inputs(
            output,
            repository_root=repository_root,
            sealed_support_directory=output / "sealed-plan",
            count_stream_descriptor_path=(output / "sealed-plan/p1-count-stream-descriptor.json"),
        )
    except BaseException as error:
        raise ContainedExecutionError(
            "successful worker stage failed source-free semantic verification"
        ) from error
    manifest_entry = entries["materialization-manifest.json"]
    if (
        type(fingerprint) is not str
        or fingerprint != manifest_entry.sha256
        or len(fingerprint) != 64
    ):
        raise ContainedExecutionError(
            "source-free materialization checker returned a contradictory fingerprint"
        )
    return fingerprint


def _worker_success_is_exact(
    report: ContainedTrainingWorkerTerminalReport,
    execution: ContainedExecutionObservation,
    *,
    policy_fingerprint: str,
    runtime_image_digest: str,
    code_closure_sha256: str,
    input_closure_sha256: str,
    inventory: StagedTrainingInventory,
) -> bool:
    worker = report.success_observation
    return bool(
        report.outcome == "success"
        and report.execution_id == CONTAINED_EXECUTION_ID
        and report.expected_source_sha256 == SCIPLEX3_SOURCE_SHA256
        and report.expected_source_byte_count == SCIPLEX3_SOURCE_BYTE_COUNT
        and report.source_matches_expected
        and report.source_post_authentication_completed
        and report.staged_tree_sha256 == inventory.fingerprint
        and report.staged_file_count == len(inventory.entries)
        and worker is not None
        and worker.execution_id == execution.execution_id
        and worker.policy_fingerprint == policy_fingerprint
        and worker.runtime_image_digest == runtime_image_digest
        and worker.training_code_closure_sha256 == code_closure_sha256
        and worker.execution_input_closure_sha256 == input_closure_sha256
        and worker.staged_inventory == inventory
    )


def _quarantine_stage(output: Path) -> Path:
    """Atomically remove a failed stage from the success path, then no-follow restrict it."""

    quarantine = output.parent / "quarantine"
    try:
        if quarantine.exists() or quarantine.is_symlink():
            raise ContainedExecutionError("execution quarantine path already exists")
        output.replace(quarantine)
        directories = [quarantine]
        files: list[Path] = []
        for current_name, directory_names, file_names in os.walk(quarantine, followlinks=False):
            current = Path(current_name)
            for name in directory_names:
                child = current / name
                child_state = child.lstat()
                if stat.S_ISDIR(child_state.st_mode):
                    directories.append(child)
                elif not stat.S_ISLNK(child_state.st_mode):
                    raise ContainedExecutionError("quarantine contains an invalid directory entry")
            files.extend(current / name for name in file_names)
        for path in files:
            state = path.lstat()
            if stat.S_ISREG(state.st_mode):
                path.chmod(0o400)
            elif not stat.S_ISLNK(state.st_mode):
                raise ContainedExecutionError("quarantine contains an invalid file entry")
        for directory in sorted(directories, key=lambda item: len(item.parts), reverse=True):
            directory.chmod(0o500)
        directory_descriptor = os.open(
            quarantine.parent,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except OSError as error:
        raise ContainedExecutionError("cannot durably quarantine failed worker stage") from error
    return quarantine


def _canonical_after_worker(
    canonical_root: Path,
) -> tuple[CanonicalPublicationTreeIdentity | None, bool]:
    try:
        return canonical_publication_tree_identity(canonical_root), True
    except ContainedExecutionError:
        return None, False


def _execute_contained_training(
    *,
    capability: VerifiedItem123ExecutionCapability,
    execution_start: Item123ExecutionStart,
    source_path: Path,
    repository_root: Path,
    protected_source_acquired: Literal[True],
) -> tuple[Path, ContainedTrainingTerminalObservation]:
    """Execute after authorization consumption; never open source bytes or publish output."""

    root = Path(repository_root)
    verify_capability_for_execution(capability, repository_root=root)
    verify_execution_start(capability, execution_start)
    if protected_source_acquired is not True:
        raise ContainedExecutionError("supervisor requires completed protected-source acquisition")
    paths = capability.proposal.execution_paths
    if (
        paths.execution_id != CONTAINED_EXECUTION_ID
        or paths.canonical_publication_root != CANONICAL_PUBLICATION_RELATIVE_TEXT
    ):
        raise ContainedExecutionError("verified capability carries different execution paths")
    approved_source = Path(paths.protected_source_path)
    provided_source = Path(source_path)
    if (
        not approved_source.is_absolute()
        or ".." in approved_source.parts
        or Path(os.path.normpath(os.fspath(approved_source))) != approved_source
        or provided_source != approved_source
    ):
        raise ContainedExecutionError("protected source path differs from the verified capability")
    staging_root = Path(paths.staging_root)
    if not root.is_absolute() or ".." in root.parts:
        raise ContainedExecutionError("repository root must be one normalized absolute path")
    canonical_root = root / CANONICAL_PUBLICATION_RELATIVE_PATH
    canonical_before = canonical_publication_tree_identity(canonical_root)
    policy, code_closure, input_closure, image_lock = contained_training_contracts(root)
    executor = DockerExecutor(
        policy,
        lock_root=staging_root / ".locks",
        staging_root=staging_root,
        canonical_publication_root=canonical_root,
        execution_input_closure=input_closure,
    )
    output = executor.output_stage_path(CONTAINED_EXECUTION_ID)
    try:
        prior_execution_state = output.parent.lstat()
    except FileNotFoundError:
        pass
    else:
        if not stat.S_ISDIR(prior_execution_state.st_mode):
            raise ContainedExecutionError("prior execution root is not one real directory")
        raise ContainedExecutionError(
            "execution ID has already been consumed; prior terminal evidence is preserved"
        )
    execution: ContainedExecutionObservation | None = None
    executor_failed = False
    try:
        execution = executor.run(
            execution_id=CONTAINED_EXECUTION_ID,
            source_path=approved_source,
            code_path=root,
            output_path=output,
        )
    except ExecutionStageAlreadyClaimed:
        raise
    except BaseException:
        if not executor.owns_output_stage_claim(
            execution_id=CONTAINED_EXECUTION_ID,
            output_path=output,
        ):
            raise
        executor_failed = True
    try:
        output_state = output.lstat()
    except OSError as error:
        raise ContainedExecutionError(
            "contained executor failed before a terminal stage could be preserved"
        ) from error
    if not stat.S_ISDIR(output_state.st_mode):
        raise ContainedExecutionError("contained executor output is not one real directory")
    canonical_after, canonical_after_valid = _canonical_after_worker(canonical_root)

    worker_report: ContainedTrainingWorkerTerminalReport | None = None
    worker_report_status: Literal["verified", "missing", "invalid"]
    try:
        worker_report = _read_bounded_worker_report(output / WORKER_TERMINAL_REPORT_FILENAME)
    except FileNotFoundError:
        worker_report_status = "missing"
    except ContainedExecutionError:
        worker_report_status = "invalid"
    else:
        worker_report_status = "verified"

    inventory: StagedTrainingInventory | None
    try:
        inventory = inventory_staged_training_tree(output)
    except ContainedExecutionError:
        inventory = None
    if inventory is not None and len(inventory.entries) > _PARENT_TERMINAL_INVENTORY_MAX_ENTRIES:
        inventory = None

    terminal_status: Literal[
        "success",
        "timeout",
        "oom_killed",
        "worker_failure",
        "stage_rejected",
        "supervisor_failure",
    ]
    failure_code: Literal[
        "none",
        "worker_timed_out",
        "worker_oom_killed",
        "worker_exited_nonzero",
        "worker_report_missing",
        "worker_report_invalid",
        "worker_report_contradiction",
        "stage_inventory_invalid",
        "stage_semantic_verification_failed",
        "canonical_publication_changed",
        "canonical_publication_identity_invalid",
        "contained_executor_failed",
    ]
    if executor_failed:
        terminal_status = "supervisor_failure"
        failure_code = "contained_executor_failed"
    elif not canonical_after_valid:
        terminal_status = "stage_rejected"
        failure_code = "canonical_publication_identity_invalid"
    elif canonical_after != canonical_before:
        terminal_status = "stage_rejected"
        failure_code = "canonical_publication_changed"
    elif inventory is None:
        terminal_status = "stage_rejected"
        failure_code = "stage_inventory_invalid"
    elif execution is None:
        terminal_status = "supervisor_failure"
        failure_code = "contained_executor_failed"
    elif execution.outcome == "timeout":
        terminal_status = "timeout"
        failure_code = "worker_timed_out"
    elif execution.outcome == "oom_killed":
        terminal_status = "oom_killed"
        failure_code = "worker_oom_killed"
    elif execution.outcome == "worker_failure":
        terminal_status = "worker_failure"
        failure_code = "worker_exited_nonzero"
    elif worker_report_status == "missing":
        terminal_status = "stage_rejected"
        failure_code = "worker_report_missing"
    elif worker_report_status == "invalid":
        terminal_status = "stage_rejected"
        failure_code = "worker_report_invalid"
    elif worker_report is None or not _worker_success_is_exact(
        worker_report,
        execution,
        policy_fingerprint=policy.fingerprint,
        runtime_image_digest=image_lock.runtime_image.digest,
        code_closure_sha256=code_closure.fingerprint,
        input_closure_sha256=input_closure.fingerprint,
        inventory=inventory,
    ):
        terminal_status = "stage_rejected"
        failure_code = "worker_report_contradiction"
    else:
        terminal_status = "success"
        failure_code = "none"

    semantic_stage_sha256: str | None = None
    success_observation: ContainedTrainingObservation | None = None
    if (
        terminal_status == "success"
        and execution is not None
        and inventory is not None
        and worker_report is not None
    ):
        try:
            semantic_stage_sha256 = _semantic_stage_fingerprint(
                output,
                repository_root=root,
                inventory=inventory,
            )
            worker_success = worker_report.success_observation
            if worker_success is None:  # pragma: no cover - narrowed by the exact join above
                raise ContainedExecutionError("successful worker report lost its observation")
            success_observation = ContainedTrainingObservation(
                training_plan_fingerprint=worker_success.training_plan_fingerprint,
                policy_fingerprint=policy.fingerprint,
                runtime_image_digest=image_lock.runtime_image.digest,
                training_code_closure_sha256=code_closure.fingerprint,
                execution_input_closure_sha256=input_closure.fingerprint,
                staged_inventory=inventory,
                staged_tree_sha256=inventory.fingerprint,
                worker_observation=worker_success,
                execution_observation=execution,
                wall_clock_limit_seconds=policy.wall_clock_seconds,
                memory_max_bytes=policy.memory_max_bytes,
                memory_swap_max_bytes=policy.memory_swap_max_bytes,
            )
            _write_exclusive(
                output / SUCCESS_OBSERVATION_FILENAME,
                canonical_json_bytes(success_observation.model_dump(mode="json")),
            )
            seal_staged_training_tree(output, expected_inventory=inventory)
        except BaseException:
            terminal_status = "stage_rejected"
            failure_code = "stage_semantic_verification_failed"
            semantic_stage_sha256 = None
            success_observation = None

    if terminal_status == "success":
        stage = output
        stage_disposition: Literal["sealed", "quarantined"] = "sealed"
        stage_relative_path: Literal["output", "quarantine"] = "output"
    else:
        stage = _quarantine_stage(output)
        stage_disposition = "quarantined"
        stage_relative_path = "quarantine"

    terminal = ContainedTrainingTerminalObservation(
        execution_id=CONTAINED_EXECUTION_ID,
        terminal_status=terminal_status,
        failure_code=failure_code,
        policy_fingerprint=policy.fingerprint,
        runtime_image_digest=image_lock.runtime_image.digest,
        training_code_closure_sha256=code_closure.fingerprint,
        execution_input_closure_sha256=input_closure.fingerprint,
        wall_clock_limit_seconds=policy.wall_clock_seconds,
        memory_max_bytes=policy.memory_max_bytes,
        memory_swap_max_bytes=policy.memory_swap_max_bytes,
        aggregate_container_limits_enforced=execution is not None,
        container_cleanup_disposition=("proved_removed" if execution is not None else "unproved"),
        snapshot_volume_cleanup_disposition=(
            "proved_removed" if execution is not None else "unproved"
        ),
        protected_source_acquired_before_supervisor=protected_source_acquired,
        execution_observation=execution,
        worker_report_status=worker_report_status,
        worker_terminal_report=(worker_report if worker_report_status == "verified" else None),
        staged_inventory=inventory,
        staged_tree_sha256=inventory.fingerprint if inventory is not None else None,
        staged_file_count=len(inventory.entries) if inventory is not None else None,
        stage_disposition=stage_disposition,
        stage_relative_path=stage_relative_path,
        semantic_stage_sha256=semantic_stage_sha256,
        success_observation=success_observation,
        canonical_publication_relative_path=CANONICAL_PUBLICATION_RELATIVE_TEXT,
        canonical_publication_before=canonical_before,
        canonical_publication_after=canonical_after,
        canonical_publication_unchanged=(canonical_after == canonical_before),
    )
    terminal_payload = canonical_json_bytes(terminal.model_dump(mode="json"))
    if len(terminal_payload) > PARENT_TERMINAL_REPORT_MAX_BYTES:  # pragma: no cover - model guard
        raise ContainedExecutionError("parent terminal observation exceeds its exact byte bound")
    _write_exclusive(output.parent / PARENT_TERMINAL_REPORT_FILENAME, terminal_payload)
    return stage, terminal


def run_contained_training(
    *,
    capability: VerifiedItem123ExecutionCapability,
    execution_start: Item123ExecutionStart,
    repository_root: Path,
    protected_source_acquired: Literal[True],
) -> tuple[Path, ContainedTrainingTerminalObservation]:
    """Derive the capability-bound source, then enter the fixed non-retryable supervisor."""

    root = Path(repository_root)
    verify_capability_for_execution(capability, repository_root=root)
    verify_execution_start(capability, execution_start)
    return _execute_contained_training(
        capability=capability,
        execution_start=execution_start,
        source_path=Path(capability.proposal.execution_paths.protected_source_path),
        repository_root=root,
        protected_source_acquired=protected_source_acquired,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    parser.error(
        "raw contained execution is disabled; use scripts/run_item12_3_authorized_execution.py"
    )
    return 2  # pragma: no cover - argparse exits


if __name__ == "__main__":
    raise SystemExit(main())
