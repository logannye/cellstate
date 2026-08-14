#!/usr/bin/env python3
"""Perform the one authorized Item 12.3 runtime recheck, source fit, and cleanup."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import stat
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal

from cellstate.domain.common import canonical_json_bytes
from cellstate.training.item12_3_authorization import (
    ITEM12_3_MAX_TERMINAL_REPORT_BYTES,
    Item123AuthorizationError,
    Item123TerminalReport,
    VerifiedItem123ExecutionCapability,
    claim_execution_start,
    load_verified_capability,
    record_contained_terminal_report,
    record_terminal_report,
    require_proposal_current,
    verify_capability_for_execution,
)

if __package__:
    from scripts.verify_sciplex3_v5_runtime_distribution import verify_runtime_distribution
else:  # pragma: no cover - direct CLI execution uses the sibling script directory.
    from verify_sciplex3_v5_runtime_distribution import (  # type: ignore[import-not-found,no-redef]
        verify_runtime_distribution,
    )

_CURL = "/usr/bin/curl"
_DISTRIBUTION_LOCK = Path("containers/sciplex3-v5-runtime/runtime-distribution-lock.json")
_IMAGE_LOCK = Path("containers/sciplex3-v5-runtime/runtime-image-lock.json")
_MARKER_MAX_BYTES = 4096


def _private_directory(path: Path, *, create: bool) -> None:
    if create:
        try:
            path.mkdir(mode=0o700)
        except FileExistsError as error:
            raise Item123AuthorizationError("protected source directory already exists") from error
        except OSError as error:
            raise Item123AuthorizationError("cannot create protected source directory") from error
    try:
        state = path.lstat()
    except OSError as error:
        raise Item123AuthorizationError("private execution directory is unavailable") from error
    if (
        not stat.S_ISDIR(state.st_mode)
        or stat.S_ISLNK(state.st_mode)
        or state.st_uid != os.geteuid()
        or state.st_gid != os.getegid()
        or stat.S_IMODE(state.st_mode) != 0o700
    ):
        raise Item123AuthorizationError("private execution directory is not runner-owned 0700")


def _write_marker(path: Path, value: dict[str, Any]) -> None:
    payload = canonical_json_bytes(value)
    if not payload or len(payload) > _MARKER_MAX_BYTES:
        raise Item123AuthorizationError("source state marker violates its size bound")
    _private_directory(path.parent, create=False)
    parent = os.open(
        path.parent,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    descriptor = -1
    try:
        descriptor = os.open(
            path.name,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o400,
            dir_fd=parent,
        )
        os.fchmod(descriptor, 0o400)
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise Item123AuthorizationError("source state marker write was incomplete")
            offset += written
        os.fsync(descriptor)
        os.fsync(parent)
    except FileExistsError as error:
        raise Item123AuthorizationError("source state marker already exists") from error
    except OSError as error:
        raise Item123AuthorizationError("cannot seal source state marker") from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        os.close(parent)


def _consumption_path(capability: VerifiedItem123ExecutionCapability) -> Path:
    paths = capability.proposal.execution_paths
    return Path(paths.attempt_ledger_root) / (
        f"{capability.attempt_consumption.attempt_key}.consumed.json"
    )


def _source_acquisition_may_have_started(
    capability: VerifiedItem123ExecutionCapability,
) -> bool:
    marker = Path(capability.proposal.execution_paths.source_acquisition_started_marker)
    try:
        marker.lstat()
    except FileNotFoundError:
        return False
    except OSError:
        return True
    return True


def _reauthenticate_runtime(
    capability: VerifiedItem123ExecutionCapability, repository_root: Path
) -> None:
    proposal = capability.proposal
    observation = verify_runtime_distribution(
        repository_root / _DISTRIBUTION_LOCK,
        repository_root / _IMAGE_LOCK,
        Path(proposal.execution_paths.runtime_archive_path),
    )
    observed = asdict(observation)
    prepared = json.loads(capability.runtime_preparation_canonical_json)
    if (
        type(prepared) is not dict
        or prepared.pop("load_performed_from_verified_descriptor", None) is not True
        or observed.pop("load_performed_from_verified_descriptor", None) is not False
        or canonical_json_bytes(observed) != canonical_json_bytes(prepared)
    ):
        raise Item123AuthorizationError("runtime changed after capability issuance")


def _open_source_destination(path: Path) -> tuple[int, os.stat_result]:
    _private_directory(path.parent, create=True)
    descriptor = -1
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        os.fchmod(descriptor, 0o600)
        identity = os.fstat(descriptor)
    except OSError as error:
        if descriptor >= 0:
            os.close(descriptor)
        raise Item123AuthorizationError("cannot create private protected-source file") from error
    if not stat.S_ISREG(identity.st_mode) or identity.st_nlink != 1:
        os.close(descriptor)
        raise Item123AuthorizationError("protected-source destination is not one regular file")
    return descriptor, identity


def _acquire_source(capability: VerifiedItem123ExecutionCapability) -> tuple[Path, os.stat_result]:
    proposal = capability.proposal
    require_proposal_current(proposal)
    source = proposal.protected_source
    limits = proposal.resource_limits
    destination = Path(proposal.execution_paths.protected_source_path)
    started_marker = Path(proposal.execution_paths.source_acquisition_started_marker)
    _write_marker(
        started_marker,
        {
            "artifact_schema": "cellstate-item12-3-source-acquisition-started",
            "artifact_schema_version": "1.0.0",
            "attempt_key": capability.attempt_consumption.attempt_key,
            "proposal_sha256": capability.proposal_sha256,
            "retry_permitted": False,
        },
    )
    descriptor, identity = _open_source_destination(destination)
    sha256 = hashlib.sha256()
    md5 = hashlib.md5(usedforsecurity=False)
    byte_count = 0
    process: subprocess.Popen[bytes] | None = None
    try:
        environment = {"LANG": "C", "PATH": "/usr/bin:/bin"}
        process = subprocess.Popen(
            (
                _CURL,
                "--disable",
                "--fail",
                "--location",
                "--silent",
                "--show-error",
                "--proto",
                "=https",
                "--proto-redir",
                "=https",
                "--connect-timeout",
                str(limits.source_download_connect_timeout_seconds),
                "--max-time",
                str(limits.source_download_total_timeout_seconds),
                "--retry",
                str(limits.source_download_retry_count),
                "--max-redirs",
                "5",
                "--netrc-file",
                "/dev/null",
                source.public_locator_uri,
            ),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            env=environment,
            close_fds=True,
        )
        if process.stdout is None:  # pragma: no cover - guaranteed by PIPE
            raise Item123AuthorizationError("source downloader has no bounded byte stream")
        while True:
            chunk = process.stdout.read(8 * 1024 * 1024)
            if not chunk:
                break
            byte_count += len(chunk)
            if byte_count > source.byte_count:
                raise Item123AuthorizationError("protected source exceeds its approved byte count")
            sha256.update(chunk)
            md5.update(chunk)
            offset = 0
            while offset < len(chunk):
                written = os.write(descriptor, chunk[offset:])
                if written <= 0:
                    raise Item123AuthorizationError("protected source write was incomplete")
                offset += written
        if process.wait(timeout=10.0) != 0:
            raise Item123AuthorizationError("protected source transfer failed")
        if (
            byte_count != source.byte_count
            or sha256.hexdigest() != source.sha256
            or md5.hexdigest() != source.md5
        ):
            raise Item123AuthorizationError("protected source identity differs from approval")
        os.fchmod(descriptor, 0o400)
        os.fsync(descriptor)
        identity = os.fstat(descriptor)
        parent = os.open(
            destination.parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            os.fsync(parent)
        finally:
            os.close(parent)
    except BaseException:
        if process is not None and process.poll() is None:
            process.kill()
            with contextlib.suppress(subprocess.TimeoutExpired):
                process.wait(timeout=10.0)
        raise
    finally:
        os.close(descriptor)
    _write_marker(
        Path(proposal.execution_paths.source_acquisition_completed_marker),
        {
            "artifact_schema": "cellstate-item12-3-source-acquisition-completed",
            "artifact_schema_version": "1.0.0",
            "attempt_key": capability.attempt_consumption.attempt_key,
            "byte_count": byte_count,
            "sha256": sha256.hexdigest(),
        },
    )
    return destination, identity


def _remove_source(
    capability: VerifiedItem123ExecutionCapability,
    source_path: Path,
    identity: os.stat_result | None,
) -> bool:
    removed = False
    try:
        if identity is not None:
            observed = source_path.lstat()
            if (
                not stat.S_ISREG(observed.st_mode)
                or observed.st_nlink != 1
                or (observed.st_dev, observed.st_ino) != (identity.st_dev, identity.st_ino)
            ):
                return False
            source_path.unlink()
        else:
            try:
                observed = source_path.lstat()
            except FileNotFoundError:
                pass
            else:
                if not stat.S_ISREG(observed.st_mode) or observed.st_nlink != 1:
                    return False
                source_path.unlink()
        parent = source_path.parent
        parent_descriptor = os.open(
            parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            os.fsync(parent_descriptor)
        finally:
            os.close(parent_descriptor)
        parent.rmdir()
        execution_root = Path(capability.proposal.execution_paths.execution_root)
        root_descriptor = os.open(
            execution_root,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            os.fsync(root_descriptor)
        finally:
            os.close(root_descriptor)
        removed = True
    except OSError:
        removed = False
    if removed:
        _write_marker(
            Path(capability.proposal.execution_paths.host_source_removed_marker),
            {
                "artifact_schema": "cellstate-item12-3-host-source-removed",
                "artifact_schema_version": "1.0.0",
                "attempt_key": capability.attempt_consumption.attempt_key,
            },
        )
    return removed


def _run(capability_path: Path, repository_root: Path) -> tuple[int, Item123TerminalReport]:
    root = Path(os.path.abspath(repository_root))  # noqa: PTH100 - lexical, no resolution
    capability = load_verified_capability(capability_path)
    verify_capability_for_execution(capability, repository_root=root)
    consumption_path = _consumption_path(capability)
    try:
        _reauthenticate_runtime(capability, root)
    except Exception:
        _, report = record_terminal_report(
            consumption_path,
            outcome="pre_source_failure",
            reason="runtime_distribution_failed",
            protected_source_acquired=False,
        )
        return 1, report
    _, execution_start = claim_execution_start(capability)
    source_identity: os.stat_result | None = None
    terminal: Any | None = None
    acquisition_completed = False
    try:
        _source_path, source_identity = _acquire_source(capability)
        acquisition_completed = True
        if __package__:
            from scripts.run_sciplex3_k562_v5_contained import run_contained_training
        else:  # pragma: no cover - direct CLI execution uses the sibling script directory.
            from run_sciplex3_k562_v5_contained import (  # type: ignore[import-not-found,no-redef]
                run_contained_training,
            )

        _, terminal = run_contained_training(
            capability=capability,
            execution_start=execution_start,
            repository_root=root,
            protected_source_acquired=True,
        )
    except BaseException:
        terminal = None
    acquisition_started = acquisition_completed or _source_acquisition_may_have_started(capability)
    if not acquisition_started:
        raise Item123AuthorizationError(
            "protected source did not start after the exclusive execution claim"
        )
    cleanup_removed = _remove_source(
        capability,
        Path(capability.proposal.execution_paths.protected_source_path),
        source_identity,
    )
    cleanup: Literal["removed", "failed"] = "removed" if cleanup_removed else "failed"
    if terminal is not None:
        _, report = record_contained_terminal_report(
            consumption_path,
            terminal,
            protected_source_acquired=True,
            runtime_preparation_observation_sha256=(
                capability.runtime_preparation_observation_sha256
            ),
            protected_source_cleanup_disposition=cleanup,
        )
        return (0 if report.outcome == "success" else 1), report
    _, report = record_terminal_report(
        consumption_path,
        outcome="runtime_failure",
        reason=(
            "contained_execution_failed"
            if acquisition_completed
            else "protected_source_acquisition_failed"
        ),
        protected_source_acquired=True,
        protected_source_acquisition=(
            "completed" if acquisition_completed else "started_incomplete"
        ),
        protected_source_cleanup_disposition=(
            "failed" if cleanup == "failed" else "unknown" if acquisition_completed else "removed"
        ),
        host_source_cleanup_disposition=cleanup,
        container_cleanup_disposition=("unproved" if acquisition_completed else "not_started"),
        snapshot_volume_cleanup_disposition=(
            "unproved" if acquisition_completed else "not_started"
        ),
        contained_terminal_status=(
            "supervisor_failure" if acquisition_completed else "not_started"
        ),
        contained_failure_code=(
            "contained_executor_failed" if acquisition_completed else "not_applicable"
        ),
        execution_outcome=("unavailable" if acquisition_completed else "not_started"),
        runtime_preparation_observation_sha256=(capability.runtime_preparation_observation_sha256),
        source_match_disposition="acquired_unverified",
        worker_report_status="unknown" if acquisition_completed else "not_started",
        stage_disposition="unknown" if acquisition_completed else "not_created",
        stage_semantic_verification="unknown" if acquisition_completed else "not_attempted",
        stage_preserved_on_runner=None if acquisition_completed else False,
    )
    return 1, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verified-capability", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    args = parser.parse_args()
    try:
        return_code, report = _run(args.verified_capability, args.repository_root)
        payload = canonical_json_bytes(report.model_dump(mode="json"))
        if len(payload) > ITEM12_3_MAX_TERMINAL_REPORT_BYTES:
            raise Item123AuthorizationError("terminal report exceeds its upload bound")
        print(payload.decode("utf-8"))
    except Item123AuthorizationError:
        print("item12_3_execution_denied", file=sys.stderr)
        raise SystemExit(1) from None
    except Exception:
        print("item12_3_execution_failed_closed", file=sys.stderr)
        raise SystemExit(1) from None
    raise SystemExit(return_code)


if __name__ == "__main__":
    main()
