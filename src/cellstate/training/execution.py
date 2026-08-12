"""Parent-owned OCI containment for source-touching training workers.

The executor deliberately stops at an isolated staging directory.  It never publishes model
artifacts.  Docker supplies one aggregate cgroup for the complete worker process tree; the host
supervisor owns the wall deadline, kills the container on expiry, waits for termination, and
removes the container before returning an observation.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import subprocess
import tempfile
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal, Protocol, Self

from pydantic import ConfigDict, Field, field_validator, model_validator

from cellstate.domain.common import SchemaModel, canonical_fingerprint, canonical_json_bytes

_HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_DIGEST = re.compile(r"^sha256:([0-9a-f]{64})$")
_EXECUTION_ID = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,62}$")
_CONTAINER_ID = re.compile(r"^[0-9a-f]{12,64}$")
_ANONYMOUS_VOLUME_ID = re.compile(r"^[0-9a-f]{64}$")
_ENVIRONMENT_KEY = re.compile(r"^[A-Z][A-Z0-9_]*$")
_LABEL_KEY = "org.cellstate.contained-execution.owner"
_POLICY_LABEL_KEY = "org.cellstate.contained-execution.policy-sha256"
_EXECUTION_LABEL_KEY = "org.cellstate.contained-execution.id"
WORKER_TERMINAL_REPORT_MAX_BYTES = 64 * 1024
PARENT_TERMINAL_REPORT_MAX_BYTES = 64 * 1024


class ContainedExecutionError(RuntimeError):
    """Raised when the executor cannot prove containment or complete process-tree cleanup."""


class ExecutionStageAlreadyClaimed(ContainedExecutionError):
    """Raised before source work when another process owns the one-use execution stage."""


class ContainerCommandTimeout(TimeoutError):
    """Raised by a CLI adapter when one Docker command crosses its parent deadline."""


@dataclass(frozen=True, slots=True)
class ContainerCommandResult:
    """Sanitized subprocess result used by real and deterministic fake Docker CLIs."""

    returncode: int
    stdout: str = ""
    stderr: str = ""

    def __post_init__(self) -> None:
        if type(self.returncode) is not int:
            raise TypeError("container command return code must be an exact integer")
        if type(self.stdout) is not str or type(self.stderr) is not str:
            raise TypeError("container command output must be text")


class ContainerCLI(Protocol):
    """Injectable Docker command boundary."""

    def run(
        self,
        command: Sequence[str],
        *,
        timeout_seconds: float | None = None,
    ) -> ContainerCommandResult: ...


class SubprocessDockerCLI:
    """Non-shell Docker CLI adapter with a parent-enforced timeout on every blocking call."""

    def run(
        self,
        command: Sequence[str],
        *,
        timeout_seconds: float | None = None,
    ) -> ContainerCommandResult:
        try:
            completed = subprocess.run(
                tuple(command),
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired as error:
            raise ContainerCommandTimeout("Docker command crossed its parent deadline") from error
        except OSError as error:
            raise ContainedExecutionError("Docker CLI is unavailable") from error
        return ContainerCommandResult(completed.returncode, completed.stdout, completed.stderr)


class RuntimeImageIdentity(SchemaModel):
    """An immutable OCI image reference; mutable tags are never accepted."""

    model_config = ConfigDict(strict=True)

    reference: str = Field(min_length=1)
    digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    platform: Literal["linux/amd64"] = "linux/amd64"

    @model_validator(mode="after")
    def reference_is_digest_pinned(self) -> Self:
        if type(self.reference) is not str or self.reference.count("@") != 1:
            raise ValueError("runtime image reference must contain exactly one digest separator")
        name, digest = self.reference.rsplit("@", 1)
        if not name or digest != self.digest or _DIGEST.fullmatch(digest) is None:
            raise ValueError("runtime image reference must be pinned to its exact SHA-256 digest")
        if any(character.isspace() for character in self.reference):
            raise ValueError("runtime image reference must not contain whitespace")
        return self


class RuntimeBuilderIdentity(SchemaModel):
    """Exact source-free builder and command semantics for one runtime artifact."""

    model_config = ConfigDict(strict=True)

    buildx_version: str = Field(pattern=r"^v[0-9]+\.[0-9]+\.[0-9]+$")
    buildx_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    buildkit_version: str = Field(pattern=r"^v[0-9]+\.[0-9]+\.[0-9]+$")
    buildkit_image_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    dockerfile_frontend_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    dockerfile_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    requirements_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_date_epoch: int = Field(gt=0)
    no_cache: bool
    platform: Literal["linux/amd64"] = "linux/amd64"
    provenance_attestation_disabled: bool
    image_tag: str = Field(min_length=1)
    output_options: tuple[str, ...] = Field(min_length=1)

    @field_validator("source_date_epoch")
    @classmethod
    def epoch_is_exact(cls, value: int) -> int:
        if type(value) is not int:
            raise ValueError("runtime source-date epoch must be an exact integer")
        return value

    @field_validator("output_options")
    @classmethod
    def output_options_are_canonical(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(
            type(option) is not str or not option or option.strip() != option for option in value
        ):
            raise ValueError("runtime build options must be non-empty canonical text")
        if value != tuple(sorted(value)) or len(value) != len(set(value)):
            raise ValueError("runtime build options must be unique and sorted")
        return value

    @model_validator(mode="after")
    def immutable_build_semantics_are_enabled(self) -> Self:
        if self.no_cache is not True:
            raise ValueError("runtime build must disable the cache")
        if self.provenance_attestation_disabled is not True:
            raise ValueError("runtime build must disable manifest-changing provenance attestations")
        if any(character.isspace() for character in self.image_tag):
            raise ValueError("runtime image tag must not contain whitespace")
        return self


class RuntimeImageLayerIdentity(SchemaModel):
    """One exact ordered blob in the runnable OCI image layer closure."""

    model_config = ConfigDict(strict=True)

    media_type: Literal["application/vnd.oci.image.layer.v1.tar+gzip"] = (
        "application/vnd.oci.image.layer.v1.tar+gzip"
    )
    digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    byte_count: int = Field(gt=0)

    @field_validator("byte_count")
    @classmethod
    def byte_count_is_exact(cls, value: int) -> int:
        if type(value) is not int:
            raise ValueError("runtime image layer byte count must be an exact integer")
        return value


class RuntimeImageLock(SchemaModel):
    """Canonical image lock bound to the complete executable training-code closure."""

    model_config = ConfigDict(strict=True)

    artifact_schema: Literal["cellstate-contained-training-runtime-image-lock"] = (
        "cellstate-contained-training-runtime-image-lock"
    )
    artifact_schema_version: Literal["1.0.0"] = "1.0.0"
    runtime_image: RuntimeImageIdentity
    runtime_entrypoint: Literal["/usr/bin/timeout"] = "/usr/bin/timeout"
    container_user_mode: Literal["host-effective-uid-gid"] = "host-effective-uid-gid"
    snapshot_volume_initialization: Literal["empty-image-directory-mode-1777"] = (
        "empty-image-directory-mode-1777"
    )
    builder: RuntimeBuilderIdentity
    archive_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    oci_index_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    config_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    layers: tuple[RuntimeImageLayerIdentity, ...] = Field(min_length=1)
    training_code_closure_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    image_provenance_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator(
        "archive_sha256",
        "training_code_closure_sha256",
        "image_provenance_sha256",
    )
    @classmethod
    def code_closure_digest_is_canonical(cls, value: str) -> str:
        if type(value) is not str or _HEX_SHA256.fullmatch(value) is None:
            raise ValueError("runtime lock SHA-256 must be lowercase canonical hex")
        return value

    @model_validator(mode="after")
    def layer_closure_is_exact(self) -> Self:
        digests = tuple(layer.digest for layer in self.layers)
        if len(digests) != len(set(digests)):
            raise ValueError("runtime image layer digests must be unique")
        return self

    @property
    def fingerprint(self) -> str:
        return canonical_fingerprint(self.model_dump(mode="json"))


class TrainingCodeClosureEntry(SchemaModel):
    """One exact regular file in a contained-training closure."""

    model_config = ConfigDict(strict=True)

    relative_path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    byte_count: int = Field(ge=0)

    @field_validator("relative_path")
    @classmethod
    def path_is_canonical(cls, value: str) -> str:
        if type(value) is not str or "\x00" in value or "\\" in value:
            raise ValueError("training-code path must be canonical relative POSIX text")
        path = PurePosixPath(value)
        if (
            not value
            or path.is_absolute()
            or path.as_posix() != value
            or any(part in {"", ".", ".."} for part in path.parts)
        ):
            raise ValueError("training-code path must remain inside its closure root")
        return value

    @field_validator("sha256")
    @classmethod
    def digest_is_canonical(cls, value: str) -> str:
        if type(value) is not str or _HEX_SHA256.fullmatch(value) is None:
            raise ValueError("training-code SHA-256 must be lowercase canonical hex")
        return value

    @field_validator("byte_count")
    @classmethod
    def byte_count_is_exact(cls, value: int) -> int:
        if type(value) is not int:
            raise ValueError("training-code byte count must be an exact integer")
        return value


class TrainingCodeClosureManifest(SchemaModel):
    """Canonical complete Python-source inventory for one contained fit."""

    model_config = ConfigDict(strict=True)

    artifact_schema: Literal["cellstate-contained-training-code-closure"] = (
        "cellstate-contained-training-code-closure"
    )
    artifact_schema_version: Literal["1.0.0"] = "1.0.0"
    entries: tuple[TrainingCodeClosureEntry, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def entries_are_closed(self) -> Self:
        paths = tuple(entry.relative_path for entry in self.entries)
        if paths != tuple(sorted(paths)) or len(paths) != len(set(paths)):
            raise ValueError("training-code closure paths must be unique and sorted")
        return self

    @property
    def fingerprint(self) -> str:
        return canonical_fingerprint(self.model_dump(mode="json"))


class ExecutionInputClosureManifest(SchemaModel):
    """Exact mounted tree: the code closure plus every declared public control input."""

    model_config = ConfigDict(strict=True)

    artifact_schema: Literal["cellstate-contained-training-execution-input-closure"] = (
        "cellstate-contained-training-execution-input-closure"
    )
    artifact_schema_version: Literal["1.0.0"] = "1.0.0"
    training_code_closure_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    entries: tuple[TrainingCodeClosureEntry, ...] = Field(min_length=1)

    @field_validator("training_code_closure_sha256")
    @classmethod
    def code_closure_digest_is_canonical(cls, value: str) -> str:
        if type(value) is not str or _HEX_SHA256.fullmatch(value) is None:
            raise ValueError("execution-input code-closure SHA-256 is not canonical")
        return value

    @model_validator(mode="after")
    def entries_are_closed(self) -> Self:
        paths = tuple(entry.relative_path for entry in self.entries)
        if paths != tuple(sorted(paths)) or len(paths) != len(set(paths)):
            raise ValueError("execution-input closure paths must be unique and sorted")
        return self

    @property
    def fingerprint(self) -> str:
        return canonical_fingerprint(self.model_dump(mode="json"))


_STAGE_OBSERVATION_EXCLUSIONS = (
    "contained-training-observation.json",
    "contained-worker-observation.json",
)
_STAGED_ARTIFACT_ROLES = {
    "candidate-model.json": "model_artifact",
    "candidate-training-plan.json": "training_plan",
    "p1-assembly-receipt.json": "p1_assembly_receipt",
    "p1-finalized-count-scan-receipt.json": "p1_finalized_count_scan",
    "training-execution-observation.json": "training_result",
}


def _staged_artifact_role(relative_path: str) -> str:
    return _STAGED_ARTIFACT_ROLES.get(relative_path, "support")


class StagedTrainingEntry(SchemaModel):
    """One exact regular file emitted by the worker before observation metadata."""

    model_config = ConfigDict(strict=True)

    relative_path: str = Field(min_length=1)
    artifact_role: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    byte_count: int = Field(ge=0)

    @field_validator("relative_path")
    @classmethod
    def path_is_canonical(cls, value: str) -> str:
        if type(value) is not str or "\x00" in value or "\\" in value:
            raise ValueError("staged training path must be canonical relative POSIX text")
        path = PurePosixPath(value)
        if (
            not value
            or path.is_absolute()
            or path.as_posix() != value
            or any(part in {"", ".", ".."} for part in path.parts)
            or value in _STAGE_OBSERVATION_EXCLUSIONS
        ):
            raise ValueError("staged training path escapes or collides with observation metadata")
        return value

    @field_validator("sha256")
    @classmethod
    def digest_is_canonical(cls, value: str) -> str:
        if type(value) is not str or _HEX_SHA256.fullmatch(value) is None:
            raise ValueError("staged training SHA-256 is not canonical")
        return value

    @model_validator(mode="after")
    def role_matches_path(self) -> Self:
        if self.artifact_role != _staged_artifact_role(self.relative_path):
            raise ValueError("staged training artifact role differs from its exact path")
        return self

    @field_validator("byte_count")
    @classmethod
    def byte_count_is_exact(cls, value: int) -> int:
        if type(value) is not int:
            raise ValueError("staged training byte count must be an exact integer")
        return value


class StagedTrainingInventory(SchemaModel):
    """Canonical worker/parent-shared inventory with a frozen metadata exclusion rule."""

    model_config = ConfigDict(strict=True)

    artifact_schema: Literal["cellstate-contained-staged-training-tree"] = (
        "cellstate-contained-staged-training-tree"
    )
    artifact_schema_version: Literal["1.0.0"] = "1.0.0"
    excluded_observation_paths: tuple[str, str] = _STAGE_OBSERVATION_EXCLUSIONS
    entries: tuple[StagedTrainingEntry, ...]

    @model_validator(mode="after")
    def inventory_is_closed(self) -> Self:
        if self.excluded_observation_paths != _STAGE_OBSERVATION_EXCLUSIONS:
            raise ValueError("staged inventory observation exclusions are not exact")
        paths = tuple(entry.relative_path for entry in self.entries)
        if paths != tuple(sorted(paths)) or len(paths) != len(set(paths)):
            raise ValueError("staged inventory paths must be unique and sorted")
        return self

    @property
    def fingerprint(self) -> str:
        return canonical_fingerprint(self.model_dump(mode="json"))


def _read_stable_regular_file(path: Path, *, name: str) -> bytes:
    try:
        before = path.lstat()
    except OSError as error:
        raise ContainedExecutionError(f"cannot inspect {name}") from error
    if not stat.S_ISREG(before.st_mode):
        raise ContainedExecutionError(f"{name} is not a regular file")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
        try:
            opened = os.fstat(descriptor)
            if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
                raise ContainedExecutionError(f"{name} changed while opening")
            chunks: list[bytes] = []
            while chunk := os.read(descriptor, 1024 * 1024):
                chunks.append(chunk)
            after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
    except OSError as error:
        raise ContainedExecutionError(f"cannot read {name}") from error
    payload = b"".join(chunks)
    if (opened.st_dev, opened.st_ino, opened.st_size) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
    ) or opened.st_size != len(payload):
        raise ContainedExecutionError(f"{name} changed while reading")
    return payload


def inventory_staged_training_tree(root: Path) -> StagedTrainingInventory:
    """No-follow, same-FD inventory shared by the worker and source-free host supervisor."""

    directory = Path(root)
    try:
        root_state = directory.lstat()
    except OSError as error:
        raise ContainedExecutionError("cannot inspect staged training root") from error
    if not stat.S_ISDIR(root_state.st_mode):
        raise ContainedExecutionError("staged training root must be one real directory")
    entries: list[StagedTrainingEntry] = []
    for current, directory_names, file_names in os.walk(directory, followlinks=False):
        current_path = Path(current)
        for name in directory_names:
            child = current_path / name
            if not stat.S_ISDIR(child.lstat().st_mode):
                raise ContainedExecutionError("staged training tree contains a directory link")
        for name in file_names:
            path = current_path / name
            relative = path.relative_to(directory).as_posix()
            if relative in _STAGE_OBSERVATION_EXCLUSIONS:
                _read_stable_regular_file(path, name="staged observation metadata")
                continue
            payload = _read_stable_regular_file(path, name="staged training artifact")
            entries.append(
                StagedTrainingEntry(
                    relative_path=relative,
                    artifact_role=_staged_artifact_role(relative),
                    sha256=hashlib.sha256(payload).hexdigest(),
                    byte_count=len(payload),
                )
            )
    return StagedTrainingInventory(
        entries=tuple(sorted(entries, key=lambda item: item.relative_path))
    )


def seal_staged_training_tree(
    root: Path,
    *,
    expected_inventory: StagedTrainingInventory,
) -> None:
    """Seal the parent-reverified stage and prove its inventory did not change."""

    directory = Path(root)
    if inventory_staged_training_tree(directory) != expected_inventory:
        raise ContainedExecutionError("staged training tree changed before sealing")
    directories = [directory]
    files: list[Path] = []
    for current, names, file_names in os.walk(directory, followlinks=False):
        current_path = Path(current)
        for name in names:
            child = current_path / name
            if not stat.S_ISDIR(child.lstat().st_mode):
                raise ContainedExecutionError("cannot seal staged directory link")
            directories.append(child)
        files.extend(current_path / name for name in file_names)
    for path in files:
        if not stat.S_ISREG(path.lstat().st_mode):
            raise ContainedExecutionError("cannot seal staged non-regular file")
        path.chmod(0o444)
    for path in sorted(directories, key=lambda item: len(item.parts), reverse=True):
        path.chmod(0o555)
    if inventory_staged_training_tree(directory) != expected_inventory:
        raise ContainedExecutionError("staged training tree changed while sealing")


class CanonicalPublicationTreeIdentity(SchemaModel):
    """Content identity for a no-follow snapshot of the canonical publication tree."""

    model_config = ConfigDict(strict=True)

    artifact_schema: Literal["cellstate-canonical-publication-tree-identity"] = (
        "cellstate-canonical-publication-tree-identity"
    )
    artifact_schema_version: Literal["1.0.0"] = "1.0.0"
    state: Literal["absent", "present"]
    tree_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    directory_count: int = Field(ge=0)
    file_count: int = Field(ge=0)
    byte_count: int = Field(ge=0)

    @field_validator("directory_count", "file_count", "byte_count")
    @classmethod
    def counts_are_exact(cls, value: int) -> int:
        if type(value) is not int:
            raise ValueError("canonical-publication counts must be exact integers")
        return value

    @model_validator(mode="after")
    def absent_tree_is_empty(self) -> Self:
        if self.state == "absent" and (
            self.directory_count != 0 or self.file_count != 0 or self.byte_count != 0
        ):
            raise ValueError("an absent canonical-publication tree cannot contain entries")
        if self.state == "present" and self.directory_count < 1:
            raise ValueError("a present canonical-publication tree must contain its root")
        return self


def canonical_publication_tree_identity(root: Path) -> CanonicalPublicationTreeIdentity:
    """Hash one canonical publication tree without following links or unstable file handles."""

    directory = Path(root)
    if (
        not directory.is_absolute()
        or ".." in directory.parts
        or Path(os.path.normpath(os.fspath(directory))) != directory
    ):
        raise ContainedExecutionError(
            "canonical-publication identity requires one normalized absolute path"
        )
    current = Path(directory.anchor)
    missing = False
    for part in directory.parts[1:]:
        current /= part
        if missing:
            continue
        try:
            observed = current.lstat()
        except FileNotFoundError:
            missing = True
            continue
        except OSError as error:
            raise ContainedExecutionError(
                "cannot inspect canonical-publication identity ancestors"
            ) from error
        if stat.S_ISLNK(observed.st_mode):
            raise ContainedExecutionError(
                "canonical-publication identity must not use a symlinked ancestor"
            )
        if current != directory and not stat.S_ISDIR(observed.st_mode):
            raise ContainedExecutionError(
                "canonical-publication identity has a non-directory ancestor"
            )
    if missing:
        summary: dict[str, object] = {"state": "absent", "entries": ()}
        return CanonicalPublicationTreeIdentity(
            state="absent",
            tree_sha256=canonical_fingerprint(summary),
            directory_count=0,
            file_count=0,
            byte_count=0,
        )
    try:
        root_state = directory.lstat()
    except OSError as error:  # pragma: no cover - ancestor loop already read the root
        raise ContainedExecutionError("cannot inspect canonical-publication root") from error
    if not stat.S_ISDIR(root_state.st_mode):
        raise ContainedExecutionError("canonical-publication root must be one real directory")

    entries: list[dict[str, object]] = []
    byte_count = 0
    directory_count = 0
    file_count = 0
    for current_name, directory_names, file_names in os.walk(directory, followlinks=False):
        current_path = Path(current_name)
        relative_directory = current_path.relative_to(directory).as_posix() or "."
        current_state = current_path.lstat()
        if not stat.S_ISDIR(current_state.st_mode):  # pragma: no cover - guarded by os.walk
            raise ContainedExecutionError("canonical-publication walk left the real tree")
        entries.append(
            {
                "kind": "directory",
                "mode": stat.S_IMODE(current_state.st_mode),
                "path": relative_directory,
            }
        )
        directory_count += 1
        for name in directory_names:
            child = current_path / name
            if not stat.S_ISDIR(child.lstat().st_mode):
                raise ContainedExecutionError(
                    "canonical-publication tree contains a directory link"
                )
        for name in file_names:
            path = current_path / name
            payload = _read_stable_regular_file(path, name="canonical-publication regular file")
            state = path.lstat()
            relative_path = path.relative_to(directory).as_posix()
            entries.append(
                {
                    "byte_count": len(payload),
                    "kind": "file",
                    "mode": stat.S_IMODE(state.st_mode),
                    "path": relative_path,
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
            )
            byte_count += len(payload)
            file_count += 1
    entries.sort(key=lambda item: str(item["path"]))
    summary = {
        "state": "present",
        "entries": tuple(entries),
    }
    return CanonicalPublicationTreeIdentity(
        state="present",
        tree_sha256=canonical_fingerprint(summary),
        directory_count=directory_count,
        file_count=file_count,
        byte_count=byte_count,
    )


def _canonical_container_path(value: str, *, name: str) -> str:
    if type(value) is not str or not value or "\x00" in value or "\\" in value:
        raise ValueError(f"{name} must be one canonical absolute POSIX path")
    path = PurePosixPath(value)
    if not path.is_absolute() or path.as_posix() != value or ".." in path.parts:
        raise ValueError(f"{name} must be one canonical absolute POSIX path")
    return value


class ContainedExecutionPolicy(SchemaModel):
    """Frozen whole-container resource, isolation, mount, and runtime identity."""

    model_config = ConfigDict(strict=True)

    artifact_schema: Literal["cellstate-contained-execution-policy"] = (
        "cellstate-contained-execution-policy"
    )
    artifact_schema_version: Literal["1.0.0"] = "1.0.0"
    policy_id: str = Field(min_length=1)
    owner_id: str = Field(min_length=1)
    executor: Literal["docker"] = "docker"
    runtime_image: RuntimeImageIdentity
    runtime_entrypoint: Literal["/usr/bin/timeout"] = "/usr/bin/timeout"
    container_user_mode: Literal["host-effective-uid-gid"] = "host-effective-uid-gid"
    snapshot_volume_initialization: Literal["empty-image-directory-mode-1777"] = (
        "empty-image-directory-mode-1777"
    )
    training_code_closure_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    execution_input_closure_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    wall_clock_seconds: int = Field(gt=0)
    cleanup_timeout_seconds: int = Field(gt=0)
    memory_max_bytes: int = Field(gt=0)
    memory_swap_max_bytes: int = Field(gt=0)
    pids_limit: int = Field(gt=1)
    temporary_max_bytes: int = Field(gt=0)
    snapshot_max_bytes: int = Field(gt=0)
    observed_training_peak_memory_bytes: int = Field(gt=0)
    network_mode: Literal["none"] = "none"
    read_only_root_filesystem: Literal[True] = True
    source_mount_read_only: Literal[True] = True
    code_mount_read_only: Literal[True] = True
    output_mount_read_write: Literal[True] = True
    snapshot_mount_read_write: Literal[True] = True
    cap_drop_all: Literal[True] = True
    no_new_privileges: Literal[True] = True
    init_process: Literal[True] = True
    source_container_path: str
    code_container_path: str
    output_container_path: str
    snapshot_container_path: str
    temporary_container_path: str
    workdir: str
    environment: Mapping[str, str] = Field(default_factory=dict)
    worker_command: tuple[str, ...] = Field(min_length=1)

    @field_validator("policy_id", "owner_id")
    @classmethod
    def identifiers_are_canonical(cls, value: str) -> str:
        if type(value) is not str or _EXECUTION_ID.fullmatch(value) is None:
            raise ValueError("execution policy identifiers must be canonical label-safe text")
        return value

    @field_validator("training_code_closure_sha256", "execution_input_closure_sha256")
    @classmethod
    def training_code_closure_is_canonical(cls, value: str) -> str:
        if type(value) is not str or _HEX_SHA256.fullmatch(value) is None:
            raise ValueError("execution policy code closure must be lowercase canonical hex")
        return value

    @field_validator(
        "source_container_path",
        "code_container_path",
        "output_container_path",
        "snapshot_container_path",
        "temporary_container_path",
        "workdir",
    )
    @classmethod
    def container_paths_are_canonical(cls, value: str, info: object) -> str:
        field_name = getattr(info, "field_name", "container path")
        return _canonical_container_path(value, name=str(field_name))

    @field_validator(
        "wall_clock_seconds",
        "cleanup_timeout_seconds",
        "memory_max_bytes",
        "memory_swap_max_bytes",
        "pids_limit",
        "temporary_max_bytes",
        "snapshot_max_bytes",
        "observed_training_peak_memory_bytes",
    )
    @classmethod
    def integers_are_exact(cls, value: int) -> int:
        if type(value) is not int:
            raise ValueError("execution limits must be exact integers")
        return value

    @field_validator("worker_command")
    @classmethod
    def worker_command_is_exact(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if type(value) is not tuple or any(
            type(item) is not str or not item or "\x00" in item for item in value
        ):
            raise ValueError("worker command must be a nonempty exact argument vector")
        return value

    @model_validator(mode="after")
    def limits_and_isolation_are_coherent(self) -> Self:
        if self.memory_swap_max_bytes != self.memory_max_bytes:
            raise ValueError("execution policy must disable additional swap above memory.max")
        mount_paths = tuple(
            PurePosixPath(value)
            for value in (
                self.source_container_path,
                self.code_container_path,
                self.output_container_path,
                self.snapshot_container_path,
                self.temporary_container_path,
            )
        )
        if any(
            left == right or left in right.parents or right in left.parents
            for index, left in enumerate(mount_paths)
            for right in mount_paths[index + 1 :]
        ):
            raise ValueError(
                "source, code, output, snapshot, and temporary mounts must not overlap"
            )
        workdir = PurePosixPath(self.workdir)
        mutable_mounts = (
            PurePosixPath(self.source_container_path),
            PurePosixPath(self.output_container_path),
            PurePosixPath(self.snapshot_container_path),
            PurePosixPath(self.temporary_container_path),
        )
        if any(workdir == mount or mount in workdir.parents for mount in mutable_mounts):
            raise ValueError("workdir must not be inside a source, output, or temporary mount")
        code_mount = PurePosixPath(self.code_container_path)
        if workdir != code_mount and code_mount not in workdir.parents:
            raise ValueError("workdir must remain on the explicit read-only code mount")
        if self.temporary_max_bytes > self.memory_max_bytes:
            raise ValueError("bounded temporary storage cannot exceed container memory.max")
        if self.observed_training_peak_memory_bytes >= self.memory_max_bytes:
            raise ValueError("observed training peak must leave positive cgroup headroom")
        environment = dict(self.environment)
        if any(
            type(key) is not str
            or _ENVIRONMENT_KEY.fullmatch(key) is None
            or type(value) is not str
            or not value
            or "\x00" in value
            for key, value in environment.items()
        ):
            raise ValueError("execution environment must contain exact canonical text entries")
        if tuple(environment) != tuple(sorted(environment)):
            raise ValueError("execution environment keys must be canonically sorted")
        if environment.get("TMPDIR") != self.temporary_container_path:
            raise ValueError("TMPDIR must resolve only inside the bounded writable tmpfs")
        return self

    @property
    def fingerprint(self) -> str:
        return canonical_fingerprint(self.model_dump(mode="json"))

    @classmethod
    def from_canonical_json(cls, payload: bytes) -> ContainedExecutionPolicy:
        try:
            policy = cls.model_validate_json(payload)
        except ValueError as error:
            raise ContainedExecutionError("execution policy is invalid") from error
        if canonical_json_bytes(policy.model_dump(mode="json")) != payload:
            raise ContainedExecutionError("execution policy is not canonical JSON")
        return policy


class ContainedExecutionObservation(SchemaModel):
    """Sanitized parent observation; it grants no publication or lifecycle authority."""

    model_config = ConfigDict(strict=True)

    artifact_schema: Literal["cellstate-contained-execution-observation"] = (
        "cellstate-contained-execution-observation"
    )
    artifact_schema_version: Literal["1.1.0"] = "1.1.0"
    execution_id: str = Field(min_length=1)
    policy_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    runtime_image_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    container_user_mode: Literal["host-effective-uid-gid"]
    observed_container_uid: int = Field(ge=0)
    observed_container_gid: int = Field(ge=0)
    outcome: Literal["success", "timeout", "oom_killed", "worker_failure"]
    exit_code: int
    timed_out: bool
    worker_watchdog_timed_out: bool = False
    oom_killed: bool
    parent_wall_clock_elapsed_seconds: float = Field(ge=0.0)
    container_removed: Literal[True] = True
    snapshot_volume_removed: Literal[True] = True
    process_tree_cleaned: Literal[True] = True
    canonical_publication_performed: Literal[False] = False

    @field_validator("execution_id")
    @classmethod
    def execution_id_is_canonical(cls, value: str) -> str:
        if type(value) is not str or _EXECUTION_ID.fullmatch(value) is None:
            raise ValueError("execution ID must be canonical label-safe text")
        return value

    @field_validator("policy_fingerprint")
    @classmethod
    def fingerprint_is_canonical(cls, value: str) -> str:
        if type(value) is not str or _HEX_SHA256.fullmatch(value) is None:
            raise ValueError("policy fingerprint must be lowercase canonical hex")
        return value

    @field_validator("exit_code", "observed_container_uid", "observed_container_gid")
    @classmethod
    def observation_integers_are_exact(cls, value: int) -> int:
        if type(value) is not int:
            raise ValueError("execution observation integers must be exact")
        return value

    @field_validator("parent_wall_clock_elapsed_seconds")
    @classmethod
    def elapsed_time_is_finite(cls, value: float) -> float:
        if type(value) is not float or not math.isfinite(value):
            raise ValueError("parent wall-clock observation must be one finite float")
        return value

    @model_validator(mode="after")
    def outcome_is_exact(self) -> Self:
        expected = (
            "timeout"
            if self.timed_out or self.worker_watchdog_timed_out
            else "oom_killed"
            if self.oom_killed
            else "success"
            if self.exit_code == 0
            else "worker_failure"
        )
        if (
            self.outcome != expected
            or (self.timed_out and self.worker_watchdog_timed_out)
            or ((self.timed_out or self.worker_watchdog_timed_out) and self.oom_killed)
            or (self.worker_watchdog_timed_out and self.exit_code not in {124, 137})
        ):
            raise ValueError("execution outcome contradicts timeout, OOM, or exit status")
        return self

    @property
    def fingerprint(self) -> str:
        return canonical_fingerprint(self.model_dump(mode="json"))


class ContainedTrainingWorkerObservation(SchemaModel):
    """Worker-side proof over the Docker-pinned source bind and staged fit closure."""

    model_config = ConfigDict(strict=True)

    artifact_schema: Literal["cellstate-contained-training-worker-observation"] = (
        "cellstate-contained-training-worker-observation"
    )
    artifact_schema_version: Literal["1.0.0"] = "1.0.0"
    execution_id: str = Field(min_length=1)
    training_plan_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    policy_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    runtime_image_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    training_code_closure_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    execution_input_closure_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    expected_source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_pre_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_post_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    expected_source_byte_count: int = Field(gt=0)
    source_pre_byte_count: int = Field(gt=0)
    source_post_byte_count: int = Field(gt=0)
    staged_inventory: StagedTrainingInventory
    staged_tree_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    staged_file_count: int = Field(gt=0)
    training_succeeded: Literal[True] = True
    source_verified_before_training: Literal[True] = True
    source_closed_after_post_verification: Literal[True] = True
    source_verified_after_training: Literal[True] = True
    canonical_publication_performed: Literal[False] = False

    @field_validator("execution_id")
    @classmethod
    def worker_execution_id_is_canonical(cls, value: str) -> str:
        if type(value) is not str or _EXECUTION_ID.fullmatch(value) is None:
            raise ValueError("worker execution ID must be canonical label-safe text")
        return value

    @field_validator(
        "training_plan_fingerprint",
        "policy_fingerprint",
        "training_code_closure_sha256",
        "execution_input_closure_sha256",
        "expected_source_sha256",
        "source_pre_sha256",
        "source_post_sha256",
        "staged_tree_sha256",
    )
    @classmethod
    def worker_digests_are_canonical(cls, value: str) -> str:
        if type(value) is not str or _HEX_SHA256.fullmatch(value) is None:
            raise ValueError("worker SHA-256 fields must be lowercase canonical hex")
        return value

    @field_validator(
        "expected_source_byte_count",
        "source_pre_byte_count",
        "source_post_byte_count",
        "staged_file_count",
    )
    @classmethod
    def worker_counts_are_exact(cls, value: int) -> int:
        if type(value) is not int:
            raise ValueError("worker counts must be exact integers")
        return value

    @model_validator(mode="after")
    def pinned_source_is_reauthenticated(self) -> Self:
        if not (
            self.source_pre_sha256 == self.source_post_sha256 == self.expected_source_sha256
        ) or not (
            self.source_pre_byte_count
            == self.source_post_byte_count
            == self.expected_source_byte_count
        ):
            raise ValueError("worker did not reauthenticate the same expected pinned source")
        if (
            self.staged_tree_sha256 != self.staged_inventory.fingerprint
            or self.staged_file_count != len(self.staged_inventory.entries)
        ):
            raise ValueError("worker staged-tree summary differs from its typed inventory")
        return self

    @property
    def fingerprint(self) -> str:
        return canonical_fingerprint(self.model_dump(mode="json"))


class ContainedTrainingWorkerTerminalReport(SchemaModel):
    """Bounded worker report with fixed failure codes and no exception or path text."""

    model_config = ConfigDict(strict=True)

    artifact_schema: Literal["cellstate-contained-training-worker-terminal-report"] = (
        "cellstate-contained-training-worker-terminal-report"
    )
    artifact_schema_version: Literal["1.0.0"] = "1.0.0"
    execution_id: str = Field(min_length=1)
    outcome: Literal["success", "worker_failure"]
    terminal_phase: Literal[
        "source_authentication",
        "contract_authentication",
        "preparation",
        "training",
        "stage_finalization",
        "completed",
    ]
    failure_code: Literal[
        "none",
        "source_authentication_failed",
        "contract_authentication_failed",
        "preparation_failed",
        "training_failed",
        "stage_finalization_failed",
        "source_post_authentication_failed",
        "stage_inventory_failed",
    ]
    expected_source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    expected_source_byte_count: int = Field(gt=0)
    source_pre_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    source_pre_byte_count: int | None = Field(default=None, ge=0)
    source_post_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    source_post_byte_count: int | None = Field(default=None, ge=0)
    source_access_started: bool
    protected_source_acquired: bool
    source_post_authentication_attempted: bool
    source_post_authentication_completed: bool
    source_matches_expected: bool
    staged_inventory_status: Literal["verified", "unavailable"]
    staged_tree_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    staged_file_count: int | None = Field(default=None, ge=0)
    success_observation: ContainedTrainingWorkerObservation | None = None
    exception_detail_recorded: Literal[False] = False
    canonical_publication_performed: Literal[False] = False

    @field_validator("execution_id")
    @classmethod
    def terminal_execution_id_is_canonical(cls, value: str) -> str:
        if type(value) is not str or _EXECUTION_ID.fullmatch(value) is None:
            raise ValueError("worker terminal execution ID must be canonical label-safe text")
        return value

    @field_validator("expected_source_byte_count", "staged_file_count")
    @classmethod
    def terminal_counts_are_exact(cls, value: int | None) -> int | None:
        if value is not None and type(value) is not int:
            raise ValueError("worker terminal counts must be exact integers")
        return value

    @model_validator(mode="after")
    def terminal_report_is_bounded_and_coherent(self) -> Self:
        pre_complete = self.source_pre_sha256 is not None and self.source_pre_byte_count is not None
        post_complete = (
            self.source_post_sha256 is not None and self.source_post_byte_count is not None
        )
        if (self.source_pre_sha256 is None) != (self.source_pre_byte_count is None) or (
            self.source_post_sha256 is None
        ) != (self.source_post_byte_count is None):
            raise ValueError("worker terminal source hash/count pairs must be complete")
        if self.protected_source_acquired and not self.source_access_started:
            raise ValueError("worker terminal source acquisition predates source access")
        if self.source_access_started != self.protected_source_acquired:
            raise ValueError("worker terminal source-access disposition is not conservative")
        if (post_complete and not self.source_post_authentication_attempted) or (
            self.source_post_authentication_attempted and not self.protected_source_acquired
        ):
            raise ValueError("worker terminal post-authentication attempt is contradictory")
        if self.source_post_authentication_completed != post_complete:
            raise ValueError("worker terminal post-authentication disposition is contradictory")
        exact_source = (
            pre_complete
            and post_complete
            and self.source_pre_sha256 == self.source_post_sha256 == self.expected_source_sha256
            and self.source_pre_byte_count
            == self.source_post_byte_count
            == self.expected_source_byte_count
        )
        if self.source_matches_expected != exact_source:
            raise ValueError("worker terminal source identity disposition is contradictory")
        if self.staged_inventory_status == "verified":
            if self.staged_tree_sha256 is None or self.staged_file_count is None:
                raise ValueError("verified worker terminal stage lacks its exact summary")
        elif self.staged_tree_sha256 is not None or self.staged_file_count is not None:
            raise ValueError("unavailable worker terminal stage cannot claim an exact summary")
        if self.outcome == "success":
            success = self.success_observation
            if (
                self.terminal_phase != "completed"
                or self.failure_code != "none"
                or not self.source_access_started
                or not self.protected_source_acquired
                or not self.source_post_authentication_attempted
                or not exact_source
                or self.staged_inventory_status != "verified"
                or success is None
                or success.execution_id != self.execution_id
                or success.staged_tree_sha256 != self.staged_tree_sha256
                or success.staged_file_count != self.staged_file_count
            ):
                raise ValueError("successful worker terminal report is incomplete")
        elif (
            self.terminal_phase == "completed"
            or self.failure_code == "none"
            or self.success_observation is not None
        ):
            raise ValueError("failed worker terminal report claims success evidence")
        if (
            len(canonical_json_bytes(self.model_dump(mode="json")))
            > WORKER_TERMINAL_REPORT_MAX_BYTES
        ):
            raise ValueError("worker terminal report exceeds its exact byte bound")
        return self

    @property
    def fingerprint(self) -> str:
        return canonical_fingerprint(self.model_dump(mode="json"))


class ContainedTrainingObservation(SchemaModel):
    """Typed success proof joining worker byte evidence to parent process-tree containment."""

    model_config = ConfigDict(strict=True)

    artifact_schema: Literal["cellstate-contained-training-observation"] = (
        "cellstate-contained-training-observation"
    )
    artifact_schema_version: Literal["1.1.0"] = "1.1.0"
    training_plan_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    policy_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    runtime_image_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    training_code_closure_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    execution_input_closure_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    staged_inventory: StagedTrainingInventory
    staged_tree_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    worker_observation: ContainedTrainingWorkerObservation
    execution_observation: ContainedExecutionObservation
    wall_clock_limit_seconds: int = Field(gt=0)
    memory_max_bytes: int = Field(gt=0)
    memory_swap_max_bytes: int = Field(gt=0)
    aggregate_container_limits_enforced: Literal[True] = True
    canonical_publication_performed: Literal[False] = False

    @field_validator(
        "training_plan_fingerprint",
        "policy_fingerprint",
        "training_code_closure_sha256",
        "execution_input_closure_sha256",
        "staged_tree_sha256",
    )
    @classmethod
    def training_digests_are_canonical(cls, value: str) -> str:
        if type(value) is not str or _HEX_SHA256.fullmatch(value) is None:
            raise ValueError("contained-training SHA-256 fields must be lowercase canonical hex")
        return value

    @field_validator("wall_clock_limit_seconds", "memory_max_bytes", "memory_swap_max_bytes")
    @classmethod
    def resource_limits_are_exact(cls, value: int) -> int:
        if type(value) is not int:
            raise ValueError("contained-training resource limits must be exact integers")
        return value

    @model_validator(mode="after")
    def success_and_identity_are_exact(self) -> Self:
        worker = self.worker_observation
        execution = self.execution_observation
        if (
            execution.outcome != "success"
            or execution.exit_code != 0
            or execution.timed_out
            or execution.oom_killed
        ):
            raise ValueError("contained training observation requires a successful worker tree")
        if (
            self.memory_swap_max_bytes != self.memory_max_bytes
            or execution.parent_wall_clock_elapsed_seconds > self.wall_clock_limit_seconds
        ):
            raise ValueError("contained training resource evidence contradicts a successful fit")
        if (
            execution.execution_id != worker.execution_id
            or self.training_plan_fingerprint != worker.training_plan_fingerprint
            or not (
                self.policy_fingerprint == worker.policy_fingerprint == execution.policy_fingerprint
            )
            or not (
                self.runtime_image_digest
                == worker.runtime_image_digest
                == execution.runtime_image_digest
            )
            or self.training_code_closure_sha256 != worker.training_code_closure_sha256
            or self.execution_input_closure_sha256 != worker.execution_input_closure_sha256
            or self.staged_tree_sha256 != worker.staged_tree_sha256
            or self.staged_inventory != worker.staged_inventory
        ):
            raise ValueError("worker and parent containment identities do not form one execution")
        return self

    @property
    def fingerprint(self) -> str:
        return canonical_fingerprint(self.model_dump(mode="json"))


class ContainedTrainingTerminalObservation(SchemaModel):
    """Universal parent-owned terminal evidence for one non-retryable contained run."""

    model_config = ConfigDict(strict=True)

    artifact_schema: Literal["cellstate-contained-training-terminal-observation"] = (
        "cellstate-contained-training-terminal-observation"
    )
    artifact_schema_version: Literal["1.1.0"] = "1.1.0"
    execution_id: str = Field(min_length=1)
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
    policy_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    runtime_image_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    training_code_closure_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    execution_input_closure_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    wall_clock_limit_seconds: int = Field(gt=0)
    memory_max_bytes: int = Field(gt=0)
    memory_swap_max_bytes: int = Field(gt=0)
    aggregate_container_limits_enforced: bool
    container_cleanup_disposition: Literal["proved_removed", "unproved"]
    snapshot_volume_cleanup_disposition: Literal["proved_removed", "unproved"]
    protected_source_acquired_before_supervisor: Literal[True]
    execution_observation: ContainedExecutionObservation | None = None
    worker_report_status: Literal["verified", "missing", "invalid"]
    worker_terminal_report: ContainedTrainingWorkerTerminalReport | None = None
    staged_inventory: StagedTrainingInventory | None = None
    staged_tree_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    staged_file_count: int | None = Field(default=None, ge=0)
    stage_disposition: Literal["sealed", "quarantined"]
    stage_relative_path: Literal["output", "quarantine"]
    semantic_stage_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    success_observation: ContainedTrainingObservation | None = None
    canonical_publication_relative_path: Literal[
        "backends/vertical-a/sciplex3-k562-24h-v1/candidate-publication"
    ]
    canonical_publication_before: CanonicalPublicationTreeIdentity
    canonical_publication_after: CanonicalPublicationTreeIdentity | None = None
    canonical_publication_unchanged: bool
    retry_performed: Literal[False] = False
    canonical_publication_performed: Literal[False] = False
    evaluation_performed: Literal[False] = False
    lifecycle_evidence_minted: Literal[False] = False
    scientifically_admissible: Literal[False] = False

    @field_validator("execution_id")
    @classmethod
    def terminal_execution_id_is_canonical(cls, value: str) -> str:
        if type(value) is not str or _EXECUTION_ID.fullmatch(value) is None:
            raise ValueError("parent terminal execution ID must be canonical label-safe text")
        return value

    @field_validator("wall_clock_limit_seconds", "memory_max_bytes", "memory_swap_max_bytes")
    @classmethod
    def terminal_resource_limits_are_exact(cls, value: int) -> int:
        if type(value) is not int:
            raise ValueError("parent terminal resource limits must be exact integers")
        return value

    @field_validator("staged_file_count")
    @classmethod
    def terminal_stage_count_is_exact(cls, value: int | None) -> int | None:
        if value is not None and type(value) is not int:
            raise ValueError("parent terminal stage count must be an exact integer")
        return value

    @model_validator(mode="after")
    def terminal_evidence_is_closed(self) -> Self:
        execution = self.execution_observation
        if self.memory_swap_max_bytes != self.memory_max_bytes:
            raise ValueError("parent terminal memory limits differ")
        if self.aggregate_container_limits_enforced != (execution is not None):
            raise ValueError("parent terminal container-limit disposition is contradictory")
        expected_cleanup = "proved_removed" if execution is not None else "unproved"
        if (
            self.container_cleanup_disposition != expected_cleanup
            or self.snapshot_volume_cleanup_disposition != expected_cleanup
        ):
            raise ValueError("parent terminal cleanup disposition contradicts Docker evidence")
        if execution is not None and (
            execution.execution_id != self.execution_id
            or execution.policy_fingerprint != self.policy_fingerprint
            or execution.runtime_image_digest != self.runtime_image_digest
        ):
            raise ValueError("parent terminal execution and policy identities differ")
        if self.worker_report_status == "verified":
            if (
                self.worker_terminal_report is None
                or self.worker_terminal_report.execution_id != self.execution_id
            ):
                raise ValueError("verified worker terminal report is absent or mismatched")
        elif self.worker_terminal_report is not None:
            raise ValueError("unverified worker terminal report cannot enter parent evidence")
        if self.staged_inventory is None:
            if self.staged_tree_sha256 is not None or self.staged_file_count is not None:
                raise ValueError("unavailable parent stage cannot claim an exact summary")
        elif (
            self.staged_tree_sha256 != self.staged_inventory.fingerprint
            or self.staged_file_count != len(self.staged_inventory.entries)
        ):
            raise ValueError("parent terminal stage summary differs from its inventory")
        identities_equal = (
            self.canonical_publication_after is not None
            and self.canonical_publication_before == self.canonical_publication_after
        )
        if self.canonical_publication_unchanged != identities_equal:
            raise ValueError("canonical-publication before/after disposition is contradictory")
        if self.terminal_status == "success":
            worker = self.worker_terminal_report
            success = self.success_observation
            if (
                self.failure_code != "none"
                or execution is None
                or execution.outcome != "success"
                or execution.timed_out
                or execution.oom_killed
                or execution.parent_wall_clock_elapsed_seconds > self.wall_clock_limit_seconds
                or self.worker_report_status != "verified"
                or worker is None
                or worker.outcome != "success"
                or self.staged_inventory is None
                or self.stage_disposition != "sealed"
                or self.stage_relative_path != "output"
                or self.semantic_stage_sha256 is None
                or success is None
                or success.execution_observation != execution
                or success.wall_clock_limit_seconds != self.wall_clock_limit_seconds
                or success.memory_max_bytes != self.memory_max_bytes
                or success.memory_swap_max_bytes != self.memory_swap_max_bytes
                or not self.canonical_publication_unchanged
            ):
                raise ValueError("successful parent terminal evidence is incomplete")
        else:
            if (
                self.failure_code == "none"
                or self.stage_disposition != "quarantined"
                or self.stage_relative_path != "quarantine"
                or self.semantic_stage_sha256 is not None
                or self.success_observation is not None
            ):
                raise ValueError("failed parent terminal evidence did not quarantine the stage")
            if execution is None:
                if (
                    self.terminal_status != "supervisor_failure"
                    or self.failure_code != "contained_executor_failed"
                ):
                    raise ValueError("missing Docker evidence must be a supervisor failure")
            elif self.terminal_status == "supervisor_failure":
                raise ValueError("supervisor failure cannot invent Docker terminal evidence")
            elif self.terminal_status == "timeout":
                if execution.outcome != "timeout" or self.failure_code != "worker_timed_out":
                    raise ValueError("parent timeout evidence contradicts the Docker outcome")
            elif self.terminal_status == "oom_killed":
                if execution.outcome != "oom_killed" or self.failure_code != "worker_oom_killed":
                    raise ValueError("parent OOM evidence contradicts the Docker outcome")
            elif self.terminal_status == "worker_failure":
                if (
                    execution.outcome != "worker_failure"
                    or self.failure_code != "worker_exited_nonzero"
                ):
                    raise ValueError(
                        "parent worker-failure evidence contradicts the Docker outcome"
                    )
            elif self.terminal_status == "stage_rejected":
                if self.failure_code not in {
                    "worker_report_missing",
                    "worker_report_invalid",
                    "worker_report_contradiction",
                    "stage_inventory_invalid",
                    "stage_semantic_verification_failed",
                    "canonical_publication_changed",
                    "canonical_publication_identity_invalid",
                }:
                    raise ValueError("parent stage rejection has a non-stage failure code")
            else:  # pragma: no cover - Literal exhaustiveness defense
                raise ValueError("parent terminal status hides the Docker terminal outcome")
        if (
            len(canonical_json_bytes(self.model_dump(mode="json")))
            > PARENT_TERMINAL_REPORT_MAX_BYTES
        ):
            raise ValueError("parent terminal observation exceeds its exact byte bound")
        return self

    @property
    def fingerprint(self) -> str:
        return canonical_fingerprint(self.model_dump(mode="json"))


@dataclass(frozen=True, slots=True)
class _ContainerState:
    running: bool
    oom_killed: bool
    exit_code: int


def _docker_json(result: ContainerCommandResult, *, name: str) -> object:
    if result.returncode != 0:
        raise ContainedExecutionError(f"Docker failed while reading {name}")
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise ContainedExecutionError(f"Docker returned malformed {name}") from error


class DockerExecutor:
    """Exact Docker lifecycle with aggregate memory and parent-owned wall containment."""

    def __init__(
        self,
        policy: ContainedExecutionPolicy,
        *,
        cli: ContainerCLI | None = None,
        monotonic: object = time.monotonic,
        sleep: object = time.sleep,
        lock_root: Path | None = None,
        staging_root: Path | None = None,
        canonical_publication_root: Path | None = None,
        execution_input_closure: ExecutionInputClosureManifest | None = None,
    ) -> None:
        self.policy = policy
        self._cli = cli or SubprocessDockerCLI()
        effective_uid = os.geteuid()
        effective_gid = os.getegid()
        if type(effective_uid) is not int or effective_uid < 0:
            raise ContainedExecutionError("host effective UID is not canonical")
        if type(effective_gid) is not int or effective_gid < 0:
            raise ContainedExecutionError("host effective GID is not canonical")
        self._effective_uid = effective_uid
        self._effective_gid = effective_gid
        if not callable(monotonic):
            raise TypeError("monotonic clock must be callable")
        if not callable(sleep):
            raise TypeError("sleep must be callable")
        self._monotonic = monotonic
        self._sleep = sleep
        self._lock_root = Path(
            lock_root or Path(tempfile.gettempdir()) / "cellstate-execution-locks"
        )
        self._staging_root = Path(staging_root) if staging_root is not None else None
        self._canonical_publication_root = (
            Path(canonical_publication_root) if canonical_publication_root is not None else None
        )
        self._execution_input_closure = (
            ExecutionInputClosureManifest.model_validate(
                execution_input_closure.model_dump(mode="python")
            )
            if execution_input_closure is not None
            else None
        )
        self._claimed_output_stage: tuple[str, Path] | None = None
        if (
            self._execution_input_closure is not None
            and self._execution_input_closure.fingerprint
            != self.policy.execution_input_closure_sha256
        ):
            raise ContainedExecutionError("executor input closure differs from the frozen policy")

    def output_stage_path(self, execution_id: str) -> Path:
        """Return the only host output path this executor will create for an execution."""

        if _EXECUTION_ID.fullmatch(execution_id) is None:
            raise ContainedExecutionError("execution ID is not canonical")
        if self._staging_root is None:
            raise ContainedExecutionError("executor has no isolated staging root")
        return self._staging_root / self.policy.owner_id / execution_id / "output"

    def code_stage_path(self, execution_id: str) -> Path:
        """Return the executor-owned sealed code/input snapshot path for an execution."""

        if _EXECUTION_ID.fullmatch(execution_id) is None:
            raise ContainedExecutionError("execution ID is not canonical")
        if self._staging_root is None:
            raise ContainedExecutionError("executor has no isolated staging root")
        return self._staging_root / self.policy.owner_id / execution_id / "code"

    def owns_output_stage_claim(self, *, execution_id: str, output_path: Path) -> bool:
        """Report only this executor instance's successful exclusive stage claim."""

        return self._claimed_output_stage == (execution_id, Path(output_path))

    @staticmethod
    def _paths_overlap(left: Path, right: Path) -> bool:
        left_parts = left.parts
        right_parts = right.parts
        shared = min(len(left_parts), len(right_parts))
        return left_parts[:shared] == right_parts[:shared]

    @staticmethod
    def _no_symlink_realpath(path: Path, *, name: str) -> Path:
        """Reject every symlinked ancestor before returning a normalized physical path."""

        if not path.is_absolute() or ".." in path.parts:
            raise ContainedExecutionError(f"{name} must be one normalized absolute path")
        current = Path(path.anchor)
        missing = False
        for part in path.parts[1:]:
            current /= part
            if missing:
                continue
            try:
                observed = current.lstat()
            except FileNotFoundError:
                missing = True
                continue
            except OSError as error:
                raise ContainedExecutionError(f"cannot inspect {name} ancestors") from error
            if stat.S_ISLNK(observed.st_mode):
                raise ContainedExecutionError(f"{name} must not use a symlinked ancestor")
            if current != path and not stat.S_ISDIR(observed.st_mode):
                raise ContainedExecutionError(f"{name} has a non-directory ancestor")
        resolved = Path(os.path.realpath(path))
        if resolved != path:
            raise ContainedExecutionError(f"{name} does not have one exact physical path")
        return resolved

    @staticmethod
    def _tree_contains_inode(root: Path, *, device: int, inode: int) -> bool:
        if not root.exists():
            return False
        for current, names, file_names in os.walk(root, followlinks=False):
            current_path = Path(current)
            for name in (*names, *file_names):
                try:
                    observed = (current_path / name).lstat()
                except OSError as error:
                    raise ContainedExecutionError(
                        "cannot authenticate canonical-publication inode closure"
                    ) from error
                if stat.S_ISLNK(observed.st_mode):
                    raise ContainedExecutionError(
                        "canonical publication contains an untrusted symlink"
                    )
                if (observed.st_dev, observed.st_ino) == (device, inode):
                    return True
        return False

    def _owned_directory(self, path: Path, *, token_payload: bytes) -> None:
        """Create or reauthenticate one real mode-0700 directory with an exact owner token."""

        try:
            observed = path.lstat()
        except FileNotFoundError:
            try:
                path.mkdir(parents=True, mode=0o700)
            except FileExistsError:
                observed = path.lstat()
            else:
                observed = path.lstat()
        if not stat.S_ISDIR(observed.st_mode) or stat.S_IMODE(observed.st_mode) != 0o700:
            raise ContainedExecutionError(
                "execution staging owner must be a real mode-0700 directory"
            )
        token = path / ".cellstate-execution-owner.json"
        try:
            token.lstat()
        except FileNotFoundError:
            entries = tuple(path.iterdir())
            unexpected = tuple(
                entry
                for entry in entries
                if entry != token and not entry.name.startswith(".cellstate-owner-token-")
            )
            if unexpected:
                try:
                    token.lstat()
                except FileNotFoundError:
                    raise ContainedExecutionError(
                        "unowned execution staging directory is not empty"
                    ) from None
            temporary_descriptor = -1
            temporary_name = ""
            try:
                temporary_descriptor, temporary_name = tempfile.mkstemp(
                    prefix=".cellstate-owner-token-",
                    dir=path,
                )
                try:
                    os.fchmod(temporary_descriptor, 0o400)
                    offset = 0
                    while offset < len(token_payload):
                        written = os.write(temporary_descriptor, token_payload[offset:])
                        if written <= 0:
                            raise ContainedExecutionError("short execution staging token write")
                        offset += written
                    os.fsync(temporary_descriptor)
                finally:
                    os.close(temporary_descriptor)
                    temporary_descriptor = -1
                os.link(temporary_name, token, follow_symlinks=False)
            except FileExistsError:
                pass
            except OSError as error:
                raise ContainedExecutionError(
                    "cannot create execution staging owner token"
                ) from error
            finally:
                if temporary_descriptor >= 0:
                    os.close(temporary_descriptor)
                if temporary_name:
                    with suppress(OSError):
                        Path(temporary_name).unlink()
        try:
            token_state = token.lstat()
            token_bytes = token.read_bytes()
        except OSError as error:
            raise ContainedExecutionError(
                "cannot reauthenticate execution staging owner"
            ) from error
        if not stat.S_ISREG(token_state.st_mode) or token_bytes != token_payload:
            raise ContainedExecutionError("execution staging owner token differs")

    def _prepare_output_stage(self, *, execution_id: str, output_path: Path) -> None:
        if self._claimed_output_stage is not None:
            raise ContainedExecutionError("executor instance already owns one output-stage claim")
        if self._staging_root is None or self._canonical_publication_root is None:
            raise ContainedExecutionError(
                "contained execution requires explicit staging and canonical-publication roots"
            )
        staging_root = self._staging_root
        canonical_root = self._canonical_publication_root
        if not staging_root.is_absolute() or not canonical_root.is_absolute():
            raise ContainedExecutionError("execution host roots must already be absolute")
        staging_root = self._no_symlink_realpath(staging_root, name="execution staging root")
        canonical_root = self._no_symlink_realpath(
            canonical_root, name="canonical-publication root"
        )
        if self._paths_overlap(staging_root, canonical_root):
            raise ContainedExecutionError(
                "execution staging must not overlap canonical publication"
            )
        expected = self.output_stage_path(execution_id)
        if output_path != expected:
            raise ContainedExecutionError("worker output path is not the executor-owned stage")
        root_token = canonical_json_bytes(
            {
                "artifact_schema": "cellstate-execution-staging-owner",
                "artifact_schema_version": "1.0.0",
                "owner_id": self.policy.owner_id,
            }
        )
        self._owned_directory(staging_root, token_payload=root_token)
        self._no_symlink_realpath(staging_root, name="execution staging root")
        owner_root = staging_root / self.policy.owner_id
        self._owned_directory(owner_root, token_payload=root_token)
        execution_root = owner_root / execution_id
        try:
            execution_root.mkdir(mode=0o700)
        except FileExistsError as error:
            raise ExecutionStageAlreadyClaimed(
                "execution ID has already been consumed; prior terminal evidence is preserved"
            ) from error
        except OSError as error:
            raise ContainedExecutionError("cannot claim the one-use execution stage") from error
        self._claimed_output_stage = (execution_id, output_path)
        self._owned_directory(execution_root, token_payload=root_token)
        output_path.mkdir(mode=0o700)
        observed = output_path.lstat()
        if (
            not stat.S_ISDIR(observed.st_mode)
            or stat.S_IMODE(observed.st_mode) != 0o700
            or tuple(output_path.iterdir())
        ):
            raise ContainedExecutionError("worker output stage is not empty, real, and restrictive")

    def _authenticate_prepared_output_stage(self, *, execution_id: str, output_path: Path) -> None:
        """Reauthenticate the one-use stage after lock acquisition without recreating it."""

        if self._staging_root is None or self._canonical_publication_root is None:
            raise ContainedExecutionError(
                "contained execution requires explicit staging and canonical-publication roots"
            )
        expected = self.output_stage_path(execution_id)
        if output_path != expected:
            raise ContainedExecutionError("worker output path is not the executor-owned stage")
        token_payload = canonical_json_bytes(
            {
                "artifact_schema": "cellstate-execution-staging-owner",
                "artifact_schema_version": "1.0.0",
                "owner_id": self.policy.owner_id,
            }
        )
        for directory in (
            self._staging_root,
            self._staging_root / self.policy.owner_id,
            output_path.parent,
        ):
            try:
                observed = directory.lstat()
                token = _read_stable_regular_file(
                    directory / ".cellstate-execution-owner.json",
                    name="execution staging owner token",
                )
            except OSError as error:
                raise ContainedExecutionError(
                    "prepared execution staging owner disappeared"
                ) from error
            if (
                not stat.S_ISDIR(observed.st_mode)
                or stat.S_IMODE(observed.st_mode) != 0o700
                or token != token_payload
            ):
                raise ContainedExecutionError("prepared execution staging ownership differs")
        try:
            output_state = output_path.lstat()
        except OSError as error:
            raise ContainedExecutionError("prepared worker output stage disappeared") from error
        if (
            not stat.S_ISDIR(output_state.st_mode)
            or stat.S_IMODE(output_state.st_mode) != 0o700
            or tuple(output_path.iterdir())
        ):
            raise ContainedExecutionError(
                "prepared worker output stage is not empty, real, and restrictive"
            )

    @staticmethod
    def _read_stable_closure_entry(path: Path) -> bytes:
        try:
            before = path.lstat()
        except OSError as error:
            raise ContainedExecutionError("cannot inspect execution-input closure entry") from error
        if not stat.S_ISREG(before.st_mode):
            raise ContainedExecutionError("execution-input closure entry is not a regular file")
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(path, flags)
            try:
                opened = os.fstat(descriptor)
                if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
                    raise ContainedExecutionError("execution-input entry changed while opening")
                chunks: list[bytes] = []
                while chunk := os.read(descriptor, 1024 * 1024):
                    chunks.append(chunk)
                after = os.fstat(descriptor)
            finally:
                os.close(descriptor)
        except OSError as error:
            raise ContainedExecutionError("cannot read execution-input closure entry") from error
        if (opened.st_dev, opened.st_ino, opened.st_size) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
        ):
            raise ContainedExecutionError("execution-input entry changed while copying")
        return b"".join(chunks)

    def _stage_code_closure(self, *, execution_id: str, source_root: Path) -> Path:
        """Copy, verify, and seal only the frozen closure before Docker can import it."""

        manifest = self._execution_input_closure
        if manifest is None:
            raise ContainedExecutionError("contained execution requires an exact input closure")
        if manifest.fingerprint != self.policy.execution_input_closure_sha256:
            raise ContainedExecutionError("execution policy input-closure identity drifted")
        if manifest.training_code_closure_sha256 != self.policy.training_code_closure_sha256:
            raise ContainedExecutionError("execution input carries another code closure")
        if not source_root.is_absolute():
            raise ContainedExecutionError("execution-input source root must already be absolute")
        source_root = self._no_symlink_realpath(source_root, name="execution-input source root")
        try:
            source_state = source_root.lstat()
        except OSError as error:
            raise ContainedExecutionError("cannot inspect execution-input source root") from error
        if not stat.S_ISDIR(source_state.st_mode):
            raise ContainedExecutionError("execution-input source root must be one real directory")
        if self._staging_root is None or self._paths_overlap(source_root, self._staging_root):
            raise ContainedExecutionError("execution-input source aliases writable staging")

        code_root = self.code_stage_path(execution_id)
        code_root.mkdir(mode=0o700)
        expected_files: set[str] = set()
        directories = {code_root}
        for entry in manifest.entries:
            payload = self._read_stable_closure_entry(source_root / entry.relative_path)
            if (
                len(payload) != entry.byte_count
                or hashlib.sha256(payload).hexdigest() != entry.sha256
            ):
                raise ContainedExecutionError("execution-input closure bytes differ from policy")
            target = code_root / Path(entry.relative_path)
            target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            current = target.parent
            while current != code_root:
                directories.add(current)
                current = current.parent
            flags = (
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0)
            )
            try:
                descriptor = os.open(target, flags, 0o400)
                try:
                    offset = 0
                    while offset < len(payload):
                        written = os.write(descriptor, payload[offset:])
                        if written <= 0:
                            raise ContainedExecutionError("short execution-input snapshot write")
                        offset += written
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
            except OSError as error:
                raise ContainedExecutionError("cannot stage execution-input closure") from error
            expected_files.add(entry.relative_path)
        actual_files = {
            path.relative_to(code_root).as_posix()
            for path in code_root.rglob("*")
            if path.is_file() and not path.is_symlink()
        }
        if actual_files != expected_files:
            raise ContainedExecutionError("staged execution-input closure contains extra paths")
        for directory in sorted(directories, key=lambda item: len(item.parts), reverse=True):
            directory.chmod(0o555)
            flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0)
            descriptor = os.open(directory, flags)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        for entry in manifest.entries:
            target = code_root / Path(entry.relative_path)
            payload = self._read_stable_closure_entry(target)
            if (
                stat.S_IMODE(target.lstat().st_mode) != 0o400
                or len(payload) != entry.byte_count
                or hashlib.sha256(payload).hexdigest() != entry.sha256
            ):
                raise ContainedExecutionError("sealed execution-input snapshot failed re-read")
        return code_root

    @contextmanager
    def _owner_lock(self) -> Iterator[None]:
        """Serialize one owner namespace; process death releases the stable inode lock."""

        try:
            import fcntl
        except ImportError as error:  # pragma: no cover - v5 execution is POSIX-only
            raise ContainedExecutionError("contained execution requires POSIX flock") from error
        try:
            self._lock_root.mkdir(parents=True, exist_ok=True)
            observed = self._lock_root.lstat()
            if not stat.S_ISDIR(observed.st_mode):
                raise ContainedExecutionError("execution lock root must be a real directory")
            path = self._lock_root / f"{self.policy.owner_id}.lock"
            flags = (
                os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
            )
            descriptor = os.open(path, flags, 0o600)
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
                os.close(descriptor)
        except OSError as error:
            raise ContainedExecutionError("cannot acquire stable execution-owner lock") from error

    def _run(
        self,
        *arguments: str,
        timeout_seconds: float | None = None,
    ) -> ContainerCommandResult:
        return self._cli.run(("docker", *arguments), timeout_seconds=timeout_seconds)

    def preflight_image(self) -> None:
        """Require the exact locally present digest before any worker or source operation."""

        result = self._run(
            "image",
            "inspect",
            "--format",
            "{{json .}}",
            self.policy.runtime_image.reference,
            timeout_seconds=float(self.policy.cleanup_timeout_seconds),
        )
        value = _docker_json(result, name="runtime image identity")
        if type(value) is not dict:
            raise ContainedExecutionError("Docker runtime image identity is not an object")
        image = value
        repo_digests = image.get("RepoDigests")
        if (
            type(repo_digests) is not list
            or self.policy.runtime_image.reference not in repo_digests
            or image.get("Os") != "linux"
            or image.get("Architecture") != "amd64"
        ):
            raise ContainedExecutionError("local runtime image differs from the frozen digest")

    def _labels(self, execution_id: str) -> tuple[str, ...]:
        return (
            f"{_LABEL_KEY}={self.policy.owner_id}",
            f"{_POLICY_LABEL_KEY}={self.policy.fingerprint}",
            f"{_EXECUTION_LABEL_KEY}={execution_id}",
        )

    def _container_name(self, execution_id: str) -> str:
        if _EXECUTION_ID.fullmatch(execution_id) is None:
            raise ContainedExecutionError("execution ID is not canonical")
        return f"cellstate-{self.policy.owner_id}-{execution_id}"

    def build_create_command(
        self,
        *,
        execution_id: str,
        source_path: Path,
        code_path: Path,
        output_path: Path,
    ) -> tuple[str, ...]:
        """Build the non-shell command without opening, resolving, or statting the source."""

        if _EXECUTION_ID.fullmatch(execution_id) is None:
            raise ContainedExecutionError("execution ID is not canonical")
        source = Path(source_path)
        code = Path(code_path)
        output = Path(output_path)
        if not source.is_absolute() or not code.is_absolute() or not output.is_absolute():
            raise ContainedExecutionError("container bind paths must already be absolute")
        for path in (source, code, output):
            rendered = os_fspath(path)
            if any(character in rendered for character in ("\x00", "\n", ",")):
                raise ContainedExecutionError("container bind path cannot be represented exactly")
        name = self._container_name(execution_id)
        command = [
            "docker",
            "create",
            "--name",
            name,
            "--platform",
            self.policy.runtime_image.platform,
            "--pull",
            "never",
            "--memory",
            str(self.policy.memory_max_bytes),
            "--memory-swap",
            str(self.policy.memory_swap_max_bytes),
            "--pids-limit",
            str(self.policy.pids_limit),
            "--user",
            f"{self._effective_uid}:{self._effective_gid}",
            "--tmpfs",
            f"{self.policy.temporary_container_path}:"
            "rw,noexec,nosuid,nodev,"
            f"size={self.policy.temporary_max_bytes},mode=0700,"
            f"uid={self._effective_uid},gid={self._effective_gid}",
            "--network",
            self.policy.network_mode,
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--init",
            "--entrypoint",
            self.policy.runtime_entrypoint,
        ]
        for label in self._labels(execution_id):
            command.extend(("--label", label))
        for key, value in self.policy.environment.items():
            command.extend(("--env", f"{key}={value}"))
        command.extend(
            (
                "--mount",
                f"type=bind,source={source},target={self.policy.source_container_path},readonly",
                "--mount",
                f"type=bind,source={code},target={self.policy.code_container_path},readonly",
                "--mount",
                f"type=bind,source={output},target={self.policy.output_container_path}",
                "--mount",
                f"type=volume,target={self.policy.snapshot_container_path}",
                "--workdir",
                self.policy.workdir,
                self.policy.runtime_image.reference,
                *self.policy.worker_command,
            )
        )
        return tuple(command)

    def _inspect_container_user(
        self, container_id: str, *, timeout_seconds: float
    ) -> tuple[int, int]:
        """Read back Docker's exact configured numeric user before worker start."""

        value = _docker_json(
            self._run(
                "inspect",
                "--format",
                "{{json .Config.User}}",
                container_id,
                timeout_seconds=timeout_seconds,
            ),
            name="container user identity",
        )
        expected = f"{self._effective_uid}:{self._effective_gid}"
        if type(value) is not str or value != expected:
            raise ContainedExecutionError(
                "Docker container user differs from the host-effective policy"
            )
        return self._effective_uid, self._effective_gid

    def _inspect_state(self, container_id: str) -> _ContainerState:
        value = _docker_json(
            self._run(
                "inspect",
                "--format",
                "{{json .State}}",
                container_id,
                timeout_seconds=float(self.policy.cleanup_timeout_seconds),
            ),
            name="container state",
        )
        if type(value) is not dict:
            raise ContainedExecutionError("Docker container state is not an object")
        running = value.get("Running")
        oom_killed = value.get("OOMKilled")
        exit_code = value.get("ExitCode")
        if type(running) is not bool or type(oom_killed) is not bool or type(exit_code) is not int:
            raise ContainedExecutionError("Docker container state has malformed primitive fields")
        return _ContainerState(running, oom_killed, exit_code)

    def _inspect_labels(
        self,
        container_id: str,
        *,
        expected_execution_id: str | None = None,
        allow_historical_policy: bool = False,
    ) -> Mapping[str, str]:
        value = _docker_json(
            self._run(
                "inspect",
                "--format",
                "{{json .Config.Labels}}",
                container_id,
                timeout_seconds=float(self.policy.cleanup_timeout_seconds),
            ),
            name="container ownership labels",
        )
        if type(value) is not dict or any(
            type(key) is not str or type(item) is not str for key, item in value.items()
        ):
            raise ContainedExecutionError("Docker container labels are malformed")
        labels = dict(value)
        policy_fingerprint = labels.get(_POLICY_LABEL_KEY)
        execution_id = labels.get(_EXECUTION_LABEL_KEY)
        if labels.get(_LABEL_KEY) != self.policy.owner_id:
            raise ContainedExecutionError("refusing to recover a container from another owner")
        if type(policy_fingerprint) is not str or _HEX_SHA256.fullmatch(policy_fingerprint) is None:
            raise ContainedExecutionError("container policy identity is malformed")
        if not allow_historical_policy and policy_fingerprint != self.policy.fingerprint:
            raise ContainedExecutionError("refusing to recover a container from another policy")
        if type(execution_id) is not str or _EXECUTION_ID.fullmatch(execution_id) is None:
            raise ContainedExecutionError("container execution identity is malformed")
        if expected_execution_id is not None and execution_id != expected_execution_id:
            raise ContainedExecutionError("refusing to recover a container from another execution")
        return labels

    def _inspect_container_name(self, container_id: str, *, execution_id: str) -> None:
        value = _docker_json(
            self._run(
                "inspect",
                "--format",
                "{{json .Name}}",
                container_id,
                timeout_seconds=float(self.policy.cleanup_timeout_seconds),
            ),
            name="container deterministic name",
        )
        if type(value) is not str or value != f"/{self._container_name(execution_id)}":
            raise ContainedExecutionError(
                "refusing to recover a container outside the deterministic owner namespace"
            )

    def _query_exact_container_name(self, container_name: str) -> tuple[str, ...]:
        result = self._run(
            "ps",
            "-aq",
            "--no-trunc",
            "--filter",
            f"name=^/{re.escape(container_name)}$",
            timeout_seconds=float(self.policy.cleanup_timeout_seconds),
        )
        if result.returncode != 0:
            raise ContainedExecutionError("Docker exact-name container query failed")
        container_ids = tuple(line.strip() for line in result.stdout.splitlines() if line.strip())
        if len(container_ids) > 1 or any(
            _CONTAINER_ID.fullmatch(item) is None for item in container_ids
        ):
            raise ContainedExecutionError("Docker exact-name query returned invalid IDs")
        return container_ids

    def _recover_ambiguous_create(self, execution_id: str) -> tuple[str, ...]:
        """Poll one deterministic name through the complete cleanup window after create doubt."""

        name = self._container_name(execution_id)
        deadline = float(self._monotonic()) + self.policy.cleanup_timeout_seconds
        recovered: list[str] = []
        while float(self._monotonic()) < deadline:
            container_ids = self._query_exact_container_name(name)
            for container_id in container_ids:
                self._inspect_labels(
                    container_id,
                    expected_execution_id=execution_id,
                )
                self._remove_and_verify(container_id, kill=True)
                if container_id not in recovered:
                    recovered.append(container_id)
            remaining = deadline - float(self._monotonic())
            if remaining > 0.0:
                self._sleep(min(0.1, remaining))

        final_ids = self._query_exact_container_name(name)
        for container_id in final_ids:
            self._inspect_labels(
                container_id,
                expected_execution_id=execution_id,
            )
            self._remove_and_verify(container_id, kill=True)
            if container_id not in recovered:
                recovered.append(container_id)
        if self._query_exact_container_name(name):
            raise ContainedExecutionError(
                "ambiguous Docker create remained after stable exact-name recovery"
            )
        return tuple(recovered)

    def _inspect_snapshot_volume(self, container_id: str) -> str:
        value = _docker_json(
            self._run(
                "inspect",
                "--format",
                "{{json .Mounts}}",
                container_id,
                timeout_seconds=float(self.policy.cleanup_timeout_seconds),
            ),
            name="container snapshot volume",
        )
        if type(value) is not list:
            raise ContainedExecutionError("Docker container mounts are malformed")
        if any(type(item) is not dict for item in value):
            raise ContainedExecutionError("Docker container mount entries are malformed")
        mounts = tuple(value)
        expected_mounts = {
            self.policy.source_container_path: ("bind", False),
            self.policy.code_container_path: ("bind", False),
            self.policy.output_container_path: ("bind", True),
            self.policy.snapshot_container_path: ("volume", True),
        }
        actual_mounts: dict[str, tuple[object, object]] = {}
        for mount in mounts:
            destination = mount.get("Destination")
            mount_type = mount.get("Type")
            writable = mount.get("RW")
            if mount_type == "tmpfs" and destination == self.policy.temporary_container_path:
                if writable is not True:
                    raise ContainedExecutionError("contained temporary mount is not writable")
                continue
            if type(destination) is not str or destination in actual_mounts:
                raise ContainedExecutionError("contained worker mount destinations are malformed")
            actual_mounts[destination] = (mount_type, writable)
        if actual_mounts != expected_mounts:
            raise ContainedExecutionError("contained worker mount shape differs from policy")
        volumes = tuple(mount for mount in mounts if mount.get("Type") == "volume")
        if len(volumes) != 1:
            raise ContainedExecutionError("contained worker must own one anonymous volume")
        volume = volumes[0]
        name = volume.get("Name")
        if (
            type(name) is not str
            or _ANONYMOUS_VOLUME_ID.fullmatch(name) is None
            or volume.get("Destination") != self.policy.snapshot_container_path
            or volume.get("RW") is not True
        ):
            raise ContainedExecutionError("contained snapshot volume identity is malformed")
        return name

    def _remove_and_verify(self, container_id: str, *, kill: bool) -> _ContainerState:
        snapshot_volume_id = self._inspect_snapshot_volume(container_id)
        state = self._inspect_state(container_id)
        # A create request can finish just before its CLI deadline while start never happens.  A
        # wait on that non-running ``created`` state can block indefinitely, so kill/wait only a
        # tree Docker still reports as running; an exited or never-started tree is removed directly.
        if kill and state.running:
            killed = self._run(
                "kill",
                "--signal",
                "KILL",
                container_id,
                timeout_seconds=float(self.policy.cleanup_timeout_seconds),
            )
            if killed.returncode != 0:
                raise ContainedExecutionError("Docker could not kill the contained process tree")
            try:
                waited = self._run(
                    "wait",
                    container_id,
                    timeout_seconds=float(self.policy.cleanup_timeout_seconds),
                )
            except ContainerCommandTimeout as error:
                raise ContainedExecutionError("killed container did not terminate") from error
            if waited.returncode != 0:
                raise ContainedExecutionError("Docker could not reap the contained process tree")
            state = self._inspect_state(container_id)
        if state.running:
            raise ContainedExecutionError("container process tree remained alive after execution")
        removed = self._run(
            "rm",
            "--force",
            "--volumes",
            container_id,
            timeout_seconds=float(self.policy.cleanup_timeout_seconds),
        )
        if removed.returncode != 0:
            raise ContainedExecutionError("Docker could not remove the contained process tree")
        absent = self._run(
            "ps",
            "-aq",
            "--no-trunc",
            "--filter",
            f"id={container_id}",
            timeout_seconds=float(self.policy.cleanup_timeout_seconds),
        )
        if absent.returncode != 0 or absent.stdout.strip():
            raise ContainedExecutionError("removed container absence could not be proved")
        volume_absent = self._run(
            "volume",
            "inspect",
            snapshot_volume_id,
            timeout_seconds=float(self.policy.cleanup_timeout_seconds),
        )
        if volume_absent.returncode == 0:
            raise ContainedExecutionError("anonymous source-snapshot volume remained after cleanup")
        volume_query = self._run(
            "volume",
            "ls",
            "--quiet",
            "--filter",
            f"name=^{snapshot_volume_id}$",
            timeout_seconds=float(self.policy.cleanup_timeout_seconds),
        )
        if volume_query.returncode != 0 or volume_query.stdout.strip():
            raise ContainedExecutionError(
                "anonymous source-snapshot volume absence could not be proved"
            )
        return state

    def _recover_owned_containers_locked(self) -> tuple[str, ...]:
        """Recover current or historical-policy trees inside one authenticated owner namespace."""

        result = self._run(
            "ps",
            "-aq",
            "--filter",
            f"label={_LABEL_KEY}={self.policy.owner_id}",
            timeout_seconds=float(self.policy.cleanup_timeout_seconds),
        )
        if result.returncode != 0:
            raise ContainedExecutionError("Docker orphan-container query failed")
        container_ids = tuple(line.strip() for line in result.stdout.splitlines() if line.strip())
        if len(container_ids) != len(set(container_ids)) or any(
            _CONTAINER_ID.fullmatch(item) is None for item in container_ids
        ):
            raise ContainedExecutionError("Docker orphan-container query returned invalid IDs")
        for container_id in container_ids:
            labels = self._inspect_labels(container_id, allow_historical_policy=True)
            execution_id = labels[_EXECUTION_LABEL_KEY]
            self._inspect_container_name(container_id, execution_id=execution_id)
            self._inspect_container_user(
                container_id,
                timeout_seconds=float(self.policy.cleanup_timeout_seconds),
            )
            self._inspect_snapshot_volume(container_id)
            self._remove_and_verify(container_id, kill=True)
        return container_ids

    def recover_owned_containers(self) -> tuple[str, ...]:
        """Recover authenticated owner trees while the stable namespace lock is held."""

        with self._owner_lock():
            return self._recover_owned_containers_locked()

    def run(
        self,
        *,
        execution_id: str,
        source_path: Path,
        code_path: Path,
        output_path: Path,
    ) -> ContainedExecutionObservation:
        """Run one worker and return only after the complete container tree is removed."""

        output = Path(output_path)
        if not output.is_absolute():
            raise ContainedExecutionError("contained output stage must already be absolute")
        # The one-use stage is claimed before source normalization and owner-lock acquisition so
        # every subsequent executor failure has a durable, quarantinable parent evidence root.
        self._prepare_output_stage(execution_id=execution_id, output_path=output)
        with self._owner_lock():
            return self._run_contained(
                execution_id=execution_id,
                source_path=source_path,
                code_path=code_path,
                output_path=output,
            )

    def _run_contained(
        self,
        *,
        execution_id: str,
        source_path: Path,
        code_path: Path,
        output_path: Path,
    ) -> ContainedExecutionObservation:
        """Execute while the stable owner lock excludes concurrent orphan recovery."""

        source = Path(source_path)
        code_source = Path(code_path)
        output = Path(output_path)
        if not source.is_absolute() or not code_source.is_absolute() or not output.is_absolute():
            raise ContainedExecutionError("contained host paths must already be absolute")
        source = self._no_symlink_realpath(source, name="contained source")
        code_source = self._no_symlink_realpath(code_source, name="execution-input source root")
        output = self._no_symlink_realpath(output, name="contained output stage")
        self._authenticate_prepared_output_stage(execution_id=execution_id, output_path=output)
        self._recover_owned_containers_locked()
        self.preflight_image()
        started_at = float(self._monotonic())
        if not math.isfinite(started_at):
            raise ContainedExecutionError("parent monotonic clock returned a non-finite value")
        deadline = started_at + self.policy.wall_clock_seconds

        def remaining() -> float:
            value = deadline - float(self._monotonic())
            if value <= 0.0:
                raise ContainerCommandTimeout("contained execution crossed its parent deadline")
            return value

        if self._staging_root is None:  # pragma: no cover - narrowed by stage preparation
            raise ContainedExecutionError("executor staging root disappeared")
        if self._paths_overlap(source, self._staging_root):
            raise ContainedExecutionError(
                "contained source aliases executor-owned writable staging"
            )
        try:
            source_state = source.lstat()
        except OSError as error:
            raise ContainedExecutionError("cannot inspect contained source bind") from error
        if not stat.S_ISREG(source_state.st_mode):
            raise ContainedExecutionError("contained source bind must be one real regular file")
        if self._canonical_publication_root is None:  # narrowed by stage preparation
            raise ContainedExecutionError("canonical-publication root disappeared")
        canonical_root = self._no_symlink_realpath(
            self._canonical_publication_root,
            name="canonical-publication root",
        )
        if self._paths_overlap(source, canonical_root) or self._tree_contains_inode(
            canonical_root,
            device=source_state.st_dev,
            inode=source_state.st_ino,
        ):
            raise ContainedExecutionError(
                "contained source aliases the canonical-publication inode closure"
            )
        if self._paths_overlap(code_source, canonical_root):
            raise ContainedExecutionError("execution-input source aliases canonical publication")
        remaining()
        code = self._stage_code_closure(
            execution_id=execution_id,
            source_root=code_source,
        )
        remaining()
        create_command = self.build_create_command(
            execution_id=execution_id,
            source_path=source,
            code_path=code,
            output_path=output,
        )

        try:
            created = self._cli.run(create_command, timeout_seconds=remaining())
        except ContainerCommandTimeout as error:
            try:
                self._recover_ambiguous_create(execution_id)
            except BaseException as cleanup_error:
                raise ContainedExecutionError(
                    "ambiguous Docker create cleanup could not be proved"
                ) from cleanup_error
            raise ContainedExecutionError(
                "Docker create crossed the wall deadline before worker start"
            ) from error
        container_id = created.stdout.strip()
        if created.returncode != 0 or _CONTAINER_ID.fullmatch(container_id) is None:
            try:
                self._recover_ambiguous_create(execution_id)
            except BaseException as cleanup_error:
                raise ContainedExecutionError(
                    "ambiguous Docker create cleanup could not be proved"
                ) from cleanup_error
            raise ContainedExecutionError("Docker failed to create the contained worker")

        timed_out = False
        state: _ContainerState | None = None
        container_may_be_running = False
        observed_uid: int | None = None
        observed_gid: int | None = None
        terminal_observed_at: float | None = None
        try:
            observed_uid, observed_gid = self._inspect_container_user(
                container_id, timeout_seconds=remaining()
            )
            container_may_be_running = True
            start = self._run("start", container_id, timeout_seconds=remaining())
            if start.returncode != 0:
                raise ContainedExecutionError("Docker failed to start the contained worker")
            try:
                waited = self._run("wait", container_id, timeout_seconds=remaining())
                if waited.returncode != 0:
                    raise ContainedExecutionError("Docker could not wait for the contained worker")
            except ContainerCommandTimeout:
                timed_out = True
            terminal_observed_at = float(self._monotonic())
            state = self._remove_and_verify(container_id, kill=timed_out)
        except BaseException:
            if state is None:
                try:
                    state = self._remove_and_verify(container_id, kill=container_may_be_running)
                except BaseException as cleanup_error:
                    raise ContainedExecutionError(
                        "contained worker failed and process-tree cleanup could not be proved"
                    ) from cleanup_error
            raise

        if state is None:  # pragma: no cover - defensive narrowing
            raise ContainedExecutionError("contained worker produced no final state")
        if observed_uid is None or observed_gid is None:  # pragma: no cover - defensive narrowing
            raise ContainedExecutionError("contained worker user identity was not observed")
        if terminal_observed_at is None:  # pragma: no cover - defensive narrowing
            raise ContainedExecutionError("contained worker terminal time was not observed")
        parent_elapsed = terminal_observed_at - started_at
        if not math.isfinite(parent_elapsed) or parent_elapsed < 0.0:
            raise ContainedExecutionError(
                "parent monotonic clock moved backwards or became invalid"
            )
        oom_killed = state.oom_killed and not timed_out
        worker_watchdog_timed_out = (
            not timed_out and not oom_killed and state.exit_code in {124, 137}
        )
        outcome: Literal["success", "timeout", "oom_killed", "worker_failure"] = (
            "timeout"
            if timed_out or worker_watchdog_timed_out
            else "oom_killed"
            if oom_killed
            else "success"
            if state.exit_code == 0
            else "worker_failure"
        )
        return ContainedExecutionObservation(
            execution_id=execution_id,
            policy_fingerprint=self.policy.fingerprint,
            runtime_image_digest=self.policy.runtime_image.digest,
            container_user_mode=self.policy.container_user_mode,
            observed_container_uid=observed_uid,
            observed_container_gid=observed_gid,
            outcome=outcome,
            exit_code=state.exit_code,
            timed_out=timed_out,
            worker_watchdog_timed_out=worker_watchdog_timed_out,
            oom_killed=oom_killed,
            parent_wall_clock_elapsed_seconds=parent_elapsed,
        )


def os_fspath(path: Path) -> str:
    """Small typed boundary kept separate for command-construction tests."""

    return str(path)


__all__ = [
    "PARENT_TERMINAL_REPORT_MAX_BYTES",
    "WORKER_TERMINAL_REPORT_MAX_BYTES",
    "CanonicalPublicationTreeIdentity",
    "ContainedExecutionError",
    "ContainedExecutionObservation",
    "ContainedExecutionPolicy",
    "ContainedTrainingObservation",
    "ContainedTrainingTerminalObservation",
    "ContainedTrainingWorkerObservation",
    "ContainedTrainingWorkerTerminalReport",
    "ContainerCLI",
    "ContainerCommandResult",
    "ContainerCommandTimeout",
    "DockerExecutor",
    "ExecutionInputClosureManifest",
    "ExecutionStageAlreadyClaimed",
    "RuntimeBuilderIdentity",
    "RuntimeImageIdentity",
    "RuntimeImageLayerIdentity",
    "RuntimeImageLock",
    "StagedTrainingEntry",
    "StagedTrainingInventory",
    "SubprocessDockerCLI",
    "TrainingCodeClosureEntry",
    "TrainingCodeClosureManifest",
    "canonical_publication_tree_identity",
    "inventory_staged_training_tree",
    "seal_staged_training_tree",
]
