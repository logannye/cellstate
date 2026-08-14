#!/usr/bin/env python3
"""Verify and prepare the immutable Item 12.3 runtime distribution before source access."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import stat
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

if __package__:
    from scripts.verify_sciplex3_v5_runtime_oci import (
        ArchiveIdentity,
        VerificationError,
        _load_lock,
        _verify_loaded_image_payload,
        verify_archive,
    )
else:  # pragma: no cover - exercised by direct CLI use rather than import-based tests.
    from verify_sciplex3_v5_runtime_oci import (  # type: ignore[import-not-found,no-redef]
        ArchiveIdentity,
        VerificationError,
        _load_lock,
        _verify_loaded_image_payload,
        verify_archive,
    )

EXPECTED_REPOSITORY = "logannye/cellstate"
EXPECTED_RELEASE_TAG = "sciplex3-v5-runtime-20260811-locked"
EXPECTED_RELEASE_TARGET_COMMIT = "d254d7c128e7025f19d2c84cdb3eb9901c5e69cb"
EXPECTED_ARCHIVE_SHA256 = "37c2fa5846acfbd8357476859bd7f8f0ac6591261d79c2f6f46f0aa22fb76454"
EXPECTED_ARCHIVE_BYTE_COUNT = 115_639_808
EXPECTED_ASSET_NAME = f"sciplex3-v5-runtime-linux-amd64-{EXPECTED_ARCHIVE_SHA256}.oci.tar"
EXPECTED_OCI_INDEX_DIGEST = (
    "sha256:e0f0afd6c66197a37d0ab7a05e7cccfe5990da1fd8497e175fdf3ab909a67812"
)
EXPECTED_IMAGE_DIGEST = "sha256:12c2faa6019fb60cdcabaa8f38f70e99be7998997b97ddb0ca59fbe2e82f1e25"
EXPECTED_CONFIG_DIGEST = "sha256:80ed48f278d7a46c0ae7811285efc69181ae59872a358cc9b176079aa09f3cc8"
EXPECTED_RUNTIME_IMAGE_LOCK_SHA256 = (
    "ea7ecdb3b7f0bef452da562b8d1267e9fc8a3e218f897555831718a9afca3f7b"
)
EXPECTED_RUNTIME_DISTRIBUTION_LOCK_SHA256 = (
    "9b3e14ced6b223666318d6495f5762b414321186aa5e0606d093566c7dcb1259"
)
EXPECTED_HOST_OPERATING_SYSTEM = "linux"
EXPECTED_HOST_ARCHITECTURE = "x86_64"
EXPECTED_DOCKER_SERVER_VERSION = "29.7.2"
EXPECTED_DOCKER_OPERATING_SYSTEM = "linux"
EXPECTED_DOCKER_ARCHITECTURE = "x86_64"
EXPECTED_IMAGE_STORE_STATUS = ("driver-type", "io.containerd.snapshotter.v1")
EXPECTED_CGROUP_VERSION = "2"
EXPECTED_ATTESTATION_STATEMENT_TYPE = "https://in-toto.io/Statement/v1"
EXPECTED_ATTESTATION_PREDICATE_TYPE = "https://in-toto.io/attestation/release/v0.2"
EXPECTED_ATTESTATION_CERTIFICATE_IDENTITY = "https://dotcom.releases.github.com"
_COMMAND_TIMEOUT_SECONDS = 30.0
_LOAD_TIMEOUT_SECONDS = 300.0
_PROHIBITED_DOCKER_ENVIRONMENT = (
    "DOCKER_API_VERSION",
    "DOCKER_CERT_PATH",
    "DOCKER_CONFIG",
    "DOCKER_CONTEXT",
    "DOCKER_DEFAULT_PLATFORM",
    "DOCKER_HOST",
    "DOCKER_TLS_VERIFY",
)


@dataclass(frozen=True, slots=True)
class RuntimeDistributionLock:
    """Exact immutable locator joined to one runtime-image lock and daemon contract."""

    repository: str
    release_tag: str
    release_target_commit: str
    asset_name: str
    archive_sha256: str
    archive_byte_count: int
    oci_index_digest: str
    image_digest: str
    config_digest: str
    distribution_lock_sha256: str
    runtime_image_lock_sha256: str
    host_operating_system: str
    host_architecture: str
    docker_server_version: str
    docker_operating_system: str
    docker_architecture: str
    image_store_status: tuple[str, str]
    cgroup_version: str
    memory_limit_supported: bool
    memory_swap_limit_supported: bool
    pids_limit_supported: bool


@dataclass(frozen=True, slots=True)
class ImmutableReleaseIdentity:
    """Observed immutable GitHub release and exact release-asset identity."""

    repository: str
    release_tag: str
    release_target_commit: str
    asset_name: str
    asset_sha256: str
    asset_byte_count: int
    asset_url: str
    attestation_predicate_type: str
    release_attestation_verified: bool
    asset_attestation_verified: bool


@dataclass(frozen=True, slots=True)
class RuntimeDaemonIdentity:
    """Observed local Docker daemon identity required for protected execution."""

    context_name: str
    endpoint: str
    host_operating_system: str
    host_architecture: str
    server_version: str
    operating_system: str
    architecture: str
    image_store_status: tuple[str, str]
    cgroup_version: str
    memory_limit_supported: bool
    memory_swap_limit_supported: bool
    pids_limit_supported: bool


@dataclass(frozen=True, slots=True)
class RuntimeArchiveFileIdentity:
    """Host file identity held continuously while Docker consumes the verified archive."""

    device: int
    inode: int
    byte_count: int
    mode: int
    modification_time_ns: int
    change_time_ns: int


@dataclass(frozen=True, slots=True)
class RuntimeDistributionObservation:
    """Source-free proof that the distributed archive is loaded under the exact daemon."""

    distribution_lock_sha256: str
    runtime_image_lock_sha256: str
    archive: ArchiveIdentity
    archive_file: RuntimeArchiveFileIdentity
    release: ImmutableReleaseIdentity
    daemon: RuntimeDaemonIdentity
    loaded_image_verified: bool
    load_performed_from_verified_descriptor: bool

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class CommandResult:
    """Minimal injectable external-command result for source-free verification."""

    returncode: int
    stdout: str
    stderr: str


CommandRunner = Callable[[Sequence[str], float, tuple[int, ...]], CommandResult]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_object(path: Path, *, label: str) -> tuple[dict[str, Any], bytes]:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise VerificationError(f"cannot read {label} {path}: {exc}") from exc

    def pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise VerificationError(f"{label} contains duplicate JSON key {key!r}")
            value[key] = item
        return value

    try:
        parsed = json.loads(payload, object_pairs_hook=pairs_hook)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise VerificationError(f"{label} is not valid UTF-8 JSON") from exc
    if not isinstance(parsed, dict):
        raise VerificationError(f"{label} must be a JSON object")
    canonical = json.dumps(parsed, sort_keys=True, separators=(",", ":")).encode()
    if payload not in {canonical, canonical + b"\n"}:
        raise VerificationError(f"{label} is not canonical JSON")
    return parsed, payload


def _expected_distribution_payload() -> dict[str, Any]:
    return {
        "archive": {
            "byte_count": EXPECTED_ARCHIVE_BYTE_COUNT,
            "config_digest": EXPECTED_CONFIG_DIGEST,
            "image_digest": EXPECTED_IMAGE_DIGEST,
            "oci_index_digest": EXPECTED_OCI_INDEX_DIGEST,
            "sha256": EXPECTED_ARCHIVE_SHA256,
        },
        "distribution": {
            "asset_attestation_required": True,
            "asset_name": EXPECTED_ASSET_NAME,
            "immutable_release_required": True,
            "provider": "github-release-asset",
            "release_attestation_required": True,
            "release_tag": EXPECTED_RELEASE_TAG,
            "release_target_commit": EXPECTED_RELEASE_TARGET_COMMIT,
            "repository": EXPECTED_REPOSITORY,
        },
        "runtime_daemon": {
            "architecture": EXPECTED_DOCKER_ARCHITECTURE,
            "cgroup_version": EXPECTED_CGROUP_VERSION,
            "executor": "docker",
            "host_architecture": EXPECTED_HOST_ARCHITECTURE,
            "host_operating_system": EXPECTED_HOST_OPERATING_SYSTEM,
            "image_store_driver_status": list(EXPECTED_IMAGE_STORE_STATUS),
            "local_unix_socket_required": True,
            "memory_limit_supported": True,
            "memory_swap_limit_supported": True,
            "operating_system": EXPECTED_DOCKER_OPERATING_SYSTEM,
            "pids_limit_supported": True,
            "server_version": EXPECTED_DOCKER_SERVER_VERSION,
        },
        "runtime_distribution_lock_schema": ("cellstate-sciplex3-v5-runtime-distribution-lock"),
        "runtime_distribution_lock_version": "1.0.0",
        "runtime_image_lock_sha256": EXPECTED_RUNTIME_IMAGE_LOCK_SHA256,
    }


def verify_distribution_lock(
    distribution_lock_path: Path,
    runtime_image_lock_path: Path,
) -> RuntimeDistributionLock:
    """Verify one predeclared release locator without contacting GitHub or Docker."""

    distribution, distribution_bytes = _canonical_json_object(
        distribution_lock_path, label="runtime distribution lock"
    )
    if hashlib.sha256(distribution_bytes).hexdigest() != (
        EXPECTED_RUNTIME_DISTRIBUTION_LOCK_SHA256
    ):
        raise VerificationError("runtime distribution lock raw SHA-256 differs")
    if distribution != _expected_distribution_payload():
        raise VerificationError("runtime distribution lock differs from the reviewed locator")
    try:
        image_lock_sha256 = _sha256_file(runtime_image_lock_path)
    except OSError as exc:
        raise VerificationError(
            f"cannot hash runtime image lock {runtime_image_lock_path}: {exc}"
        ) from exc
    if image_lock_sha256 != EXPECTED_RUNTIME_IMAGE_LOCK_SHA256:
        raise VerificationError("runtime distribution points to another runtime image lock")
    image_lock = _load_lock(runtime_image_lock_path)
    archive = distribution["archive"]
    if (
        image_lock.get("archive_sha256") != archive["sha256"]
        or image_lock.get("oci_index_digest") != archive["oci_index_digest"]
        or image_lock.get("image_digest") != archive["image_digest"]
        or image_lock.get("config_digest") != archive["config_digest"]
    ):
        raise VerificationError("runtime distribution and runtime image lock identities differ")
    return RuntimeDistributionLock(
        repository=EXPECTED_REPOSITORY,
        release_tag=EXPECTED_RELEASE_TAG,
        release_target_commit=EXPECTED_RELEASE_TARGET_COMMIT,
        asset_name=EXPECTED_ASSET_NAME,
        archive_sha256=EXPECTED_ARCHIVE_SHA256,
        archive_byte_count=EXPECTED_ARCHIVE_BYTE_COUNT,
        oci_index_digest=EXPECTED_OCI_INDEX_DIGEST,
        image_digest=EXPECTED_IMAGE_DIGEST,
        config_digest=EXPECTED_CONFIG_DIGEST,
        distribution_lock_sha256=EXPECTED_RUNTIME_DISTRIBUTION_LOCK_SHA256,
        runtime_image_lock_sha256=EXPECTED_RUNTIME_IMAGE_LOCK_SHA256,
        host_operating_system=EXPECTED_HOST_OPERATING_SYSTEM,
        host_architecture=EXPECTED_HOST_ARCHITECTURE,
        docker_server_version=EXPECTED_DOCKER_SERVER_VERSION,
        docker_operating_system=EXPECTED_DOCKER_OPERATING_SYSTEM,
        docker_architecture=EXPECTED_DOCKER_ARCHITECTURE,
        image_store_status=EXPECTED_IMAGE_STORE_STATUS,
        cgroup_version=EXPECTED_CGROUP_VERSION,
        memory_limit_supported=True,
        memory_swap_limit_supported=True,
        pids_limit_supported=True,
    )


def _default_command_runner(
    command: Sequence[str], timeout_seconds: float, pass_fds: tuple[int, ...]
) -> CommandResult:
    try:
        completed = subprocess.run(
            tuple(command),
            check=False,
            capture_output=True,
            pass_fds=pass_fds,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        raise VerificationError(f"external command timed out: {command[0]}") from exc
    except OSError as exc:
        raise VerificationError(f"cannot invoke external command {command[0]}") from exc
    return CommandResult(completed.returncode, completed.stdout, completed.stderr)


def _checked_command(
    command: Sequence[str],
    *,
    label: str,
    runner: CommandRunner,
    timeout_seconds: float = _COMMAND_TIMEOUT_SECONDS,
    pass_fds: tuple[int, ...] = (),
) -> str:
    result = runner(tuple(command), timeout_seconds, pass_fds)
    if result.returncode != 0:
        detail = result.stderr.strip() or "no diagnostic"
        raise VerificationError(f"{label} failed: {detail}")
    return result.stdout


def _json_output(payload: str, *, label: str) -> object:
    try:
        return json.loads(payload)
    except json.JSONDecodeError as exc:
        raise VerificationError(f"{label} returned malformed JSON") from exc


def _verify_release_attestation_statement(
    payload: str,
    *,
    label: str,
    lock: RuntimeDistributionLock,
) -> None:
    evidence = _json_output(payload, label=label)
    if not isinstance(evidence, dict):
        raise VerificationError(f"{label} evidence must be an object")
    verification = evidence.get("verificationResult")
    if not isinstance(verification, dict):
        raise VerificationError(f"{label} has no verified result")
    signature = verification.get("signature")
    certificate = signature.get("certificate") if isinstance(signature, dict) else None
    if (
        not isinstance(certificate, dict)
        or certificate.get("subjectAlternativeName") != EXPECTED_ATTESTATION_CERTIFICATE_IDENTITY
    ):
        raise VerificationError(f"{label} has another certificate identity")
    statement = verification.get("statement")
    if not isinstance(statement, dict):
        raise VerificationError(f"{label} has no verified statement")
    package_url = f"pkg:github/{lock.repository}@{lock.release_tag}"
    expected_subjects = [
        {"digest": {"sha1": lock.release_target_commit}, "uri": package_url},
        {"digest": {"sha256": lock.archive_sha256}, "name": lock.asset_name},
    ]
    predicate = statement.get("predicate")
    if (
        statement.get("_type") != EXPECTED_ATTESTATION_STATEMENT_TYPE
        or statement.get("predicateType") != EXPECTED_ATTESTATION_PREDICATE_TYPE
        or statement.get("subject") != expected_subjects
        or not isinstance(predicate, dict)
        or predicate.get("purl") != package_url
        or predicate.get("repository") != lock.repository
        or predicate.get("tag") != lock.release_tag
    ):
        raise VerificationError(f"{label} statement differs from the immutable locator")


def _verified_archive_path(path: Path, lock: RuntimeDistributionLock) -> tuple[int, int]:
    try:
        observed = path.lstat()
    except OSError as exc:
        raise VerificationError(
            f"cannot inspect distributed runtime archive {path}: {exc}"
        ) from exc
    if not stat.S_ISREG(observed.st_mode):
        raise VerificationError("distributed runtime archive must be one regular non-symlink file")
    if path.name != lock.asset_name:
        raise VerificationError("distributed runtime archive filename differs from its locator")
    if observed.st_size != lock.archive_byte_count:
        raise VerificationError("distributed runtime archive byte count differs from its locator")
    return observed.st_dev, observed.st_ino


class _VerifiedArchiveDescriptor:
    """One no-follow archive descriptor held from verification through Docker load."""

    def __init__(self, path: Path, lock: RuntimeDistributionLock) -> None:
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        try:
            self.descriptor = os.open(path, flags)
        except OSError as exc:
            raise VerificationError(
                f"cannot securely open distributed runtime archive: {exc}"
            ) from exc
        try:
            observed = os.fstat(self.descriptor)
            if not stat.S_ISREG(observed.st_mode):
                raise VerificationError("distributed runtime descriptor is not a regular file")
            self.identity = RuntimeArchiveFileIdentity(
                device=observed.st_dev,
                inode=observed.st_ino,
                byte_count=observed.st_size,
                mode=stat.S_IMODE(observed.st_mode),
                modification_time_ns=observed.st_mtime_ns,
                change_time_ns=observed.st_ctime_ns,
            )
            self._lock = lock
            self.verify()
        except BaseException:
            os.close(self.descriptor)
            raise

    def __enter__(self) -> _VerifiedArchiveDescriptor:
        return self

    def __exit__(self, *_: object) -> None:
        os.close(self.descriptor)

    def _sha256(self) -> str:
        digest = hashlib.sha256()
        offset = 0
        while chunk := os.pread(self.descriptor, 1024 * 1024, offset):
            digest.update(chunk)
            offset += len(chunk)
        if offset != self.identity.byte_count:
            raise VerificationError("distributed runtime descriptor changed while being hashed")
        return digest.hexdigest()

    def verify(self) -> None:
        def identity() -> RuntimeArchiveFileIdentity:
            observed = os.fstat(self.descriptor)
            return RuntimeArchiveFileIdentity(
                device=observed.st_dev,
                inode=observed.st_ino,
                byte_count=observed.st_size,
                mode=stat.S_IMODE(observed.st_mode),
                modification_time_ns=observed.st_mtime_ns,
                change_time_ns=observed.st_ctime_ns,
            )

        if identity() != self.identity:
            raise VerificationError("distributed runtime descriptor identity changed")
        if self.identity.byte_count != self._lock.archive_byte_count:
            raise VerificationError("distributed runtime descriptor byte count differs")
        if self._sha256() != self._lock.archive_sha256:
            raise VerificationError("distributed runtime descriptor SHA-256 differs")
        if identity() != self.identity:
            raise VerificationError("distributed runtime descriptor changed while being hashed")


def _load_runtime_archive_from_descriptor(
    archive: _VerifiedArchiveDescriptor,
    *,
    runner: CommandRunner,
) -> None:
    """Make Docker consume the already verified descriptor, never its mutable pathname."""

    descriptor_path = f"/proc/self/fd/{archive.descriptor}"
    _checked_command(
        ("docker", "load", "--input", descriptor_path),
        label="exact runtime archive descriptor load",
        runner=runner,
        timeout_seconds=_LOAD_TIMEOUT_SECONDS,
        pass_fds=(archive.descriptor,),
    )
    archive.verify()


def verify_immutable_release_asset(
    lock: RuntimeDistributionLock,
    archive_path: Path,
    *,
    runner: CommandRunner = _default_command_runner,
) -> ImmutableReleaseIdentity:
    """Verify GitHub's immutable release metadata and attestations for one local archive."""

    before_identity = _verified_archive_path(archive_path, lock)
    release_payload = _checked_command(
        (
            "gh",
            "release",
            "view",
            lock.release_tag,
            "--repo",
            lock.repository,
            "--json",
            "tagName,targetCommitish,isDraft,isImmutable,assets",
        ),
        label="immutable runtime release lookup",
        runner=runner,
    )
    release = _json_output(release_payload, label="immutable runtime release lookup")
    if not isinstance(release, dict):
        raise VerificationError("immutable runtime release metadata must be an object")
    assets = release.get("assets")
    if (
        release.get("tagName") != lock.release_tag
        or release.get("targetCommitish") != lock.release_target_commit
        or release.get("isDraft") is not False
        or release.get("isImmutable") is not True
        or not isinstance(assets, list)
        or len(assets) != 1
        or not isinstance(assets[0], dict)
    ):
        raise VerificationError("runtime release is not the exact published immutable release")
    asset = assets[0]
    expected_asset_digest = f"sha256:{lock.archive_sha256}"
    expected_url = (
        f"https://github.com/{lock.repository}/releases/download/"
        f"{lock.release_tag}/{lock.asset_name}"
    )
    if (
        asset.get("name") != lock.asset_name
        or asset.get("size") != lock.archive_byte_count
        or asset.get("digest") != expected_asset_digest
        or asset.get("state") != "uploaded"
        or asset.get("url") != expected_url
    ):
        raise VerificationError("immutable runtime release asset differs from its locator")

    release_attestation = _checked_command(
        (
            "gh",
            "release",
            "verify",
            lock.release_tag,
            "--repo",
            lock.repository,
            "--format",
            "json",
        ),
        label="runtime release attestation verification",
        runner=runner,
    )
    if not release_attestation.strip():
        raise VerificationError("runtime release attestation verification returned no evidence")
    _verify_release_attestation_statement(
        release_attestation,
        label="runtime release attestation verification",
        lock=lock,
    )
    asset_attestation = _checked_command(
        (
            "gh",
            "release",
            "verify-asset",
            lock.release_tag,
            str(archive_path),
            "--repo",
            lock.repository,
            "--format",
            "json",
        ),
        label="runtime release asset attestation verification",
        runner=runner,
    )
    if not asset_attestation.strip():
        raise VerificationError("runtime release asset attestation returned no evidence")
    _verify_release_attestation_statement(
        asset_attestation,
        label="runtime release asset attestation verification",
        lock=lock,
    )

    after_identity = _verified_archive_path(archive_path, lock)
    if after_identity != before_identity or _sha256_file(archive_path) != lock.archive_sha256:
        raise VerificationError("distributed runtime archive changed during release verification")
    return ImmutableReleaseIdentity(
        repository=lock.repository,
        release_tag=lock.release_tag,
        release_target_commit=lock.release_target_commit,
        asset_name=lock.asset_name,
        asset_sha256=lock.archive_sha256,
        asset_byte_count=lock.archive_byte_count,
        asset_url=expected_url,
        attestation_predicate_type=EXPECTED_ATTESTATION_PREDICATE_TYPE,
        release_attestation_verified=True,
        asset_attestation_verified=True,
    )


def _docker_json_value(
    command: Sequence[str],
    *,
    label: str,
    runner: CommandRunner,
) -> object:
    return _json_output(
        _checked_command(command, label=label, runner=runner),
        label=label,
    )


def _host_identity() -> tuple[str, str]:
    return sys.platform, platform.machine()


def verify_runtime_daemon(
    lock: RuntimeDistributionLock,
    *,
    runner: CommandRunner = _default_command_runner,
) -> RuntimeDaemonIdentity:
    """Require the exact local native-linux Docker daemon and containerd image store."""

    host_operating_system, host_architecture = _host_identity()
    if host_operating_system != lock.host_operating_system:
        raise VerificationError("runtime host operating system is not native Linux")
    if host_architecture != lock.host_architecture:
        raise VerificationError("runtime host architecture is not native x86_64")
    overrides = tuple(key for key in _PROHIBITED_DOCKER_ENVIRONMENT if key in os.environ)
    if overrides:
        raise VerificationError("Docker environment overrides are prohibited")
    context_name = _checked_command(
        ("docker", "context", "show"), label="Docker context lookup", runner=runner
    ).strip()
    if not context_name or any(character.isspace() for character in context_name):
        raise VerificationError("active Docker context name is not canonical")
    endpoint = _docker_json_value(
        (
            "docker",
            "context",
            "inspect",
            context_name,
            "--format",
            "{{json .Endpoints.docker.Host}}",
        ),
        label="Docker endpoint lookup",
        runner=runner,
    )
    server_version = _docker_json_value(
        ("docker", "version", "--format", "{{json .Server.Version}}"),
        label="Docker server-version lookup",
        runner=runner,
    )
    operating_system = _docker_json_value(
        ("docker", "info", "--format", "{{json .OSType}}"),
        label="Docker operating-system lookup",
        runner=runner,
    )
    architecture = _docker_json_value(
        ("docker", "info", "--format", "{{json .Architecture}}"),
        label="Docker architecture lookup",
        runner=runner,
    )
    driver_status = _docker_json_value(
        ("docker", "info", "--format", "{{json .DriverStatus}}"),
        label="Docker image-store lookup",
        runner=runner,
    )
    cgroup_version = _docker_json_value(
        ("docker", "info", "--format", "{{json .CgroupVersion}}"),
        label="Docker cgroup-version lookup",
        runner=runner,
    )
    memory_limit = _docker_json_value(
        ("docker", "info", "--format", "{{json .MemoryLimit}}"),
        label="Docker memory-limit support lookup",
        runner=runner,
    )
    swap_limit = _docker_json_value(
        ("docker", "info", "--format", "{{json .SwapLimit}}"),
        label="Docker memory-swap-limit support lookup",
        runner=runner,
    )
    pids_limit = _docker_json_value(
        ("docker", "info", "--format", "{{json .PidsLimit}}"),
        label="Docker PID-limit support lookup",
        runner=runner,
    )
    if not isinstance(endpoint, str) or not endpoint.startswith("unix:///"):
        raise VerificationError("Docker daemon must use one local Unix-socket endpoint")
    if server_version != lock.docker_server_version:
        raise VerificationError("Docker server version differs from the runtime distribution lock")
    if operating_system != lock.docker_operating_system:
        raise VerificationError(
            "Docker operating system differs from the runtime distribution lock"
        )
    if architecture != lock.docker_architecture:
        raise VerificationError("Docker architecture differs from the runtime distribution lock")
    if not isinstance(driver_status, list) or list(lock.image_store_status) not in driver_status:
        raise VerificationError("Docker daemon does not use the required containerd image store")
    if cgroup_version != lock.cgroup_version:
        raise VerificationError("Docker daemon does not use the required cgroup version")
    if memory_limit is not lock.memory_limit_supported or memory_limit is not True:
        raise VerificationError("Docker daemon does not enforce memory limits")
    if swap_limit is not lock.memory_swap_limit_supported or swap_limit is not True:
        raise VerificationError("Docker daemon does not enforce memory-plus-swap limits")
    if pids_limit is not lock.pids_limit_supported or pids_limit is not True:
        raise VerificationError("Docker daemon does not enforce PID limits")
    return RuntimeDaemonIdentity(
        context_name=context_name,
        endpoint=endpoint,
        host_operating_system=host_operating_system,
        host_architecture=host_architecture,
        server_version=lock.docker_server_version,
        operating_system=lock.docker_operating_system,
        architecture=lock.docker_architecture,
        image_store_status=lock.image_store_status,
        cgroup_version=lock.cgroup_version,
        memory_limit_supported=lock.memory_limit_supported,
        memory_swap_limit_supported=lock.memory_swap_limit_supported,
        pids_limit_supported=lock.pids_limit_supported,
    )


def _verify_loaded_runtime(
    runtime_image_lock_path: Path,
    *,
    runner: CommandRunner,
) -> None:
    image_lock = _load_lock(runtime_image_lock_path)
    reference = image_lock.get("image_reference")
    if not isinstance(reference, str) or not reference:
        raise VerificationError("runtime image lock has no exact image reference")
    payload = _docker_json_value(
        ("docker", "image", "inspect", reference),
        label="loaded runtime image lookup",
        runner=runner,
    )
    _verify_loaded_image_payload(image_lock, payload)


def verify_runtime_distribution(
    distribution_lock_path: Path,
    runtime_image_lock_path: Path,
    archive_path: Path,
    *,
    runner: CommandRunner = _default_command_runner,
) -> RuntimeDistributionObservation:
    """Verify an immutable archive, exact daemon, and already-loaded image before source access."""

    lock = verify_distribution_lock(distribution_lock_path, runtime_image_lock_path)
    _verified_archive_path(archive_path, lock)
    with _VerifiedArchiveDescriptor(archive_path, lock) as stable_archive:
        archive = verify_archive(runtime_image_lock_path, archive_path)
        if archive.archive_sha256 != lock.archive_sha256:
            raise VerificationError("distributed archive differs from its distribution lock")
        release = verify_immutable_release_asset(lock, archive_path, runner=runner)
        daemon = verify_runtime_daemon(lock, runner=runner)
        _verify_loaded_runtime(runtime_image_lock_path, runner=runner)
        stable_archive.verify()
        return RuntimeDistributionObservation(
            distribution_lock_sha256=lock.distribution_lock_sha256,
            runtime_image_lock_sha256=lock.runtime_image_lock_sha256,
            archive=archive,
            archive_file=stable_archive.identity,
            release=release,
            daemon=daemon,
            loaded_image_verified=True,
            load_performed_from_verified_descriptor=False,
        )


def prepare_runtime_distribution(
    distribution_lock_path: Path,
    runtime_image_lock_path: Path,
    archive_path: Path,
    *,
    runner: CommandRunner = _default_command_runner,
) -> RuntimeDistributionObservation:
    """Verify, load, reverify, and observe the exact runtime before any protected-source work."""

    lock = verify_distribution_lock(distribution_lock_path, runtime_image_lock_path)
    _verified_archive_path(archive_path, lock)
    with _VerifiedArchiveDescriptor(archive_path, lock) as stable_archive:
        archive = verify_archive(runtime_image_lock_path, archive_path)
        release = verify_immutable_release_asset(lock, archive_path, runner=runner)
        stable_archive.verify()
        daemon = verify_runtime_daemon(lock, runner=runner)
        _load_runtime_archive_from_descriptor(stable_archive, runner=runner)
        _verify_loaded_runtime(runtime_image_lock_path, runner=runner)
        stable_archive.verify()
        return RuntimeDistributionObservation(
            distribution_lock_sha256=lock.distribution_lock_sha256,
            runtime_image_lock_sha256=lock.runtime_image_lock_sha256,
            archive=archive,
            archive_file=stable_archive.identity,
            release=release,
            daemon=daemon,
            loaded_image_verified=True,
            load_performed_from_verified_descriptor=True,
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distribution-lock", type=Path, required=True)
    parser.add_argument("--runtime-image-lock", type=Path, required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("verify-lock", help="verify the source-free distribution locator")
    release = commands.add_parser(
        "verify-release-archive", help="verify the exact immutable release asset"
    )
    release.add_argument("--archive", type=Path, required=True)
    commands.add_parser("verify-daemon", help="verify the exact source-free Docker daemon")
    runtime = commands.add_parser(
        "verify-runtime", help="verify the release, daemon, and already-loaded runtime"
    )
    runtime.add_argument("--archive", type=Path, required=True)
    prepare = commands.add_parser(
        "prepare-runtime", help="verify, load, and reverify the complete runtime"
    )
    prepare.add_argument("--archive", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        lock = verify_distribution_lock(args.distribution_lock, args.runtime_image_lock)
        result: object
        if args.command == "verify-lock":
            result = lock
        elif args.command == "verify-release-archive":
            _verified_archive_path(args.archive, lock)
            archive = verify_archive(args.runtime_image_lock, args.archive)
            release = verify_immutable_release_asset(lock, args.archive)
            result = {"archive": asdict(archive), "release": asdict(release)}
        elif args.command == "verify-daemon":
            result = verify_runtime_daemon(lock)
        elif args.command == "verify-runtime":
            result = verify_runtime_distribution(
                args.distribution_lock, args.runtime_image_lock, args.archive
            )
        elif args.command == "prepare-runtime":
            result = prepare_runtime_distribution(
                args.distribution_lock, args.runtime_image_lock, args.archive
            )
        else:  # pragma: no cover - argparse enforces the closed command set.
            raise VerificationError(f"unsupported command {args.command!r}")
        if isinstance(
            result,
            (RuntimeDistributionLock, RuntimeDaemonIdentity, RuntimeDistributionObservation),
        ):
            payload = asdict(result)
        else:
            payload = result
        print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    except VerificationError as exc:
        print(f"verification failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
