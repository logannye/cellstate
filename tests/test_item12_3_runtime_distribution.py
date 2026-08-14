from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
SCRIPT_PATH = ROOT / "scripts/verify_sciplex3_v5_runtime_distribution.py"
SCRIPT_SPEC = importlib.util.spec_from_file_location("item12_3_runtime_distribution", SCRIPT_PATH)
assert SCRIPT_SPEC is not None
assert SCRIPT_SPEC.loader is not None
distribution = importlib.util.module_from_spec(SCRIPT_SPEC)
sys.modules[SCRIPT_SPEC.name] = distribution
sys.path.insert(0, str(SCRIPT_PATH.parent))
try:
    SCRIPT_SPEC.loader.exec_module(distribution)
finally:
    sys.path.remove(str(SCRIPT_PATH.parent))

ArchiveIdentity = distribution.ArchiveIdentity
VerificationError = distribution.VerificationError
DISTRIBUTION_LOCK = ROOT / "containers/sciplex3-v5-runtime/runtime-distribution-lock.json"
IMAGE_LOCK = ROOT / "containers/sciplex3-v5-runtime/runtime-image-lock.json"


@pytest.fixture(autouse=True)
def _native_linux_x86_64_host(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        distribution,
        "_host_identity",
        lambda: (
            distribution.EXPECTED_HOST_OPERATING_SYSTEM,
            distribution.EXPECTED_HOST_ARCHITECTURE,
        ),
    )


def _json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _release_payload(*, immutable: bool = True) -> str:
    return _json(
        {
            "assets": [
                {
                    "digest": f"sha256:{distribution.EXPECTED_ARCHIVE_SHA256}",
                    "name": distribution.EXPECTED_ASSET_NAME,
                    "size": distribution.EXPECTED_ARCHIVE_BYTE_COUNT,
                    "state": "uploaded",
                    "url": (
                        "https://github.com/logannye/cellstate/releases/download/"
                        f"{distribution.EXPECTED_RELEASE_TAG}/"
                        f"{distribution.EXPECTED_ASSET_NAME}"
                    ),
                }
            ],
            "isDraft": False,
            "isImmutable": immutable,
            "tagName": distribution.EXPECTED_RELEASE_TAG,
            "targetCommitish": distribution.EXPECTED_RELEASE_TARGET_COMMIT,
        }
    )


def _attestation_payload(*, asset_sha256: str = distribution.EXPECTED_ARCHIVE_SHA256) -> str:
    package_url = (
        f"pkg:github/{distribution.EXPECTED_REPOSITORY}@{distribution.EXPECTED_RELEASE_TAG}"
    )
    return _json(
        {
            "verificationResult": {
                "signature": {
                    "certificate": {
                        "subjectAlternativeName": (
                            distribution.EXPECTED_ATTESTATION_CERTIFICATE_IDENTITY
                        )
                    }
                },
                "statement": {
                    "_type": distribution.EXPECTED_ATTESTATION_STATEMENT_TYPE,
                    "predicate": {
                        "purl": package_url,
                        "repository": distribution.EXPECTED_REPOSITORY,
                        "tag": distribution.EXPECTED_RELEASE_TAG,
                    },
                    "predicateType": distribution.EXPECTED_ATTESTATION_PREDICATE_TYPE,
                    "subject": [
                        {
                            "digest": {"sha1": distribution.EXPECTED_RELEASE_TARGET_COMMIT},
                            "uri": package_url,
                        },
                        {
                            "digest": {"sha256": asset_sha256},
                            "name": distribution.EXPECTED_ASSET_NAME,
                        },
                    ],
                },
            }
        }
    )


def _runtime_archive(tmp_path: Path) -> Path:
    path = tmp_path / distribution.EXPECTED_ASSET_NAME
    with path.open("wb") as stream:
        stream.truncate(distribution.EXPECTED_ARCHIVE_BYTE_COUNT)
    return path


def _successful_runner(
    commands: list[tuple[str, ...]], *, immutable: bool = True
) -> distribution.CommandRunner:
    def run(
        command: Sequence[str], _: float, _pass_fds: tuple[int, ...]
    ) -> distribution.CommandResult:
        argv = tuple(command)
        commands.append(argv)
        if argv[:3] == ("gh", "release", "view"):
            return distribution.CommandResult(0, _release_payload(immutable=immutable), "")
        if argv[:3] in {
            ("gh", "release", "verify"),
            ("gh", "release", "verify-asset"),
        }:
            return distribution.CommandResult(0, _attestation_payload(), "")
        if argv == ("docker", "context", "show"):
            return distribution.CommandResult(0, "item12-runtime\n", "")
        if argv[:3] == ("docker", "context", "inspect"):
            return distribution.CommandResult(0, _json("unix:///run/item12/docker.sock"), "")
        if argv == ("docker", "version", "--format", "{{json .Server.Version}}"):
            return distribution.CommandResult(
                0, _json(distribution.EXPECTED_DOCKER_SERVER_VERSION), ""
            )
        if argv == ("docker", "info", "--format", "{{json .OSType}}"):
            return distribution.CommandResult(
                0, _json(distribution.EXPECTED_DOCKER_OPERATING_SYSTEM), ""
            )
        if argv == ("docker", "info", "--format", "{{json .Architecture}}"):
            return distribution.CommandResult(
                0, _json(distribution.EXPECTED_DOCKER_ARCHITECTURE), ""
            )
        if argv == ("docker", "info", "--format", "{{json .DriverStatus}}"):
            return distribution.CommandResult(
                0, _json([list(distribution.EXPECTED_IMAGE_STORE_STATUS)]), ""
            )
        if argv == ("docker", "info", "--format", "{{json .CgroupVersion}}"):
            return distribution.CommandResult(0, _json(distribution.EXPECTED_CGROUP_VERSION), "")
        if argv in {
            ("docker", "info", "--format", "{{json .MemoryLimit}}"),
            ("docker", "info", "--format", "{{json .SwapLimit}}"),
            ("docker", "info", "--format", "{{json .PidsLimit}}"),
        }:
            return distribution.CommandResult(0, _json(True), "")
        if argv[:2] == ("docker", "load"):
            return distribution.CommandResult(0, "Loaded image\n", "")
        if argv[:3] == ("docker", "image", "inspect"):
            return distribution.CommandResult(
                0,
                _json(
                    [
                        {
                            "Architecture": "amd64",
                            "Descriptor": {"digest": distribution.EXPECTED_IMAGE_DIGEST},
                            "Id": distribution.EXPECTED_IMAGE_DIGEST,
                            "Os": "linux",
                            "RepoDigests": [
                                "cellstate-sciplex3-v5-runtime@"
                                f"{distribution.EXPECTED_IMAGE_DIGEST}"
                            ],
                        }
                    ]
                ),
                "",
            )
        raise AssertionError(f"unexpected source-free command: {argv}")

    return run


def test_checked_in_distribution_lock_is_exact_and_noncircular() -> None:
    lock = distribution.verify_distribution_lock(DISTRIBUTION_LOCK, IMAGE_LOCK)

    assert lock.repository == "logannye/cellstate"
    assert lock.release_target_commit == "d254d7c128e7025f19d2c84cdb3eb9901c5e69cb"
    assert lock.archive_sha256 == distribution.EXPECTED_ARCHIVE_SHA256
    assert lock.distribution_lock_sha256 == distribution.EXPECTED_RUNTIME_DISTRIBUTION_LOCK_SHA256
    assert lock.runtime_image_lock_sha256 == distribution.EXPECTED_RUNTIME_IMAGE_LOCK_SHA256
    assert lock.host_operating_system == "linux"
    assert lock.host_architecture == "x86_64"
    assert lock.docker_server_version == "29.7.2"
    assert lock.image_store_status == ("driver-type", "io.containerd.snapshotter.v1")
    assert lock.cgroup_version == "2"
    assert lock.memory_limit_supported is True
    assert lock.memory_swap_limit_supported is True
    assert lock.pids_limit_supported is True


def test_distribution_lock_rejects_mutable_release_or_image_lock_drift(tmp_path: Path) -> None:
    payload = json.loads(DISTRIBUTION_LOCK.read_bytes())
    payload["distribution"]["immutable_release_required"] = False
    changed_distribution = tmp_path / "runtime-distribution-lock.json"
    changed_distribution.write_text(_json(payload), encoding="utf-8")

    with pytest.raises(VerificationError, match="raw SHA-256"):
        distribution.verify_distribution_lock(changed_distribution, IMAGE_LOCK)

    changed_image = tmp_path / "runtime-image-lock.json"
    changed_image.write_bytes(IMAGE_LOCK.read_bytes() + b"\n")
    with pytest.raises(VerificationError, match="another runtime image lock"):
        distribution.verify_distribution_lock(DISTRIBUTION_LOCK, changed_image)


def test_immutable_release_verification_binds_metadata_and_both_attestations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = _runtime_archive(tmp_path)
    lock = distribution.verify_distribution_lock(DISTRIBUTION_LOCK, IMAGE_LOCK)
    commands: list[tuple[str, ...]] = []
    original_sha256 = distribution._sha256_file

    def sha256(path: Path) -> str:
        if path == archive:
            return distribution.EXPECTED_ARCHIVE_SHA256
        return original_sha256(path)

    monkeypatch.setattr(distribution, "_sha256_file", sha256)
    observed = distribution.verify_immutable_release_asset(
        lock, archive, runner=_successful_runner(commands)
    )

    assert observed.asset_name == distribution.EXPECTED_ASSET_NAME
    assert observed.asset_sha256 == distribution.EXPECTED_ARCHIVE_SHA256
    assert observed.release_attestation_verified is True
    assert observed.asset_attestation_verified is True
    assert any(command[:3] == ("gh", "release", "verify") for command in commands)
    assert any(command[:3] == ("gh", "release", "verify-asset") for command in commands)


def test_immutable_release_verification_rejects_mutable_release(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = _runtime_archive(tmp_path)
    lock = distribution.verify_distribution_lock(DISTRIBUTION_LOCK, IMAGE_LOCK)
    monkeypatch.setattr(
        distribution,
        "_sha256_file",
        lambda _: distribution.EXPECTED_ARCHIVE_SHA256,
    )

    with pytest.raises(VerificationError, match="exact published immutable release"):
        distribution.verify_immutable_release_asset(
            lock, archive, runner=_successful_runner([], immutable=False)
        )


def test_immutable_release_verification_rejects_attested_subject_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = _runtime_archive(tmp_path)
    lock = distribution.verify_distribution_lock(DISTRIBUTION_LOCK, IMAGE_LOCK)
    monkeypatch.setattr(
        distribution,
        "_sha256_file",
        lambda _: distribution.EXPECTED_ARCHIVE_SHA256,
    )
    good_runner = _successful_runner([])

    def changed_attestation(
        command: Sequence[str], timeout: float, pass_fds: tuple[int, ...]
    ) -> distribution.CommandResult:
        result = good_runner(command, timeout, pass_fds)
        if tuple(command)[:3] == ("gh", "release", "verify"):
            return distribution.CommandResult(0, _attestation_payload(asset_sha256="0" * 64), "")
        return result

    with pytest.raises(VerificationError, match="immutable locator"):
        distribution.verify_immutable_release_asset(lock, archive, runner=changed_attestation)


def test_runtime_daemon_requires_exact_version_native_linux_and_containerd_store() -> None:
    lock = distribution.verify_distribution_lock(DISTRIBUTION_LOCK, IMAGE_LOCK)
    commands: list[tuple[str, ...]] = []

    observed = distribution.verify_runtime_daemon(lock, runner=_successful_runner(commands))

    assert observed.server_version == "29.7.2"
    assert observed.host_operating_system == "linux"
    assert observed.host_architecture == "x86_64"
    assert observed.architecture == "x86_64"
    assert observed.endpoint.startswith("unix://")
    assert observed.cgroup_version == "2"
    assert observed.memory_limit_supported is True
    assert observed.memory_swap_limit_supported is True
    assert observed.pids_limit_supported is True

    good_runner = _successful_runner([])

    def old_daemon(
        command: Sequence[str], timeout: float, pass_fds: tuple[int, ...]
    ) -> distribution.CommandResult:
        result = good_runner(command, timeout, pass_fds)
        if tuple(command) == (
            "docker",
            "version",
            "--format",
            "{{json .Server.Version}}",
        ):
            return distribution.CommandResult(0, _json("28.4.0"), "")
        return result

    with pytest.raises(VerificationError, match="server version"):
        distribution.verify_runtime_daemon(lock, runner=old_daemon)


@pytest.mark.parametrize(
    ("host_identity", "message"),
    (
        (("darwin", "x86_64"), "operating system"),
        (("linux", "aarch64"), "host architecture"),
    ),
)
def test_runtime_daemon_rejects_non_native_host_before_any_docker_call(
    host_identity: tuple[str, str],
    message: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock = distribution.verify_distribution_lock(DISTRIBUTION_LOCK, IMAGE_LOCK)
    commands: list[tuple[str, ...]] = []
    monkeypatch.setattr(distribution, "_host_identity", lambda: host_identity)

    with pytest.raises(VerificationError, match=message):
        distribution.verify_runtime_daemon(lock, runner=_successful_runner(commands))

    assert commands == []


@pytest.mark.parametrize(
    ("field", "changed", "message"),
    (
        ("CgroupVersion", "1", "cgroup version"),
        ("MemoryLimit", False, "memory limits"),
        ("SwapLimit", False, "memory-plus-swap"),
        ("PidsLimit", False, "PID limits"),
    ),
)
def test_runtime_daemon_rejects_missing_cgroup_limit_support(
    field: str,
    changed: object,
    message: str,
) -> None:
    lock = distribution.verify_distribution_lock(DISTRIBUTION_LOCK, IMAGE_LOCK)
    good_runner = _successful_runner([])

    def changed_daemon(
        command: Sequence[str], timeout: float, pass_fds: tuple[int, ...]
    ) -> distribution.CommandResult:
        result = good_runner(command, timeout, pass_fds)
        if tuple(command) == (
            "docker",
            "info",
            "--format",
            f"{{{{json .{field}}}}}",
        ):
            return distribution.CommandResult(0, _json(changed), "")
        return result

    with pytest.raises(VerificationError, match=message):
        distribution.verify_runtime_daemon(lock, runner=changed_daemon)


@pytest.mark.parametrize("environment_key", ("DOCKER_CONFIG", "DOCKER_CONTEXT"))
def test_runtime_daemon_rejects_environment_overrides(
    environment_key: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock = distribution.verify_distribution_lock(DISTRIBUTION_LOCK, IMAGE_LOCK)

    monkeypatch.setenv(environment_key, "unreviewed")
    with pytest.raises(VerificationError, match="environment overrides"):
        distribution.verify_runtime_daemon(lock, runner=_successful_runner([]))


def test_prepare_runtime_distribution_orders_all_preflights_before_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive_path = _runtime_archive(tmp_path)
    archive = ArchiveIdentity(
        archive_sha256=distribution.EXPECTED_ARCHIVE_SHA256,
        index_digest=distribution.EXPECTED_OCI_INDEX_DIGEST,
        image_digest=distribution.EXPECTED_IMAGE_DIGEST,
        config_digest=distribution.EXPECTED_CONFIG_DIGEST,
        layers=(),
    )
    monkeypatch.setattr(distribution, "verify_archive", lambda *_: archive)
    descriptor = os.open(archive_path, os.O_RDONLY)

    class FakeDescriptor:
        identity = distribution.RuntimeArchiveFileIdentity(
            device=1,
            inode=2,
            byte_count=distribution.EXPECTED_ARCHIVE_BYTE_COUNT,
            mode=0o400,
            modification_time_ns=3,
            change_time_ns=4,
        )

        def __init__(self, *_: object) -> None:
            self.descriptor = descriptor

        def __enter__(self) -> FakeDescriptor:
            return self

        def __exit__(self, *_: object) -> None:
            os.close(self.descriptor)

        def verify(self) -> None:
            return None

    monkeypatch.setattr(distribution, "_VerifiedArchiveDescriptor", FakeDescriptor)
    original_sha256 = distribution._sha256_file

    def sha256(path: Path) -> str:
        if path == archive_path:
            return distribution.EXPECTED_ARCHIVE_SHA256
        return original_sha256(path)

    monkeypatch.setattr(distribution, "_sha256_file", sha256)
    commands: list[tuple[str, ...]] = []

    observed = distribution.prepare_runtime_distribution(
        DISTRIBUTION_LOCK,
        IMAGE_LOCK,
        archive_path,
        runner=_successful_runner(commands),
    )

    asset_attestation = next(
        index
        for index, command in enumerate(commands)
        if command[:3] == ("gh", "release", "verify-asset")
    )
    daemon = commands.index(("docker", "context", "show"))
    load = next(
        index for index, command in enumerate(commands) if command[:2] == ("docker", "load")
    )
    loaded_image = next(
        index
        for index, command in enumerate(commands)
        if command[:3] == ("docker", "image", "inspect")
    )
    assert asset_attestation < daemon < load < loaded_image
    assert observed.loaded_image_verified is True
    assert observed.load_performed_from_verified_descriptor is True
    assert len(observed.fingerprint) == 64


def test_docker_load_consumes_held_verified_descriptor_after_path_substitution(
    tmp_path: Path,
) -> None:
    exact_payload = b"exact reviewed runtime archive bytes"
    path = tmp_path / "runtime.oci.tar"
    path.write_bytes(exact_payload)
    base_lock = distribution.verify_distribution_lock(DISTRIBUTION_LOCK, IMAGE_LOCK)
    lock = replace(
        base_lock,
        asset_name=path.name,
        archive_sha256=hashlib.sha256(exact_payload).hexdigest(),
        archive_byte_count=len(exact_payload),
    )

    with distribution._VerifiedArchiveDescriptor(path, lock) as archive:
        original = tmp_path / "original.oci.tar"
        path.rename(original)
        path.write_bytes(b"malicious substituted bytes")

        def load(
            command: Sequence[str], _: float, pass_fds: tuple[int, ...]
        ) -> distribution.CommandResult:
            assert tuple(command) == (
                "docker",
                "load",
                "--input",
                f"/proc/self/fd/{archive.descriptor}",
            )
            assert pass_fds == (archive.descriptor,)
            assert os.pread(pass_fds[0], len(exact_payload), 0) == exact_payload
            return distribution.CommandResult(0, "loaded\n", "")

        with pytest.raises(VerificationError, match="identity changed"):
            distribution._load_runtime_archive_from_descriptor(archive, runner=load)


def test_verified_descriptor_rejects_in_place_mutation_even_if_bytes_are_restored(
    tmp_path: Path,
) -> None:
    exact_payload = b"exact reviewed runtime archive bytes"
    path = tmp_path / "runtime.oci.tar"
    path.write_bytes(exact_payload)
    base_lock = distribution.verify_distribution_lock(DISTRIBUTION_LOCK, IMAGE_LOCK)
    lock = replace(
        base_lock,
        asset_name=path.name,
        archive_sha256=hashlib.sha256(exact_payload).hexdigest(),
        archive_byte_count=len(exact_payload),
    )

    with distribution._VerifiedArchiveDescriptor(path, lock) as archive:
        with path.open("r+b") as mutable:
            mutable.write(b"X")
            mutable.seek(0)
            mutable.write(exact_payload[:1])
            mutable.flush()
            os.fsync(mutable.fileno())
        os.utime(
            path,
            ns=(
                archive.identity.modification_time_ns,
                archive.identity.modification_time_ns + 2_000_000_000,
            ),
        )

        with pytest.raises(VerificationError, match="identity changed"):
            archive.verify()
