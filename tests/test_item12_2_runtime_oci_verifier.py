from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import Any

import pytest

SCRIPT_PATH = Path(__file__).parents[1] / "scripts/verify_sciplex3_v5_runtime_oci.py"
SCRIPT_SPEC = importlib.util.spec_from_file_location("item12_2_runtime_oci_verifier", SCRIPT_PATH)
assert SCRIPT_SPEC is not None
assert SCRIPT_SPEC.loader is not None
SCRIPT_MODULE = importlib.util.module_from_spec(SCRIPT_SPEC)
sys.modules[SCRIPT_SPEC.name] = SCRIPT_MODULE
SCRIPT_SPEC.loader.exec_module(SCRIPT_MODULE)

VerificationError = SCRIPT_MODULE.VerificationError
EXPECTED_BUILDKIT_IMAGE_DIGEST = SCRIPT_MODULE.EXPECTED_BUILDKIT_IMAGE_DIGEST
EXPECTED_BUILDKIT_VERSION = SCRIPT_MODULE.EXPECTED_BUILDKIT_VERSION
EXPECTED_BUILDX_COMMIT = SCRIPT_MODULE.EXPECTED_BUILDX_COMMIT
EXPECTED_BUILDX_VERSION = SCRIPT_MODULE.EXPECTED_BUILDX_VERSION
EXPECTED_DOCKERFILE_FRONTEND_DIGEST = SCRIPT_MODULE.EXPECTED_DOCKERFILE_FRONTEND_DIGEST
EXPECTED_IMAGE_TAG = SCRIPT_MODULE.EXPECTED_IMAGE_TAG
_verify_loaded_image_payload = SCRIPT_MODULE._verify_loaded_image_payload
main = SCRIPT_MODULE.main
verify_archive = SCRIPT_MODULE.verify_archive
verify_archives = SCRIPT_MODULE.verify_archives
verify_builder = SCRIPT_MODULE.verify_builder
verify_build_inputs = SCRIPT_MODULE.verify_build_inputs


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _digest(value: bytes) -> str:
    return f"sha256:{hashlib.sha256(value).hexdigest()}"


def _blob_path(digest: str) -> str:
    return f"blobs/sha256/{digest.removeprefix('sha256:')}"


def _fixture_payloads() -> tuple[dict[str, bytes], dict[str, Any]]:
    layer = b"source-free synthetic OCI layer"
    layer_digest = _digest(layer)
    config = _json_bytes(
        {
            "architecture": "amd64",
            "config": {"Entrypoint": ["/opt/runtime/bin/python"]},
            "os": "linux",
            "rootfs": {"diff_ids": [_digest(b"uncompressed layer")], "type": "layers"},
        }
    )
    config_digest = _digest(config)
    manifest = _json_bytes(
        {
            "config": {
                "digest": config_digest,
                "mediaType": "application/vnd.oci.image.config.v1+json",
                "size": len(config),
            },
            "layers": [
                {
                    "digest": layer_digest,
                    "mediaType": "application/vnd.oci.image.layer.v1.tar+gzip",
                    "size": len(layer),
                }
            ],
            "mediaType": "application/vnd.oci.image.manifest.v1+json",
            "schemaVersion": 2,
        }
    )
    manifest_digest = _digest(manifest)
    index = _json_bytes(
        {
            "manifests": [
                {
                    "digest": manifest_digest,
                    "mediaType": "application/vnd.oci.image.manifest.v1+json",
                    "platform": {"architecture": "amd64", "os": "linux"},
                    "size": len(manifest),
                }
            ],
            "mediaType": "application/vnd.oci.image.index.v1+json",
            "schemaVersion": 2,
        }
    )
    files = {
        "blobs/sha256/" + layer_digest.removeprefix("sha256:"): layer,
        "blobs/sha256/" + config_digest.removeprefix("sha256:"): config,
        "blobs/sha256/" + manifest_digest.removeprefix("sha256:"): manifest,
        "index.json": index,
        "oci-layout": _json_bytes({"imageLayoutVersion": "1.0.0"}),
    }
    lock = {
        "architecture": "amd64",
        "archive_sha256": "0" * 64,
        "build": {
            "dockerfile_sha256": "0" * 64,
            "image_tag": EXPECTED_IMAGE_TAG,
            "no_cache": True,
            "output_options": ["type=oci"],
            "platform": "linux/amd64",
            "provenance_attestation_disabled": True,
            "reproducibility_build_count": 2,
            "requirements_sha256": "0" * 64,
            "source_date_epoch": 1_786_406_400,
        },
        "builder": {
            "buildkit_image_digest": EXPECTED_BUILDKIT_IMAGE_DIGEST,
            "buildkit_version": EXPECTED_BUILDKIT_VERSION,
            "buildx_commit": EXPECTED_BUILDX_COMMIT,
            "buildx_version": EXPECTED_BUILDX_VERSION,
            "dockerfile_frontend_digest": EXPECTED_DOCKERFILE_FRONTEND_DIGEST,
        },
        "config_digest": config_digest,
        "image_digest": manifest_digest,
        "image_reference": f"synthetic-runtime@{manifest_digest}",
        "layers": [
            {
                "byte_count": len(layer),
                "digest": layer_digest,
                "media_type": "application/vnd.oci.image.layer.v1.tar+gzip",
            }
        ],
        "oci_index_digest": _digest(index),
        "operating_system": "linux",
        "platform": "linux/amd64",
        "runtime_image_lock_schema": "cellstate-sciplex3-v5-runtime-image-lock",
        "runtime_image_lock_version": "1.0.0",
    }
    return files, lock


def _write_archive(
    path: Path,
    files: dict[str, bytes],
    *,
    reverse: bool = False,
    extra_member: tarfile.TarInfo | None = None,
) -> None:
    items = list(files.items())
    if reverse:
        items.reverse()
    with tarfile.open(path, "w") as archive:
        for index, (name, payload) in enumerate(items):
            member = tarfile.TarInfo(name)
            member.mtime = index + int(reverse)
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))
        if extra_member is not None:
            archive.addfile(extra_member)


def _write_lock(path: Path, lock: dict[str, Any]) -> None:
    path.write_bytes(_json_bytes(lock))


def _bind_archive(lock: dict[str, Any], path: Path) -> None:
    lock["archive_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()


def test_two_independent_oci_archives_match_the_complete_locked_identity(tmp_path: Path) -> None:
    files, lock = _fixture_payloads()
    lock_path = tmp_path / "runtime-image-lock.json"
    first = tmp_path / "first.oci.tar"
    second = tmp_path / "second.oci.tar"
    _write_archive(first, files)
    _write_archive(second, files)
    _bind_archive(lock, first)
    _write_lock(lock_path, lock)

    identity = verify_archives(lock_path, [first, second])

    assert identity.index_digest == lock["oci_index_digest"]
    assert identity.image_digest == lock["image_digest"]
    assert identity.config_digest == lock["config_digest"]
    assert identity.archive_sha256 == lock["archive_sha256"]
    assert len(identity.layer_digests) == 1


def test_single_distributed_archive_cli_verifies_the_complete_locked_identity(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    files, lock = _fixture_payloads()
    lock_path = tmp_path / "runtime-image-lock.json"
    archive_path = tmp_path / "runtime.oci.tar"
    _write_archive(archive_path, files)
    _bind_archive(lock, archive_path)
    _write_lock(lock_path, lock)

    assert (
        main(
            [
                "--lock",
                str(lock_path),
                "verify-archive",
                "--archive",
                str(archive_path),
            ]
        )
        == 0
    )
    output = json.loads(capsys.readouterr().out)
    assert output["archive_sha256"] == lock["archive_sha256"]
    assert output["image_digest"] == lock["image_digest"]


def test_oci_verifier_rejects_blob_content_that_does_not_match_its_digest(tmp_path: Path) -> None:
    files, lock = _fixture_payloads()
    layer_path = next(name for name in files if files[name].startswith(b"source-free"))
    files[layer_path] = b"x" * len(files[layer_path])
    lock_path = tmp_path / "runtime-image-lock.json"
    archive_path = tmp_path / "tampered.oci.tar"
    _write_archive(archive_path, files)
    _bind_archive(lock, archive_path)
    _write_lock(lock_path, lock)

    with pytest.raises(VerificationError, match="does not match path digest"):
        verify_archive(lock_path, archive_path)


def test_oci_verifier_rejects_unreferenced_blobs(tmp_path: Path) -> None:
    files, lock = _fixture_payloads()
    extra = b"unreferenced"
    files[_blob_path(_digest(extra))] = extra
    lock_path = tmp_path / "runtime-image-lock.json"
    archive_path = tmp_path / "extra.oci.tar"
    _write_archive(archive_path, files)
    _bind_archive(lock, archive_path)
    _write_lock(lock_path, lock)

    with pytest.raises(VerificationError, match="missing or unreferenced blobs"):
        verify_archive(lock_path, archive_path)


def test_oci_verifier_rejects_nonregular_archive_members(tmp_path: Path) -> None:
    files, lock = _fixture_payloads()
    symlink = tarfile.TarInfo("blobs/sha256/" + "f" * 64)
    symlink.type = tarfile.SYMTYPE
    symlink.linkname = "../../index.json"
    lock_path = tmp_path / "runtime-image-lock.json"
    archive_path = tmp_path / "symlink.oci.tar"
    _write_archive(archive_path, files, extra_member=symlink)
    _bind_archive(lock, archive_path)
    _write_lock(lock_path, lock)

    with pytest.raises(VerificationError, match="non-regular OCI archive member"):
        verify_archive(lock_path, archive_path)


def test_oci_verifier_never_follows_an_archive_path_symlink(tmp_path: Path) -> None:
    files, lock = _fixture_payloads()
    lock_path = tmp_path / "runtime-image-lock.json"
    archive_path = tmp_path / "runtime.oci.tar"
    linked_path = tmp_path / "linked.oci.tar"
    _write_archive(archive_path, files)
    _bind_archive(lock, archive_path)
    _write_lock(lock_path, lock)
    linked_path.symlink_to(archive_path)

    with pytest.raises(VerificationError, match="securely open OCI archive"):
        verify_archive(lock_path, linked_path)


def test_build_input_verifier_binds_hashes_platform_and_epoch(tmp_path: Path) -> None:
    files, lock = _fixture_payloads()
    del files
    context = tmp_path / "context"
    context.mkdir()
    dockerfile = context / "Dockerfile"
    requirements = context / "requirements.lock"
    dockerfile.write_bytes(
        b"# syntax=docker/dockerfile:1.7@"
        + EXPECTED_DOCKERFILE_FRONTEND_DIGEST.encode()
        + b"\nFROM scratch\n"
    )
    requirements.write_bytes(b"example==1 --hash=sha256:" + b"0" * 64 + b"\n")
    lock["build"]["dockerfile_sha256"] = _digest(dockerfile.read_bytes()).removeprefix("sha256:")
    lock["build"]["requirements_sha256"] = _digest(requirements.read_bytes()).removeprefix(
        "sha256:"
    )
    lock_path = tmp_path / "runtime-image-lock.json"
    _write_lock(lock_path, lock)

    assert verify_build_inputs(lock_path, context) == 1_786_406_400

    dockerfile.write_bytes(b"FROM busybox\n")
    with pytest.raises(VerificationError, match="frozen build input drift for Dockerfile"):
        verify_build_inputs(lock_path, context)


def test_oci_verifier_rejects_archive_wrapper_and_locked_layer_drift(tmp_path: Path) -> None:
    files, lock = _fixture_payloads()
    lock_path = tmp_path / "runtime-image-lock.json"
    archive_path = tmp_path / "runtime.oci.tar"
    _write_archive(archive_path, files)
    _bind_archive(lock, archive_path)
    _write_lock(lock_path, lock)

    reordered = tmp_path / "reordered.oci.tar"
    _write_archive(reordered, files, reverse=True)
    with pytest.raises(VerificationError, match="archive SHA-256 drift"):
        verify_archive(lock_path, reordered)

    lock["layers"][0]["byte_count"] += 1
    _write_lock(lock_path, lock)
    with pytest.raises(VerificationError, match="layer closure"):
        verify_archive(lock_path, archive_path)


def test_build_input_verifier_rejects_builder_identity_substitution(tmp_path: Path) -> None:
    _, lock = _fixture_payloads()
    context = tmp_path / "context"
    context.mkdir()
    dockerfile = context / "Dockerfile"
    requirements = context / "requirements.lock"
    dockerfile.write_bytes(
        b"# syntax=docker/dockerfile:1.7@"
        + EXPECTED_DOCKERFILE_FRONTEND_DIGEST.encode()
        + b"\nFROM scratch\n"
    )
    requirements.write_bytes(b"locked\n")
    lock["build"]["dockerfile_sha256"] = hashlib.sha256(dockerfile.read_bytes()).hexdigest()
    lock["build"]["requirements_sha256"] = hashlib.sha256(requirements.read_bytes()).hexdigest()
    lock["builder"]["buildx_commit"] = "f" * 40
    lock_path = tmp_path / "runtime-image-lock.json"
    _write_lock(lock_path, lock)

    with pytest.raises(VerificationError, match="builder identity"):
        verify_build_inputs(lock_path, context)


def test_active_builder_verifier_binds_client_worker_and_image(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, lock = _fixture_payloads()
    lock_path = tmp_path / "runtime-image-lock.json"
    _write_lock(lock_path, lock)
    builder_inventory = json.dumps(
        {
            "Current": True,
            "Driver": "docker-container",
            "Nodes": [
                {
                    "Name": "frozen0",
                    "Platforms": ["linux/amd64"],
                    "Status": "running",
                    "Version": EXPECTED_BUILDKIT_VERSION,
                }
            ],
        }
    )

    def run(command: list[str] | tuple[str, ...], **_: object) -> subprocess.CompletedProcess[str]:
        argv = tuple(command)
        if argv == ("docker", "buildx", "version"):
            output = (
                f"github.com/docker/buildx {EXPECTED_BUILDX_VERSION} {EXPECTED_BUILDX_COMMIT}\n"
            )
        elif argv == ("docker", "buildx", "ls", "--format", "json"):
            output = builder_inventory + "\n"
        else:
            assert argv == ("docker", "buildx", "inspect", "--bootstrap")
            output = (
                "Name: frozen\nDriver: docker-container\n"
                f'Driver Options: image="moby/buildkit@{EXPECTED_BUILDKIT_IMAGE_DIGEST}"\n'
            )
        return subprocess.CompletedProcess(argv, 0, output, "")

    monkeypatch.setattr(SCRIPT_MODULE.subprocess, "run", run)
    verify_builder(lock_path)

    def wrong_client(
        command: list[str] | tuple[str, ...], **_: object
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(tuple(command), 0, "wrong\n", "")

    monkeypatch.setattr(SCRIPT_MODULE.subprocess, "run", wrong_client)
    with pytest.raises(VerificationError, match="Buildx client identity"):
        verify_builder(lock_path)

    def wrong_worker_authority(
        command: list[str] | tuple[str, ...], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        result = run(command, **kwargs)
        if tuple(command) == ("docker", "buildx", "inspect", "--bootstrap"):
            return subprocess.CompletedProcess(
                tuple(command),
                0,
                result.stdout.replace(EXPECTED_BUILDKIT_IMAGE_DIGEST, "sha256:" + "0" * 64),
                "",
            )
        return result

    monkeypatch.setattr(SCRIPT_MODULE.subprocess, "run", wrong_worker_authority)
    with pytest.raises(VerificationError, match="worker image authority"):
        verify_builder(lock_path)


def test_loaded_image_verifier_accepts_classic_docker_config_identity() -> None:
    _, lock = _fixture_payloads()

    _verify_loaded_image_payload(
        lock,
        [
            {
                "Architecture": "amd64",
                "Id": lock["config_digest"],
                "Os": "linux",
                "RepoDigests": [lock["image_reference"]],
            }
        ],
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ({"Id": "sha256:" + "0" * 64}, "image ID"),
        ({"Architecture": "arm64"}, "platform"),
        ({"RepoDigests": []}, "exact locked digest reference"),
        ({"Descriptor": {"digest": "sha256:" + "0" * 64}}, "descriptor"),
    ),
)
def test_loaded_image_verifier_fails_closed(mutation: dict[str, Any], message: str) -> None:
    _, lock = _fixture_payloads()
    image = {
        "Architecture": "amd64",
        "Descriptor": {"digest": lock["image_digest"]},
        "Id": lock["image_digest"],
        "Os": "linux",
        "RepoDigests": [lock["image_reference"]],
    }
    image.update(mutation)

    with pytest.raises(VerificationError, match=message):
        _verify_loaded_image_payload(lock, [image])
